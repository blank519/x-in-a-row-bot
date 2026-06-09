import base64
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from google.oauth2 import service_account
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO

import game_utils as utils

# Keep these imports available so SB3 can deserialize models trained with this repo's classes.
from train_ppo_tic_tac_toe import BoardCnnExtractor, MaskableActorCriticPolicy  # noqa: F401
from train_ppo_gomoku import BoardCnnExtractor as GomokuBoardCnnExtractor  # noqa: F401


APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

GAME_CONFIGS = {
    "tic_tac_toe": {
        "height": 3,
        "width": 3,
        "win_con": 3,
        "model_path_env": "MODEL_PATH_TIC_TAC_TOE",
        "gcs_object_env": "GCS_OBJECT_TIC_TAC_TOE",
        "default_model_path": "outputs/ppo_tic_tac_toe_baseline3.zip",
    },
    "gomoku": {
        "height": 15,
        "width": 15,
        "win_con": 5,
        "model_path_env": "MODEL_PATH_GOMOKU",
        "gcs_object_env": "GCS_OBJECT_GOMOKU",
        "default_model_path": "outputs/ppo_gomoku.zip",
    },
}


def _resolve_model_path(configured: str) -> Path:
    model_path = Path(configured)
    if model_path.exists():
        return model_path

    if model_path.suffix != ".zip":
        with_zip = Path(f"{configured}.zip")
        return with_zip

    return model_path


def _download_model_from_gcs(target_path: Path, object_name: str | None = None) -> None:
    encoded_sa = os.getenv("GCP_SERVICE_ACCOUNT_JSON_B64")
    bucket_name = os.getenv("GCS_BUCKET")
    gcs_object_name = object_name or os.getenv("GCS_OBJECT")

    if not encoded_sa or not bucket_name or not gcs_object_name:
        raise RuntimeError(
            "Model file is missing locally and GCS download is not configured. "
            "Set GCP_SERVICE_ACCOUNT_JSON_B64, GCS_BUCKET, and GCS_OBJECT (or per-game override)."
        )

    try:
        sa_info = json.loads(base64.b64decode(encoded_sa).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Failed to decode GCP_SERVICE_ACCOUNT_JSON_B64.") from exc

    credentials = service_account.Credentials.from_service_account_info(sa_info)
    project = os.getenv("GCP_PROJECT") or sa_info.get("project_id")
    client = storage.Client(project=project, credentials=credentials)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_object_name)
    blob.download_to_filename(str(target_path))
    print("Model downloaded from GCS")


def _load_model_for_game(game_type: str, cfg: dict) -> MaskablePPO:
    configured_path = os.getenv(cfg["model_path_env"], cfg["default_model_path"])
    model_path = _resolve_model_path(configured_path)
    if not model_path.exists():
        object_name = os.getenv(cfg["gcs_object_env"], os.getenv("GCS_OBJECT"))
        print(f"Model for '{game_type}' not found locally, downloading from GCS...")
        _download_model_from_gcs(model_path, object_name=object_name)

    model = MaskablePPO.load(str(model_path))
    print(f"Model for '{game_type}' loaded successfully")
    return model


MODELS: dict[str, MaskablePPO] = {}
MODEL_LOCK = threading.Lock()


@dataclass
class GameSession:
    game_id: str
    game_type: Literal["tic_tac_toe", "gomoku"]
    height: int
    width: int
    win_con: int
    board: list[list[str | None]]
    player_symbol: Literal["X", "O"]
    ai_symbol: Literal["X", "O"]
    turn: Literal["player", "ai"]
    status: Literal["in_progress", "draw", "player_won", "ai_won"]
    winner: str | None
    num_turns: int
    max_turns: int


class NewGameRequest(BaseModel):
    player_symbol: Literal["X", "O"] = "X"
    game_type: Literal["tic_tac_toe", "gomoku"] = "tic_tac_toe"


class MoveRequest(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)


class Move(BaseModel):
    row: int
    col: int


class GameResponse(BaseModel):
    game_id: str
    game_type: Literal["tic_tac_toe", "gomoku"]
    height: int
    width: int
    win_con: int
    board: list[list[str | None]]
    player_symbol: Literal["X", "O"]
    ai_symbol: Literal["X", "O"]
    turn: Literal["player", "ai"]
    status: Literal["in_progress", "draw", "player_won", "ai_won"]
    winner: str | None
    ai_move: Move | None = None
    num_turns: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS

    yield

    with MODEL_LOCK:
        MODELS = {}


app = FastAPI(title="Tic-Tac-Toe AI Service", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


SESSIONS: dict[str, GameSession] = {}
SESSIONS_LOCK = threading.Lock()


def _get_model_for_game(game_type: str) -> MaskablePPO:
    cfg = GAME_CONFIGS[game_type]
    with MODEL_LOCK:
        model = MODELS.get(game_type)
        if model is not None:
            return model

        # Keep only one loaded model to reduce memory footprint on small instances.
        MODELS.clear()
        model = _load_model_for_game(game_type, cfg)
        MODELS[game_type] = model
        return model


def _make_response(session: GameSession, ai_move: Move | None = None) -> GameResponse:
    return GameResponse(
        game_id=session.game_id,
        game_type=session.game_type,
        height=session.height,
        width=session.width,
        win_con=session.win_con,
        board=session.board,
        player_symbol=session.player_symbol,
        ai_symbol=session.ai_symbol,
        turn=session.turn,
        status=session.status,
        winner=session.winner,
        ai_move=ai_move,
        num_turns=session.num_turns,
    )


def _apply_terminal_state(session: GameSession, symbol: str, last_move: Move | None = None) -> None:
    if last_move is not None:
        if utils.check_winner(session.board, symbol, last_move.row, last_move.col, session.win_con):
            session.winner = symbol
            session.status = "player_won" if symbol == session.player_symbol else "ai_won"
            session.turn = "player"
            return

    if utils.is_draw(session.num_turns, session.max_turns):
        session.winner = None
        session.status = "draw"
        session.turn = "player"


def _run_ai_turn(session: GameSession) -> Move | None:
    if session.status != "in_progress" or session.turn != "ai":
        return None

    obs = utils.build_observation(
        session.board,
        learner_symbol=session.ai_symbol,
        opponent_symbol=session.player_symbol,
        height=session.height,
        width=session.width,
    )
    mask = utils.action_mask(session.board, session.height, session.width)

    if not mask.any():
        # Apply draw logic
        _apply_terminal_state(session, session.ai_symbol, None)
        return None

    model = _get_model_for_game(session.game_type)
    with MODEL_LOCK:
        action, _state = model.predict(obs, action_masks=mask, deterministic=True)

    action = int(action)
    if mask[action] == 0:
        legal = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        action = int(legal[0])

    row = action // session.width
    col = action % session.width
    session.board[row][col] = session.ai_symbol
    session.num_turns += 1

    #Check if AI won
    _apply_terminal_state(session, session.ai_symbol, Move(row=row, col=col))
    if session.status == "in_progress":
        session.turn = "player"

    return Move(row=row, col=col)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/games", response_model=GameResponse)
def create_game(payload: NewGameRequest) -> GameResponse:
    cfg = GAME_CONFIGS[payload.game_type]

    try:
        _get_model_for_game(payload.game_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    player_symbol = payload.player_symbol
    ai_symbol: Literal["X", "O"] = "O" if player_symbol == "X" else "X"

    session = GameSession(
        game_id=str(uuid.uuid4()),
        game_type=payload.game_type,
        height=cfg["height"],
        width=cfg["width"],
        win_con=cfg["win_con"],
        board=utils.new_board(cfg["height"], cfg["width"]),
        player_symbol=player_symbol,
        ai_symbol=ai_symbol,
        turn="player" if player_symbol == "X" else "ai",
        status="in_progress",
        winner=None,
        num_turns=0,
        max_turns=cfg["height"] * cfg["width"],
    )

    ai_move: Move | None = None
    if session.turn == "ai":
        ai_move = _run_ai_turn(session)

    with SESSIONS_LOCK:
        SESSIONS[session.game_id] = session

    return _make_response(session, ai_move=ai_move)


@app.get("/games/{game_id}", response_model=GameResponse)
def get_game(game_id: str) -> GameResponse:
    with SESSIONS_LOCK:
        session = SESSIONS.get(game_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    return _make_response(session)


@app.post("/games/{game_id}/move", response_model=GameResponse)
def player_move(game_id: str, payload: MoveRequest) -> GameResponse:
    with SESSIONS_LOCK:
        session = SESSIONS.get(game_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Game is already finished")

    if session.turn != "player":
        raise HTTPException(status_code=400, detail="It is not the player's turn")

    row = payload.row
    col = payload.col
    if row >= session.height or col >= session.width:
        raise HTTPException(status_code=400, detail="Move is outside the board dimensions")

    if session.board[row][col] is not None:
        raise HTTPException(status_code=400, detail="Illegal move: cell is already occupied")

    session.board[row][col] = session.player_symbol
    session.num_turns += 1
    # Check if player won
    _apply_terminal_state(session, session.player_symbol, Move(row=row, col=col))

    ai_move: Move | None = None
    if session.status == "in_progress":
        session.turn = "ai"
        ai_move = _run_ai_turn(session)

    return _make_response(session, ai_move=ai_move)
