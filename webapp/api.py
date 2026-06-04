import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO

import game_utils as utils

# Keep these imports available so SB3 can deserialize models trained with this repo's classes.
from train_ppo_tic_tac_toe import BoardCnnExtractor, MaskableActorCriticPolicy  # noqa: F401


APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

HEIGHT = 3
WIDTH = 3
WIN_CON = 3


def _resolve_model_path() -> Path:
    configured = os.getenv("MODEL_PATH", "ppo_tic_tac_toe_baseline3.zip")
    model_path = Path(configured)
    if model_path.exists():
        return model_path

    if model_path.suffix != ".zip":
        with_zip = Path(f"{configured}.zip")
        if with_zip.exists():
            return with_zip

    raise FileNotFoundError(
        f"Could not find model file '{configured}'. Set MODEL_PATH to a valid .zip file path."
    )


MODEL = MaskablePPO.load(str(_resolve_model_path()))
MODEL_LOCK = threading.Lock()


@dataclass
class GameSession:
    game_id: str
    board: list[list[str | None]]
    player_symbol: Literal["X", "O"]
    ai_symbol: Literal["X", "O"]
    turn: Literal["player", "ai"]
    status: Literal["in_progress", "draw", "player_won", "ai_won"]
    winner: str | None = None
    num_turns: int = 0
    max_turns: int = HEIGHT * WIDTH


class NewGameRequest(BaseModel):
    player_symbol: Literal["X", "O"] = "X"


class MoveRequest(BaseModel):
    row: int = Field(ge=0, lt=HEIGHT)
    col: int = Field(ge=0, lt=WIDTH)


class Move(BaseModel):
    row: int
    col: int


class GameResponse(BaseModel):
    game_id: str
    board: list[list[str | None]]
    player_symbol: Literal["X", "O"]
    ai_symbol: Literal["X", "O"]
    turn: Literal["player", "ai"]
    status: Literal["in_progress", "draw", "player_won", "ai_won"]
    winner: str | None
    ai_move: Move | None = None
    num_turns: int = 0


app = FastAPI(title="Tic-Tac-Toe AI Service")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


SESSIONS: dict[str, GameSession] = {}
SESSIONS_LOCK = threading.Lock()


def _make_response(session: GameSession, ai_move: Move | None = None) -> GameResponse:
    return GameResponse(
        game_id=session.game_id,
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
        if utils.check_winner(session.board, symbol, last_move.row, last_move.col, WIN_CON):
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

    obs = utils.build_observation(session.board, learner_symbol=session.ai_symbol, opponent_symbol=session.player_symbol, height=HEIGHT, width=WIDTH)
    mask = utils.action_mask(session.board, HEIGHT, WIDTH)

    if not mask.any():
        # Apply draw logic
        _apply_terminal_state(session, session.ai_symbol, None)
        return None

    with MODEL_LOCK:
        action, _state = MODEL.predict(obs, action_masks=mask, deterministic=True)

    action = int(action)
    if mask[action] == 0:
        legal = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        action = int(legal[0])

    row = action // WIDTH
    col = action % WIDTH
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
    player_symbol = payload.player_symbol
    ai_symbol: Literal["X", "O"] = "O" if player_symbol == "X" else "X"

    session = GameSession(
        game_id=str(uuid.uuid4()),
        board=utils.new_board(HEIGHT, WIDTH),
        player_symbol=player_symbol,
        ai_symbol=ai_symbol,
        turn="player" if player_symbol == "X" else "ai",
        status="in_progress",
        num_turns=0,
        max_turns=HEIGHT * WIDTH,
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
