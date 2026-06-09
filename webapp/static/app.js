const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const gameSelect = document.getElementById("game-select");
const symbolSelect = document.getElementById("symbol-select");
const newGameBtn = document.getElementById("new-game-btn");

let game = null;
let movePending = false;

function statusText(current) {
  if (!current) {
    return "No game loaded.";
  }

  if (current.status === "in_progress") {
    return current.turn === "player" ? "Your turn." : "AI is thinking...";
  }

  if (current.status === "draw") {
    return "Draw game.";
  }

  if (current.status === "player_won") {
    return "You won!";
  }

  return "AI won.";
}

function canClickCell(r, c) {
  return (
    game &&
    !movePending &&
    game.status === "in_progress" &&
    game.turn === "player" &&
    game.board[r][c] === null
  );
}

async function createGame() {
  statusEl.textContent = "Starting game...";
  const response = await fetch("/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      game_type: gameSelect.value,
      player_symbol: symbolSelect.value,
    }),
  });

  if (!response.ok) {
    statusEl.textContent = "Failed to create game.";
    return;
  }

  game = await response.json();
  render();
}

async function playMove(row, col) {
  if (!game) {
    return;
  }

  if (!canClickCell(row, col)) {
    return;
  }

  const previousGame = structuredClone(game);
  game.board[row][col] = game.player_symbol;
  game.turn = "ai";
  movePending = true;
  statusEl.textContent = "AI is thinking...";
  render();

  const response = await fetch(`/games/${game.game_id}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    game = previousGame;
    movePending = false;
    render();
    statusEl.textContent = payload.detail || "Move failed.";
    return;
  }

  game = await response.json();
  movePending = false;
  render();
}

function render() {
  boardEl.innerHTML = "";

  if (!game) {
    statusEl.textContent = "No game loaded.";
    return;
  }

  boardEl.style.gridTemplateColumns = `repeat(${game.width}, minmax(0, 1fr))`;
  boardEl.classList.toggle("board--large", game.width > 6);

  for (let r = 0; r < game.height; r += 1) {
    for (let c = 0; c < game.width; c += 1) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.textContent = game.board[r][c] || "";
      cell.disabled = !canClickCell(r, c);
      cell.addEventListener("click", () => {
        playMove(r, c);
      });
      boardEl.appendChild(cell);
    }
  }

  statusEl.textContent = statusText(game);
}

newGameBtn.addEventListener("click", () => {
  createGame();
});

gameSelect.addEventListener("change", () => {
  createGame();
});

createGame();
