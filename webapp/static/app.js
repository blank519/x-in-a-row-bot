const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const symbolSelect = document.getElementById("symbol-select");
const newGameBtn = document.getElementById("new-game-btn");

let game = null;

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
    body: JSON.stringify({ player_symbol: symbolSelect.value }),
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

  statusEl.textContent = "Submitting move...";

  const response = await fetch(`/games/${game.game_id}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    statusEl.textContent = payload.detail || "Move failed.";
    return;
  }

  game = await response.json();
  render();
}

function render() {
  boardEl.innerHTML = "";

  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
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

createGame();
