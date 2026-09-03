# Orca variant of the ticket pipeline

This is the [Orca](https://github.com/stablyai/orca) port of the planner ->
implementer -> evaluator pipeline. It reuses the role prompts and tickets from the
Pi version (`.pi/`), but moves the orchestration (the loop, retry cap, and
PASS/FAIL branching) onto Orca's Run / Task-DAG / decision-gate machinery instead
of an LLM-followed skill.

The Pi version under `.pi/` is left intact — this is an alternative front end, not
a replacement.

## Files

```
.orca/
├── coordinator.md        # prompt for the coordinator agent that drives Orca
└── roles/
    ├── planner.md        # harness-neutral briefing (from .pi/agents/planner.md)
    ├── implementer.md    # harness-neutral briefing
    └── evaluator.md      # harness-neutral briefing (emits VERDICT: PASS|FAIL)
```

Tickets are shared with the Pi pipeline: `.pi/tickets/*.md` (see
`.pi/tickets/TEMPLATE.md`). The reference docs the roles read are also shared:
`AGENTS.md`, `.pi/skills/experiments/SKILL.md`, `.pi/skills/code-changes/SKILL.md`
(these are plain markdown, readable by any agent's file tools).

## How it maps

| Pi pipeline | Orca pipeline |
|-------------|---------------|
| `.pi/skills/pipeline/SKILL.md` (LLM-followed loop) | `.orca/coordinator.md` driving a Run + task DAG + decision gate |
| `.pi/agents/*.md` (Pi subagent defs) | `.orca/roles/*.md` (briefings injected as task specs) |
| evaluator `VERDICT:` line parsed by the orchestrating LLM | evaluator `worker_done` payload consumed by a **decision gate** |
| retry cap in the skill's instructions | coordinator loop honoring the ticket's `max_iterations` |
| sequential runs | parallel worker tasks in isolated worktrees for experiments |

### What changed vs `.pi/agents/*.md`
- Dropped the Pi frontmatter (`name`/`tools`/`model`) — Orca sets agent, tools,
  model, and worktree per worker via `worker-start`, not via file frontmatter.
- Reframed "you are a Pi subagent" -> "you are the worker for this task".
- Reference docs are cited **by path** (e.g. `.pi/skills/experiments/SKILL.md`)
  rather than as Pi "skills", so a non-Pi worker can still read them.
- The evaluator now also ties its `VERDICT` block to its `worker_done` message,
  which the coordinator's decision gate consumes.

### What stayed identical
- The three role bodies (responsibilities, branch-on-`type`, output formats).
- The `VERDICT: PASS|FAIL` + `FEEDBACK:` contract.
- The ticket format and the shared reference docs.

## Running it

1. **Install Orca** and its `orchestration` skill (see
   <https://github.com/stablyai/orca> and `orca skills get orchestration --full`).
2. **Start a coordinator agent** with `.orca/coordinator.md` as its instructions
   (a Pi/Claude/Codex session holding Orca's `orchestration` skill).
3. Give it a ticket: "run the pipeline on `.pi/tickets/EXAMPLE-experiment.md`".
   The coordinator opens a Run, dispatches the planner, then loops
   implement -> evaluate -> gate until PASS or `max_iterations`.

## Installing the orchestration skill (don't inject the full guide)

Install Orca's `orchestration` skill as a **discovery stub**, not by pasting the
full guide into the coordinator's prompt:

```
npx skills add https://github.com/stablyai/orca --skill orchestration --global
```

Only the stub's short `description` stays in context; the coordinator pulls the
full guide on demand with `orca skills get orchestration --full` when it starts
orchestrating. Injecting the whole ~435-line guide every run would waste a large
slice of the local Qwen's context window for no benefit — `coordinator.md`
already embeds the verified command subset this pipeline needs.

## Verify before relying on it

- **CLI flags.** The command shapes in `coordinator.md` were checked against the
  `orchestration` skill (`task-create --spec/--deps`, `worker-start`,
  `check --wait/--ack`, `worker-release`, `gate-create/gate-resolve`). Flags can
  still drift with Orca versions; the coordinator falls back to
  `orca skills get orchestration --full` if one is rejected.
- **Prompt injection.** Workers receive the task spec + a lifecycle preamble via
  `worker-start` (or the low-level `dispatch --inject`). Confirm the `--spec`
  payload arrives intact for your Orca version.
- **Local Qwen.** Use `--agent pi` workers so each role runs your local Qwen from
  Pi's `models.json`; Orca's `--model` override applies to Claude/Codex/Cursor
  only, not Pi.
- **Worktree strategy.** Sequential code loop = `--worktree current`; parallel
  experiment runs = `--worktree new-child`.
- **Preconditions.** The orchestration experimental feature must be enabled
  (Settings > Experimental) and the runtime running (`orca status --json`).

## Possible next step
Once the flags are confirmed, this coordinator prompt can be replaced by a thin
driver **script** (`orca orchestration ...` calls) for fully deterministic control
of the loop and gate — the role briefings and tickets stay exactly as they are.
