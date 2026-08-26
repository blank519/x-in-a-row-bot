# Ticket pipeline for Pi (planner -> implementer -> evaluator)

A self-correcting, three-role agent pipeline for this repo, built on Pi's
subagent mechanism:

1. **planner** reads a *ticket* and produces an actionable plan.
2. **implementer** executes the plan (code changes, or launching training runs).
3. **evaluator** checks the result against the ticket and emits `VERDICT: PASS`
   or `VERDICT: FAIL` + feedback. On `FAIL`, the orchestrator loops back to the
   implementer with that feedback, up to `max_iterations`.

This is the **config-only ("Tier 1") implementation**: the loop is driven by the
main Pi session (the LLM), using project config only — no TypeScript. See
[Upgrade path](#upgrade-path) for the deterministic version.

## Files

```
.pi/
├── agents/
│   ├── planner.md        # ticket -> plan (read-only)
│   ├── implementer.md    # executes the plan (full tools)
│   └── evaluator.md      # judges vs "Done when"; emits VERDICT: PASS|FAIL
├── skills/
│   ├── pipeline/SKILL.md      # the orchestrator loop
│   ├── experiments/SKILL.md   # on-demand: running runs + mlruns analysis
│   └── code-changes/SKILL.md  # on-demand: verification / test suite
├── tickets/
│   ├── TEMPLATE.md
│   ├── EXAMPLE-code.md
│   └── EXAMPLE-experiment.md
└── README.md             # this file
```

The agents branch on the ticket's `type:` (`code` vs `experiment`) and read the
matching on-demand skill: `.pi/skills/experiments/SKILL.md` (running runs +
`mlruns/` analysis — per-(heuristic, side) win-rate + episode-length trajectories
vs. a baseline) or `.pi/skills/code-changes/SKILL.md` (verification). Repo-root
`AGENTS.md` stays lean and holds only always-relevant orientation.

## Prerequisites

1. **Install a subagent extension.** The planner/implementer/evaluator run as
   delegated Pi subagents, which requires Pi's subagent (or workflow) extension —
   it is an example extension, not part of core Pi. Install it into
   `~/.pi/agent/extensions/` per its README:
   <https://github.com/earendil-works/pi/tree/main/packages/coding-agent/examples/extensions/subagent>

2. **Enable project-local agents.** By default the subagent tool only loads
   user-level agents (`~/.pi/agent/agents/`). To use the agents in `.pi/agents/`,
   configure the subagent tool with `agentScope: "project"` (or `"both"`), and
   **trust this project** (`/trust`) when prompted. Only do this because you
   control this repo.

3. **Model:** the agents omit a `model:` in their frontmatter, so each subagent
   inherits the dispatching session's active model and thinking level — i.e. your
   locally-served Qwen. No cloud model is pinned.

## Running it

From a Pi session at the repo root:

- Invoke the skill: `/skill:pipeline`, then give it a ticket path, or
- Just ask: `run the pipeline on .pi/tickets/EXAMPLE-experiment.md`.

The orchestrator will plan, then implement/evaluate in a loop until PASS or
`max_iterations` (default 3), then report.

Write new tickets by copying `tickets/TEMPLATE.md`. The most important field is
**Done when** — it is the evaluator's target, so make it concrete and checkable.

## Upgrade path

To make the loop deterministic (hard retry cap, verdict parsing, and step
sequencing enforced in code rather than by the model), reimplement the
orchestrator as a Pi **`workflow-core`** TypeScript extension under
`.pi/extensions/`. It would spawn the same three agents, parse the evaluator's
`VERDICT:` line programmatically, and loop with an explicit counter — while
keeping the agent `.md` files and tickets exactly as they are here. The
config-only version and the extension version share all of this config; only the
control flow moves into code.
