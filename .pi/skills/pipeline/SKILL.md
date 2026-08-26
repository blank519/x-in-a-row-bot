---
name: pipeline
description: Run the planner -> implementer -> evaluator loop on a ticket file. Use when asked to "run the pipeline", "work a ticket", or process a file under .pi/tickets/.
---

# Ticket pipeline (planner -> implementer -> evaluator)

You are the **orchestrator**. Drive a self-correcting loop over three subagents
using the **subagent tool** (in single mode, one call at a time, so you can branch
on the evaluator's verdict). The three agents are defined in `.pi/agents/`:
`planner`, `implementer`, `evaluator`.

## Inputs
The user gives you a **ticket** file path (usually under `.pi/tickets/`). If they
don't, ask for it. Read the ticket yourself so you know its `type:`
(`code` or `experiment`), its **Done when** criteria, and its `max_iterations`
(default **3** if not specified).

## Procedure

1. **Read the ticket** at the given path.

2. **Plan.** Call the `planner` subagent with the ticket path/contents. Keep the
   returned plan; it is the shared reference for the loop.

3. **Loop** for up to `max_iterations`:
   a. **Implement.** Call the `implementer` subagent with: the ticket, the plan,
      and — on any attempt after the first — the evaluator's `FEEDBACK` from the
      previous iteration (state clearly that fixing it is the priority).
   b. **Evaluate.** Call the `evaluator` subagent with: the ticket (its **Done
      when** criteria) and the implementer's report from step (a).
   c. **Read the verdict.** Parse the evaluator's final `VERDICT:` line.
      - `PASS` -> stop the loop; go to step 4.
      - `FAIL` -> carry its `FEEDBACK` into the next iteration's implementer call.

4. **Report to the user:**
   - PASS: summarize what was done and the evidence the evaluator cited.
   - Still FAIL after `max_iterations`: stop (do **not** loop forever). Summarize
     progress and report the outstanding `FEEDBACK` so the user can intervene.

## Rules
- Use **single-mode** subagent calls and inspect each result before the next step
  — the conditional loop-back cannot be expressed as a fixed chain.
- Rely on the subagents' **isolated context**: pass concise handoffs (plan,
  report, feedback, run ids), not giant transcripts. Keep your own context lean.
- For `experiment` tickets, remember runs are long: the implementer may launch
  them in the background, so evaluation may need to wait for a run to progress.
  Don't fabricate results — only pass the evaluator real run ids to analyze.
- Never let a subagent weaken tests or edit `mlruns/` to force a PASS.

## Note on this design
This is the config-only ("Tier 1") orchestrator: the loop is driven by you, the
LLM. For deterministic looping, a hard retry cap enforced in code, and verdict
parsing outside the model, upgrade to a `workflow-core` TypeScript extension — see
`.pi/README.md` ("Upgrade path").
