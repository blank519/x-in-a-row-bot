# Ticket contract

A ticket under `.pi/tickets/` is the pipeline's source of truth. It contains:

- `type`: exactly `code` or `experiment`.
- `max_iterations`: optional positive integer; default 3. HOLD does not consume
  an iteration.
- `Goal`: desired outcome and why it matters.
- `Constraints`: required boundaries.
- `Done when`: explicit, observable pass criteria.
- Experiment tickets additionally identify a hypothesis, immutable baseline run,
  and preferably an evidence-ready condition.

Workers may make criteria more concrete but must not weaken, silently reinterpret,
or replace them. Raise ambiguity in the plan or escalate it.

## Verdict semantics

- PASS ends the pipeline.
- FAIL sends actionable feedback to a new implementation attempt and consumes one
  iteration.
- HOLD is valid only for an immature experiment; orchestration waits and
  reevaluates without rerunning implementation or consuming an iteration.
