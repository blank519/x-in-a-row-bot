# Experiment evidence readiness

A run is evidence-ready when either:

1. it completed normally at `total_timesteps` and saved its final output; or
2. the ticket's target metrics clearly plateaued over a sustained recent window
   with no continuing upward or downward trend.

Completion makes a run evaluable, not successful. A couple of checkpoints cannot
demonstrate convergence. Read metric files in step order and pair rates with
corresponding episode lengths. A ticket may impose a stricter
`evidence_ready_condition`.

A live incomplete run with too little or still-trending evidence is not ready.
`HOLD` records this evidence state; it is neither PASS nor FAIL and should state
which trajectories are missing or immature and a reasonable recheck interval.

A process that died with little or no usable evidence is a failed run, not an
immature one. Confirm this using process state and logs. A live but permanently
stuck run should be bounded by the planned runtime plus a grace period rather
than treated as indefinitely pending.

Prefer waiting until the expected batch-completion time before the first full
observation rather than polling. For reference, a 10M-step run may take roughly
8 hours. Readiness decisions should come from the trajectory evidence, not the
wall-clock estimate.
