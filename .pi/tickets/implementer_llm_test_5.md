# Evaluate different LLMs for the implementer agent

type: code change
max_iterations: 1

## Goal
For every experiment work unit mentioned in `LLM_test_plan.md`, create 6 different implementer 
agents, each using a different LLM to implement the experimental change. Instead of launching 
the runs, simple analyze and compare their respective costs, how well they follow the plan, 
whether they make excessive changes, and whether or not the run would fail on launch. Make
a ranking based on these criteria.

## Constraints
- Skip the planner stage, since we are only evaluating the implementer agents.
- For each work unit, create 6 implementers, each using a different one of the following LLMs
  through OpenRouter:
  - GLM 5.3
  - GLM 5.3 Flash
  - Qwen 3.8 Max
  - Qwen 3.8 27B
  - DeepSeek V4 Pro
  - DeepSeek V4 Flash
- Each implementer should act as if they are in an experiment pipeline.
- Each implementer should create a copy of the current `train_ppo_gomoku.py` file with a 
  descriptive name.
- Each implementer **MUST NOT** open, read, cat, diff, copy, or byte-duplicate any other 
  worker's file or report, only the original `train_ppo_gomoku.py`. 
- Every worker agent must include in its report a section describing what agent tools were used
  and how they were used.
- DO NOT LAUNCH ANY OF THE CREATED RUN FILES.

## Done when
- All implementer agents have completed their tasks and provided reports.
- A ranking of the implementer agents based on the evaluation criteria has been created.
