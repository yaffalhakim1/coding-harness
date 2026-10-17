# Replicating the paper's findings

This directory holds the scripts and fixtures used to reproduce the results in
`task.md`.

## What is replicated

| Paper claim | How it is measured | Command |
|---|---|---|
| Context management prevents overflow, most valuable at tight budgets | T0 vs T1–T4 pass rate and overflow count across window sizes | `harness ablate` |
| T4 (elision → summarization) is the most efficient tier | Summarization-call count and total tokens per tier | `harness ablate` |
| Planning saves cost for capable models | T4 with `--planning true` vs `false` | `harness ablate` |
| bash-only lowers cost for bash-capable models | T4 predefined vs `bash-only` | `harness ablate` |
| Failure stages differ by capability | `harness analyze` per trajectory | `harness analyze` |

## Quick run (tight budget)

The clearest signal appears at a window tight enough to bind. With the 3-task
fixture:

```bash
# 1k window, all five tiers + two component ablations (7 settings)
harness ablate \
  --tasks bench/tiny-tasks.json \
  --config config/bench-tight.toml \
  --windows 1000 \
  --work-dir /tmp/ablate-work \
  --results-dir /tmp/ablate-results
```

Expected shape of the result:

```
setting                        pass     rate   mean tok mean turn overflow
T0/plan/1k/predefined          0/3      0.0%       2734         0        3
T1/plan/1k/predefined          3/3    100.0%      26406         5        0
T2/plan/1k/predefined          3/3    100.0%      29200         6        0
T3/plan/1k/predefined          3/3    100.0%      29046         6        0
T4/plan/1k/predefined          3/3    100.0%      27064         5        0
T4/noplan/1k/predefined        3/3    100.0%      18037         4        0
T4/plan/1k/bash-only           3/3    100.0%      18683         6        0
```

Three claims are visible in that table alone:

1. T0 fails every task with `context_overflow`; every managed tier passes.
2. T4 matches T1–T3 on accuracy while issuing fewer summarization calls than T3
   (14 vs 15 in the recorded run) — staging elision first is the efficiency win.
3. Planning off and bash-only are both cheaper than the T4 baseline, which is
   the paper's cost finding for capable models.

## Full grid

`--windows 32000,64000,96000,128000` produces the paper's 22 settings per
benchmark (20 context settings + 2 component ablations). It is expensive: each
setting is a full task sweep.

## Inspecting a single run

```bash
harness analyze --trajectory /tmp/ablate-results
```

Prints the failure stage (file localization → line localization → patch
implementation → verification), phase counts, and termination reason, which is
the paper's §C.1 attribution.

## Output files

`--results-dir` receives, per setting:

- `<benchmark>-<tier>-<window>k-<plan|noplan>-<action-space>.json` — full report
  including per-task rows
- the same name with `.csv` — one row per task, for plotting

Trajectories land in `--trajectory-dir`, one JSON per session.

## Caveats

- The fixture has 3 tasks. It demonstrates the mechanism; it cannot establish
  statistical significance. The paper uses 500 SWE-Bench and 89 Terminal-Bench
  tasks with McNemar tests.
- Isolation is a per-task directory, not the paper's container. Tasks that
  install packages or mutate global state are not safely parallelisable here.
- Model differences dominate. `deepseek-v4-1-flash` is not one of the paper's
  four models, so absolute numbers are not comparable — only the direction of
  each effect is.
