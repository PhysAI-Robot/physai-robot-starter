# Scripted experts

Privileged-ground-truth expert policies used to generate demonstrations for
imitation learning. "Privileged" means the policy reads simulator ground
truth (e.g. object pose from MuJoCo directly) that a deployed policy would
not have access to — this is what makes it an "expert" rather than a
perception-driven baseline.

- `so101_pick_place_expert.py` — scripted pick-and-place expert for SO-101,
  driven by `scripts/collect_demos.py`.

Registers itself with `physai.robots.registry` / `physai.policy.registry` on
import; core never imports this package.

## SO-101 scripted pick-and-place baseline

Run one episode:

```bash
uv run python scripts/eval_policy.py --policy scripted --episodes 5 --seed 0 --max-steps 500
```

Run the documented deterministic reliability check (at least 100 seeds; see
below for why 20 is not enough):

```bash
uv run python scripts/eval_policy.py --policy scripted --episodes 100 --seed 0 --max-steps 600
```

See [Results](#results) for the current success rates.

Use the same seed when comparing parameter changes. Save local results when
needed:

```bash
mkdir -p outputs/local
uv run python scripts/eval_policy.py --policy scripted --episodes 5 --seed 0 --max-steps 500 --json-out outputs/local/so101_scripted_seed0.json
```

Inspect the policy interactively in the browser (see the
[Web Viewer Runbook](../../docs/WEB_VIEWER_RUNBOOK.md) for the full
workflow):

```bash
uv run python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml --policy scripted --serve --seed 0
```

## Results

Measured 2026-09-25 on commit `71b95ec`: deterministic scene, no
randomization, `--max-steps 600`, seeds starting at 0, Wilson 95% intervals.
These are simulation results that depend on the calibrated jaw pads; they are
not hardware or randomized results. This table is the one place the project's
success rates are recorded; other documents link here.

| Policy | Task | Seeds | Success | 95% CI | Failures |
| --- | --- | --- | --- | --- | --- |
| `scripted` | single cube | 300 | 300/300 (100%) | [98.7%, 100%] | none |
| `scripted` | sorting (`--sorting`) | 300 | 300/300 (100%) | [98.7%, 100%] | none |
| `visual_servo` | single cube | 100 | 95/100 (95%) | [88.8%, 97.8%] | 5 timeouts (seeds 13, 15, 28, 64, 76); no collisions or unsafe actions |

Reproduce with
`uv run python scripts/eval_policy.py --policy <name> --episodes <n> --seed 0 --max-steps 600`
(add `--sorting` for the sorting task). Use at least 100 seeds when comparing
changes: a 20-seed run could not resolve the pre-fix scripted policy, which
scored 45% on seeds 0-19 and 60% on seeds 100-149
([FINDINGS.md](FINDINGS.md)).

The `visual_servo` rate is below the 100% recorded on 2026-09-21. That
measurement predates the fingertip pad refit (`fad205d`); the cause of the
timeouts is not yet diagnosed and is tracked in
[ROADMAP.md](../../ROADMAP.md).

## Collecting demonstrations

This expert drives `scripts/collect_demos.py` for imitation-learning data
collection — see
[research/imitation_learning/README.md](../imitation_learning/README.md) for
the full collect -> train -> evaluate workflow.

## Findings

The root cause of the pre-2026-09-21 failures, the fixes that worked, and the
ideas that were tried and reverted are recorded in
[FINDINGS.md](FINDINGS.md). The rule that came out of it: evaluate with at
least 100 seeds, because a 20-seed run scored 45% and 60% on two seed ranges
of the same pre-fix policy.
