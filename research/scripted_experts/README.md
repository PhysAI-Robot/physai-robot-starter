# Scripted experts

Privileged-ground-truth expert policies used to generate demonstrations for imitation
learning. "Privileged" means the policy reads simulator state (such as object pose
from MuJoCo) that a deployed policy would not have, which is what makes it an expert
rather than a perception-driven baseline.

- `so101_pick_place_expert.py`: scripted pick-and-place expert for SO-101, driven by
  `scripts/collect_demos.py`.

It registers itself with `physai.robots.registry` and `physai.policy.registry` on
import; core never imports this package.

## Run and evaluate

```bash
uv run python scripts/eval_policy.py --policy scripted --episodes 5 --seed 0 --max-steps 500
uv run python scripts/eval_policy.py --policy scripted --episodes 100 --seed 0 --max-steps 600
uv run python scripts/eval_policy.py --policy scripted --episodes 5 --seed 0 --json outputs/local/so101_scripted_seed0.json
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy scripted --serve --seed 0
```

The last command opens the policy in the browser viewer
([runbook](../../docs/WEB_VIEWER_RUNBOOK.md)). Add `--sorting` to the evaluation for the
three-cube task. Use the same seed when comparing parameter changes, and at least 100
seeds: a 20-seed run could not resolve the pre-fix policy
([FINDINGS.md](FINDINGS.md#why-the-protocol-needs-at-least-100-seeds)).

## Results

The default task is a fixed target with the cube drawn from a small range; the
randomized rows say so. `--max-steps 600`, Wilson 95% intervals. These are simulation
results that depend on the calibrated jaw pads, not hardware results. This table is the one place the project's success rates
are recorded; other documents link here.

| Policy | Task | Simulator | Seeds | Success | 95% CI | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| `scripted` | single cube | MuJoCo | 0-299 | 300/300 (100%) | [98.7%, 100%] | none |
| `scripted` | sorting (`--sorting`) | MuJoCo | 0-299 | 300/300 (100%) | [98.7%, 100%] | none |
| `scripted` | [randomized pick-place](../studies/01_randomized_pick_place.md) | MuJoCo | 0-299 | 300/300 (100%) | [98.7%, 100%] | none |
| `scripted` | randomized pick-place, held out | MuJoCo | 1000-1299 | 300/300 (100%) | [98.7%, 100%] | none |
| `visual_servo` | single cube | MuJoCo | 0-99 | 100/100 (100%) | [96.3%, 100%] | none |
| `visual_servo` | single cube | Isaac Sim | 0-99 | 100/100 (100%) | [96.3%, 100%] | none |
| `visual_servo` | single cube, held out | MuJoCo | 100-149 | 50/50 (100%) | [92.9%, 100%] | none |
| `visual_servo`, target from camera | single cube | MuJoCo | 0-99 | 100/100 (100%) | [96.3%, 100%] | none |
| `visual_servo`, target from camera | randomized pick-place, held out | MuJoCo | 1000-1299 | 296/300 (98.7%) | [96.6%, 99.5%] | 4, cube near the base on the centre line |
| `lerobot` (ACT, 200 demos, chunk 100, 60k steps) | randomized pick-place, held out | MuJoCo | 1000-1299 | 294/300 (98.0%) | [95.7%, 99.1%] | 6 timeouts |
| `lerobot` (ACT, 100 demos, chunk 100, 60k steps) | randomized pick-place, held out | MuJoCo | 1000-1299 | 275/300 (91.7%) | [88.0%, 94.3%] | 25 timeouts |
| `lerobot` (ACT trained on Isaac images, 100 demos, chunk 100, 60k steps) | randomized pick-place, held out | MuJoCo | 1000-1299 | 229/300 (76.3%) | [71.2%, 80.8%] | 71 timeouts |
| `lerobot` (ACT trained on Isaac images, 100 demos, chunk 100, 60k steps) | randomized pick-place, held out | Isaac Sim | 1000-1099 | 85/100 (85%) | [76.7%, 90.7%] | 15 timeouts |
| `lerobot` (ACT, 200 demos, chunk 100, 60k steps) | randomized pick-place, held out | Isaac Sim | 1000-1099 | 58/100 (58%) | [48.2%, 67.2%] | 42 timeouts |
| `lerobot` (ACT, 100 demos, chunk 100, 60k steps) | randomized pick-place, held out | Isaac Sim | 1000-1099 | 55/100 (55%) | [45.2%, 64.4%] | 45 timeouts |
| `visual_servo`, target from camera | randomized pick-place, held out | Isaac Sim | 1000-1299 | 271/300 (90.3%) | [86.5%, 93.2%] | 29, 26 with the cube within 0.19 m of the base |

The scripted rows were measured on 2026-09-25 (commit `71b95ec`); the `visual_servo`
rows on 2026-10-03 and later (the camera-target rows on 2026-10-07), after the pad refit and the timing and perception fixes
([classical control findings](../classical_control/FINDINGS.md)).
Reproduce with `uv run python scripts/eval_policy.py --policy <name> --episodes <n> --seed 0 --max-steps 600`.

## Collecting demonstrations

This expert drives `scripts/collect_demos.py`; the collect, train and evaluate workflow
is in [research/imitation_learning/README.md](../imitation_learning/README.md).

The root cause of the pre-2026-09-21 failures, the fixes and the reverted ideas are in
[FINDINGS.md](FINDINGS.md).
