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

The current baseline is **100%** over 100 seeds (95% CI [96%, 100%], measured
2026-09-21), matching the deterministic `visual_servo` policy (see
[research/classical_control/README.md](../classical_control/README.md)). Note
that a 20-seed run could not resolve the pre-fix policy — the same
configuration scored 45% on seeds 0-19 and 60% on seeds 100-149 — so still
use at least 100 seeds when comparing changes. The root cause and fix are
recorded below. Results depend on the calibrated jaw pads and the
deterministic scene; they are not hardware or randomized results.

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
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --policy scripted --serve --seed 0
```

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
