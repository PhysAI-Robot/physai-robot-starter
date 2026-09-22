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
