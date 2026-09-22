# Classical control

Closed-loop control baselines that do not use a learned model: visual servo,
trajectory generation, and similar techniques.

- `so101_visual_servo.py` — calibrated-camera color-blob visual servo policy
  for SO-101.

Registers itself with `physai.policy.registry` on import; core never imports
this package.
