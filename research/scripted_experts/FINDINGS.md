# Scripted expert findings

Measurements and reverted experiments behind the scripted expert's current
behavior. The runbook and current results are in [README.md](README.md).

## The 2026-09-21 reliability fix

Opened 2026-09-20, resolved 2026-09-21. The scripted expert was far weaker
than the `20/20` previously recorded. Root cause and fix:

`ExpertConfig.approach_dir` defaulted to `None` and `_solve()` passed it to
`ArmKinematics.ik_pinch()` explicitly, which **overrode** `ik_pinch()`'s own
`TOP_DOWN` default — so the scripted expert solved IK with no wrist-orientation
term at all. The redundant arm could satisfy the pinch-point position target
with the wrist at an arbitrary tilt; the moving jaw sits farther from the
wrist than the static one, so a few degrees of unwanted tilt was enough to
swing it clear of the cube at squeeze time (pads measured 20-50 mm off and
17-20 mm high on failing episodes, versus ~16 mm and a few mm on successes).
`visual_servo` never passed `approach_dir`, so it always got the `TOP_DOWN`
constraint — which is why "the camera baseline is 100% cheating knows the
coordinates but still can't beat it" was the right question to ask. Setting
`ExpertConfig.approach_dir` to default to `TOP_DOWN` closed the gap with no
other change (`approach_rate` stayed at its original 0.5).

The orientation fix alone brought single-cube success to 100% and sorting to
90% (from 46%). Sorting's remaining gap was then traced with
`mujoco.mj_contactForce`: the cube was grasped correctly and lifted, but the
pad normal force at `gripper_grip=0.19` measured only **0.1-1 N** (the
actuator can produce up to ~2.94 N), so the grip gradually let go during
TRANSFER's acceleration and the cube fell back to the table. The cause is
`gripper_grip=0.19` sitting barely past `gripper_touch=0.21` — with a
position-controlled jaw, squeeze force against a rigid cube comes entirely
from how far past first contact the commanded position sits, and 0.02 of
normalized aperture was not far enough. Deepening the squeeze target to
`gripper_grip=0.06` raised the measured contact force and closed most of the
gap, with **no other change** (pad friction, pad size, and a LIFT settle dwell
were all tried and made things worse or did nothing — see below).

Measured over held-out seeds with Wilson 95% intervals, original vs. final:

| Policy | Variant | Seeds | Success (original) | Success (final) |
| --- | --- | --- | --- | --- |
| `scripted` | Single cube | 100 | 59% [49%, 68%] | **100% [98%, 100%]** (300 seeds) |
| `scripted` | Sorting | 50 | 46% [33%, 60%] | **98% [97%, 99%]** (900 seeds) |
| `visual_servo` | Single cube | 20 | 100% [84%, 100%] | 100% (unchanged) |

The single-cube result matches `visual_servo` exactly (200/200 through the
production `scripts/eval_policy.py` path, zero collisions/timeouts/unsafe
actions across every validation run), so the scripted expert is now a valid
upper bound and demonstration teacher.

**A third sorting-specific mechanism, found after the first two fixes.** With
the grip force fixed, sorting still sat at 94% while pick-place — same robot,
same expert code, same physics — was at 100%. The two tasks differ only in
scene layout: sorting places three cubes 6 cm apart; pick-place has one cube
on an otherwise empty table. Bucketing sorting failures by which of the three
spawn bands held the target cube showed a 6x spread (middle band, flanked by
a neighbor on both sides, failed 12.6% of the time vs. 1.9% and 4.0% for the
two edge bands, which only have one neighbor each) — ruling out workspace
reach and pointing at the neighbors specifically. Direct contact logging
confirmed the mechanism: the arm's pads touch a neighboring cube in every
single middle-band episode, success or failure alike, during APPROACH's wide
lateral sweep (jaws command `gripper_open`, spanning 79 mm against 28 mm
cubes 6 cm apart). That contact is usually harmless, but `_grasp_xy` is
locked once at the start of APPROACH and never revisited, so on the episodes
where the nudge lands on the *target* cube, every later phase keeps aiming at
a now-stale position: cube drift from the locked aim point averaged 6 mm
(median 0 mm) in successes vs. 79 mm (median 70 mm) in failures. Refreshing
`_grasp_xy` once — at the APPROACH-to-DESCEND handoff, not continuously —
took sorting from 94% to 98% over 900 held-out seeds across three seed
ranges, with pick-place (nothing nearby to nudge) unaffected.

A companion idea — narrowing the APPROACH/DESCEND aperture to just clear the
cube (11 mm margin at `gripper_grip=0.25` vs. 51 mm at the default
`gripper_open=0.55`) — was tried on the reasoning that a narrower jaw would
be less likely to reach a neighbor in the first place. It measurably made
things worse (94% -> 81%): removing the wide-open slack made the jaws far
more likely to clip the *target* cube itself during approach, given the
system's existing positioning precision, which is a worse failure than the
one it was meant to prevent. Not applied; `gripper_open` stays `0.55` for
APPROACH/DESCEND.

Two more things tried during the sorting work did not survive validation and
were reverted:

- **A pad-geometry-aware IK offset regressed and was reverted.** Suspecting
  `PINCH_OFFSET` (a fixed constant, measured accurate to ~3 mm near a 0.19
  aperture but drifting past 10 mm by 0.06) was still costing accuracy,
  `ArmKinematics` was extended to compute the offset from the live pad geoms
  instead. It measurably made things worse (98.7% -> ~85% on a 150-seed
  check): the geometric midpoint of the pad geoms' origins is not the same
  reference point the calibrated constant represents, so "exact" was exact
  for the wrong target. Fully reverted; `kinematics.py` is unchanged from
  before this investigation.
- **`gripper_force_limit` (a pre-existing `EnvConfig` field, default `0.3`)
  turned out to be load-bearing for a different policy.** With the deep
  0.06 squeeze, contact force is already low without any cap (~0.1-1 N
  measured via `mj_contactForce`), and a fair 300-seed sorting comparison
  showed every tested cap value (0.3-2.0 N) doing 2-5 points *worse* than no
  cap at all (96.0% uncapped vs 92.7-94.7% capped). But `visual_servo` uses
  its own, shallower, hardcoded 0.19 aperture, and disabling the cap broke
  `test_visual_servo_pick_place_settles_from_multiple_seeds` outright: at
  that shallow aperture, uncapped force can reach the actuator's full ~2.94 N
  rating against a rigid cube and destabilize the contact, which is exactly
  what the cap exists to prevent. The shared default was kept at `0.3` to
  keep `visual_servo` correct.
- **The grasp pads were later refitted to the fingertips** (flush with each
  tip's inner face and along its angle, centred on the finger, 12 x 12 x 6 mm,
  one 28 mm cube apart at the pad centres; the moving pad is tilted ~8
  degrees to follow its finger's face; see `ManipulationSceneConfig`), and the web viewer stopped
  overriding the torque cap. First touch of the cube moved from about 0.21 to
  about 0.176 normalized aperture, so the hardcoded 0.19 above no longer
  squeezed at all (`visual_servo` dropped the cube on every seed) and became
  `SQUEEZE_GRIP = 0.15` in `so101_visual_servo.py`; the expert's deep 0.06
  squeeze is unaffected, and its `gripper_touch=0.21` now just stops a little
  short of contact. The refit also exposed friction creep (a held cube slid out at ~1 mm/s
  under its own weight whatever the grip force, which the old ~3 mm
  "drift in hand" figures had been hiding); manipulation scenes now enable
  MuJoCo's no-slip solver (`noslip_iterations = 5`). Both policies were
  rechecked afterwards (expert 20/20,
  `visual_servo` 12/12 seeds). Narrower 8-10 mm pads matched the tip more
  tightly but cost `visual_servo` one seed in twelve, because it grasps up to
  about 15 mm off the pinch centre.
- **Requiring both pads in contact before accepting a grasp** (tightening
  `_cube_grasped()`, which currently accepts either pad touching) was tried
  on the theory that a one-sided contact — measured directly on one failing
  seed as 0 N on one pad, 4.2 N on the other — should trigger a retry before
  ever lifting. It made sorting *worse* (94% -> 91.7%): the stricter check
  also retries some contacts that would have held fine, and each retry
  re-runs the full APPROACH-to-SQUEEZE cycle, so the added retries burned
  step budget into new timeouts faster than they fixed weak grasps.

What remains after the grasp-refresh fix (98% over 900 seeds) is believed to
be a small, genuine task/physics floor: a handful of seeds still fail
regardless of `gripper_grip` depth (0.06-0.19) or the force cap, and
`visual_servo` -- independent grip logic, unaffected by `ExpertConfig` --
fails on some of the same ones. No retry-on-drop was added anywhere in this
investigation.

Two things were established while investigating, and they still shape
[ROADMAP.md](../../ROADMAP.md)'s Phase 2.0 protocol:

1. **A real bug was fixed along the way.** Grasp detection looked up a
   hardcoded `cube_geom`, which does not exist in the sorting scene (its
   cubes are `cube_red`, `cube_blue`, `cube_yellow`). The lookup returned -1,
   no grasp was ever detected, and the expert retried until timeout. Covered
   by a regression test.
2. **The 20-seed protocol cannot measure this policy.** The same
   configuration scored 45% on seeds 0-19 and 60% on seeds 100-149 before the
   fix. The historical `20/20` was partly sampling luck. This is why
   ROADMAP.md's Phase 2.0 requires at least 100 seeds.
