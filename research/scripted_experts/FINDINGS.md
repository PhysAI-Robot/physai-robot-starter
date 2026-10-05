# Scripted expert findings

What was measured and tried behind the scripted expert's current behavior. Commands
and current results are in [README.md](README.md).

## The 2026-09-21 reliability fix

The expert was far weaker than the recorded `20/20`: the same configuration scored
45% on seeds 0-19 and 60% on seeds 100-149. Three causes:

1. **No wrist-orientation term.** `ExpertConfig.approach_dir` defaulted to `None` and
   `_solve()` passed it to `ArmKinematics.ik_pinch()`, overriding its own `TOP_DOWN`
   default. The redundant arm met the position target with the wrist at an arbitrary
   tilt, and the moving jaw (farther from the wrist) swung clear of the cube at
   squeeze (pads 20-50 mm off, 17-20 mm high on failures against about 16 mm and a few
   mm on successes). Defaulting it to `TOP_DOWN` took single-cube to 100% and sorting
   from 46% to 90%. `visual_servo` always had the constraint, which is why it was
   already at 100%.
2. **Grip too shallow.** Sorting's remaining gap was traced with `mj_contactForce`:
   at `gripper_grip=0.19`, barely past `gripper_touch=0.21`, pad force was 0.1-1 N
   (the actuator can give about 2.94 N), so the cube slipped out during TRANSFER's
   acceleration. With a position-controlled jaw the squeeze comes entirely from how far
   past first contact the target sits. `gripper_grip=0.06` closed most of the gap.
3. **Stale grasp point in sorting (6x spread).** Failures by spawn band showed the
   middle band, flanked by cubes on both sides, failing 12.6% against 1.9% and 4.0%.
   APPROACH's wide jaw sweep (79 mm against 28 mm cubes 6 cm apart) nudges a neighbor
   in every middle-band episode; when it nudges the target, `_grasp_xy`, locked once at
   the start of APPROACH, is stale (cube drift 6 mm median 0 in successes, 79 mm median
   70 in failures). Refreshing it once at the APPROACH-to-DESCEND handoff took sorting
   from 94% to 98% over 900 held-out seeds; pick-place is unaffected.

| Policy | Variant | Success before | Success after |
| --- | --- | --- | --- |
| `scripted` | single cube | 59% [49%, 68%] (100 seeds) | 100% [98%, 100%] (300 seeds) |
| `scripted` | sorting | 46% [33%, 60%] (50 seeds) | 98% over 900 seeds, now 300/300 on seeds 0-299 |

Also fixed: grasp detection looked up a hardcoded `cube_geom`, which does not exist in
sorting (`cube_red`, `cube_blue`, `cube_yellow`), so no grasp was ever detected and the
expert retried to timeout (regression test added).

## Tried and reverted

| Idea | Result |
| --- | --- |
| Narrow the APPROACH/DESCEND aperture to just clear the cube | sorting 94% to 81%: jaws clip the target cube itself |
| Pad-geometry-aware IK offset in `ArmKinematics` instead of the calibrated `PINCH_OFFSET` | about 98.7% to 85% on 150 seeds: the pad-geom midpoint is not the reference the constant represents |
| Remove or raise `gripper_force_limit` (0.3-2.0 N tested) | 2-5 points worse for the expert uncapped, but uncapped breaks `visual_servo` (a shallow aperture lets force reach about 2.94 N and destabilizes the contact), so the shared 0.3 stays |
| Pad friction, pad size, a LIFT settle dwell | no effect or worse |
| Require both pads in contact before accepting a grasp | sorting 94% to 91.7%: extra retries burn step budget into timeouts |

Later, the grasp pads were refitted to the fingertips (12 x 12 x 6 mm, flush with each
tip's inner face, moving pad tilted about 8 degrees; see `ManipulationSceneConfig`).
First touch moved from about 0.21 to about 0.176 normalized aperture, so
`visual_servo`'s old 0.19 squeeze stopped squeezing and became `SQUEEZE_GRIP = 0.15`;
the expert's deep 0.06 is unaffected. The refit exposed friction creep (a held cube
slid at about 1 mm/s whatever the grip force), so manipulation scenes now use MuJoCo's
no-slip pass (`noslip_iterations = 5`). Narrower 8-10 mm pads matched the tip more
tightly but cost `visual_servo` one seed in twelve.

What remains (98% before the 300/300 re-measure) was believed to be a small task and
physics floor; no retry-on-drop was added.

## Why the protocol needs at least 100 seeds

A 20-seed run cannot resolve a policy's reliability (the pre-fix expert scored 45% and
60% on two seed ranges). [ROADMAP.md](../../ROADMAP.md)'s evaluation protocol and the
README results table both rest on this.
