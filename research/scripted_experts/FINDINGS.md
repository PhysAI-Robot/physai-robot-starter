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

## Randomized pick-and-place (study 1, M1)

Cube and target both random over the reachable region (`configs/manifests/so101_randomized_pick_place.yaml`:
x 0.14-0.27, |y| <= 0.14, 0.16-0.255 m from the base, at least 8 cm apart). The first
run scored 295/300 on seeds 1000-1299, with 5 timeouts, no collisions and no refused actions.
Two causes, both found by tracing `pinch_center` and the cube's contacts per step:

1. **The jaw swept the cube out of reach from HOME.** HOME's static jaw sits at
   x about 0.21 and low; a direct joint-space move to the hover pose dragged it through a
   cube spawned at x 0.17-0.20 near the centre line, shoving the cube to x about 0.11,
   where IK no longer converges. Fix: a `RISE` phase straight up to hover height first
   (wrist left as it is: turning it top-down from HOME is itself the sweep), also after a
   missed grasp. All 5 failing held-out seeds and both failures on 0-299 passed on rerun, but a full rerun still failed 4 of 300 held-out and 1 of 300 training seeds (cause 2).
2. **DESCEND started before the wrist had turned top-down.** After `RISE` the pinch point
   reaches the hover while the rate-limited wrist is still rotating, so the `reached`
   test passed early and a jaw landed on the cube's top face (seeds 1047, 1055, 1209 and 1230
   held out, 296 on 0-299). Fix: APPROACH also waits until the commanded joints are
   within 0.02 rad of the IK solution.

| Variant | Seeds | Before | After |
| --- | --- | --- | --- |
| randomized | 1000-1299 | 295/300 | 300/300 |
| randomized | 0-299 | 299/300 (after fix 1) | 300/300 |
| fixed target (regression) | 0-299 | 300/300 | 300/300 |
| sorting (regression) | 0-299 | 299/300 (seed 95) | 300/300 |

The `RISE` phase first broke sorting seed 95: a cube at r = 0.268 m has no top-down hover
pose, so the old sequence timed out APPROACH and DESCEND swept in from HOME, and rising
first changed where that sweep started. `RISE` is now skipped when the hover pose is out of
top-down reach, which restores the old path for that cube.

Expert efficiency reference on 1000-1299 (median over successes): 6.5 s, 0.40 m of
gripper path, RMS jerk 105 m/s^3 (`completion_time_s`, `path_length_m`, `rms_jerk` in the
`eval_policy.py` JSON).

The visible target disc did not move on MuJoCo before this check: domain randomization
restored every geom position after the layout had moved the pad, so the task scored the
site while cameras saw a disc at the scene default. State-based tests could not see it;
`show_target()` now places the pad after randomization, with a test. The expert and its
numbers are unaffected (the pad has no collision).

Isaac Sim: the spawn is the same on both engines (targets identical, cubes within 3 mm
over seeds 1000-1009; the difference is the jaw touching the cube during the one reset
hold step). The expert itself is not evaluated on Isaac: it reads MuJoCo contact data
directly, and `eval_policy.py --sim isaac` accepts only `visual_servo`, `constant` and `lerobot`.

## Why the protocol needs at least 100 seeds

A 20-seed run cannot resolve a policy's reliability (the pre-fix expert scored 45% and
60% on two seed ranges). [study 1's evaluation protocol](../studies/01_randomized_pick_place.md#evaluation-protocol-shared-by-all-methods) and the
README results table both rest on this.
