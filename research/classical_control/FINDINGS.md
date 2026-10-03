# Classical control findings

Measurements and reverted experiments behind `visual_servo`'s current
behavior. The runbook and current results are in [README.md](README.md).

## The 2026-09-29 Isaac Sim CLOSE-exit fix

Opened by ROADMAP.md's 2E tier 4: `visual_servo` ran its full phase sequence
against real Isaac Sim but never delivered the cube. Root cause, fix, and
what's still open are summarized in [ROADMAP.md](../../ROADMAP.md)'s 2E
entry; this records the trace evidence behind it.

### The bug

`act()`'s CLOSE/RELEASE settle check was:

```python
settled_gripper = abs(grip_now - self._grip) < 0.06
```

`self._grip` is the *ramping* commanded position (moves at a fixed
`0.9 rad/s` toward `grip_goal`, not `grip_goal` itself). On MuJoCo the real
gripper lags that ramp by well over 0.06 until the ramp is nearly done, so
the check accidentally waits for the real squeeze. Isaac's PhysX position
drive tracks the ramp with ~no lag, so the check was satisfied almost
immediately — traced directly against real Isaac Sim (`squeeze_grip=0.06`,
seed 0, front+wrist cameras, `render=True`):

```
step= 144 phase=CLOSE  _grip=1.0000 grip_now=1.0000 diff=0.0000
step= 145 phase=CLOSE  _grip=0.9700 grip_now=1.0000 diff=0.0300
...
step= 151 phase=CLOSE  _grip=0.7900 grip_now=0.8200 diff=0.0300
step= 152 phase=LIFT   _grip=0.7600 grip_now=0.7900 diff=0.0300   <- CLOSE exits, ~21% closed
```

8 consecutive steps at `diff=0.03 < 0.06` end CLOSE while `_grip` is still
0.76 — nowhere near the 0.06 target. The gripper never touches the cube;
LIFT starts on an empty hand.

### The fix

`ramp_done = self._grip == grip_goal` gates settling on the ramp actually
reaching its target first. A jaw genuinely jammed against the object (the
whole point of a deep `squeeze_grip`) then never satisfies `tracking`
(`abs(grip_now - self._grip) < 0.06`), so a `stalled` path (no real motion
for a while) is the alternative. A first version used the same 8-step
patience for `stalled` as for `tracking` and re-traced clean:

```
step= 176 phase=CLOSE  _grip=0.0600 grip_now=0.2623 diff=0.2023
...
step= 189 phase=CLOSE  _grip=0.0600 grip_now=0.2740 diff=0.2140   <- 8 stalled steps, CLOSE exits ~73% closed
step= 190 phase=LIFT   _grip=0.0600 grip_now=0.2740 diff=0.2140
step= 192 phase=LIFT   _grip=0.0600 grip_now=0.3012 diff=0.2412
step= 193 phase=LIFT   _grip=0.0600 grip_now=0.2230               cube z: 0.021 -> 0.061 in one step
```

Real progress (grip now reaches ~73% closed instead of ~21%, and the cube
visibly lifts during CLOSE), but a **new** failure mode: the jaw was stuck
on static friction at ~0.27 open, 0.21 short of the 0.06 target, for about
15 steps — long enough to pass the 8-step stall check — then slipped the
rest of the way through *after* LIFT had already started moving the arm.
The slip released stored spring/contact energy into a moving system and
launched the cube several cm in one step.

`_GRIP_STALL_SETTLE_STEPS = 30` (versus `tracking`'s 8) gives that stick-slip
pause more time to resolve into further closing *before* the phase advances,
so the eventual slip lands during CLOSE (arm still stationary) rather than
during LIFT (arm already moving) — mirroring how
`test_so101_grasp_hold_isaac.py`'s `grasp_and_lift` avoids the same failure
by never moving the arm until its own fixed-step squeeze is done. Re-traced
with the longer patience (seed 0):

```
step= 144 phase=CLOSE  grip_now=1.0000
step= 205 phase=LIFT   grip_now=0.2600   <- CLOSE exits after the 30-step stall cap, still ~74% closed
step= 207 phase=LIFT   grip_now=0.2222   cube z: 0.019 -> 0.024
step= 211 phase=TRANSFER grip_now=0.0668  cube z: 0.019 -> 0.030 (smooth, no single-step jump)
```

No more multi-cm single-step launch — the cube rises smoothly over several
steps instead. `_GRIP_STALL_SETTLE_STEPS=30` reduces the failure but does not
eliminate the underlying stick-slip: the 30-step cap can still be reached
before the slip finishes (as above), so the final close-through can still
land after a phase transition. It just lands more gently.

### Resolution: Isaac gripped with a different contact geometry (2026-09-29)

The TRANSFER slip was not a friction, force, or solver-tuning problem. MuJoCo
grips with two fitted 12 x 12 x 6 mm box pads (`contact_pads` in
`description.yaml`, tilted ~8 degrees, with the jaw's own collision mesh
disabled -- `sim.mujoco.scenes.common`). Isaac only ever received the pads'
*friction material*, applied to the raw jaw collision mesh
(`apply_contact_friction`); the pad boxes themselves were never built. The two
simulators gripped with different shapes at different positions, so no amount
of matching coefficients or solver settings could make them comparable.

Fix: `sim.isaac.description.apply_contact_pad_colliders` builds the same boxes
from the shared spec and deactivates the jaw links' own collision wrappers (the
importer's `<mesh>_1` Xforms hold instance-proxy meshes, which are read-only,
so the wrapper prim is deactivated instead). Checked independently before
running any policy: at the pad-alignment angle (`pad_align_gripper_q=0.16`) the
pad centres are 33.9 mm apart in Isaac (expected 28 mm cube + 2 x 3 mm pad
half-thickness = 34 mm), which also confirms the USD link frames match the
MJCF body frames the pad poses are expressed in; only the two pads remain
collidable under `gripper_link`.

Result, with no per-simulator overrides (`squeeze_grip` is the shared default
0.15; the only test-side override is `target_plane_z`, scene geometry):
`visual_servo` detects, approaches, grasps, lifts (cube z ~0.054 through
TRANSFER), and places the cube 9 mm from the target on seed 0. The tier-4
test (now asserting delivery within 4 cm) passed 3 of 3 consecutive runs, and
tier 3 (grasp-hold), the Isaac env tests and the non-Isaac suite all still
pass -- unlike every parameter experiment below, which broke tier 3.

Two earlier "noise" observations are worth keeping in perspective: an
identical parameter config gave different results run-to-run while the
geometry was wrong, i.e. the system was near an instability that the correct
geometry does not have (3 of 3 identical passes since).

### Experiments before the fix (kept as the record of what did *not* work)

These were all run against the mismatched geometry above; their negative
results are best read in that light, not as statements about PhysX itself.

#### Hold security through TRANSFER (as it looked before the fix)

Even with a genuine squeeze, the cube does not reach the target: it is
carried through LIFT and partway through TRANSFER, then settles back down
before the arm finishes (cube ends ~17cm short of the target in both
post-fix traces; not asserted by
`test_visual_servo_runs_the_full_pick_and_place_loop`, which checks real
lift height instead of final position).

**Squeeze depth is not the lever (verified on MuJoCo, cheap to check).**
`squeeze_grip=0.06` is tuned to just barely jam in
`test_so101_grasp_hold_isaac.py`'s *static* grasp-and-hold; the obvious next
guess was a deeper squeeze. Tested on MuJoCo (identical control logic, much
cheaper to iterate on) over 50 seeds: `squeeze_grip=0.06` scored 47/50 (94%),
and the deepest available value, `squeeze_grip=0.0`, scored identically —
47/50, the *same* failing seeds (13, 15, 28), the *same* per-episode step
counts. Both simulators cap the gripper actuator's torque well below its
rating (`gripper_force_limit`, 0.3 N·m in both — see below), so once the
jaw is jammed against the cube, commanding a deeper position target past the
jam point does not increase the actual squeeze force in either simulator;
it was already at the force cap. This also confirms `squeeze_grip=0.06` is
not compensating for weaker Isaac physics — it performs identically on
MuJoCo, so it is safe to treat as a single shared value rather than a
per-simulator override.

**Raising `gripper_force_limit` was tried and made it worse, not better
(reverted).** If squeeze depth doesn't raise the force cap, the cap itself
seemed like the next lever: `IsaacEnvConfig.gripper_force_limit` was raised
from 0.3 to 0.5 N·m (still well under the joint's 3.35 N·m rating) on the
theory that PhysX's effective grip at the shared 0.3 N·m cap needed more
friction margin than MuJoCo's contact model does for the same visible
depth. Verified against real Isaac Sim (both
`test_so101_grasp_hold_isaac.py` and
`test_visual_servo_runs_the_full_pick_and_place_loop` in one run): tier 4
was unaffected (both its tests still passed), but tier 3 -- the
*previously 100% reliable* 12-second static grasp-and-hold -- regressed to
a complete failure to lift the cube at all
(`cube_pos[2] == 0.0189`, at rest height, not the required
`> spawn_height + 0.02`). This is exactly the "position-controlled squeeze
into contact chatter" `EnvConfig.gripper_force_limit`'s own MuJoCo-side
docstring already warns the rated 3.35 N·m torque causes -- a *stronger* cap
did not add margin, it destabilized the grasp Isaac already had. Reverted to
0.3.

**Raising the cube's PhysX solver iteration counts (MuJoCo's
`noslip_iterations` analogue) was also tried and also made it worse
(reverted).** Friction *coefficients* are not the mismatch: `add_cube`
already copies MuJoCo's own sliding-friction value
(`SingleCubeFixedPlaceSceneConfig`'s cube `friction=[1.2, ...]` ==
`GraspCubeConfig.friction=1.2`) onto the PhysX material, and both
`add_cube` and `apply_contact_friction` already force PhysX's
friction-combine mode to "max" to match MuJoCo's own combine policy
(tier 3's own earlier fix). What MuJoCo's `noslip_iterations` actually
buys is different: MuJoCo's friction constraint is inherently soft, so a
held object creeps out even when the friction force is well within its
limit (`description.yaml`'s own comment: "~1 mm/s ... regardless of grip
force"), and `noslip_iterations` is an extra solver pass that removes that
creep without touching the coefficient. PhysX's analogous lever is solver
*iteration count*, not the material. Checked against the actual installed
PhysX schema (`omni.usd.schema.physx`'s `schema.usda`, not assumed):
`PhysxRigidBodyAPI`'s defaults are `solverPositionIterationCount=16`,
`solverVelocityIterationCount=1`; `PhysxArticulationAPI`'s (the robot,
including its gripper links) are `solverPositionIterationCount=32`,
`solverVelocityIterationCount=1`. Since a contact pair's iteration count is
documented as the max of the two bodies', the gripper's own default (32)
already exceeds a plain rigid body's (16) -- so setting the cube's position
count to 32 to "match" was likely a no-op for the actual contact, but the
same change also set the cube's *velocity* count to 4, introducing an
asymmetry with the gripper's unchanged 1 that hadn't existed before (both
were 1 by default). Verified against real Isaac Sim (same two-test run as
the force-limit experiment): tier 4 unaffected, tier 3 regressed to the
*same* complete failure to lift (`cube_pos[2] == 0.0189`, matching the
force-limit failure's value almost exactly). Reverted (`git checkout --
src/physai/sim/isaac/objects.py`).

**Raising solver velocity iterations scene-wide (not per-body, to avoid
the asymmetry above) was tried next and failed the same way.**
`PhysxSceneAPI.minVelocityIterationCount` raises a *floor* that applies to
every actor uniformly (rigid bodies, articulations, everything) instead of
one body's own count, which should raise the cube and the gripper's
effective velocity iterations together and preserve whatever symmetry
otherwise holds between them -- ruling out the specific "asymmetric
counts" mechanism guessed above, not just retrying the same thing. Its
default is 0 (confirmed in the same installed schema), so the per-body
default of 1 was the effective floor already; raised to 4. Verified
against real Isaac Sim (same two tests): tier 4 unaffected, tier 3
regressed to the same complete failure to lift again
(`cube_pos[2] == 0.018949955701828003` -- to 7 decimal places, the same
resting value as the two prior failures). Reverted
(`git checkout -- src/physai/robots/so101/isaac_env.py`).

Three independently-reasoned "add more force/accuracy" interventions --
raising the torque cap, raising (and separately, ruling out an asymmetry
in) solver iterations -- all broke the same previously 100%-reliable
static grasp the same way. That the symmetric, scene-wide version failed
identically to the per-body one rules out "unbalanced iteration counts"
as the mechanism; all three failures landing on nearly the same resting
height is also consistent with a much simpler explanation than three
different physical mechanisms -- in each case the fingers may simply never
have engaged the cube at all, so it just sits at its one, deterministic
resting equilibrium regardless of which change caused that. Either way,
the pattern says `test_so101_grasp_hold_isaac.py`'s current success is not
"under-tuned, push any knob higher" -- it looks like a stability optimum,
and pushing past it in more than one unrelated direction fails the same
way. Further tuning here should not be more blind parameter sweeps (each
costs a full Isaac Sim iteration -- 20 minutes to a few hours in this
environment); it needs either a verified mechanism (e.g. actually
inspecting per-substep contact forces/penetration interactively in Isaac
Sim's own GUI, not headless) or a differently-shaped fix (see the "not
yet tried" motion-speed idea below), not another guess at a scalar.

**Doubling the effective pad-cube friction (independent of force/solver
iterations -- a genuinely different lever) was tried on `visual_servo`'s
tier-4 loop and also didn't help, but by a different and precisely
diagnosed mechanism.** `friction=[1.2, ...]` was already matched to
MuJoCo's own value (confirmed above), but the pads' own friction
(`description.yaml`'s `pad_static`/`pad_moving`, `2.0`) dominates under
"max" combine regardless of the cube's -- so matching the cube's number to
MuJoCo was never the binding constraint; the pads' shared `2.0` already
was. Tested doubling it to `4.0` for Isaac only (`apply_contact_friction`
gained an opt-in `friction_multiplier` param, default `1.0` = unchanged;
`IsaacEnvConfig.contact_friction_multiplier` wired it through), leaving
MuJoCo's own value untouched. Traced against real Isaac Sim
(`squeeze_grip=0.06`, seed 0): CLOSE never settled at all within its
120-step safety cap -- `grip_now` oscillated in a ~0.003-0.005-per-step
band around 0.22 open (e.g. 0.2238, 0.2184, 0.2210, 0.2183, 0.2189, ...,
step 243 to 263) instead of converging, so neither `tracking` nor 30
consecutive `stalled` steps (`_GRIP_STALL_EPS=0.003`) were ever satisfied,
and the phase was force-ended by the step cap instead. The cube was never
lifted at all (z stayed at its ~0.0189 rest height through CLOSE, LIFT,
*and* TRANSFER -- it was dragged along the ground, not carried), ending
0.237m from the target, worse than the unmodified-friction runs. This
isn't "more friction is bad for grasping" -- it's that raising friction
changed the contact's micro stick-slip *noise* enough to break
`_GRIP_STALL_EPS`'s noise tolerance, which was tuned against the default
friction's jitter characteristics, not against this one. Reverted
(`git checkout -- src/physai/sim/isaac/description.py
src/physai/robots/so101/isaac_env.py`); the `friction_multiplier` knob
itself was default-safe (`1.0` no-ops) but removed anyway per this
project's KISS convention -- no passing caller needs it yet.

**Soft (compliant) pad contact, translated from MuJoCo's own solref, did not
fix it either -- but it fixed a different problem and exposed run-to-run
noise.** MuJoCo's pads are already deliberately soft (`description.yaml`:
`solref=[0.004, 1.0]`, "already at the stability floor ... a stiffer direct
solref throws the cube out of the grasp"), so the force/solver/friction
experiments above all pushed PhysX *stiffer* than the working reference.
PhysX's analogue is `PhysxMaterialAPI`'s `compliantContactStiffness`/
`compliantContactDamping` (default 0 = rigid; found in the installed
`schema.usda`; no FEM deformable body needed), with
`compliantContactAccelerationSpring=True` (sink-in depth = acceleration /
stiffness, independent of mass). Scratchpad-only, no repo change.

- *Direct solref translation on pads AND cube* (stiffness 1/tau^2 = 62500,
  damping 2/tau = 500): the cube was shoved ~7 cm in x during DESCEND, before
  the gripper closed; max cube z 0.0515 (launched), 0.19 m from target. So the
  soft *cube* material disturbs the open-finger approach.
- *Pads only, cube rigid* (five configs, cube reset to its spawn pose
  (0.20, 0.08) each time -- `env.reset()` alone does NOT move the cube back, so
  an earlier 3-config pass that relied on it was invalid and discarded):
  stiffness/damping 20000/283 -> max z 0.036, 0.247 m from target;
  10000/200 -> 0.049, 0.169; 20000/566 -> 0.027, 0.209; 20000/140 -> 0.066,
  0.106 (but pre-grasp drift 0.137 m: the cube was pushed during DESCEND,
  not carried); 40000/400 -> 0.068, 0.147. Pre-grasp drift is ~0 for the
  well-damped configs, so pad-only compliance does remove the descent
  disturbance, and several configs lift the cube 2-5 cm (versus 0 lift at
  the shared default) -- but *none* carried it toward the target: it was
  dropped within ~4 cm of the pick point in every case.
- *Run-to-run noise is as large as the effect being tuned.* The identical
  config 20000/283 was run twice from the same start: max z 0.052 vs 0.036 and
  final distance 0.145 vs 0.247 m. Any single-run comparison between configs
  above (and in the earlier experiments, which were also single runs) is
  within that noise; only large, repeatable effects -- like tier 3 failing to
  lift at all, three times in a row -- are trustworthy from one run.

So pad compliance is a reasonable, MuJoCo-consistent knob, but on this
evidence it does not by itself make the hold survive TRANSFER. Deciding
between the remaining ideas needs repeated runs per setting (N >= 5) rather
than one, which is what the Isaac cost per run makes expensive.

**Not yet tried:** a slower TRANSFER motion (`max_joint_rate`, the resolver
step that actually bounds Cartesian speed -- note `SO101VisualServoPolicy`'s
own `max_speed` constructor parameter is dead code, stored but never read by
`act()`) was tested at half rate on MuJoCo alone: 46/50 (92%), a different
and slightly larger failing set (5, 13, 28, 41) than the 1.2 rad/s default.
Inconclusive on MuJoCo, and not yet tested on Isaac -- unlike the squeeze
depth, force-cap, and solver-iteration experiments above, slowing the
motion changes something MuJoCo's own contact model is not the bottleneck
for, so a MuJoCo-only test cannot rule it in or out for Isaac.

Each Isaac Sim iteration in this environment costs 20 minutes to a few hours
of wall-clock time (`SimulationApp` boot plus per-step rendering -- the
culprit for the "few hours" end was, more than once, a background `uv sync`
from an unrelated concurrent command silently swapping the `isaac` extra's
packages out from under the running process; never run another `uv`/`uv run`
command against this repo's venv while an Isaac Sim process is running), so
further tuning here should be scoped and budgeted deliberately rather than
done by trial and error. Cheap MuJoCo-side checks (same policy code, much
faster to iterate) are worth ruling a hypothesis in or out on first, as
above, before spending an Isaac run -- but only for hypotheses MuJoCo's own
contact model actually exercises the same way; the force-cap result above
shows a MuJoCo-side result can still fail to transfer.

### Verified not to regress MuJoCo

The ramp/stall rework only changes CLOSE/RELEASE's *timing*, and MuJoCo's own
settle condition was already dominated by real gripper lag (see the trace
above), not by the buggy early-exit path. Confirmed unaffected:

- The same seed-0 trace against MuJoCo is bit-for-bit identical before and
  after (`success=True dist_cube_target=0.0069...`).
- 100/100 MuJoCo seeds: 95/100 success, same 5 timeout seeds (13, 15, 28, 64,
  76) as the documented baseline, before and after.
- `tests/research/classical_control/test_so101_visual_servo.py` and
  `test_so101_visual_servo_acceptance.py`: unaffected.
- Full repo test suite (`tests/` excluding `isaac`-marked tests): unaffected.

## The 2026-10-02 identical-scene follow-up (jaw lands on the cube)

With the contact geometry matched (above), `visual_servo` still failed on
Isaac once the scene was made identical to MuJoCo's (table, target pad, the
same camera) and the wrist camera was calibrated. This records what the
failures were and the measurements behind the policy defaults.

### What was wrong

1. **The static jaw landed on top of the cube.** At DESCEND the pinch stopped
   about 22 mm above the cube centre (z ~0.057 vs 0.034): the static pad's
   underside rested on the cube top. The static pad sits ~18 mm toward -x of
   the pinch point and the cube is 28 mm wide, so there is ~1 mm of
   clearance; a few mm of estimation bias toward +x puts the pad over the
   cube. Pad geometry itself matches MuJoCo (pad positions relative to the
   pinch agree to the millimetre in xy, 2.5 mm in z).
2. **The first estimate is off by about a centimetre, and DESCEND outran the
   refinement.** The pinch dropped in ~5 control steps while the wrist
   refinement was still moving xy.
3. **Isaac had no wrist calibration at all**, so `_refine_from_final_camera`
   silently skipped. `SO101IsaacEnv.camera_calibration("wrist")` now composes
   the mount with the link pose from the display mirror (matches the USD
   camera prim to ~3 mm).
4. **MuJoCo's `fx = fy * width / height` was a hidden tuning.** On a 4:3 image
   it scaled x by 4/3, which shifted the y estimate ~5.5 mm to the safe side;
   the policy had been tuned on it (20/20), and "fixing" it alone dropped
   MuJoCo to 17/20. At 16:9 the factor is wrong by a different amount.
5. **`reset()` did not restore the cube** after a previous episode threw it,
   which invalidated one whole parameter sweep until it was caught.
6. **Rendering advanced Isaac's physics.** Each `rep.orchestrator.step` in
   `render_camera` stepped the simulation once, so with two cameras one
   control step ran 4 physics steps instead of 2 (counted: 48 steps for 12
   `env.step`s against 24 with rendering off): Isaac ran at twice the
   intended simulated time per control step, and everything measured on it
   before this was fixed (including an earlier 100/100) is void. It now
   renders with `delta_time=0.0`; joint trajectories with rendering on and
   off are identical.
7. **The first image after `reset()` was stale.** Poses written through the
   tensor API reach the renderer only after a physics step, so the first
   observation still showed the previous episode, a cube on the target pad,
   and the policy aimed at it. With rendering no longer stepping physics this
   showed up as a perfect alternation (even seeds succeed, odd seeds fail);
   `reset()` now holds for one control step before observing.

### The two perception gaps that remained (found by matching seeds)

Once the above was fixed and both simulators ran the same seeds, Isaac still
scored 79/100 against MuJoCo's 100/100 (21 failures, all seeds MuJoCo won,
concentrated at the far corner of the cube layout: none for x < 0.211 or
y < 0.07). Two causes, both in what the cameras see rather than in physics:

- **Colour thresholding was lighting-dependent.** `ColorBlobDetector` kept
  pixels within a distance of one bright red, so it dropped the shadowed part
  of the cube, and Isaac's shadows are deeper. The wrist blob was ~1700 px
  against MuJoCo's ~3900 and its centroid sat 13-20 px above the cube's
  projection: the wrist estimate was +9 to +15 mm off in y in Isaac and
  ~-1 mm in MuJoCo. The detector now uses saturation and hue
  (`(r - max(g, b)) / r >= 0.25`, and `(g - b) / (r - b) <= 0.35` to keep
  out the yellow arm): the same blob, to within 1 px, in both simulators.
- **Isaac washed the cube out.** Its default ACES tone mapping and sRGB
  encoding gave the cube's top face a saturation of 0.2 (MuJoCo: 0.7), so the
  front-camera blob was a 28-136 px fragment, the initial estimate wandered
  from -36 to +15 mm, and for the farthest cubes the hover point fell outside
  what IK reaches from HOME: the arm did not move for 230 steps (APPROACH and
  DESCEND each hit their 120-step cap). Linear output without the gamma step
  (`sim.isaac.core.configure_render_output`, applied once the stage exists)
  gives the cube faces to within a few levels of MuJoCo's (front face 111, 32,
  25 against 112, 33, 26) and an initial estimate within +1..+5 mm.

Light intensities were then fitted for that output (`sim/studio.py`): table
colour 224, 216, 197 against MuJoCo's 226, 217, 199 and a wrist-camera cube
mean of 100, 30, 24, identical to MuJoCo's. Table brightness is about linear
in dome + key; a sum near 1400 matches and much more clips the table to
white. The far floor near the horizon still renders darker than MuJoCo's,
which is why the mean pixel difference (~50) is not a useful score: no policy
reads that region.

### What changed

- `fx = fy` in both simulators (square pixels at any aspect ratio).
- `SO101VisualServoPolicy` defaults: `align_before_descend=True` (hover at the
  approach height until the pinch is within 4 mm of the refined xy, then
  lower) and `grasp_offset_xy=(-0.006, -0.006)` (bias the pick target to the
  safe side). The offset was found by sweeping on Isaac: y = -6 mm succeeded
  6/8 without align versus 1/4 at y = 0; x had no effect inside the swept
  range. With align on, 4/4 at (-6, -6) and 3/4 at (0, 0).
- `ColorBlobDetector` segments by saturation and hue; Isaac renders linear
  output with matte materials and a checkered floor (tile 1/6 m, aligned to
  MuJoCo's) from the shared palette in `sim/studio.py`.
- Isaac builds its scene from the same `ManipulationSceneConfig` as MuJoCo and
  draws the cube per seed with the same RNG order (`layout.draw_cube_xy`,
  checked against `golden_layouts.json`); a `TaskRuntime` wraps it.
- Camera resolution is one of 320x240 / 640x480 / 1280x720
  (`physai.contracts.CAMERA_RESOLUTIONS`); every camera in a run uses it.

### Results

Same seeds 0-99, same scene, policy defaults, 320 x 240
(`scripts/eval_policy.py` for each simulator, `scripts/compare_evaluations.py`):

| simulator | success | Wilson 95% | mean steps |
|---|---:|---|---:|
| MuJoCo | 100/100 | 96% to 100% | 200 |
| Isaac | 100/100 | 96% to 100% | 117 |

All 100 seeds agree (both succeed). The Isaac evaluation was run in ten-seed
shards, each in its own process, because of the render glitch below (a shard
that the glitch check aborted was simply repeated). Before the perception
fixes Isaac scored 79/100; with the saturation detector alone it scored
95/100 (the five misses were the far-corner cubes whose hover point IK could
not reach; repeated, those seeds succeeded about half the time). Those runs,
and a later 100/100, were taken with the two Isaac measurement bugs above
(6 and 7) still in, so they are not evidence of parity; the table is the run
after both were fixed.

Other resolutions (MuJoCo 20 seeds / Isaac 8 seeds, after the fixes):

| resolution | MuJoCo | Isaac |
|---|---:|---:|
| 640 x 480 | 20/20 | 8/8 |
| 1280 x 720 | 20/20 | 8/8 |

MuJoCo with `fx = fy` and the previous colour detector, align off: 17/20,
18/20, 19/20 at the three resolutions; align on: 20/20 at all three.

Isaac finishes episodes in 117 steps on average against MuJoCo's 200, and
that gap is one phase. Mean phase length over seeds 0-3 (steps), MuJoCo /
Isaac: APPROACH 29.8 / 29.5, DESCEND 6.5 / 6.8, CLOSE 117 / 36, LIFT 4.2 /
4.5, TRANSFER 25 / 25, LOWER 2 / 2, RELEASE 9 / 9.8. The arm motion matches
to within a step, which also confirms the timing fix; CLOSE differs because
MuJoCo's gripper lags its ramping command, so the settle check rarely passes
and CLOSE runs to its 120-step cap, while Isaac's PhysX position drive tracks
the ramp with no lag and settles in ~36 (see the CLOSE-exit section above).

### Bugs found along the way

- The wrist calibration took its pixel size from the description (320 x 240)
  instead of the chosen resolution, which at 640 x 480 halved cx/cy/fy and
  misplaced the cube by ~12 cm.
- Isaac's joint limits are float32, so an IK solution exactly at a limit
  exceeded them by 3e-6 and the safety gate rejected it; the spec now takes
  its limits from the kinematics model and `send_action` clips to Isaac's.
- Isaac Kit replaces `asyncio.run`, which broke uvicorn's server thread for
  `--serve`; the server now runs on its own event loop.
- The tier-3 grasp-hold test swept the cube with a jumping joint command once
  the ground plane was made flush with z = 0; it now rate-limits commands like
  `visual_servo` does and grasps in the table scene.
- **Render glitch (open).** Isaac Sim sometimes renders without the robot (no
  arm, no shadow) for a whole process while its physics works, so camera
  policies fail for reasons unrelated to the policy. It is random per
  process, comes in bursts (none in dozens of consecutive processes, then
  several in a row) and shows at every resolution. VRAM is not the cause
  (peak 1.3-1.8 GB of 6 GB); extra `app.update()` calls, toggling visibility
  and re-selecting the physics variant were tried, and it did not reproduce
  while collecting logs, so the cause is unknown. `SO101IsaacEnv.robot_is_rendered()`
  detects it without relying on colours (render with the robot shown and
  hidden, compare), the env refuses to start if it fails, and
  `eval_policy --sim isaac` checks at every episode start and stops, so no
  episode is recorded with a robot-less camera. Run long Isaac evaluations in
  shards and retry an aborted one.
- **Determinism (negative result).** Setting PhysX `enableEnhancedDeterminism`
  did not make runs identical (the same seed gave 128, 134 and 600 steps
  without it and 600, 133 and 134 with it), so it was reverted. Run-to-run
  variation remains; the far-corner cubes were the most sensitive.
