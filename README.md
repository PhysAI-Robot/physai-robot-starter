# physai-robot-starter — Phase 0

Simulation-first starter for physical-AI research on the **SO-101** 5-DoF arm.
Phase 0 runs the whole stack from your architecture diagram in one process, in
MuJoCo, with no ROS2. The sim, scripted expert, and planner path need no GPU;
training a real VLA policy (`scripts/train_act.py`) does.

```
  natural language ──▶ VLM planner ──▶ sub-goals (PoseStamped)
                                          │
                        VLA policy ◀──────┘   ◀── images + joint state
                            │
                    joint targets / Twist
                            │
                        MuJoCo (rigid body + contact)
```

Every interface between those boxes already speaks **ROS2-shaped messages**
(`JointState`, `PoseStamped`, `Twist`, `GripperCommand`), so Phase 1 is a port,
not a rewrite. See [`src/physai/bridge/ros2_contract.py`](src/physai/bridge/ros2_contract.py).

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_assets.py          # downloads the SO-101 MJCF + meshes (~16 MB)
python scripts/run_sim.py               # one episode + an mp4 in outputs/
python -m pytest tests/ -q
```

Then:

```bash
python scripts/eval_policy.py --policy scripted --episodes 20
python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1
python scripts/plan_task.py --planner scripted        # planner path, no API key
python scripts/workspace_map.py                       # what the arm can actually grasp
python scripts/run_sim.py --viewer                    # interactive
python scripts/teleop_keyboard.py                     # jog it by hand
```

**Training a real VLA** (optional, GPU strongly recommended — see `requirements.txt`
for the `torch`/`lerobot` install, which needs a hardware-specific command so it
isn't pinned in the base install):

```bash
python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1 --width 128 --height 128
python scripts/train_act.py --dataset data/pickplace_v1 --steps 5000
python scripts/eval_policy.py --policy lerobot --checkpoint outputs/act_ckpt --episodes 20 --camera-size 128
```

## Verified status

Measured on this repo, Python 3.13 + MuJoCo 3.11:

| Check | Result |
|---|---|
| Scripted expert, 30 seeds | **30/30 success** (mean ~145 control steps ≈ 5.8 s) |
| Demo collection, 50 episodes | 50/50 kept, 0 discarded |
| Record → replay round-trip | success |
| Planner → PlanRunner (scripted planner) | success |
| **ACT trained from 50 demos, 5000 steps** (GTX 1650, 4 GB, ~15 min) | **7/20 = 35%** success, camera-only, vs a **95%** scripted-expert ceiling on the same 20 seeds |
| Test suite | 24 passed |
| Sim speed | ~700 control steps/s headless without cameras |

The VLM planner path (`--planner claude`) is implemented and type-checked
against the Messages API but **has not been run against the live API here** —
it needs your `ANTHROPIC_API_KEY`.

35% is not meant to look good in isolation — it is the measured cost of
replacing the scripted expert's privileged simulator access with two camera
feeds, at this data and compute scale. The two most likely levers are more
demonstrations and more training steps (L1 loss was still trending down at
step 5000); see the training report generated alongside this run for the
loss curve, per-episode breakdown, and two preprocessing bugs that were
caught and fixed while producing it (`training_meta.json` filename collision;
non-square camera renders silently degrading the policy — both now fixed in
`vla_adapter.py` and `act_dataset.py`).

---

## Answers to the three open questions on your slide

### 1. Model?

**Two different models, two different rates. Don't conflate them.**

| Layer | Rate | Recommendation |
|---|---|---|
| VLM planner | ~0.2 Hz | Claude Opus 5 via the Messages API — implemented in [`planner/claude_vlm.py`](src/physai/planner/claude_vlm.py) with structured outputs, so the plan always parses. |
| VLA policy | 25–30 Hz | Start with **SmolVLA** (`lerobot/smolvla_base`): built for exactly this arm, fine-tunes on consumer hardware, ~50 episodes is the usual starting dataset size. |

For the VLA, in order of what I'd actually try:

1. **ACT** — the cheapest honest baseline. Trains in under an hour on 50 episodes.
   If ACT can't do your task, the problem is the data, not the model.
2. **SmolVLA** — language-conditioned, the natural fit for "VLA takes the VLM's
   sub-goal as text". Fine-tune from `lerobot/smolvla_base`.
3. **TurboVLA** — 0.2B params, 32 Hz, <1 GB VRAM on an RTX 4090, 97.7 % on
   LIBERO. Worth benchmarking against SmolVLA if inference latency becomes the
   bottleneck on real hardware. It is newer, so treat it as the experiment, not
   the baseline.

The seam is [`policy/vla_adapter.py`](src/physai/policy/vla_adapter.py) — you
implement one method (`_infer`) and observation packing, action chunking, and
unit conversion are already done.

### 2. Robot description?

`TheRobotStudio/SO-ARM100`, directory `Simulation/SO101`. It ships both URDF
(for ROS2) and MJCF (for MuJoCo), generated from the Onshape CAD.
`scripts/fetch_assets.py` mirrors it.

Use **`so101_new_calib.xml`**, not `so101_old_calib.xml` — "new calib" puts
each joint's zero at the middle of its range, which matches what LeRobot expects
from the real follower arm. The pad positions in this repo are measured against
the new-calib meshes; switching files means re-measuring them.

Joints, in the canonical order used everywhere in this repo:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.

### 3. ROS2 version?

**Jazzy Jalisco.** It is LTS on Ubuntu 24.04 with support through 2029, which
is the only sensible choice for a multi-year research project.

- *Kilted Kaiju* (May 2025) is non-LTS with a short support window.
- *Lyrical Luth* (2026) requires Ubuntu 26.04.
- *Humble* is LTS but ends in 2027 and is a generation behind on `ros2_control`.

---

## What actually turned out to be hard

These are real findings from getting the expert to 100 %, not hypotheticals.
They are the things that will cost you a week each if you rediscover them.

**The arm cannot do a high top-down approach.** With 5 DoF and no shoulder roll,
"reach this point" and "reach this point with the jaws pointing down" are
different problems. Position-only IK converges anywhere on the table; a
top-down-constrained solve only converges within ~5 cm of the surface. So the
IK here constrains the *approach axis* (2 rotational constraints), not a full
6-DoF pose — a full pose target on a 5-DoF arm just oscillates.

**Solvable ≠ graspable.** `scripts/workspace_map.py` shows top-down IK solving
across almost the whole table, but the expert only succeeds reliably for cubes
in x ∈ [0.20, 0.24]. IK does not model the descent path or the grasp itself.
The default spawn range is the band that works; widen it deliberately.

**The table was silently jamming the arm.** A table slab overlapping the robot
base at the origin pinned `shoulder_pan` against its 2.94 N·m limit. The symptom
was "IK returns a solution the arm never reaches" — it looks exactly like a
broken solver. `tests/test_sim.py` now asserts the table clears the base.

**Sphere-shaped finger pads eject the object.** Squeezing a cube between two
spheres is unstable; it squirts out sideways. The pads are flat plates whose
orientation is computed at build time from the EE site frame
([`sim/scene.py`](src/physai/sim/scene.py)) so both jaw faces are genuinely
parallel.

**Grip force is a knife-edge if you command one aperture.** Command the exact
object width and there is ~zero interference, so the hold decays from 25 N to
5 N during the lift and the cube slides out. Command tighter in one step and
the jaws close at ~5 rad/s and bat the cube away before contact captures it.
The fix is a two-stage close — `CLOSE` to just wider than the object, then
`SQUEEZE` — with the gripper rate-limited.

**Post-grasp waypoints must not track the object.** If the LIFT target is
`cube_z + lift_height` and the cube is in the gripper, the goal recedes exactly
as fast as the arm approaches it and the phase never completes. Pre-grasp
phases track the cube; post-grasp phases use absolute heights.

**Rate-limit everything.** Position actuators will happily accept a step input,
and the arm swats the cube off the table on the first tick.

---

## Layout

```
src/physai/
  contracts.py            ROS2-shaped message dataclasses — the common language
  sim/
    scene.py              MjSpec scene composition (table, cube, cameras, jaw pads)
    kinematics.py         FK, approach-constrained IK, pinch-centre IK
    env.py                SO101PickPlaceEnv — reset/step over Observation/Action
  control/
    resolver.py           PoseStamped -> joints (IK), Twist -> joints (Jacobian),
                          joint rate limiting
  policy/
    base.py               the Policy interface everything implements
    scripted.py           privileged expert; the demonstration generator
    plan_runner.py        executes a Plan open-loop (the VLA baseline)
    act_dataset.py        recorded .npz episodes -> ACT training batches
    vla_adapter.py        LeRobotPolicy (real ACT/SmolVLA checkpoints); ReplayPolicy
  planner/
    base.py               Plan / SubGoal, plus an offline ScriptedPlanner
    claude_vlm.py         Claude Messages API planner, structured outputs
  data/recorder.py        LeRobot-shaped episode recorder
  bridge/ros2_contract.py Phase 1 topics, types, rates — declared now, used later
scripts/                  runnable entry points (see Quick start), incl. train_act.py
configs/                  the numbers that matter, with reasoning attached
tests/                    24 tests; the sim ones skip if assets aren't fetched
```

### Data format

Episodes are `.npz` with LeRobot v2 key names, so conversion is a rename:

```
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32   5 arm joints + gripper, radians
action                    (T, 6) float32   absolute joint targets, radians
```

Actions are **absolute joint targets**, not deltas, because that is what the
real SO-101 follower takes — so a policy trained here transfers to hardware
without an action-space change.

---

## Is the plan feasible?

Yes, and Phase 0 is the right place to start. Two things worth flagging before
you commit:

**Sim-to-real on contact is the hard part, not the models.** Everything in "what
turned out to be hard" above is contact and kinematics, and none of it transfers
for free. Budget for the real arm behaving differently on grasping specifically.
The mitigation is that this env produces demos in the same format LeRobot records
from the real arm, so you can train on a mix.

**The VLM is not in the control loop, and shouldn't be.** At ~1–3 s per plan it
runs once per task. If you find yourself wanting it at 10 Hz, that is the VLA's
job. Keep the rate separation — it is the main reason this architecture works.

**A 5-DoF arm constrains the task set.** No shoulder roll means no side grasps,
no reorientation in hand, and a shallow top-down envelope. Pick tasks that fit
it, or plan for the 7-DoF variant if your "many robots / embodiment diversity"
axis matters.

## Suggested next steps

1. `python scripts/collect_demos.py --episodes 50` → your first dataset.
2. Convert to a `LeRobotDataset` and fine-tune SmolVLA; evaluate with
   `eval_policy.py` against the scripted expert's 100 % as the ceiling.
3. Turn on `randomize_target=True` and widen the cube range — measure how fast
   the VLA degrades. That is your first real result.
4. Run `plan_task.py --planner claude` and check grounding accuracy against the
   printed ground truth. That is the VLM half's first real result.
5. Only then start Phase 1 (ROS2 Jazzy). `scripts/show_ros2_contract.py` prints
   the node graph you'll be implementing.

## Sources

- [SO-ARM100 / SO-101 simulation models](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
- [LeRobot](https://github.com/huggingface/lerobot) · [SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla) · [SmolVLA paper](https://arxiv.org/pdf/2506.01844)
- [TurboVLA](https://arxiv.org/abs/2607.27205) · [code](https://github.com/H-EmbodVis/TurboVLA)
- [ROS 2 Kilted Kaiju release notes](https://docs.ros.org/en/jazzy/Releases/Release-Kilted-Kaiju.html) (for the LTS comparison)
- [Claude Messages API](https://platform.claude.com/docs/en/api/messages)
