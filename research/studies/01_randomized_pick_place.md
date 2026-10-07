# Study 1: randomized pick-and-place on the SO-101

**Status:** active. Framework features this study needs are in [ROADMAP.md](../../ROADMAP.md);
results live in each topic's `FINDINGS.md`. When the paper is out this file becomes its
record (status `done`); follow-up work is a new study file, not edits to this one.
`[ ]` marks planned, missing or partial work.

**Question:** on one task, how far does each paradigm go, from a privileged scripted
expert to a camera-only classical controller to imitation learning, and can RL make the
learned policy both more reliable and more efficient? The output is a study paper: an
open, reproducible comparison with confidence intervals on two simulators.

| Method | Perception | Role in the study |
| --- | --- | --- |
| Scripted expert | Privileged simulator state | Ceiling and demonstration teacher; must reach 100% |
| Visual servo (classical) | Camera only | How far hand-engineered perception and control go |
| ACT (imitation learning) | Camera + proprioception | The learning baseline, trained on scripted demonstrations |
| ACT + RL refinement | Camera + proprioception | Does RL lift success and motion efficiency over ACT? |

## Research questions

1. **Coverage:** with the cube and the place target both random over the whole reachable
   workspace, what success does each method reach, and where in the workspace does it fail?
2. **Classical against learning:** where does camera-only visual servo match ACT, and where
   does it not?
3. **Sim-to-sim:** how much does each method lose moving from MuJoCo to Isaac Sim without
   retraining, and what closes the gap? The gap already seen on the narrow task is in
   the [findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap).
4. **RL refinement:** does RL on top of ACT raise success and shorten, smooth the motion
   (completion time, path length, jerk)?
5. **Cost of the top-down grasp:** how much of the table does the fixed top-down grasp leave
   out, and how much would a tilted grasp add? Answered by the measurement under
   [Task definition](#task-definition-frozen-at-m0-before-any-training), not by a method.

## Candidate contributions

No new algorithm is claimed. The contribution is measured evidence, picked at M3 from
what the results support:

- **Benchmark (baseline contribution):** an open, seeded SO-101 pick-and-place benchmark
  on MuJoCo and Isaac Sim, four methods, one protocol, Wilson intervals on at least 300
  held-out seeds.
- **Engine gap as a proxy for visual domain shift (strongest candidate):** a controlled
  ablation of why ACT loses on Isaac (colour, lighting, texture) and which training change
  closes it, against a classical method that does not lose. Cheap to run, rarely measured.
- **Teacher quality to student quality:** does a smoother scripted expert (minimum-jerk)
  give a better or more efficient ACT student? One extra demo set and one extra training run.
- **RL refinement on a 6 GB GPU budget:** residual RL on a frozen ACT on a low-cost arm.

## Task definition (frozen at M0, before any training)

- **Cube and target spawn:** both sampled independently over the region where a top-down
  grasp with hover clearance is reachable, from `scripts/workspace_map.py`: x 0.14-0.27 m,
  y within about ±0.14 m, with the far corners excluded (the map's `o` cells). Left,
  right and front are all covered. Today's task spawns the cube in x 0.20-0.24, y
  0.05-0.13 only, with a fixed target.
- **Grasp approach: top-down only, frozen.** Every method grasps with the gripper pointing
  down. The region above is a limit of the SO-101 (5 DoF, no shoulder roll) with that grasp,
  not of a method, and the paper says so. Side or tilted grasps are not part of this study;
  they are a follow-up study (`studies/02_*.md`) if the results call for one.
- **Sampler as frozen** (`configs/manifests/so101_randomized_pick_place.yaml`): x 0.14-0.27 m,
  |y| <= 0.14 m, and 0.16-0.255 m from the base, for cube and target alike.
- **What the top-down limit costs** (IK check, grasp at z = 0.034 m and a pre-grasp 5 cm back
  along the approach direction, approach pointing away from the base, no collision or contact
  test, 504 grid cells of 1 cm x 2 cm over x 0.10-0.33, |y| <= 0.20, of which 210 lie in the
  region above; `scripts/workspace_map.py --tilt T --hover 0.05 --x-range 0.10 0.33
  --y-range -0.20 0.20` prints each grid):

  | Gripper tilt below horizontal | Reachable cells | In the region | Distance from base |
  | --- | ---: | ---: | --- |
  | 90 degrees (top-down, used) | 218 | 154 | 0.14-0.26 m |
  | 75 | 86 | 33 | 0.16-0.35 m |
  | 60 | 29 | 14 | 0.17-0.34 m |
  | 45 | 18 | 6 | 0.23-0.33 m |
  | 30 or less | 6 or fewer | 0 | 0.28 m and beyond |
  | 0 (flat, "prone") | 0 | 0 | none |

  A side grasp does not add coverage inside the region: top-down reaches the most cells near
  the base, and a flat grasp reaches none at cube height. A tilt of 60-75 degrees only
  extends the outer reach to about 0.33-0.35 m, in a narrow band. Reaching it would need a
  new grasp geometry in the expert and in visual servo, and a repeat of M1.
- **Minimum cube-to-target distance** so that no episode is a near no-op (0.08 m).
- **Target is visible** in the front camera everywhere in the region, as the existing
  target disc; no method except the scripted expert reads its position from state.
- **Success:** unchanged (`success_xy_tol` 0.04 m, held 10 steps).
- **Seeds:** demonstrations from 0-999, evaluation on held-out 1000-1299 (300 episodes);
  the same seeds for every method and both engines.

## Milestones

### M0: freeze the task

- [x] Core features from [ROADMAP.md](../../ROADMAP.md#tasks-and-scenes):
  reachability-shaped spawn sampling (a radius range from the base, 0.16-0.255 m), a random
  target on Isaac, the minimum separation.
- [x] A manifest for the randomized task beside `so101_single_cube_fixed_place.yaml`:
  `so101_randomized_pick_place.yaml`.
- [x] Spawn parity: the same seed gives the same target pose and a cube within 3 mm on both
  engines (seeds 1000-1009).
- [x] The target disc and the cube are visible in the front camera at episode start on all 300
  held-out seeds (fewest pixels: cube 132, disc 485, counted by hiding the object and
  diffing the render). The wrist camera at HOME sees them on only about two thirds of the
  seeds, so a camera-only method must start from the front camera.

Done when: the task above is runnable on both engines and the manifest is committed.

### M1: scripted expert at 100% (gate for everything after it)

- [x] 300/300 on seeds 0-299 and 300/300 on 1000-1299 on MuJoCo.
- [ ] Run the expert on Isaac as well (it reads MuJoCo contact data; needs a port).
- [x] Any failure is fixed at its root cause, or the region is shrunk and the limit
  documented in [FINDINGS](../scripted_experts/FINDINGS.md#randomized-pick-and-place-study-1-m1);
  no method is trained before this.
- [x] Trajectory-quality numbers for the expert (time, path length, jerk) as the efficiency
  reference.

### M2: visual servo on the randomized task

- [x] Detect the place target from the camera (`TargetDiscDetector`); it used to read
  `env.target_pos`, which is privileged and would break the camera-only claim once the target
  is random.
- [ ] Evaluate on the protocol below on both engines, with a success heatmap over the
  workspace. Done so far: MuJoCo 296/300 on seeds 1000-1299, Isaac Sim 88/100 on seeds
  1000-1099 (seeds 1100-1299 on Isaac and the heatmap remain). No further grasp-recovery or robustness tuning: report it as it stands.

### M3: ACT

- [ ] Demonstrations from the M1 expert: 100, 200 and 500 episodes.
- [ ] One fixed training configuration (image size, chunk size, batch, precision, steps)
  that fits the 6 GB GPU, with seeds and loss curves logged.
- [ ] Evaluate on both engines; ablations: number of demonstrations, wrist camera on/off.
- [ ] Choose the paper's contributions from the results so far.

### M4: sim-to-sim gap

- [ ] Colour and lighting augmentation, then visual randomization in the MuJoCo renderer;
  retrain and measure the Isaac drop after each, one change at a time.
- [ ] Run the difficulty sweep (lighting, camera shift) for visual servo and the best ACT.

### M5: RL refinement

- [ ] Residual policy on a frozen ACT: RL learns a small correction to ACT's action,
  trained with PPO through the existing Gymnasium adapter and `task.reward` plus
  efficiency terms (time, path length, jerk).
- [ ] If image-based RL does not fit the GPU, train the critic on simulator state and the
  actor on ACT's inputs (asymmetric actor-critic), and say so in the paper.
- [ ] Report ACT against ACT + RL on success and on the efficiency metrics.

Done when: the RL row is in the results table, or the attempt is documented as negative
with its cause (a negative result is still reportable).

### M6: paper and release

- [ ] Paper: setup, task, four-method comparison, workspace heatmaps, sim-to-sim gap,
  ablations, RL refinement, failure analysis, limitations (simulation only).
- [ ] Tag a release with the exact configs, seeds and checkpoint download links that
  reproduce every reported number.

## Evaluation protocol (shared by all methods)

300 held-out seeds (1000-1299) per method and engine, at least 100 for an ablation
([why](../scripted_experts/FINDINGS.md#why-the-protocol-needs-at-least-100-seeds)). Success
rate with a 95% Wilson interval. Metrics: success, collision, timeout, unsafe-action
rejections, completion time, path length, jerk. Failures broken down by spawn region and
by cause.

## What to prepare

- **Compute budget:** Isaac evaluation is slow and needs shards
  ([findings](../imitation_learning/FINDINGS.md#causes-found-along-the-way)); budget the
  300-episode Isaac runs per method before M3.
- **Related work to read and cite:** ACT (Zhao et al., 2023), LeRobot (Cadene et al.,
  2024), Diffusion Policy (Chi et al., 2023), domain randomization (Tobin et al., 2017),
  residual RL (Silver et al., 2018; Johannink et al., 2019; Ankile et al., 2024), and
  manipulation benchmarks (robosuite, ManiSkill, RLBench) for positioning.
- **Venue:** a robotics workshop paper or an arXiv report first; pick the venue before M6
  so its page limit shapes the figures.
- **Record as you go:** causes and final numbers in each topic's `FINDINGS.md`, so the
  paper's results and failure sections are written from them, not reconstructed.
