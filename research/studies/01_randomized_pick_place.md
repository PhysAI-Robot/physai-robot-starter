# Study 1: randomized pick-and-place on the SO-101

**Status:** active. Framework features this study needs are in [ROADMAP.md](../../ROADMAP.md);
results live in each topic's `FINDINGS.md`. When the paper is out this file becomes its
record (status `done`); follow-up work is a new study file, not edits to this one.
`[ ]` marks planned, missing or partial work.

**Question:** on one task, how far does each paradigm go, from a privileged scripted
expert to a camera-only classical controller to imitation learning, and why does the
learned policy lose success when only the simulator's renderer changes? The output is a
study paper: an open, reproducible comparison with confidence intervals on two simulators,
and a controlled attribution of the renderer gap. RL refinement on top of ACT was planned
here and moved to [study 02](02_act_rl_refinement.md): ACT is at 98% on MuJoCo, RL on Isaac
does not fit the evaluation budget, and residual RL on ACT is a crowded area.

| Method | Perception | Role in the study |
| --- | --- | --- |
| Scripted expert | Privileged simulator state | Ceiling and demonstration teacher; must reach 100% |
| Visual servo (classical) | Camera only | How far hand-engineered perception and control go |
| ACT (imitation learning) | Camera + proprioception | The learning baseline, trained on scripted demonstrations |

## Research questions

1. **Coverage:** with the cube and the place target both random over the whole reachable
   workspace, what success does each method reach, and where in the workspace does it fail?
2. **Classical against learning:** where does camera-only visual servo match ACT, and where
   does it not?
3. **Sim-to-sim:** how much does each method lose moving from MuJoCo to Isaac Sim without
   retraining, and what closes the gap? The gap already seen on the narrow task is in
   the [findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap).
4. **Attribution:** which image content carries the renderer gap, and which training recipe
   (one renderer, a randomized renderer, both renderers) holds on a renderer never seen in
   training?
5. **Cost of the top-down grasp:** how much of the table does the fixed top-down grasp leave
   out, and how much would a tilted grasp add? Answered by the measurement under
   [Task definition](#task-definition-frozen-at-m0-before-any-training), not by a method.

## Candidate contributions

No new algorithm is claimed. The contribution is measured evidence. Status after M4; each
point links to where the numbers are:

- **Benchmark (baseline contribution), supported:** an open, seeded SO-101 pick-and-place
  benchmark on MuJoCo and Isaac Sim with one protocol (300 held-out seeds, Wilson intervals,
  validation seeds kept apart from test seeds). Expert, visual servo and ACT are measured
  ([results table](../scripted_experts/README.md#results)).
- **Engine gap as a proxy for visual domain shift (strongest candidate), supported, cause not
  yet isolated:** on the same 100 seeds the move from MuJoCo to Isaac costs ACT 40 points
  (98 to 58) and the camera-only classical method 9 (97 to 88), with no extra demonstrations
  closing it ([findings](../imitation_learning/FINDINGS.md#randomized-pick-and-place-study-1-m3)).
  M4: physics is ruled out, and training on Isaac's own images with the same actions recovers
  most of the gap (55 to 85 of 100) while the same model loses 15-20 points on MuJoCo, so the
  gap is a pure observation-domain shift
  ([findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap-on-the-randomized-task-study-1-m4)).
  Colour, brightness, sharpness and gripper-reading differences each fail to explain it alone;
  which image content does is open (M5), but training on both engines' images closes it
  (94/100 on Isaac with no loss on MuJoCo). The transfer is asymmetric: MuJoCo-trained ACT
  loses 40 points on Isaac, Isaac-trained ACT loses 12 on MuJoCo at 100, 200 and 400
  demonstrations alike (paired, same 100 seeds).
- **Limit to state in the paper:** the scripted teacher only runs on MuJoCo, so the Isaac data are its
  actions replayed on Isaac with Isaac's observations (see
  [findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap-on-the-randomized-task-study-1-m4));
  that isolates the observation shift but is not an Isaac expert.
- **Teacher quality to student quality:** not started; needs a minimum-jerk expert, one demo set
  and one training run.

### Findings worth a section in the paper

1. **Longer action chunks played in full beat reactive execution** for ACT on this task
   (validation 62% to 82% to 92% with chunk 30, chunk 100, then 60k steps; fewer actions per
   chunk or temporal ensembling lowered success monotonically). The opposite of the usual
   advice to re-plan often. [Findings](../imitation_learning/FINDINGS.md#randomized-pick-and-place-study-1-m3).
2. **Demonstration count only matters once the training setup is right:** at chunk 30 200
   demonstrations were no better than 100 (206 against 220 of 300); at chunk 100 and 60k steps
   they were (294 against 275). An ablation run at a fixed, weak configuration would have
   concluded the opposite.
3. **The classical method is also fragile across engines, until perception is fixed.** The
   target-disc detector's first threshold scored 51/100 on Isaac and 296/300 on MuJoCo; a
   threshold set from measured background chroma gave 271/300. Renderer differences hit
   hand-built perception as well as learned vision
   ([findings](../classical_control/FINDINGS.md#place-target-from-the-camera)).
4. **Different failure geometry:** visual servo fails on cubes near the base on both engines
   (a grasp and sweep problem), ACT on Isaac fails evenly across the workspace (a perception
   problem: M4 ruled out physics). They are complementary, which supports using the classical method as a diagnostic
   baseline and not only a competitor.
5. **A coverage limit of the top-down grasp**, measured: flat or low-tilt grasps reach no cell
   of the sampled region, and a 60-75 degree tilt only extends the outer reach (table under
   [Task definition](#task-definition-frozen-at-m0-before-any-training)).
6. **Benchmark hygiene, with examples.** State-based tests missed that the visible target
   disc did not move in MuJoCo; an expert change fixed one region and broke a sorting seed that
   only a regression run caught; the same expert scored 45% and 60% on two 20-50 seed ranges before its fix, so small evaluations cannot rank policies
   ([expert findings](../scripted_experts/FINDINGS.md)). Useful as a short reproducibility
   section.
7. **The gap is a pure observation-domain shift, shown causally.** The same expert actions,
   replayed on Isaac, deliver the cube 49 times in 50; ACT trained on Isaac's pictures of those
   actions scores 85/100 on Isaac against 55/100 trained on MuJoCo's, and falls from 92% to 76%
   on MuJoCo. It is not colour or brightness (every measured region gain applied together
   leaves 96%), not sharpness, not the gripper-reading lag, and colour augmentation did not
   move Isaac (58 against 62). One policy trained on both engines' images is 98% on MuJoCo and
   94% on Isaac, so the shift is learnable from data of both renderers. Isaac-only
   controls get 91% (200 replays) and 97% (400 different replays) on Isaac and 76% and 84% on
   MuJoCo, so what mixing adds is keeping MuJoCo at 98%; on Isaac the mixed model and both
   controls are not distinguishable (paired p 0.38-0.51), and only 100 against 400 Isaac
   replays is a significant rise (85 to 97, p 0.008). A policy that survives lighting and camera
   shift but collapses with distractor boxes (60 / 46 / 23% for 1 / 2 / 4) is reading scene
   content, which a camera-only detector (95 / 96 / 92%) ignores.
8. **The final ACT is close to open loop.** With chunk 100 played in full it reads the cameras
   and joint state at steps 0, 100 and 200 only, which explains why long chunks win, why
   per-frame edits barely move it, and why half of its Isaac failures never touch the cube.
9. **Everything runs on a 6 GB laptop GPU:** the final ACT trains in about 2 hours, which
   makes the benchmark reproducible without a cluster.

## Task definition (frozen at M0, before any training)

- **Cube and target spawn:** both sampled independently over the region where a top-down
  grasp with hover clearance is reachable, from `scripts/workspace_map.py`: x 0.14-0.27 m,
  y within about ±0.14 m, with the far corners excluded (the map's `o` cells). Left,
  right and front are all covered. Today's task spawns the cube in x 0.20-0.24, y
  0.05-0.13 only, with a fixed target.
- **Grasp approach: top-down only, frozen.** Every method grasps with the gripper pointing
  down. The region above is a limit of the SO-101 (5 DoF, no shoulder roll) with that grasp,
  not of a method, and the paper says so. Side or tilted grasps are not part of this study;
  they are a later study if the results call for one.
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
- **Seeds:** demonstrations from 0-899, validation 900-999 (choosing a training or
  execution setting, never reported as a result), evaluation on held-out 1000-1299 (300
  episodes); the same evaluation seeds for every method and both engines. The first ACT runs
  (below) were scored on 1000-1299 before this split existed, so they are the initial
  configuration; every later setting is chosen on 900-999 and only the chosen one is run on
  1000-1299.

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
- Not needed: the expert on Isaac. Demonstrations come from MuJoCo (the expert reads its
  contact data); Isaac is only where trained policies are evaluated.
- [x] Any failure is fixed at its root cause, or the region is shrunk and the limit
  documented in [FINDINGS](../scripted_experts/FINDINGS.md#randomized-pick-and-place-study-1-m1);
  no method is trained before this.
- [x] Trajectory-quality numbers for the expert (time, path length, jerk) as the efficiency
  reference.

### M2: visual servo on the randomized task

- [x] Detect the place target from the camera (`TargetDiscDetector`); it used to read
  `env.target_pos`, which is privileged and would break the camera-only claim once the target
  is random.
- [x] Evaluate on the protocol below on both engines, with a success map over the
  workspace (`scripts/plot_workspace.py`): MuJoCo 296/300 and Isaac Sim 271/300 on seeds
  1000-1299, with the map in `outputs/study01/`. No further grasp-recovery or robustness
  tuning: report it as it stands.

### M3: ACT

- [x] Demonstrations from the M1 expert: 100 and 200 episodes (seeds 0-199, all kept). 500 is
  deferred: its policy-sized frames are about 11 GB in RAM.
- [x] One fixed training configuration that fits the 6 GB GPU: chunk 100 played in full, 60k
  steps, batch 16, image 128, lr 1e-5, front and wrist cameras, chosen on validation seeds
  900-999 and frozen (settings tried and numbers:
  [FINDINGS](../imitation_learning/FINDINGS.md#randomized-pick-and-place-study-1-m3)).
- [x] Evaluate on both engines; ablation: number of demonstrations. MuJoCo 294/300 (200 demos)
  and 275/300 (100 demos); Isaac Sim 58/100 and 55/100. The wrist-camera ablation and the
  equal-epoch demonstration ablation are not done.
- [x] Choose the paper's contributions from the results so far (see Candidate contributions).

### M4: sim-to-sim gap

- [x] Diagnose the gap before changing training: physics ruled out by replaying the expert's
  actions on Isaac, the images shown to differ by region and not by a global brightness, and
  dimming MuJoCo shown not to reproduce it.
- [x] Causal test: replay the expert's actions on Isaac while recording Isaac's frames, train on
  them and evaluate on both engines. Trained on Isaac's images ACT scores 85/100 on Isaac
  (55/100 when trained on MuJoCo's) and 229/300 on MuJoCo (275/300), so the gap goes both
  ways and comes from the images alone; it is not symmetric (40 points one way, 12 the other).
- [x] Probes of what the policy relies on (region gains, blur, background, camera blanking,
  gripper reading): no single measured difference explains the gap.
- [x] Colour and lighting augmentation at training time, chosen on validation seeds: no gain
  on Isaac (58 against 62 of 100), so not frozen
  ([findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap-on-the-randomized-task-study-1-m4)).
- [x] Train one policy on both engines' images (200 MuJoCo demonstrations and the same 200 replayed
  on Isaac Sim, 400 episodes, 120k steps) and evaluate it on both engines. Criterion fixed
  beforehand (about 90% on MuJoCo and 80% on Isaac means one policy covers both): met, 294/300
  on MuJoCo and 94/100 on Isaac. Control (Isaac-only, same 200 replays and Isaac exposure): 91/100
  on Isaac, 229/300 on MuJoCo; Isaac-only on 400 different replays: 97/100 on Isaac, 253/300 on
  MuJoCo. On Isaac the mixed model is not distinguishable from either control (paired p 0.51
  and 0.38); what mixing buys is keeping MuJoCo at 98%.
- Moved to M5: visual randomization in the MuJoCo renderer.
- [x] Difficulty sweep (lighting, camera shift, clutter) for visual servo and the final ACT.

### M5: attribution and a held-out renderer

Which image content carries the gap, and which training recipe holds on a renderer no model
was trained on. Every comparison is paired on the same seeds (exact McNemar, discordant
counts reported); a difference below p 0.05 is reported as "not distinguishable". Criteria
are fixed here, before the runs.

- [x] **First-frame swap.** The final ACT reads the cameras at steps 0, 100 and 200 only, and
  both engines spawn the same scene (M0), so run it closed loop on MuJoCo with only the step-0
  frames replaced by Isaac's render of the same spawn (test seeds 1000-1099). A drop of 20
  points or more (half the gap) means the first read carries the gap. Then swap one region of
  that frame at a time (floor and sky, table, arm, cube and disc, wrist camera): a region
  carries the gap if swapping it alone costs 10 points or more. Result: 98 to 95/100
  (validation) and 98 to 90/100 (test, paired p 0.02), under the 20 point bar, so the
  per-region swaps were skipped
  ([findings](../imitation_learning/FINDINGS.md#first-frame-swap-study-1-m5)).
- [ ] **Later reads.** The first read carries little, so the gap sits in the reads at steps
  100 and 200 (the arm in view, the cube about to be grasped) or in Isaac's execution of the
  same actions. Measure it without Isaac in the loop: record MuJoCo rollouts' state at the
  reads, render the same states on Isaac, and compare the chunk ACT predicts from each
  render (joint-space distance per read). The same state renderer is what the Newton
  spike needs.
- [ ] **Floor pattern.** The policy loses 13-18 points when the image edit flattens the
  checker background (M4 probes), so the checker floor may be a position cue. Render a
  single-colour floor natively in MuJoCo (needs a floor-look option in the scene) and evaluate
  the final ACT, the Isaac-trained and the mixed ACT and visual servo on 1000-1099. A drop of
  10 points or more is a limit of the benchmark the paper states, and the floor joins the
  randomization below. The benchmark's own floor stays the checker (frozen at M0).
- [ ] **Visual randomization in MuJoCo** (floor, table texture, lighting, distractor boxes):
  demonstrations collected with randomized looks, the final configuration retrained, scored on
  both engines. 80% or more on Isaac means the gap closes without any target-renderer data.
- [ ] **Newton as a third, held-out renderer:** a render-only spike first (MuJoCo physics,
  Newton draws the front and wrist frames from the state). Go if the camera poses match MuJoCo
  within a few pixels, a frame takes under about 1 s at 320 x 240, memory stays under about
  4 GB, and visual servo still reaches 80% there. Then score the MuJoCo, Isaac-400, mixed and
  randomized models on Newton (1000-1099). No-go is documented and a second Isaac look
  (lights, materials) stands in.
- [ ] **Training-seed variance:** two more training seeds of the final ACT, scored on both
  engines, so the headline gap is not one run.
- Not planned: shorter execution or temporal ensembling on Isaac. It lowered success in M3
  and the first-frame swap tests the same mechanism more cheaply.

Done when: the training-recipe by renderer table (MuJoCo, randomized MuJoCo, Isaac, both;
scored on MuJoCo, Isaac, Newton or its stand-in) is filled and the region that carries the gap
is named, or each missing part is documented with its cause.

### M6: paper and release

- [ ] Paper: setup, task, three-method comparison, workspace heatmaps, sim-to-sim gap and its
  attribution, held-out renderer, ablations, failure analysis, limitations (simulation only,
  demonstrations on Isaac are MuJoCo actions replayed).
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
  manipulation benchmarks (robosuite, ManiSkill, RLBench) for positioning, cross-engine
  evaluation (ROBOGATE, arXiv 2603.22126; GS-Playground, 2604.25459), open-loop chunk
  execution (2608.15938) and rigorous policy comparison (TRI, 2405.05439; Badithela et al.,
  2510.04354).
- **Venue:** a robotics workshop paper or an arXiv report first; pick the venue before M6
  so its page limit shapes the figures.
- **Record as you go:** causes and final numbers in each topic's `FINDINGS.md`, so the
  paper's results and failure sections are written from them, not reconstructed.
