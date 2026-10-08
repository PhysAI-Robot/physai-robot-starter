# Imitation learning findings

ACT trained on scripted-expert demonstrations (`scripts/collect_demos.py`, seeds 0-49
or 0-99), evaluated closed loop on held-out seeds 1000 and up with
`scripts/eval_policy.py`. Success is out of the episodes shown, with the Wilson 95%
interval.

## Randomized pick-and-place (study 1, M3)

ACT on the randomized task of [study 1](../studies/01_randomized_pick_place.md), trained on
the scripted expert's demonstrations (seeds 0-199, all kept, one dataset) and scored on
held-out seeds 1000-1299 on MuJoCo and 1000-1099 on Isaac Sim. The sections below this one
are the earlier fixed-target task; their checkpoints no longer load (the task was renamed
`single_cube_place`).

**Final configuration:** chunk 100 played in full (`n_action_steps = 100`), 60k steps,
batch 16, image 128, lr 1e-5, front and wrist cameras; about 2 hours on the 6 GB laptop GPU
(8 steps/s).

| Demos | Chunk | Steps | MuJoCo 1000-1299 | 95% CI | Isaac Sim 1000-1099 |
| --- | --- | --- | --- | --- | --- |
| 100 | 30 | 30k | 220/300 (73%) | 68-78% | 26/100 |
| 200 | 30 | 30k | 206/300 (69%) | 63-74% | not run |
| 100 | 100 | 60k | 275/300 (92%) | 88-94% | 55/100 |
| **200** | **100** | **60k** | **294/300 (98%)** | **96-99%** | **58/100** |

For comparison on the same seeds: expert 300/300, visual servo 296/300 on MuJoCo and
271/300 on Isaac ([results table](../scripted_experts/README.md#results)). No run had a
collision or a refused action; every failure is a timeout. On the 100 Isaac seeds the MuJoCo
rate is 93/100 for the 100-demo and 98/100 for the 200-demo model, so the engine gap is about 40
points for both.

**What moved the number.** Settings were chosen on validation seeds 900-999 (ACT, 100 demos),
and only the chosen one was run on 1000-1299:

| Setting (100 demos) | Validation |
| --- | --- |
| chunk 30, 30 of 30 actions, 30k steps | 62/100 |
| chunk 30, first 15 / 10 / 5 actions | 38 / 23 / 2 of 100 |
| chunk 30, temporal ensembling 0.01 | 48/100 |
| chunk 100, 30k steps, 100 / 50 actions | 82 / 65 of 100 |
| chunk 100, 60k steps, 100 actions | 92/100 |

- **Longer chunks played in full are better, and looking again sooner is worse.** Running fewer
  actions per chunk or blending overlapping chunks lowered success monotonically (timeouts rose
  with it), the opposite of what a more reactive policy would give. Chunk 100 (3.3 s) took
  62% to 82%.
- **More steps.** The chunk-100 loss halved from 0.064 to 0.031 between 30k and 60k steps, and
  validation went 82% to 92%.
- **More demonstrations only help once that holds.** At chunk 30 and 30k steps 200 demos
  scored below 100 (206 against 220 of 300, intervals overlapping, final loss 0.042
  against 0.035); at chunk 100 and 60k steps they scored 294 against 275, intervals apart.
  The 200-demo runs saw half as many passes over the data at equal steps; an equal-epoch run
  (120k steps) was not made.
- **Where it fails.** At chunk 30 most failures ended 14 cm from the target (the cube was
  not delivered; only 5% were near misses), and the cube at the sides or far from the base
  failed more. On Isaac the 200-demo model fails evenly (12 of 29 near-base cubes, 30 of 71
  far ones), unlike visual servo, whose Isaac failures cluster near the base.

**Against visual servo, on the same seeds.** On MuJoCo (1000-1299) ACT-200 and visual servo
differ on 10 of 300 seeds (4 only ACT, 6 only visual servo) and both fail 0, so they are
close to interchangeable there. On the first 100 seeds the move from MuJoCo to Isaac costs
visual servo 9 points (97 to 88) and ACT-200 40 (98 to 58); on Isaac 37 seeds pass only for
visual servo and 7 only for ACT-200.

Not done: 500 demonstrations (its policy-sized frames take about 11 GB of the 16 GB RAM),
the wrist-camera ablation, and the equal-epoch demonstration ablation.

## Sim-to-sim gap on the randomized task (study 1, M4)

The final ACT-200 scores 98% on MuJoCo and 58% on Isaac Sim over the same 100 seeds, a 40
point gap (visual servo loses 9). What was tried to find and close it, on validation seeds
900-999 unless stated:

- **Not the physics.** Replaying the expert's recorded actions open loop (no images) on 50
  demonstration seeds scored 50/50 on MuJoCo; on Isaac only 26/50 reached the success rule,
  but 49/50 ended with the cube within 4 cm of the target and all 24 misses used exactly the
  recorded number of steps, so they ran out of actions before the 10 step hold, not out of
  reach. Isaac's contact physics deliver the cube.
- **The ACT failures on Isaac are real misses.** None of the 42 failures on seeds 1000-1099
  ended within 4 cm of the target and 35 ended 10 cm or more away (median 0.16 m); on MuJoCo
  the same seeds have 2 failures.
- **The images differ, but not by a global brightness.** At an identical pose the front
  camera's background is 0.52x as bright on Isaac (100 against 192) while the table is 0.86x
  (183 against 212) and the cube slightly brighter (105 against 91); the wrist camera's
  background is 0.8x. The cube position and the colour detector's output agree to under 1 px.
  Mean pixel difference outside the arm is 56 (front) and 47 (wrist) of 255.
- **Dimming MuJoCo does not reproduce it.** ACT-200 on MuJoCo at lighting x1.0 / 0.7 / 0.5 / 1.4
  scored 98 / 97 / 96 / 96 of 100, while the table also darkens to 133 (Isaac 183) so the
  scene differs from Isaac's.
- **Colour augmentation did not help.** Retraining with random brightness (0.4-1.4),
  contrast, saturation, a slight hue shift and a vertical brightness ramp (to scale the
  background apart from the table) gave the same loss (0.0317 against 0.0318), 96/100 on
  MuJoCo (98 before) and 58/100 on Isaac (62 before). Per seed on Isaac the two models
  differ on 36 of 100 seeds in both directions, so it is noise. The candidate was therefore not
  frozen and not run on the test seeds.

**Sensitivity sweep** (MuJoCo, test seeds 1000-1099, 100 episodes per cell, camera shift not
told to the policy except where marked):

| Cell | Visual servo | ACT-200 |
| --- | --- | --- |
| nominal | 97 | 98 |
| lighting x0.5 / x0.7 / x1.3 / x1.6 | 99 / 99 / 99 / 99 | 95 / 98 / 99 / 95 |
| camera shift 5 / 10 / 20 mm | 100 / 90 / 57 | 97 / 86 / 54 |
| camera shift 20 mm, told | 87 | 53 (cannot use it) |
| clutter 1 / 2 / 4 boxes | 95 / 96 / 92 | 60 / 46 / 23 |

ACT is as robust as the classical method to lighting and camera shift, and far less robust to
distractor boxes it never saw in training; visual servo ignores them because it detects only
red. That fits a policy that conditions on the whole image and fails when something new
appears, which Isaac's different floor, shadows and arm rendering also are. The cause on Isaac
is therefore narrowed to image content beyond colour and brightness, not identified.

Not done: visual randomization in the MuJoCo renderer (floor, textures, distractors) as the
next training change, and a feature-level test (ACT on Isaac frames recoloured to MuJoCo's
statistics). The jitter run trained at 5.7 steps/s against 8.3 without it (the augmentation
runs on the CPU), about 3 hours.

## Results

| Demos | Steps | Engine | Seeds | Success | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 50 | 4k | MuJoCo | 1000-1049 | 17/50 | 22-48% |
| 50 | 30k | MuJoCo | 1000-1049 | 41/50 | 69-90% |
| 100 | 30k | MuJoCo | 1000-1049 | 48/50 | 87-99% |
| 50 | 60k | MuJoCo | 1000-1049 | 49/50 | 90-100% |
| 100 | 60k | MuJoCo | 1000-1049 | 48/50 | 87-99% |
| 100 | 30k | MuJoCo | 1000-1099 | 92/100 | 85-96% |
| 100 | 30k | **Isaac Sim** | 1000-1099 | **54/100** | 44-63% |

The 4k and the first 50-demo 30k runs (17/50 and 37/50) ran before the joint-limit clip
below; the 30k row shows the same checkpoint with the clip.

More demos and more steps each lift 82% to 96-98% on 50 seeds; together they add
nothing, and 50 episodes cannot separate 96% from 98%. The scripted expert and
`visual_servo` rows are in the [results table](../scripted_experts/README.md#results).

## Sim-to-sim gap (earlier fixed-target task)

Superseded in part by the [randomized-task diagnosis](#sim-to-sim-gap-on-the-randomized-task-study-1-m4):
colour augmentation did not close the gap there, so the colour suspect below is not enough.

Per seed, 45 pass on MuJoCo and fail on Isaac, 7 the reverse, 47 pass on both, 1 fails
on both. Isaac failures are timeouts (46), with no collisions and no refused actions.
`scripts/compare_cameras.py` at an identical pose shows the same cube pixel position
and detector output (within 0.1-2.5 px) but very different colours: front background
(192, 203, 196) on MuJoCo against (100, 112, 104) on Isaac, mean absolute pixel
difference outside the arm 56 (front) and 47 (wrist) of 255. That fits `visual_servo`
(colour of the cube) passing on both engines while ACT (whole image) does not, but it
is a suspect, not a proven cause: the gap has not been closed yet.

## Causes found along the way

- ACT overshot a joint limit by 1-6 mrad (`wrist_flex` 1.659 against 1.658), which
  the safety gate rightly refused; clipping in `LeRobotPolicy` took 37/50 to 41/50.
- Training stores only policy-sized frames: full 320 x 240 episodes are about 95 MB
  each, so 100 demos exhausted memory and failed to pickle into DataLoader workers on
  Windows.
- The checkpoint metadata held the instruction text instead of the task name, so
  evaluation refused every checkpoint until the name was stored.
- Isaac evaluation needs shards: 100 episodes took about 2 hours, with no render
  glitch in 10 shards.
