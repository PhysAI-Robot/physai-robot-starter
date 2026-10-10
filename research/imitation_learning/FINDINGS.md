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

**Probes on one model, one change at a time** (ACT-200 on MuJoCo, validation seeds 900-999,
unedited 98/100; `--policy-arg image_edit=` and `state_edit=`, regions segmented from the
projected table top and from colour, gains measured per region on the two renders):

| Edit to what the policy sees | Success |
| --- | --- |
| background x0.67, table x0.93, cube x1.16, arm tinted orange (Isaac's measured gains), each alone | 97 / 99 / 98 / 98 |
| all four together (`isaac_like`, wrist camera x0.8 as well) | 96 |
| Gaussian blur matching Isaac's sharpness (front 0.5 px, wrist 1.0 px), both / front / wrist | 98 / 97 / 98 |
| background replaced by one flat colour / by grey (checker pattern removed) | 80 / 85 |
| front camera blanked / wrist camera blanked | 1 / 47 |
| gripper reading follows the command at once (Isaac-like) / ramps at 0.0142 per step (MuJoCo-like control) | 98 / 98 |
| colour-jitter model: background x0.67 / flat / arm tint / all gains | 97 / 96 / 97 / 98 |

Measured differences that turned out not to matter: Isaac renders the arm orange (red, green,
blue gain 1.03, 0.77, 0.48), the background darker (0.67 by region; 0.52 when taken from the
upper third) and the wrist image far smoother (Laplacian variance 0.06x); and Isaac's gripper
reading reaches its command in a few steps while MuJoCo's ramps at a constant 0.0142 rad per
step (the arm joints agree to 0.002 rad for the same actions). None of them moves the policy
when applied alone, so the 40 point gap is not a colour, brightness, sharpness or gripper-lag
effect. What the policy does need is the background pattern (it loses 13-18 points without
it, and the colour-jitter model does not) and both cameras (front is indispensable).

**How the final policy reads the world.** LeRobot's ACT only reads the observation when its
action queue is empty, so with chunk 100 played in full it looks at the cameras and the joint
state at steps 0, 100 and 200 only: it is close to an open-loop trajectory planned from the
first frame. That explains why longer chunks were better, why edits applied at every frame
move it so little, and the shape of the failures. Of the 42 Isaac failures on seeds
1000-1099, 21 never moved the cube (the jaws missed it; the clearance is about 1 mm) and 21
moved it somewhere else. Where the cube and the target are, as read from the first
frame, is therefore what has to transfer.

**The causal test: train on the other engine's images.** The expert's recorded actions were
replayed on Isaac Sim (49/50 reached the target there, 98/100 for the full set) while
recording Isaac's camera frames and joint states, giving 100 demonstrations with the same
actions, the same physics and different pictures. ACT trained on them with the final
configuration (chunk 100, 60k steps, 100 demos; loss 0.0326 against 0.0314) and scored, on
the test seeds:

| Trained on | MuJoCo 1000-1299 | Isaac 1000-1099 | MuJoCo, same 100 seeds |
| --- | --- | --- | --- |
| MuJoCo images | 275/300 (92%) | 55/100 | 93/100 |
| Isaac images | 229/300 (76%), CI 71-81% | 85/100, CI 77-91% | 73/100 |

Training on Isaac's pictures recovers 30 of the 38 points lost on Isaac, and the same model
loses 15-20 on MuJoCo, so the gap goes both ways and comes from the images alone: an ACT
follows the renderer it was trained on. It is not symmetric: on the same 100 seeds the
MuJoCo-trained models lose 38-40 points on Isaac, the Isaac-trained ones 12 on MuJoCo (85 to
73, 91 to 79, 97 to 85 for 100, 200 and 400 replays; paired p < 0.05 each). The Isaac-trained
100 and 200 models both score 229/300 on MuJoCo by coincidence: they differ on 80 seeds. On Isaac the two models differ on 40 of 100 seeds
(35 only the Isaac-trained one passes, 5 only the MuJoCo-trained one, 10 neither). The
Isaac-trained model's own 85% against 93% in-domain on MuJoCo is most likely the cost of
the replayed actions (2 of 100 replayed demonstrations did not reach the target, and the cube
arrives a little later than under MuJoCo's physics), not a rendering effect, though that was
not tested; its Isaac failures are 15
timeouts, 11 of them ending 10 cm or more from the target. Which image content carries the
shift is still open: no single colour, brightness, sharpness or gripper-state edit above
explains it.

**Training on both engines' images.** One ACT on the 200 MuJoCo demonstrations plus the same
200 replayed on Isaac Sim (400 episodes, 120k steps so the passes over the data match the
200-demo model, same configuration otherwise; loss 0.0297) scored, on the test seeds:

| Trained on | MuJoCo 1000-1299 | Isaac 1000-1099 |
| --- | --- | --- |
| MuJoCo images, 200 demos | 294/300 (98%) | 58/100 |
| Isaac images, 100 demos | 229/300 (76%) | 85/100 |
| Isaac images, 200 replays (control, 60k steps) | 229/300 (76%) | 91/100 (91%), CI 84-95% |
| Isaac images, 400 different replays (control, 120k steps) | 253/300 (84%), CI 80-88% | 97/100 (97%), CI 92-99% |
| **both, 200 + 200** | **294/300 (98%)**, CI 96-99% | **94/100 (94%)**, CI 88-97% |

One policy is as good as the MuJoCo-only one on MuJoCo and far better than either single-engine
model on Isaac (6 failures, all timeouts, 5 of them 10 cm or more from the target, no
collisions). Two Isaac-only controls separate mixing from more data. On the same 200 replays and the same
Isaac exposure (60k steps) the model scores 91/100 on Isaac, against 94/100 mixed (6 seeds only
the mixed one passes, 3 only the control; paired exact McNemar p 0.51). On 400 different replays
(spawns 0-399, 120k steps, so more layouts than the mixed model's 200) it scores 97/100 on
Isaac (CI 92-99%; 4 seeds only it passes, 1 only the mixed one, p 0.38) and 253/300 on MuJoCo
(84%). So the MuJoCo images add nothing measurable on Isaac. More Isaac data trends up (100,
200, 400 demonstrations: 85, 91, 97%), but only 100 against 400 is significant (p 0.008;
100 against 200 p 0.21, 200 against 400 p 0.11). What mixing buys is the other engine: the same
policy keeps 98% on MuJoCo, where the Isaac-only models reach 76% (100 and 200 demonstrations)
and 84% (400). More Isaac data raises MuJoCo too, but not to the MuJoCo-trained level. Limits:
the mixed model sees 200 layouts, each in both renderers, against 400 for the longest control
(equal episode count, half the variety), and the steps differ between runs (the passes over
each model's data are about equal).

**What the replay design does and does not show.** The scripted expert reads MuJoCo's contact
data and only runs on MuJoCo; the "Isaac" demonstrations are its actions replayed open loop on
Isaac Sim with Isaac's frames and joint states recorded (spawns agree: same target, cube within
3 mm). That is the right design for the question asked, whether the gap comes from what the
policy sees, because actions and physics are held fixed and only the observations change. It is
not an Isaac expert: the actions are timed for MuJoCo's dynamics and are not corrected on Isaac
(the cube arrives a little later, and 1-2 of 100 replays miss the target), so the Isaac-trained
numbers are probably a lower bound on what native Isaac demonstrations would give, and no
"expert on Isaac" success rate exists. Say "demonstrations replayed on Isaac (MuJoCo teacher)",
not "Isaac demonstrations".

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
is therefore image content beyond colour, brightness and sharpness; training on Isaac's own
images removes it (above), but which content it is stays unidentified.

## First-frame swap (study 1, M5)

ACT-200 closed loop on MuJoCo, with only the first read of the cameras (step 0, the arm at HOME)
replaced by Isaac's render of the same seed; the reads at steps 100 and 200 stay MuJoCo's
(`--policy-arg image_edit=first_all first_frames=data/m5_first first_seed=...`, frames
captured with `eval_policy.py --policy constant --max-steps 2 --save-dataset`, 200 seeds
900-1099 on each engine). The two first frames are the same scene (cube within 3 mm for 196
of 200 seeds; the other 4, where Isaac knocked the cube at reset, are left unswapped). The
swap is live: the step count changed on 85 of 100 seeds.

| Seeds | Unedited | First frame from Isaac | Paired p |
| --- | --- | --- | --- |
| 900-999 (validation) | 98/100 | 95/100 | not tested |
| 1000-1099 (test) | 98/100 | 90/100 | 0.02 (9 seeds only unedited, 1 only swapped) |

So the first read carries 3-8 points of the 40 point gap on Isaac: most of the gap sits in the
reads at steps 100 and 200 (or in how Isaac executes the same actions), not in where ACT
sees the cube and target at the start. The criterion fixed beforehand (20 points or more) is
not met, and the per-region swaps were not run since the whole frame costs less than 10.

**Later reads and execution.** ACT-200 rolled out on MuJoCo seeds 900-999 (98/100), with
the front and wrist frames and the full state kept at each read (steps 0, 100 and, on 37
seeds that ran past 200 steps, 200). `read_probe.py render` put Isaac's arm and cube at each of
those states (joints within 0.002 rad after the step; 4 of 237 reads moved more than 0.02 rad,
a cube held in the jaws) and rendered both cameras; `compare` asked the model for the 100-step
chunk from each render and measured the pinch-centre distance between the two chunks:

| Isaac frames replace | read 0 | read 100 | read 200 |
| --- | --- | --- | --- |
| both cameras (median of the largest gap along the chunk) | 9.6 mm | 11.3 mm | 14.7 mm |
| front camera only | 8.1 | 9.8 | 9.6 |
| wrist camera only | 2.4 | 5.5 | 8.6 |
| arm / cube and disc / table / background (front region) | 8.2 / 2.0 / 2.8 / 5.1 | 4.3 / 1.4 / 5.0 / 4.9 | 4.6 / 4.9 / 4.4 / 3.2 |
| MuJoCo frames with every measured colour gain (no effect on success) | 7.1 | 5.7 | 5.2 |

Later reads are 1.2-1.5 times the first read, not a different order of magnitude, and no one
region stands out: the chunk moves by about a centimetre whatever part of the picture is
Isaac's, and by 7 mm under colour gains that cost nothing, so this distance does not separate
harmless from harmful edits (the wrist camera's share grows with the read, 2 mm to 9 mm, as
the cube enters the jaws). Execution is not the cause: ACT's own MuJoCo actions replayed open
loop on Isaac (seeds 900-999, `outputs/study01/m5/l5_videos/videos/`) leave the cube within
4 cm of the target in 96 of 100 episodes (70 by the success rule, which the replay cuts short
of the 10 step hold, as for the expert replays above), where ACT closed loop on Isaac scores
58/100. The same actions that fail in Isaac when ACT chooses them from Isaac's pictures succeed
when they were chosen from MuJoCo's, so the gap is in what the policy infers from the pictures.

**Floor pattern (study 1, M5).** The same models on MuJoCo with the floor painted one colour
(the mean of the two tile colours, `floor_style: plain`,
`so101_randomized_pick_place_plainfloor.yaml`), 100 seeds each, paired against the checker floor:

| Model | Validation 900-999 | Test 1000-1099 |
| --- | --- | --- |
| ACT-200 (MuJoCo images) | 98 to 72 (26 seeds only the checker passes, 0 the reverse) | 98 to 83 (15 and 0) |
| ACT, Isaac images, 200 replays | 76 to 42 (40 and 6) | not run |
| ACT, Isaac images, 400 replays | 78 to 17 (63 and 2) | 85 to 20 (65 and 0) |
| ACT mixed, 200 + 200 | 99 to 95 (4 and 0, p 0.125) | 98 to 96 (2 and 0, p 0.5) |
| Visual servo | 100 to 100 | 97 to 97 |

Every single-renderer ACT reads the checker, and the more data of one renderer it has the more
it does (400 Isaac replays: 17% without the pattern); the model trained on both renderers' checkers
barely moves, and the colour-only classical method does not use the floor at all. So the checker
floor is a position cue the benchmark gives learned vision and not a free background, and a
policy that survives a change of renderer may still not survive a change of floor. The benchmark
keeps the checker (frozen at M0); the plain floor is a test condition.

The floor cue is the pattern's presence, not its geometry: ACT-200 on MuJoCo with the tiles
resized to 4, 5, 7 and 9 repeats (6 is nominal) scores 96, 96, 97 and 95 of 100 on validation
seeds (98 at nominal), against 72 with no pattern.

## A third renderer: Newton (study 1, M5)

MuJoCo keeps the physics, the state and the camera poses; `newton_eval.py` sends the state to
`newton_server.py` (Newton 1.6.1 on Warp, its own Python) and uses its pictures of the front and
wrist cameras instead. Newton draws the arm, table, cube and target pad from the same MJCF
and the floor as 1/6 m tiles in the studio's two colours, a directional light with shadows
and the studio sky; the cube, the disc, the arm pose and the tile edges line up with MuJoCo's
frames (`outputs/study01/m5/newton_probe_montage.png`). It is a different renderer (lighting,
shading, no textures or blur of Isaac's kind), unseen by every model. A frame takes 3-5 ms. The go
criteria set beforehand are met (poses line up, under 1 s per frame, memory a few hundred MB,
visual servo 100/100 against the 80% bound), so the models were scored on it with no
training on Newton.

| Model | MuJoCo | Newton, validation 900-999 | Newton, test 1000-1099 (MuJoCo, same seeds) | Paired p |
| --- | --- | --- | --- | --- |
| ACT, MuJoCo images, 200 demos | 98 | 82 | 85 (98) | 0.002 |
| ACT, Isaac images, 200 replays | 79 | 31 | 32 (79) | < 0.001 |
| ACT, Isaac images, 400 replays | 85 | 58 | 54 (85) | < 0.001 |
| **ACT, both renderers, 200 + 200** | **98** | **97** | **95 (98)** | **0.25** |
| Visual servo | 97 | 100 | 100 (97) | 0.25 |

The model trained on MuJoCo's and Isaac's pictures keeps its score on a renderer neither showed
it, and every single-renderer model loses 13 to 47 points there, so training on two renderers
learned something that carries to a third, which one renderer's data (even 400 episodes of
it) does not. Limits: Newton's floor is also a checker (E2 shows the policies read it), the
scene is Newton's reading of the same MJCF and not a photograph, and one third renderer is
one sample of "unseen".

Not done: a mixed model on 400 different layouts (to compare equal variety), visual
randomization in the MuJoCo renderer (floor, textures, distractors), and finding which image
content carries the shift (for example by blending the two renders region by region). The jitter run trained at 5.7 steps/s against 8.3 without it (the augmentation
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
