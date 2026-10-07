# Imitation learning findings

ACT trained on scripted-expert demonstrations (`scripts/collect_demos.py`, seeds 0-49
or 0-99), evaluated closed loop on held-out seeds 1000 and up with
`scripts/eval_policy.py`. Success is out of the episodes shown, with the Wilson 95%
interval.

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
nothing, and 50 episodes cannot separate 96% from 98%. Scripted expert: 300/300;
`visual_servo`: 100/100 on both engines.

## Sim-to-sim gap

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
