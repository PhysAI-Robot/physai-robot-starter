# Study 2: RL refinement of ACT on the SO-101

**Status:** planned, not started. Split out of [study 01](01_randomized_pick_place.md), whose
task, seeds, evaluation protocol and ACT checkpoints it reuses. `[ ]` marks planned work.

**Question:** does RL on top of a frozen ACT raise success and shorten, smooth the motion
(completion time, path length, jerk) on the randomized pick-and-place task?

**Why a separate study:** ACT already reaches 98% on MuJoCo, so success has little room there;
RL on Isaac Sim does not fit the evaluation budget (about 2 hours per 100 episodes); residual
RL on imitation policies is a crowded area (Ankile et al., 2024 and later). Study 01 is about
the renderer gap and does not need RL to answer it.

## Before starting

- Name what is new against residual-RL work on ACT, or do not start.
- The final ACT is close to open loop (chunk 100 played in full, cameras read at steps 0, 100
  and 200; [findings](../imitation_learning/FINDINGS.md#sim-to-sim-gap-on-the-randomized-task-study-1-m4)),
  so a per-step residual makes the policy closed loop: decide whether that is the question.

## Milestones

### M1: RL refinement

- [ ] Residual policy on a frozen ACT: RL learns a small correction to ACT's action,
  trained with PPO through the existing Gymnasium adapter and `task.reward` plus
  efficiency terms (time, path length, jerk).
- [ ] If image-based RL does not fit the GPU, train the critic on simulator state and the
  actor on ACT's inputs (asymmetric actor-critic), and say so in the paper.
- [ ] Report ACT against ACT + RL on success and on the efficiency metrics, with study 01's
  protocol.

Done when: the RL row is in the results table, or the attempt is documented as negative
with its cause (a negative result is still reportable).

## Related work

Residual RL (Silver et al., 2018; Johannink et al., 2019; Ankile et al., 2024).
