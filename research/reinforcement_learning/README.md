# Reinforcement learning

Deep RL research built on top of the core Gymnasium adapter
(`physai.data.gym_env.GymnasiumAdapter`). No training code lives here yet —
this directory is a placeholder for the optional RL work described in
[ROADMAP.md](../../ROADMAP.md).

Any RL policy or training script added here registers itself with
`physai.policy.registry`; core never imports this package.
