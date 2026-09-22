# VLA

Vision-language-action research beyond the checkpoint-backed policy adapter
already covered by `research/imitation_learning/vla_adapter.py`. No module
lives here yet — this directory is a placeholder for future work described
in [ROADMAP.md](../../ROADMAP.md) (Phase 4).

Any module added here registers itself with `physai.policy.registry`; core
never imports this package.
