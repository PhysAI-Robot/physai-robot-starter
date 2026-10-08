import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))


def test_eval_command_carries_the_policy_flags_and_the_cell():
    from sweep_difficulty import CELLS, eval_command

    cell = next(c for c in CELLS if c.name == "lighting_x0.5")
    command = eval_command(
        cell,
        Path("out/lighting_x0.5.json"),
        seed=1000,
        episodes=100,
        max_steps=None,
        policy_flags=("--policy", "lerobot", "--checkpoint", "ckpt"),
    )
    assert command[command.index("--policy") + 1] == "lerobot"
    assert command[command.index("--checkpoint") + 1] == "ckpt"
    assert command[command.index("--lighting-scale") + 1] == "0.5"
    assert command[command.index("--seed") + 1] == "1000"
    assert "visual_servo" not in command
