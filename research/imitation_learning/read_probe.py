"""Where ACT's renderer gap enters: compare what it predicts from two renders of one state.

ACT with chunk 100 played in full reads the cameras at steps 0, 100 and 200 only. Given
MuJoCo rollouts saved with `eval_policy.py --save-dataset`, `render` puts Isaac Sim's arm and
cube at the state of each read (teleported, not driven there) and renders both cameras;
`compare` asks the model for the 100-step chunk from the MuJoCo frames and from the Isaac
frames at that state and reports how far apart the two chunks are, per read and per image
region swapped. No policy runs on Isaac, so one pass over 100 seeds takes minutes.

    OMNI_KIT_ACCEPT_EULA=YES python research/imitation_learning/read_probe.py render \
        --dataset data/m5_reads/act200_mujoco --out outputs/study01/m5/reads_isaac.npz
    python research/imitation_learning/read_probe.py compare \
        --reads outputs/study01/m5/reads_isaac.npz --checkpoint outputs/study01/act_200_c100_60k

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402

READS = (0, 100, 200)
KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "observation.state",
    "observation.environment_state",
)
# Isaac's table top and the arm's HOME pose differ from MuJoCo's by under a millimetre, so
# the state written to Isaac settles by less than this over the one physics step.
SETTLE_TOLERANCE_RAD = 0.02


def render(args) -> None:
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session

    from physai.data import load_episode

    manifest = with_overrides(
        load_manifest(args.manifest), policy="idle", simulator="isaac"
    )
    session = create_session(manifest, render=True)
    env = session.runtime.robot
    meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
    rows: dict[str, list] = {
        k: []
        for k in (
            "seed",
            "read",
            "mujoco_front",
            "mujoco_wrist",
            "isaac_front",
            "isaac_wrist",
            "state",
            "env_state",
            "isaac_drift",
        )
    }
    for entry in meta["episodes"][: args.limit]:
        episode = load_episode(args.dataset / entry["file"], KEYS)
        env.reset(seed=entry["seed"])
        if not env.robot_is_rendered():
            raise SystemExit(f"seed {entry['seed']}: Isaac is not drawing the robot")
        for read in READS:
            if read >= len(episode["observation.state"]):
                continue
            env_state = episode["observation.environment_state"][read]
            isaac_front, isaac_wrist, drift = _render_state(env, env_state)
            for key, value in (
                ("seed", entry["seed"]),
                ("read", read),
                ("mujoco_front", episode[KEYS[0]][read]),
                ("mujoco_wrist", episode[KEYS[1]][read]),
                ("isaac_front", isaac_front),
                ("isaac_wrist", isaac_wrist),
                ("state", episode[KEYS[2]][read]),
                ("env_state", env_state),
                ("isaac_drift", drift),
            ):
                rows[key].append(value)
        print(f"seed {entry['seed']}: {len(rows['seed'])} reads so far")
    session.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{key: np.asarray(value) for key, value in rows.items()})
    drift = np.asarray(rows["isaac_drift"])
    print(
        f"wrote {len(drift)} reads -> {args.out}; arm drift after the step "
        f"median {np.median(drift):.4f} max {drift.max():.4f} rad "
        f"({(drift > SETTLE_TOLERANCE_RAD).sum()} above {SETTLE_TOLERANCE_RAD})"
    )


def _render_state(env, env_state: np.ndarray):
    """Isaac's front and wrist frames with the arm and cube at `env_state` (MuJoCo qpos)."""
    target = np.zeros((1, len(env.dof_names)), dtype=np.float32)
    target[0, env._arm_indices] = env_state[:5]
    target[0, env._gripper_index] = env_state[5]
    env.articulation.set_dof_positions(target)
    env.articulation.set_dof_position_targets(target)
    env.articulation.set_dof_velocities(np.zeros_like(target))
    env._cube_body.set_world_poses(
        positions=[env_state[6:9]], orientations=[env_state[9:13]]
    )
    env._cube_body.set_velocities(
        linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]]
    )
    # Poses written through the tensor API reach the renderer after a physics step.
    env.core.step_simulation()
    observation = env.observe()
    drift = float(np.abs(np.asarray(env._arm_qpos()) - env_state[:5]).max())
    return (
        np.asarray(observation.images["front"].data),
        np.asarray(observation.images["wrist"].data),
        drift,
    )


VARIANTS = (
    "all",
    "front",
    "wrist",
    "arm",
    "cubedisc",
    "table",
    "background",
    "colour_only",
)
# Pinch-centre distance (mm) between two chunks: a grasp needs about 1 mm of clearance and
# the placement 40 mm, so 10 mm marks a chunk that may well grasp elsewhere.
LARGE_MM = 10.0


def _variant_frames(variant: str, row: dict, table: np.ndarray):
    """(front, wrist) that stand for "this part of the scene looked like Isaac's"."""
    from research.imitation_learning import image_edits as ie

    m_front, m_wrist = row["mujoco_front"], row["mujoco_wrist"]
    i_front, i_wrist = row["isaac_front"], row["isaac_wrist"]
    if variant == "all":
        return i_front, i_wrist
    if variant == "front":
        return i_front, m_wrist
    if variant == "wrist":
        return m_front, i_wrist
    if variant == "colour_only":  # every measured colour gain, applied to MuJoCo
        return ie.edit_front("isaac_like", m_front, table), ie.edit_wrist(
            "isaac_like", m_wrist
        )
    return ie.swap_front(f"first_{variant}", m_front, i_front, table), m_wrist


def compare(args) -> None:
    import torch

    import research.imitation_learning.vla_adapter  # noqa: F401  (registers "lerobot")
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session

    manifest = with_overrides(
        load_manifest(args.manifest), policy="lerobot", simulator="mujoco"
    )
    session = create_session(
        manifest, render=True, policy_kwargs={"checkpoint": args.checkpoint}
    )
    env, policy = session.runtime.robot, session.runtime.policy
    frame = env.reset(seed=900).images["front"]
    from research.imitation_learning.image_edits import table_polygon_mask

    scene = env.cfg.scene
    table = table_polygon_mask(
        frame.data.shape[:2],
        frame.intrinsics,
        frame.extrinsics,
        scene.table_pos,
        scene.table_size,
    )
    with np.load(args.reads) as archive:
        data = {key: archive[key] for key in archive.files}
    rows = [{k: v[i] for k, v in data.items()} for i in range(len(data["seed"]))]

    def chunk(front, wrist, state):
        sample = {
            "observation.images.front": policy._resize(front),
            "observation.images.wrist": policy._resize(wrist),
            "observation.state": torch.from_numpy(np.asarray(state)).float(),
            "task": policy.instruction,
        }
        with torch.no_grad():
            processed = policy.preprocessor(sample)
            out = policy.policy.predict_action_chunk(processed)
            out = policy.postprocessor(out)
        return policy._clip_to_joint_limits(out.squeeze(0).cpu().numpy())

    # The chunk must start with the action the running policy would take.
    first = rows[0]
    policy.policy.reset()
    sample = {
        "observation.images.front": policy._resize(first["mujoco_front"]),
        "observation.images.wrist": policy._resize(first["mujoco_wrist"]),
        "observation.state": torch.from_numpy(first["state"]).float(),
        "task": policy.instruction,
    }
    with torch.no_grad():
        action = policy.postprocessor(
            policy.policy.select_action(policy.preprocessor(sample))
        )
    reference = chunk(first["mujoco_front"], first["mujoco_wrist"], first["state"])
    gap = np.abs(
        policy._clip_to_joint_limits(action.squeeze(0).cpu().numpy()) - reference[0]
    ).max()
    if gap > 1e-4:
        raise SystemExit(f"chunk[0] differs from select_action by {gap:.2e}")

    kin = env.kin
    times = np.arange(0, reference.shape[0], 5)

    def pinch(c):
        return np.array([kin.pinch_center_from_qpos(c[t, :5]) for t in times])

    result = {}
    for variant in VARIANTS:
        for row in rows:
            base = chunk(row["mujoco_front"], row["mujoco_wrist"], row["state"])
            front, wrist = _variant_frames(variant, row, table)
            other = chunk(front, wrist, row["state"])
            mm = np.linalg.norm(pinch(base) - pinch(other), axis=1) * 1000.0
            result.setdefault((variant, int(row["read"])), []).append(
                (
                    float(mm.max()),
                    float(mm.mean()),
                    float(np.abs(base[:, :5] - other[:, :5]).mean()),
                )
            )
        print(f"{variant} done")
    lines = [
        "| variant | read | n | median max pinch gap mm | p90 | share over "
        f"{LARGE_MM:g} mm | median joint gap rad |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (variant, read), values in sorted(
        result.items(), key=lambda kv: (VARIANTS.index(kv[0][0]), kv[0][1])
    ):
        a = np.asarray(values)
        lines.append(
            f"| {variant} | {read} | {len(a)} | {np.median(a[:, 0]):.1f} | "
            f"{np.percentile(a[:, 0], 90):.1f} | {(a[:, 0] > LARGE_MM).mean():.0%} | "
            f"{np.median(a[:, 2]):.4f} |"
        )
    text = chr(10).join(lines)
    print(text)
    if args.table:
        args.table.write_text(text + chr(10), encoding="utf-8")
    session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("render", help="render MuJoCo rollout states on Isaac Sim")
    one.add_argument("--dataset", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--limit", type=int, help="only the first N episodes")
    one.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "configs/manifests/so101_randomized_pick_place.yaml",
    )
    two = sub.add_parser("compare", help="chunk ACT predicts from each render")
    two.add_argument("--reads", type=Path, required=True)
    two.add_argument("--checkpoint", type=Path, required=True)
    two.add_argument("--table", type=Path, help="write the markdown table here")
    two.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "configs/manifests/so101_randomized_pick_place.yaml",
    )
    args = parser.parse_args()
    {"render": render, "compare": compare}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
