"""Fine-tune ACT (Action Chunking Transformer) on recorded demonstrations.

Uses the real `lerobot` ACTPolicy/ACTConfig — every input/output shape and key
name was verified against the installed library before this script was
written (see conversation history / commit message), not guessed from memory.

    python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
    python scripts/train_act.py --dataset data/pickplace_v1 --steps 4000

Chosen over SmolVLA for this hardware: SmolVLA carries a VLM backbone and
needs far more than 4 GB of VRAM to fine-tune. ACT is the standard
lightweight LeRobot baseline — small ResNet18 vision encoder plus a
transformer, designed for exactly this data scale (tens of demos).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from physai.policy.act_dataset import ACTEpisodeDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/act_ckpt"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors

    meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
    task = meta["task"]
    print(f"dataset: {args.dataset}  episodes: {meta['num_episodes']}  "
          f"task: {task!r}  device: {args.device}")

    train_set = ACTEpisodeDataset(
        args.dataset, chunk_size=args.chunk_size, image_size=args.image_size, task=task,
    )
    print(f"{len(train_set)} (timestep) training samples")

    print("computing normalization stats...")
    stats = train_set.compute_stats()
    args.out.mkdir(parents=True, exist_ok=True)
    stats.to_json(args.out / "dataset_stats.json")

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
    )

    H = W = args.image_size
    input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        device=args.device,
        optimizer_lr=args.lr,
    )
    # NOT config.json — ACTPolicy.save_pretrained() below writes its own
    # config.json into this same directory (the full ACTConfig dump) and
    # would silently clobber this one. Learned by having it happen: the
    # loader read the ACTConfig dump instead, image_size came back None, and
    # LeRobotPolicy._resize() quietly skipped resizing for every inference call.
    # The training seeds are recorded so evaluation can tell whether it is
    # measuring generalisation or reciting the training set. Without this a
    # sorting run was evaluated on seeds 0-19 while training had consumed
    # 0-65, and the resulting 70% was mostly memorised layouts.
    train_seeds = sorted(
        e["seed"] for e in meta.get("episodes", []) if e.get("seed") is not None
    )
    with (args.out / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "chunk_size": args.chunk_size, "image_size": args.image_size,
            "task": task, "steps": args.steps, "batch_size": args.batch_size,
            "lr": args.lr, "device": args.device,
            "dataset": str(args.dataset), "train_seeds": train_seeds,
        }, f, indent=2)

    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=stats.per_key)
    policy = ACTPolicy(cfg).to(args.device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay,
    )

    history: list[dict] = []
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()

        step += 1
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            entry = {"step": step, "loss": loss.item(), **loss_dict,
                     "elapsed_s": round(elapsed, 1)}
            history.append(entry)
            print(f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"{loss_dict}  ({elapsed:.0f}s)")

        if step % args.save_every == 0 or step == args.steps:
            # Only the policy weights round-trip through save/from_pretrained
            # cleanly in this lerobot version — DataProcessorPipeline.from_pretrained
            # loses the ACT-specific tensor<->transition converters and breaks on
            # reload (verified). Regenerate pre/post at load time from the
            # reloaded config + these saved stats instead of trying to persist
            # the pipeline objects themselves.
            policy.save_pretrained(args.out)

    (args.out / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    total_time = time.time() - t0
    print(f"\ndone: {args.steps} steps in {total_time:.0f}s "
          f"({args.steps / total_time:.2f} steps/s)")
    print(f"checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
