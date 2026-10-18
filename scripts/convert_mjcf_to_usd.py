"""Convert a fetched robot MJCF into a USD asset with Isaac Sim's MJCF importer and
print what came out, so the result can be judged before any env is built on it.

    python scripts/convert_mjcf_to_usd.py --robot viperx300s

Writes `outputs/<robot>/usd/` (git-ignored; Isaac's output is not committed, same as
the URDF path) and a plain-text report next to it. Needs the `isaac` extra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli import new_parser

REPO = Path(__file__).resolve().parents[1]

# robot name -> MJCF file under assets/<robot>/
MJCF = {"viperx300s": "vx300s.xml"}


def main() -> int:
    parser = new_parser(__doc__)
    parser.add_argument(
        "--robot",
        default="viperx300s",
        choices=sorted(MJCF),
        help="robot whose MJCF to convert",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default outputs/<robot>/usd)",
    )
    parser.add_argument(
        "--variant",
        default="physx",
        help="USD Physics variant to inspect in the report",
    )
    parser.add_argument(
        "--fix-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fix the robot's base to the world",
    )
    args = parser.parse_args()

    mjcf = REPO / "assets" / args.robot / MJCF[args.robot]
    if not mjcf.exists():
        print(
            f"{mjcf} not found - run `python scripts/fetch_assets.py --robot {args.robot}`"
        )
        return 2
    out = (args.out or REPO / "outputs" / args.robot / "usd").resolve()
    out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        config = MJCFImporterConfig(
            mjcf_path=str(mjcf),
            usd_path=str(out),
            import_scene=False,
            fix_base=args.fix_base,
            collision_from_visuals=False,
            allow_self_collision=False,
        )
        usd_path = MJCFImporter(config).import_mjcf()
        print(f"USD: {usd_path}")
        report = describe(usd_path, variant=args.variant)
        text = "\n".join(report)
        (out / "report.txt").write_text(text, encoding="utf-8")
        print(text)
    finally:
        app.close()
    return 0


def describe(usd_path: str, variant: str = "physx") -> list[str]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    lines = [
        (
            f"upAxis={UsdGeom.GetStageUpAxis(stage)} "
            f"metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}"
        )
    ]
    root = stage.GetDefaultPrim()
    lines.append(f"defaultPrim={root.GetPath() if root else None}")
    if root and variant:
        # The importer leaves the Physics variant unselected, so a plain traversal
        # finds no joints; core's own import (sim/isaac/description.py) selects one too.
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    if root:
        for name in root.GetVariantSets().GetNames():
            vs = root.GetVariantSets().GetVariantSet(name)
            lines.append(
                f"variantSet {name}: {vs.GetVariantNames()} selected={vs.GetVariantSelection()!r}"
            )
    n_art = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            n_art += 1
            lines.append(f"ARTICULATION_ROOT {path}")
        if prim.IsA(UsdPhysics.Joint):
            kind = prim.GetTypeName()
            lo = prim.GetAttribute("physics:lowerLimit").Get()
            hi = prim.GetAttribute("physics:upperLimit").Get()
            b0 = prim.GetRelationship("physics:body0").GetTargets()
            b1 = prim.GetRelationship("physics:body1").GetTargets()
            extra = {
                a.GetName(): a.Get()
                for a in prim.GetAttributes()
                if a.HasAuthoredValue()
                and any(
                    k in a.GetName()
                    for k in ("drive:", "armature", "friction", "mimic", "maxJoint")
                )
            }
            lines.append(
                f"JOINT {path} {kind} lim=({lo},{hi}) schemas={list(prim.GetAppliedSchemas())}"
            )
            lines.append(f"      body0={b0} body1={b1}")
            lines.append(f"      attrs={extra}")
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            lines.append(f"BODY {path} mass={prim.GetAttribute('physics:mass').Get()}")
        elif prim.IsA(UsdGeom.Camera):
            lines.append(f"CAMERA {path}")
        elif "site" in path.lower() and prim.GetTypeName() in (
            "Xform",
            "Sphere",
            "Cube",
            "Scope",
        ):
            lines.append(f"SITE? {path} {prim.GetTypeName()}")
    lines.append(f"articulation roots: {n_art}")
    lines.append("---- prim tree (depth<=4)")
    for prim in stage.Traverse():
        depth = len(prim.GetPath().pathString.strip("/").split("/"))
        if depth <= 4:
            lines.append(
                "  " * (depth - 1) + f"{prim.GetName()} <{prim.GetTypeName()}>"
            )
    return lines


if __name__ == "__main__":
    sys.exit(main())
