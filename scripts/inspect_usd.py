"""List the renderable geometry of a USD file: which meshes exist, whether they are
visible, their purpose, and where they sit. For finding out why a robot is not drawn.

    python scripts/inspect_usd.py .isaac_cache/viperx300s/vx300s/vx300s.usda [--variant physx]

Uses plain `pxr` (no Isaac app). Writes the report next to the file as `inspect.txt`.
"""

from __future__ import annotations

from pathlib import Path

from _cli import new_parser


def main() -> int:
    from pxr import Usd, UsdGeom

    parser = new_parser(__doc__)
    parser.add_argument("usd", help="path of the .usd/.usda file to inspect")
    parser.add_argument(
        "--variant",
        default="physx",
        help="USD Physics variant to select before inspecting",
    )
    args = parser.parse_args()
    stage = Usd.Stage.Open(args.usd)
    root = stage.GetDefaultPrim()
    if args.variant:
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection(args.variant)
    lines = [
        f"default prim {root.GetPath()}  loaded paths ok={stage.GetLoadSet() is not None}"
    ]
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"]
    )
    n = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        n += 1
        img = UsdGeom.Imageable(prim)
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lines.append(
            f"MESH {prim.GetPath()}  purpose={img.ComputePurpose()} "
            f"vis={img.ComputeVisibility()} active={prim.IsActive()} "
            f"bbox={tuple(round(v, 3) for v in box.GetMin())}..{tuple(round(v, 3) for v in box.GetMax())}"
        )
    # One visual mesh in full: what the renderer is given.
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if (
            prim.IsA(UsdGeom.Mesh)
            and prim.GetName() == "vx300s_1_base"
            and "_1/" not in str(prim.GetPath())
        ):
            mesh = UsdGeom.Mesh(prim)
            lines.append(
                f"DETAIL {prim.GetPath()} instance_proxy={prim.IsInstanceProxy()}"
            )
            lines.append(
                f"  points={len(mesh.GetPointsAttr().Get() or [])} faces={len(mesh.GetFaceVertexCountsAttr().Get() or [])}"
            )
            lines.append(
                f"  subdiv={mesh.GetSubdivisionSchemeAttr().Get()} orientation={mesh.GetOrientationAttr().Get()} doubleSided={mesh.GetDoubleSidedAttr().Get()}"
            )
            lines.append(f"  extent={mesh.GetExtentAttr().Get()}")
            lines.append(
                f"  primvars={[p.GetPrimvarName() for p in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]}"
            )
            lines.append(
                f"  schemas={prim.GetAppliedSchemas()} xformOps={[o.GetOpName() for o in UsdGeom.Xformable(prim).GetOrderedXformOps()]}"
            )
            from pxr import UsdShade

            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            lines.append(f"  bound material={bound.GetPath() if bound else None}")
            lines.append(
                "  parent chain: "
                + " > ".join(
                    f"{a.GetName()}[{a.GetTypeName()}|inst={a.IsInstanceable()}]"
                    for a in reversed(
                        list(prim.GetPath().GetParentPath().GetPrefixes())
                        and [
                            stage.GetPrimAtPath(x) for x in prim.GetPath().GetPrefixes()
                        ][-4:]
                    )
                )
            )
            break
    lines.append(f"{n} meshes")
    text = "\n".join(lines)
    Path(args.usd).with_name("inspect.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
