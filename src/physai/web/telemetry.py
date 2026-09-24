"""Small, transport-neutral payloads for a browser 3D viewer."""

from __future__ import annotations

import struct
from typing import Any

import mujoco
import numpy as np

_GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_HFIELD: "hfield",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"{object_type}_{index}"


def _quat_xyzw(matrix: Any) -> list[float]:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
    # MuJoCo uses wxyz internally; Three.js uses xyzw.
    return [
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
        float(quaternion[0]),
    ]


def _contact_ring_quat(frame: Any, *, pad_is_geom1: bool) -> list[float]:
    """Orient a flat ring so its face lies flush against the contacted surface.

    `mjContact.frame` is documented as rows [normal, tangent1, tangent2], the
    opposite convention from `xmat`/`geom_xmat` (whose *columns* are the
    local axes in world coordinates, which `_quat_xyzw` assumes). Rebuilding
    the matrix with the normal as the third column -- verified numerically
    against a known flat-table contact -- makes a Three.js ring (whose local
    +Z is its face normal) land exactly on the contact surface.

    MuJoCo's normal points from geom1 to geom2, a detail of internal geom
    ordering the caller has no control over -- verified empirically that for
    a real pad/table contact the pad ends up as geom1, so the raw normal
    points *into* the table. The browser nudges the ring along local +Z to
    avoid z-fighting, so an unflipped normal there buries it invisibly
    inside the surface instead of sitting on top where the pad touched it.
    Swapping the tangents when negating the normal keeps the frame a proper
    (det=+1) rotation, since tangent2 x tangent1 = -normal exactly when
    tangent1 x tangent2 = normal.
    """
    normal, tangent1, tangent2 = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    if pad_is_geom1:
        normal, tangent1, tangent2 = -normal, tangent2, tangent1
    matrix = np.column_stack([tangent1, tangent2, normal])
    return _quat_xyzw(matrix)


def _quat_xyzw_from_wxyz(quaternion: Any) -> list[float]:
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return [float(values[1]), float(values[2]), float(values[3]), float(values[0])]


def _owner(name: str, instance_prefixes: dict[str, str] | None) -> str | None:
    if not instance_prefixes:
        return None
    for instance_id, prefix in instance_prefixes.items():
        if name.startswith(prefix):
            return instance_id
    return None


def _geom_owner(
    model: mujoco.MjModel,
    geom_id: int,
    instance_prefixes: dict[str, str] | None,
) -> str | None:
    owner = _owner(_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), instance_prefixes)
    if owner is not None:
        return owner
    return _owner(
        _name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ),
        instance_prefixes,
    )


def build_scene_manifest(
    model: mujoco.MjModel,
    *,
    robot: str = "",
    instance_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe static MuJoCo geometry that the browser can instantiate."""
    geometries: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        is_mesh = geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
        mesh_id = int(model.geom_dataid[geom_id]) if is_mesh else -1
        mesh_name = _name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if is_mesh else None
        material_id = int(model.geom_matid[geom_id])
        rgba = (
            model.mat_rgba[material_id]
            if material_id >= 0
            else model.geom_rgba[geom_id]
        )
        geometries.append(
            {
                "id": geom_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                **(
                    {"instance_id": _geom_owner(model, geom_id, instance_prefixes)}
                    if instance_prefixes is not None
                    else {}
                ),
                "type": _GEOM_TYPES.get(geom_type, "unknown"),
                "size": [float(value) for value in model.geom_size[geom_id]],
                "rgba": [float(value) for value in rgba],
                "visual": bool(
                    model.geom_contype[geom_id] == 0
                    or (
                        instance_prefixes is not None and model.geom_group[geom_id] != 3
                    )
                ),
                "asset": (
                    f"/assets/so101/assets/{mesh_name}.stl" if mesh_name else None
                ),
                "mesh_id": mesh_id if is_mesh else None,
                "scale": (
                    [float(value) for value in model.mesh_scale[mesh_id]]
                    if is_mesh
                    else None
                ),
                "mesh_position": (
                    [float(value) for value in model.mesh_pos[mesh_id]]
                    if is_mesh
                    else None
                ),
                "mesh_quaternion": (
                    _quat_xyzw_from_wxyz(model.mesh_quat[mesh_id]) if is_mesh else None
                ),
                "body": _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[geom_id]),
                ),
            }
        )
    return {
        "type": "scene",
        "version": 1,
        "robot": robot,
        "geometries": geometries,
    }


def build_mesh_payload(model: mujoco.MjModel, mesh_id: int) -> bytes:
    """Return one MuJoCo-compiled mesh as compact binary geometry data."""
    if mesh_id < 0 or mesh_id >= model.nmesh:
        raise ValueError(f"mesh id {mesh_id} is out of range")
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_end = vert_start + int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_end = face_start + int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_start:vert_end], dtype="<f4")
    indices = np.asarray(model.mesh_face[face_start:face_end], dtype="<u4")
    header = struct.pack("<II", len(vertices), indices.size)
    return header + vertices.tobytes() + indices.tobytes()


def build_state_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    step: int,
    robot: str = "",
    instance_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Serialize dynamic body and geometry transforms without copying meshes."""
    bodies = []
    for body_id in range(1, model.nbody):
        bodies.append(
            {
                "id": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                **(
                    {
                        "instance_id": _owner(
                            _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                            instance_prefixes,
                        )
                    }
                    if instance_prefixes is not None
                    else {}
                ),
                "position": [float(value) for value in data.xpos[body_id]],
                "quaternion": _quat_xyzw(data.xmat[body_id]),
            }
        )
    geometries = []
    for geom_id in range(model.ngeom):
        geometries.append(
            {
                "id": geom_id,
                **(
                    {"instance_id": _geom_owner(model, geom_id, instance_prefixes)}
                    if instance_prefixes is not None
                    else {}
                ),
                "position": [float(value) for value in data.geom_xpos[geom_id]],
                "quaternion": _quat_xyzw(data.geom_xmat[geom_id]),
            }
        )
    joint_widths = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    joints = []
    for joint_id in range(model.njnt):
        start = int(model.jnt_qposadr[joint_id])
        width = joint_widths[int(model.jnt_type[joint_id])]
        joints.append(
            {
                "id": joint_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                **(
                    {
                        "instance_id": _owner(
                            _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                            instance_prefixes,
                        )
                    }
                    if instance_prefixes is not None
                    else {}
                ),
                "qpos": [float(value) for value in data.qpos[start : start + width]],
            }
        )
    # Keyed by (instance, pad, other_geom): a single real touch (e.g. a
    # flat pad resting on the table) commonly shows up as several nearby
    # MuJoCo contact points, and reporting each separately would make the
    # browser's contact-ring indicator jitter between them frame to frame.
    # Averaging position keeps it at one stable spot; the surfaces involved
    # are near-flat at that scale, so the first point's frame (orientation)
    # is representative of the rest too. The same reasoning applies to force:
    # a group's `force_n` sums the normal force over its contact points.
    contact_groups: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    contact_force = np.zeros(6)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        for geom_id, other_id in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name.endswith("pad_static"):
                pad = "static"
            elif name.endswith("pad_moving"):
                pad = "moving"
            else:
                continue
            instance_id = (
                _geom_owner(model, geom_id, instance_prefixes)
                if instance_prefixes is not None
                else None
            )
            other_geom = _name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id)
            key = (instance_id, pad, other_geom)
            group = contact_groups.setdefault(
                key,
                {
                    "pad": pad,
                    "other_geom": other_geom,
                    **(
                        {"instance_id": instance_id}
                        if instance_prefixes is not None
                        else {}
                    ),
                    "quaternion": _contact_ring_quat(
                        contact.frame, pad_is_geom1=(geom_id == contact.geom1)
                    ),
                    "_positions": [],
                    "_normal_force": 0.0,
                },
            )
            group["_positions"].append(np.asarray(contact.pos, dtype=np.float64))
            # Force is in the contact frame, so component 0 is the normal
            # force in newtons; it is zero for contacts MuJoCo excluded.
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            group["_normal_force"] += float(contact_force[0])
    gripper_contacts = []
    for group in contact_groups.values():
        positions = group.pop("_positions")
        group["pos"] = [float(value) for value in np.mean(positions, axis=0)]
        group["force_n"] = group.pop("_normal_force")
        gripper_contacts.append(group)
    return {
        "type": "state",
        "version": 1,
        "robot": robot,
        "step": int(step),
        "sim_time": float(data.time),
        "bodies": bodies,
        "joints": joints,
        "geometries": geometries,
        "gripper_contacts": gripper_contacts,
    }
