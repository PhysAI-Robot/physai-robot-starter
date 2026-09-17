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


def _quat_xyzw_from_wxyz(quaternion: Any) -> list[float]:
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return [float(values[1]), float(values[2]), float(values[3]), float(values[0])]


def build_scene_manifest(model: mujoco.MjModel, *, robot: str = "") -> dict[str, Any]:
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
                "type": _GEOM_TYPES.get(geom_type, "unknown"),
                "size": [float(value) for value in model.geom_size[geom_id]],
                "rgba": [float(value) for value in rgba],
                "visual": bool(model.geom_contype[geom_id] == 0),
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
) -> dict[str, Any]:
    """Serialize dynamic body and geometry transforms without copying meshes."""
    bodies = []
    for body_id in range(1, model.nbody):
        bodies.append(
            {
                "id": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "position": [float(value) for value in data.xpos[body_id]],
                "quaternion": _quat_xyzw(data.xmat[body_id]),
            }
        )
    geometries = []
    for geom_id in range(model.ngeom):
        geometries.append(
            {
                "id": geom_id,
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
                "qpos": [float(value) for value in data.qpos[start : start + width]],
            }
        )
    return {
        "type": "state",
        "version": 1,
        "robot": robot,
        "step": int(step),
        "sim_time": float(data.time),
        "bodies": bodies,
        "joints": joints,
        "geometries": geometries,
    }
