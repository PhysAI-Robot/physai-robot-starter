"""Translate the browser's command payload into the shared ``Action`` contract."""

from __future__ import annotations

from typing import Any

from ..contracts import Action, GripperCommand, Twist, Vector3


def action_from_payload(payload: dict[str, Any]) -> Action:
    """Convert the browser command schema to the shared ``Action`` contract."""
    mode = payload.get("mode")
    if mode == "joint_position":
        return Action(
            joint_position=payload["position"],
            joint_names=tuple(payload["names"]) if payload.get("names") else None,
            gripper=GripperCommand(float(payload.get("gripper", 1.0))),
        )
    if mode == "twist":
        linear = payload.get("linear", {})
        angular = payload.get("angular", {})
        gripper = payload.get("gripper")
        return Action(
            ee_twist=Twist(
                linear=Vector3(
                    float(linear.get("x", 0.0)),
                    float(linear.get("y", 0.0)),
                    float(linear.get("z", 0.0)),
                ),
                angular=Vector3(
                    float(angular.get("x", 0.0)),
                    float(angular.get("y", 0.0)),
                    float(angular.get("z", 0.0)),
                ),
            ),
            gripper=(GripperCommand(float(gripper)) if gripper is not None else None),
        )
    raise ValueError("command mode must be 'joint_position' or 'twist'")


__all__ = ["action_from_payload"]
