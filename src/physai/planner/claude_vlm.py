"""VLM high-level planner backed by the Claude Messages API.

Sends the front + wrist camera frames and the current joint state, gets back a
structured plan of sub-goals. Structured outputs (`output_config.format`)
guarantee the response parses — no regex over prose, no retry-on-JSON-error
loop.

    export ANTHROPIC_API_KEY=...
    python scripts/plan_task.py --instruction "pick up the red cube"

Cost note: this runs at plan time (once per task), not at 25 Hz. Do not put it
in the control loop.
"""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

from ..contracts import Observation, Pose, PoseStamped, Vector3
from .base import Plan, Planner, SubGoal

DEFAULT_MODEL = "claude-opus-5"

SKILLS = ["move_above", "grasp", "release", "move_to", "retreat"]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "string",
            "description": "One sentence on what you see and how you decomposed the task.",
        },
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": SKILLS},
                    "target_description": {
                        "type": "string",
                        "description": "The object this sub-goal acts on, in plain language.",
                    },
                    "position_xyz_m": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Goal position of the gripper pinch centre in the robot "
                            "base frame, metres: [x, y, z]."
                        ),
                    },
                    "gripper": {
                        "type": "number",
                        "description": "Normalised aperture, 0 = closed, 1 = open.",
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "skill",
                    "target_description",
                    "position_xyz_m",
                    "gripper",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["notes", "subgoals"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the high-level task planner for an SO-101 5-DoF robot arm.

Frame and workspace facts you must respect:
- Coordinates are metres in the robot base frame. +x points away from the base
  across the table, +y to the robot's left, +z up.
- The table surface is at z = 0.02. Objects resting on it have their centre at
  about z = 0.034.
- The reachable workspace on the table is roughly x in [0.15, 0.30],
  y in [-0.15, 0.15].
- The arm has no shoulder roll, so a straight top-down approach only works
  within about 5 cm of the table. Keep every waypoint z below 0.10.
- `position_xyz_m` is where the centre of the closed jaws should end up, not
  where the wrist goes.

Produce the shortest plan that accomplishes the instruction. A pick-and-place
is normally four sub-goals: move_above the object, grasp it, move_above the
destination, release. Set `gripper` to about 0.55 when open and 0.20 when
holding an object.

If the instruction cannot be carried out from what you can see, return an empty
subgoals list and say why in `notes`."""


def _png_b64(image: np.ndarray) -> str:
    import imageio.v3 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, image, extension=".png")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class ClaudePlanner(Planner):
    name = "claude_planner"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8000,
        effort: str = "medium",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Claude planner needs the anthropic SDK: uv sync --extra vlm"
            ) from exc
        if api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or use ScriptedPlanner to develop offline."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    # ------------------------------------------------------------------
    def _build_content(self, instruction: str, observation: Observation) -> list[dict]:
        content: list[dict] = []
        for name, frame in observation.images.items():
            content.append({"type": "text", "text": f"Camera `{name}`:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _png_b64(frame.data),
                },
            })
        js = observation.joint_state
        state = {
            "joint_names": list(js.name),
            "joint_positions_rad": [round(v, 4) for v in js.position.tolist()],
        }
        if observation.ee_pose is not None:
            state["gripper_site_xyz_m"] = [
                round(v, 4)
                for v in observation.ee_pose.pose.position.as_array().tolist()
            ]
        content.append({
            "type": "text",
            "text": f"Current robot state:\n{json.dumps(state, indent=2)}",
        })
        content.append({"type": "text", "text": f"Instruction: {instruction}"})
        return content

    def plan(self, instruction: str, observation: Observation) -> Plan:
        if not observation.images:
            raise ValueError(
                "ClaudePlanner needs camera frames; construct the env with render=True."
            )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": PLAN_SCHEMA},
            },
            messages=[{"role": "user", "content": self._build_content(instruction, observation)}],
        )

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"Claude declined this request (category="
                f"{getattr(detail, 'category', None)}). Nothing was planned."
            )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Plan was truncated at max_tokens; raise ClaudePlanner(max_tokens=...)."
            )

        text = next(b.text for b in response.content if b.type == "text")
        payload = json.loads(text)
        return self._to_plan(instruction, payload)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_plan(instruction: str, payload: dict) -> Plan:
        subgoals = []
        for sg in payload.get("subgoals", []):
            xyz = np.asarray(sg["position_xyz_m"], dtype=np.float64).reshape(3)
            subgoals.append(
                SubGoal(
                    skill=sg["skill"],
                    target_description=sg.get("target_description", ""),
                    waypoint=PoseStamped(pose=Pose(position=Vector3.from_array(xyz))),
                    gripper=float(sg["gripper"]),
                    rationale=sg.get("rationale", ""),
                )
            )
        return Plan(
            instruction=instruction,
            subgoals=subgoals,
            notes=payload.get("notes", ""),
            raw=payload,
        )
