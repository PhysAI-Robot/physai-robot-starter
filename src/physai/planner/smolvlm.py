"""Local VLM planner backed by Hugging Face SmolVLM."""

from __future__ import annotations

import json
import re

import numpy as np

from ..contracts import Observation
from ..model_store import resolve_local_model
from .base import Plan, Planner
from .claude_vlm import PLAN_SCHEMA, SYSTEM_PROMPT, ClaudePlanner

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"SmolVLM did not return a JSON plan: {text!r}")
    return json.loads(match.group(0))


class SmolVLMPlanner(Planner):
    """Run SmolVLM locally and translate its JSON response into a ``Plan``."""

    name = "smolvlm_planner"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 800,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SmolVLM needs torch and transformers: "
                "pip install -e '.[smolvlm]'"
            ) from exc

        local_model = resolve_local_model(model, model_name="smolvlm")
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            local_model, local_files_only=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            local_model, local_files_only=True
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def plan(self, instruction: str, observation: Observation) -> Plan:
        if not observation.images:
            raise ValueError(
                "SmolVLMPlanner needs camera frames; construct the env with render=True."
            )

        from PIL import Image

        state = observation.joint_state
        state_text = {
            "joint_names": list(state.name),
            "joint_positions_rad": [round(v, 4) for v in state.position.tolist()],
        }
        if observation.ee_pose is not None:
            state_text["gripper_site_xyz_m"] = [
                round(v, 4)
                for v in observation.ee_pose.pose.position.as_array().tolist()
            ]

        schema_text = json.dumps(PLAN_SCHEMA, separators=(",", ":"))
        prompt = (
            f"{SYSTEM_PROMPT}\n\nReturn only valid JSON matching this schema:\n"
            f"{schema_text}\n\nCurrent robot state:\n{json.dumps(state_text)}\n"
            f"Instruction: {instruction}"
        )
        content = [{"type": "text", "text": prompt}]
        images = []
        for frame in observation.images.values():
            image = Image.fromarray(np.asarray(frame.data))
            images.append(image)
            content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            images=images,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            generated[:, prompt_tokens:], skip_special_tokens=True
        )[0]
        return ClaudePlanner._to_plan(instruction, _extract_json(text))