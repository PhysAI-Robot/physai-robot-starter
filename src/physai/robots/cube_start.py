"""Where the cube starts, taken from another simulator (study 01).

Off unless an env config names a file. The file is JSON in metres with one of:

    {"source": "...", "offset_xy": [dx, dy]}            added to the drawn xy
    {"source": "...", "xy_by_seed": {"1000": [x, y]}}   replaces the drawn xy

`source` says where the numbers came from (which simulator, which seeds and in
what order they were run). The drawn xy still comes from the env's rng first,
so the target and the randomization are drawn exactly as without the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

XY = tuple[float, float]


def _xy(value: object, where: str) -> XY:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: expected [x, y] in metres, got {value!r}")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class CubeStart:
    offset_xy: XY | None = None
    xy_by_seed: dict[int, XY] | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if (self.offset_xy is None) == (self.xy_by_seed is None):
            raise ValueError("give exactly one of offset_xy and xy_by_seed")

    @classmethod
    def load(cls, path: str | Path) -> CubeStart:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {"source", "offset_xy", "xy_by_seed"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        offset = data.get("offset_xy")
        table = data.get("xy_by_seed")
        return cls(
            offset_xy=None if offset is None else _xy(offset, f"{path}: offset_xy"),
            xy_by_seed=None
            if table is None
            else {int(k): _xy(v, f"{path}: seed {k}") for k, v in table.items()},
            source=str(data.get("source", "")),
        )

    def apply(self, seed: int | None, drawn: XY) -> XY:
        """The xy the cube starts at in this episode, given the xy the env drew."""
        if self.offset_xy is not None:
            return drawn[0] + self.offset_xy[0], drawn[1] + self.offset_xy[1]
        if seed is None:
            raise ValueError("a per-seed cube start needs a seed at reset")
        if seed not in self.xy_by_seed:
            raise KeyError(f"seed {seed} is not in the cube-start table")
        return self.xy_by_seed[seed]
