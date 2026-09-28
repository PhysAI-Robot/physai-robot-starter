# 16. isaacsim as a project extra, not a separate venv

## Status

Accepted. Supersedes the separate-venv recipe in
[ADR 13](0013-isaac-sim-optional-backend.md).

## Context

ADR 13 ruled out a normal `pyproject.toml` extra for two reasons: (1)
`isaacsim==6.1.0.0` pins `numpy==2.3.1` against this project's own
`numpy>=2.0,<2.3`, a conflict against a *base* dependency that `[tool.uv]
conflicts` (extras-only) cannot express; (2) isaacsim needs a custom
package index and pre-release versions, neither set globally here.

Reason (1) turned out to trace to `lerobot==0.6.1` (`numpy<2.3.0`), not
MuJoCo. Dropping `lerobot` from `vla` removed the ceiling entirely, which
reopened whether `uv`'s per-extra conflict forking and index scoping could
handle reason (2) without a separate venv.

## Decision

Add `isaac = ["isaacsim[all,extscache]==6.1.0.0"]` as a normal extra,
`uv sync --extra isaac` straight into this project's own `.venv`. Found by
iterating on real `uv lock` errors, not by guessing from docs:

1. `[[tool.uv.index]] name = "nvidia" url = "https://pypi.nvidia.com"` —
   isaacsim's ecosystem (`isaacsim-*`, `warp-lang`, `mujoco-usd-converter`,
   `usd-exchange`, `nvidia-cu*`, and NVIDIA-repackaged
   `opencv-python-headless-noffmpeg`/`tinyobjloader`) isn't on regular PyPI.
2. `[tool.uv] index-strategy = "unsafe-best-match"` — otherwise `uv lock`
   refuses a package (`mujoco-usd-converter`) found only on the second
   index, to guard against dependency confusion. Project-wide, not scoped
   to `isaac`; accepted as low risk given these NVIDIA-specific names, but
   worth knowing it isn't Isaac-scoped.
3. `fastapi`/`uvicorn[standard]` moved into base `dependencies`, uvicorn's
   floor relaxed to `>=0.29` (was a `web` extra requiring `>=0.53.0`).
   `isaacsim-kernel` pins `uvicorn==0.29.0` exactly; a loose-enough base
   specifier still lets each fork resolve its own compatible version (see
   `mujoco`/`numpy` below) — the same trick `[tool.uv] conflicts` does for
   extras, done for a base dependency instead, which ADR 13 assumed
   impossible. The browser viewer was never really a mode distinct from
   the rest of the project anyway (`--viewer` needs neither it nor
   `isaac`; `physai.web.host.Host` has no `fastapi`/`uvicorn` import).
4. `webtest = ["httpx2>=2.13"]`, new, with `[tool.uv] conflicts =
   [[{extra="isaac"}, {extra="webtest"}]]` — `httpx2` needs `idna>=3.18`,
   `isaacsim-kernel` pins `idna==3.10` exactly, and no `httpx2` release has
   a loose enough floor to bridge that (a genuine conflict, unlike
   uvicorn's). `httpx2` is used only by
   `tests/core/unit/test_web_app_streaming.py`, already guarded by
   `pytest.importorskip`, so isolating it costs one skipped test.
5. `pytest`/`ruff`/`lark`/`import-linter` (the old `dev` extra) moved into
   base too, same reasoning as (3): this is a starter repo worked in
   directly, not a library, so nobody wants it without a runnable test
   suite. Already verified compatible with `isaac` as an extra; folding it
   in only changes who has to ask for it.
6. `vla`'s `lerobot==0.6.1` conflicts with `isaac` the same way `httpx2`
   does (`numpy<2.3.0` vs. isaacsim's `numpy==2.3.1`, no loose-enough floor
   on lerobot's side either), so `[tool.uv] conflicts` marks `vla` against
   `isaac` too rather than dropping `lerobot` from the project or asking
   for a manual `pip install` outside the lockfile.

That leaves two extras that combine freely — `training` (Gymnasium) and
`isaac` — plus `webtest` and `vla`, both narrow and both genuinely
isaac-incompatible (not just heavy).

No pre-release allowance was needed anywhere, unlike ADR 13's manual
recipe: `uv`'s resolver, given a wider search space, picked stable releases
where manual pip had picked newer pre-releases (`pydantic` 2.13.5 not
2.14.0b2, `pyopengl` 3.1.10 not 4.0.0a6, `warp-lang` 1.17.0 not a `.dev`
build), falling back to a pre-release only for the two packages with no
stable release at all.

`mujoco`/`numpy` are base dependencies with two resolutions in the
lockfile: the `isaac` fork gets `mujoco==3.11.0`/`numpy==2.3.1` (what
isaacsim needs), any other fork resolves `mujoco>=3.2`/`numpy>=2.0`
independently. `physai.sim.mujoco` already only assumes `mujoco>=3.2`, so
this isn't a new constraint.

## Consequences

Verified end to end, not just resolved on paper: `uv sync --extra isaac
--extra training` installs cleanly; the full test suite passes under it
(170 passed, 3 skipped), including `test_isaac_env.py` running for real
against Isaac Sim 6.1.0.0 on an RTX 3060; a real 400-step headless episode
ran via `scripts/run_sim.py --manifest configs/manifests/so101_isaac.yaml`
with `OMNI_KIT_ACCEPT_EULA=YES` set to accept the NVIDIA EULA
non-interactively (an environment variable the person running it sets —
nothing here sets it on their behalf).

Three real bugs found and fixed along the way: `run_sim.py`'s headless
default policy (`"scripted"`) needs MuJoCo-only `ArmKinematics` and crashed
confusingly on other simulators — `run_episodes()` now raises a clear error
instead, and the shipped `so101_isaac.yaml` sets `policy: constant`. `uv
sync` leaves a stale, empty `isaacsim/` namespace package behind when the
`isaac` extra is dropped (it only removes files a wheel's RECORD lists,
not directories isaacsim writes at runtime) — `test_isaac_env.py` now
checks for `SimulationApp` specifically instead of trusting a bare import.
And `lerobot`'s `draccus` dependency installs its own test suite as a
top-level `tests` package in site-packages; per PEP 420 a regular package
anywhere on `sys.path` wins over a namespace-package portion regardless of
order, so every `from tests.core... import ...` absolute import in this
suite silently broke whenever `vla` was installed — `tests/__init__.py`
(previously absent; this repo's own `tests/` was a namespace package) fixes
it by making this project's `tests` win instead, which in turn required
`tests`'s own bare `from conftest import ...` pattern (16 files) to become
`from tests.conftest import ...`, since packaging `tests/` changes where
pytest resolves that name from too.

The browser viewer and dev tooling are no longer modes to switch away
from: `uv sync --extra isaac --extra training` gets everything, with no
`web`/`dev` extra left to forget. CI and normal dev flows are unaffected
by `isaac` existing unless someone asks for it.

ADR 13 remains for its numpy-conflict analysis and the
`SO101IsaacEnv`-is-robot-only decision, both unchanged; its separate-venv
recipe is what this ADR replaces.
