# 16. isaacsim as a project extra, not a separate venv

## Status

Accepted. Supersedes the separate-virtual-environment recipe in
[ADR 13](0013-isaac-sim-optional-backend.md).

## Context

ADR 13 ruled out a normal `pyproject.toml` extra for two reasons: (1)
`isaacsim==6.1.0.0` pins `numpy==2.3.1`, conflicting with this project's own
`numpy>=2.0,<2.3`, a conflict against a *base* dependency that `[tool.uv]
conflicts` (which only forks *extras* against each other) cannot express;
(2) isaacsim needs a custom package index (`pypi.nvidia.com`) and allows
pre-release versions, neither of which this project set globally. The
mandated workaround was a fully separate virtual environment with
`PYTHONPATH` pointed at this repo's `src/`.

Two things changed since: `lerobot==0.6.1` (not MuJoCo, which has no numpy
pin at all) turned out to be the actual source of the `<2.3` ceiling, and it
was dropped from the `vla` extra pending a lerobot release that supports
`numpy>=2.3` — after which reason (1) no longer applied to anything in this
project. That reopened the question of reason (2): could `uv`'s per-extra
conflict forking and per-package index scoping handle the rest, without a
separate venv?

## Decision

Add `isaac = ["isaacsim[all,extscache]==6.1.0.0"]` as a normal
`project.optional-dependencies` extra, installed with `uv sync --extra
isaac` straight into this project's own `.venv`. Three `uv` settings make
the resolution work, discovered by iterating on real `uv lock` errors
rather than guessing from documentation:

1. **`[[tool.uv.index]] name = "nvidia" url = "https://pypi.nvidia.com"`** —
   a second package index, needed because isaacsim's own ecosystem
   (`isaacsim-*`, `warp-lang`, `mujoco-usd-converter`, `usd-exchange`, the
   `nvidia-cu*` CUDA wheels, and a couple of NVIDIA-repackaged builds of
   `opencv-python-headless-noffmpeg`/`tinyobjloader`) is not on regular
   PyPI.
2. **`[tool.uv] index-strategy = "unsafe-best-match"`** — without it, `uv
   lock` fails on `mujoco-usd-converter==0.5.0` ("found on
   pypi.nvidia.com, but not at the requested version... by default, uv
   will only consider versions published on the first index that contains
   a given package, to avoid dependency confusion"). This is a
   project-wide setting, not scoped to the `isaac` extra — accepted as low
   risk given the package names involved are NVIDIA/Omniverse-specific and
   unlikely to be squatted on PyPI, but it is a standing, not
   Isaac-specific, security posture change worth knowing about.
3. **`fastapi`/`uvicorn[standard]` moved into base `dependencies`, with
   `uvicorn`'s floor relaxed to `>=0.29`** (was a `web` extra requiring
   `>=0.53.0`). `isaacsim-kernel` pins `uvicorn==0.29.0` exactly and some
   exact `fastapi` version; both are base dependencies pulled into every
   fork, but a *loose enough* base specifier still lets each fork resolve
   its own compatible version (see the `mujoco`/`numpy` forking below) —
   this is the general form of what `[tool.uv] conflicts` is for (forking
   incompatible requirements apart) applied to a base dependency instead of
   an extra, which reason (1) in ADR 13 said could not be done; it turns
   out it can, as long as the base specifier's floor doesn't rule out
   whichever exact version the conflicting extra needs. The browser viewer
   (`physai.web`, `--serve`) is not really an optional "mode" distinct from
   the rest of the project (`--viewer` needs neither it nor `isaac`,
   `physai.web.host.Host` itself has no `fastapi`/`uvicorn` import at all),
   so there is no reason to make choosing it a decision against `isaac` in
   the first place.
4. **`webtest = ["httpx2>=2.13"]`, a new extra, with `[tool.uv] conflicts =
   [[{ extra = "isaac" }, { extra = "webtest" }]]`** — `httpx2` (used only
   by `tests/core/unit/test_web_app_streaming.py`, guarded by
   `pytest.importorskip("httpx2")`) itself requires `idna>=3.18`, and
   `isaacsim-kernel` pins `idna==3.10` exactly. Unlike `uvicorn`, there is
   no loose-enough floor on `httpx2`'s own side that resolves this (every
   released `httpx2>=2.1` requires `idna>=3.18`; older ones lack the
   `ASGITransport` behavior the test exercises) — a genuine conflict, not
   a specifier that was just drawn too tight. Isolating it into its own
   extra keeps it out of everyone else's sync (only that one test skips
   without it) and lets `conflicts` fork it away from `isaac` explicitly
   instead of `uv lock` refusing the whole project.
5. **`pytest`/`ruff`/`lark`/`import-linter` (the old `dev` extra) moved
   into base `dependencies` too**, the same reasoning as item 3: this is a
   starter repo worked in directly, not a library, so there is no audience
   that wants it without its own test suite runnable, and folding it in
   removes one more extra someone would otherwise have to remember to add
   alongside `isaac`. Verified compatible the same way (already resolved
   and ran cleanly as an extra alongside `isaac` before the fold; folding
   changes nothing about that compatibility, only who has to ask for it).

This leaves exactly two extras with real, opt-in content:
**`training`** (Gymnasium) and **`isaac`**, plus the narrow, isaac
-incompatible **`webtest`** and the heavy, ACT/LeRobot-specific **`vla`**
(kept separate from base deliberately — `torch` is large and only needed
for that one workflow, unlike the small, universally-useful dev tooling
and browser viewer above).

No pre-release allowance (`--prerelease=allow`, `[tool.uv] prerelease`) was
needed anywhere, contrary to ADR 13's original recipe and to what a manual
`uv pip install --prerelease=allow` resolves. `uv`'s resolver, given the
wider search space of two indexes plus its own default "prefer stable,
fall back only when necessary" policy, found stable releases where the
manual pip-based recipe had picked newer pre-releases (`pydantic` 2.13.5
instead of 2.14.0b2, `pyopengl` 3.1.10 instead of 4.0.0a6, `warp-lang`
1.17.0 instead of a `.dev` snapshot); it fell back to a pre-release only
for the two packages with no stable release at all
(`opencv-python-headless-noffmpeg==4.14.0.94rc1`,
`tinyobjloader==2.0.0rc13`), scoped to just those packages by construction,
not a global policy change.

Because `mujoco` and `numpy` are base dependencies pulled into every fork,
and isaacsim's own dependency tree needs specific compatible versions of
each, the lockfile carries **two resolutions** of both: the `isaac` fork
gets `mujoco==3.11.0`/`numpy==2.3.1` (what isaacsim's tree needs), any
other fork gets whatever the unconstrained ranges
(`mujoco>=3.2`/`numpy>=2.0`) resolve to independently (`3.13.0`/`2.2.6` at
the time of writing). `physai.sim.mujoco` already only assumes
`mujoco>=3.2`, so this is not a new constraint — it is verified working
under `mujoco==3.11.0` (see Consequences).

## Consequences

**Verified end to end**, not just resolved on paper:
- `uv sync --extra isaac` (or `--extra isaac --extra training`) installs
  cleanly into this project's own `.venv`.
- The full test suite passes under it: 170 passed, 3 skipped, including
  `tests/core/acceptance/so101/test_isaac_env.py` running for real against
  Isaac Sim 6.1.0.0 (previously skipped everywhere this project's own venv
  was used, since `isaacsim` was never importable there).
- A real headless episode ran via `scripts/run_sim.py --manifest
  configs/manifests/so101_isaac.yaml` (400 steps, `policy: constant`) end
  to end against real Isaac Sim on an RTX 3060 Laptop GPU, with
  `OMNI_KIT_ACCEPT_EULA=YES` set to accept the NVIDIA Omniverse EULA
  non-interactively (a one-time, explicit decision — this project does not
  set it anywhere on the user's behalf; it is an environment variable the
  person running the command sets, not project configuration).
- Found and fixed a real bug in the process: `run_sim.py`'s headless
  default policy (`"scripted"`, `so101_pick_place_expert.py`) needs
  MuJoCo-only `ArmKinematics` and crashed with a confusing
  `AttributeError: 'SO101IsaacEnv' object has no attribute 'kin'` deep
  inside research code. `run_episodes()` now raises a clear error naming
  the actual cause when no `--policy` is given for a non-`"mujoco"`
  simulator, and the shipped `so101_isaac.yaml` sets `policy: constant`
  explicitly.
- Found and fixed a second one: `uv sync` only removes files a wheel's own
  RECORD lists, not `extsUser`/`kit` directories `isaacsim` writes under
  its own install path at runtime, so switching away from the `isaac`
  extra can leave a stale, empty `isaacsim` namespace package behind.
  `pytest.importorskip("isaacsim")` alone does not notice — it only checks
  that the name imports, not that anything is inside it — so
  `test_isaac_env.py` now also checks `hasattr(isaacsim, "SimulationApp")`
  before running, skipping cleanly instead of erroring when this happens.

**The browser viewer and the dev/test tooling are no longer "modes" to
switch away from at all.** `fastapi`/`uvicorn`/`pytest`/`ruff` are base
dependencies now, so `uv sync --extra isaac --extra training` installs
everything — a working `physai.web`/`--serve` stack and the full test
suite included — in one sync, with no `web` or `dev` extra left to choose
between or forget. The only thing that still cannot combine with `isaac`
is `webtest` (`httpx2`, for one streaming test), a narrow, rarely-needed
exception rather than the project's whole web-serving or testing
capability. CI and normal dev flows (`training`) are unaffected by
`isaac` existing as an extra unless someone explicitly asks for it.

ADR 13 remains for its numpy-conflict analysis and the
`SO101IsaacEnv`-is-robot-only design decision, which are unchanged; its
separate-venv recipe is what this ADR replaces.
