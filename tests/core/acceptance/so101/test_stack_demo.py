from pathlib import Path

import pytest
from tests.conftest import requires_assets

pytestmark = pytest.mark.assets

MANIFEST = Path("configs/manifests/so101_stack_cubes.yaml")


@requires_assets
@pytest.mark.parametrize("seed", [0, 1])
def test_the_demo_policy_stacks_three_cubes_from_the_side(seed):
    """The whole stack, end to end, on a spread of seeds."""
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session, run_episode

    manifest = with_overrides(load_manifest(MANIFEST), policy="demo", seed=seed)
    session = create_session(manifest, render=False)
    try:
        result = run_episode(session.runtime, seed)
        assert result.info["success"], result.info["cubes_stacked"]
        assert result.info["cubes_stacked"] == 3
    finally:
        session.close()
