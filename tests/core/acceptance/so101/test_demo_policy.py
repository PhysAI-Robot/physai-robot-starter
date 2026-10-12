import pytest
from tests.conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_core_demo_policy_picks_and_places_the_cube(seed):
    from physai.config import load_manifest
    from physai.runtime import create_session, run_episode

    manifest = load_manifest("configs/manifests/so101_single_cube_fixed_place.yaml")
    session = create_session(
        manifest.__class__(**{**manifest.__dict__, "policy": "demo"})
    )
    try:
        assert run_episode(session.runtime, seed).success
    finally:
        session.close()
