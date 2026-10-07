import numpy as np


def test_spawn_draws_stay_in_the_region_and_apart():
    from physai.robots.so101.layout import draw_xy

    rng = np.random.default_rng(0)
    for _ in range(1000):
        cube = draw_xy(rng, (0.14, 0.27), (-0.14, 0.14), (0.16, 0.255))
        target = draw_xy(rng, (0.14, 0.27), (-0.14, 0.14), (0.16, 0.255), [cube], 0.08)
        for x, y in (cube, target):
            assert 0.16 <= np.hypot(x, y) <= 0.255
        assert np.hypot(cube[0] - target[0], cube[1] - target[1]) >= 0.08


def test_without_constraints_a_spawn_is_one_draw_of_x_then_y():
    """Recorded datasets and golden layouts depend on this draw order."""
    from physai.robots.so101.layout import draw_xy

    rng, ref = np.random.default_rng(7), np.random.default_rng(7)
    assert draw_xy(rng, (0.2, 0.24), (0.05, 0.13)) == (
        ref.uniform(0.2, 0.24),
        ref.uniform(0.05, 0.13),
    )
