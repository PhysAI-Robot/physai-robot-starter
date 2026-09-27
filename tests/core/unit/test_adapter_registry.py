import pytest


def test_backends_are_discoverable_and_new_ones_are_additive():
    from physai.robots import available_adapters, create_adapter
    from physai.robots.adapters import _ADAPTERS, register_adapter

    assert set(available_adapters()) == {
        "direct",
        "ros2_sim",
        "ros2_real",
    }
    with pytest.raises(ValueError, match="direct"):
        create_adapter("does-not-exist", None)  # the error lists what is available

    sentinel = object()
    register_adapter("_fake_test_backend", lambda direct, **_: sentinel)
    try:
        assert create_adapter("_fake_test_backend", None) is sentinel
    finally:
        del _ADAPTERS["_fake_test_backend"]


def test_direct_adapter_wraps_the_port():
    from physai.robots import DirectAdapter, create_adapter, create_robot

    robot = create_robot("so101", render=False)
    try:
        adapter = create_adapter("direct", robot._environment)
        assert isinstance(adapter, DirectAdapter)
    finally:
        robot.close()
