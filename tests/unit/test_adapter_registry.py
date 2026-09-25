import pytest


def test_builtin_backends_are_discoverable():
    from physai.robots import available_adapters

    assert set(available_adapters()) == {
        "direct_mujoco",
        "ros2_mujoco",
        "ros2_hardware",
    }


def test_direct_mujoco_adapter_wraps_the_port():
    from physai.robots import DirectMuJoCoAdapter, create_adapter, create_robot

    robot = create_robot("so101", render=False)
    try:
        adapter = create_adapter("direct_mujoco", robot._environment)
        assert isinstance(adapter, DirectMuJoCoAdapter)
    finally:
        robot.close()


def test_unknown_adapter_lists_available_adapters():
    from physai.robots import create_adapter

    with pytest.raises(ValueError, match="direct_mujoco"):
        create_adapter("does-not-exist", None)


def test_new_backend_is_additive_through_register_adapter():
    from physai.robots.adapters import _ADAPTERS, register_adapter

    sentinel = object()
    register_adapter("_fake_test_backend", lambda direct, **_: sentinel)
    try:
        from physai.robots import create_adapter

        assert create_adapter("_fake_test_backend", None) is sentinel
    finally:
        del _ADAPTERS["_fake_test_backend"]
