import pytest

from physai.robots import (
    available_kinematics_adapters,
    create_kinematics_adapter,
    register_kinematics_adapter,
)


class FakeAdapter:
    pass


def test_kinematics_registry_keeps_embodiment_adapters_separate():
    name = "test_future_arm"
    register_kinematics_adapter(name, FakeAdapter)
    assert name in available_kinematics_adapters()
    assert isinstance(create_kinematics_adapter(name), FakeAdapter)

    with pytest.raises(ValueError, match="already registered"):
        register_kinematics_adapter(name, FakeAdapter)


def test_kinematics_registry_reports_unknown_adapter():
    with pytest.raises(ValueError, match="unknown kinematics adapter"):
        create_kinematics_adapter("missing_future_arm")