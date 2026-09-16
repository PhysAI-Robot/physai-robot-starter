from physai.robots.turtlebot.navigation import Nav2AcceptanceResult


def test_nav2_acceptance_result_requires_zero_collisions_for_success():
    result = Nav2AcceptanceResult(4, True, False, 0.04, 0, None)
    assert result.succeeded
    assert result.as_dict()["succeeded"] is True

    collision = Nav2AcceptanceResult(4, True, False, 0.04, 1, "collision_detected")
    assert not collision.succeeded
    assert collision.as_dict()["failure_reason"] == "collision_detected"


def test_nav2_acceptance_result_reports_action_timeout():
    result = Nav2AcceptanceResult(None, True, True, None, 0, "action_timeout")
    assert not result.succeeded
    assert result.as_dict()["timed_out"] is True
