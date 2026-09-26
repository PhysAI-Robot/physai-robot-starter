from physai.bridge import ALL_ENDPOINTS, EXTERNAL_INPUTS, Direction
from physai.bridge.ros2_contract import describe


def test_ros2_contract_is_self_consistent():
    published = {e.topic for e in ALL_ENDPOINTS if e.direction is Direction.PUBLISH}
    subscribed = {e.topic for e in ALL_ENDPOINTS if e.direction is Direction.SUBSCRIBE}
    orphans = subscribed - published - EXTERNAL_INPUTS
    assert not orphans, f"subscribed but nothing publishes: {sorted(orphans)}"

    types = {}
    for endpoint in ALL_ENDPOINTS:
        types.setdefault(endpoint.topic, set()).add(endpoint.msg_type)
    clashes = {topic: values for topic, values in types.items() if len(values) > 1}
    assert not clashes, f"topic type mismatch: {clashes}"


def test_describe_lists_every_endpoint_owner():
    text = describe()

    for owner in {e.owner for e in ALL_ENDPOINTS}:
        assert f"[{owner}]" in text
