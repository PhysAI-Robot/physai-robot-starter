from physai.bridge import ALL_ENDPOINTS, EXTERNAL_INPUTS, Direction


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
