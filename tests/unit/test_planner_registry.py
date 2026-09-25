import pytest


def test_planners_are_discovered_and_research_ones_register_on_import():
    import research.vlm_planners.sorting_planner  # noqa: F401
    from physai.planner import available_planners, create_planner

    assert "scripted_planner" in available_planners()
    planner = create_planner(
        "scripted_planner", pick_xyz=(0.2, 0.08, 0.036), place_xyz=(0.2, -0.1, 0.021)
    )
    assert planner.name == "scripted_planner"
    assert "sorting_planner" in available_planners()
    with pytest.raises(ValueError, match="scripted_planner"):
        create_planner("does-not-exist")  # the error lists what is available
