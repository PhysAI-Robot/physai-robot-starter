def test_scripted_planner_is_discoverable():
    from physai.planner import available_planners, create_planner

    assert "scripted_planner" in available_planners()
    planner = create_planner(
        "scripted_planner", pick_xyz=(0.2, 0.08, 0.036), place_xyz=(0.2, -0.1, 0.021)
    )
    assert planner.name == "scripted_planner"


def test_sorting_planner_registers_itself_when_its_research_module_is_imported():
    import research.vlm_planners.sorting_planner  # noqa: F401
    from physai.planner import available_planners

    assert "sorting_planner" in available_planners()


def test_unknown_planner_lists_available_planners():
    import pytest

    from physai.planner import create_planner

    with pytest.raises(ValueError, match="scripted_planner"):
        create_planner("does-not-exist")
