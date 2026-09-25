"""Scene configs serialize into dataset metadata without machine-specific paths."""

import json

from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig
from physai.sim.scenes.common import REPO_ROOT, build_manipulation_spec


def test_scene_metadata_is_portable_json(tmp_path):
    xml = REPO_ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"
    for scene_type in (PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig):
        metadata = scene_type(robot_xml=xml).to_metadata()
        assert metadata["robot_xml"] == "assets/so101/so101_new_calib_camera.xml"

    scene = PickPlaceMinimalSceneConfig(
        robot_xml=REPO_ROOT / "assets" / "x.xml", camera_width=224
    )
    metadata = json.loads(json.dumps(scene.to_metadata()))  # must serialize
    assert metadata["camera_width"] == 224
    assert metadata["table_size"] == list(scene.table_size)

    outside = tmp_path / "robot.xml"  # a path outside the repository stays as given
    metadata = PickPlaceMinimalSceneConfig(robot_xml=outside).to_metadata()
    assert metadata["robot_xml"] == outside.as_posix()
    assert PickPlaceMinimalSceneConfig().to_metadata()["robot_xml"] is None
    assert build_manipulation_spec.__doc__  # its docstring once sat after code
