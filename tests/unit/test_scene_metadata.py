"""Scene configs serialize into dataset metadata without machine-specific paths."""

import json

from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig
from physai.sim.scenes.common import REPO_ROOT


def test_repo_paths_are_stored_relative_to_the_repository():
    xml = REPO_ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"

    for scene_type in (PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig):
        metadata = scene_type(robot_xml=xml).to_metadata()

        assert metadata["robot_xml"] == "assets/so101/so101_new_calib_camera.xml"


def test_metadata_is_json_serializable_and_keeps_other_fields():
    scene = PickPlaceMinimalSceneConfig(
        robot_xml=REPO_ROOT / "assets" / "x.xml", camera_width=224
    )

    metadata = json.loads(json.dumps(scene.to_metadata()))

    assert metadata["camera_width"] == 224
    assert metadata["table_size"] == list(scene.table_size)


def test_a_path_outside_the_repository_is_kept_as_given(tmp_path):
    outside = tmp_path / "robot.xml"

    metadata = PickPlaceMinimalSceneConfig(robot_xml=outside).to_metadata()

    assert metadata["robot_xml"] == outside.as_posix()


def test_an_unset_path_stays_none():
    assert PickPlaceMinimalSceneConfig().to_metadata()["robot_xml"] is None
