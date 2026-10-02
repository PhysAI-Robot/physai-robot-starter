"""Scene configs serialize into dataset metadata without machine-specific paths."""

import json

from physai.contracts import CAMERA_SIZE
from physai.sim.mujoco import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig
from physai.sim.mujoco.scenes.common import REPO_ROOT, build_manipulation_spec


def test_scene_metadata_is_portable_json(tmp_path):
    xml = REPO_ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"
    for scene_type in (PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig):
        metadata = scene_type(robot_xml=xml).to_metadata()
        assert metadata["robot_xml"] == "assets/so101/so101_new_calib_camera.xml"

    scene = PickPlaceMinimalSceneConfig(robot_xml=REPO_ROOT / "assets" / "x.xml")
    metadata = json.loads(json.dumps(scene.to_metadata()))  # must serialize
    assert (metadata["camera_width"], metadata["camera_height"]) == CAMERA_SIZE
    assert metadata["table_size"] == list(scene.table_size)

    outside = tmp_path / "robot.xml"  # a path outside the repository stays as given
    metadata = PickPlaceMinimalSceneConfig(robot_xml=outside).to_metadata()
    assert metadata["robot_xml"] == outside.as_posix()
    assert PickPlaceMinimalSceneConfig().to_metadata()["robot_xml"] is None
    assert build_manipulation_spec.__doc__  # its docstring once sat after code


def test_camera_resolution_is_one_of_three_presets_defaulting_to_the_smallest():
    import pytest

    from physai.contracts import (
        CAMERA_RESOLUTIONS,
        CAMERA_SIZE,
        DEFAULT_CAMERA_RESOLUTION,
        parse_camera_resolution,
    )
    from physai.robots.description import load_robot_description
    from physai.robots.so101.isaac_env import _DESCRIPTION_PATH

    assert CAMERA_RESOLUTIONS == ("320x240", "640x480", "1280x720")
    assert DEFAULT_CAMERA_RESOLUTION == "320x240"
    scene = PickPlaceMinimalSceneConfig()
    assert (scene.camera_width, scene.camera_height) == CAMERA_SIZE
    for name, size in zip(CAMERA_RESOLUTIONS, ((320, 240), (640, 480), (1280, 720))):
        chosen = PickPlaceMinimalSceneConfig(camera_resolution=name)
        assert (chosen.camera_width, chosen.camera_height) == size
        assert parse_camera_resolution(name) == size
    # Only the presets: no free-form size, and the pixel size is derived.
    with pytest.raises(ValueError, match="unsupported camera resolution"):
        PickPlaceMinimalSceneConfig(camera_resolution="64x64")
    with pytest.raises(TypeError):
        PickPlaceMinimalSceneConfig(camera_width=64)
    # The description's camera entries are data, so they must agree with the default.
    for camera in load_robot_description(_DESCRIPTION_PATH).cameras:
        assert (camera.width, camera.height) == CAMERA_SIZE
