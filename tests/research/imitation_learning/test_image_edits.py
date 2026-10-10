import numpy as np
import pytest

from research.imitation_learning import image_edits as ie

BACKGROUND = (200, 210, 205)
TABLE = (220, 215, 200)
ARM = (150, 130, 20)
CUBE = (90, 28, 23)


def _scene():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    table = np.zeros((40, 60), dtype=bool)
    table[20:, :] = True
    image[table] = TABLE
    image[5:10, 5:10] = ARM  # arm over the background
    image[25:30, 20:25] = CUBE  # cube on the table
    return image, table


def _changed(before, after):
    return (before != after).any(axis=-1)


@pytest.mark.parametrize(
    ("name", "region"),
    [
        ("bg_dim", "background"),
        ("bg_gray", "background"),
        ("table_dim", "table"),
        ("cube_gain", "cube"),
        ("arm_tint", "arm"),
    ],
)
def test_an_edit_changes_only_its_region(name, region):
    image, table = _scene()
    after = ie.edit_front(name, image, table)
    regions = ie.front_regions(image, table)
    changed = _changed(image, after)
    assert changed[regions[region]].mean() > 0.9
    assert not changed[~regions[region]].any()


def test_bg_flat_removes_background_texture_and_leaves_the_rest():
    image, table = _scene()
    image[0, :5] = (100, 100, 100)  # a darker background tile
    after = ie.edit_front("bg_flat", image, table)
    background = ie.front_regions(image, table)["background"]
    assert np.ptp(after[background], axis=0).max() <= 1
    assert (after[~background] == image[~background]).all()


def test_blank_camera_edits_replace_the_whole_frame_and_other_edits_pass_through():
    image, table = _scene()
    assert (ie.edit_front("blank_front", image, table) == ie.GRAY).all()
    assert (ie.edit_wrist("blank_wrist", image) == ie.GRAY).all()
    assert (ie.edit_wrist("blank_front", image) == image).all()


def test_blur_softens_only_the_cameras_it_names():
    image, table = _scene()
    image[::2, ::2] = (255, 255, 255)  # high-frequency detail to remove
    front = ie.edit_front("blur_front", image, table)
    wrist = ie.edit_wrist("blur_wrist", image)

    def detail(img):
        return np.abs(np.diff(img.astype(int), axis=1)).mean()

    assert detail(front) < detail(image) and detail(wrist) < detail(image)
    assert (ie.edit_front("blur_wrist", image, table) == image).all()
    assert (ie.edit_wrist("blur_front", image) == image).all()


def test_an_unknown_edit_is_refused():
    with pytest.raises(ValueError, match="unknown image_edit"):
        ie.ImageEditor("nope", (0.3, 0.0, 0.01), (0.2, 0.25, 0.01))


def test_first_frame_swap_changes_only_its_region():
    mujoco, table = _scene()
    isaac = np.full_like(mujoco, 77)
    regions = ie.front_regions(mujoco, table)
    for name, region in (
        ("first_table", "table"),
        ("first_background", "background"),
        ("first_cubedisc", "cube"),
    ):
        after = ie.swap_front(name, mujoco, isaac, table)
        mask = regions[region] | (regions["disc"] if region == "cube" else False)
        assert (after[mask] == 77).all() and (after[~mask] == mujoco[~mask]).all()
    assert (ie.swap_front("first_all", mujoco, isaac, table) == 77).all()
    assert (ie.swap_front("first_wrist", mujoco, isaac, table) == mujoco).all()


def _capture(root, engine, seeds, value):
    import json

    (root / engine).mkdir(parents=True)
    episodes = []
    for seed in seeds:
        state = np.zeros((2, 13))
        state[:, 6:9] = (0.2, 0.0, 0.036)
        np.savez(
            root / engine / f"e{seed}.npz",
            **{
                "observation.images.front": np.full((2, 4, 4, 3), value, np.uint8),
                "observation.images.wrist": np.full((2, 4, 4, 3), value, np.uint8),
                "observation.environment_state": state,
            },
        )
        episodes.append({"file": f"e{seed}.npz", "seed": seed})
    (root / engine / "meta.json").write_text(json.dumps({"episodes": episodes}))


class _Frame:
    def __init__(self, value):
        self.data = np.full((4, 4, 3), value, np.uint8)


def test_first_frame_swap_swaps_each_episodes_first_read_only(tmp_path):
    _capture(tmp_path, "mujoco", (5, 6), 10)
    _capture(tmp_path, "isaac", (5, 6), 99)
    editor = ie.FirstFrameSwap(
        "first_wrist", (0.3, 0, 0.01), (0.2, 0.25, 0.01), tmp_path, 5
    )
    for _ in range(2):  # two episodes, seeds 5 and 6
        editor.reset()
        assert (editor("wrist", _Frame(10)) == 99).all()
        assert (editor("front", _Frame(10)) == 10).all()  # not the swapped camera
        assert (editor("wrist", _Frame(10)) == 10).all()  # later reads pass through
    editor.reset()
    with pytest.raises(KeyError):
        editor("wrist", _Frame(10))


def test_first_frame_swap_refuses_a_mismatched_seed_range(tmp_path):
    _capture(tmp_path, "mujoco", (5,), 10)
    _capture(tmp_path, "isaac", (5,), 99)
    editor = ie.FirstFrameSwap(
        "first_all", (0.3, 0, 0.01), (0.2, 0.25, 0.01), tmp_path, 5
    )
    editor.reset()
    with pytest.raises(ValueError, match="not the captured MuJoCo frame"):
        editor("wrist", _Frame(200))
