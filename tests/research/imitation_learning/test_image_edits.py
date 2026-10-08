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


def test_an_unknown_edit_is_refused():
    with pytest.raises(ValueError, match="unknown image_edit"):
        ie.ImageEditor("nope", (0.3, 0.0, 0.01), (0.2, 0.25, 0.01))
