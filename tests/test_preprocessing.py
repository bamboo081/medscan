import numpy as np
import pytest
from PIL import Image
from medscan.preprocessing import intensity_scale, image_tensor


def test_ct_hu_clips_and_maps():
    raw = np.array([[-1000, 0, 1000, 2000]], dtype=np.float32)
    out = intensity_scale(raw, "ct_hu", (-1000, 1000))
    np.testing.assert_allclose(out, [[0, 0.5, 1, 1]])


def test_mri_background_preserved():
    raw = np.array([[0, 2, 4, 6], [0, 0, 9, 12]], dtype=np.float32)
    out = intensity_scale(raw, "mri_slice")
    assert np.all(out[raw == 0] == 0)
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_constant_finite():
    image = np.ones((8,8), dtype=np.float32)
    out = intensity_scale(image, "xray_exported")
    assert np.all(out == 0)


def test_preprocessing_requires_true_hu_arrays(tmp_path):
    src = tmp_path / "exported.png"
    Image.fromarray(np.uint8(np.full((64,64), 128))).save(src)
    with pytest.raises(ValueError, match="HU"):
        image_tensor(src, mode="ct_hu", hu_window=(-100, 200))
    x = image_tensor(src, mode="xray_exported", size=64)
    assert x.shape == (3,64,64) and x.isfinite().all()


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        intensity_scale(np.array([[float("nan")]]), "mri_slice")
