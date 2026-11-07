import numpy as np
import pandas as pd
import torch
from PIL import Image

from train_nih import NIHData


def test_cache_matches_uncached_and_keeps_augmentation_fresh(tmp_path, monkeypatch):
    path = tmp_path / 'image.png'
    Image.fromarray(np.random.default_rng(42).integers(0, 256, (80, 90), dtype=np.uint8)).save(path)
    df = pd.DataFrame({'path': [str(path)], 'labels': ['Pneumonia']})
    plain = NIHData(df, 64)
    cached = NIHData(df, 64, cache_dir=tmp_path / 'cache')
    expected, y = plain[0]
    actual, cy = cached[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cy, y)
    assert len(list((tmp_path / 'cache').rglob('*.npy'))) == 1
    augmented = NIHData(df, 64, augment=True, cache_dir=tmp_path / 'cache')
    torch.manual_seed(42)
    cached_augmented = augmented[0][0]
    torch.manual_seed(42)
    plain_augmented = NIHData(df, 64, augment=True)[0][0]
    torch.testing.assert_close(cached_augmented, plain_augmented, rtol=0, atol=0)
    import medscan.image_cache as module
    def fail(*args, **kwargs):
        raise AssertionError('Cache hit must not decode the source')
    monkeypatch.setattr(module, 'prepare_image', fail)
    torch.testing.assert_close(cached[0][0], expected, rtol=0, atol=0)
    aug = NIHData(df, 64, augment=True, cache_dir=tmp_path / 'cache')
    torch.manual_seed(1)
    a = aug[0][0]
    torch.manual_seed(2)
    b = aug[0][0]
    assert not torch.equal(a, b)


def test_cache_invalidates_changed_source_and_size(tmp_path):
    from medscan.image_cache import cached_image
    path = tmp_path / 'image.npy'
    np.save(path, np.arange(80*80, dtype=np.float32).reshape(80, 80))
    cache = tmp_path / 'cache'
    before = cached_image(path, size=64, cache_dir=cache).copy()
    np.save(path, np.zeros((80, 80), dtype=np.float32))
    after = cached_image(path, size=64, cache_dir=cache)
    assert not np.array_equal(before, after)
    assert cached_image(path, size=96, cache_dir=cache).shape == (1, 96, 96)
    assert len(list(cache.rglob('*.npy'))) == 3


def test_bad_cache_entry_fails_explicitly(tmp_path):
    import pytest
    from medscan.image_cache import cached_image
    path = tmp_path / 'source.npy'
    np.save(path, np.arange(64*64, dtype=np.float32).reshape(64, 64))
    cache = tmp_path / 'cache'
    cached_image(path, size=64, cache_dir=cache)
    entry = next(cache.rglob('*.npy'))
    np.save(entry, np.zeros((1, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match='Invalid image cache entry'):
        cached_image(path, size=64, cache_dir=cache)
