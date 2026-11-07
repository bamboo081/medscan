"""On-demand lossless float32 NIH preprocessing cache; no labels or augmentation.

Version the key whenever deterministic preprocessing changes. File identity uses
resolved path, size and nanosecond mtime; original scans must remain immutable
during a run. Only accessed images are cached (training and validation, not test).
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .preprocessing import prepare_image

CACHE_VERSION = 'xray-percentile-resize-float32-v1'


def cached_image(path, *, size, cache_dir):
    source = Path(path).resolve()
    stat = source.stat()
    identity = [CACHE_VERSION, str(source), stat.st_size, stat.st_mtime_ns, size]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    destination = Path(cache_dir) / key[:2] / (key + '.npy')
    if destination.is_file():
        image = np.load(destination, allow_pickle=False)
        if image.shape != (1, size, size) or image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError(f'Invalid image cache entry: {destination}')
        return image
    image = prepare_image(source, mode='xray_exported', size=size).numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            np.save(f, image, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return image
