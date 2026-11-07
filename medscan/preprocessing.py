"""Modality-aware intensity transforms. All image paths are local."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

MODES = ("mri_slice", "xray_exported", "ct_exported", "ct_hu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_grayscale(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        img = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        with Image.open(path) as im:
            img = np.asarray(im.convert("F"))
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Convert raw DICOM first.")
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2 or img.size == 0 or not np.isfinite(img).all():
        raise ValueError(f"Expected a finite, nonempty 2D grayscale image: {path}")
    return img


def intensity_scale(image: np.ndarray, mode: str, hu_window: tuple[float, float] | None = None) -> np.ndarray:
    """Map a 2D image to [0,1] WITHOUT reading statistics from val/test.

    For exported scans: robust within-image percentiles, not CT Hounsfield units.
    For MRI slices: foreground-only percentiles; preserves exact zero background.
    For CT HU: use fixed task-appropriate HU window, only valid on original HU data.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Image must be a finite 2D array")
    if mode not in MODES:
        raise ValueError(f"Unknown normalization mode {mode}")
    if mode == "ct_hu":
        if hu_window is None or len(hu_window) != 2 or hu_window[0] >= hu_window[1]:
            raise ValueError("ct_hu requires a valid task-specific HU window")
        low, high = hu_window
        return np.clip((x - low) / (high - low), 0, 1).astype(np.float32)

    foreground = x[x != 0] if mode == "mri_slice" else x.reshape(-1)
    if foreground.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    low, high = np.percentile(foreground, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - low) / (high - low), 0, 1)
    if mode == "mri_slice":
        out[x == 0] = 0
    return out.astype(np.float32)


def prepare_image(path: str | Path, *, mode: str, size: int = 224,
                  hu_window: tuple[float, float] | None = None) -> torch.Tensor:
    """Deterministic single-channel scaling and resizing, before augmentation."""
    if size < 64:
        raise ValueError("Image size must be at least 64")
    if mode == "ct_hu" and Path(path).suffix.lower() != ".npy":
        raise ValueError("ct_hu requires a .npy array containing actual HU, not a rendered JPEG/PNG")
    img = intensity_scale(load_grayscale(path), mode, hu_window)
    ten = torch.from_numpy(img).unsqueeze(0)  # 1,H,W
    ten = TF.resize(ten, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return ten


def finish_image(ten: torch.Tensor, *, augment=False, mean=None, std=None) -> torch.Tensor:
    """Apply fresh augmentation, channel replication and checkpoint normalization."""
    # Conservative flips only on an explicit, task-validated request. No defaults:
    # medical image left/right and localization matter and some scans carry markers.
    if augment:
        # Mild brightness/contrast only; no geometric flips, crops, or rotations.
        ten = TF.adjust_brightness(ten, float(torch.empty(1).uniform_(0.95, 1.05)))
        ten = TF.adjust_contrast(ten, float(torch.empty(1).uniform_(0.95, 1.05)))
        ten = ten.clamp(0, 1)
    ten = ten.repeat(3, 1, 1)
    return TF.normalize(ten, mean or IMAGENET_MEAN, std or IMAGENET_STD)


def image_tensor(path: str | Path, *, mode: str, size: int = 224,
                 hu_window: tuple[float, float] | None = None, augment: bool = False,
                 mean: list[float] | None = None, std: list[float] | None = None) -> torch.Tensor:
    return finish_image(prepare_image(path, mode=mode, size=size, hu_window=hu_window),
                        augment=augment, mean=mean, std=std)
