"""Image utilities: comparison metrics, contact sheets and crops."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from attrs import frozen
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFont, ImageOps

BACKGROUND = (128, 128, 128)


@frozen
class ImageComparison:
    ssim: float
    color_delta: float  # mean absolute RGB difference in [0, 1]

    @property
    def score(self) -> float:
        """Round-trip score in [0, 1]: structural similarity, capped by the global color error."""
        return float(np.clip(min(self.ssim, 1.0 - 2.0 * self.color_delta), 0.0, 1.0))

    def to_dict(self) -> dict[str, float]:
        return {"ssim": round(self.ssim, 4), "color_delta": round(self.color_delta, 4), "score": round(self.score, 4)}


def open_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            flat = Image.new("RGB", rgba.size, BACKGROUND)
            flat.paste(rgba, mask=rgba.getchannel("A"))
            image = flat
        else:
            image = image.convert("RGB")
        if size is not None:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image.copy()


def dominant_color(path: Path, fallback: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> tuple[float, float, float]:
    """Median color of the object in a reference, as linear RGB (0..1) for a Principled base color.

    The object is its alpha channel, or else everything that differs from the border color."""
    with Image.open(path) as original:
        rgba = ImageOps.exif_transpose(original).convert("RGBA")
    rgba.thumbnail((256, 256))
    pixels = np.asarray(rgba, dtype=np.float32) / 255
    rgb, alpha = pixels[..., :3], pixels[..., 3]
    if np.any(alpha < 0.95):
        mask = alpha > 0.5
    else:
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
        mask = np.max(np.abs(rgb - np.median(border, axis=0)), axis=2) > 0.12
    if mask.sum() < 16:
        return fallback
    srgb = np.array([np.median(rgb[..., i][mask]) for i in range(3)])
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return (round(float(linear[0]), 4), round(float(linear[1]), 4), round(float(linear[2]), 4))


def compare_images(first: Path, second: Path, size: tuple[int, int] = (256, 256)) -> ImageComparison:
    a = np.asarray(open_rgb(first, size), dtype=np.float64) / 255.0
    b = np.asarray(open_rgb(second, size), dtype=np.float64) / 255.0
    return ImageComparison(ssim=ssim(_luma(a), _luma(b)), color_delta=float(np.mean(np.abs(a - b))))


def ssim(x: np.ndarray, y: np.ndarray, *, sigma: float = 1.5, radius: int = 5) -> float:
    """Mean structural similarity of two grayscale images in [0, 1] (Wang et al. 2004)."""
    c1, c2 = 0.01**2, 0.03**2
    kernel = _gaussian_kernel(sigma, radius)
    mu_x, mu_y = _blur(x, kernel), _blur(y, kernel)
    sigma_x = _blur(x * x, kernel) - mu_x**2
    sigma_y = _blur(y * y, kernel) - mu_y**2
    sigma_xy = _blur(x * y, kernel) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.clip(np.mean(numerator / denominator), 0.0, 1.0))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def _gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(image, pad, mode="reflect")
    rows = sliding_window_view(padded, len(kernel), axis=1) @ kernel
    return sliding_window_view(rows, len(kernel), axis=0) @ kernel


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def contact_sheet(paths: Sequence[Path], labels: Sequence[str], out: Path, *, cell: int = 320, columns: int = 4) -> Path:
    """Grid of thumbnails with a numbered label on each; used to let a model pick a candidate."""
    rows = max(1, -(-len(paths) // columns))
    sheet = Image.new("RGB", (columns * cell, rows * (cell + 28)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        thumb = open_rgb(path)
        thumb.thumbnail((cell - 8, cell - 8))
        x, y = (index % columns) * cell, (index // columns) * (cell + 28)
        sheet.paste(thumb, (x + (cell - thumb.width) // 2, y + 28 + (cell - thumb.height) // 2))
        draw.rectangle((x, y, x + cell - 1, y + cell + 27), outline=(200, 200, 200))
        draw.text((x + 6, y + 4), label, fill=(200, 0, 0), font=_font(18))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def side_by_side(paths: Sequence[Path], labels: Sequence[str], out: Path, *, height: int = 512) -> Path:
    images = [open_rgb(p) for p in paths]
    scaled = [im.resize((max(1, round(im.width * height / im.height)), height)) for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in scaled), height + 28), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, label in zip(scaled, labels, strict=True):
        canvas.paste(image, (x, 28))
        draw.text((x + 6, 4), label, fill=(0, 0, 0), font=_font(18))
        x += image.width
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def crop_normalized(path: Path, bbox: Sequence[float], out: Path, *, pad: float = 0.08, min_side: int = 0) -> Path:
    """Crop a normalized ``[x0, y0, x1, y1]`` box (top-left origin) with padding, saved as PNG.
    Crops whose shorter side is below ``min_side`` are upscaled to it."""
    image = open_rgb(path)
    x0, y0, x1, y1 = bbox
    w, h = image.size
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    box = (
        max(0, round((x0 - px) * w)),
        max(0, round((y0 - py) * h)),
        min(w, round((x1 + px) * w)),
        min(h, round((y1 + py) * h)),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise ValueError(f"Crop box {bbox} is too small for {path.name}")
    crop = image.crop(box)
    if min_side and min(crop.size) < min_side:
        factor = min_side / min(crop.size)
        crop = crop.resize((round(crop.width * factor), round(crop.height * factor)), Image.Resampling.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)
    return out


def normalize_to_png(source: Path, out: Path, *, max_side: int = 2048) -> Path:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")
        image.thumbnail((max_side, max_side))
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, format="PNG")
    return out


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
