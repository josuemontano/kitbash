"""Cheap reference screening. These signals cannot establish identity or detect occlusion."""

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from kitbash.domain.inventory import InventoryItem
from kitbash.infra.image_search import Candidate
from kitbash.infra.imaging import open_rgb


def rights_allowed(candidate: Candidate, allowed: Sequence[str]) -> bool:
    if candidate.source in {"user", "input_crop"}:
        return candidate.license_id == "user-provided"
    if candidate.license_id not in allowed or not candidate.provider_id or not candidate.page_url:
        return False
    # Attribution is required for BY licenses; unknown authors are not silently treated as public domain.
    if candidate.license_id.startswith("cc-by") and not candidate.creator.strip():
        return False
    try:
        url = urlsplit(candidate.license_url)
        page = urlsplit(candidate.page_url)
        if url.username is not None or url.password is not None or url.port not in (None, 80, 443):
            return False
        if page.scheme not in {"http", "https"} or not page.hostname:
            return False
    except ValueError:
        return False
    if url.scheme not in {"http", "https"} or url.hostname not in {"creativecommons.org", "www.creativecommons.org"}:
        return False
    path = url.path.rstrip("/").lower()
    if candidate.license_id == "public-domain":
        return path == "/publicdomain/mark/1.0"
    if candidate.license_id == "cc0-1.0":
        return path == "/publicdomain/zero/1.0"
    match = re.fullmatch(r"cc-(by(?:-sa)?)-(\d\.\d)", candidate.license_id)
    return bool(match and path == f"/licenses/{match[1]}/{match[2]}")


def token_overlap(candidate: Candidate, item: InventoryItem) -> float:
    wanted = _tokens(item.search_name or item.name)
    found = _tokens(" ".join((candidate.title, *candidate.categories)))
    name_overlap = len(wanted & found) / max(1, len(wanted))
    category = _tokens(item.category)
    category_overlap = len(category & found) / max(1, len(category))
    return min(1.0, name_overlap + 0.1 * category_overlap)


def _tokens(text: str) -> set[str]:
    return {t.removesuffix("s") for t in re.findall(r"[^\W_]+", text.lower()) if len(t) > 2}


def assess(candidate: Candidate, item: InventoryItem) -> dict[str, float | bool]:
    """Measure at bounded resolution; alpha or border color only approximates foreground."""
    assert candidate.path is not None
    with Image.open(candidate.path) as original:
        rgba = original.convert("RGBA")
        rgba.thumbnail((256, 256))
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255
    rgb_image = open_rgb(candidate.path)
    rgb_image.thumbnail((256, 256))
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255
    gray = rgb.mean(axis=2)
    laplacian = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    sharpness = min(1.0, float(np.var(laplacian)) / 0.012) if laplacian.size else 0.0
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background_color = np.median(border, axis=0)
    if np.any(alpha < 0.95):
        foreground = alpha > 0.1
        background = float(np.mean(np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1])) < 0.1))
    else:
        foreground = np.max(np.abs(rgb - background_color), axis=2) > 0.12
        background = float(np.mean(np.max(np.abs(border - background_color), axis=1) <= 0.12))
    occupancy = float(np.mean(foreground))
    clipping = float(np.mean(np.concatenate((foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))))
    native_side = min(candidate.original_width or candidate.width, candidate.original_height or candidate.height)
    decoded_side = min(candidate.width, candidate.height)
    detail = min(1.0, min(native_side, decoded_side) / 512)
    overlap = 1.0 if candidate.source == "input_crop" else token_overlap(candidate, item)
    occupancy_score = min(1.0, occupancy / 0.15, max(0.0, (0.9 - occupancy) / 0.15))
    score = (
        0.35 * overlap + 0.15 * detail + 0.15 * sharpness + 0.15 * background
        + 0.1 * occupancy_score + 0.05 * (1 - clipping) + 0.05 / (1 + candidate.provider_rank)
    )
    # Hard caps prevent title/rank from compensating for weak physical evidence.
    eligible = overlap >= 0.65 and min(native_side, decoded_side) >= 384 and sharpness >= 0.2 and 0.08 <= occupancy <= 0.75 and clipping <= 0.02 and background >= 0.9
    if not eligible:
        score = min(score, 0.79)
    return {
        "score": round(score, 6), "token_overlap": round(overlap, 4), "sharpness": round(sharpness, 4),
        "background": round(background, 4), "occupancy": round(occupancy, 4), "clipping": round(clipping, 4),
        "detail": round(detail, 4), "automatic_eligible": eligible,
    }
