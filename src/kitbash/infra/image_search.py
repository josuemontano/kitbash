"""Reference image candidates from the web (Wikimedia Commons, Openverse) or the input image itself."""

from pathlib import Path
from typing import Protocol

import httpx
from attrs import frozen
from PIL import Image, UnidentifiedImageError

from kitbash.domain.inventory import InventoryItem
from kitbash.infra.imaging import crop_normalized, normalize_to_png

USER_AGENT = "kitbash/0.1 (reference search; https://github.com/)"


@frozen
class Candidate:
    source: str
    title: str
    url: str | None = None
    path: Path | None = None  # set once the image is local
    license: str = ""


class CandidateProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def once_per_item(self) -> bool:
        """True for providers that search on their own (they get only the most specific query)."""
        ...

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]: ...


def make_http_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT})


class InputCropProvider:
    """The object's own crop from the reference image (image mode only)."""

    name = "input_crop"
    once_per_item = True

    def __init__(self, input_image: Path | None, min_side: int = 512) -> None:
        self._image = input_image
        self._min_side = min_side

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]:
        bbox = item.position.image_bbox
        if self._image is None or bbox is None:
            return []
        try:
            path = crop_normalized(self._image, bbox, workdir / "input_crop.png", min_side=self._min_side)
        except ValueError:
            return []
        return [Candidate(source=self.name, title=f"crop of {item.name} from the reference", path=path)]


class WikimediaProvider:
    name = "wikimedia"
    once_per_item = False
    endpoint = "https://commons.wikimedia.org/w/api.php"

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]:
        params = {
            "action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
            "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024,
            "format": "json",
        }
        data = self._client.get(self.endpoint, params=params).raise_for_status().json()
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        results = []
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            license_name = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            results.append(
                Candidate(self.name, page.get("title", ""), url=info.get("thumburl") or info.get("url"), license=license_name)
            )
        return results


class OpenverseProvider:
    name = "openverse"
    once_per_item = False
    endpoint = "https://api.openverse.org/v1/images/"

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]:
        params = {"q": query, "page_size": limit, "mature": "false"}
        data = self._client.get(self.endpoint, params=params).raise_for_status().json()
        return [
            Candidate(self.name, r.get("title") or "", url=r.get("url"), license=f"{r.get('license', '')} {r.get('license_version', '')}".strip())
            for r in data.get("results", [])
            if r.get("url")
        ]


class ImageDownloader:
    def __init__(self, client: httpx.Client, *, max_bytes: int, min_side: int) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._min_side = min_side

    def fetch(self, candidate: Candidate, destination: Path) -> Candidate | None:
        """Download (or copy) a candidate to ``destination`` as PNG; None if unusable."""
        raw = destination.with_suffix(".download")
        try:
            if candidate.path is not None:
                source = candidate.path
            else:
                with self._client.stream("GET", candidate.url) as response:
                    response.raise_for_status()
                    if not response.headers.get("content-type", "image/").startswith("image/"):
                        return None
                    size = 0
                    with raw.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > self._max_bytes:
                                return None
                            handle.write(chunk)
                source = raw
            with Image.open(source) as image:
                if min(image.size) < self._min_side:
                    return None
            normalize_to_png(source, destination)
        except (httpx.HTTPError, UnidentifiedImageError, OSError):
            return None
        finally:
            raw.unlink(missing_ok=True)
        return Candidate(candidate.source, candidate.title, candidate.url, destination, candidate.license)
