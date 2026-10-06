"""Reference image candidates from the web (Wikimedia Commons, Openverse) or the input image itself."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from attrs import asdict, evolve, fields, frozen
from PIL import Image, ImageOps

from kitbash.domain.inventory import InventoryItem
from kitbash.infra.imaging import crop_normalized
from kitbash.infra.public_http import PublicHTTPTransport, validate_public_url

USER_AGENT = "kitbash/0.1 (reference search; https://github.com/)"


@frozen
class Candidate:
    source: str
    title: str
    url: str | None = None
    path: Path | None = None
    license: str = ""
    provider_id: str = ""
    page_url: str = ""
    creator: str = ""
    license_id: str = ""
    license_url: str = ""
    width: int = 0
    height: int = 0
    original_width: int = 0
    original_height: int = 0
    query: str = ""
    provider_rank: int = 0
    categories: tuple[str, ...] = ()
    mime: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path) if self.path is not None else None
        data["categories"] = list(self.categories)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Candidate:
        values = {field.name: data[field.name] for field in fields(cls) if field.name in data}
        values["path"] = Path(data["path"]) if data.get("path") else None
        values["categories"] = tuple(data.get("categories") or ())
        return cls(**values)


class CandidateProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def once_per_item(self) -> bool:
        """True for providers that search on their own (they get only the most specific query)."""
        ...

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]: ...


def make_http_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(timeout=timeout_s, follow_redirects=False, trust_env=False,
                        transport=PublicHTTPTransport(), headers={"User-Agent": USER_AGENT})


class InputCropProvider:
    """The object's own crop, requiring sufficient ORIGINAL unpadded image detail."""

    name = "input_crop"
    once_per_item = True

    def __init__(self, input_image: Path | None, min_side: int = 512) -> None:
        self._image = input_image
        self._min_side = min_side

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]:
        bbox = item.position.image_bbox
        if self._image is None or bbox is None or limit <= 0:
            return []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(self._image) as image:
                    oriented = ImageOps.exif_transpose(image)
                    width, height = oriented.size
                x0, y0, x1, y1 = bbox
                original_width = max(0, round(min(1, x1) * width) - round(max(0, x0) * width))
                original_height = max(0, round(min(1, y1) * height) - round(max(0, y0) * height))
                if min(original_width, original_height) < self._min_side:
                    return []
                # Padding can add context but must never rescue insufficient detail.
                path = crop_normalized(self._image, bbox, workdir / "input_crop.png")
                with Image.open(path) as crop:
                    crop_width, crop_height = crop.size
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return []
        return [Candidate(source=self.name, title=f"crop of {item.name} from the reference", path=path,
                          license="User provided", license_id="user-provided", width=crop_width, height=crop_height,
                          original_width=original_width, original_height=original_height, query=query,
                          provider_rank=1, mime="image/png")]


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
        for rank, page in enumerate(pages, 1):
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in _MIME_FORMATS:
                continue
            metadata = info.get("extmetadata") or {}
            license_name = _plain_text(metadata.get("LicenseShortName", {}).get("value", ""))
            license_url = metadata.get("LicenseUrl", {}).get("value", "")
            if license_url.startswith("//"):
                license_url = "https:" + license_url
            width, height = _dimension(info.get("width")), _dimension(info.get("height"))
            thumbnail = bool(info.get("thumburl"))
            results.append(Candidate(
                self.name, _plain_text(page.get("title", "")), url=info.get("thumburl") or info.get("url"),
                license=license_name, provider_id=str(page.get("pageid", "")), page_url=info.get("descriptionurl", ""),
                creator=_plain_text(metadata.get("Artist", {}).get("value", "")),
                license_id=_license_id(license_name, license_url), license_url=license_url,
                width=_dimension(info.get("thumbwidth")) if thumbnail else width,
                height=_dimension(info.get("thumbheight")) if thumbnail else height,
                original_width=width, original_height=height, query=query, provider_rank=rank,
                categories=tuple(filter(None, _plain_text(metadata.get("Categories", {}).get("value", "")).split("|"))),
                mime=info["mime"],
            ))
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
        results = []
        for rank, result in enumerate(data.get("results", []), 1):
            if not result.get("url"):
                continue
            license_name = f"{result.get('license', '')} {result.get('license_version', '')}".strip()
            license_url = result.get("license_url") or ""
            license_id = _license_id(license_name, license_url)
            if not license_url:
                license_url = _canonical_license_url(license_id)
            categories = result.get("categories") or []
            tags = result.get("tags") or []
            width, height = _dimension(result.get("width")), _dimension(result.get("height"))
            results.append(Candidate(
                self.name, _plain_text(result.get("title") or ""), url=result["url"], license=license_name,
                provider_id=str(result.get("id", "")), page_url=result.get("foreign_landing_url") or "",
                creator=_plain_text(result.get("creator") or ""), license_id=license_id, license_url=license_url,
                width=width, height=height, original_width=width, original_height=height,
                query=query, provider_rank=rank,
                categories=tuple(dict.fromkeys(_plain_text(value.get("name", "") if isinstance(value, dict) else value)
                                               for value in [*categories, *tags] if value)),
                mime=result.get("mime_type") or result.get("mime") or "",
            ))
        return results


_MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
_FORMAT_MIMES = {value: key for key, value in _MIME_FORMATS.items()}
_REDIRECTS = {301, 302, 303, 307, 308}


class _MetadataText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _MetadataText()
    parser.feed(str(value))
    return " ".join(" ".join(parser.parts).split())


def _dimension(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _license_id(name: str, url: str) -> str:
    normalized = re.sub(r"[\s_]+", "-", name.strip().lower())
    # Prefer an explicit declaration; never reinterpret a restrictive name as a
    # permissive license merely because a conflicting URL accompanies it.
    if normalized in ("public-domain", "public-domain-mark", "pdm", "pdm-1.0"):
        return "public-domain"
    if normalized in ("cc0", "cc0-1.0", "cc-zero"):
        return "cc0-1.0"
    match = re.fullmatch(r"(?:cc-)?(by(?:-nc)?(?:-sa|-nd)?)-(\d\.\d)", normalized)
    if match:
        return f"cc-{match[1]}-{match[2]}"
    if normalized:
        return ""
    parsed = urlsplit(url)
    if parsed.hostname not in ("creativecommons.org", "www.creativecommons.org"):
        return ""
    match = re.match(r"^/licenses/(by(?:-nc)?(?:-sa|-nd)?)/(\d\.\d)(?:/|$)", parsed.path)
    if match:
        return f"cc-{match[1]}-{match[2]}"
    if parsed.path.rstrip("/") == "/publicdomain/zero/1.0":
        return "cc0-1.0"
    if parsed.path.rstrip("/") == "/publicdomain/mark/1.0":
        return "public-domain"
    return ""


def _canonical_license_url(license_id: str) -> str:
    if license_id == "cc0-1.0":
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if license_id == "public-domain":
        return "https://creativecommons.org/publicdomain/mark/1.0/"
    match = re.fullmatch(r"cc-(by(?:-nc)?(?:-sa|-nd)?)-(\d\.\d)", license_id)
    return f"https://creativecommons.org/licenses/{match[1]}/{match[2]}/" if match else ""


class ImageDownloader:
    """Bounded image decoding and content-addressed PNG reuse.

    Network clients must come from make_http_client. An explicitly injected
    transport (for example MockTransport) is a trusted dependency boundary.
    Cache entries never replace the caller's current rights/provenance metadata.
    """

    def __init__(self, client: httpx.Client, *, max_bytes: int, min_side: int,
                 cache_dir: Path | None = None, max_side: int = 16384, max_pixels: int = 40_000_000) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._min_side = min_side
        self._cache_dir = cache_dir
        self._max_side = max_side
        self._max_pixels = max_pixels

    def fetch(self, candidate: Candidate, destination: Path) -> Candidate | None:
        """Download (or copy) a candidate as PNG; invalid candidates return None."""
        raw: Path | None = None
        try:
            if candidate.path is None:
                if not candidate.url:
                    return None
                validate_public_url(candidate.url)
                cached = self._cached(candidate, destination)
                if cached is not None:
                    return cached
                raw = _temporary_path(destination)
                mime = self._download(candidate.url, raw)
                source = raw
            else:
                source = candidate.path
                mime = ""
            source_bytes = source.stat().st_size
            if not 0 < source_bytes <= self._max_bytes:
                return None
            with self._decode(source, _MIME_FORMATS.get(mime)) as image:
                content_hash = _pixel_hash(image)
                self._save_png(image, destination)
                result = self._result(candidate, destination, image.size, content_hash, mime)
                if candidate.path is None and candidate.url:
                    self._store(candidate.url, result, source_bytes, image.size)
                return result
        except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError, SyntaxError, EOFError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            return None
        finally:
            if raw is not None:
                raw.unlink(missing_ok=True)

    def _download(self, url: str, raw: Path) -> str:
        current = validate_public_url(url)
        for hop in range(6):
            # Never rely on the client's redirect setting, including injected clients.
            with self._client.stream("GET", current, follow_redirects=False,
                                     headers={"Accept-Encoding": "identity"}) as response:
                if response.status_code in _REDIRECTS:
                    if hop == 5 or not response.headers.get("location"):
                        raise ValueError("Too many redirects or missing redirect destination")
                    location = response.headers["location"]
                    if "\\" in location or any(ord(char) <= 32 or ord(char) == 127 for char in location):
                        raise ValueError("Invalid redirect destination")
                    current = validate_public_url(current.join(location))
                    continue
                response.raise_for_status()
                mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if mime not in _MIME_FORMATS:
                    raise ValueError("Unsupported image MIME type")
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ValueError("Encoded HTTP bodies are not supported")
                length = response.headers.get("content-length")
                if length is not None and not 0 < int(length) <= self._max_bytes:
                    raise ValueError("Image body exceeds byte limit")
                size = 0
                with raw.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        size += len(chunk)
                        if size > self._max_bytes:
                            raise ValueError("Image body exceeds byte limit")
                        handle.write(chunk)
                return mime
        raise ValueError("Too many redirects")

    def _check_dimensions(self, size: tuple[int, int]) -> None:
        width, height = size
        if min(size) < self._min_side or min(size) <= 0 or max(size) > self._max_side or width * height > self._max_pixels:
            raise ValueError("Image dimensions outside configured limits")

    def _decode(self, source: Path, expected_format: str | None = None) -> Image.Image:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.format not in _FORMAT_MIMES or (expected_format is not None and image.format != expected_format):
                    raise ValueError("Unsupported or mismatched image format")
                self._check_dimensions(image.size)
                image.verify()
            with Image.open(source) as image:
                image.load()
                with ImageOps.exif_transpose(image) as oriented:
                    # Canonical RGBA makes hashes independent of source encoding,
                    # palettes, EXIF orientation, and RGB versus opaque RGBA mode.
                    return oriented.convert("RGBA")

    def _save_png(self, image: Image.Image, destination: Path) -> None:
        temporary = _temporary_path(destination)
        try:
            image.save(temporary, format="PNG")
            if temporary.stat().st_size > self._max_bytes:
                raise ValueError("Normalized PNG exceeds byte limit")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _result(self, candidate: Candidate, destination: Path, size: tuple[int, int], content_hash: str,
                mime: str) -> Candidate:
        return evolve(candidate, path=destination, width=size[0], height=size[1], content_hash=content_hash,
                      original_width=candidate.original_width or size[0], original_height=candidate.original_height or size[1],
                      mime=candidate.mime or mime or "image/png")

    def _index_path(self, url: str) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / "urls" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def _cached(self, candidate: Candidate, destination: Path) -> Candidate | None:
        if self._cache_dir is None or candidate.url is None:
            return None
        index = self._index_path(candidate.url)
        try:
            if index.stat().st_size > 65536:
                raise ValueError("Oversized cache index")
            record = json.loads(index.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("version") != 1 or record.get("url") != candidate.url:
                raise ValueError("Invalid cache index")
            content_hash = record["content_hash"]
            if not isinstance(content_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", content_hash):
                raise ValueError("Invalid content hash")
            if not 0 < int(record["source_bytes"]) <= self._max_bytes or record["mime"] not in _MIME_FORMATS:
                raise ValueError("Cached source exceeds current limits")
            size = (int(record["source_width"]), int(record["source_height"]))
            self._check_dimensions(size)
            source = self._cache_dir / "content" / f"{content_hash}.png"
            if not 0 < source.stat().st_size <= self._max_bytes:
                raise ValueError("Cached PNG exceeds current limits")
            with self._decode(source, "PNG") as image:
                if image.size != size or _pixel_hash(image) != content_hash:
                    raise ValueError("Cached pixels do not match the index")
                self._save_png(image, destination)
                return self._result(candidate, destination, image.size, content_hash, record["mime"])
        except (KeyError, TypeError, ValueError, OSError, SyntaxError, EOFError, OverflowError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            # A damaged entry is a miss, not authority to accept bytes or rights.
            return None

    def _store(self, url: str, candidate: Candidate, source_bytes: int, size: tuple[int, int]) -> None:
        if self._cache_dir is None or candidate.path is None:
            return
        content = self._cache_dir / "content" / f"{candidate.content_hash}.png"
        temporary: Path | None = None
        try:
            temporary = _temporary_path(content)
            shutil.copyfile(candidate.path, temporary)
            os.replace(temporary, content)
            record = {"version": 1, "url": url, "content_hash": candidate.content_hash, "source_bytes": source_bytes,
                      "source_width": size[0], "source_height": size[1], "mime": candidate.mime,
                      "candidate": candidate.to_dict()}
            index = self._index_path(url)
            temporary = _temporary_path(index)
            temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, index)
        except OSError:
            # An unavailable cache must not discard an otherwise valid download.
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(name)


def _pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256(f"RGBA:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()
