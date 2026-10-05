import io
import json

import httpx
import pytest
from attrs import evolve
from PIL import Image

from kitbash.domain.inventory import Dimensions, InventoryItem, Placement
from kitbash.infra.image_search import Candidate, ImageDownloader, InputCropProvider, OpenverseProvider, WikimediaProvider


@pytest.fixture
def item() -> InventoryItem:
    return InventoryItem("lamp", "Desk lamp", "Brass desk lamp", "lighting", Dimensions(1, 1, 1),
                         Placement(image_bbox=(0.1, 0.1, 0.9, 0.9)))


def image_bytes(size=(64, 48), *, format="PNG", color=(30, 80, 120), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=format)
    return buffer.getvalue()


def downloader(handler, *, cache_dir=None, **limits) -> ImageDownloader:
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return ImageDownloader(client, max_bytes=limits.pop("max_bytes", 1_000_000),
                           min_side=limits.pop("min_side", 16), cache_dir=cache_dir, **limits)


def test_provenance_roundtrip_is_json_serializable(tmp_path):
    candidate = Candidate("wikimedia", "Brass lamp", "https://images.example/lamp.jpg", tmp_path / "lamp.png", "CC BY 4.0",
                          provider_id="42", page_url="https://commons.wikimedia.org/wiki/File:Lamp.jpg", creator="A. Artist",
                          license_id="cc-by-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                          width=640, height=480, original_width=2560, original_height=1920,
                          query="brass desk lamp", provider_rank=3, categories=("Desk lamps", "Brass"),
                          mime="image/jpeg", content_hash="abc")
    assert Candidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate


def test_wikimedia_metadata_and_api_search_order(item, tmp_path):
    def response(request):
        assert request.url.params["gsrsearch"] == "filetype:bitmap brass desk lamp"
        return httpx.Response(200, json={"query": {"pages": {
            "42": {"pageid": 42, "title": "File:Brass lamp.jpg", "index": 2, "imageinfo": [{
                "url": "https://upload.wikimedia.org/original.jpg", "thumburl": "https://upload.wikimedia.org/thumb.jpg",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Brass_lamp.jpg", "mime": "image/jpeg",
                "width": 4000, "height": 3000, "thumbwidth": 1024, "thumbheight": 768,
                "extmetadata": {"Artist": {"value": '<a href="/Artist">A. Artist</a>'},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                "Categories": {"value": "Desk lamps|Brass"}},
            }]},
            "9": {"pageid": 9, "title": "File:Diagram.svg", "index": 1,
                  "imageinfo": [{"mime": "image/svg+xml"}]},
        }}})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        candidates = WikimediaProvider(client).search("brass desk lamp", item, 5, tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider_id == "42"
    assert candidate.provider_rank == 2
    assert candidate.page_url == "https://commons.wikimedia.org/wiki/File:Brass_lamp.jpg"
    assert candidate.creator == "A. Artist"
    assert candidate.license_id == "cc-by-sa-4.0"
    assert candidate.license_url == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert (candidate.width, candidate.height, candidate.original_width, candidate.original_height) == (1024, 768, 4000, 3000)
    assert (candidate.query, candidate.categories, candidate.mime) == ("brass desk lamp", ("Desk lamps", "Brass"), "image/jpeg")


def test_openverse_metadata_retains_restrictive_rights(item, tmp_path):
    def response(request):
        return httpx.Response(200, json={"results": [{
            "id": "provider-uuid", "title": "Brass desk lamp", "url": "https://images.example/lamp.webp",
            "foreign_landing_url": "https://photos.example/123", "creator": "Photographer",
            "license": "by-nc", "license_version": "4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "width": 1200, "height": 900, "mime_type": "image/webp", "tags": [{"name": "lamp"}, {"name": "brass"}],
        }]})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        candidate, = OpenverseProvider(client).search("brass desk lamp", item, 5, tmp_path)
    assert (candidate.provider_id, candidate.page_url, candidate.creator) == (
        "provider-uuid", "https://photos.example/123", "Photographer")
    assert candidate.license_id == "cc-by-nc-4.0"
    assert candidate.license_url == "https://creativecommons.org/licenses/by-nc/4.0/"
    assert (candidate.width, candidate.height, candidate.original_width, candidate.original_height) == (1200, 900, 1200, 900)
    assert (candidate.query, candidate.provider_rank, candidate.categories, candidate.mime) == (
        "brass desk lamp", 1, ("lamp", "brass"), "image/webp")


@pytest.mark.parametrize(("license_name", "version", "expected"), [("cc0", "1.0", "cc0-1.0"), ("pdm", "1.0", "public-domain")])
def test_openverse_public_domain_declarations(item, tmp_path, license_name, version, expected):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": [{
        "url": "https://images.example/lamp.png", "license": license_name, "license_version": version,
    }]}))) as client:
        candidate, = OpenverseProvider(client).search("lamp", item, 1, tmp_path)
    assert candidate.license_id == expected


def test_crop_rejects_insufficient_unpadded_detail(item, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 800)).save(source)
    small = evolve(item, position=Placement(image_bbox=(0.2, 0.2, 0.5, 0.7)))
    # Padding would give a 348px width; original object detail is only 300px.
    assert InputCropProvider(source, min_side=320).search("lamp", small, 1, tmp_path / "small") == []
    assert not (tmp_path / "small" / "input_crop.png").exists()


def test_crop_reports_original_detail_without_upscaling(item, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 800)).save(source)
    candidate, = InputCropProvider(source, min_side=640).search("lamp", item, 1, tmp_path / "crop")
    assert (candidate.original_width, candidate.original_height) == (800, 640)
    assert (candidate.width, candidate.height) == (928, 742)
    with Image.open(candidate.path) as crop:
        assert crop.size == (928, 742)
    assert candidate.license_id == "user-provided"


def test_cache_reuses_across_downloaders_without_replacing_current_metadata(tmp_path):
    calls = []
    body = image_bytes()

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    original = Candidate("wikimedia", "Original title", "https://images.example/lamp.png", license_id="cc-by-4.0",
                         provider_id="42", creator="Artist", original_width=4000, original_height=3000)
    cache = tmp_path / "cache"
    first = downloader(response, cache_dir=cache).fetch(original, tmp_path / "first.png")
    changed = evolve(original, title="Current title", license_id="", creator="")
    second = downloader(response, cache_dir=cache).fetch(changed, tmp_path / "second.png")
    assert calls == [original.url]
    assert first is not None and second is not None
    assert (second.width, second.height, second.original_width, second.original_height) == (64, 48, 4000, 3000)
    assert (second.title, second.license_id, second.creator, second.provider_id) == ("Current title", "", "", "42")
    assert second.content_hash == first.content_hash
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()


def test_content_hash_deduplicates_decoded_pixels_and_survives_local_recheck(tmp_path):
    paths = []
    for mode in ("RGB", "RGBA"):
        path = tmp_path / f"{mode}.png"
        path.write_bytes(image_bytes(mode=mode, color=(30, 80, 120) if mode == "RGB" else (30, 80, 120, 255)))
        paths.append(path)
    fetcher = downloader(lambda request: pytest.fail("Local images must not use HTTP"))
    first = fetcher.fetch(Candidate("input_crop", "one", path=paths[0], license_id="user-provided"), tmp_path / "one.png")
    second = fetcher.fetch(Candidate("input_crop", "two", path=paths[1], license_id="user-provided"), tmp_path / "two.png")
    assert first is not None and second is not None
    assert first.content_hash == second.content_hash
    rechecked = fetcher.fetch(first, tmp_path / "reviewed.png")
    assert rechecked is not None and rechecked.content_hash == first.content_hash


@pytest.mark.parametrize("limit", [{"min_side": 49}, {"max_side": 63}, {"max_pixels": 3071}, {"max_bytes": 100}])
def test_cache_hit_revalidates_current_limits(tmp_path, limit):
    body = image_bytes()

    def response(request):
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    candidate = Candidate("openverse", "lamp", "https://images.example/lamp.png")
    cache = tmp_path / "cache"
    assert downloader(response, cache_dir=cache).fetch(candidate, tmp_path / "before.png") is not None
    assert downloader(response, cache_dir=cache, **limit).fetch(candidate, tmp_path / "after.png") is None
    assert not (tmp_path / "after.png").exists()


@pytest.mark.parametrize("damage", ["invalid-index", "invalid-png", "changed-pixels", "wrong-format", "unsafe-hash"])
def test_corrupt_cache_is_redownloaded_not_trusted(tmp_path, damage):
    body = image_bytes()
    calls = []

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    candidate = Candidate("openverse", "lamp", "https://images.example/lamp.png")
    cache = tmp_path / "cache"
    original = downloader(response, cache_dir=cache).fetch(candidate, tmp_path / "before.png")
    assert original is not None
    index, = (cache / "urls").glob("*.json")
    blob, = (cache / "content").glob("*.png")
    if damage == "invalid-index":
        index.write_text("[]")
    elif damage == "invalid-png":
        blob.write_bytes(b"not an image")
    elif damage == "changed-pixels":
        blob.write_bytes(image_bytes(color=(0, 255, 0)))
    elif damage == "wrong-format":
        blob.write_bytes(image_bytes(format="JPEG"))
    else:
        record = json.loads(index.read_text())
        record["content_hash"] = "../../outside"
        index.write_text(json.dumps(record))
    restored = downloader(response, cache_dir=cache).fetch(candidate, tmp_path / "after.png")
    assert calls == [candidate.url, candidate.url]
    assert restored is not None and restored.content_hash == original.content_hash


@pytest.mark.parametrize(("mime", "body"), [
    ("image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg"/>'),
    ("image/png", b"not an image"),
    ("image/jpeg", image_bytes()),
    ("image/gif", image_bytes(format="GIF")),
])
def test_invalid_or_mislabeled_content_is_rejected(tmp_path, mime, body):
    def response(request):
        return httpx.Response(200, content=body, headers={"content-type": mime})

    target = tmp_path / "invalid.png"
    assert downloader(response).fetch(Candidate("web", "bad", "https://images.example/bad"), target) is None
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_stream_byte_limit_without_content_length(tmp_path):
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * 100
            yield b"y" * 100

    def response(request):
        return httpx.Response(200, stream=Chunks(), headers={"content-type": "image/png"})

    assert downloader(response, max_bytes=150).fetch(Candidate("web", "large", "https://images.example/large"),
                                                   tmp_path / "large.png") is None
    assert list(tmp_path.iterdir()) == []


def test_decompression_bomb_warning_is_rejected_without_aborting(tmp_path, monkeypatch):
    body = image_bytes((64, 48))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2000)

    def response(request):
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    assert downloader(response).fetch(Candidate("web", "bomb", "https://images.example/bomb"), tmp_path / "bomb.png") is None


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://images.example/image.png", "http://user:password@images.example/image.png",
    "https://images.example:8443/image.png", "http://127.0.0.1/image.png", "http://10.0.0.1/image.png",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/image.png", "http://[::ffff:127.0.0.1]/image.png",
    "http://224.0.0.1/image.png", "http://0.0.0.0/image.png", "http://192.0.2.1/image.png",
])
def test_unsafe_destination_is_rejected_before_request(tmp_path, url):
    fetcher = downloader(lambda request: pytest.fail("Unsafe destination reached transport"))
    assert fetcher.fetch(Candidate("web", "unsafe", url), tmp_path / "unsafe.png") is None


@pytest.mark.parametrize("location", ["http://127.0.0.1/private", "http://[::1]/private", "http://images.example:8080/private",
                                      "file:///private", "http://user@images.example/private"])
def test_redirect_destination_is_validated_even_with_auto_redirect_client(tmp_path, location):
    calls = []

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": location})

    assert downloader(response).fetch(Candidate("web", "redirect", "https://images.example/start"), tmp_path / "bad.png") is None
    assert calls == ["https://images.example/start"]


def test_public_relative_redirect_and_bounded_redirect_cycle(tmp_path):
    calls = []

    def response(request):
        calls.append(request.url.path)
        if request.url.path == "/image":
            return httpx.Response(200, content=image_bytes(), headers={"content-type": "image/png"})
        return httpx.Response(302, headers={"location": "/image" if request.url.path == "/start" else "/cycle"})

    fetcher = downloader(response)
    result = fetcher.fetch(Candidate("web", "redirect", "https://images.example/start"), tmp_path / "good.png")
    assert result is not None and (result.width, result.height) == (64, 48)
    assert calls == ["/start", "/image"]
    calls.clear()
    assert fetcher.fetch(Candidate("web", "cycle", "https://images.example/cycle"), tmp_path / "bad.png") is None
    assert calls == ["/cycle"] * 6
