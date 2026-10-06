import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from attrs import evolve
from PIL import Image, ImageFilter

from kitbash.config import load_config
from kitbash.domain.inventory import Inventory
from kitbash.infra.image_search import Candidate, ImageDownloader, InputCropProvider
from kitbash.services.reference_quality import rights_allowed
from kitbash.services.references import ReferenceFinder


class NoModel:
    def ask_json(self, **kwargs):
        pytest.fail("deterministic acquisition invoked a model")


class Provider:
    name = "wikimedia"
    once_per_item = False

    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def search(self, query, item, limit, workdir):
        self.calls += 1
        return [evolve(c, query=query) for c in self.candidates]


def photo(path, *, color=0, blurred=False):
    pixels = np.full((640, 640, 3), 255, dtype=np.uint8)
    y, x = np.indices((384, 384))
    texture = np.where((x // 4 + y // 4) % 2, 40, 160).astype(np.uint8)
    pixels[128:512, 128:512] = texture[..., None]
    pixels[128:512, 128:512, color] //= 2
    image = Image.fromarray(pixels)
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(18))
    image.save(path)
    return path


def licensed(path, **changes):
    return evolve(Candidate(
        "wikimedia", "Wooden crate", url="https://images.example/crate.png", path=path,
        license="CC BY 4.0", provider_id="42", page_url="https://commons.wikimedia.org/wiki/File:Crate.png",
        creator="A Photographer", license_id="cc-by-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        width=640, height=640, original_width=640, original_height=640,
    ), **changes)


@pytest.fixture
def item(sample_inventory_dict):
    return Inventory.from_dict(sample_inventory_dict).items[0]


def finder(tmp_path, providers, *, config=None, llm=None):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail(f"unexpected download: {r.url}")))
    return ReferenceFinder(
        providers, ImageDownloader(client, max_bytes=2_000_000, min_side=256), llm or NoModel(),
        config or load_config().reference, SimpleNamespace(event=lambda *a, **kw: None), cache_dir=tmp_path / "cache",
    )


def test_explicit_user_image_precedes_search_and_needs_no_model(tmp_path, item):
    provider = Provider([])
    path = photo(tmp_path / "explicit.png", blurred=True)
    choice = finder(tmp_path, [provider]).find(evolve(item, user_reference=str(path)), tmp_path / "run")
    assert choice.source == "user"
    assert choice.provenance["license_id"] == "user-provided"
    assert provider.calls == 0


def test_detailed_crop_precedes_api_but_tiny_native_crop_does_not(tmp_path, item):
    source = photo(tmp_path / "source.png")
    provider = Provider([])
    cropped_item = evolve(item, position=evolve(item.position, image_bbox=(0, 0, 1, 1)))
    choice = finder(tmp_path, [InputCropProvider(source, 384), provider]).find(cropped_item, tmp_path / "large")
    assert choice.source == "input_crop" and provider.calls == 0
    tiny_item = evolve(item, position=evolve(item.position, image_bbox=(0.3, 0.3, 0.4, 0.4)))
    assert finder(tmp_path, [InputCropProvider(source, 384), provider]).find(tiny_item, tmp_path / "tiny") is None
    assert provider.calls > 0


def test_high_quality_single_candidate_selects_without_model_and_preserves_provenance(tmp_path, item):
    candidate = licensed(photo(tmp_path / "crate.png"))
    choice = finder(tmp_path, [Provider([candidate])]).find(item, tmp_path / "run")
    assert choice is not None
    selected = json.loads((tmp_path / "run/selection.json").read_text())
    assert selected["provenance"]["page_url"] == candidate.page_url
    assert selected["provenance"]["creator"] == candidate.creator
    assert selected["provenance"]["url"] == candidate.url
    assert selected["provenance"]["query"]
    assert selected["provenance"]["quality"]["score"] >= 0.88


def test_lexical_match_does_not_override_blur_and_human_choice_keeps_rights(tmp_path, item):
    candidate = licensed(photo(tmp_path / "blurry.png", blurred=True))
    service = finder(tmp_path, [Provider([candidate])])
    directory = tmp_path / "run"
    assert service.find(item, directory) is None
    review = json.loads((directory / "review.json").read_text())
    assert Path(review["contact_sheet"]).is_file()
    assert review["candidates"][0]["quality"]["automatic_eligible"] is False
    choice = service.select_reviewed(item, directory, 0)
    assert choice.source == "wikimedia"
    assert choice.provenance["creator"] == candidate.creator
    assert choice.provenance["license_id"] == candidate.license_id


def test_tied_candidates_require_review_and_changed_file_cannot_be_approved(tmp_path, item):
    first = licensed(photo(tmp_path / "first.png"))
    second = licensed(photo(tmp_path / "second.png", color=1), provider_id="43", url="https://images.example/second.png")
    service = finder(tmp_path, [Provider([second, first])])
    directory = tmp_path / "run"
    assert service.find(item, directory) is None
    rows = json.loads((directory / "review.json").read_text())["candidates"]
    assert [row["provider_id"] for row in rows] == ["42", "43"]
    Image.new("RGB", (640, 640), "black").save(rows[0]["path"])
    assert service.select_reviewed(item, directory, 0) is None
    assert service.select_reviewed(item, directory, -1) is None


def test_search_cache_crosses_finder_instances_and_rechecks_rights(tmp_path, item):
    candidate = licensed(photo(tmp_path / "crate.png"))
    provider = Provider([candidate])
    assert finder(tmp_path, [provider]).find(item, tmp_path / "first") is not None
    cached_provider = Provider([])
    restricted = evolve(load_config().reference, allowed_licenses=("cc0-1.0",))
    assert finder(tmp_path, [cached_provider], config=restricted).find(item, tmp_path / "second") is None
    assert cached_provider.calls == 0
    expired = evolve(load_config().reference, search_cache_ttl_s=0)
    finder(tmp_path, [cached_provider], config=expired).find(item, tmp_path / "third")
    assert cached_provider.calls > 0


def test_decoded_duplicates_cannot_create_false_ambiguity(tmp_path, item):
    first = licensed(photo(tmp_path / "same.png"))
    duplicate = evolve(first, url="https://other.example/same.png", provider_id="43")
    service = finder(tmp_path, [Provider([first, duplicate])])
    assert service.find(item, tmp_path / "run") is not None
    rows = json.loads((tmp_path / "run/candidates/candidates.json").read_text())
    assert [row["provider_id"] for row in rows] == ["42"]


@pytest.mark.parametrize("changes", [
    {"license_id": ""}, {"license_id": "cc-by-nc-4.0"}, {"license_id": "cc-by-nd-4.0"},
    {"creator": ""}, {"page_url": ""}, {"license_url": "https://example.org/free"},
    {"license_url": "https://creativecommons.org/licenses/by-nc/4.0/"},
])
def test_rights_fail_closed_for_unknown_restricted_or_inconsistent_claims(tmp_path, item, changes):
    candidate = licensed(tmp_path / "must-not-be-read.png", **changes)
    assert not rights_allowed(candidate, load_config().reference.allowed_licenses)
    assert finder(tmp_path, [Provider([candidate])]).find(item, tmp_path / "run") is None


def test_vision_is_opt_in_and_cannot_bypass_quality_threshold(tmp_path, item):
    candidate = licensed(photo(tmp_path / "blurry.png", blurred=True))
    calls = []

    class Visual:
        def ask_json(self, **kwargs):
            calls.append(kwargs["task"])
            return kwargs["validate"]({"choice": 1, "reason": "whole object"})

    service = finder(tmp_path, [Provider([candidate])], config=evolve(load_config().reference, vision_fallback=True), llm=Visual())
    assert service.find(item, tmp_path / "run") is None
    assert calls == []


def test_opt_in_vision_can_resolve_high_quality_tie(tmp_path, item):
    first = licensed(photo(tmp_path / "first.png"))
    second = licensed(photo(tmp_path / "second.png", color=1), provider_id="43", url="https://images.example/second.png")

    class Visual:
        def ask_json(self, **kwargs):
            return kwargs["validate"]({"choice": 2, "reason": "correct shape"})

    service = finder(tmp_path, [Provider([first, second])], config=evolve(load_config().reference, vision_fallback=True), llm=Visual())
    assert service.find(item, tmp_path / "run").provenance["provider_id"] == "43"


def test_pixel_ranking_considers_candidates_beyond_display_budget(tmp_path, item):
    blurry = licensed(photo(tmp_path / "blur.png", blurred=True))
    duplicate = evolve(blurry, provider_id="43", provider_rank=1, url="https://images.example/blur2.png")
    sharp = licensed(photo(tmp_path / "sharp.png"), provider_id="44", provider_rank=2, url="https://images.example/sharp.png")
    service = finder(tmp_path, [Provider([blurry, duplicate, sharp])], config=evolve(load_config().reference, max_candidates=2))
    choice = service.find(item, tmp_path / "run")
    assert choice.provenance["provider_id"] == "44"
