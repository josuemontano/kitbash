import json

import httpx
import pytest

from kitbash.infra.embeddings import HashingEmbedder
from tests.fakes.clef_flash import answer


@pytest.fixture(autouse=True)
def clef_api(monkeypatch):
    """Only the explicit fake endpoint is intercepted; adapter tests use their own transport."""
    requests = []
    send = httpx.Client.send

    def fake_send(client, request, **kwargs):
        if request.url.host != "kitbash-integration-clef.test":
            return send(client, request, **kwargs)
        assert request.method == "POST" and request.url.path == "/v1/systemone"
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=answer(payload), request=request)

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    return requests


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(64)


@pytest.fixture
def sample_inventory_dict() -> dict:
    return {
        "scene": {
            "description": "A wooden crate with a mug on top, in a bright studio.",
            "environment": "studio",
            "lighting": "soft key light from the left",
            "camera": {"location_m": [0.0, -3.0, 1.2], "rotation_deg": [78.0, 0.0, 0.0], "focal_length_mm": 50},
        },
        "items": [
            {
                "id": "wooden_crate",
                "name": "Wooden crate",
                "description": "Slatted pine crate",
                "category": "prop",
                "dimensions_m": {"width": 0.6, "depth": 0.4, "height": 0.35},
                "position": {"image_bbox": [0.3, 0.4, 0.7, 0.9], "location_m": [0.0, 0.0, 0.0], "rotation_deg": [0, 0, 10]},
                "relationships": [],
                "materials_hint": ["pine wood"],
                "confidence": 0.9,
            },
            {
                "id": "ceramic_mug",
                "name": "Ceramic mug",
                "description": "White glazed coffee mug",
                "category": "decor",
                "dimensions_m": [0.12, 0.09, 0.1],
                "position": {"image_bbox": [0.45, 0.3, 0.55, 0.42], "location_m": [0.05, 0.0, 0.35]},
                "relationships": [{"type": "on", "target": "Wooden crate"}],
                "materials_hint": "glazed ceramic, white",
                "confidence": 0.8,
            },
        ],
    }
