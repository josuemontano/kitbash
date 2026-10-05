from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from kitbash.errors import BlenderScriptError
from kitbash.infra.imaging import ImageComparison
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.usd_fidelity import UsdCheck, UsdFidelityChecker


@pytest.mark.parametrize("mode", ["preview_surface_baked", "materialx"])
def test_known_preview_loss_rejected_despite_perfect_roundtrip(tmp_path, mode):
    image = tmp_path / "render.png"
    Image.new("RGB", (32, 32), (160, 80, 40)).save(image)
    material = {
        "usd_prim": "glass_material",
        "rung": mode,
        "materialx_lossless": mode == "materialx",
        "expected_preview_channels": ["Base Color"],
        "lost_in_preview": [],
    }
    toolkit = Mock(spec=BlenderToolkit)
    toolkit.roundtrip_resolution = 32
    toolkit.roundtrip_samples = 1
    toolkit.export_usd.return_value = {
        "materials": {"Glass Material": material},
        "usd_material_mode": mode,
    }
    toolkit.usd_roundtrip.return_value = {
        "materials_total": 1,
        "materials_ok": 1,
        "materials": {"glass_material": {"found": True, "principled": True, "missing_channels": []}},
        "images": [str(image)],
    }
    toolkit.render_views.return_value = [image]
    checker = UsdFidelityChecker(toolkit)
    kwargs = {
        "work_dir": tmp_path / "work",
        "roundtrip_dir": tmp_path / "roundtrip",
        "prefix": "asset",
        "log_dir": tmp_path,
    }
    clean = checker.check(tmp_path / "asset.blend", tmp_path / "asset.usdc", **kwargs)
    assert clean.score == pytest.approx(1.0)
    assert clean.facts()["usd_broken_materials"] == 0

    material["lost_in_preview"] = ["Transmission Weight", "Subsurface Weight"]
    with pytest.raises(BlenderScriptError) as failure:
        checker.check(tmp_path / "asset.blend", tmp_path / "asset.usdc", **kwargs)
    assert "Glass Material" in str(failure.value)
    assert "Transmission Weight" in str(failure.value)
    assert "Subsurface Weight" in str(failure.value)
    # The clean check ran once; a known lossy export never reaches the round trip.
    assert toolkit.usd_roundtrip.call_count == 1
    assert toolkit.render_views.call_count == 1


@pytest.mark.parametrize("broken_prim, expected_broken", [(None, 1), ("glass_material", 1), ("wood_material", 2)])
def test_raw_check_facts_count_preview_loss_without_double_counting(broken_prim, expected_broken):
    materials = {
        prim: {"found": True, "principled": True, "missing_channels": ["Base Color"] if prim == broken_prim else []}
        for prim in ("glass_material", "wood_material")
    }
    check = UsdCheck(
        usd_path=Path("asset.usdc"),
        export={
            "usd_material_mode": "materialx",
            "materials": {
                "Glass Material": {
                    "usd_prim": "glass_material",
                    "materialx_lossless": True,
                    "lost_in_preview": ["Transmission Weight"],
                },
                "Wood Material": {"usd_prim": "wood_material", "lost_in_preview": []},
            },
        },
        roundtrip={"materials_total": 2, "materials_ok": 2 - int(broken_prim is not None), "materials": materials},
        comparisons=(ImageComparison(ssim=1.0, color_delta=0.0),),
        compare_image=None,
    )
    assert check.facts()["usd_roundtrip_score"] == 1.0
    assert check.facts()["usd_broken_materials"] == expected_broken
    assert check.report()["materials"]["Glass Material"]["lost_in_preview"] == ["Transmission Weight"]
