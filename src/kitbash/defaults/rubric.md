# kitbash rubric

Critics score each criterion that applies to the current phase from 0 to 1 and decide
whether it passes. Edit this file (or pass `--rubric my_rubric.md`) to change critic
behavior. No code changes are needed.

Columns:

- **criterion**: what is judged. Its slug (lowercase, underscores) is the criterion id.
- **weight**: relative weight in the phase score.
- **pass condition**: what the critics must verify. A condition in backticks is also
  checked by machine against measured facts, for example `usd_roundtrip_score >= 0.85`.
  When every fact it names is available, the machine check decides pass or fail.
- **applies to**: comma-separated phases: `breakdown`, `modelling`, `layout`, `assembly` or `all`.
- **critic** (optional): `visual`, `technical` or `both` (default `both`).

| criterion | weight | pass condition | applies to | critic |
|---|---|---|---|---|
| Real-world scale is plausible | 2 | Dimensions are plausible for the object in meters and within 25% of the inventory estimate `scale_error <= 0.25` | breakdown, modelling | both |
| Origin at the base, Z up | 1 | The origin sits at the bottom center of the bounds and the object stands upright along +Z `origin_offset_m <= 0.01 and up_axis_ok` | modelling | technical |
| Asset name is correct and follows the naming convention | 1 | The object, mesh data, materials and images use the configured naming convention and the name describes the object `naming_violations == 0` | modelling, layout | technical |
| Matches the reference (shape, proportions, color) | 3 | Silhouette, proportions, part layout and dominant colors match the reference image | breakdown, modelling | visual |
| Uses Principled BSDF with sensible PBR values | 2 | Every material ends in a Principled BSDF, is built from editable nodes, metallic is near 0 or 1, roughness is plausible and base colors avoid pure black or white `non_principled_materials == 0` | modelling | technical |
| USD material fidelity | 2 | The exported USD re-imports with a Principled BSDF graph per material, the round-trip render is within threshold and no textures are missing or have broken paths `usd_roundtrip_score >= 0.85 and missing_textures == 0 and usd_broken_materials == 0` | modelling, assembly | technical |
| Inventory is complete and correctly identified | 2 | Every salient object in the reference is listed once, with a specific name, a clear description, a sensible category and plausible materials | breakdown | both |
| Spatial arrangement matches the reference | 3 | Relative positions, rotations, relationships (on, next to, inside) and camera framing match the reference | breakdown, layout | visual |
| Style compliance | 2 | Shading, lighting, camera and color treatment follow the selected style | layout | both |
| Lighting | 2 | Lighting is plausible for the scene, exposure is balanced and nothing is blown out or crushed | layout | visual |
| Scene integrity | 1 | Every approved asset is placed, skipped assets have labelled placeholders, nothing floats or intersects unintentionally `missing_assets == 0` | layout | technical |
