"""Standard USD orthographic cameras, plus Blender's aperture-unit import correction.

Projection, aperture, offsets, clipping and transforms are USD schema data, not a
private camera recipe. Only selection identity and raster/pixel aspect are application
metadata: UsdGeomCamera does not specify a render resolution or an active camera.
"""

import bpy
from pxr import Gf, Sdf, UsdGeom


def author_orthographic_cameras(stage, scene):
    """Fill cameras omitted by Blender, retaining its exported object/parent Xforms."""
    roots = {}
    for prim in stage.Traverse():
        name = prim.GetAttribute("userProperties:kb_camera_name")
        if name and name.Get():
            roots[name.Get()] = prim
    unit_scale = 1.0 / UsdGeom.GetStageMetersPerUnit(stage)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type != "CAMERA" or obj.data.type != "ORTHO" or obj.hide_render:
            continue
        root = roots.get(obj.name)
        if root is None:
            raise RuntimeError(f"USD export omitted camera object transform: {obj.name}")
        # Future Blender exporters may already support orthographic cameras. Do not
        # author a second camera or overwrite their standard camera properties.
        if root.IsA(UsdGeom.Camera) or any(child.IsA(UsdGeom.Camera) for child in root.GetChildren()):
            continue
        path = root.GetPath().AppendChild("Camera")
        index = 1
        while stage.GetPrimAtPath(path):
            path = root.GetPath().AppendChild(f"Camera_{index}")
            index += 1
        camera = UsdGeom.Camera.Define(stage, path)
        data = obj.evaluated_get(depsgraph).data
        # view_frame includes sensor fit, render aspect, non-square pixels and shift.
        # Its corners are camera-local, so the existing parent Xforms carry the full
        # evaluated world transform exactly once. USD apertures use tenths of a unit.
        frame = data.view_frame(scene=scene)
        left, right = min(v.x for v in frame), max(v.x for v in frame)
        bottom, top = min(v.y for v in frame), max(v.y for v in frame)
        aperture_units = 10.0 * unit_scale
        camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
        camera.CreateHorizontalApertureAttr((right - left) * aperture_units)
        camera.CreateVerticalApertureAttr((top - bottom) * aperture_units)
        camera.CreateHorizontalApertureOffsetAttr((right + left) * 0.5 * aperture_units)
        camera.CreateVerticalApertureOffsetAttr((top + bottom) * 0.5 * aperture_units)
        camera.CreateClippingRangeAttr(Gf.Vec2f(data.clip_start * unit_scale, data.clip_end * unit_scale))
        camera.CreateFocalLengthAttr(data.lens / (100.0 * UsdGeom.GetStageMetersPerUnit(stage)))
        camera.CreateFStopAttr(0.0)
        camera.GetPrim().CreateAttribute("userProperties:blender:data_name", Sdf.ValueTypeNames.String).Set(data.name)


def normalize_imported_orthographic_scale(stage, scene):
    """Correct Blender's missing tenth-unit conversion using standard USD data only.

    Blender 5.2 imports max(apertures) directly as ortho_scale, although apertures
    are tenths of a stage unit. The camera itself, its transform, shifts and identity
    must already have survived the normal USD importer; no camera is synthesized.
    Assigning the schema-derived value is also correct when Blender fixes its reader.
    """
    imported = {obj.get("kb_camera_name", obj.name): obj for obj in scene.objects if obj.type == "CAMERA"}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        camera = UsdGeom.Camera(prim)
        if camera.GetProjectionAttr().Get() != UsdGeom.Tokens.orthographic:
            continue
        root = prim
        name = None
        while root and not root.IsPseudoRoot():
            attr = root.GetAttribute("userProperties:kb_camera_name")
            if attr and attr.Get():
                name = attr.Get()
                break
            root = root.GetParent()
        obj = imported.get(name or prim.GetName())
        if obj is not None and obj.data.type == "ORTHO":
            obj.data.ortho_scale = max(
                camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()
            ) * UsdGeom.GetStageMetersPerUnit(stage) / 10.0
