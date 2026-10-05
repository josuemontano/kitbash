"""Export the open .blend to USD with the material fidelity ladder.

1. MaterialX network, when this Blender can export it and the export keeps every linked Principled input.
2. UsdPreviewSurface otherwise; inputs it cannot express (procedural textures, ramps, math...) are baked
   to image textures stored next to the USD and referenced by relative path.

Work happens in a temporary copy of the .blend, so the original procedural node graphs are never touched.
"""

import os

import bpy
import kb_materials as km
import kb_render
import kitbash_bpy as kb
from pxr import Sdf, Usd, UsdShade

MATERIALX = "materialx"
PREVIEW_BAKED = "preview_surface_baked"
VALUE_TYPES = {"color3f": Sdf.ValueTypeNames.Color3f, "float": Sdf.ValueTypeNames.Float, "normal3f": Sdf.ValueTypeNames.Normal3f}
TEXTURE_OUTPUTS = {"rgb": Sdf.ValueTypeNames.Float3, "r": Sdf.ValueTypeNames.Float}

options = kb.args()
output_usd = os.path.abspath(options["output_usd"])
usd_dir = os.path.dirname(output_usd)
textures_dir = os.path.join(usd_dir, "textures")
work_dir = os.path.abspath(options["work_dir"])
include_scene = bool(options.get("scene", False))
export_options = {p.identifier for p in bpy.ops.wm.usd_export.get_rna_type().properties}
materialx_supported = options.get("materialx", "auto") == "auto" and "generate_materialx_network" in export_options
os.makedirs(work_dir, exist_ok=True)
os.makedirs(textures_dir, exist_ok=True)

# The rest of this script only ever modifies (and saves) the temporary copy.
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(work_dir, "export_copy.blend"), check_existing=False, relative_remap=True)


def localize_instances():
    """Linked libraries and collection instances (assembly.mode = "link") become local, real objects in
    the copy, so their materials can be analysed and baked like any other."""
    for datablocks in (bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.images):
        for datablock in list(datablocks):
            if datablock.library is not None:
                datablock.make_local()
    instancers = [o for o in bpy.context.scene.objects if o.instance_type == "COLLECTION" and o.instance_collection]
    if instancers:
        with bpy.context.temp_override(
            active_object=instancers[0], object=instancers[0], selected_objects=instancers, selected_editable_objects=instancers
        ):
            bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
        # Keep the tagged placement root as the parent of its realized geometry, without exporting
        # the collection instance a second time. Only the root carries kb_asset_key.
        for obj in instancers:
            obj.instance_type = "NONE"
            obj.instance_collection = None


localize_instances()
if include_scene:
    for obj in bpy.context.scene.objects:
        if obj.type == "CAMERA":
            obj["kb_camera_name"] = obj.name
            obj["kb_active_camera"] = obj == bpy.context.scene.camera


def export(path, materialx):
    wanted = {
        "filepath": path,
        "export_materials": True,
        "generate_preview_surface": True,
        "generate_materialx_network": materialx,
        "export_textures_mode": "NEW",
        "overwrite_textures": True,
        "relative_paths": True,
        "export_uvmaps": True,
        "rename_uvmaps": True,
        "export_normals": True,
        "selected_objects_only": False,
        "evaluation_mode": "RENDER",
        "use_instancing": False,
        "export_custom_properties": True,
        "custom_properties_namespace": "userProperties",
        "author_blender_name": True,
        "merge_parent_xform": False,
        "export_cameras": include_scene,
        "export_lights": include_scene,
        "convert_world_material": include_scene,
    }
    bpy.ops.wm.usd_export(**{k: v for k, v in wanted.items() if k in export_options})


def usd_materials(stage):
    """Blender material name -> UsdShade.Material (the exporter records the original name)."""
    found = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            attr = prim.GetAttribute("userProperties:blender:data_name")
            found[attr.Get() if attr and attr.Get() else prim.GetName()] = UsdShade.Material(prim)
    return found


def mtlx_connected(material, principled_inputs):
    surface = material.ComputeSurfaceSource(renderContext="mtlx")[0]
    if not surface:
        return False
    for name in principled_inputs:
        target = km.MATERIALX_INPUTS.get(name)
        usd_input = surface.GetInput(target) if target else None
        if usd_input is None or not usd_input.HasConnectedSource():
            return False
    return True


# -- 1. analyse every material ---------------------------------------------------------------------
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
materials = km.mesh_materials(meshes)
analysis = {}
for material in materials:
    bsdfs = km.principled_nodes(material)
    linked, direct = {}, []
    for bsdf in bsdfs[:1]:
        for socket in bsdf.inputs:
            if socket.is_linked:
                linked[socket.name] = km.upstream_types(socket)
                if km.is_direct_image(socket):
                    direct.append(socket.name)
    analysis[material.name] = {
        "principled_count": len(bsdfs),
        "ends_in_principled": km.ends_in_principled(material),
        "linked_inputs": linked,
        "direct_image_inputs": direct,
    }

# -- 2. can MaterialX carry each graph without losing links? ---------------------------------------
if materialx_supported and materials:
    probe_path = os.path.join(work_dir, "materialx_probe.usda")
    export(probe_path, True)
    probe_stage = Usd.Stage.Open(probe_path)  # keep the stage alive while its prims are used
    probed = usd_materials(probe_stage)
    for name, info in analysis.items():
        info["materialx_lossless"] = (
            name in probed
            and info["principled_count"] == 1
            and info["ends_in_principled"]
            and mtlx_connected(probed[name], info["linked_inputs"])
        )
else:
    for info in analysis.values():
        info["materialx_lossless"] = False

# -- 3. choose the rung and the channels to bake ---------------------------------------------------
for info in analysis.values():
    info["rung"] = MATERIALX if materialx_supported and info["materialx_lossless"] else PREVIEW_BAKED
    needs_preview_bake = info["rung"] == PREVIEW_BAKED or options.get("bake_preview_fallback", True)
    procedural = [c for c in info["linked_inputs"] if c not in info["direct_image_inputs"]]
    info["bake_channels"] = [c for c in procedural if c in km.PREVIEW_INPUTS] if needs_preview_bake else []
    info["lost_in_preview"] = [c for c in procedural if c not in km.PREVIEW_INPUTS]
    if not info["ends_in_principled"]:
        info["lost_in_preview"].append("surface (not a Principled BSDF)")
    info["baked"] = {}


def split_shared_materials():
    """Bake targets must map to one mesh: duplicate a baked material used by several meshes."""
    users = {}
    for obj in meshes:
        for slot in obj.material_slots:
            if slot.material is not None:
                users.setdefault(slot.material.name, []).append(obj)
    for name, info in list(analysis.items()):
        datas = {o.data.name for o in users.get(name, [])}
        if not info["bake_channels"] or len(datas) < 2:
            continue
        for mesh_name in sorted(datas)[1:]:
            mesh = bpy.data.meshes[mesh_name]
            copy = bpy.data.materials[name].copy()
            copy.name = f"{name}_{kb.part_name(mesh_name)}"  # no dots: USD prim names drop them
            for index, slot_material in enumerate(mesh.materials):
                if slot_material is not None and slot_material.name == name:
                    mesh.materials[index] = copy
            analysis[copy.name] = {**info, "baked": {}, "copy_of": name}


def bake_channel(channel, targets):
    """Bake ``channel`` of every material in ``targets`` into its own image in one Cycles pass."""
    objects = [o for o in meshes if any(s.material and s.material.name in targets for s in o.material_slots)]
    for obj in objects:
        uv = kb.ensure_uv(obj)
        layer = obj.data.uv_layers[uv]
        obj.data.uv_layers.active = layer
        layer.active_render = True
    restore, images = [], {}
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or any(material.name == m.name for m, *_ in restore):
                continue
            tree = material.node_tree
            node = tree.nodes.new("ShaderNodeTexImage")
            added = [node]
            if material.name in targets:
                size = int(options.get("bake_resolution", 1024))
                image = bpy.data.images.new(f"bake_{kb.part_name(material.name)}_{kb.part_name(channel)}", size, size)
                image.colorspace_settings.name = "sRGB" if km.PREVIEW_INPUTS[channel][3] == "sRGB" else "Non-Color"
                images[material.name] = image
                node.image = image
                if channel != "Normal":
                    bsdf = km.principled_nodes(material)[0]
                    output = km.output_node(material)
                    original = output.inputs["Surface"].links[0].from_socket if output.inputs["Surface"].is_linked else None
                    emission = tree.nodes.new("ShaderNodeEmission")
                    tree.links.new(bsdf.inputs[channel].links[0].from_socket, emission.inputs["Color"])
                    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
                    added.append(emission)
                    restore.append((material, added, output, original))
                    tree.nodes.active = node
                    continue
            else:
                node.image = bpy.data.images.get("kb_bake_dummy") or bpy.data.images.new("kb_bake_dummy", 8, 8)
            tree.nodes.active = node
            restore.append((material, added, None, None))
    kb.activate(objects[0])
    for obj in objects:
        obj.select_set(True)
    with bpy.context.temp_override(active_object=objects[0], object=objects[0], selected_objects=objects, selected_editable_objects=objects):
        bpy.ops.object.bake(type="NORMAL" if channel == "Normal" else "EMIT", normal_space="TANGENT", margin=8, use_clear=True)
    for material_name, image in images.items():
        filename = f"{image.name}.png"
        image.filepath_raw = os.path.join(textures_dir, filename)
        image.file_format = "PNG"
        image.save()
        analysis[material_name]["baked"][channel] = f"./textures/{filename}"
    for material, added, output, original in restore:
        tree = material.node_tree
        if output is not None and original is not None:
            tree.links.new(original, output.inputs["Surface"])
        for node in added:
            tree.nodes.remove(node)


# -- 4. bake in the temporary copy ------------------------------------------------------------------
if any(info["bake_channels"] for info in analysis.values()):
    split_shared_materials()
    kb_render.configure_render("CYCLES", options.get("bake_samples", 16), options.get("device", "CPU"), (64, 64))
    channels = sorted({c for info in analysis.values() for c in info["bake_channels"]})
    for channel in channels:
        targets = {name for name, info in analysis.items() if channel in info["bake_channels"]}
        bake_channel(channel, targets)

# -- 5. final export and USD authoring --------------------------------------------------------------
export(output_usd, materialx_supported)
stage = Usd.Stage.Open(output_usd)
exported = usd_materials(stage)
for name, info in analysis.items():
    material = exported.get(name)
    if material is None:
        info["notes"] = "material not found in the exported USD"
        continue
    info["usd_prim"] = material.GetPrim().GetName()  # the name Blender's importer gives the material
    if info["rung"] == PREVIEW_BAKED and materialx_supported:
        material.GetPrim().RemoveProperty("outputs:mtlx:surface")
    if not info["baked"]:
        continue
    surface = material.ComputeSurfaceSource()[0]
    base = material.GetPath()
    reader = UsdShade.Shader.Define(stage, base.AppendChild("kb_st_reader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for channel, relative in info["baked"].items():
        usd_input, value_type, texture_output, colorspace = km.PREVIEW_INPUTS[channel]
        texture = UsdShade.Shader.Define(stage, base.AppendChild(f"kb_bake_{kb.part_name(channel)}"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_output)
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if channel == "Normal":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
        source = texture.CreateOutput(texture_output, TEXTURE_OUTPUTS[texture_output])
        surface.CreateInput(usd_input, VALUE_TYPES[value_type]).ConnectToSource(source)
stage.Save()

# -- 6. check every texture reference resolves and is relative ---------------------------------------
textures, missing, absolute = [], [], []
for prim in stage.Traverse():
    if not prim.IsA(UsdShade.Shader):
        continue
    for shader_input in UsdShade.Shader(prim).GetInputs():
        if shader_input.GetTypeName() != Sdf.ValueTypeNames.Asset:
            continue
        value = shader_input.Get()
        if not value or not value.path:
            continue
        resolved = value.resolvedPath or os.path.join(usd_dir, value.path)
        textures.append(value.path)
        if not os.path.isfile(resolved):
            missing.append(value.path)
        if os.path.isabs(value.path):
            absolute.append(value.path)

for info in analysis.values():
    direct = [c for c in info["direct_image_inputs"] if c in km.PREVIEW_INPUTS]
    info["expected_preview_channels"] = sorted(set(direct) | set(info["baked"]))

rungs = [info["rung"] for info in analysis.values()]
kb.emit(
    "usd",
    {
        "usd_path": output_usd,
        "materialx_supported": materialx_supported,
        "usd_material_mode": MATERIALX if rungs and all(r == MATERIALX for r in rungs) else PREVIEW_BAKED,
        "materials": analysis,
        "textures": sorted(set(textures)),
        "missing_textures": sorted(set(missing)),
        "absolute_texture_paths": sorted(set(absolute)),
    },
)
