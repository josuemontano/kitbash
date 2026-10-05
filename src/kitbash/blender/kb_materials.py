"""Material graph analysis shared by the inspection, USD export and round-trip scripts."""

import os

import bpy

# Principled input -> (UsdPreviewSurface input, USD value type, texture output, source color space)
PREVIEW_INPUTS = {
    "Base Color": ("diffuseColor", "color3f", "rgb", "sRGB"),
    "Roughness": ("roughness", "float", "r", "raw"),
    "Metallic": ("metallic", "float", "r", "raw"),
    "Normal": ("normal", "normal3f", "rgb", "raw"),
    "Emission Color": ("emissiveColor", "color3f", "rgb", "sRGB"),
    "Alpha": ("opacity", "float", "r", "raw"),
    "Coat Weight": ("clearcoat", "float", "r", "raw"),
    "Coat Roughness": ("clearcoatRoughness", "float", "r", "raw"),
}
# Principled input -> OpenPBR (MaterialX) surface input written by Blender's exporter
MATERIALX_INPUTS = {
    "Base Color": "base_color",
    "Metallic": "base_metalness",
    "Roughness": "specular_roughness",
    "IOR": "specular_ior",
    "Alpha": "geometry_opacity",
    "Normal": "geometry_normal",
    "Diffuse Roughness": "base_diffuse_roughness",
    "Subsurface Weight": "subsurface_weight",
    "Specular IOR Level": "specular_weight",
    "Anisotropic": "specular_roughness_anisotropy",
    "Tangent": "geometry_tangent",
    "Transmission Weight": "transmission_weight",
    "Coat Weight": "coat_weight",
    "Coat Roughness": "coat_roughness",
    "Coat Normal": "geometry_coat_normal",
    "Sheen Weight": "fuzz_weight",
    "Emission Color": "emission_color",
    "Emission Strength": "emission_luminance",
}
SHADER_MIXERS = ("ShaderNodeMixShader", "ShaderNodeAddShader")
UV_SOURCES = ("ShaderNodeUVMap", "ShaderNodeTexCoord", "ShaderNodeMapping")


def output_node(material):
    tree = material.node_tree
    if tree is None:
        return None
    outputs = [n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    active = [n for n in outputs if n.is_active_output and n.target in ("ALL", "CYCLES")]
    return (active or outputs or [None])[0]


def surface_node(material):
    output = output_node(material)
    if output is None or not output.inputs["Surface"].is_linked:
        return None
    return output.inputs["Surface"].links[0].from_node


def principled_nodes(material):
    """Principled BSDFs reachable from the surface output through Mix/Add Shader nodes."""
    found, stack, seen = [], [surface_node(material)], set()
    while stack:
        node = stack.pop()
        if node is None or node.name in seen:
            continue
        seen.add(node.name)
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            found.append(node)
        elif node.bl_idname in SHADER_MIXERS:
            stack.extend(link.from_node for socket in node.inputs for link in socket.links)
    return found


def ends_in_principled(material):
    node = surface_node(material)
    if node is None:
        return False
    if node.bl_idname == "ShaderNodeBsdfPrincipled":
        return True
    return node.bl_idname in SHADER_MIXERS and bool(principled_nodes(material))


def upstream_types(socket):
    types, stack, seen = [], [link.from_node for link in socket.links], set()
    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        types.append(node.bl_idname)
        stack.extend(link.from_node for s in node.inputs for link in s.links)
    return sorted(set(types))


def _uv_mapped_image(node):
    if node is None or node.bl_idname != "ShaderNodeTexImage" or node.projection != "FLAT" or node.image is None:
        return False
    vector = node.inputs["Vector"]
    if not vector.is_linked:
        return True
    source = vector.links[0].from_node
    if source.bl_idname == "ShaderNodeTexCoord":
        return vector.links[0].from_socket.name == "UV"
    return source.bl_idname in UV_SOURCES and all(t in UV_SOURCES for t in upstream_types(vector))


def is_direct_image(socket):
    """True when UsdPreviewSurface can express this input without baking (a UV-mapped image texture)."""
    if not socket.is_linked:
        return True
    node = socket.links[0].from_node
    if socket.name == "Normal" and node.bl_idname == "ShaderNodeNormalMap":
        color = node.inputs["Color"]
        return color.is_linked and _uv_mapped_image(color.links[0].from_node)
    return _uv_mapped_image(node)


def linked_inputs(bsdf):
    return {socket.name: upstream_types(socket) for socket in bsdf.inputs if socket.is_linked}


def _value(socket):
    value = getattr(socket, "default_value", None)
    try:
        return [round(float(v), 4) for v in value]
    except TypeError:
        return round(float(value), 4) if isinstance(value, (int, float)) else None


def image_info(image, base_dir=None):
    filepath = image.filepath or image.filepath_raw
    absolute = bpy.path.abspath(filepath, start=base_dir) if filepath else ""
    return {
        "name": image.name,
        "filepath": filepath,
        "relative": filepath.startswith("//"),
        "exists": bool(absolute) and os.path.isfile(absolute),
        "packed": image.packed_file is not None,
        "colorspace": image.colorspace_settings.name,
    }


def images_of(material):
    tree = material.node_tree
    if tree is None:
        return []
    return [n.image for n in tree.nodes if getattr(n, "image", None) is not None]


def material_summary(material):
    tree = material.node_tree
    summary = {
        "name": material.name,
        "ends_in_principled": ends_in_principled(material),
        "node_types": sorted({n.bl_idname for n in tree.nodes}) if tree else [],
        "node_count": len(tree.nodes) if tree else 0,
        "link_count": len(tree.links) if tree else 0,
        "images": [image_info(image) for image in images_of(material)],
        "principled": [],
    }
    for bsdf in principled_nodes(material):
        inputs = {}
        for socket in bsdf.inputs:
            if socket.is_linked:
                inputs[socket.name] = {"linked": upstream_types(socket), "direct_image": is_direct_image(socket)}
            elif socket.name in ("Base Color", "Metallic", "Roughness", "IOR", "Alpha", "Transmission Weight", "Emission Strength", "Coat Weight"):
                inputs[socket.name] = _value(socket)
        summary["principled"].append({"node": bsdf.name, "inputs": inputs})
    return summary


def mesh_materials(objects):
    """Unique materials used by mesh ``objects``, in first-use order."""
    seen, result = set(), []
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material is not None and slot.material.name not in seen:
                seen.add(slot.material.name)
                result.append(slot.material)
    return result
