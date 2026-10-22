"""
Convert a static FBX scene with PBR materials into a .gltf (+ .bin) for threepp.

  blender --background --python scripts/fbx_to_gltf.py -- <src.fbx> <out.gltf> [--material-mode auto|pbr|phong]

Blender reads the FBX (geometry, node transforms, instancing, lights) and its
glTF exporter writes everything except the materials. Those are rebuilt here
from what the FBX names, with the rules threepp's old FBXLoader used:

  * SPECULAR slot: an ORM pack (R = occlusion, G = roughness, B = metalness,
    the Unreal / Falcor "_Specular" convention) becomes metallicRoughnessTexture
    with both factors at 1. `--material-mode` decides: auto guesses from the
    file name (_specular, _orm, _rma, ...), pbr always, phong never (the map is
    dropped; roughness comes from the FBX shininess as Blender converts it).
  * Glass (opacity < 0.99, or "glass" / "window" / "crystal" in the name):
    KHR_materials_transmission + KHR_materials_ior 1.5, double-sided, smooth
    unless an ORM map drives roughness.
  * Cutouts (foliage names, or a BC1 diffuse whose blocks use punch-through
    alpha): alphaMode MASK at 0.5, double-sided.
  * Emissive: map and/or colour, EmissiveFactor as KHR_materials_emissive_strength.

Textures are NOT re-encoded. Images point at the original files, relative to
the output, so the .gltf must stay beside the asset's texture folder (write it
next to the .fbx). DDS maps go through MSFT_texture_dds and stay
block-compressed with their mip chains; TGA keeps a plain uri, which threepp
decodes but strict glTF viewers will not.

Normal maps are written as they are. FBX assets from DirectX tools (Bistro)
carry green-down normals that glTF cannot flag; the loading code flips
normalScale.y (see examples/vulkan/vulkan_bistro.cpp).

Animation, skinning and morphs are not handled here; see mixamo_to_glb.py.
"""
import bpy, sys, os, json, struct, argparse, urllib.parse, addon_utils
from bpy_extras import node_shader_utils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="fbx_to_gltf.py")
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--material-mode", choices=("auto", "pbr", "phong"), default="auto")
args = ap.parse_args(argv)
src = os.path.abspath(args.src)
out = os.path.abspath(args.out)
if not out.lower().endswith(".gltf"):
    sys.exit("[fbx] output must be a .gltf (images are referenced, not embedded)")


def log(*a):
    print("[fbx]", *a); sys.stdout.flush()


for m in ("io_scene_fbx", "io_scene_gltf2"):
    try:
        addon_utils.enable(m, default_set=True, persistent=True)
    except Exception as e:
        log("enable", m, e)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Blender 5.0: io_scene_fbx still writes lamp.cycles.cast_shadow, which Cycles
# dropped, so any FBX with lights aborts the import. Same shim as prop_to_glb.py.
try:
    _probe = bpy.data.lights.new(name="_cast_shadow_shim", type='POINT')
    _cls = type(_probe.cycles)
    if not hasattr(_cls, "cast_shadow"):
        _cls.cast_shadow = bpy.props.BoolProperty(default=True)
    bpy.data.lights.remove(_probe)
except Exception as e:
    log("cycles cast_shadow shim", e)

log("import", src)
bpy.ops.import_scene.fbx(filepath=src, use_image_search=False)

src_dir = os.path.dirname(src)
out_dir = os.path.dirname(out)

# ---- texture paths -----------------------------------------------------------

_by_name = None
unresolved = set()


def resolve(img):
    """Where the image file is: as Blender resolved it, beside the .fbx, or
    anywhere below the .fbx's folder by file name (case-insensitive)."""
    global _by_name
    if img is None:
        return None
    p = bpy.path.abspath(img.filepath_raw or img.filepath)
    if p and os.path.isfile(p):
        return os.path.normpath(p)
    name = os.path.basename(p.replace("\\", "/"))
    if not name:
        return None
    if _by_name is None:
        _by_name = {}
        for root, _, files in os.walk(src_dir):
            for f in files:
                _by_name.setdefault(f.lower(), os.path.join(root, f))
    found = _by_name.get(name.lower())
    if found is None:
        unresolved.add(name)
    return found


def contains_any(s, words):
    s = s.lower()
    return any(w in s for w in words)


def stem(p):
    return os.path.splitext(os.path.basename(p))[0] if p else ""


ORM_HINTS = ("_specular", "_orm", "_rma", "_occlusionroughnessmetallic",
             "_roughness", "_metalrough", "_metallicrough")
GLASS_HINTS = ("glass", "window", "crystal", "transparent")
FOLIAGE_HINTS = ("foliage", "leaf", "leaves", "plant", "tree", "ivy", "hedge",
                 "flower", "grass", "fern", "vine", "bush", "shrub")


def bc1_has_cutout_alpha(path):
    """BC1 blocks with color0 <= color1 use the 3-colour + transparent mode.
    Measured on Bistro (2026-07-01): opaque textures stay under ~2% from
    endpoint ties, real cutouts (chairs 24%, barriers 31%, chains 14%) far
    above; 5% splits them. Other formats carry explicit alpha."""
    if not path or not path.lower().endswith(".dds"):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(128)
            if len(head) < 128 or head[:4] != b"DDS ":
                return False
            height, width = struct.unpack_from("<II", head, 12)
            pf_flags, fourcc = struct.unpack_from("<I4s", head, 80)
            if not (pf_flags & 0x4) or fourcc != b"DXT1":
                return False
            blocks = ((max(1, width) + 3) // 4) * ((max(1, height) + 3) // 4)
            data = f.read(blocks * 8)
    except OSError:
        return False
    if blocks == 0 or len(data) < blocks * 8:
        return False
    words = memoryview(data[:blocks * 8]).cast("H")  # c0, c1, 4 index bytes
    alpha = sum(1 for c0, c1 in zip(words[0::4], words[1::4]) if c0 <= c1)
    return alpha / blocks > 0.05


# ---- material survey (before export, while the FBX node trees are live) -----

survey = {}
for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    w = node_shader_utils.PrincipledBSDFWrapper(mat, is_readonly=True)
    if w.node_principled_bsdf is None:
        continue
    survey[mat.name] = dict(
        base_color=list(w.base_color)[:3],
        alpha=float(w.alpha),
        roughness=float(w.roughness),
        metallic=float(w.metallic),
        emission_color=list(w.emission_color)[:3],
        emission_strength=float(w.emission_strength),
        normal_strength=float(w.normalmap_strength),
        diffuse=resolve(w.base_color_texture.image),
        specular=resolve(w.specular_texture.image),
        normal=resolve(w.normalmap_texture.image),
        emissive=resolve(w.emission_color_texture.image),
    )
log("materials:", len(survey))

# ---- export: geometry, nodes, lights; materials as bare names ----------------

bpy.ops.export_scene.gltf(filepath=out, export_format='GLTF_SEPARATE',
                          export_image_format='NONE', export_materials='EXPORT',
                          export_yup=True, export_lights=True,
                          # Blender holds FBX Intensity / 100, which is what
                          # FBXLoader handed threepp; pass it through as is.
                          export_import_convert_lighting_mode='RAW',
                          export_cameras=False, export_animations=False,
                          export_extras=False)

# ---- rebuild the materials ---------------------------------------------------

with open(out, encoding="utf-8") as f:
    gltf = json.load(f)

images, textures, image_of = [], [], {}
# Blender's own material extensions go with its materials; keep the rest
# (KHR_lights_punctual, ...).
used = {e for e in gltf.get("extensionsUsed", []) if not e.startswith("KHR_materials_")}
required = {e for e in gltf.get("extensionsRequired", []) if not e.startswith("KHR_materials_")}


def texture_ref(path):
    if not path:
        return None
    if path not in image_of:
        uri = os.path.relpath(path, out_dir).replace("\\", "/")
        img = {"uri": urllib.parse.quote(uri)}
        tex = {}
        if path.lower().endswith(".dds"):
            img["mimeType"] = "image/vnd-ms.dds"
            # No PNG fallback beside it, so a reader without the
            # extension has nothing to show: required, not just used.
            tex["extensions"] = {"MSFT_texture_dds": {"source": len(images)}}
            used.add("MSFT_texture_dds"); required.add("MSFT_texture_dds")
        else:
            tex["source"] = len(images)
        images.append(img)
        image_of[path] = len(textures)
        textures.append(tex)
    return {"index": image_of[path]}


counts = dict(pbr=0, glass=0, cutout=0, emissive=0)
for md in gltf.get("materials", []):
    info = survey.get(md.get("name"))
    if info is None:
        continue
    name = md["name"]
    spec = info["specular"]
    if args.material_mode == "phong":
        is_pbr = False
    elif args.material_mode == "pbr":
        is_pbr = spec is not None
    else:
        is_pbr = spec is not None and contains_any(stem(spec), ORM_HINTS)
    alpha = info["alpha"]
    diffuse = info["diffuse"]
    is_glass = alpha < 0.99 or contains_any(name, GLASS_HINTS) \
        or contains_any(stem(diffuse), GLASS_HINTS[:3])
    cutout = diffuse is not None and not is_glass and (
        contains_any(name, FOLIAGE_HINTS) or contains_any(stem(diffuse), FOLIAGE_HINTS)
        or bc1_has_cutout_alpha(diffuse))

    pbr = {}
    base = info["base_color"] if diffuse is None else [1.0, 1.0, 1.0]
    opacity = alpha if (alpha < 0.99 and not is_glass) else 1.0
    pbr["baseColorFactor"] = [*base, opacity]
    if diffuse:
        pbr["baseColorTexture"] = texture_ref(diffuse)
    if is_pbr:
        pbr["metallicRoughnessTexture"] = texture_ref(spec)
        pbr["metallicFactor"] = 1.0
        pbr["roughnessFactor"] = 1.0
        counts["pbr"] += 1
    else:
        pbr["metallicFactor"] = info["metallic"]
        pbr["roughnessFactor"] = 0.0 if is_glass else info["roughness"]
    new = {"name": name, "pbrMetallicRoughness": pbr}

    if info["normal"]:
        new["normalTexture"] = texture_ref(info["normal"])
        if info["normal_strength"] != 1.0:
            new["normalTexture"]["scale"] = info["normal_strength"]

    ec, emap = info["emission_color"], info["emissive"]
    if emap or max(ec) > 0.0:
        if emap:
            new["emissiveTexture"] = texture_ref(emap)
            if max(ec) <= 0.0:
                ec = [1.0, 1.0, 1.0]
        strength = info["emission_strength"]
        peak = max(ec)
        if peak > 1.0:  # glTF caps the factor at 1; the rest goes to strength
            ec = [c / peak for c in ec]
            strength *= peak
        new["emissiveFactor"] = ec
        if strength != 1.0:
            new.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                "emissiveStrength": strength}
            used.add("KHR_materials_emissive_strength")
        counts["emissive"] += 1

    if is_glass:
        ext = new.setdefault("extensions", {})
        ext["KHR_materials_transmission"] = {
            "transmissionFactor": max(0.01, 1.0 - alpha) if alpha < 0.99 else 1.0}
        ext["KHR_materials_ior"] = {"ior": 1.5}
        used.update(("KHR_materials_transmission", "KHR_materials_ior"))
        new["doubleSided"] = True
        counts["glass"] += 1
    elif cutout:
        new["alphaMode"] = "MASK"
        new["alphaCutoff"] = 0.5
        new["doubleSided"] = True
        counts["cutout"] += 1
    elif opacity < 1.0:
        new["alphaMode"] = "BLEND"
        new["doubleSided"] = True

    md.clear()
    md.update(new)

for key in ("images", "textures", "samplers"):
    gltf.pop(key, None)
if images:
    gltf["images"] = images
    gltf["textures"] = textures
if used:
    gltf["extensionsUsed"] = sorted(used)
if required:
    gltf["extensionsRequired"] = sorted(required)

with open(out, "w", encoding="utf-8") as f:
    json.dump(gltf, f, separators=(",", ":"))

log("materials: %d (%d ORM, %d glass, %d cutout, %d emissive), images: %d, meshes: %d, lights: %d"
    % (len(gltf.get("materials", [])), counts["pbr"], counts["glass"], counts["cutout"],
       counts["emissive"], len(images), len(gltf.get("meshes", [])),
       len(gltf.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", []))))
if unresolved:
    log("textures not found (slots left empty):", ", ".join(sorted(unresolved)))
log("wrote", out)
