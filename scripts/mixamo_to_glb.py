"""
Merge a Mixamo "character + separate animation FBX" pack into ONE GLB whose
animations are all bound to the single character skeleton, so threepp's
GLTFLoader + AnimationMixer can drive them (same path as Soldier.glb).

Usage (headless):
  blender --background --python mixamo_to_glb.py -- <src_dir> <out.glb> [character.fbx]

If [character.fbx] is omitted, runs in DRY-RUN mode: imports only the animation
files (skeleton-only), keeps the first armature, and exports a mesh-less GLB.
This validates the multi-action merge + glTF export quickly, without paying for
the (large) character import.

Why automatic_bone_orientation=False: it preserves the RAW Mixamo bone matrices,
which are byte-identical across every export of the same rig. That makes an
action authored in one file play correctly on the armature from another file
(the action's data paths are pose.bones["mixamorig:..."], rest pose must match).
"""

import bpy, sys, os, glob, json, struct, math, addon_utils

# Textures are downscaled to <= this (snapped to a power of two) and JPEG-packed.
# Mixamo characters ship 2K PNGs; at TPS distance 1K is ample, and POT sizes
# avoid the GL backend's non-power-of-two upload crash.
TARGET_TEX = 1024
JPEG_QUALITY = 90

# --- args after the "--" separator ------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    print("[mixamo] ERROR: need <src_dir> <out.glb> [character.fbx]")
    sys.exit(1)
src_dir = argv[0]
out_glb = argv[1]
character = argv[2] if len(argv) > 2 else None

IMPORT_KW = dict(automatic_bone_orientation=False, use_anim=True, ignore_leaf_bones=False)


def log(*a):
    print("[mixamo]", *a)
    sys.stdout.flush()


# --- make sure the IO addons are available ----------------------------------
for mod in ("io_scene_fbx", "io_scene_gltf2"):
    try:
        addon_utils.enable(mod, default_set=True, persistent=True)
    except Exception as e:
        log("addon enable", mod, "->", e)

log("blender", bpy.app.version_string)
log("src_dir", src_dir)
log("out_glb", out_glb)
log("character", character or "(dry-run, anims only)")

# --- clean slate -------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

arm = None
mesh_objs = []

# --- import the character (mesh + skeleton), if given ------------------------
if character:
    path = os.path.join(src_dir, character)
    before = set(bpy.data.actions)
    log("importing character (this is the slow one) ...", path)
    bpy.ops.import_scene.fbx(filepath=path, **IMPORT_KW)
    for o in bpy.context.selected_objects:
        if o.type == 'ARMATURE':
            arm = o
        elif o.type == 'MESH':
            mesh_objs.append(o)
    for a in (set(bpy.data.actions) - before):
        a.name = "base_" + os.path.splitext(character)[0]
        a.use_fake_user = True
    log("character armature:", arm.name if arm else None,
        "| meshes:", [m.name for m in mesh_objs])
    if arm:
        bones = [b.name for b in arm.data.bones]
        log("bones:", len(bones), "sample:", bones[:6])

# --- import every animation file, capture + rename its action ----------------
collected = []
for fbx in sorted(glob.glob(os.path.join(src_dir, "*.fbx"))):
    base = os.path.basename(fbx)
    if character and base == character:
        continue
    before = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=fbx, **IMPORT_KW)
    imported = list(bpy.context.selected_objects)
    new_actions = list(set(bpy.data.actions) - before)
    clip = os.path.splitext(base)[0]
    for i, a in enumerate(new_actions):
        a.name = clip if i == 0 else f"{clip}.{i}"
        a.use_fake_user = True
        collected.append(a.name)
        fr = a.frame_range
        log(f"  + {a.name}  frames [{fr[0]:.0f}, {fr[1]:.0f}]")
    # In dry-run, keep the first imported armature as the export target.
    if arm is None:
        for o in imported:
            if o.type == 'ARMATURE':
                arm = o
                break
    # Delete every imported object except the armature we are keeping; the
    # actions survive because use_fake_user is set. (Skeleton-only Mixamo files
    # carry no mesh, so this is usually just stray armatures / empties.)
    for o in imported:
        if o is arm:
            continue
        bpy.data.objects.remove(o, do_unlink=True)

log("collected", len(collected), "clips")
log("scene objects at export:", [o.name for o in bpy.context.scene.objects])
log("actions in file:", len(bpy.data.actions))

# --- shrink textures: downscale to POT <= TARGET_TEX -------------------------
def pot(n):
    n = min(n, TARGET_TEX)
    return max(64, 1 << int(round(math.log2(max(1, n)))))

any_alpha = False
for img in list(bpy.data.images):
    if not img.has_data or img.size[0] == 0:
        log("  image", img.name, "(no data, skipped)")
        continue
    w, h = img.size
    if img.depth == 32:
        any_alpha = True
    nw, nh = pot(w), pot(h)
    log(f"  image {img.name}: {w}x{h} depth{img.depth}" + (f" -> {nw}x{nh}" if (nw, nh) != (w, h) else ""))
    if (nw, nh) != (w, h):
        img.scale(nw, nh)

# Force JPEG when nothing needs alpha (big win); keep source formats otherwise.
img_format = 'AUTO' if any_alpha else 'JPEG'
log("texture export format:", img_format, "(any_alpha=%s)" % any_alpha)

# --- export: one glTF animation per action -----------------------------------
# The scene now holds only the export targets, so use_selection is unnecessary.
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format='GLB',
    export_animations=True,
    export_animation_mode='ACTIONS',
    export_force_sampling=True,
    export_yup=True,
    export_skins=True,
    export_apply=False,
    export_image_format=img_format,
    export_jpeg_quality=JPEG_QUALITY,
)
log("exported", out_glb, os.path.getsize(out_glb), "bytes")

# --- verify: parse the GLB JSON chunk and list animations --------------------
with open(out_glb, "rb") as f:
    data = f.read()
_magic, _ver, _len = struct.unpack_from("<III", data, 0)
clen, _ctype = struct.unpack_from("<II", data, 12)
gltf = json.loads(data[20:20 + clen].decode("utf-8"))
anims = [a.get("name", "?") for a in gltf.get("animations", [])]
log(f"GLB animations ({len(anims)}):", anims)
log("GLB meshes:", len(gltf.get("meshes", [])),
    "| skins:", len(gltf.get("skins", [])),
    "| images:", len(gltf.get("images", [])))
log("DONE")
