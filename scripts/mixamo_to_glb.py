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
from mathutils import Vector

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
    # The character file's own action is the bind pose sampled over two frames.
    # It is not a clip, and shipping it would put a 0.07 s nothing at the FRONT
    # of the exported list — which is exactly what "play the first clip" picks
    # up. Drop it; the rest pose the skin binds against comes from the
    # armature, not from an action.
    for a in (set(bpy.data.actions) - before):
        log("  - dropping the character's own bind action", a.name)
        bpy.data.actions.remove(a)
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

# --- measure what each clip actually travels ---------------------------------
# A clip is authored at ONE speed. Playing it at 1x while the character moves at
# some other speed is precisely what foot-sliding is, so the consumer needs the
# authored number — and the only honest source for it is the clip itself. Read
# here from the Hips bone's WORLD displacement across the clip (armature scale
# and orientation included), so the figures are metres per second in the same
# frame the exported GLB lands in.
#
# A turn clip gives itself away by the other column: near-zero travel and a
# large net yaw.
def measure_clips(arm):
    if arm is None:
        return
    root = next((b for b in arm.pose.bones if b.parent is None), None)
    if root is None:
        return
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    was = arm.animation_data.action if arm.animation_data else None
    if arm.animation_data is None:
        arm.animation_data_create()

    def sample(frame):
        scene.frame_set(int(round(frame)))
        bpy.context.view_layer.update()
        pb = arm.pose.bones[root.name]
        m = arm.matrix_world @ pb.matrix
        # Bone X in world, flattened: a stable heading reference for the hips.
        ax = m.to_3x3() @ Vector((1.0, 0.0, 0.0))
        return (arm.matrix_world @ pb.head), math.degrees(math.atan2(ax.y, ax.x))

    log(f"  {'clip':<26}{'dur s':>7}{'travel m':>10}{'m/s':>8}{'net yaw':>9}")
    for a in sorted(bpy.data.actions, key=lambda x: x.name):
        arm.animation_data.action = a
        try:
            if a.slots:
                arm.animation_data.action_slot = a.slots[0]
        except Exception:
            pass
        f0, f1 = a.frame_range
        dur = max((f1 - f0) / fps, 1e-6)
        p0, y0 = sample(f0)
        p1, y1 = sample(f1)
        d = p1 - p0
        travel = math.hypot(d.x, d.y)
        dyaw = (y1 - y0 + 180.0) % 360.0 - 180.0
        log(f"  {a.name:<26}{dur:7.2f}{travel:10.3f}{travel / dur:8.2f}{dyaw:8.0f}°")
    if was is not None:
        arm.animation_data.action = was


try:
    measure_clips(arm)
except Exception as e:
    log("clip measurement failed (harmless):", e)

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
