"""
Convert a static prop (dae / fbx / glb / obj) into an optimised GLB for threepp:
textures downscaled to POT <= 1024 and JPEG-packed (POT matters for the GL
backend's texture upload; JPEG keeps the file small).

  blender --background --python prop_to_glb.py -- <src> <out.glb>

Prints the world-space bounding-box size so the caller knows the prop's native
scale (used to size it in-hand).
"""
import bpy, sys, os, math, addon_utils, mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
src, out = argv[0], argv[1]
TARGET, QUALITY = 1024, 90

for m in ("io_scene_fbx", "io_scene_gltf2"):
    try:
        addon_utils.enable(m, default_set=True, persistent=True)
    except Exception as e:
        print("[prop] enable", m, e)

def log(*a):
    print("[prop]", *a); sys.stdout.flush()

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(src)[1].lower()
log("import", ext, src)
if ext == ".dae":
    bpy.ops.wm.collada_import(filepath=src)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=src)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=src)
else:
    log("unsupported extension", ext); sys.exit(1)

# downscale textures to POT
def pot(n):
    n = min(n, TARGET)
    return max(64, 1 << int(round(math.log2(max(1, n)))))

any_alpha = False
for img in list(bpy.data.images):
    if not img.has_data or img.size[0] == 0:
        continue
    w, h = img.size
    if img.depth == 32:
        any_alpha = True
    nw, nh = pot(w), pot(h)
    log(f"image {img.name}: {w}x{h} depth{img.depth}" + (f" -> {nw}x{nh}" if (nw, nh) != (w, h) else ""))
    if (nw, nh) != (w, h):
        img.scale(nw, nh)
fmt = 'AUTO' if any_alpha else 'JPEG'

# report world bbox so the caller knows the native size
mins = [1e9] * 3
maxs = [-1e9] * 3
for o in bpy.context.scene.objects:
    if o.type == 'MESH':
        for v in o.bound_box:
            wv = o.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                mins[i] = min(mins[i], wv[i]); maxs[i] = max(maxs[i], wv[i])
size = [round(maxs[i] - mins[i], 4) for i in range(3)]
log("meshes:", sum(1 for o in bpy.context.scene.objects if o.type == 'MESH'),
    "| bbox size (x,y,z):", size, "| tex format:", fmt)

bpy.ops.export_scene.gltf(filepath=out, export_format='GLB',
                          export_image_format=fmt, export_jpeg_quality=QUALITY,
                          export_yup=True)
log("wrote", out, os.path.getsize(out), "bytes")
