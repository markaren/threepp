"""LaRS panoptic labels for the harbour scene (phase C of plans/harbour-scene.md).

Every frame of the Mariner's camera_main comes with a panoptic mask in LaRS's classes, written
in LaRS's own on-disk format so a LaRS model or the LaRS evaluator reads it without conversion.

The LaRS format, as matched (read 2026-10-01 from the evaluator's code and the dataset page:
github.com/lojzezust/lars_evaluator (lars_eval/config.py, lars_eval/panoptic.py,
lars_eval/panopticapi.py, test/dataset/*.py) and lojzezust.github.io/lars-dataset):

  <split>/images/<name>.jpg                 the keyframe
  <split>/images_seq/<seq>_<%05d>.jpg       the keyframe and the 9 frames before it (frame_i - 0..9);
                                            <name> = <seq>_<%05d>, seq = everything before the last '_'
  <split>/panoptic_masks/<name>.png         RGB: R = category id, G * 256 + B = instance id
                                            (segment id = R + 256 G + 256^2 B, COCO's rgb2id);
                                            (0, 0, 0) = VOID (category 0): ignored by the evaluator
  <split>/semantic_masks/<name>.png         one channel: 0 obstacle (static + every thing class),
                                            1 water, 2 sky, 255 ignore
  <split>/panoptic_annotations.json         COCO panoptic: images [{id, file_name, width, height}],
                                            annotations [{image_id, file_name, segments_info:
                                            [{id, category_id, bbox [x, y, w, h], area, iscrowd}]}],
                                            categories [{id, name, supercategory, isthing, color}];
                                            one segment per stuff class per image
  <split>/image_list.txt                    one <name> per line

  categories: stuff 1 Static Obstacle, 3 Water, 5 Sky; things 11 Boat/ship, 12 Row boats,
  13 Paddle board, 14 Buoy, 15 Swimmer, 16 Animal, 17 Float, 19 Other (config.py DYN_OBST_IDS).

Not matched: LaRS's per-image scene attributes (image_annotations.json) are not written, the
category colours are ours (the evaluator's panoptic path does not read them), and LaRS's
image_id / instance numbering conventions are not documented: ours are an int per keyframe and
a STABLE instance id per object (the registry index + 1), so an instance id is also a track id.

Decisions (phase C, 2026-10-01): 1280x720 JPG (q 95) like LaRS's common size; a keyframe every
13 POV frames from frame 9 of each cut (82 keyframes, 820 sequence frames, no overlap); one
sequence per POV cut (harbour_c0 = camera_main over the film's chase window, when she runs west
along the quay past the moored sjarks; harbour_c1, harbour_c2, harbour_c3); the Mariner's own hull is VOID (LaRS's
ignore), never a boat; the sjarks and Trollfjord are Boat/ship, the marks and the mooring buoy
Buoy, the gillnet floats (blaase) Float; quay, fenders, bollards, mooring lines, the mole and the
whole town are one Static Obstacle segment; the ocean sheet and the far ring are Water, so the
mirrored boats and marks in it are water too.

Ids: renderer.set_instance_id(mesh, id) keys a single Object3D (the G-buffer's stable id is
looked up per drawn MESH, VulkanCoreImpl::stableIdForObject), so a glb's hierarchy is labelled by
walking it and tagging every mesh. Ids we assign live at LABEL_BASE + code, far above the
renderer's auto-numbering (1, 2, 3, ... for every unlabelled mesh it draws: the terrain tiles,
buildings, roads, trees GeoScene streams in), so any id below LABEL_BASE is the town: Static
Obstacle. 0 is sky.
"""
import json
import math
import os

import numpy as np

import threepp as tp

LABEL_BASE = 1 << 24
C_STATIC, C_WATER, C_VOID, C_REG = 1, 3, 9, 1000          # codes above LABEL_BASE

VOID, STATIC, WATER, SKY = 0, 1, 3, 5
BOAT, ROWBOAT, PADDLE, BUOY, SWIMMER, ANIMAL, FLOAT, OTHER = 11, 12, 13, 14, 15, 16, 17, 19
CATEGORIES = [  # (id, name, supercategory, isthing, colour: ours, for the film and the overlays)
    (STATIC, "Static Obstacle", "obstacle", 0, (247, 195, 37)),
    (WATER, "Water", "water", 0, (41, 167, 224)),
    (SKY, "Sky", "sky", 0, (90, 75, 164)),
    (BOAT, "Boat/ship", "obstacle", 1, (230, 57, 70)),
    (ROWBOAT, "Row boats", "obstacle", 1, (255, 120, 160)),
    (PADDLE, "Paddle board", "obstacle", 1, (255, 170, 90)),
    (BUOY, "Buoy", "obstacle", 1, (60, 220, 90)),
    (SWIMMER, "Swimmer", "obstacle", 1, (255, 255, 255)),
    (ANIMAL, "Animal", "obstacle", 1, (150, 90, 40)),
    (FLOAT, "Float", "obstacle", 1, (255, 90, 255)),
    (OTHER, "Other", "obstacle", 1, (200, 200, 200)),
]
CAT = {c[0]: c for c in CATEGORIES}
REG_CLASS = {"boat": BOAT, "buoy": BUOY, "float": FLOAT, "static_obstacle": STATIC}
SEMANTIC = {STATIC: 0, WATER: 1, SKY: 2, VOID: 255}           # things -> 0 (obstacle)
VOID_COLOUR = (25, 25, 25)

# the class lookup for the codes above LABEL_BASE (rebuilt by assign_ids)
_LUT = {}


def _meshes(node):
    out = []
    node.traverse(lambda o: out.append(o) if isinstance(o, tp.Mesh) else None)
    return out


def assign_ids(renderer, S):
    """Tag every mesh the scene draws outside GeoScene. Call once the mooring lines exist (after
    the first step_floats); idempotent. Returns how many meshes fell to the generic static tag
    (the bollards and the mooring lines; anything else there is a labelling gap)."""
    global _LUT
    lut = {C_STATIC: (STATIC, 0, "static"), C_WATER: (WATER, 0, "water"), C_VOID: (VOID, 0, "own_vessel")}
    labelled = set()
    n_static = [0]

    def tag(node, code):
        for m in _meshes(node):
            renderer.set_instance_id(m, LABEL_BASE + code)
            labelled.add(m.id)
    S.inst = []                               # (instance id, category, name, node) for the things
    for k, (node, cls, name) in enumerate(S.registry):
        if node is S.mariner["obj"]:
            continue                          # her own hull: below
        cat = REG_CLASS[cls]
        code = C_REG + k
        lut[code] = (cat, k + 1 if cat != STATIC else 0, name)
        tag(node, code)
        S.inst.append((k + 1, cat, name, node))
    tag(S.mariner["obj"], C_VOID)             # LaRS: the own vessel is ignore (VOID), never a boat
    tag(S.ocean, C_WATER)
    tag(S.far, C_WATER)                       # the far sea ring beyond the sheet
    for o in S.scene.children:                # the bollards, the mooring lines (meshes not yet tagged)
        if o is S.geo:
            continue
        for m in _meshes(o):
            if m.id not in labelled:
                renderer.set_instance_id(m, LABEL_BASE + C_STATIC)
                labelled.add(m.id)
                n_static[0] += 1
    _LUT = lut
    return n_static[0]


def decode(ids):
    """(H, W) uint32 stable ids -> (cat (H, W) uint8 LaRS category, inst (H, W) uint16, 0 for
    stuff and void)."""
    cat = np.full(ids.shape, STATIC, np.uint8)            # auto ids: GeoScene's town
    inst = np.zeros(ids.shape, np.uint16)
    cat[ids == 0] = SKY
    hi = ids >= LABEL_BASE
    if hi.any():
        codes, inv = np.unique(ids[hi] - LABEL_BASE, return_inverse=True)
        cc = np.array([_LUT.get(int(c), (VOID, 0, "?"))[0] for c in codes], np.uint8)
        ii = np.array([_LUT.get(int(c), (VOID, 0, "?"))[1] for c in codes], np.uint16)
        cat[hi] = cc[inv]
        inst[hi] = ii[inv]
    return cat, inst


def panoptic_rgb(cat, inst):
    out = np.zeros(cat.shape + (3,), np.uint8)
    out[..., 0] = cat
    out[..., 1] = (inst >> 8).astype(np.uint8)
    out[..., 2] = (inst & 255).astype(np.uint8)
    return out


def semantic(cat):
    sem = np.zeros(cat.shape, np.uint8)                    # obstacle
    sem[cat == WATER] = 1
    sem[cat == SKY] = 2
    sem[cat == VOID] = 255
    return sem


def segments(cat, inst):
    """COCO panoptic segments_info (VOID left out, as LaRS does)."""
    seg = cat.astype(np.int64) + 256 * (inst >> 8).astype(np.int64) + 65536 * (inst & 255).astype(np.int64)
    out = []
    for sid in np.unique(seg):
        c = int(sid % 256)
        if c == VOID:
            continue
        ys, xs = np.nonzero(seg == sid)
        x0, y0 = int(xs.min()), int(ys.min())
        out.append({"id": int(sid), "category_id": c,
                    "bbox": [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
                    "area": int(len(xs)), "iscrowd": 0,
                    "instance_id": int(inst[ys[0], xs[0]])})
    return out


def colour_mask(cat, inst):
    """The film's mask panel: class colours, VOID dark, things of >= 400 px outlined in white."""
    lut = np.zeros((256, 3), np.uint8)
    for c in CATEGORIES:
        lut[c[0]] = c[4]
    lut[VOID] = VOID_COLOUR
    rgb = lut[cat]
    key = cat.astype(np.int32) * 65536 + inst.astype(np.int32)
    edge = np.zeros(cat.shape, bool)
    edge[:-1] |= key[:-1] != key[1:]
    edge[1:] |= key[:-1] != key[1:]
    edge[:, :-1] |= key[:, :-1] != key[:, 1:]
    edge[:, 1:] |= key[:, :-1] != key[:, 1:]
    # outline the things big enough to keep their colour inside the line (a 3 px spar stays green)
    area = np.bincount(inst.ravel(), minlength=1)
    big = (cat >= 10) & (area[inst] >= 400)
    rgb[edge & big] = (255, 255, 255)
    return rgb


class LarsWriter:
    """Writes one split in LaRS's layout under root (a temp dir the caller renames)."""

    def __init__(self, root, W, H):
        self.root, self.W, self.H = root, W, H
        for d in ("images", "images_seq", "panoptic_masks", "semantic_masks"):
            os.makedirs(os.path.join(root, d), exist_ok=False)
        self.images, self.anns, self.names = [], [], []
        self.stats_seg = []

    def seq_frame(self, seq, i, rgb):
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(self.root, "images_seq", f"{seq}_{i:05d}.jpg"), quality=95)

    def keyframe(self, seq, i, rgb, cat, inst, meta):
        from PIL import Image
        name = f"{seq}_{i:05d}"
        Image.fromarray(rgb).save(os.path.join(self.root, "images", name + ".jpg"), quality=95)
        Image.fromarray(panoptic_rgb(cat, inst)).save(os.path.join(self.root, "panoptic_masks", name + ".png"))
        Image.fromarray(semantic(cat)).save(os.path.join(self.root, "semantic_masks", name + ".png"))
        segs = segments(cat, inst)
        iid = len(self.images) + 1
        self.images.append({"id": iid, "file_name": name + ".jpg", "width": self.W, "height": self.H, **meta})
        self.anns.append({"image_id": iid, "file_name": name + ".png",
                          "segments_info": [{k: s[k] for k in ("id", "category_id", "bbox", "area", "iscrowd")}
                                            for s in segs]})
        self.names.append(name)
        for s in segs:
            self.stats_seg.append({"image": name, **s})
        return name, segs

    def close(self, inst_names):
        cats = [{"id": c[0], "name": c[1], "supercategory": c[2], "isthing": c[3], "color": list(c[4])}
                for c in CATEGORIES]
        doc = {"info": {"description": "threepp harbour scene (Alesund), synthetic, LaRS panoptic format",
                        "version": "harbour-pC"},
               "licenses": [], "images": self.images, "annotations": self.anns, "categories": cats}
        with open(os.path.join(self.root, "panoptic_annotations.json"), "w") as f:
            json.dump(doc, f)
        with open(os.path.join(self.root, "image_list.txt"), "w") as f:
            f.write("\n".join(self.names) + "\n")
        return self.stats(inst_names)

    def stats(self, inst_names):
        """Per-class instance counts and pixel areas over the export; per-instance bbox sizes."""
        out = {"keyframes": len(self.names), "image_size": [self.W, self.H], "classes": {}, "instances": {}}
        for c in CATEGORIES:
            ss = [s for s in self.stats_seg if s["category_id"] == c[0]]
            if not ss:
                continue
            e = {"segments": len(ss), "pixels": int(sum(s["area"] for s in ss)),
                 "frac_of_pixels": sum(s["area"] for s in ss) / float(self.W * self.H * max(1, len(self.names)))}
            if c[3]:
                e["distinct_instances"] = len({s["instance_id"] for s in ss})
                ws = np.array([s["bbox"][2] for s in ss])
                hs = np.array([s["bbox"][3] for s in ss])
                e["bbox_w_px"] = [int(ws.min()), int(np.median(ws)), int(ws.max())]
                e["bbox_h_px"] = [int(hs.min()), int(np.median(hs)), int(hs.max())]
                e["bbox_diag_px_min_median_max"] = [round(float(v), 1) for v in
                                                    np.percentile(np.hypot(ws, hs), [0, 50, 100])]
            out["classes"][c[1]] = e
        for s in self.stats_seg:
            if s["instance_id"] == 0:
                continue
            nm = inst_names.get(s["instance_id"], str(s["instance_id"]))
            d = out["instances"].setdefault(nm, {"category": CAT[s["category_id"]][1], "keyframes": 0, "bboxes_wh": [],
                                                 "areas": []})
            d["keyframes"] += 1
            d["bboxes_wh"].append(s["bbox"][2:])
            d["areas"].append(s["area"])
        return out


# --------------------------------------------------------------------------- #
#  The film panel: POV | mask, with a legend and live per-class counts
# --------------------------------------------------------------------------- #
def _font(size):
    from PIL import ImageFont
    for f in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def film_panel(rgb, cat, inst, title, pw=960, ph=540):
    from PIL import Image, ImageDraw
    a = Image.fromarray(rgb).resize((pw, ph), Image.LANCZOS)
    # the mask is resampled NEAREST from the full-size label image, then outlined at panel size
    H, W = cat.shape
    yy = (np.arange(ph) * H / ph + 0.5 * H / ph).astype(int)
    xx = (np.arange(pw) * W / pw + 0.5 * W / pw).astype(int)
    cs, isub = cat[yy][:, xx], inst[yy][:, xx]
    b = Image.fromarray(colour_mask(cs, isub))
    out = Image.new("RGB", (2 * pw, ph))
    out.paste(a, (0, 0))
    out.paste(b, (pw, 0))
    d = ImageDraw.Draw(out)
    f, fs = _font(18), _font(15)
    d.rectangle([0, 0, pw, 30], fill=(0, 0, 0))
    d.text((10, 5), title, font=f, fill=(255, 255, 255))
    d.rectangle([pw, 0, 2 * pw, 30], fill=(0, 0, 0))
    d.text((pw + 10, 5), "panoptic mask, LaRS classes (synthetic, free)", font=f, fill=(255, 255, 255))
    # live counts: thing instances visible in this frame (full-res label, >= 1 px)
    counts = {}
    for c in (BOAT, BUOY, FLOAT):
        counts[c] = len(np.unique(inst[(cat == c)])) if (cat == c).any() else 0
    x0, y0 = 2 * pw - 306, 42                # top right of the mask: sky in every POV cut (pC v2: top
                                             # left clipped Trollfjord's superstructure in c1)
    d.rectangle([x0 - 6, y0 - 6, x0 + 300, y0 + 112], fill=(0, 0, 0))
    rows = [(BOAT, f"Boat/ship  {counts[BOAT]}"), (BUOY, f"Buoy  {counts[BUOY]}"), (FLOAT, f"Float  {counts[FLOAT]}"),
            (STATIC, "Static obstacle"), (WATER, "Water"), (SKY, "Sky"), (VOID, "Ignore (own bow)")]
    for j, (c, txt) in enumerate(rows):
        x, y = x0 + (j // 4) * 160, y0 + (j % 4) * 28
        col = VOID_COLOUR if c == VOID else CAT[c][4]
        d.rectangle([x, y + 3, x + 16, y + 19], fill=tuple(col), outline=(255, 255, 255))
        d.text((x + 22, y + 1), txt, font=fs, fill=(255, 255, 255))
    return np.asarray(out)


def overlay_triptych(rgb, cat, inst, title):
    """RGB | mask | 50 % overlay, full size."""
    from PIL import Image, ImageDraw
    m = colour_mask(cat, inst)
    ov = (0.5 * rgb.astype(np.float32) + 0.5 * m.astype(np.float32)).astype(np.uint8)
    H, W = cat.shape
    out = Image.new("RGB", (3 * W, H + 34))
    for j, im in enumerate((rgb, m, ov)):
        out.paste(Image.fromarray(im), (j * W, 34))
    d = ImageDraw.Draw(out)
    d.text((8, 6), title + "   |   RGB   |   mask   |   50 % overlay", font=_font(20), fill=(255, 255, 255))
    return out


def project(cam_pos, cam_quat, fov_deg, W, H, p):
    """World point -> (u, v, view z) for a camera looking down its local -Z (threepp)."""
    x, y, z, w = cam_quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    q = R.T @ (np.asarray(p, float) - np.asarray(cam_pos, float))
    if q[2] >= -0.1:
        return None
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2.0)
    return 0.5 * W + f * q[0] / -q[2], 0.5 * H - f * q[1] / -q[2], -q[2]


# --------------------------------------------------------------------------- #
#  The export: the POV cuts of the film's own sim, LaRS keyframes every N frames
# --------------------------------------------------------------------------- #
N_KEY = 13           # c1..c3: 780 POV frames (8 + 8 + 10 s at 30 fps) -> 18 + 18 + 23 = 59 keyframes;
                     # c0 (the chase window, 10 s): 23 more
N_SEQ = 9            # LaRS: the 9 frames before each keyframe
FILM_W, FILM_H = 960, 540


class Exporter:
    """The labels hook of run_film. Per POV frame: one read of RGB + ids (+ depth at keyframes)
    from the same rendered frame, the film panel piped to ffmpeg; at keyframes the LaRS files and
    the gate data."""

    def __init__(self, hs, renderer, S, camera, W, H, split_dir, outdir):
        self.hs, self.r, self.S, self.cam, self.W, self.H = hs, renderer, S, camera, W, H
        self.outdir = outdir
        # c0: camera_main over the chase window (cast-off, west along the quay past the moored
        # sjarks). The cut's own times and 30 fps are kept, so the sim stays the film's.
        self.pov_over = {"b_chase": "c0_pov_quay_run"}
        self.pov = [c for c in hs.CUTS if c[3] is hs._film_pov or c[0] in self.pov_over]
        self.t_end = self.pov[-1][2]
        self.final_dir = split_dir
        self.tmp_dir = os.path.join(os.path.dirname(split_dir), "_tmp_" + os.path.basename(split_dir))
        self.writer = LarsWriter(self.tmp_dir, W, H)
        self.assigned = False
        self.ring = {}                 # seq -> {frame index: rgb} (the last N_SEQ frames)
        self.written_seq = set()
        self.idx = {}                  # seq -> next frame index
        self.keys = []                 # (name, t, cut, overlay triptych)
        self.gate1 = {}                # registry name -> expected / seen / expected_not_seen
        self.gate2 = []                # per keyframe: own-hull pixels and their labels
        self.track = []                # per POV frame: mark centroids, mask and RGB
        self.film = None
        self.film_n = 0

    # ---- ids
    def assign(self):
        n = assign_ids(self.r, self.S)
        self.assigned = True
        self.names = {k + 1: name for k, (_, _, name) in enumerate(self.S.registry)}
        print(f"[labels] ids assigned: {len(self.S.registry) - 1} registry objects + own vessel void, "
              f"{n} meshes on the generic static tag (15 bollards + 10 mooring lines expected)")

    # ---- film
    def _film_open(self):
        import subprocess
        import imageio_ffmpeg
        k = 1
        while os.path.exists(os.path.join(self.outdir, f"harbour_labels_film_v{k}.mp4")):
            k += 1
        self.film_path = os.path.join(self.outdir, f"harbour_labels_film_v{k}.mp4")
        self.film_tmp = os.path.join(self.outdir, f"_tmp_harbour_labels_film_v{k}.mp4")
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{2 * FILM_W}x{FILM_H}", "-framerate", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "18", "-preset", "medium", "-threads", "1", self.film_tmp]
        self.film = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def close(self):
        if self.film is not None:
            self.film.stdin.close()
            self.film.wait()
            os.replace(self.film_tmp, self.film_path)
        st = self.writer.close(dict(self.names))
        os.replace(self.tmp_dir, self.final_dir)
        return st

    # ---- one POV frame
    def pov_frame(self, t, cut):
        hs, S, r, cam = self.hs, self.S, self.r, self.cam
        r.sim_time = hs.T_BASE + t
        cp = cam.position
        S.geo.update(cp)
        r.set_fog_water_surface_y(float(S.ocean.sample_height(cp.x, cp.z)))
        cname = self.pov_over.get(cut[0], cut[0])
        seq = "harbour_" + cname.split("_")[0]             # harbour_c0, harbour_c1, harbour_c2, harbour_c3
        i = self.idx.get(seq, 0)
        self.idx[seq] = i + 1
        key = i >= N_SEQ and (i - N_SEQ) % N_KEY == 0
        a = r.read_aovs_typed(S.scene, cam, ["rgb", "instance_ids"] + (["depth"] if key else []))
        rgb, ids = np.ascontiguousarray(a["rgb"]), a["instance_ids"]
        cat, inst = decode(ids)
        self._track(cname, i, rgb, cat, inst)
        ring = self.ring.setdefault(seq, {})
        ring[i] = rgb
        for j in [j for j in ring if j < i - N_SEQ]:
            del ring[j]
        if key:
            for j in range(i - N_SEQ, i + 1):
                if (seq, j) not in self.written_seq:
                    self.writer.seq_frame(seq, j, ring[j])
                    self.written_seq.add((seq, j))
            name, segs = self.writer.keyframe(seq, i, rgb, cat, inst, {"sim_time": round(t, 4), "cut": cname})
            self._gate1(name, ids, a["depth"])
            self._gate2(name, ids, cat)
            self.keys.append((name, t, cname, overlay_triptych(rgb, cat, inst, f"{name}  t={t:.2f} s")))
        if self.film is None:
            self._film_open()
        title = f"camera_main   {cname}   sim t {t:5.2f} s"
        self.film.stdin.write(np.ascontiguousarray(film_panel(rgb, cat, inst, title, FILM_W, FILM_H)).tobytes())
        self.film_n += 1

    # ---- gate 1: every registry instance in the frustum and not occluded appears in a keyframe
    def _gate1(self, name, ids, depth):
        cam = self.cam
        cam.update_matrix_world()
        wp, wq = cam.get_world_position(), cam.get_world_quaternion()
        pos, quat = (wp.x, wp.y, wp.z), (wq.x, wq.y, wq.z, wq.w)
        present = set(np.unique(ids[ids >= LABEL_BASE] - LABEL_BASE).tolist())
        for k, (node, cls, nm) in enumerate(self.S.registry):
            if node is self.S.mariner["obj"]:
                continue
            b = tp.Box3()
            b.set_from_object(node)
            bmin, bmax = b.min(), b.max()
            lo, hi = np.array([bmin.x, bmin.y, bmin.z]), np.array([bmax.x, bmax.y, bmax.z])
            lo[1] = max(lo[1], 0.15)                          # above the sea: the waterline and up
            g = self.gate1.setdefault(nm, {"class": cls, "expected": 0, "seen": 0, "expected_not_seen": []})
            seen = (C_REG + k) in present
            g["seen"] += int(seen)
            if hi[1] <= lo[1]:
                continue
            ext = hi - lo
            tol = 0.5 * min(ext[0], ext[2]) + 0.5
            vis = False
            for fx in np.linspace(0.15, 0.85, 5):
                for fy in np.linspace(0.1, 0.9, 5):
                    for fz in np.linspace(0.15, 0.85, 5):
                        q = project(pos, quat, cam.fov, self.W, self.H, lo + ext * np.array([fx, fy, fz]))
                        if q is None:
                            continue
                        u, v, z = q
                        if not (1 <= u < self.W - 1 and 1 <= v < self.H - 1) or z > 4000.0:
                            continue
                        if depth[int(v), int(u)] >= z - tol:
                            vis = True
                            break
                    if vis:
                        break
                if vis:
                    break
            if vis:
                g["expected"] += 1
                if not seen:
                    g["expected_not_seen"].append(name)

    # ---- gate 2: no pixel of the Mariner's own hull is labelled boat (render her hidden, diff)
    def _gate2(self, name, ids, cat):
        mar = self.S.mariner["obj"]
        mar.visible = False
        ids2 = self.r.read_instance_ids(self.S.scene, self.cam)
        mar.visible = True
        # the two reads are separate renders with different TAA jitter, so every silhouette in the
        # frame shifts by a sub-pixel and the raw diff carries 1-2 px bands along all edges (pC v2:
        # ~2.5k px of town, sky and Trollfjord edges per keyframe). A 3x3 opening keeps her bow (a
        # ~7k px blob) and drops those bands; her own silhouette edge goes with them.
        # pC v3: 2-3 px jitter clusters still survived the opening in dense detail (Trollfjord's
        # railings, the town), ~4 px per keyframe; her hull is the components of >= 300 px.
        from scipy.ndimage import binary_opening, label as cc_label
        raw = ids != ids2
        opened = binary_opening(raw, structure=np.ones((3, 3), bool))
        comp, n = cc_label(opened)
        size = np.bincount(comp.ravel(), minlength=n + 1)
        size[0] = 0
        own = size[comp] >= 300
        lab = {(CAT[int(c)][1] if int(c) in CAT else "Ignore (void)"): int(((cat == c) & own).sum())
               for c in np.unique(cat[own])}
        self.gate2.append({"keyframe": name, "own_hull_px": int(own.sum()), "raw_diff_px": int(raw.sum()),
                           "labels_on_own_hull": lab, "thing_px_on_own_hull": int(((cat >= 10) & own).sum())})

    # ---- gate 3 data: a moving mark's mask centroid against its colour in the RGB
    def _track(self, cutname, i, rgb, cat, inst):
        rec = {"cut": cutname, "i": i}
        for nm, col in (("lateral_stbd", "green"), ("lateral_port", "red")):
            iid = next((k + 1 for k, (_, _, n) in enumerate(self.S.registry) if n == nm), None)
            m = inst == iid
            if m.sum() < 60:
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), max(xs.min() - 20, 0), min(xs.max() + 20, self.W - 1)
            win = rgb[y0:y1 + 1, x0:x1 + 1].astype(np.int32)
            R, G, B = win[..., 0], win[..., 1], win[..., 2]
            sel = (G > R + 25) & (G > B + 5) if col == "green" else (R > G + 40) & (R > B + 30)
            if sel.sum() < 30:
                continue
            rec[nm] = (float(xs.mean()), float(np.nonzero(sel)[1].mean() + x0), int(m.sum()))
        self.track.append(rec)


def gate3(track):
    """No one-frame lag: the RGB colour centroid of a moving mark must sit on the SAME frame's mask
    centroid, not the previous or next frame's (frames where the mark moves > 2 px)."""
    out = {}
    for nm in ("lateral_stbd", "lateral_port"):
        e0, em, ep, mv, n = [], [], [], [], 0
        for a, b, c in zip(track, track[1:], track[2:]):
            if not (nm in a and nm in b and nm in c) or not (a["cut"] == b["cut"] == c["cut"]):
                continue
            xm0, xm1, xm2 = a[nm][0], b[nm][0], c[nm][0]
            if abs(xm1 - xm0) < 2.0:
                continue
            xr = b[nm][1]
            mv.append(abs(xm1 - xm0))
            e0.append(abs(xr - xm1))
            em.append(abs(xr - xm0))
            ep.append(abs(xr - xm2))
            n += 1
        if n:
            out[nm] = {"frames": n, "mean_motion_px": round(float(np.mean(mv)), 2),
                       "err_same_frame_px": round(float(np.mean(e0)), 2),
                       "err_vs_previous_frame_px": round(float(np.mean(em)), 2),
                       "err_vs_next_frame_px": round(float(np.mean(ep)), 2),
                       "pass": bool(np.mean(e0) < min(np.mean(em), np.mean(ep)) and np.mean(e0) < 1.5)}
    return out
