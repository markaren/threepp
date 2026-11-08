"""Graphics research scenes (Cornell box, Veach MIS, BMW M6, ...) fetched on demand.

    python research_scenes.py --list                   # what the release holds, what is cached
    python research_scenes.py                          # viewer: pick a scene, it downloads on first use
    python research_scenes.py bmw-m6                   # open one scene straight away
    python research_scenes.py veach-mis --variant core
    python research_scenes.py cornell-box --shot out.png   # one headless frame
    python research_scenes.py --fetch kitchen crown    # just fill the cache
    python research_scenes.py --gl                     # GLRenderer instead of Vulkan

Scenes are picked in an ImGui panel. A threepp build without ImGui (tp.HAS_IMGUI
False) opens the first scene and steps through the rest with LEFT / RIGHT; V
switches core / extended and C returns to the file's camera.

The scenes are Erfan Momeni's glTF 2.0 conversions of the classic PBRT research
scenes (https://github.com/ErfanMo77/gltf-research-scenes). Every release of
that repository is one zip of about a gigabyte: per scene a `core` GLB (plain
metallic-roughness), an `extended` GLB (clearcoat, transmission, volume, IOR,
specular, punctual lights), the environment HDR, the licences and a Cycles
reference render. Nobody wants the whole gigabyte to look at a Cornell box, so
this demo never downloads it. GitHub serves release assets with HTTP byte
ranges: the listing reads only the zip's central directory (a few hundred
kilobytes at the end of the file), and opening a scene pulls just that scene's
entries out of the middle of the zip, checked against the CRC32 the archive
records for each one.

Everything lands in $THREEPP_MODEL_CACHE/gltf-research-scenes/<release tag>/
(default ~/.cache/threepp/models/...), laid out as in the repository, so a
scene is downloaded once and opens offline from then on. `ResearchScenes` is
the reusable part, plain standard library:

    rs = ResearchScenes()
    for s in rs.scenes(): print(s.id, s.variants)
    glb = rs.fetch("cornell-box")               # Path to the cached .glb

The scenes keep their own licences (CC0 or CC BY 3.0, per scene); each fetch
puts the scene's licence files beside its GLB.
"""
import argparse
import json
import math
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

REPO = "ErfanMo77/gltf-research-scenes"
ASSET = "gltf_research_scenes_glb.zip"
CACHE_ROOT = Path(os.environ.get("THREEPP_MODEL_CACHE",
                                 Path.home() / ".cache" / "threepp" / "models")) / "gltf-research-scenes"
ZIP_PREFIX = "gltf_research_scenes/"     # every entry sits under this folder in the zip
RELEASE_TTL = 24 * 3600                  # how long a resolved release is trusted before asking GitHub again
USER_AGENT = "threepp-research-scenes"


def _get(url, start=None, timeout=60):
    """urlopen with our User-Agent, optionally from byte `start` to the end."""
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise IOError(f"stream ended {n - len(buf)} bytes early")
        buf += chunk
    return bytes(buf)


class _RangeFile:
    """A read-only, seekable file over HTTP byte ranges, enough for zipfile.

    A read continues the open response while reads are sequential and opens a
    new ranged GET when zipfile seeks elsewhere. GitHub answers explicit and
    open-ended ranges but not suffix ranges (bytes=-N), so the size comes from
    the release metadata and every request names its start byte.
    """

    def __init__(self, url, size):
        self.url, self.size = url, size
        self.pos = 0
        self._resp, self._resp_pos = None, -1
        self.requests = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self._resp is None or self._resp_pos != self.pos:
            self.close()
            self._resp = _get(self.url, self.pos)
            self._resp_pos = self.pos
            self.requests += 1
        data = _read_exact(self._resp, n)
        self.pos += n
        self._resp_pos = self.pos
        return data

    def close(self):
        if self._resp is not None:
            self._resp.close()
        self._resp = None


@dataclass
class Scene:
    id: str
    variants: dict                                  # "core" / "extended" / "glb" -> zip entry name
    files: list = field(default_factory=list)       # every entry of this scene in the zip

    @property
    def default_variant(self):
        return "extended" if "extended" in self.variants else next(iter(self.variants))


class ResearchScenes:
    """The scenes of one release of gltf-research-scenes, fetched entry by entry."""

    def __init__(self, release="latest", cache_root=CACHE_ROOT, log=print):
        self.cache_root = Path(cache_root)
        self.log = log
        self.release = self._resolve_release(release)
        self.dir = self.cache_root / self.release["tag"]
        self._index = self._load_index()
        self._scenes = None
        self._manifest_path = self.dir / "manifest.json"
        self._manifest = self._read_json(self._manifest_path) or {}
        self._lock = threading.Lock()

    # --- release + index -------------------------------------------------------

    @staticmethod
    def _read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(path)

    def _resolve_release(self, release):
        """Tag, asset URL, size and upload time of the release zip; cached for a day,
        and the cached answer is used as is when GitHub cannot be reached."""
        memo = self.cache_root / f"release-{release}.json"
        cached = self._read_json(memo)
        if cached and time.time() - cached.get("checked", 0) < RELEASE_TTL:
            return cached
        api = f"https://api.github.com/repos/{REPO}/releases/" + \
              ("latest" if release == "latest" else f"tags/{release}")
        try:
            with _get(api, timeout=20) as r:
                rel = json.load(r)
        except (urllib.error.URLError, OSError) as e:
            if cached:
                self.log(f"[research-scenes] GitHub unreachable ({e}); using release {cached['tag']} from the cache")
                return cached
            offline = self._newest_cached_tag() if release == "latest" else None
            if offline:
                self.log(f"[research-scenes] GitHub unreachable ({e}); using cached release {offline['tag']}")
                return offline
            raise
        asset = next((a for a in rel.get("assets", []) if a["name"] == ASSET), None)
        if asset is None:
            raise RuntimeError(f"release {rel.get('tag_name')} of {REPO} has no {ASSET}")
        info = {"tag": rel["tag_name"], "url": asset["browser_download_url"], "size": asset["size"],
                "updated": asset["updated_at"], "checked": time.time()}
        self._write_json(memo, info)
        return info

    def _newest_cached_tag(self):
        best = None
        for idx in self.cache_root.glob("*/index.json"):
            data = self._read_json(idx)
            if data and (best is None or data["release"]["updated"] > best["updated"]):
                best = data["release"]
        return best

    def _load_index(self):
        """Every entry of the zip: offset, sizes, CRC. Built once per release upload
        from the central directory, read over a handful of range requests."""
        path = self.dir / "index.json"
        data = self._read_json(path)
        rel = self.release
        if data and data["release"]["updated"] == rel["updated"] and data["release"]["size"] == rel["size"]:
            return data["entries"]
        self.log(f"[research-scenes] reading the {rel['tag']} index ({rel['size'] / 2**20:.0f} MB zip, "
                 f"central directory only)")
        rf = _RangeFile(rel["url"], rel["size"])
        try:
            with zipfile.ZipFile(rf) as zf:
                entries = {i.filename: {"offset": i.header_offset, "csize": i.compress_size,
                                        "size": i.file_size, "crc": i.CRC, "method": i.compress_type}
                           for i in zf.infolist() if not i.is_dir()}
        finally:
            rf.close()
        self.log(f"[research-scenes] {len(entries)} entries, {rf.requests} range requests")
        self._write_json(path, {"release": {k: rel[k] for k in ("tag", "url", "size", "updated")},
                                "entries": entries})
        return entries

    # --- listing ---------------------------------------------------------------

    def scenes(self):
        """The release's scenes, sorted by id."""
        if self._scenes is None:
            self._scenes = self._group_scenes()
        return self._scenes

    def _group_scenes(self):
        by_id = {}
        for name in self._index:
            parts = name[len(ZIP_PREFIX):].split("/")
            if len(parts) < 3 or parts[0] != "scenes" or parts[1] == "common":
                continue
            sid = parts[1]
            s = by_id.setdefault(sid, Scene(sid, {}))
            s.files.append(name)
            if parts[2] == "gltf" and name.endswith(".glb"):
                stem = Path(name).stem
                variant = stem.rsplit("_", 1)[1] if stem.endswith(("_core", "_extended")) else "glb"
                s.variants[variant] = name
        return [by_id[k] for k in sorted(by_id) if by_id[k].variants]

    def scene(self, scene_id):
        for s in self.scenes():
            if s.id == scene_id:
                return s
        raise KeyError(f"no scene '{scene_id}' in {self.release['tag']}; "
                       f"have: {', '.join(s.id for s in self.scenes())}")

    def size(self, scene_id, variant=None):
        """Bytes a fetch of this scene + variant writes (the GLB and its environment map)."""
        s = self.scene(scene_id)
        names = [s.variants[variant or s.default_variant]]
        names += [n for n in s.files if "_environment" in n and n.endswith(".hdr")]
        return sum(self._index[n]["size"] for n in names)

    def is_cached(self, scene_id, variant=None):
        s = self.scene(scene_id)
        return self._is_cached(s.variants[variant or s.default_variant])

    def local_path(self, entry):
        return self.dir / entry[len(ZIP_PREFIX):]

    def _is_cached(self, entry):
        p = self.local_path(entry)
        e = self._index[entry]
        return p.is_file() and p.stat().st_size == e["size"] and self._manifest.get(entry) == e["crc"]

    # --- fetching --------------------------------------------------------------

    def fetch(self, scene_id, variant=None, progress=None):
        """Path to the scene's GLB in the cache, downloading what is missing: the GLB,
        the environment map its extras name, and the scene's licence files.
        `progress(file_name, done_bytes, total_bytes)` is called while each file downloads."""
        s = self.scene(scene_id)
        variant = variant or s.default_variant
        if variant not in s.variants:
            raise KeyError(f"{scene_id} has no '{variant}' variant; it has {', '.join(s.variants)}")
        glb_entry = s.variants[variant]
        for n in [glb_entry] + [n for n in s.files if "license" in Path(n).name.lower()]:
            if not self._is_cached(n):
                self._extract(n, progress)
        glb = self.local_path(glb_entry)
        for env in environment_maps(glb):    # known only once the GLB is here
            for n in (self._entry_beside(glb_entry, env["uri"]),
                      self._entry_beside(glb_entry, env.get("license_uri"))):
                if n and not self._is_cached(n):
                    self._extract(n, progress)
        return glb

    def _entry_beside(self, entry, rel):
        """The zip entry a URI relative to `entry`'s folder points at (None if absent)."""
        if not rel:
            return None
        parts = []
        for p in (Path(entry).parent.as_posix() + "/" + rel).split("/"):
            if p == "..":
                parts.pop()
            elif p not in ("", "."):
                parts.append(p)
        name = "/".join(parts)
        return name if name in self._index else None

    def _extract(self, entry, progress=None):
        """Stream one zip entry out of the remote zip into the cache, CRC-checked."""
        e = self._index[entry]
        dest = self.local_path(entry)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        self.log(f"[research-scenes] fetching {entry[len(ZIP_PREFIX):]} ({e['size'] / 2**20:.1f} MB)")
        if e["method"] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RuntimeError(f"{entry}: zip method {e['method']} not supported")
        with _get(self.release["url"], e["offset"]) as r:
            head = _read_exact(r, 30)
            sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", head)
            if sig != 0x04034B50:
                raise RuntimeError(f"{entry}: no local file header at {e['offset']}; the release changed? "
                                   f"delete {self.dir / 'index.json'}")
            _read_exact(r, name_len + extra_len)
            inflate = zlib.decompressobj(-15) if e["method"] == zipfile.ZIP_DEFLATED else None
            crc, written, left = 0, 0, e["csize"]
            with open(tmp, "wb") as f:
                while left:
                    chunk = r.read(min(left, 1 << 20))
                    if not chunk:
                        raise IOError(f"{entry}: connection closed with {left} bytes to go")
                    left -= len(chunk)
                    out = inflate.decompress(chunk) if inflate else chunk
                    if inflate and not left:
                        out += inflate.flush()
                    crc = zlib.crc32(out, crc)
                    written += len(out)
                    f.write(out)
                    if progress:
                        progress(dest.name, written, e["size"])
        if crc != e["crc"] or written != e["size"]:
            tmp.unlink(missing_ok=True)
            raise IOError(f"{entry}: CRC/size mismatch (got {crc:08x}/{written}, zip says {e['crc']:08x}/{e['size']})")
        tmp.replace(dest)
        with self._lock:
            self._manifest[entry] = e["crc"]
            self._write_json(self._manifest_path, self._manifest)


# --- what the GLB says about itself -------------------------------------------


def glb_json(path):
    """The JSON chunk of a .glb."""
    with open(path, "rb") as f:
        magic, _version, _length, chunk_len, chunk_type = struct.unpack("<IIIII", f.read(20))
        if magic != 0x46546C67 or chunk_type != 0x4E4F534A:
            raise ValueError(f"{path} is not a GLB")
        return json.loads(f.read(chunk_len))


def environment_maps(glb):
    """The environment maps the file records in its extras (glTF has no environment
    light). Each: uri relative to the GLB, licence uri, and where given intensity_scale
    and rotation_degrees. The PBRT conversions put them under the top-level
    extras.pbrt.light_sources; the Blender export (classroom2) under its scene's
    extras.environment_map, at a strength of 8."""
    doc = glb_json(glb)
    pbrt = (doc.get("extras") or {}).get("pbrt", {})
    envs = [ls["environment_map"] for ls in pbrt.get("light_sources", []) if "environment_map" in ls]
    for sc in doc.get("scenes", []):
        env = (sc.get("extras") or {}).get("environment_map")
        if env and all(e.get("uri") != env.get("uri") for e in envs):
            envs.append(env)
    return envs


def _node_matrix(node):
    if "matrix" in node:
        return [node["matrix"][c * 4 + r] for r in range(4) for c in range(4)]   # column-major -> rows
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    sx, sy, sz = node.get("scale", (1, 1, 1))
    tx, ty, tz = node.get("translation", (0, 0, 0))
    return [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y - z * w) * sy, 2 * (x * z + y * w) * sz, tx,
            2 * (x * y + z * w) * sx, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z - x * w) * sz, ty,
            2 * (x * z - y * w) * sx, 2 * (y * z + x * w) * sy, (1 - 2 * (x * x + y * y)) * sz, tz,
            0, 0, 0, 1]


def _mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def gltf_camera(glb):
    """The file's first perspective camera, or None: a dict of position, forward, up (world),
    yfov (degrees), aspect (the reference render's, when the file records it) and mirrored.
    threepp's GLTFLoader does not build glTF cameras, so the demo reads the node itself."""
    doc = glb_json(glb)
    nodes = doc.get("nodes", [])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    for i, n in enumerate(nodes):
        if "camera" not in n:
            continue
        cam = doc["cameras"][n["camera"]]
        if cam.get("type") != "perspective":
            continue
        m, j = _node_matrix(n), i
        while j in parent:
            j = parent[j]
            m = _mul(_node_matrix(nodes[j]), m)
        col = lambda c: [m[r * 4 + c] for r in range(3)]
        norm = lambda v: [x / (math.sqrt(sum(y * y for y in v)) or 1) for x in v]
        x, y, z = col(0), col(1), col(2)
        det = (x[0] * (y[1] * z[2] - y[2] * z[1]) - y[0] * (x[1] * z[2] - x[2] * z[1])
               + z[0] * (x[1] * y[2] - x[2] * y[1]))
        p = cam["perspective"]
        res = doc.get("extras", {}).get("pbrt", {}).get("render", {}).get("resolution")
        return {"position": col(3), "forward": norm([-v for v in z]), "up": norm(y),
                "yfov": math.degrees(p["yfov"]),
                "aspect": p.get("aspectRatio") or (res[0] / res[1] if res else None),
                # Some conversions keep PBRT's image flip (Scale -1 1 1) as a mirrored camera
                # node. Cycles renders through it; a threepp camera cannot be mirrored, so
                # those scenes show their reference image flipped left to right.
                "mirrored": det < 0}
    return None


# --- command line: list / fetch -----------------------------------------------


def mb(n):
    return f"{n / 2**20:.1f} MB" if n < 10 * 2**20 else f"{n / 2**20:.0f} MB"


def print_listing(rs):
    print(f"{REPO} {rs.release['tag']}  ->  cache {rs.dir}")
    for s in rs.scenes():
        marks = [f"{v}{'*' if rs.is_cached(s.id, v) else ''} {mb(rs.size(s.id, v))}" for v in s.variants]
        print(f"  {s.id:<20} {'   '.join(marks)}")
    print("  (* = cached; sizes leave out shared environment maps)")


def text_progress(name, done, total, _last={"t": 0.0}):
    """A one-line console progress bar for files of a megabyte or more."""
    if total < 2**20:
        return
    now = time.time()
    if now - _last["t"] > 0.5 or done >= total:
        _last["t"] = now
        print(f"\r  {name}: {mb(done)} / {mb(total)}", end="\n" if done >= total else "", flush=True)


# --- viewer -------------------------------------------------------------------

BACKDROP = 0x101014


class Viewer:
    """One scene at a time in a threepp scene, swapped on request.

    A scene that is not cached downloads on a worker thread while the window keeps
    drawing; `frame()` (what the window's loop calls) loads it on the main thread
    once it is there, since building the meshes makes GPU resources.
    """

    def __init__(self, tp, rs, canvas, renderer, vulkan, ui=None, controls=True):
        self.tp, self.rs, self.canvas, self.renderer, self.vulkan, self.ui = tp, rs, canvas, renderer, vulkan, ui
        self.scenes = rs.scenes()
        self.ids = [s.id for s in self.scenes]
        # The reference renders are Cycles with Khronos PBR Neutral at exposure 1.
        renderer.tone_mapping = tp.ToneMapping.Neutral
        renderer.tone_mapping_exposure = 1.0

        self.scene = tp.Scene()
        self.scene.background = BACKDROP
        if not vulkan:
            # GL has no emitter lighting: the area lights of these scenes are emissive
            # meshes, so without a fill most interiors would be black.
            self.scene.add(tp.HemisphereLight(0xffffff, 0x303038, 0.6))
        self.camera = tp.PerspectiveCamera(40, canvas.aspect(), 0.01, 1000)
        self.controls = tp.OrbitControls(self.camera, canvas) if controls else None
        if self.controls:
            self.controls.enable_damping = True

        self.model = None
        self.loaded = None          # (scene id, variant) on screen
        self.env_name = None
        self.home = None            # (camera dict or None, bbox centre, bbox radius)
        self.fetching = None        # scene id downloading on the worker thread
        self.progress = ("", 0, 0)  # (file, bytes done, bytes total) of that download
        self.pending = None         # (scene id, variant, glb path) waiting for the main thread
        self.error = None
        self.pick = 0               # panel state
        self.variant = 0
        self.exposure = 1.0

    # --- loading ---------------------------------------------------------------

    def request(self, scene_id, variant=None):
        """Show this scene: at once if cached, else after a background download."""
        if self.fetching:
            return False
        s = self.rs.scene(scene_id)
        variant = variant or s.default_variant
        self.error = None
        if self.rs.is_cached(scene_id, variant):
            # fetch() still fills in a missing environment map or licence (small files)
            self.pending = (scene_id, variant, self.rs.fetch(scene_id, variant))
            return True
        self.fetching = scene_id
        self.progress = ("", 0, self.rs.size(scene_id, variant))

        def report(name, done, total):
            self.progress = (name, done, total)
            if not self.ui:                 # no panel to show it in
                text_progress(name, done, total)

        def work():
            try:
                glb = self.rs.fetch(scene_id, variant, progress=report)
                self.pending = (scene_id, variant, glb)
            except Exception as e:      # shown in the panel; the viewer keeps running
                self.error = f"{scene_id}: {e}"
            finally:
                self.fetching = None
        threading.Thread(target=work, daemon=True).start()
        return True

    def pump(self):
        """Load a scene whose files have arrived (main thread)."""
        if self.pending:
            scene_id, variant, glb = self.pending
            self.pending = None
            try:
                self.show(scene_id, variant, glb)
            except Exception as e:
                self.error = f"{scene_id}: {e}"

    def show(self, scene_id, variant, glb):
        tp = self.tp
        t0 = time.time()
        result = tp.GLTFLoader().load(str(glb))
        if self.model is not None:
            self.scene.remove(self.model)
        self.model = result.scene
        # threepp's GLTFLoader hides the lights a file brings (the caller opts in).
        # These scenes' lights are part of them: the extended PBRT conversions'
        # directional sun, classroom2's window sunlight.
        self.model.traverse(lambda o: setattr(o, "visible", True)
                            if type(o).__name__ in ("DirectionalLight", "SpotLight", "PointLight") else None)
        self.scene.add(self.model)
        envs = environment_maps(glb)
        if envs:
            # RGBELoader hands back an equirect map, and the converter wrote the HDR in that
            # layout already in glTF's frame (top row +Y, centre column +X), so no rotation.
            env = tp.RGBELoader().load(str((glb.parent / envs[0]["uri"]).resolve()))
            self.scene.environment = env
            self.scene.background = env
            # The strength the reference render used for its world (classroom2: 8), on
            # the light and the backdrop alike, as Cycles' world strength is.
            strength = float(envs[0].get("intensity_scale", 1.0))
            self.scene.environment_intensity = strength
            self.scene.background_intensity = strength
            rx, ry, rz = (math.radians(a) for a in envs[0].get("rotation_degrees", (0.0, 0.0, 0.0)))
            for rot in (self.scene.environment_rotation, self.scene.background_rotation):
                rot.set(rx, ry, rz)
        else:
            self.scene.environment = None
            self.scene.background = BACKDROP
            self.scene.environment_intensity = 1.0
            self.scene.background_intensity = 1.0
        self.env_name = Path(envs[0]["uri"]).name if envs else None
        box = tp.Box3().set_from_object(self.model)
        c, sz = box.get_center(), box.get_size()
        radius = 0.5 * math.sqrt(sz.x ** 2 + sz.y ** 2 + sz.z ** 2) or 1.0
        cam = gltf_camera(glb)
        self.home = (cam, (c.x, c.y, c.z), radius)
        self.reset_camera()
        self.loaded = (scene_id, variant)
        print(f"[research-scenes] {scene_id} ({variant}) loaded in {time.time() - t0:.1f} s"
              + (f", environment {self.env_name}" if envs else "")
              + (", camera mirrored in the file: the view is the reference flipped left to right"
                 if cam and cam["mirrored"] else ""))

    def reset_camera(self):
        """The file's own camera (the reference render's view), else a three-quarter view."""
        cam, center, radius = self.home
        camera = self.camera
        camera.near = max(radius * 1e-4, 1e-3)
        camera.far = radius * 50
        if cam:
            pos, fwd = cam["position"], cam["forward"]
            camera.fov = cam["yfov"]
            camera.up.set(*cam["up"])
            camera.position.set(*pos)
            # Orbit about the point straight ahead at the depth of the scene's centre.
            d = max(sum((c - p) * f for c, p, f in zip(center, pos, fwd)), radius * 0.05)
            target = [p + f * d for p, f in zip(pos, fwd)]
        else:
            camera.fov = 40
            camera.up.set(0, 1, 0)
            target = center
            camera.position.set(center[0] + radius * 1.6, center[1] + radius * 0.8, center[2] + radius * 1.6)
        camera.look_at(*target)
        camera.update_projection_matrix()
        if self.controls:
            self.controls.target = self.tp.Vector3(*target)
            self.controls.update()

    def on_key(self, key, mods=0):
        """Keys for a build without ImGui: LEFT / RIGHT step through the scenes,
        V switches core / extended, C goes back to the file's camera."""
        if key in ("LEFT", "RIGHT"):
            if self.fetching:
                print(f"[research-scenes] still downloading {self.fetching}")
                return
            self.pick = (self.pick + (1 if key == "RIGHT" else -1)) % len(self.scenes)
            s = self.scenes[self.pick]
            self.variant = list(s.variants).index(s.default_variant)
            self.request(s.id)
        elif key == "V" and self.loaded:
            variants = list(self.rs.scene(self.loaded[0]).variants)
            if len(variants) > 1 and not self.fetching:
                self.variant = (variants.index(self.loaded[1]) + 1) % len(variants)
                self.request(self.loaded[0], variants[self.variant])
        elif key == "C" and self.home:
            self.reset_camera()

    # --- per frame -------------------------------------------------------------

    def frame(self):
        self.pump()
        if self.controls:
            self.controls.enabled = not (self.ui and self.ui.want_capture_mouse)
            self.controls.update()
        self.renderer.render(self.scene, self.camera)
        if self.ui:
            self.ui.render(self.draw_ui)

    def draw_ui(self):
        im = self.tp.imgui
        rs = self.rs
        im.set_next_window_pos(10, 10)
        im.set_next_window_size(360, 0, True)    # height 0 + always: fit the content every frame
        im.begin(f"Research scenes {rs.release['tag']}")
        labels = [f"{s.id}{'  *' if rs.is_cached(s.id) else ''}" for s in self.scenes]
        changed, self.pick = im.combo("scene", self.pick, labels)
        s = self.scenes[self.pick]
        variants = list(s.variants)
        if changed:
            self.variant = variants.index(s.default_variant)
        self.variant = min(self.variant, len(variants) - 1)
        _, self.variant = im.combo("variant", self.variant, variants)
        variant = variants[self.variant]
        if self.fetching:
            name, done, total = self.progress
            im.text(f"downloading {name or self.fetching}")
            im.text(f"  {mb(done)} / {mb(total)}")
        elif self.loaded == (s.id, variant):
            im.text("on screen")
        elif im.button("open" if rs.is_cached(s.id, variant) else f"download {mb(rs.size(s.id, variant))} and open"):
            self.request(s.id, variant)
        if self.error:
            im.text(self.error[:120])
        im.separator()
        changed, self.exposure = im.slider_float("exposure", self.exposure, 0.05, 8.0)
        if changed:
            self.renderer.tone_mapping_exposure = self.exposure
        if self.home and im.button("reference camera"):
            self.reset_camera()
        if self.loaded:
            im.text(f"{self.loaded[0]} / {self.loaded[1]}")
            if self.env_name:
                im.text(f"environment {self.env_name}")
            if self.home[0] and self.home[0]["mirrored"]:
                im.text("mirrored file camera: L/R flipped vs ref")
        im.text("* = cached")
        im.text(f"{im.get_framerate():.0f} fps")
        im.end()


def run_viewer(rs, args):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import threepp as tp
    from demo_common import resize_handler

    ids = [s.id for s in rs.scenes()]
    first = args.scene or "cornell-box"
    if first not in ids:
        sys.exit(f"no scene '{first}'; --list shows what there is")
    vulkan = tp.HAS_VULKAN and not args.gl

    if args.shot:
        glb = rs.fetch(first, args.variant, progress=text_progress)
        cam = gltf_camera(glb)
        h = 720                            # the reference render's aspect at 720 rows
        w = int(round(h * ((cam and cam["aspect"]) or 16 / 9) / 2)) * 2
        canvas = tp.Canvas("research scenes", width=w, height=h, headless=True, vsync=False)
        renderer = tp.VulkanRenderer(canvas) if vulkan else tp.GLRenderer(canvas)
        v = Viewer(tp, rs, canvas, renderer, vulkan, controls=False)
        v.show(first, args.variant or rs.scene(first).default_variant, glb)
        for _ in range(args.frames):        # let TAA, the denoiser and probe GI settle
            v.frame()
        if vulkan:
            renderer.save_frame(v.scene, v.camera, args.shot)   # renders one more frame and reads it back
        else:
            renderer.save_frame(args.shot)
        print(f"wrote {args.shot}")
        return

    canvas = tp.Canvas("threepp - research scenes", width=1280, height=800, vsync=False)
    renderer = tp.VulkanRenderer(canvas) if vulkan else tp.GLRenderer(canvas)
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None   # after the renderer
    v = Viewer(tp, rs, canvas, renderer, vulkan, ui=ui)
    canvas.on_window_resize(resize_handler(v.camera, renderer))
    v.pick = ids.index(first)
    s = rs.scene(first)
    v.variant = list(s.variants).index(args.variant or s.default_variant)
    if ui is None:
        # Without the panel the keys pick scenes, so open one to start from.
        print("[research-scenes] this threepp build has no ImGui: LEFT / RIGHT = previous / next scene, "
              "V = core / extended, C = the file's camera")
        canvas.on_key_pressed(v.on_key)
        v.request(first, args.variant)
    elif args.scene:
        v.request(first, args.variant)
    canvas.animate(v.frame)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scene id to open (see --list)")
    ap.add_argument("--list", action="store_true", help="list the release's scenes and what is cached")
    ap.add_argument("--fetch", nargs="+", metavar="ID", help="download these scenes into the cache and exit")
    ap.add_argument("--variant", help="core, extended, or glb (classroom2); default extended")
    ap.add_argument("--release", default="latest", help="release tag, e.g. v1.1.0 (default: latest)")
    ap.add_argument("--shot", metavar="PNG", help="render one headless frame of `scene` and exit")
    ap.add_argument("--frames", type=int, default=48, help="frames rendered before --shot saves (default 48)")
    ap.add_argument("--gl", action="store_true", help="use the GLRenderer even when Vulkan is available")
    args = ap.parse_args()

    rs = ResearchScenes(args.release)
    if args.list:
        print_listing(rs)
        return
    if args.fetch:
        for sid in args.fetch:
            print(rs.fetch(sid, args.variant, progress=text_progress))
        return
    run_viewer(rs, args)


if __name__ == "__main__":
    main()
