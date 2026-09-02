"""The LOOK of the Calico Tanks demo: sky, sun, contact shadow.

`spot_calico.py` owns the simulation; this module owns everything that makes the
robot belong to the rock it stands on. Three problems, three answers:

SKY  The splats carry no sky. The scan ships one -- the `env/` chunk, 76k splats
     on a 2 km sphere -- but `from_sog_lod` skips it (it is not a LOD level), and
     it is a photographic panorama band: 88 % of its splats sit below 20 deg
     elevation, its zenith is nearly empty, and its bottom half is the ground of
     the capture, which would draw a second, wrong floor under the trail. So the
     default here is threepp's own procedural sky, `warp_common.sky_env`'s maths
     rebuilt for a Z-UP world (that function's elevation is world +Y), used as
     both `scene.background` and `scene.environment`. `sky="env"` loads the scan's
     own sphere as a second `SplatCloud` for the A/B.
     TRIED, and the A/B is not close: shots/wp4_envsky_lookback.png. The sky it
     gives IS the deep blue SuperSplat shows, and it buries everything else --
     76k Gaussians up to 355 m across drawn at 1 km smear the canyon into a dome,
     and the trail with the robot on it survives as a 30-pixel speck. It is a
     viewer's backdrop, not a scene element.

SUN  The splats are unlit -- their sun is baked in -- so the scene's one
     DirectionalLight (EnvSunPolicy: a scene DirectionalLight claims the sun)
     has to point where the scan's sun pointed, or Spot is lit from one side and
     the canyon from the other and the robot reads as pasted on.
     MEASURED, not guessed: `sun_direction()` documents the shading fit that
     produced SUN_DIR_ZUP below.

SHADOW  The splats receive nothing: they are not geometry, no shadow map and no
     ray-traced occluder touches them. The renderer has no shadow-catcher /
     receive-only material either (`ShadowMaterial` exists in the GL path only;
     `grep -ri shadowmaterial src/threepp/renderers/vulkan` is empty). So the
     contact shadow is FAKE and deliberate: `ContactShadows` puts a soft dark
     disc under each foot and one under the body, each sized and faded by how far
     that foot is above the probed floor.

Nothing in here builds anything -- import it, call `apply_look`, keep the handle.
"""
import math
import os

import numpy as np

import threepp as tp

# ── the sun ───────────────────────────────────────────────────────────────────
# Elevation 65 deg, azimuth -25 deg in WP0's post-flip Y-up frame; see
# sun_direction() for how that was measured. Z-up world = (x, -z, y) of the Y-up
# vector, the same map Frame.dir applies in spot_calico.py.
SUN_ELEV_DEG = 65.0
SUN_AZ_DEG = -25.0
SUN_DIR_YUP = (0.3830, 0.9063, -0.1786)
SUN_DIR_ZUP = (0.3830, 0.1786, 0.9063)

SUN_COLOR = 0xfff1d8
SUN_INTENSITY = 3.1
SKY_HEMI_TOP = 0xbcd2f2
SKY_HEMI_BOTTOM = 0x8c7a63
SKY_HEMI_INTENSITY = 0.35

# ── exposure ──────────────────────────────────────────────────────────────────
# PINNED (auto-exposure off), and MEASURED, not dialled by eye. The first build
# ran at 1.05 and the slab the robot stands on came out at mean luma 189.6/255
# against WP2's 173.7 -- brighter than the frame it was meant to match, with the
# left cliff on the way to blown. `look_harness.py --exposures 0.75,0.9,1.05`
# renders one scene at each and measures the slab (the centre-bottom quarter of
# the lookback frame) and the whole frame's 99th percentile:
#
#     exposure   slab mean   frame p99
#       0.75       169.9       212.0     <- chosen
#       0.90       180.9       219.9
#       1.05       189.6       225.2
#       (WP2 reference, shots/wp2_lookback.png: 173.7 / 234.7)
#
# 0.75 and 0.90 both land in the 165-185 window with p99 well clear of 250;
# 0.75 is the one that keeps the cliff's banding readable and leaves the most
# headroom, and it sits within 4 levels of WP2's slab. Captures:
# shots/wp4b_exposure_ab.png.
EXPOSURE = 0.75


def sun_direction():
    """The scan's baked sun as a unit vector TOWARD the sun, in the Z-up world.

    METHOD (a shading fit on the splats themselves, not an eyeball of a render):
    decode a coarse level of the SOG directly (level 5, 597k splats), keep the
    93k inside the trail corridor with opacity > 0.6, and give each splat a
    normal -- a Gaussian squashed onto a surface has its THIN axis along the
    surface normal, so the normal is the rotation applied to the axis of the
    smallest scale. Keep the flat ones (smin/smid < 0.4, 70k), sign each normal
    into the upper hemisphere, and fit

        luminance  ~  ambient + gain * max(0, n . s)

    by least squares over a 5-degree grid of sun directions s. Best fit:
    elevation 65 deg, azimuth -25 deg, ambient 0.35, gain 0.30.

    Independent check, same data: mean luminance binned by normal azimuth peaks
    at -60..0 deg (0.61) and bottoms at +90..+150 (0.40) -- the same azimuth the
    fit chose, from a statistic that knows nothing about the fit.
    Third check, by eye: in shots/wp2_lookback.png the shadows fall away from the
    upper left. That camera's right vector is (-0.32, -0.94, 0) and
    SUN_DIR_ZUP . right = -0.29, so this sun sits high and to the LEFT of that
    view, which is where the scan's shadows say it is.
    """
    return SUN_DIR_ZUP


# ── sky ───────────────────────────────────────────────────────────────────────
def sky_texture(sun_dir=SUN_DIR_ZUP, w=1024, h=512,
                horizon=(0.50, 0.62, 0.84), zenith=(0.06, 0.20, 0.56),
                below_horizon=(0.55, 0.50, 0.44), below_nadir=(0.24, 0.20, 0.16),
                sun_gain=42.0, glow_gain=2.6,
                haze=(0.13, 0.12, 0.10), haze_width_deg=2.6):
    """`warp_common.sky_env`, rebuilt for a Z-UP world.

    sky_env computes elevation from the direction's +Y component because every
    caller so far (hull sculpt, prop vortex) is a Y-up scene. This world is Z-up,
    so the only change that matters is which component is "up" -- the texel ->
    direction parameterisation is the renderer's and is left exactly as it was.
    The lower hemisphere is desert haze, not sea: it is what the rock's own
    image-based light picks up from below the horizon.
    """
    sun_dir = np.asarray(sun_dir, np.float32)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    # Texel -> WORLD direction. float_texture's row 0 is the ZENITH end of the
    # equirect (its docstring says so) while the parameterisation above puts
    # elevation -90 on row 0, so the map's vertical axis is flipped relative to
    # the renderer's: the world direction of a texel is d with its pole
    # component negated. MEASURED, not assumed -- built the other way round the
    # first render came out with the desert haze overhead and the deep blue
    # under the horizon.
    D = np.stack([d[..., 0], -d[..., 1], d[..., 2]], axis=-1)
    up_c = D[..., 2]                     # world +Z is up in this scene
    up = np.clip(up_c, 0.0, 1.0)[..., None] ** 0.40
    down = np.clip(-up_c, 0.0, 1.0)[..., None] ** 0.60
    col = np.where(up_c[..., None] >= 0.0,
                   np.float32(horizon) * (1.0 - up) + np.float32(zenith) * up,
                   np.float32(below_horizon) * (1.0 - down)
                   + np.float32(below_nadir) * down).astype(np.float32)
    # Horizon haze. The first build's band (amplitude 0.30/0.24/0.17 over a
    # 3.6-degree sigma) washed the whole upper-left of the lookback frame to a
    # white-grey -- Mojave noon at 65 degrees of sun elevation has a hard, deep
    # horizon, not the milky one of a humid coast -- so both the amplitude and
    # the width came down (0.13 over 2.6 deg) and the gradient underneath got
    # deeper at both ends. Still additive, still warm-tinted: some haze is
    # correct, an unbroken white band is not.
    hs = math.sin(math.radians(haze_width_deg))
    col += (np.exp(-(up_c * up_c) / (2.0 * hs * hs))[..., None]
            * np.float32(haze))
    ang = np.arccos(np.clip(D @ sun_dir, -1.0, 1.0))       # sun_dir is world
    col += ((np.exp(-(ang / math.radians(1.7)) ** 2) * sun_gain
             + np.exp(-(ang / math.radians(13.0)) ** 2) * glow_gain)[..., None]
            * np.float32([1.0, 0.96, 0.88]))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def load_env_cloud(asset, position=None, rotation_x=math.pi + math.pi / 2):
    """The scan's OWN sky: `<asset>/env` as a second SplatCloud, same transform.

    76k splats at ~1 km. Returns None (with a printed reason) if the chunk is
    missing. Costs a second cloud's fixed per-frame tax (~1.3 ms measured
    elsewhere) and draws the panorama's ground band as well as its sky.
    """
    env_dir = os.path.join(str(asset), "env")
    if not os.path.isdir(env_dir):
        print(f"[look] no env chunk at {env_dir}")
        return None
    data = tp.SogLoader.load(env_dir, level=0)
    cloud = tp.SplatCloud(data)
    cloud.rotation.x = rotation_x
    if position is not None:
        cloud.position.set(float(position[0]), float(position[1]), float(position[2]))
    return cloud


# ── the whole look, in one call ───────────────────────────────────────────────
class Look:
    """Handles for whatever `apply_look` put in the scene."""

    def __init__(self, sun, hemi, sky, env_cloud, sun_target, distance):
        self.sun = sun
        self.hemi = hemi
        self.sky = sky
        self.env_cloud = env_cloud
        self.sun_target = sun_target
        self.distance = distance
        self.dir = np.asarray(SUN_DIR_ZUP, np.float64)

    def follow(self, center):
        """Keep the shadow frustum around a moving robot WITHOUT turning the sun.

        The light stays on its line; only the point it is anchored to moves, so
        the shading never changes as the robot walks.
        """
        c = np.asarray(center, np.float64)
        p = c + self.dir * self.distance
        self.sun.position.set(float(p[0]), float(p[1]), float(p[2]))
        if self.sun_target is not None:
            self.sun_target.position.set(float(c[0]), float(c[1]), float(c[2]))


def apply_look(scene, renderer, *, sky="proc", asset=None, cloud_position=None,
               sun_dir=None, exposure=EXPOSURE, shadow_extent=6.0, distance=40.0,
               sun_intensity=SUN_INTENSITY, hemi_intensity=SKY_HEMI_INTENSITY):
    """Sky + one sun + pinned exposure. Returns a `Look`.

    sky: "proc" (threepp's own, the default), "env" (the scan's own sphere as a
         second cloud, plus the procedural map as the *environment* so the
         image-based light still exists), or "none".
    cloud_position: the splat cloud's world position, so an "env" sky lands on
         the same origin the scan does.
    """
    sd = np.asarray(sun_dir if sun_dir is not None else SUN_DIR_ZUP, np.float64)
    sd = sd / np.linalg.norm(sd)

    tex = sky_texture(sd)
    scene.environment = tex
    env_cloud = None
    if sky == "env":
        env_cloud = load_env_cloud(asset, cloud_position)
        if env_cloud is not None:
            scene.add(env_cloud)
        else:
            scene.background = tex
    elif sky == "proc":
        scene.background = tex
    # "none": whatever the caller already set stays

    hemi = tp.HemisphereLight(SKY_HEMI_TOP, SKY_HEMI_BOTTOM, hemi_intensity)
    scene.add(hemi)

    sun = tp.DirectionalLight(SUN_COLOR, sun_intensity)
    p = sd * distance
    sun.position.set(float(p[0]), float(p[1]), float(p[2]))
    sun.cast_shadow = True
    sun.set_shadow_frustum(-shadow_extent, shadow_extent, shadow_extent, -shadow_extent)
    sun.set_shadow_bias(-0.0004)
    scene.add(sun)
    try:
        target = sun.get_target()
        scene.add(target)
    except Exception:
        target = None

    renderer.auto_exposure = False          # PINNED: nothing about this scene may drift
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = float(exposure)
    renderer.shadow_map_enabled = True

    print(f"[look] sky={sky}  sun elev {SUN_ELEV_DEG:.0f} az {SUN_AZ_DEG:.0f} deg "
          f"(world dir {tuple(round(float(v), 3) for v in sd)})  exposure {exposure}")
    return Look(sun, hemi, tex, env_cloud, target, distance)


# ── contact shadows ───────────────────────────────────────────────────────────
# LAYERS, not a texture. The Vulkan deferred path routes an unlit transparent
# MeshBasicMaterial with NO map and NO vertex colours to the raster overlay,
# where material.opacity is honoured; give the same material a `map` and it
# falls back into the G-buffer and draws OPAQUE -- which is exactly what the
# first attempt at this did, a hard black square under the robot
# (shots/wp4_after_trackside.png before this note). So the soft edge is built
# out of concentric ANNULI instead, each carrying the alpha the falloff wants in
# its own radius band.
#
# The rings do NOT composite -- they tile the disc, so what you see at radius r
# IS that ring's alpha, and the falloff is only as smooth as the step between
# neighbours. The first build used 12 rings on a 1-(k/N)^2.4 ramp, whose largest
# step is 0.133 of alpha: at 1:1 that is ~26 levels of luminance at a hard
# circular edge, and shots/wp4_shadow_crop_after.png shows all twelve of them as
# concentric rings. So: 96 rings, largest step 0.022 -- about 4 levels, which
# the upscaler smears past visibility.
#
# The PROFILE is a flat core out to 0.45 R at the peak alpha, then a
# smootherstep to exactly zero at R (zero value AND zero slope at both ends, so
# neither the core boundary nor the rim is an edge). A plain Gaussian was tried
# first and is wrong for this: at sigma 0.55 R it carries an area-weighted mean
# of 0.23 against the old ramp's 0.48, and shots/wp4b_shadow_crop_gauss.png is
# what that looks like -- no rings, and very nearly no shadow either (the
# shadow-pool-to-rock luminance ratio under the left front foot went 0.51 -> 0.65
# where it should have stayed near 0.51). The flat core buys the mean back
# (0.33, ratio 0.54) and the radii carry the rest: 0.15 -> 0.18 m at the feet,
# 0.34 -> 0.40 m under the body, which puts the integrated darkening
# R^2 * mean(alpha) at 0.0108 against the old 0.0109 at a foot and 0.0258
# against 0.0237 under the body. Peak alpha 0.62.
_DISC_LAYERS = 96
_DISC_SEGMENTS = 48
_DISC_CORE = 0.45           # fraction of the outer radius held at full alpha


def _ring_alpha(k, n=_DISC_LAYERS, core=_DISC_CORE):
    """Peak-normalised falloff at the k-th ring's mid-radius. 1 at 0, 0 at R."""
    x = (k + 0.5) / float(n)
    if x <= core:
        return 1.0
    t = min(1.0, max(0.0, (1.0 - x) / (1.0 - core)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class ContactShadows:
    """Soft dark discs under the feet and the body.

    The renderer draws no shadow onto the splats -- there is nothing there to
    receive one -- so this is the whole contact cue. Each disc is an unlit,
    transparent, depth-write-off quad lying in the world XY plane (a
    PlaneGeometry's normal is +Z, which in this Z-up world already points up) at
    the floor height, and its size and opacity come from how high its foot is:

        h = 0        tight, dark disc  (the foot is planted)
        h = fade_m   gone             (the foot is in the air)

    which is the one thing a fake shadow has to get right, because a disc that
    stays put under a lifted foot is worse than no shadow at all.

    `art` is spot_deploy's articulation: links are [base, (hip, uleg, lleg) x 4],
    and the foot is the far end of each lleg capsule.
    """

    FOOT_LINKS = (3, 6, 9, 12)
    SHIN_HALF = 0.15            # lleg capsule half length (spot_deploy: 0.30 m)

    def __init__(self, scene, art, floor_fn=None, floor_z=0.0,
                 foot_radius=0.18, body_radius=0.40,
                 foot_opacity=0.62, body_opacity=0.30,
                 fade_m=0.22, lift=0.012, sun_dir=SUN_DIR_ZUP):
        self.art = art
        self.floor_fn = floor_fn if floor_fn is not None else (lambda x, y: floor_z)
        self.fade_m = float(fade_m)
        self.lift = float(lift)
        # A 65-degree sun does not put the shadow straight under the thing that
        # casts it: it offsets it by height/tan(elev) along the sun's azimuth,
        # away from the sun. 0.47 m of body height is 0.22 m of offset here, and
        # a body blob sitting dead centre is the tell that a fake shadow is fake.
        s = np.asarray(sun_dir, np.float64)
        s = s / np.linalg.norm(s)
        self.slide = -s[:2] / max(s[2], 1e-3)
        self.group = tp.Group()
        self.discs = []          # one tp.Group of stacked rings per contact point
        self.layers = []         # its per-layer meshes
        self.base_op = []
        for r, op in ([(foot_radius, foot_opacity)] * 4 + [(body_radius, body_opacity)]):
            g = tp.Group()
            layers = []
            # ANNULI, not a stack: coplanar discs on top of one another fight the
            # overlay's depth test and only the first survives, so each layer owns
            # its own radius band and carries the alpha the falloff wants there.
            for k in range(_DISC_LAYERS):
                r0 = r * k / float(_DISC_LAYERS)
                r1 = r * (k + 1) / float(_DISC_LAYERS)
                a = op * _ring_alpha(k)
                mat = tp.MeshBasicMaterial()
                mat.color = 0x120c07          # warm near-black, not pure black
                mat.transparent = True
                mat.opacity = a
                mat.depth_write = False
                mat.tone_mapped = False
                mat.fog = False
                geom = (tp.CircleGeometry(r1, _DISC_SEGMENTS) if k == 0 else
                        tp.RingGeometry(r0, r1, _DISC_SEGMENTS, 1))
                m = tp.Mesh(geom, mat)
                m.render_order = 3 + k
                g.add(m)
                layers.append((m, a))
            self.group.add(g)
            self.discs.append(g)
            self.layers.append(layers)
            self.base_op.append(float(op))
        scene.add(self.group)

    # -- geometry ---------------------------------------------------------
    def foot_positions(self):
        """World positions of the four feet, from the shin links' own poses."""
        links = self.art.links                 # a property, not a call
        out = []
        for i in self.FOOT_LINKS:
            L = links[i]
            p = L.position
            q = L.quaternion
            # the capsule's local axis is +Y (spot_deploy._capsule builds it that
            # way) and the foot is its far end
            v = _rotate((q.x, q.y, q.z, q.w), (0.0, self.SHIN_HALF, 0.0))
            out.append((p.x + v[0], p.y + v[1], p.z + v[2]))
        return out

    @property
    def visible(self):
        return self.group.visible

    @visible.setter
    def visible(self, v):
        self.group.visible = bool(v)

    def update(self):
        pts = self.foot_positions()
        rs = self.art.root_state()
        pts.append((float(rs[0]), float(rs[1]), float(rs[2])))
        for i, (x, y, z) in enumerate(pts):
            fz = float(self.floor_fn(x, y))
            h = max(0.0, z - fz - (0.0 if i < 4 else 0.42))
            t = min(1.0, h / self.fade_m)
            s = 1.0 + 0.9 * t                      # spreads as the foot lifts
            d = self.discs[i]
            off = self.slide * (z - fz)            # sun-driven offset
            d.position.set(float(x + off[0]), float(y + off[1]), fz + self.lift)
            d.scale.set(s, s, 1.0)
            fade = (1.0 - t) ** 1.3
            for m, a in self.layers[i]:
                m.material.opacity = a * fade
            d.visible = fade > 0.02


def _rotate(q, v):
    x, y, z, w = q
    # v + 2w(q x v) + 2 q x (q x v)
    ux, uy, uz = x, y, z
    cx = uy * v[2] - uz * v[1]
    cy = uz * v[0] - ux * v[2]
    cz = ux * v[1] - uy * v[0]
    c2x = uy * cz - uz * cy
    c2y = uz * cx - ux * cz
    c2z = ux * cy - uy * cx
    return (v[0] + 2.0 * (w * cx + c2x),
            v[1] + 2.0 * (w * cy + c2y),
            v[2] + 2.0 * (w * cz + c2z))
