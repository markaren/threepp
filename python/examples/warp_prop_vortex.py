"""Propeller tip vortices that read as ROPES -- NVIDIA Warp sim, Vulkan rendering.

warp_nebula_vk.py proved the thesis on a cloud: particles carry the IMAGE, the
field's own density volume carries the LIGHT TRANSPORT. A nebula is the easy
case for it, because a cloud has no shape you can be wrong about. This is the
hard case, and the one the feature was built for: a three-bladed propeller
shedding tip vortices into the classic interleaved helices, which braid,
destabilise and diffuse into a self-shadowed slipstream.

WHY IT NEEDS THE VOLUME. A helix is a curve that passes IN FRONT OF ITSELF.
Additive sprites cannot say which loop is nearer -- every crossing is the same
sum from either side -- so a flat render of this is a tangle of uniform glowing
wire. With the two marches on, the near loop occludes the far one (T_cam) and
the slipstream shadows its own far side (T_sun), and the same particles resolve
into ropes wound around a lit cylinder of smoke. --flat at the same timestamp
is the whole acceptance test.

    pip install warp-lang
    python warp_prop_vortex.py                   # window; drag to orbit
    python warp_prop_vortex.py --shot 0.9        # headless PNG, the spin-up
    python warp_prop_vortex.py --shot 4.0        # the developed wake
    python warp_prop_vortex.py --shot 4.0 --flat # the SAME frame, knobs at 0
    python warp_prop_vortex.py --n 5000000       # more particles: a grainless far wake
    python warp_prop_vortex.py --bench           # frame time, knobs on vs --flat

THE SIM IS A CLOSED FORM, like the renderer's own emitter and unlike the
nebula's integrator: slot s is reborn every `lifetime` seconds and its whole
state is f(seed, slot, t). Nothing accumulates, so --shot 4.0 is 240 frames of
evaluation rather than 240 frames of integration, and the wake is the same wake
whether you seek to it or walk to it.

    tb    shed time; the slot's Lagrangian label
    tau   age, t - tb
    phi   theta(tb) + blade * 2pi/B + swirl(tau)   -- swirl DECAYS with age
    x     U_wake * tau, eased from the disc velocity over the first ~0.1 s
    r     r0 contracting toward 0.78 r0 (slipstream contraction)
    w     sqrt(w0^2 + 2 nu_t tau)                  -- Lamb-Oseen core diffusion
    +     a curl-noise displacement growing as tau^1.9, sampled at the
          Lagrangian LABEL rather than at the position, so the wobble convects
          WITH the filament instead of standing still in the world while the
          wake slides through it.

THE CREDIBILITY DETAIL is theta(t): the visible propeller's rotation and the
kernel's birth azimuth are the same function of the same clock, so every
filament is rooted at a blade tip and stays welded to it. Get that wrong by one
frame and the whole thing reads as a smoke machine behind a fan.

BillboardRepr::stretchSeconds is NOT used, and not for lack of trying: the
backend gates the streak on `rendererOwned || hostPrevIsPrevStep`, and an
Interop field is neither, so it is a hard no-op here (ParticleFieldPass.cpp).
The rope's thickness comes from the per-particle tube offset instead, which is
where it belongs -- a vortex core is a tube, not a smeared point.

DETERMINISM is forfeit on the interop leg exactly as it is for the nebula (the
bytes are authored past the import); the sim clock is nevertheless the frame
index times DT and never a wall clock, so a shot is reproducible modulo the
import. --no-interop falls back to the host ring at a fraction of the count.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp

import threepp as tp
from warp_common import cli_arg

N = cli_arg("--n", 2_000_000, int)
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 4.0, float)
FLAT = "--flat" in sys.argv
INTEROP = "--no-interop" not in sys.argv
W, H = cli_arg("--width", 1280, int), cli_arg("--height", 800, int)
HEADLESS = SHOT or BENCH
FPS = 60
DT = 1.0 / FPS

# ── The propeller ───────────────────────────────────────────────────────────
# Axis is +X: the disc lies in YZ at x = 0 and the wake blows toward +X. A
# point at (0, r, 0) rotated about X by theta lands at (0, r cos, r sin), which
# is exactly the kernel's (y, z) -- so the mesh's rotation.x and the kernel's
# azimuth are the same number with no sign to get wrong.
BLADES = cli_arg("--blades", 3, int)
R_TIP = 0.90                        # m
R_HUB = 0.14
RPS = cli_arg("--rps", 6.0, float)  # revolutions per second at full speed
OMEGA = 2.0 * math.pi * RPS
T_RAMP = cli_arg("--ramp", 2.0, float)   # spin-up, seconds

# ── The wake ────────────────────────────────────────────────────────────────
LIFETIME = 0.85          # s of wake held on screen; also the slot recycle period
U_DISC = 3.6             # axial velocity AT the disc (m/s)
U_WAKE = 5.6             # fully developed slipstream velocity; ~2x inflow
TAU_A = 0.13             # s over which the flow eases from U_DISC to U_WAKE
CONTRACT = 0.78          # slipstream contracts to this fraction of its birth radius
TAU_C = 0.16             # s of that contraction
SWIRL_F = 0.22           # self-induced rotation of the helix, as a fraction of OMEGA
TAU_S = 0.40             # s over which the swirl decays away
W_CORE0 = 0.010          # vortex core radius at shedding (m)
# ── Core growth: BURST, not sqrt ────────────────────────────────────────────
# Lamb-Oseen's w = sqrt(w0^2 + 2 nu_t tau) was the first thing tried and it is
# the wrong law for this picture, for a reason worth writing down: a sqrt
# spends most of its growth in the FIRST instants, so any nu_t large enough to
# fog the far wake has already fogged the near one and there is no rope left to
# look at. A tip vortex does not do that. It holds a tight, coherent core while
# it is stable and then BURSTS when the helical instability catches up with it,
# which is a delayed growth -- and that is both what the photographs show and
# what makes the near/far split of this demo exist at all.
W_BURST = cli_arg("--burst", 0.20, float)   # core radius added by end of life (m)
W_BURST_P = 2.4                             # its delay: flat, then a knee
TURB = cli_arg("--turb", 0.30, float)    # m of curl displacement at end of life
TURB_P = 1.9             # its growth exponent in age -- tight at the tip, wild downstream
TURB_FREQ = 0.55         # label-space wavelength of the meander
SHEET = cli_arg("--sheet", 0.30, float)  # fraction of slots born across the SPAN
# ── Sprite size and the flux it carries ─────────────────────────────────────
# The sprite is a GRAIN of the core, not the core: tying its radius to wc (the
# obvious first move) grows its area ~400x over the life, and since a sprite
# contributes radiance x area, the far wake then out-blazes the very filaments
# it diffused from -- the picture inverts, which is what the first pass did.
# The grain grows instead, and its radiance is divided by the area it gained,
# so a parcel's total flux is roughly conserved and the far wake dims because
# it is SPREAD rather than because it was told to.
#
# GROWTH IS A KNEE, NOT A RAMP -- the second thing this had to learn. The first
# pass grew the grain LINEARLY in age (1 + 3.2 af), which is the one law that
# cannot serve both halves of this picture: small enough to keep the near ropes
# tight left the aged grains at ~1.5 px, far too small to touch their
# neighbours, and the far wake came out as SAND -- discrete specks reading as
# static rather than smoke. Growing them enough to fuse then fattened the young
# ones and dissolved the ropes. Putting the grain on the same af^p knee the
# burst core already uses (W_BURST_P) separates the two populations: below
# af~0.4 the grain is still essentially SPRITE_R0 and the rope is a rope, above
# it the grain balloons ~13x and neighbouring sprites OVERLAP into a continuum.
# Overlap is the whole point -- a smooth far wake is not a smoother sprite, it
# is enough sprites per pixel that no individual one is visible.
SPRITE_R0 = cli_arg("--grain", 0.0026, float)   # grain radius at shedding (m)
SPRITE_GROW = cli_arg("--grow", 13.0, float)    # x radius by end of life
SPRITE_GROW_P = 2.2                             # its delay -- the same knee as the burst
# The flux split pays for that area. Radius x13 is area x170, so a grain that
# kept its radiance would bloom the far wake to white; FLUX_P near 1 divides the
# radiance by very nearly the area gained, which holds the far wake at the
# per-pixel brightness it had while spreading its energy over 170x the film.
# 0.80 is a deliberate half-step back from conserving: it trades a little
# smoothness for a far wake that is still THERE.
FLUX_P = cli_arg("--flux", 0.80, float)         # 1 = flux-conserving, >1 = dims faster
# The last speckle source was the sprite's t^9 CORE spike: the falloff plants
# a 1-2 px hard dot at every sprite's centre regardless of drawn radius, and
# `softness` deliberately does not reach it (it is what keeps a few-pixel ember
# from reading as a blob). It is a knob now -- BillboardRepr.core_weight, 0.85
# being the old hardcoded constant -- and this demo turns it OFF below, which
# is what lets 2M read smooth where it used to take --n 5000000 to average the
# dots away. 2M is where the 25 fps floor put the default on a 4070 at 1280x800.

# ── The look ────────────────────────────────────────────────────────────────
# The density volume is LATCHED at these bounds and never refitted: a box that
# tracks its own matter re-phases the lattice every frame and the volume swims.
# Elongated along the wake and tight across it -- the sun leg's cost is the box
# crossing distance, so a fat box is paid for in milliseconds.
BOX_CENTER = (2.10, 0.0, 0.0)
BOX_HALF = (2.45, 1.20, 1.20)
BOX_RES = cli_arg("--vol-res", 160, int)
SIGMA = cli_arg("--sigma", 0.32, float)   # sigma_t one particle contributes
EXPOSURE = cli_arg("--exposure", 1.0, float)
EXTINCTION = 0.0 if FLAT else cli_arg("--extinction", 1.0, float)
SHADOW = 0.0 if FLAT else cli_arg("--shadow", 0.72, float)
# Additive sums clip and overlap grows with the particle count, so exposure
# follows the same 1/sqrt(N) law the other field demos use.
BRIGHT = cli_arg("--bright", 0.038, float) * math.sqrt(2_000_000 / N)

TWO_PI = 2.0 * math.pi


def prop_theta(t):
    """Blade azimuth at time t, in radians. Omega ramps linearly over T_RAMP, so
    this is the exact integral of that ramp -- the mesh and the kernel must not
    merely agree at steady state, they must agree DURING the spin-up too."""
    if t < T_RAMP:
        return OMEGA * t * t / (2.0 * T_RAMP)
    return OMEGA * (t - 0.5 * T_RAMP)


# ── The kernel ──────────────────────────────────────────────────────────────
# One launch, no state: out_pos is the exported POSITION allocation (xyz + w =
# the per-particle world radius, WSemantic.Radius) and out_col the exported
# ATTRIBUTE allocation (rgb = linear HDR radiance). Both written in place,
# device to device.
@wp.func
def theta_of(t: float, omega: float, ramp: float) -> float:
    if t < ramp:
        return omega * t * t / (2.0 * ramp)
    return omega * (t - 0.5 * ramp)


@wp.func
def omega_frac(t: float, ramp: float) -> float:
    # How far up the spin-up ramp the flow was when this parcel was shed. The
    # wake it convects into is the wake that existed AT SHEDDING, so a parcel
    # born during the ramp stays slow for the rest of its life.
    return wp.clamp(t / ramp, 0.0, 1.0)


@wp.kernel
def shed(out_pos: wp.array(dtype=wp.vec4),
         out_col: wp.array(dtype=wp.vec4),
         t: float, n: int, blades: int,
         omega: float, ramp: float, turb: float, sheet: float,
         burst: float, sprite_r0: float, flux_p: float, bright: float):
    i = wp.tid()
    s = wp.rand_init(90210, i)

    # ── The slot's own clock ────────────────────────────────────────────────
    # Uniform phase offsets over one lifetime give a uniform shedding RATE, so
    # the filament has even density along its whole length rather than beads.
    u = (float(i) + 0.5) / float(n)
    ph = t / LIFETIME + u
    tau = (ph - wp.floor(ph)) * LIFETIME        # age
    tb = t - tau                                # shed time: the Lagrangian label
    af = tau / LIFETIME                         # age fraction, [0, 1)
    if tb < 0.0:
        # Before the propeller started there is no wake. Park the slot on the
        # w < 0 dead sentinel every consumer already tests.
        out_pos[i] = wp.vec4(0.0, 0.0, 0.0, -1.0)
        out_col[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return

    blade = i % blades
    is_sheet = wp.randf(s) < sheet

    # ── Birth on the blade ──────────────────────────────────────────────────
    # The tip population is born in the last 2% of the span (that IS the tip
    # vortex); the sheet population is born across the whole span and rolls up
    # behind the blade as the entrained vortex sheet.
    r0 = R_TIP * (0.985 + 0.02 * wp.randf(s))
    if is_sheet:
        r0 = R_TIP * (0.26 + 0.72 * wp.randf(s))
    span = r0 / R_TIP

    # ── Downstream ──────────────────────────────────────────────────────────
    ofr = omega_frac(tb, ramp)
    # Axial: eased from the disc velocity to the developed slipstream. The
    # inner span pushes less air, so it lags -- which is what shears the sheet
    # against the tip helix and starts the braiding.
    ax = 0.55 + 0.45 * span
    u_w = U_WAKE * ofr * ax
    u_d = U_DISC * ofr * ax
    x = u_w * tau - (u_w - u_d) * TAU_A * (1.0 - wp.exp(-tau / TAU_A))
    # Radial: contraction toward CONTRACT * r0.
    r = r0 * (CONTRACT + (1.0 - CONTRACT) * wp.exp(-tau / TAU_C))
    # Azimuth: the blade's own phase at shedding, plus a self-induced rotation
    # of the helical system that DECAYS with age (integral of a decaying rate).
    # The inner sheet sits deeper in the swirl and turns faster, which is the
    # roll-up.
    sw = SWIRL_F * omega * ofr * (1.6 - 0.6 * span)
    phi = theta_of(tb, omega, ramp) + float(blade) * TWO_PI / float(blades) \
        + sw * TAU_S * (1.0 - wp.exp(-tau / TAU_S))

    p = wp.vec3(x, r * wp.cos(phi), r * wp.sin(phi))

    # ── The core is a TUBE, not a curve ─────────────────────────────────────
    # Lamb-Oseen diffusion: the core fattens as sqrt(t), which is the whole
    # reason the far wake is smoke and the near wake is a rope.
    wc = W_CORE0 + burst * wp.pow(af, W_BURST_P)
    if is_sheet:
        wc = wc * 2.6                     # the sheet is diffuse from the start
    p = p + wp.vec3(wp.randn(s), wp.randn(s), wp.randn(s)) * (0.55 * wc)

    # ── Instability: curl noise at the LABEL, growing with age ──────────────
    # Sampled at (shed time, birth azimuth) rather than at the position, so a
    # bump on the filament CONVECTS with it. Position-keyed noise would leave a
    # standing disturbance in the world that the wake slides through, and the
    # wake would shimmer instead of braid. Neighbouring shed times get
    # neighbouring labels, so the meander is smooth ALONG a filament while the
    # three blades wander independently -- which is what makes them collide.
    phi0 = theta_of(tb, omega, ramp) + float(blade) * TWO_PI / float(blades)
    lab = wp.vec4(tb * 3.1, wp.cos(phi0) * TURB_FREQ, wp.sin(phi0) * TURB_FREQ,
                  span * 1.3)
    amp = turb * wp.pow(af, TURB_P)
    if is_sheet:
        amp = amp * 1.7
    n1 = wp.rand_init(11)
    n2 = wp.rand_init(29)
    p = p + wp.curlnoise(n1, lab) * amp
    p = p + wp.curlnoise(n2, lab * 2.7) * (amp * 0.35)

    # ── Colour, in kernel ───────────────────────────────────────────────────
    # Young and tight: a bright, slightly cyan-white condensation core, which
    # is what a tip vortex actually looks like when it fogs. Old and diffused:
    # dim neutral smoke, so the far wake's shape is carried by the SUN term
    # (volume_shadow) and not by its own emission.
    mix = wp.pow(af, 0.55)
    c = wp.vec3(0.58, 0.80, 1.00) * (1.0 - mix) + wp.vec3(0.46, 0.47, 0.50) * mix
    grow = 1.0 + SPRITE_GROW * wp.pow(af, SPRITE_GROW_P)
    gain = (1.0 - 0.20 * af) / wp.pow(grow, flux_p)
    if is_sheet:
        gain = gain * 0.42
    c = c * (bright * gain)

    out_pos[i] = wp.vec4(p[0], p[1], p[2], sprite_r0 * grow)
    out_col[i] = wp.vec4(c[0], c[1], c[2], 1.0)


wp.init()
device = wp.get_preferred_device()
print(f"prop vortex: {N:,} particles on {device}"
      f"{'  [--flat: volumetrics OFF]' if FLAT else ''}")
INTEROP = INTEROP and device.is_cuda

# ── Scene ───────────────────────────────────────────────────────────────────
canvas = tp.Canvas("threepp x warp - propeller tip vortices", width=W, height=H,
                   vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
# PINNED: the billboards composite after the post stack and so bypass auto
# exposure entirely; a drifting exposure would move the sprites and the
# background against each other and make the --flat A/B meaningless.
renderer.tone_mapping_exposure = EXPOSURE

scene = tp.Scene()
scene.background = tp.Color(0.004, 0.005, 0.008)

camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.02, 200)
camera.position.set(cli_arg("--cam-x", 1.10, float),
                    cli_arg("--cam-y", 0.56, float),
                    cli_arg("--cam-z", 3.00, float))
camera.look_at(1.42, -0.02, 0.0)

# The KEY, and the thing volume_shadow marches toward. Placed across the wake
# and slightly BEHIND it: a backlit slipstream is the arrangement that makes
# self-shadowing legible -- the sunward flank of every loop flares through the
# forward HG lobe while the near flank sits in the wake's own shadow.
sun = tp.DirectionalLight(0xFFEFD8, 3.2)
sun.position.set(-0.9, 1.5, -2.4)
scene.add(sun)

# The RAKE -- the second half of "the prop is not a hole". It exists for the
# propeller alone and is aimed to graze the blade faces from the camera's own
# side, low and off to the right, so the twist catches it as a moving edge down
# the span rather than as flat frontal fill.
#
# IT MUST STAY DIMMER THAN THE SUN. The billboards do not sum the light rig:
# they take the single BRIGHTEST DirectionalLight in the scene as their sun and
# march T_sun toward it (VulkanCoreUploads.cpp). Let this one win on max
# channel and the whole wake's self-shadowing silently flips to a front-lit
# read, which is the one arrangement that makes a helix illegible. 1.5 vs 3.2
# is the margin; anything that raises it has to check the far crop.
rake = tp.DirectionalLight(0xD8E4FF, 1.5)
rake.position.set(2.4, 0.95, 3.1)
scene.add(rake)
scene.add(tp.AmbientLight(0x8C9BC0, 1.0))


# ── The propeller, from primitives and one lofted blade ─────────────────────
def blade_geometry(stations=14, around=10):
    """One twisted, tapered blade, lofted from elliptical sections.

    Span runs along +Y from the hub to R_TIP, chord along Z, and each section
    is rotated about the span axis by its own pitch angle -- high at the root,
    low at the tip, which is what a real propeller does so that every station
    meets the air at a similar angle of attack.
    """
    import numpy as np
    sp = np.linspace(R_HUB, R_TIP, stations)
    f = (sp - R_HUB) / (R_TIP - R_HUB)                  # 0 at root, 1 at tip
    chord = 0.30 * (0.55 + 0.85 * f - 0.95 * f ** 3)    # broad mid-span, fine tip
    thick = chord * 0.14
    pitch = np.radians(46.0 - 34.0 * f ** 0.75)         # geometric twist
    sweep = 0.06 * f ** 2.2                             # a little tip rake

    a = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    # Section in (chordwise, thickness), cambered a touch so the two faces are
    # not mirror images and the light breaks across the blade.
    cw = 0.5 * np.cos(a)
    th = 0.5 * np.sin(a) + 0.10 * (1.0 - np.cos(a) ** 2)
    v = np.empty((stations, around, 3), np.float32)
    for k in range(stations):
        z = cw * chord[k]
        y = th * thick[k]
        c, s = math.cos(pitch[k]), math.sin(pitch[k])
        v[k, :, 0] = y * c - z * s + sweep[k]           # axial (out of the disc)
        v[k, :, 1] = sp[k]                              # span
        v[k, :, 2] = y * s + z * c                      # in-plane chord

    # ── UVs, because a loft has none ────────────────────────────────────────
    # compute_vertex_normals() fills in normals; nothing fills in uv, and a
    # rust map on a missing attribute is a solid smear of whatever texel 0 is.
    # v runs root -> tip. u runs CHORDWISE as a triangle wave over the section
    # index rather than a sawtooth: the ring closes with j2 = (j+1) % around,
    # so a 0..1 sawtooth would put one quad per section carrying the entire
    # texture backwards across the seam -- a visible stripe down the blade. The
    # triangle wave is 0 at the seam and 1 at the far side, continuous at both,
    # and mirrors the two faces, which for a rust map with no direction in it
    # costs exactly nothing.
    uv = np.empty((stations, around, 2), np.float32)
    tri = 1.0 - np.abs(2.0 * a / (2.0 * np.pi) - 1.0)
    uv[:, :, 0] = tri[None, :]
    uv[:, :, 1] = f[:, None]
    idx = []
    for k in range(stations - 1):
        for j in range(around):
            j2 = (j + 1) % around
            a0 = k * around + j
            b0 = k * around + j2
            a1 = (k + 1) * around + j
            b1 = (k + 1) * around + j2
            # WINDING. The first pass wound these the other way round and the
            # blades were INSIDE-OUT: compute_vertex_normals() then hands every
            # blade vertex a normal pointing into the solid, N.L is negative on
            # the face the camera can see, and the blade shades black no matter
            # what light you put on it. That, not the material, is why this
            # demo's prop was a silhouette -- the hub is a stock primitive
            # with correct windings and it was always lit. Along
            # the ring, increasing j sweeps from the chord's +Z edge toward +X,
            # so (a0, b1, a1) is the order whose cross product points OUT.
            idx += [a0, b1, a1, a0, b0, b1]
    # Fan-cap both ends so the blade is a closed solid.
    for k, flip in ((0, False), (stations - 1, True)):
        base = k * around
        for j in range(1, around - 1):
            tri = [base, base + j, base + j + 1]
            idx += tri[::-1] if flip else tri
    g = tp.BufferGeometry()
    g.set_attribute("position", v.reshape(-1, 3))
    g.set_attribute("uv", uv.reshape(-1, 2))
    g.set_index(np.asarray(idx, np.uint32))
    g.compute_vertex_normals()
    return g


# ── The prop has to be an OBJECT, not a hole ────────────────────────────────
# A bare grey MeshStandardMaterial gave this demo a black silhouette, and for a
# reason that is the wake's own doing: the key is deliberately BEHIND the
# slipstream (see the light below), there is no environment map, and so the
# blade faces the camera sees have nothing at all to return. Two fixes, and it
# needs both. This one is the surface -- Poly Haven's "rust_coarse_01" (CC0,
# 1k JPGs under assets/), albedo + normal + roughness. The normal map is doing
# most of the work: it breaks the raking fill into hundreds of little
# highlights, which is what says "pitted machined artifact" in the first second
# of looking, where a flat grey lambert says "placeholder".
RUST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "rust_coarse_01")
RUST_REPEAT = 2.0     # tiles per uv unit; ~0.35 m of blade per tile


def rust_map(suffix, srgb):
    path = os.path.join(RUST_DIR, f"rust_coarse_01_{suffix}_1k.jpg")
    if not os.path.exists(path):
        return None
    t = tp.TextureLoader().load(
        path, tp.ColorSpace.SRGB if srgb else tp.ColorSpace.NoColorSpace)
    t.wrap_s = tp.TextureWrapping.Repeat
    t.wrap_t = tp.TextureWrapping.Repeat
    t.repeat = tp.Vector2(RUST_REPEAT, RUST_REPEAT)
    return t


metal = tp.MeshStandardMaterial()
albedo = rust_map("diff", True)
if albedo is not None:
    metal.map = albedo
    metal.normal_map = rust_map("nor_gl", False)
    metal.roughness_map = rust_map("rough", False)
    metal.color = tp.Color(0xC9C2BA)   # multiplies the map; a touch under white
    metal.normal_scale = tp.Vector2(1.5, 1.5)
    # roughness/metalness MULTIPLY their maps in the deferred path, so these are
    # ceilings and not values. 0.62 leaves the map's smoother pits around 0.3,
    # which is the range that still returns a specular edge to a grazing light.
    metal.roughness = 0.62
else:
    print(f"  (no rust maps in {RUST_DIR}; the prop stays plain metal)")
    metal.color = tp.Color(0x6B6259)
    metal.roughness = 0.45
metal.metalness = 0.22       # low: with no env map, ambient is the only fill a
                             # metal would refuse, and the prop must not be a hole
# aoMap is deliberately NOT set and its 1k JPG deliberately not shipped: the
# Vulkan deferred G-buffer carries albedo / rough-metal / normal texture slots
# and no occlusion term (gbuffer.frag), so it would be 640 kB in the repo for
# a guaranteed no-op.

prop = tp.Group()
hub = tp.Mesh(tp.CylinderGeometry(0.135, 0.155, 0.30, 28), metal)
hub.rotate_z(math.pi / 2)                    # the cylinder's own axis is Y
prop.add(hub)
bg = blade_geometry()
for b in range(BLADES):
    blade = tp.Mesh(bg, metal)
    blade.rotate_x(b * TWO_PI / BLADES)      # same offset the kernel uses
    prop.add(blade)
scene.add(prop)

# ── The field ───────────────────────────────────────────────────────────────
cfg = tp.ParticleField.Config()
cfg.capacity = N
cfg.ownership = tp.ParticleField.Ownership.Interop if INTEROP \
    else tp.ParticleField.Ownership.HostRing
cfg.w_semantic = tp.ParticleField.WSemantic.Radius
cfg.attributes = True              # per-particle colour, straight out of the kernel
field = tp.ParticleField.create(cfg)
field.frustum_culled = False
scene.add(field)

# The DENSITY half: the same particles, in one latched world box that only the
# LIGHT TRANSPORT reads. center/half_extent are per-frame writable and are
# deliberately never written.
field.set_density_repr(tp.Vector3(*BOX_CENTER), tp.Vector3(*BOX_HALF), SIGMA, BOX_RES)
# Nearly black, deliberately: the same volume is also marched by the deferred
# fog, which in-scatters through it at low frequency and would lay a smooth grey
# veil over exactly the filaments this demo exists to keep. The volume's job
# here is TRANSPORT; the sprites carry every photon.
field.density_repr.albedo = tp.Color(0.016, 0.018, 0.022)
field.density_repr.anisotropy = 0.0

# The IMAGE half.
field.set_billboard_repr(tp.Color(1, 1, 1), tp.Color(1, 1, 1), 1.0, 1.0)
bb = field.billboard_repr
# The falloff is skirt + a t^9 core; softness slides the skirt exponent 4 -> 1.2
# (particlefield_billboard.frag). High is what a SMOKE grain wants: the tight
# core is a spark's shape and, drawn at the aged grain's 10-odd pixels, it is
# exactly the hard speck the far wake must stop reading as. This is a per-FIELD
# knob and not per-particle, which sounds like it should soften the near ropes
# too -- it does not, because a young grain is ~1.5 px across and a radial
# profile has nowhere to act inside a pixel and a half. The knee gives softness
# its age selectivity for free.
bb.softness = 0.92
# The core dot OFF entirely: a smoke parcel has no spark at its centre. The
# ropes survive this -- a young grain is ~1.5 px and its whole profile is one
# bright sample either way -- but the wake stops being 2M pinpricks.
bb.core_weight = 0.0
bb.bright_jitter = 0.0        # the sim authors the colour; do not hash over it
bb.fade_power = 0.0           # no age exists on an interop field
bb.size_taper = 0.0
bb.stretch_seconds = 0.0      # a hard no-op on an Interop field -- see the docstring
bb.lit_phase_g = 0.58         # forward: the backlit flank flares
bb.volume_extinction = EXTINCTION
bb.volume_shadow = SHADOW
bb.volume_ambient = 0.26
bb.volume_sun_gain = 0.85

sim_time = 0.0
frame_no = 0
imported_pos = imported_col = None
host_pos = host_col = None


def launch(out_pos, out_col):
    wp.launch(shed, dim=N, device=device,
              inputs=[out_pos, out_col, sim_time, N, BLADES, OMEGA, T_RAMP,
                      TURB, SHEET, W_BURST, SPRITE_R0, FLUX_P, BRIGHT])


def advance():
    """The clock, and the phase lock. The sim time is the FRAME INDEX times DT
    and never a wall clock, and the propeller's azimuth is the same theta(t) the
    kernel evaluates for a filament's birth -- which is what welds every
    filament to the tip it was shed from."""
    global sim_time, frame_no
    frame_no += 1
    sim_time = frame_no * DT
    prop.rotation.x = prop_theta(sim_time)


def device_copy():
    """The renderer's per-frame callback: evaluate the wake straight into the
    two exported allocations, then synchronize. It runs INSIDE render(),
    pre-record, and the host ordering it provides is the ONLY thing sequencing
    these writes against the frame that reads them."""
    launch(imported_pos.array, imported_col.array)
    wp.synchronize_device(device)


def step_frame_host():
    advance()
    launch(host_pos, host_col)
    wp.synchronize_device(device)
    field.submit(host_pos.numpy(), DT)
    field.set_attributes(host_col.numpy())


def step_frame_interop():
    advance()          # device_copy, inside render(), reads sim_time


# ── ARM THE ZERO-COPY PATH ──────────────────────────────────────────────────
# enable_particle_field_interop returns None until after the FIRST render(): the
# field's device state and the renderer's field pass are both created on the
# frame the field is first seen.
field.set_live_count(0)
renderer.render(scene, camera)
step_frame = step_frame_host
if INTEROP:
    from threepp.cuda_interop import VkInteropArray
    try:
        h = renderer.enable_particle_field_interop(field, device_copy)
        if h is None:
            raise RuntimeError("this device cannot export memory "
                               f"(host_fallback={field.host_fallback})")
        if len(h) < 4:
            raise RuntimeError("the renderer exported no attribute buffer -- "
                               "Config.attributes was not honoured")
        imported_pos = VkInteropArray(h[0], h[1], wp.vec4, N, device)
        imported_col = VkInteropArray(h[2], h[3], wp.vec4, N, device)
        step_frame = step_frame_interop
        print(f"       zero-copy: {2 * 16 * N / 1e6:.0f} MB exported "
              f"(positions + attributes), 0 B/frame across the bus")
    except Exception as e:                                # noqa: BLE001
        INTEROP = False
        print(f"       no zero-copy export ({e}); host ring")

if not INTEROP:
    host_pos = wp.zeros(N, dtype=wp.vec4, device="cpu", pinned=True)
    host_col = wp.zeros(N, dtype=wp.vec4, device="cpu", pinned=True)

field.set_live_count(N)

print(f"       prop:   {BLADES} blades, R {R_TIP:g} m, {RPS:g} rev/s "
      f"(ramp {T_RAMP:g} s), wake {LIFETIME:g} s\n"
      f"       volume: {BOX_RES}^3 over {2 * BOX_HALF[0]:.1f} x "
      f"{2 * BOX_HALF[1]:.1f} x {2 * BOX_HALF[2]:.1f} m, sigma/particle {SIGMA:g}\n"
      f"       knobs:  extinction {EXTINCTION:g}, shadow {SHADOW:g}, "
      f"grain {SPRITE_R0:g} m, flux {FLUX_P:g}, bright {BRIGHT:.5f}")


def run_to(seconds):
    frames = int(round(seconds * FPS))
    t0 = time.perf_counter()
    for f in range(frames):
        step_frame()
        renderer.render(scene, camera)
        if f % 60 == 0:
            print(f"  t={sim_time:5.2f}", flush=True)
    return frames, time.perf_counter() - t0


if BENCH:
    run_to(2.5)                                   # warm the pipeline and the wake
    n = 240
    t0 = time.perf_counter()
    for _ in range(n):
        step_frame()
        renderer.render(scene, camera)
    dt = (time.perf_counter() - t0) / n
    print(f"bench {N:,} particles [{'flat' if FLAT else 'volumetric'}]: "
          f"{1e3 * dt:.2f} ms/frame ({1.0 / dt:.0f} fps)")
elif SHOT:
    frames, wall = run_to(SHOT_TIME)
    out = cli_arg("--out", "warp_prop_vortex_flat.png" if FLAT
                  else "warp_prop_vortex.png", str)
    renderer.save_frame(scene, camera, out)
    print(f"simulated {SHOT_TIME:.1f} s ({frames} frames) in {wall:.1f}s, wrote {out}")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(1.42, -0.02, 0.0)

    def animate():
        step_frame()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)
