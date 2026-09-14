"""
fireworks -- a staggered aerial firework display as one ParticleField.

Shells go up from one launch point and burst. Everything is ONE
`tp.ParticleField` (HostRing ownership, stable slots) plus one `tp.PointLight`
per shell, and the whole show is a CLOSED FORM in the scene clock:
`show.update(t)` at any t is valid with no warm-up and no integration, so a
film that pins `renderer.sim_time` can seek straight to the finale and get
the same picture every run, and two processes rendering the same frame agree
pixel for pixel.

Integration into a scene is three lines -- at the scene build, once::

    from fireworks import FireworkShow, Shell
    show = FireworkShow(scene, (x, y_ground, z),
                        FireworkShow.default_shells(76.0), seed=SEED)

and once per frame, before `renderer.render`, with the SAME clock the renderer
is pinned to::

    show.update(t)

`default_shells(t0)` is a three-shell cue sheet relative to the first launch:
two warm gold either side of one blue, the last the biggest, launched at t0,
t0 + 3.2 and t0 + 6.8 s and bursting 1.5-1.7 s later. Pass your own list of
`Shell` (or of `(t_launch, apex)` tuples) to re-time or re-colour it. The pace
is a real display's, not a slow-motion one: a shell is up in about a second
and a half and its break opens in a third of a second.

SIZING FOR THE CAMERA
---------------------
The defaults were tuned for a lens about 110 m from the bursts, where a 0.23 m
star covers ~3 px and a cloud 15-17 m across spans a fifth of the frame. Two
things to check before reusing them at another range:

  * Star size is in METRES and the billboards are additive, so at six times
    the range the stars go sub-pixel and `intensity` can do nothing about it:
    it makes an invisible dot whiter, it cannot make it cover area. Work out
    px = radius / (dist * fov_rad / height); below ~2 px only `size_scale`
    helps. Raise `size_scale`, then bring `intensity` DOWN -- scaling both
    sums the quads to a featureless white disc and the shell loses its colour.
    (A finale watched from 680 m settled on size_scale 8, intensity 0.58.)
  * `Shell.apex` is metres above the LAUNCH POINT. A burst centre plus its
    cloud radius (speed / drag, 14-17 m here) has to fit under the top of the
    frame; raise the apexes past that and the finale loses its crown.

WHY THE SHAPING IS ALL DONE ON THE HOST
---------------------------------------
A billboard field only knows a particle's AGE when the field is
`Ownership.Renderer`: `ParticleFieldPass` writes `bp.lifetime` for a device
emitter and leaves it at 0 for a host one, and the vertex stage then pins
`ageFrac = 0`. So on this field `fade_power`, `size_taper` and the hot->cool
colour ramp are silently no-ops, and the only per-particle channels that
survive are the RADIUS (w, under `WSemantic.Radius`) and the hashed
`bright_jitter`. Which is fine, because a firework is not a steady-state
emitter -- it is one population per shell with one clock -- and a closed form
in t can write the radius directly. Concretely:

  * a star fades by SHRINKING (r -> 0 over its own jittered life) and the
    population THINS (each star dies at its own time), so the cloud goes out
    the way a real one does, star by star, instead of dimming as a slab;
  * the palette is per FIELD, not per particle, so `update` cross-fades
    `color_hot`/`color_cool` between the shells that are currently burning,
    weighted by how much light each is throwing. The default cue sheet spaces
    the shells so at most one burst dominates at a time and the blend is never
    visible; a denser sheet will show it;
  * the streak is the one thing the field does for us: `host_stable_slots`
    promises slot i is the same particle every submit, which makes the
    previous ring slot a real prevPositions buffer and `stretch_seconds` the
    exact analytic velocity smear. That is what the ascending shell is.

Slot layout is FIXED for life (that is what "stable slots" means): each shell
owns a contiguous block of `trail + stars` slots. The trail slots draw the
sparks the rising shell sheds, then become the burst's white core the moment it
opens -- they are already sitting at the burst centre, so nothing has to move
across the frame and no reborn slot streaks in from somewhere else.

Vulkan only: `ParticleField` draws nothing on the GL backend (the PointLights
still flash). Creating a field is a STRUCTURAL scene change -- entry
re-expansion, a device wait, a cleared TAA history -- so the show creates its
one field in `__init__`, at the scene build, and never touches the count of
anything again.

TWO NUMBERS THAT DECIDE WHETHER IT READS
----------------------------------------
Both were found by looking, not by reasoning:

  * FIELD INTENSITY IS SET BY OVERDRAW. These quads are additive and a burst is
    a thousand of them inside a 30 m ball seen from 110 m. At 2.6 the finale
    came out as a featureless white disc 15 degrees across -- big, dramatic and
    with no stars in it. 0.60, with a 0.23 m star, resolves into individual
    stars and still clips its cores to white where they overlap, which is what
    a firework does.
  * STAR SIZE AND TRAIL SIZE ARE SEPARATE KNOBS. Shrinking the star to fix the
    overdraw dashed the ascent, because a trail sample only merges with the next
    one if `radius + speed * stretch_seconds` exceeds the gap between them.
    Hence `Shell.trail_radius`.

The billboards are fogged with the scene, so a scene's fog clamp applies to
them too.

Standalone check (a Vulkan build; headless, no display)::

    python fireworks.py                        # seven verification stills
    python fireworks.py --shot fw.png --t 9.3
    python fireworks.py --film fw.mp4          # -2 -> 13 s
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Appended, not inserted, so a caller run from elsewhere still gets its own
# modules first and `demo_common` (standalone only) resolves either way.
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threepp as tp


# --------------------------------------------------------------------------- #
#  One shell
# --------------------------------------------------------------------------- #
class Shell:
    """The cue for one shell. Times are ABSOLUTE scene seconds.

    `apex` is metres ABOVE the launch point, not above the ground, so a show
    moved up or down a slope keeps its shape.
    """

    def __init__(self, t_launch, apex=58.0, rise=2.7, stars=640,
                 speed=40.0, drag=2.05, life=3.0, star_radius=0.23,
                 trail_radius=0.30,
                 hot=(1.00, 0.76, 0.34), cool=(1.00, 0.34, 0.08),
                 light=(1.00, 0.80, 0.52), light_peak=600.0,
                 light_range=340.0, trail=48, drift=(0.0, 0.0), spin=0.0):
        self.t_launch = float(t_launch)
        self.apex = float(apex)
        self.rise = float(rise)
        self.stars = int(stars)
        self.speed = float(speed)
        self.drag = float(drag)          # linear drag, 1/s: terminal fall = g/k
        self.life = float(life)
        self.star_radius = float(star_radius)
        # NOT derived from star_radius: the two answer different questions. A
        # star is sized against the burst's OVERDRAW (a thousand additive quads
        # inside 30 m), a trail spark against the GAP between two samples of the
        # rising shell's path.
        self.trail_radius = float(trail_radius)
        self.hot = tuple(float(c) for c in hot)
        self.cool = tuple(float(c) for c in cool)
        self.light = tuple(float(c) for c in light)
        self.light_peak = float(light_peak)
        self.light_range = float(light_range)
        self.trail = int(trail)
        self.drift = (float(drift[0]), float(drift[1]))
        self.spin = float(spin)          # rad/s of roll on the star pattern

    @property
    def t_burst(self):
        return self.t_launch + self.rise

    @property
    def slots(self):
        return self.trail + self.stars

    # The last instant this shell puts ANYTHING in the frame. `life` is the
    # nominal star life; the per-star jitter below stretches it by 1.15.
    @property
    def t_end(self):
        return self.t_burst + self.life * 1.15 + 0.05


# --------------------------------------------------------------------------- #
#  The show
# --------------------------------------------------------------------------- #
class FireworkShow:
    """A staggered firework fired from one point on the ground.

    ``FireworkShow(scene, launch_pos, shells=None, seed=11,
                   intensity=0.60, size_scale=1.0)``

      scene       the tp.Scene; the field and the burst lights are added to it
      launch_pos  (x, y, z) world metres -- the mortar, on the ground
      shells      a sequence of Shell, or of (t_launch, apex) /
                  (t_launch, apex, kwargs_dict) tuples. None = default_shells(0)
      seed        every direction, life and twinkle is hashed from this
      intensity   the field's billboard intensity: set by OVERDRAW, see the
                  module docstring, and brought DOWN when size_scale goes up
      size_scale  screen-coverage multiplier on every star and spark; the one
                  knob that helps when the camera is far enough for a star to
                  be sub-pixel

    ``show.update(t)`` -- absolute scene seconds, any t, in any order. Call it
    once per frame before ``renderer.render`` with the SAME clock the renderer
    is pinned to.
    """

    def __init__(self, scene, launch_pos, shells=None, seed=11,
                 intensity=0.60, size_scale=1.0):
        self.origin = np.array(launch_pos, np.float64)
        self.shells = [self._as_shell(s) for s in
                       (FireworkShow.default_shells(0.0) if shells is None else shells)]
        self.seed = int(seed)

        # ---- slot layout, fixed for life ----------------------------------- #
        self._base = []
        n = 0
        for sh in self.shells:
            self._base.append(n)
            n += sh.slots
        self.capacity = n

        # ---- the per-shell constants, drawn ONCE ---------------------------- #
        # Everything random about a star is drawn here and never again: the
        # update is then a pure function of t over frozen arrays, which is what
        # makes a seek reproducible.
        self._star = []
        for i, sh in enumerate(self.shells):
            rng = np.random.default_rng(self.seed * 7919 + i * 101)
            self._star.append(self._draw_stars(sh, rng))
        self._trail = []
        for i, sh in enumerate(self.shells):
            rng = np.random.default_rng(self.seed * 104729 + i * 31)
            self._trail.append(dict(
                jit=rng.random(sh.trail),
                lat=(rng.random((sh.trail, 3)) - 0.5),
                core=(rng.random((sh.trail, 3)) - 0.5),
                bright=rng.random(sh.trail)))

        # ---- the field ------------------------------------------------------ #
        cfg = tp.ParticleField.Config()
        cfg.capacity = self.capacity
        cfg.ownership = tp.ParticleField.Ownership.HostRing
        cfg.w_semantic = tp.ParticleField.WSemantic.Radius   # w IS the radius
        cfg.uniform_radius = 0.45
        # THE STREAK LIVES OR DIES ON THIS FLAG. Without the stable-slot promise
        # the previous ring slot is not the previous step of the same particle,
        # so the pass refuses to publish stretchOverDt and every star is a round
        # dot -- a shell that rises as a dot is a flare, not a firework.
        cfg.host_stable_slots = True
        self.field = tp.ParticleField.create(cfg)
        self.field.name = "fireworks"
        # The bounds are a lie the moment a shell opens, and they are recomputed
        # from a geometry this field does not have anyway.
        self.field.frustum_culled = False
        # INTENSITY IS SET BY OVERDRAW, NOT BY TASTE. These quads are ADDITIVE:
        # a thousand stars inside a 30 m ball, seen from 110 m, cover most of
        # the disc they occupy, and at intensity 2.6 the burst came out as a
        # featureless white circle 15 degrees across with a bloom halo -- big
        # and dramatic and completely structureless. Star size and field
        # intensity are the two knobs on that, and both had to come down.
        self.field.set_billboard_repr(tp.Color(*self.shells[0].hot),
                                      tp.Color(*self.shells[0].cool),
                                      float(intensity), float(size_scale))
        b = self.field.billboard_repr
        b.softness = 0.22            # a spark with a little halo, not a puff
        b.bright_jitter = 0.55       # hashed per slot: the only per-star dimmer
        b.stretch_seconds = 0.035    # the ascent, and the first half-second out
        b.stretch_max = 22.0         # in RADII, so a 0.4 m star caps at ~9 m
        b.stretch_max_screen = 0.06  # and never paints a bar across the frame
        b.near_fade = 2.5
        b.glow = 0.55                # its own bloom pyramid: this is the flash
        b.glow_threshold = 0.0
        scene.add(self.field)

        # ---- one light per shell, all created here -------------------------- #
        # A PointLight is what actually puts the burst on the water and the
        # ground; the billboards are composited late and light nothing.
        # Created up front and left at zero intensity, because adding a light
        # mid-scene is the same structural churn the single field avoids.
        self.lights = []
        for sh in self.shells:
            L = tp.PointLight(tp.Color(*sh.light), 0.0, sh.light_range)
            L.cast_shadow = False        # a 500-star flash does not want a map
            L.position.set(float(self.origin[0]), float(self.origin[1]),
                           float(self.origin[2]))
            scene.add(L)
            self.lights.append(L)

        # ---- the submit buffer ---------------------------------------------- #
        # Parked slots sit AT the mortar with w < 0 (the dead sentinel). Not
        # underground: the streak reads the previous ring slot whatever the
        # sentinel said, so a slot reborn far from where it parked streaks in
        # from there for one frame. Parking on the launch point, and moving the
        # trail slots with the shell, keeps every rebirth displacement ~0.
        self.buf = np.zeros((self.capacity, 4), np.float32)
        self.buf[:, :3] = self.origin
        self.buf[:, 3] = -1.0
        self._prev_t = None
        self.update(self.shells[0].t_launch - 1.0)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def default_shells(t0=0.0):
        """A three-shell cue sheet, relative to the first launch at `t0`.

        Two warm gold either side of one blue, the last the biggest and
        highest so the show builds. Bursts land at t0+1.5, t0+4.7 and t0+8.5.
        """
        # THE APEXES ARE FRAMED, NOT CHOSEN. For the reference lens (eye 52 m
        # above the mortar and 112 m out from it, 38 deg vertical fov aimed
        # 9 deg down) the top of the frame is 71 m above the mortar at the
        # burst's range, and a star cloud is speed/drag = ~18 m in radius. So a
        # burst centre much above 52 m throws its top stars out of the picture.
        # Raise these and the finale loses its crown.
        # FAST. The first cue sheet ambled -- 2.6 s rises, 3 s star lives, five
        # and six seconds between shells -- and watched at film pace it read as
        # slow motion. A real shell is up in about a second and a half and its
        # break is over in two; speed and drag move TOGETHER (the cloud radius
        # is speed/drag, its opening time 1/drag), so the burst opens in a
        # third of a second at the same size it always had.
        return [
            Shell(t0 + 0.0, apex=45.0, rise=1.5, stars=640, speed=44.0,
                  drag=3.20, life=1.9, star_radius=0.22, trail=56,
                  hot=(1.00, 0.58, 0.16), cool=(1.00, 0.24, 0.04),
                  light=(1.00, 0.78, 0.46), light_peak=560.0,
                  drift=(-1.5, 3.0)),
            # The coloured one. A cold burst between two gold ones is what makes
            # the gold read AS gold; on its own it just looks like embers.
            Shell(t0 + 3.2, apex=49.0, rise=1.5, stars=700, speed=47.0,
                  drag=3.20, life=2.0, star_radius=0.23, trail=56,
                  hot=(0.26, 0.78, 1.00), cool=(0.10, 0.28, 1.00),
                  light=(0.55, 0.80, 1.00), light_peak=520.0,
                  drift=(2.5, 1.5)),
            # The finale. Higher, wider, more stars, longer legs.
            Shell(t0 + 6.8, apex=52.0, rise=1.7, stars=1100, speed=52.0,
                  drag=3.20, life=2.3, star_radius=0.25, trail=64,
                  hot=(1.00, 0.64, 0.20), cool=(1.00, 0.22, 0.04),
                  light=(1.00, 0.82, 0.50), light_peak=980.0,
                  light_range=380.0, drift=(-0.5, 4.0)),
        ]

    @staticmethod
    def _as_shell(s):
        if isinstance(s, Shell):
            return s
        if len(s) >= 3 and isinstance(s[2], dict):
            return Shell(s[0], apex=s[1], **s[2])
        return Shell(s[0], apex=s[1])

    # ---------------------------------------------------------------------- #
    def _draw_stars(self, sh, rng):
        """Freeze one shell's star pattern: directions, speeds, lives, twinkle."""
        n = sh.stars
        # A Fibonacci sphere, randomly rotated. Uniform random directions clump
        # -- at 640 samples the clumps are visible as blotches on the shell and
        # the burst reads as a spray rather than as an expanding sphere.
        k = np.arange(n, dtype=np.float64) + 0.5
        cz = 1.0 - 2.0 * k / n
        r = np.sqrt(np.clip(1.0 - cz * cz, 0.0, 1.0))
        phi = k * math.pi * (3.0 - math.sqrt(5.0))
        d = np.stack([r * np.cos(phi), cz, r * np.sin(phi)], axis=1)
        # Random rotation, so two shells of the same size are not the same shell.
        a, b, c = rng.random(3) * 2.0 * math.pi
        for axis, ang in ((1, a), (0, b), (2, c)):
            ca, sa = math.cos(ang), math.sin(ang)
            i, j = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
            u, v = d[:, i].copy(), d[:, j].copy()
            d[:, i] = u * ca - v * sa
            d[:, j] = u * sa + v * ca
        # Speed spread is what gives the shell THICKNESS. A single speed makes a
        # soap bubble: a perfect thin shell that reads as a wireframe sphere.
        spd = sh.speed * (0.80 + 0.34 * rng.random(n))
        return dict(
            v0=(d * spd[:, None]),
            life=sh.life * (0.62 + 0.55 * rng.random(n)),
            r0=sh.star_radius * (0.72 + 0.60 * rng.random(n)),
            # The glitter: each star crackles at its own rate and phase, which
            # is the difference between "embers" and "a firework".
            tw_w=16.0 + 22.0 * rng.random(n),
            tw_p=rng.random(n) * 2.0 * math.pi,
            tw_a=0.25 + 0.55 * rng.random(n))

    # ---------------------------------------------------------------------- #
    def _shell_pos(self, sh, tau):
        """Where the mortar shell is `tau` seconds after launch (vectorised)."""
        u = np.clip(np.asarray(tau, np.float64) / sh.rise, 0.0, 1.0)
        # y = apex * (2u - u^2): a parabola that arrives at the apex with EXACTLY
        # zero vertical speed, so the streak stops before it opens instead of
        # being cut off mid-climb.
        y = self.origin[1] + sh.apex * (2.0 * u - u * u)
        x = self.origin[0] + sh.drift[0] * u
        z = self.origin[2] + sh.drift[1] * u
        return x, y, z

    # ---------------------------------------------------------------------- #
    def update(self, t):
        """Advance the show to absolute scene time `t`. Closed form; any t."""
        t = float(t)
        buf = self.buf
        # Everything defaults to parked-at-the-mortar. Blocks that have anything
        # to say overwrite their own rows below.
        buf[:, 3] = -1.0

        pal_hot = np.zeros(3)
        pal_cool = np.zeros(3)
        pal_w = 0.0

        for i, sh in enumerate(self.shells):
            b0 = self._base[i]
            tau = t - sh.t_launch
            L = self.lights[i]
            if tau < 0.0 or t > sh.t_end:
                buf[b0:b0 + sh.slots, :3] = self.origin
                L.intensity = 0.0
                continue

            if tau < sh.rise:
                self._rise(sh, i, b0, tau)
                # The shell itself is a lamp, but a small one -- a 2 cm ball of
                # burning composition, not the burst. Enough to touch the water
                # under it on the way up.
                sx, sy, sz = self._shell_pos(sh, tau)
                L.position.set(float(sx), float(sy), float(sz))
                L.intensity = min(sh.light_peak * 0.05, 130.0)
                w = 0.05
            else:
                s = tau - sh.rise
                self._burst(sh, i, b0, s)
                cx, cy, cz = self._shell_pos(sh, sh.rise)
                L.position.set(float(cx), float(cy), float(cz))
                # The envelope: a hard flash that is gone in a tenth of a second
                # over a tail that carries the burning stars. Both zero outside
                # the burst -- an always-on light at 0.001 is still a light in
                # every shading loop in the frame.
                env = math.exp(-s / 0.085) + 0.42 * math.exp(-s / 0.85)
                env *= max(0.0, 1.0 - (s / (sh.life * 1.15)) ** 3)
                L.intensity = sh.light_peak * env
                w = 0.25 + env

            pal_w += w
            pal_hot += np.array(sh.hot) * w
            pal_cool += np.array(sh.cool) * w

        # ---- the palette ---------------------------------------------------- #
        # Per-particle colour does not exist on a host field, so the FIELD is
        # recoloured every frame toward whichever shell is currently throwing
        # the most light. The cue sheet spaces the bursts so this is a step
        # function in practice; the blend only matters while one shell is still
        # dying and the next is on its way up.
        if pal_w > 1e-6:
            bb = self.field.billboard_repr
            h = pal_hot / pal_w
            c = pal_cool / pal_w
            bb.color_hot = tp.Color(float(h[0]), float(h[1]), float(h[2]))
            bb.color_cool = tp.Color(float(c[0]), float(c[1]), float(c[2]))

        # ---- submit ---------------------------------------------------------- #
        # dt is what the stretch divides the frame displacement by. A seek (or
        # the very first frame) has no previous step, so fall back to the
        # renderer's own assumption rather than smearing over a 20-second gap.
        dt = 1.0 / 60.0
        if self._prev_t is not None:
            d = t - self._prev_t
            if 1e-4 < d < 0.2:
                dt = d
        self._prev_t = t
        self.field.submit(buf, dt)

    # ---------------------------------------------------------------------- #
    def _rise(self, sh, i, b0, tau):
        """The ascent: a head plus the sparks it has shed, sampled BACK in t."""
        tr = self._trail[i]
        n = sh.trail
        k = np.arange(n, dtype=np.float64)
        # Sample the shell's own path at earlier times. The trail is then
        # exactly the curve the head flew, which is free and always correct --
        # no integration, no history buffer.
        # 14 ms apart. Wider and the trail BEADS: at 35 m/s the samples sit
        # further apart than the stretched quad is long and the ascent reads as
        # a dotted line rather than as a burning fuse. The spacing has to be
        # read together with `trail_radius` and `stretch_seconds` -- a quad is
        # radius + velocity*stretch long, and the samples must be closer than
        # that. Shrinking the STARS (which the burst's overdraw wanted) is what
        # dashed this trail the first time it was tuned, which is why the trail
        # carries its own radius rather than deriving one from the star.
        back = k * 0.014 * (0.75 + 0.5 * tr["jit"])
        tb = tau - back
        alive = tb >= 0.0
        x, y, z = self._shell_pos(sh, np.maximum(tb, 0.0))
        age = np.maximum(back, 0.0)
        # A shed spark drifts sideways and sags. Small: this is smoke-scale, and
        # a trail that fans is a rocket, not a mortar shell.
        x = x + tr["lat"][:, 0] * age * 1.9
        z = z + tr["lat"][:, 2] * age * 1.9
        y = y - 2.2 * age * age
        r = sh.trail_radius * (0.95 - 0.78 * (k / max(n - 1, 1))) * \
            (0.6 + 0.8 * tr["bright"])
        # Fade the whole trail in over the first fifth of a second, so the
        # mortar does not pop into being at full length.
        r *= min(1.0, tau / 0.20)
        rows = slice(b0, b0 + n)
        self.buf[rows, 0] = x
        self.buf[rows, 1] = y
        self.buf[rows, 2] = z
        self.buf[rows, 3] = np.where(alive & (r > 0.01), r, -1.0)
        # The head is the brightest thing on the way up.
        self.buf[b0, 3] = sh.trail_radius * 1.5 * min(1.0, tau / 0.12)

    # ---------------------------------------------------------------------- #
    def _burst(self, sh, i, b0, s):
        """The burst, `s` seconds after it opened. Closed form with drag."""
        st = self._star[i]
        cx, cy, cz = self._shell_pos(sh, sh.rise)
        c = np.array([cx, cy, cz], np.float64)

        # ---- the core flash, drawn on the trail slots ----------------------- #
        # They are already parked at the burst centre, so nothing has to travel
        # and no reborn slot streaks in from anywhere.
        tr = self._trail[i]
        n = sh.trail
        rows = slice(b0, b0 + n)
        cf = max(0.0, 1.0 - s / 0.30)
        core = c[None, :] + tr["core"] * (1.2 + 26.0 * s)
        self.buf[rows, :3] = core
        self.buf[rows, 3] = np.where(cf > 0.0,
                                     sh.star_radius * (0.6 + 7.0 * cf * cf), -1.0)

        # ---- the stars ------------------------------------------------------ #
        # dv/dt = -k v + g has an exact solution, so the whole cloud is one
        # expression in s: no state, no substepping, and a seek to any frame of
        # the fall lands on the same metre it would have integrated to.
        k = sh.drag
        g = np.array([0.0, -9.81, 0.0])
        gk = g / k
        E = math.exp(-k * s)
        p = c[None, :] + (st["v0"] - gk[None, :]) * ((1.0 - E) / k) + gk[None, :] * s

        a = np.clip(s / st["life"], 0.0, 1.0)
        # The size IS the fade here: (1-a)^0.55 holds the star bright most of
        # its life and then lets it go, which is what a burning star does. The
        # population thins at the same time because every life is different, so
        # the cloud goes out star by star instead of dimming as one slab.
        r = st["r0"] * np.power(1.0 - a, 0.55)
        # Ignition: for a sixth of a second the stars are a fat white ball.
        if s < 0.18:
            r = r * (1.0 + 2.4 * (1.0 - s / 0.18))
        # Glitter, once the shell has opened out. Per-star rate and phase.
        gl = np.sin(st["tw_w"] * s + st["tw_p"])
        r = r * (1.0 - st["tw_a"] * np.clip((a - 0.18) / 0.25, 0.0, 1.0) * (gl * gl))

        rows = slice(b0 + n, b0 + n + sh.stars)
        self.buf[rows, :3] = p
        self.buf[rows, 3] = np.where(a < 1.0, np.maximum(r, 0.0), -1.0)


# --------------------------------------------------------------------------- #
#  Standalone verification scene
#
#  A shore at blue hour: sea, a slab of shingle with a few boulders on it for
#  the flash to land on, and the mortar on the shingle. The camera is the
#  reference lens the defaults were framed for -- 52 m above the mortar, 112 m
#  out from it, aimed 9 degrees down over the water -- so a look tuned here
#  transfers to any scene that watches from a similar range.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from demo_common import Encoder, cli_arg, find_ffmpeg

    if not tp.HAS_VULKAN:
        print("This threepp build has no Vulkan backend "
              "(configure with -DTHREEPP_WITH_VULKAN=ON).")
        sys.exit(0)

    SHOT = "--shot" in sys.argv
    FILM = "--film" in sys.argv
    OUT = cli_arg("--shot", "fireworks.png", str)
    FILM_OUT = cli_arg("--film", "fireworks.mp4", str)
    T = cli_arg("--t", 9.3, float)
    W, H = cli_arg("--width", 1100, int), cli_arg("--height", 900, int)
    FPS = 30
    SEA_Y = 0.0
    SHORE_Y = 2.3                       # the shingle the mortar stands on
    LAUNCH = (0.0, SHORE_Y + 0.6, 0.0)

    canvas = tp.Canvas("threepp - fireworks", width=W, height=H,
                       vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    # The exposure is AUTHORED. Auto-exposure was measured not to move a
    # headless Vulkan frame at all, and an offline capture wants one
    # deterministic level anyway: every still here is comparable to every
    # other because nothing metered them differently.
    renderer.auto_exposure = False
    renderer.tone_mapping_exposure = cli_arg("--exposure", 1.0, float)
    renderer.render_scale = 0.9
    renderer.gbuffer_msaa = 2
    renderer.bloom_intensity = 0.14
    renderer.bloom_clamp = 12.0
    renderer.starfield = 0.30

    scene = tp.Scene()

    # A blue-hour dome: cold overhead, a low warm afterglow lobe left over in
    # the north west. Built in a dozen lines so this file needs no asset.
    SW, SH_ = 256, 128
    el = np.linspace(-math.pi / 2, math.pi / 2, SH_)[:, None]
    az = np.linspace(0.0, 2.0 * math.pi, SW, endpoint=False)[None, :]
    up = np.clip(np.sin(el), 0.0, 1.0) + 0.0 * az
    zen = np.array([0.010, 0.020, 0.048])
    hor = np.array([0.055, 0.075, 0.115])
    rgb = hor[None, None, :] * (1.0 - up)[..., None] + zen[None, None, :] * up[..., None]
    # A broad afterglow lobe sitting ON the horizon. Broad on purpose: a tight
    # one is a sun, and the sun is well down at blue hour.
    glow = (np.clip(np.cos(el), 0.0, 1.0) ** 10) * \
        (np.clip(np.cos(az - 4.2), 0.0, 1.0) ** 3)
    rgb += glow[..., None] * np.array([0.075, 0.038, 0.020])[None, None, :]
    sky = np.zeros((SH_, SW, 4), np.float32)
    sky[..., :3] = np.clip(rgb, 0.0, None)
    sky[..., 3] = 1.0
    env = tp.float_texture(sky)
    scene.environment = env
    scene.background = env
    scene.set_fog_exp2(tp.Color(0.055, 0.075, 0.115), 0.00085)
    renderer.fog_anisotropy = 0.45
    renderer.set_height_fog(density=0.016, base_y=SEA_Y + 0.3,
                            falloff=6.0, noise_amount=0.8)

    # BELOW the horizon, which is what blue hour means. Above it the renderer
    # draws the sun disc, and the disc landed exactly where the shells burst --
    # the first pass of this rig looked like the bursts were not rendering at
    # all when they were simply inside a blown-out sun.
    renderer.sun_angular_radius = 0.6
    sun = tp.DirectionalLight(0xffd8b0, 0.05)
    sun.position.set(-260.0, -60.0, 140.0)
    scene.add(sun)

    def _mat(hex_col, rough, metal=0.0):
        m = tp.MeshStandardMaterial()
        m.color = tp.Color(hex_col)
        m.roughness = rough
        m.metalness = metal
        return m

    # The shore, as one slab: the point of this rig is what the flash lands on,
    # and a plane takes light the same way a hillside does.
    shore = tp.Mesh(tp.PlaneGeometry(520.0, 300.0), _mat(0x2a2620, 0.95))
    shore.rotate_x(-math.pi / 2)
    shore.position.set(-17.0, SHORE_Y, -150.5)
    shore.receive_shadow = True
    scene.add(shore)
    # A few boulders on the beach, purely so the flash has something with a
    # normal on it near the mortar -- a flat plane hides a lighting mistake.
    for bx, bz, br in ((0.0, 3.5, 2.2), (13.0, -4.5, 3.0), (-5.0, -10.5, 1.6)):
        r = tp.Mesh(tp.SphereGeometry(br, 20, 12), _mat(0x3a352c, 0.92))
        r.position.set(bx, SHORE_Y + br * 0.35, bz)
        scene.add(r)

    ocean = tp.Ocean(size=900.0, resolution=384, wind_speed=5.0, wind_theta=0.9,
                     choppiness=0.42, fetch=14e3)
    ocean.position.y = SEA_Y
    scene.add(ocean)

    show = FireworkShow(scene, LAUNCH, FireworkShow.default_shells(0.0), seed=11)

    cam = tp.PerspectiveCamera(38.0, W / float(H), 0.35, 3000)

    def place(t):
        # The reference lens, with a slow drift so a still is not a frame of a
        # locked-off camera: 52 m above the mortar, 112 m out, aimed 9 deg down.
        a = math.radians(4.0 + 0.9 * max(t + 1.0, 0.0))
        cam.position.set(-12.0 + 10.0 * math.sin(a), SHORE_Y + 52.4,
                         -116.5 + 4.0 * math.cos(a))
        cam.look_at(tp.Vector3(-33.0, 22.0, 85.5))

    def step(t):
        renderer.sim_time = t
        place(t)
        show.update(t)

    def settle(t, lead=1.0):
        """Walk INTO t at film rate, from `lead` seconds before it.

        A still rendered cold is a lie: the velocity streak is (pos - previous
        slot) and needs a real previous step, and the TAA history wants a few
        frames of the same motion behind it.
        """
        dt = 1.0 / FPS
        for k in range(int(lead * FPS), 0, -1):
            step(t - k * dt)
            renderer.render(scene, cam)
        step(t)

    if FILM:
        if find_ffmpeg() is None:
            sys.exit("--film needs ffmpeg on PATH (or `pip install imageio-ffmpeg`)")
        t0, t1 = -2.0, 13.0
        enc = Encoder(FILM_OUT, W, H, FPS, crf=18, preset="medium")
        n = int((t1 - t0) * FPS)
        for k in range(n):
            step(t0 + k / FPS)
            renderer.render(scene, cam)
            enc.send(renderer.read_pixels())
        enc.close()
        print(f"wrote {FILM_OUT}  ({n} frames, {t0}-{t1} s)")

    elif SHOT:
        settle(T)
        renderer.save_frame(scene, cam, OUT)
        print(f"wrote {OUT}  (t={T:.2f}s)")

    else:
        # The verification set: one still per beat that has to work.
        # The times matter. A shell 0.3 s after break is a compact ball however
        # it is tuned -- that is what a break IS -- and judging the burst there
        # says nothing about whether it opens. The frames that decide it are
        # ~1 s (open, discrete stars) and ~2.5 s (falling, thinning, dying).
        #
        # And they are captured out of ONE continuous walk from t = -2, not one
        # cold render each, for the reasons `settle` gives.
        marks = {0.9: "ascent",       # shell 1 climbing, the streak reading
                 1.55: "flash1",      # the break itself
                 2.3: "open1",        # gold, 0.8 s open
                 3.1: "fade1",        # embers falling, thinning out
                 5.5: "burst2",       # the blue one, 0.8 s open
                 9.3: "burst3",       # the finale, biggest
                 10.6: "fall3"}       # falling, still well above the water
        todo = sorted(marks)
        t0, t1 = -2.0, 11.4
        n = int((t1 - t0) * FPS)
        for k in range(n + 1):
            t = t0 + k / FPS
            step(t)
            renderer.render(scene, cam)
            if todo and t + 0.5 / FPS >= todo[0]:
                p = f"fw_{marks[todo[0]]}.png"
                renderer.save_frame(scene, cam, p)
                print(f"wrote {p}  (t={t:.2f}s, cue {todo[0]:.2f})")
                todo.pop(0)
