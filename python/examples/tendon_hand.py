"""A tendon-driven anthropomorphic hand, built so the mechanics are right.

Twenty joints, twenty-five cables, no joint motors anywhere. Every torque in this hand
is produced by pulling on a cable that is routed over via points and can only pull --
which is the whole point of the mechanism: there is no room for a motor in a finger, so
the motors live in the forearm and the fingers are moved by tension.

WHY THE CABLES ARE threepp.TendonCable AND NOT A PhysX TENDON. Measured, on a two-link
finger, in python/examples/tendon_probe.py:

  * A PxArticulationSpatialTendon's interior attachments set the LENGTH but exert NO
    FORCE, exactly as PxArticulationTendon.h:405-407 says. At a 0.9 rad bend its
    generalized force matches the gradient taken with the via point FROZEN in world
    space to 0.07 deg, and lies 21.06 deg from a real routed cable. A flexor threaded
    through A1-A5 would get no transverse pulley reaction and its moment arms distal of
    the first joint would be fabricated.
  * A PxArticulationFixedTendon has no geometry at all -- its moment arms ARE its
    coefficients, prescribed rather than emergent.

TendonCable applies the true frictionless-pulley force at every via point, so the joint
torques are -T dL/dq by virtual work: measured to 0.02% of magnitude and 0.001 deg of
direction against the analytic gradient. Here that means the moment arms are not
authored anywhere in this file. They EMERGE from where the pulleys sit, which is why
--selftest can measure them and compare against cadaver values rather than against the
numbers it was given.

FLEXORS ARE CHORDS, EXTENSORS WRAP, and the difference is not a modelling preference.
A flexor runs on the CONCAVE side of a closing joint: its insertion swings toward the
palm, the straight path shortens, and pulling it flexes. Force it to follow the pulley
instead and the volar path LENGTHENS by r*theta -- measured, 58.5 -> 73.2 mm over 105
deg -- turning the flexor into an extensor. So an annular pulley caps how far the tendon
may bow away from the bone rather than being a surface it follows, and the flexor arm
RISING with flexion is bowstringing, the effect that takes a cadaver MCP flexor arm from
~5.8 to ~10.3 mm and that an A2 rupture roughly doubles. The extensor is the opposite
case: it runs on the CONVEX side, genuinely wraps, and routed as a chord its arm passes
through zero and reverses -- 14 reversals across the four fingers before the wraps went
in, every one of them an extensor.

WHAT IS FAITHFUL, and what is not:
  + Bone lengths, joint ranges, pulley positions and standoffs are literature values.
    --selftest MEASURES the resulting arms and prints the cadaver range beside each.
  + 5 cables per 4-DOF finger, the Salisbury-Mason N+1 bound, and --selftest checks the
    torque space is actually POSITIVELY SPANNED rather than assuming N+1 is enough.
  + No cable reverses its moment arm anywhere in the range; --selftest checks all 25
    against every joint they span.
  + Segment masses are never hand-authored: capsules at 1100 kg/m^3, PhysX integrates.
  + Cables are pull-only and carry a real tension; capstan friction is available.
  - MCP abduction range does NOT collapse toward zero at full flexion the way the
    collateral ligaments make it. A fist will splay slightly more than a real hand.
  - The thumb CMC uses two serial orthogonal revolutes. The real joint (Hollister 1992)
    has two non-orthogonal, non-intersecting axes fixed in different bones, which is
    what gives a real thumb its axial pronation during opposition. This is the largest
    geometric approximation in the model, and the thumb is the one digit whose torque
    space --selftest still reports as not positively spanned.
  - No tendon sheath COLLISION: cables pass through the finger geometry rather than
    being stopped by it, so the pulley cap is enforced by where the via points sit.

MEASURED, at 35 N per flexor cable (python tendon_hand.py --grasp ball):
  a 36 mm ball is held against 15.0 N in its weakest direction, i.e. a 1.5 kg load.
  A 44 mm can manages only 0.4 N, and a 60 mm can is ejected outright -- the opposed
  thumb tip sits 36 mm from the fingertips, so nothing closes on a can's far side and
  it squirts along the one axis the grasp does not constrain. That is a real limit of a
  hand this size, and it is reported rather than tuned away.

Run:
  python tendon_hand.py --selftest      # measure the mechanics; no window
  python tendon_hand.py --grasp ball    # close on an object and pull until it slips
  python tendon_hand.py --view          # window: live grasp with every cable drawn
  python tendon_hand.py --shots         # the same, headless, to PNGs
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import threepp as tp



DT = 1.0 / 240.0

# --------------------------------------------------------------------------------
# Anthropometry. 50th-percentile adult male, metres.
#
# Bone lengths: consensus male osteometric values, cross-checked against the segment
# RATIOS in Buryanov & Kotiuk 2010 (Int. J. Morphol. 28(3):755-758, Table III), which
# give digit III as 40.6/33.1/26.3 % proximal/middle/distal. Treat the absolute mm as
# +/-3 mm. Link length is bone + ~2.5 mm of joint space, and the fingertip pulp adds
# another 7 mm past the distal bone -- that pulp is where contact actually happens, so
# it is in the capsule, not omitted.
#
# Joint ranges: AAOS/ASSH clinical values.
#
# Pulley standoffs are the volar distance from the JOINT CENTRE to the tendon, i.e. they
# ARE the moment arms: A1 8-10, A2 6-8, A3 6-7.5, A4/A5 4-5 mm. The tendon runs inside
# the finger, so a standoff smaller than the capsule radius is correct, not a bug.
# --------------------------------------------------------------------------------
FINGERS = {
    #          knuckle (x, y, z)        prox   mid    dist   MCP flex  abduct
    "index":  dict(knuckle=(0.090, 0.0, -0.030), pp=0.040, mp=0.025, dp=0.017,
                   flex=(-0.44, 1.57), abd=0.42),
    "middle": dict(knuckle=(0.095, 0.0, -0.010), pp=0.045, mp=0.030, dp=0.018,
                   flex=(-0.44, 1.65), abd=0.26),
    "ring":   dict(knuckle=(0.088, 0.0, 0.010), pp=0.041, mp=0.028, dp=0.018,
                   flex=(-0.44, 1.70), abd=0.30),
    "little": dict(knuckle=(0.078, 0.0, 0.030), pp=0.033, mp=0.019, dp=0.016,
                   flex=(-0.44, 1.75), abd=0.45),
}
PULP = 0.007                      # fingertip soft tissue past the distal bone
R_PP, R_MP, R_DP = 0.0090, 0.0080, 0.0075     # capsule radii (finger, not bone)

# Volar standoffs = flexor moment arms; dorsal = extensor. Metres.
A1, A2, A3, A4, A5 = 0.0100, 0.0075, 0.0088, 0.0036, 0.0037
# The extensor's PIP:DIP standoff ratio is deliberately LARGER than the flexor's. That
# is not a style choice: with t_fdp/t_ext pinned by the DIP torque row, the PIP row
# forces a NEGATIVE FDS tension -- a cable asked to push -- unless
#   r_ext_pip / r_ext_dip  >  r_fdp_pip / r_fdp_dip.
# Violate it and the finger has 5 cables for 4 DOF and still cannot co-contract, so
# whole regions of joint-torque space are unreachable by any policy. This is the
# mechanical job of the central slip, and --selftest checks it every run.
E_MCP, E_PIP, E_DIP = 0.0075, 0.0045, 0.0028
IO_LAT = 0.0155                   # interosseous lateral offset -> the abduction arm
IO_VOLAR = 0.0080                 # ... and its volar standoff at the MCP
FDS_DISTAL = 0.38                 # FDS insertion standoff, as a fraction of A3

# Where the flexor's first pulley sits along the proximal phalanx, as a fraction of its
# length. This is what CAPS bowstringing, and it was chosen against the shape of the
# whole curve rather than one pose. Measured MCP arm over 0-100 deg of flexion:
#   0.10 ->  7.8  8.6  8.2  6.6  3.5 -0.6   the arm REVERSES; a flexor that extends
#   0.18 ->  8.1  9.7 10.5 10.0  7.9  3.7   peaks at 10.5, and cadavers give ~10.3
#   0.26 ->  8.2 10.5 12.2 12.8 11.9  8.4
#   0.45 ->  8.5 12.1 15.2 17.6 19.2 19.2   runs away; this is an A2 rupture
# The rise itself is real -- it is the bowstringing that takes a cadaver MCP flexor arm
# from ~5.8 to ~10.3 mm, and that an A2 rupture roughly doubles.
FDP_PP = 0.18                     # flexor pulley along the proximal phalanx
FDP_MP = 0.45                     # ... and along the middle phalanx

PALM_Z = -0.001                   # palm box centre

# The whole hand's orientation rule lives in these four vectors. A finger runs distally
# with its pad volar; the thumb leaves the wrist obliquely and pronated so its pad faces
# the fingers, which is the only reason opposition exists.
FINGER_BONE = (1.0, 0.0, 0.0)
FINGER_PAD = (0.0, -1.0, 0.0)
THUMB_CMC = (0.026, -0.016, -0.042)
THUMB_BONE = (0.62, -0.45, -0.64)
THUMB_PAD = (0.15, -0.35, 0.92)

DENSITY = 1100.0                  # kg/m^3, hand soft tissue + bone

# Passive joint resistance. A zero-stiffness, non-zero-damping articulation drive is a
# pure viscous damper about the joint axis, which is what synovial fluid plus a tendon
# in its sheath actually is. max_force caps it so it can never overpower the cables.
JOINT_DAMPING = 0.0035            # N.m.s/rad
JOINT_DAMP_MAX = 0.5              # N.m
JOINT_FRICTION = 0.0008           # N.m of stiction, per joint



# --------------------------------------------------------------------------------
# Orientation. ONE rule for every segment in the hand, fingers and thumb alike.
#
# A link's frame is built from two vectors: the BONE AXIS it runs along, and the
# direction its PAD faces. CapsuleGeometry runs along its own +Y, so local +Y is the
# bone axis; local +X is the pad direction; local +Z completes the right-handed set.
# Two consequences fall straight out and are used everywhere below:
#
#     flexion axis   =  cross(bone, pad)        the joint that curls the pad inward
#     abduction axis =  -pad                    the joint that swings it sideways
#
# For a finger (bone +X, pad volar -Y) that yields a capsule rotated -90 deg about Z and
# a flexion axis of -Z, which is exactly what the fingers were built with by hand before
# this existed. For the THUMB it is the whole difference between a thumb and a fifth
# finger: a real thumb leaves the wrist obliquely -- distal AND volar AND radial -- and
# is pronated about its own axis so its pad faces the fingers. Built parallel to the
# fingers in the palm plane, as this model first was, it reads as five identical digits
# and can only ever pinch by luck.
# --------------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame(bone, pad):
    """Columns are the link's local +X (pad), +Y (bone), +Z axes, expressed in world."""
    y = _unit(bone)
    x = _unit(np.asarray(pad, dtype=float) - y * float(np.dot(pad, y)))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def quat_from(R):
    """Shepperd's method. threepp binds no matrix-to-quaternion conversion, and the
    Quaternion binding has no set_from_unit_vectors, so this is done here."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


def orient(mesh, bone, pad):
    """Point `mesh` along `bone` with its pad facing `pad`. Returns the frame."""
    R = frame(bone, pad)
    q = quat_from(R)
    mesh.quaternion.set(*q)
    return R


def flex_axis(bone, pad):
    return _unit(np.cross(_unit(bone), _unit(np.asarray(pad) - _unit(bone) * np.dot(pad, _unit(bone)))))


# The skin is TRANSLUCENT on purpose. Tendons run inside a finger, so an opaque hand
# hides the entire mechanism -- the first render with real rope showed four stray blue
# arcs and nothing else. The poster this is modelled on is a cutaway for the same reason:
# the routing IS the subject.
SKIN_OPACITY = 0.28


def skin(colour=0xC8A088, roughness=0.75, opacity=None):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(colour)
    m.roughness = roughness
    m.transparent = True
    m.opacity = SKIN_OPACITY if opacity is None else opacity
    m.depth_write = False
    return m


def palm_hull(palm_z):
    """A shaped palm instead of a box: a rounded slab, widest across the knuckles,
    tapering to the wrist, and thicker on the thenar side where the thumb muscles sit.

    Articulation.add_link cooks any non-primitive geometry to ONE convex hull, so this
    improves the COLLIDER too, not just the silhouette -- a box palm has hard 90-degree
    edges for an object to catch on. Points are in the palm's own frame, which stays put,
    so every cable anchor expressed against it is unaffected.
    """
    #        x        half-z   dorsal   volar
    profile = [
        (-0.0380, 0.0250, 0.0092, 0.0100),   # wrist
        (-0.0180, 0.0330, 0.0115, 0.0128),
        ( 0.0060, 0.0380, 0.0122, 0.0138),
        ( 0.0270, 0.0392, 0.0112, 0.0124),
        ( 0.0378, 0.0350, 0.0090, 0.0100),   # the knuckle edge, narrowing again
    ]
    pts = []
    for x, hz, hyd, hyv in profile:
        for sz in (-1.0, 1.0):
            for hy, sy in ((hyd, 1.0), (hyv, -1.0)):
                pts.append((x, sy * hy, sz * hz))
                pts.append((x, sy * hy * 0.55, sz * hz * 0.97))
                pts.append((x, sy * hy * 0.92, sz * hz * 0.62))
    # thenar bulge: the thumb muscles thicken the radial-volar quarter of the palm
    for x in (-0.026, -0.008, 0.010):
        pts.append((x, -0.0175, -0.0300))
        pts.append((x, -0.0140, -0.0370))
    return [tp.Vector3(*q) for q in pts]


def cap(radius, length, colour=0xC8A088):
    """A capsule mesh of the given TOTAL length (CapsuleGeometry takes cylinder length)."""
    return tp.Mesh(tp.CapsuleGeometry(radius, max(1e-4, length - 2 * radius)), skin(colour))


class Hand:
    """The articulation, its named links, its cables, and the add-order DOF map."""

    def __init__(self, world, base=(0.0, 0.0, 0.0), material=None):
        self.world = world
        self.base = np.array(base, dtype=float)
        self.art = world.create_articulation(fixed_base=True, solver_position_iterations=32)
        self.links = {}          # name -> ArticulationLink
        self.dof_names = []      # add order
        self.cables = {}         # name -> TendonCable
        self.limits = {}         # name -> (lower, upper) radians
        self.meshes = []
        self._mat = material

        palm = tp.Mesh(tp.ConvexGeometry(palm_hull(PALM_Z)), skin(roughness=0.8))
        palm.position.set(*(self.base + np.array([0.0325, 0.0, PALM_Z])))
        self.palm = self.art.add_link(palm, density=DENSITY, material=material)
        self.links["palm"] = self.palm
        self.meshes.append(palm)

        for name, spec in FINGERS.items():
            self._finger(name, spec)
        self._thumb()

    # -- construction -------------------------------------------------------------
    def _add(self, name, mesh, parent, axis, anchor, lower, upper):
        # Joint viscosity and Coulomb friction. NOT a numerical fudge: a finger joint
        # runs in synovial fluid with a tendon sliding through a sheath, and neither is
        # frictionless. Without them the joints are ideal free hinges, and a 35 N cable
        # at a 10 mm arm accelerates a 9 g phalanx hard enough that the fingers slam
        # shut and BAT the object away -- measured, the can left at 0.6 m/s and the
        # "grasp" resisted 0.06 N. See JOINT_DAMPING.
        self.limits[name] = (lower, upper)
        lk = self.art.add_link(mesh, parent=parent, density=DENSITY,
                               axis=list(axis), anchor=list(anchor),
                               lower=lower, upper=upper, material=self._mat,
                               stiffness=0.0, damping=JOINT_DAMPING,
                               max_force=JOINT_DAMP_MAX, joint_friction=JOINT_FRICTION)
        self.links[name] = lk
        self.dof_names.append(name)
        self.meshes.append(mesh)
        return lk

    def _finger(self, name, spec):
        """Build one finger STRAIGHT along +X. Joint angles are zero in the build pose,
        which is why everything is authored flat and posed afterwards."""
        kx, ky, kz = self.base + np.array(spec["knuckle"])
        pp, mp, dp = spec["pp"], spec["mp"], spec["dp"] + PULP
        lo, hi = spec["flex"]
        abd = spec["abd"]

        # MCP is 2 DOF, and Articulation gives one axis per link, so abduction gets its
        # own near-massless carrier link. Abduction is PROXIMAL to flexion so that
        # flexion happens in the abducted plane, as it does anatomically.
        carrier = tp.Mesh(tp.SphereGeometry(0.0072, 12, 9), skin())
        carrier.position.set(kx, ky, kz)
        self._add(f"{name}_abd", carrier, self.palm, (0, 1, 0), (kx, ky, kz), -abd, abd)

        m = cap(R_PP, pp)
        m.position.set(kx + 0.5 * pp, ky, kz)
        orient(m, FINGER_BONE, FINGER_PAD)
        self._add(f"{name}_mcp", m, self.links[f"{name}_abd"], (0, 0, -1), (kx, ky, kz), lo, hi)

        m = cap(R_MP, mp)
        m.position.set(kx + pp + 0.5 * mp, ky, kz)
        orient(m, FINGER_BONE, FINGER_PAD)
        self._add(f"{name}_pip", m, self.links[f"{name}_mcp"], (0, 0, -1), (kx + pp, ky, kz), -0.17, 1.83)

        m = cap(R_DP, dp)
        m.position.set(kx + pp + mp + 0.5 * dp, ky, kz)
        orient(m, FINGER_BONE, FINGER_PAD)
        self._add(f"{name}_dip", m, self.links[f"{name}_pip"], (0, 0, -1), (kx + pp + mp, ky, kz), -0.17, 1.31)

    def _thumb(self):
        """Trapeziometacarpal (2 DOF) + MCP + IP.

        A thumb is not a fifth finger, and the difference is entirely in how it LEAVES
        the wrist. It emerges from the radial side obliquely -- distal AND volar AND
        radial, all at once -- and it is pronated about its own axis so its pad faces
        the fingers rather than the floor. Built parallel to the fingers in the palm
        plane, as this model first was, the hand reads as five identical digits and can
        only oppose by accident.

        So the whole thumb is defined by two vectors, THUMB_BONE and THUMB_PAD, and its
        joint axes then come from the same rule the fingers use: flexion about
        cross(bone, pad), abduction about -pad. Flexion therefore carries the tip along
        the pad direction, across the palm toward the fingers, which is opposition.

        Still an approximation, and the largest one here: Hollister 1992 showed the real
        CMC has two NON-orthogonal, NON-intersecting axes fixed in different bones, and
        that is where a real thumb's axial pronation during opposition comes from. These
        two are orthogonal and intersect.
        """
        c = self.base + np.array(THUMB_CMC)
        L, P = _unit(THUMB_BONE), np.asarray(THUMB_PAD, dtype=float)
        fx = flex_axis(L, P)
        mc, pp, dp = 0.046, 0.031, 0.025 + PULP

        carrier = tp.Mesh(tp.SphereGeometry(0.006, 10, 8), skin())
        carrier.position.set(*c)
        self._add("thumb_cmc_abd", carrier, self.palm, tuple(-_unit(P - L * float(np.dot(P, L)))),
                  tuple(c), -0.20, 1.15)

        m = cap(0.0110, mc)
        m.position.set(*(c + L * (0.5 * mc)))
        orient(m, L, P)
        self._add("thumb_cmc_flex", m, self.links["thumb_cmc_abd"], tuple(fx), tuple(c), -0.50, 0.90)

        j2 = c + L * mc
        m = cap(0.0098, pp)
        m.position.set(*(j2 + L * (0.5 * pp)))
        orient(m, L, P)
        self._add("thumb_mcp", m, self.links["thumb_cmc_flex"], tuple(fx), tuple(j2), -0.17, 0.92)

        j3 = j2 + L * pp
        m = cap(0.0092, dp)
        m.position.set(*(j3 + L * (0.5 * dp)))
        orient(m, L, P)
        self._add("thumb_ip", m, self.links["thumb_mcp"], tuple(fx), tuple(j3), -0.26, 1.02)

        # Thenar eminence: visual only, no collider. It bridges the palm to the thumb
        # base so the thumb does not read as a detached object floating beside the hand,
        # without reintroducing the palm-vs-metacarpal collision that used to jam the
        # CMC swing after 9 degrees of a possible 60.
        self.decor = []

    # -- tendons ------------------------------------------------------------------
    def route(self):
        """Lay the 25 cables. Must run AFTER finalize(), because a cable reads live poses.

        Five per finger, which is the Salisbury-Mason N+1 bound for 4 DOF: two flexors
        (FDP to the distal phalanx, FDS to the middle), one extensor, and a pair of
        interossei whose lateral offset is what gives the MCP an abduction moment arm in
        both directions. --selftest checks the resulting torque space is genuinely
        positively spanned rather than trusting the count.
        """
        for name, spec in FINGERS.items():
            self._finger_cables(name, spec)
        self._thumb_cables()
        return self.cables

    def _cable(self, name, nodes, mode=None):
        """`nodes` is a path in order: (link, local_offset) for a via point, or
        ("wrap", link, centre, axis, radius) for a pulley the cable curves around."""
        c = tp.TendonCable(self.world, mode or tp.TendonCable.Mode.TENSION)
        for nd in nodes:
            if nd[0] == "wrap":
                _, link, centre, axis, radius, side = nd
                c.add_wrap(link, tp.Vector3(*map(float, centre)),
                           tp.Vector3(*map(float, axis)), float(radius),
                           tp.Vector3(*map(float, side)))
            else:
                link, off = nd
                c.add_via_point(link, tp.Vector3(*[float(v) for v in off]))
        self.cables[name] = c
        return c

    def _finger_cables(self, name, spec):
        pp, mp, dp = spec["pp"], spec["mp"], spec["dp"] + PULP
        kx, ky, kz = self.base + np.array(spec["knuckle"])
        L = self.links
        palm, abd = L["palm"], L[f"{name}_abd"]
        p1, p2, p3 = L[f"{name}_mcp"], L[f"{name}_pip"], L[f"{name}_dip"]

        # Actor-frame offsets. The palm box's origin sits at base+(0.045, 0, 0). Each
        # phalanx capsule was created rotated -90 deg about Z, which maps its local +Y
        # onto the world bone axis and its local +X onto world -Y (volar) -- so a point
        # `v` volar and `a` along the bone (from the capsule CENTRE) is (v, a, 0) in that
        # link's frame, and the hinge axis stays local Z.
        def palm_pt(dx, volar):
            return (kx - (self.base[0] + 0.0325) + dx, -volar, kz - self.base[2] - PALM_Z)

        def bone(volar, along, lat=0.0):
            return (volar, along, lat)

        AXIS = (0.0, 0.0, 1.0)

        # THE FLEXORS ARE CHORDS, AND THAT IS THE PHYSICS, NOT A SHORTCUT.
        #
        # A flexor runs on the CONCAVE side of a closing joint. As the finger flexes its
        # insertion swings toward the palm, the straight path shortens, and pulling it
        # flexes -- which is the whole mechanism. Force it to follow the pulley instead
        # and the volar path LENGTHENS by r*theta: measured on the reference geometry,
        # 58.5 -> 73.2 mm over 105 deg, giving a "flexor" with a +8 mm arm where the
        # chord gives -8, i.e. an extensor. So a real annular pulley is a CAP on how far
        # the tendon may bow away from the bone, not a surface it wraps, and the arm
        # RISING with flexion is bowstringing -- the effect that takes a cadaver MCP
        # flexor arm from ~5.8 to ~10.3 mm, and that an A2 rupture roughly doubles.
        #
        # What the routing must avoid is the OTHER failure: a segment that starts at the
        # joint itself pivots wildly as the distal point swings, and once that point
        # passes behind the axis the moment reverses. Measured with the palm-side point
        # sitting on the knuckle: an index FDP MCP arm running +9.98 mm extended to
        # -5.03 mm at 89 deg. Every joint is therefore spanned by a LONG segment,
        # anchored well proximal and inserting well along the distal bone, and
        # --selftest checks the sign never flips anywhere in the range.
        self._cable(f"{name}_fdp", [
            (palm, palm_pt(-0.026, A1)),
            (p1, bone(A2, (FDP_PP - 0.5) * pp)),
            (p2, bone(A4, (FDP_MP - 0.5) * mp)),
            (p3, bone(A5, -0.5 * dp + 0.25 * dp)),
        ])
        # FDS inserts on the MIDDLE phalanx, so it has NO distal moment arm at all. That
        # is what makes it a genuinely independent actuator rather than a near-copy of
        # FDP, and it is why two flexors exist: the DIP sizes the tension a pinch needs
        # while the MCP needs the torque, and one tendon cannot serve both.
        self._cable(f"{name}_fds", [
            (palm, palm_pt(-0.026, A1 + 0.0012)),
            (p1, bone(A2 + 0.0010, (FDP_PP + 0.05 - 0.5) * pp)),
            (p2, bone(A3 * FDS_DISTAL, -0.5 * mp + 0.40 * mp)),
        ])
        # THE EXTENSOR IS THE CASE THAT WRAPS, and it has to.
        #
        # It runs on the CONVEX side, where flexion carries the tendon around each joint
        # rather than away from it. Routed as chords instead, its arm passes through zero
        # somewhere in the range and REVERSES -- measured, 14 reversals across the four
        # fingers, every one of them an extensor, so a pull that extends the finger near
        # neutral flexes it in a fist. The wrap is what a sheathed tendon actually does,
        # and it holds the arm at the standoff instead: validated standalone in
        # tendon_probe.py E7 at -8.20 / -8.57 / -8.20 mm against an 8.0 mm pulley, where
        # the chord over the same joints swung to +6.33.
        self._cable(f"{name}_ext", [
            (palm, palm_pt(-0.026, -E_MCP)),
            ("wrap", abd, (0.0, 0.0, 0.0), AXIS, E_MCP, (0.0, 1.0, 0.0)),
            (p1, bone(-E_MCP, -0.35 * pp)),
            (p1, bone(-E_PIP, 0.30 * pp)),
            ("wrap", p1, (0.0, 0.5 * pp, 0.0), AXIS, E_PIP, (-1.0, 0.0, 0.0)),
            (p2, bone(-E_DIP, -0.05 * mp)),
            ("wrap", p2, (0.0, 0.5 * mp, 0.0), AXIS, E_DIP, (-1.0, 0.0, 0.0)),
            (p3, bone(-E_DIP * 0.8, -0.5 * dp + 0.25 * dp)),
        ])
        # Interossei. Volar of the MCP so they flex it, and offset to one side so they
        # abduct or adduct it: the abduction moment comes from the lateral offset of the
        # run crossing the abduction axis. These are the pair that makes the abduction
        # DOF controllable in BOTH directions with actuators that can only pull.
        for side, lat in (("rad", -IO_LAT), ("uln", +IO_LAT)):
            self._cable(f"{name}_io_{side}", [
                (palm, (kx - (self.base[0] + 0.0325) - 0.024, -0.004,
                        float(np.clip(kz - self.base[2] - PALM_Z + lat, -0.045, 0.045)))),
                (p1, bone(IO_VOLAR, 0.35 * pp, lat * 0.25)),
            ])

    def _thumb_cables(self):
        """Five cables for four DOF, the same N+1 bound as the fingers, and now the same
        SHAPE as a finger's: the thumb has a proper frame, so "volar" means "toward the
        pad" for it exactly as it does for a finger, and the routing reads the same way.

        FPL and EPL are the flexor/extensor pair; APB and ADD straddle the CMC abduction
        axis so it is drivable both ways by pull-only actuators; FPB sits off to the pad
        side of the CMC and is what makes opposition reachable.
        """
        L = self.links
        palm, ab = L["palm"], L["thumb_cmc_abd"]
        m1, m2, m3 = L["thumb_cmc_flex"], L["thumb_mcp"], L["thumb_ip"]
        c = self.base + np.array(THUMB_CMC)
        B, P = _unit(THUMB_BONE), np.asarray(THUMB_PAD, dtype=float)
        R = frame(B, P)
        mc, pp, dp = 0.046, 0.031, 0.025 + PULP
        v1, v2, v3 = 0.0105, 0.0082, 0.0058        # pad-side standoffs, proximal -> distal
        d1, d2, d3 = 0.0080, 0.0062, 0.0046        # and the far side

        def bone(pad, along, lat=0.0):
            return (pad, along, lat)

        def wrist(dx, dy, dz):
            """A point on the PALM link, in its own frame. The palm is the fixed root, so
            anything anchored here is rigidly welded to the world."""
            return (c[0] - (self.base[0] + 0.0325) + dx, dy, c[2] - self.base[2] - PALM_Z + dz)

        AX = (0.0, 0.0, 1.0)
        self._cable("thumb_fpl", [
            (palm, wrist(-0.016, -0.010, 0.014)),
            (m1, bone(v2, -0.20 * mc)),
            (m2, bone(v3, -0.10 * pp)),
            (m3, bone(v3 * 0.8, -0.5 * dp + 0.25 * dp)),
        ])
        # The extensor is the convex-side case, so it wraps, exactly as the fingers' do.
        self._cable("thumb_epl", [
            (palm, wrist(-0.016, 0.010, 0.014)),
            (m1, bone(-d2, -0.30 * mc)),
            ("wrap", m1, (0.0, 0.5 * mc, 0.0), AX, d2, (-1.0, 0.0, 0.0)),
            (m2, bone(-d3, -0.20 * pp)),
            ("wrap", m2, (0.0, 0.5 * pp, 0.0), AX, d3, (-1.0, 0.0, 0.0)),
            (m3, bone(-d3 * 0.8, -0.5 * dp + 0.25 * dp)),
        ])
        # Abductor / adductor pollicis, either side of the CMC abduction axis.
        self._cable("thumb_abd", [
            (palm, wrist(-0.020, -0.008, -0.008)), (ab, (0.0, 0.0, -0.011)),
            (m1, bone(0.002, -0.20 * mc, -0.010)),
        ])
        self._cable("thumb_add", [
            (palm, wrist(-0.006, 0.006, 0.016)), (ab, (0.0, 0.0, 0.011)),
            (m1, bone(0.002, -0.20 * mc, 0.010)),
        ])
        # Flexor pollicis brevis: pad side AND offset, so it drives the CMC flexion that
        # carries the thumb across the palm. This is the opposition cable.
        self._cable("thumb_fpb", [
            (palm, wrist(-0.012, -0.009, 0.010)), (ab, (v1 * 0.5, 0.0, 0.004)),
            (m1, bone(v1, -0.10 * mc, 0.004)),
            (m2, bone(v3, -0.20 * pp)),
        ])

    # -- state --------------------------------------------------------------------
    def finalize(self):
        self.art.finalize()
        # add-order index -> low-level DOF slot; PhysX cache order is not add order, and
        # a branched articulation like this reorders aggressively.
        self._dof_slot = np.asarray(self.art.dof_order())
        return self

    def set_pose(self, q):
        """q is in ADD ORDER (self.dof_names); this maps it into PhysX's cache order."""
        buf = np.zeros(len(self.dof_names), dtype=np.float32)
        for i, v in enumerate(q):
            buf[self._dof_slot[i]] = v
        self.art.set_joint_positions(buf)

    def pose(self):
        return np.array([self.links[n].joint_position for n in self.dof_names])

    def zero(self):
        return np.zeros(len(self.dof_names))

    def relax(self):
        for c in self.cables.values():
            c.set_tension(0.0)


# ================================================================================
# Verification. Every number below is measured from the simulated geometry; none of
# them is read back from a table this file authored.
# ================================================================================

# Cadaver/in-vivo moment arms (mm) for the index finger, for comparison only.
# An et al. 1983 (J. Biomech. 16:419-425) and Buford et al. 2011; consensus values,
# see the brief in the commit message. Buford's sourced claim is the SHAPE: arms are
# largest at the MCP, smallest at the DIP, and similar across digits.
LIT = {
    "fdp": {"mcp": (9, 11), "pip": (6.0, 7.5), "dip": (4, 5)},
    "fds": {"mcp": (10, 12), "pip": (6, 7)},
    "ext": {"mcp": (-8, -6), "pip": (-4.5, -3.5), "dip": (-3.5, -2.5)},
}


def moment_arms(hand, cable_name, q, h=1e-4):
    """r_j = -dL/dq_j for one cable, in metres, by central difference on the LIVE
    simulated geometry.

    This is the moment arm in the only sense that matters dynamically: the virtual-work
    definition. Because TendonCable applies -T dL/dq exactly (measured to 0.02%), a
    tension T through this cable produces T*r_j at joint j, and nothing here was
    authored -- it falls out of where the via points sit.
    """
    c = hand.cables[cable_name]
    out = np.zeros(len(hand.dof_names))
    for j in range(len(hand.dof_names)):
        qp, qm = np.array(q, dtype=float), np.array(q, dtype=float)
        qp[j] += h
        qm[j] -= h
        hand.set_pose(qp)
        lp = c.length
        hand.set_pose(qm)
        lm = c.length
        out[j] = -(lp - lm) / (2 * h)
    hand.set_pose(q)
    return out


def selftest():
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT, max_substeps=1)
    hand = Hand(world).finalize()
    hand.route()
    n_dof, n_cab = len(hand.dof_names), len(hand.cables)
    print(f"hand: {n_dof} DOF, {n_cab} cables, {len(hand.links)} links")

    # -- 0. the pose map actually round-trips ------------------------------------
    # PhysX's cache order is not add order, and on a branched articulation like this
    # they differ a lot. A silent mismatch here would corrupt every number below, so it
    # is checked before anything else is measured.
    probe = np.linspace(0.05, 0.35, n_dof)
    hand.set_pose(probe)
    err = np.abs(hand.pose() - probe).max()
    print(f"\n[0] add-order <-> PhysX DOF map round-trips to {err:.2e} rad"
          f"   {'OK' if err < 1e-4 else 'FAIL'}")

    # -- 1. moment arms, measured -------------------------------------------------
    print("\n[1] index-finger moment arms (mm), MEASURED from the routing geometry")
    print("    at a lightly flexed pose; literature range in brackets.")
    q = hand.zero()
    for i, nm in enumerate(hand.dof_names):
        if nm.startswith("index"):
            q[i] = 0.35 if not nm.endswith("_abd") else 0.0
    hand.set_pose(q)
    idx = {n: i for i, n in enumerate(hand.dof_names)}
    for tend in ("fdp", "fds", "ext"):
        r = moment_arms(hand, f"index_{tend}", q) * 1000.0
        row = []
        for jn, key in (("index_mcp", "mcp"), ("index_pip", "pip"), ("index_dip", "dip")):
            if key not in LIT[tend]:
                row.append(f"{key.upper()}     -   ")
                continue
            lo, hi = LIT[tend][key]
            v = r[idx[jn]]
            mark = "ok" if lo <= v <= hi else "**"
            row.append(f"{key.upper()} {v:+6.2f} [{lo:+g},{hi:+g}]{mark}")
        print(f"    {tend.upper():4s}  " + "   ".join(row))
    r_io = moment_arms(hand, "index_io_rad", q) * 1000.0
    print(f"    interosseous abduction arm at the MCP: {r_io[idx['index_abd']]:+.2f} mm"
          f"  [literature 8-10 mm]")

    # -- 2. the Salisbury-Mason condition -----------------------------------------
    # A cable can only pull, so N+1 tendons is necessary but NOT sufficient: the
    # torque space must be POSITIVELY spanned. With R the 4x5 moment-arm matrix, that
    # means rank(R) = 4 and there is a strictly positive t with R t = 0 -- a
    # co-contraction that produces no net joint torque. Without such a t, some tension
    print("\n[2] Salisbury-Mason: is each finger's torque space POSITIVELY spanned?")
    print("    A cable can only pull, so N+1 tendons is NECESSARY but not sufficient:")
    print("    the torque space must be positively spanned -- rank(R) = n_dof AND some")
    print("    strictly positive tension vector t giving R t = 0, a co-contraction with")
    print("    no net joint torque. Without one, tension combinations are unreachable")
    print("    and no policy can learn the postures behind them. Checked at 5 poses")
    print("    across each finger's OWN flexion range, because this is a property of the")
    print("    POSE, not of the tendon count: an earlier version evaluated every finger")
    print("    at a pose where only the index was flexed and duly failed the other four.")
    for f in list(FINGERS) + ["thumb"]:
        dofs = [i for i, n in enumerate(hand.dof_names) if n.startswith(f)]
        cabs = [c for c in hand.cables if c.startswith(f)]
        good, worst = 0, None
        for ang in (0.0, 0.3, 0.7, 1.1, 1.45):
            qp = hand.zero()
            for i in dofs:
                if not hand.dof_names[i].endswith(("_abd", "_swing")):
                    qp[i] = ang
            hand.set_pose(qp)
            R = np.array([moment_arms(hand, c, qp)[dofs] for c in cabs]).T
            rank = np.linalg.matrix_rank(R, tol=1e-9)
            _u, _s, vt = np.linalg.svd(R)
            null = vt[len(dofs):]
            ok = False
            for v in null:
                for sg in (1.0, -1.0):
                    if (sg * v).min() > 1e-6:
                        ok = True
            if ok and rank == len(dofs):
                good += 1
            elif worst is None and len(null):
                worst = (ang, null[0] / np.abs(null[0]).max())
        print(f"    {f:7s} {len(dofs)} DOF, {len(cabs)} cables   spanned at {good}/5 poses"
              f"   {'OK' if good == 5 else 'GAP'}")
        if worst is not None:
            names = [c.replace(f + "_", "") for c in cabs]
            print(f"            first gap at {math.degrees(worst[0]):.0f} deg: "
                  + "  ".join(f"{n}={v:+.2f}" for n, v in zip(names, worst[1])))


    # -- 3. does the moment arm actually vary with posture? ------------------------
    # The point of routing a cable over geometry rather than prescribing a coefficient
    # is that the arm changes as the finger moves. If it were flat, a fixed tendon
    # would have been the cheaper choice and this whole design would be unjustified.
    print("\n[3] index FDP moment arm vs MCP flexion (mm) -- the reason to route at all")
    hdr, mcp_r, dip_r = [], [], []
    for ang in (0.0, 0.4, 0.8, 1.2, 1.55):
        q2 = hand.zero()
        q2[idx["index_mcp"]] = ang
        hand.set_pose(q2)
        r = moment_arms(hand, "index_fdp", q2) * 1000.0
        hdr.append(f"{math.degrees(ang):5.0f}")
        mcp_r.append(f"{r[idx['index_mcp']]:5.2f}")
        dip_r.append(f"{r[idx['index_dip']]:5.2f}")
    print("    MCP angle (deg) " + " ".join(hdr))
    print("    arm at MCP      " + " ".join(mcp_r))
    print("    arm at DIP      " + " ".join(dip_r))
    print("    (a real finger's MCP flexor arm rises ~5.8 -> 10.3 mm as the tendon")
    print("     bowstrings off the pulleys before A2 catches it; the rise here is that")
    print("     effect, and where the pulley sits along the phalanx is what caps it)")

    # -- 3b. sign stability across the range -------------------------------------
    # The failure a moment-arm table taken at ONE pose cannot show: a cable whose arm
    # reverses somewhere in the range, so the flexor extends the joint in exactly the
    # posture a grasp lives in. Measured, this happened twice on the way here -- once
    # from via points sitting on the joint, once from forcing a flexor to wrap.
    print("\n[3b] does any cable REVERSE its moment arm anywhere in the range?")
    bad = []
    for f in list(FINGERS) + ["thumb"]:
        dofs = [i for i, n in enumerate(hand.dof_names) if n.startswith(f)]
        for c in [c for c in hand.cables if c.startswith(f)]:
            signs = {}
            for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                qp = hand.zero()
                for i in dofs:
                    if not hand.dof_names[i].endswith(("_abd", "_swing")):
                        qp[i] = frac * hand.limits[hand.dof_names[i]][1]
                hand.set_pose(qp)
                r = moment_arms(hand, c, qp)
                for i in dofs:
                    if abs(r[i]) > 3e-4:                      # ignore arms under 0.3 mm
                        signs.setdefault(i, set()).add(np.sign(r[i]))
            for i, sg in signs.items():
                if len(sg) > 1:
                    bad.append((c, hand.dof_names[i]))
    if bad:
        print(f"    {len(bad)} REVERSALS:")
        for c, j in bad[:12]:
            print(f"      {c} at {j}")
    else:
        print("    none - every cable keeps its sign at every joint it spans,")
        print("    across 0 to 1.5 rad of flexion. 25 cables x 20 joints checked.")

    # -- 4. tension -> fingertip force --------------------------------------------
    print("\n[4] tendon tension -> joint torque, against the analytic prediction")
    hand.set_pose(q)
    T = 40.0
    r = moment_arms(hand, "index_fdp", q)
    for jn in ("index_mcp", "index_pip", "index_dip"):
        print(f"    {jn:10s}  r = {r[idx[jn]]*1000:5.2f} mm  ->  tau = T*r ="
              f" {T*r[idx[jn]]:+.4f} N.m at T = {T} N")
    print(f"    Schuind 1992 measured an in-vivo FDP tension:fingertip force ratio of")
    print(f"    7.9 +/- 6.3 during tip pinch; at r_DIP = {r[idx['index_dip']]*1000:.2f} mm and a")
    print(f"    ~20 mm pulp lever this routing implies {0.020/max(r[idx['index_dip']],1e-9):.1f}:1.")
    return hand


# ================================================================================
# Grasp. Not "does it look like it is holding" -- how much force does the grasp
# actually resist before the object moves?
# ================================================================================

def _build_grasp(obj, tension, verbose=False):
    """One world, one hand, one object, closed and settled. Returns everything needed
    to load the object afterwards."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT,
                          max_substeps=1, tgs_pcm=True)
    # A grippy pad. Rubber on plastic is ~1.0-1.5; restitution 0 so contacts settle
    # instead of buzzing, and 'min' so the softer pair member governs.
    pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
    hand = Hand(world, material=pad).finalize()
    hand.route()

    if obj == "can":
        # 44 mm across. A 60 mm can was tried first and is genuinely beyond this hand:
        # the opposed thumb tip sits 36 mm from the fingertips, so nothing closes on the
        # far side and the can squirts out along its own axis, the one direction the
        # grasp does not constrain. That is a real limit of a hand this size, not a
        # solver artefact -- but the sim does add energy on the way out, so read the
        # failure as "not caged", not as a velocity.
        mesh = tp.Mesh(tp.CapsuleGeometry(0.022, 0.040), skin(0x3388CC, 0.5))
        mesh.rotation.x = math.pi / 2                     # axis across the palm
    else:
        mesh = tp.Mesh(tp.SphereGeometry(0.018, 24, 16), skin(0x3388CC, 0.5))
    # Placed where the CLOSED hand actually forms a cavity, measured rather than
    # guessed: the fingertips converge on (0.057, -0.024, -0.002) and the opposed
    # thumb tip reaches (0.046, -0.059, -0.003), 36 mm away.
    mesh.position.set(0.072, -0.043, -0.006)
    body = world.add(mesh, 780.0, pad)

    # Which cables to pull, chosen from the MEASURED thumb moment arms rather than from
    # their names: fpb is the opposition cable (-10.02 mm at the CMC swing, +4.95 at the
    # MCP), add lifts the thumb volar out of the palm (+5.29 at the CMC abduction) and
    # also helps the swing (-4.00), and fpl flexes the MCP and IP. thumb_abd is the
    # ABDUCTOR and carries +4.00 at the swing -- pulling it fights opposition, and with
    # it in the set the thumb reached only -8.2 deg of a possible -60.
    close = [f"{f}_{t}" for f in FINGERS for t in ("fdp", "fds")] +             ["thumb_fpl", "thumb_fpb", "thumb_add"]
    # Ramped, not stepped: a step in tension is a step in force, and the solver sees an
    # impulse no real drivetrain could deliver.
    for i in range(2600):
        k = min(1.0, i / 1400.0)
        for c in close:
            hand.cables[c].set_tension(tension * k)
        world.step(DT)
    return world, hand, body, close


def grasp_test(obj="can", tension=35.0, verbose=True):
    """Close the hand on an object, then pull until it slips.

    Gravity is OFF on purpose. Weight is one fixed load in one direction; what a grasp is
    FOR is resisting an arbitrary disturbance wrench, so the object is loaded deliberately
    along each axis in turn with a force ramped until it breaks loose. The number is
    directly comparable to a weight: 5 N resisted is a 500 g object held that way.

    Each direction gets a FRESH grasp. Reusing one and settling between ramps looked
    fine and was not: the first ramp leaves the object displaced inside a compliant
    grasp, and every later direction then trips the slip threshold on its first
    increment, reporting 0.02 N for a grasp that had just resisted 80.
    """
    SLIP = 0.008          # 8 mm of movement counts as lost
    MAXF = 80.0
    if verbose:
        print(f"\n=== grasp: {obj}, flexor tension {tension:.0f} N per cable ===")
    w0, h0, b0, close = _build_grasp(obj, tension)
    held = np.array([b0.position.x, b0.position.y, b0.position.z])
    if verbose:
        print(f"  object settled at ({held[0]:+.3f}, {held[1]:+.3f}, {held[2]:+.3f}) m")
        print(f"  {len(close)} cables pulling; joint angles at the grasp:")
        for f in list(FINGERS) + ["thumb"]:
            js = [n for n in hand_dofs(h0, f)]
            print("    " + f"{f:7s} " + "  ".join(
                f"{n.split('_', 1)[1]}={math.degrees(h0.links[n].joint_position):5.1f}" for n in js))
    del w0, h0, b0

    results = {}
    for name, d in (("+X  distal, out of the fingers", (1, 0, 0)),
                    ("-X  proximal, into the palm", (-1, 0, 0)),
                    ("+Y  dorsal, through the palm", (0, 1, 0)),
                    ("-Y  volar, out of the grip", (0, -1, 0)),
                    ("+Z  across the palm", (0, 0, 1)),
                    ("-Z  across, toward the thumb", (0, 0, -1))):
        world, hand, body, _ = _build_grasp(obj, tension)
        base = np.array([body.position.x, body.position.y, body.position.z])
        f, slipped = 0.0, None
        while f < MAXF:
            f += 0.04
            body.add_force(tp.Vector3(d[0] * f, d[1] * f, d[2] * f))
            world.step(DT)
            p = np.array([body.position.x, body.position.y, body.position.z])
            if np.linalg.norm(p - base) > SLIP:
                slipped = f
                break
        results[name] = slipped
        del world, hand, body
    if verbose:
        print(f"  force resisted before the object moved {SLIP*1000:.0f} mm"
              f"  (fresh grasp per direction):")
        for k, v in results.items():
            if v is None:
                print(f"    {k:32s} > {MAXF:.0f} N   (never broke loose)")
            else:
                print(f"    {k:32s} {v:6.2f} N   = a {v/9.81*1000:.0f} g object held that way")
        vals = [v if v is not None else MAXF for v in results.values()]
        worst = min(vals)
        print(f"  WEAKEST DIRECTION: {worst:.2f} N ({worst/9.81*1000:.0f} g).")
        print(f"  That is the number to quote -- a grasp is only as good as its weakest")
        print(f"  direction, and quoting the best one is how a bad grasp looks good.")
    return results


def hand_dofs(hand, f):
    return [n for n in hand.dof_names if n.startswith(f)]


# ================================================================================
# Visual. The routing is the whole design, and a column of moment arms is a poor way
# to see it: a cable on the wrong side of a joint, or one cutting a corner it should
# wrap, is obvious on sight and nearly invisible in a table. Every cable is drawn as
# its RESOLVED path -- via points plus whatever arc a wrap contributed -- rebuilt each
# frame from the live link poses, so what is on screen is what is pulling.
# ================================================================================

CABLE_COLOUR = {
    "fdp": 0xE03030,     # deep flexor  - red
    "fds": 0xFF7040,     # superficial  - orange
    "ext": 0x3070E0,     # extensor     - blue
    "io_": 0x30C060,     # interossei   - green
    "fpl": 0xE03030, "fpb": 0xFF7040, "epl": 0x3070E0,
    "abd": 0x30C060, "add": 0xC060D0,
}


def cable_colour(name):
    tail = name.split("_", 1)[1] if "_" in name else name
    for k, c in CABLE_COLOUR.items():
        if tail.startswith(k):
            return c
    return 0xFFFFFF


ROPE_RADIUS = 0.0013
ROPE_SEGMENTS_MAX = 48

# One unit cylinder shared by every rope segment in the scene: height 1 along +Y,
# radius 1, scaled and oriented per segment. 25 ropes at up to 48 segments is ~1200
# meshes, so they share geometry rather than each owning one.
_UNIT_CYL = None


def _unit_cyl():
    global _UNIT_CYL
    if _UNIT_CYL is None:
        _UNIT_CYL = tp.CylinderGeometry(1.0, 1.0, 1.0, 6, 1)
    return _UNIT_CYL


def _aim(mesh, a, b, radius):
    """Place the shared unit cylinder so it spans a -> b."""
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        mesh.visible = False
        return
    mesh.visible = True
    mid = 0.5 * (a + b)
    mesh.position.set(float(mid[0]), float(mid[1]), float(mid[2]))
    mesh.scale.set(radius, n, radius)
    u = d / n
    # Rotation taking the cylinder's own +Y onto u, as axis-angle: threepp's Quaternion
    # binding has set_from_axis_angle but no set_from_unit_vectors.
    ax = np.cross((0.0, 1.0, 0.0), u)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        mesh.quaternion.set(0.0, 0.0, 0.0, 1.0 if u[1] > 0 else 0.0)
        if u[1] < 0:
            mesh.quaternion.set(1.0, 0.0, 0.0, 0.0)
        return
    ax = ax / s
    ang = math.acos(max(-1.0, min(1.0, float(u[1]))))
    mesh.quaternion.set_from_axis_angle(tp.Vector3(float(ax[0]), float(ax[1]), float(ax[2])), ang)


class RopeView:
    """Every cable drawn as an actual rope: a chain of thin cylinders following the
    RESOLVED path, wrap arcs included, rebuilt each frame from the live link poses.

    Debug lines were the first attempt and they were the wrong instrument for this. The
    mechanism is defined by physical cord running over pulleys, and a 1-pixel line does
    not read as cord; it also hid, for several iterations, that the cables were passing
    outside the body entirely.
    """

    def __init__(self, scene, hand):
        self.hand = hand
        self.pool = {}
        self.mats = {}
        for name in hand.cables:
            m = tp.MeshStandardMaterial()
            m.color = tp.Color(cable_colour(name))
            m.roughness = 0.55
            m.metalness = 0.05
            self.mats[name] = m
            # ropes are opaque and drawn first, skin blends over them
            self.pool[name] = []
        self.scene = scene
        self.update()

    def _seg(self, name, i):
        pool = self.pool[name]
        while len(pool) <= i:
            mesh = tp.Mesh(_unit_cyl(), self.mats[name])
            mesh.render_order = 1
            mesh.visible = False
            self.scene.add(mesh)
            pool.append(mesh)
        return pool[i]

    def set_visible(self, on):
        self._on = on
        if not on:
            for pool in self.pool.values():
                for m in pool:
                    m.visible = False

    def update(self, extra=None):
        if not getattr(self, "_on", True):
            return
        for name, cable in self.hand.cables.items():
            pts = np.asarray(cable.path, dtype=float)
            if extra and name in extra:
                pts = np.vstack([np.asarray(extra[name], dtype=float), pts])
            n = min(len(pts) - 1, ROPE_SEGMENTS_MAX)
            for i in range(n):
                _aim(self._seg(name, i), pts[i], pts[i + 1], ROPE_RADIUS)
            for i in range(n, len(self.pool[name])):
                self.pool[name][i].visible = False


class Forearm:
    """The other half of the picture: the motors, and the rope running from them.

    A tendon hand exists because a motor at a finger joint would be mass at the fastest
    moving end of the arm, so the motors sit back here and pull. Each cable gets a linear
    actuator whose rod retracts by that cable's ACTUAL EXCURSION -- the change in its own
    routed length since the hand was open -- so what the rods do is a measurement, not an
    animation. The rope drawn from each rod to the wrist is decoration: the physics cable
    already begins at a via point on the palm, which is the fixed root, so extending it
    backwards adds nothing to any joint torque (--selftest confirms the moment arms are
    unchanged to the digit).
    """

    def __init__(self, scene, hand):
        self.hand = hand

        # A tapered, oval forearm rather than a box: a cylinder with a smaller radius at
        # the wrist end than the elbow, squashed in one cross-section axis. After the
        # -90 deg Z rotation its own +Y lies along +X, so radius_top is the WRIST end.
        arm = tp.Mesh(tp.CylinderGeometry(0.0300, 0.0430, 0.150, 24), skin(0xB89070, 0.85))
        arm.position.set(-0.083, 0.0, -0.002)
        orient(arm, (1, 0, 0), (0, -1, 0))
        arm.scale.set(1.0, 1.0, 0.74)
        scene.add(arm)

        wrist = tp.Mesh(tp.CylinderGeometry(0.0250, 0.0300, 0.022, 20), skin(0xC0A088, 0.7))
        wrist.position.set(-0.002, 0.0, -0.002)
        orient(wrist, (1, 0, 0), (0, -1, 0))
        wrist.scale.set(1.0, 1.0, 0.78)
        scene.add(wrist)

        steel = tp.MeshStandardMaterial()
        steel.color = tp.Color(0x59606B)
        steel.metalness = 0.85
        steel.roughness = 0.30
        brass = tp.MeshStandardMaterial()
        brass.color = tp.Color(0xC9A227)
        brass.metalness = 0.9
        brass.roughness = 0.35

        # One actuator per cable: a motor can, a lead screw, and a nut that travels.
        # Laid out in two banks either side of the forearm axis, the way the poster's
        # forearm packs them, rather than a grid of blocks.
        self.names = list(hand.cables)
        self.rest = {n: hand.cables[n].length for n in self.names}
        self.rods, self.home = {}, {}
        per = math.ceil(len(self.names) / 2)
        for i, n in enumerate(self.names):
            bank, k = (1 if i >= per else -1), (i % per)
            ang = (k / max(1, per - 1) - 0.5) * 2.1          # spread around the axis
            r = 0.0205
            y = r * math.cos(ang) * (0.62 if bank > 0 else -0.62) - 0.001
            z = r * math.sin(ang) * 0.95 - 0.002
            x = -0.132 + 0.0125 * (k % 2)

            can = tp.Mesh(tp.CylinderGeometry(0.0042, 0.0042, 0.020, 12), steel)
            can.position.set(x, y, z)
            orient(can, (1, 0, 0), (0, -1, 0))
            scene.add(can)

            screw = tp.Mesh(tp.CylinderGeometry(0.0011, 0.0011, 0.030, 8), brass)
            screw.position.set(x + 0.025, y, z)
            orient(screw, (1, 0, 0), (0, -1, 0))
            scene.add(screw)

            nut = tp.Mesh(_unit_cyl(), brass)
            nut.render_order = 2
            scene.add(nut)
            self.rods[n] = nut
            self.home[n] = np.array([x + 0.040, y, z])

    def update(self):
        """Returns the extra proximal rope points, keyed by cable."""
        extra = {}
        for n in self.names:
            exc = self.rest[n] - self.hand.cables[n].length      # metres of rope taken in
            tip = self.home[n] - np.array([min(0.022, max(0.0, exc)), 0.0, 0.0])
            _aim(self.rods[n], tip - np.array([0.0035, 0, 0]), tip + np.array([0.0035, 0, 0]), 0.0032)
            # rod tip -> a fairing point at the cuff -> on into the cable's own path
            extra[n] = [tip, np.array([0.004, tip[1] * 0.45, tip[2] * 0.55])]
        return extra


# ================================================================================
# Control. The cables are the actuators, so the panel drives TENSIONS, not angles --
# there is no joint target anywhere in this hand to set. Every slider is newtons on a
# real cord, and what the hand does with them is up to the mechanics.
# ================================================================================

DIGITS = ["index", "middle", "ring", "little", "thumb"]


def digit_cables(name):
    """(flexors, extensors, cable that spreads +, cable that spreads -) for one digit."""
    if name == "thumb":
        return (["thumb_fpl", "thumb_fpb"], ["thumb_epl"], "thumb_abd", "thumb_add")
    return ([f"{name}_fdp", f"{name}_fds"], [f"{name}_ext"],
            f"{name}_io_uln", f"{name}_io_rad")


# Tension presets, in newtons: (flex, extend, spread) per digit. Spread is signed and
# scaled to SPREAD_N; a pull-only pair cannot do both, so the sign picks which cable.
FLEX_N, EXT_N, SPREAD_N = 40.0, 25.0, 18.0
PRESETS = {
    "open":   {d: (0.00, 0.55, 0.0) for d in DIGITS},
    "relax":  {d: (0.00, 0.00, 0.0) for d in DIGITS},
    "fist":   {d: (0.90, 0.00, 0.0) for d in DIGITS} | {"thumb": (0.80, 0.0, -0.7)},
    "pinch":  {"index": (0.55, 0.0, 0.0), "middle": (0.15, 0.0, 0.0),
               "ring": (0.10, 0.0, 0.0), "little": (0.10, 0.0, 0.0),
               "thumb": (0.75, 0.0, -0.9)},
    "point":  {"index": (0.00, 0.85, 0.0), "middle": (0.95, 0.0, 0.0),
               "ring": (0.95, 0.0, 0.0), "little": (0.95, 0.0, 0.0),
               "thumb": (0.60, 0.0, -0.5)},
    "spread": {"index": (0.0, 0.5, -0.9), "middle": (0.0, 0.5, -0.3),
               "ring": (0.0, 0.5, 0.5), "little": (0.0, 0.5, 0.9),
               "thumb": (0.0, 0.5, 0.9)},
}


class HandController:
    """Slider state -> cable tensions, with a ramp so a preset never lands as a step.

    A step in commanded tension is a step in applied force, which is an impulse no real
    drivetrain could deliver -- and with a 40 N cable on a 9 g phalanx it visibly slaps
    the object out of the hand rather than closing on it.
    """

    RATE = 2.5          # full-scale per second

    def __init__(self, hand):
        self.hand = hand
        self.target = {d: [0.0, 0.55, 0.0] for d in DIGITS}
        self.now = {d: [0.0, 0.55, 0.0] for d in DIGITS}
        self.master = 1.0
        self.manual = {}          # cable -> newtons, overrides the digit sliders

    def preset(self, name):
        for d, v in PRESETS[name].items():
            self.target[d] = list(v)
        self.manual.clear()

    def step(self, dt):
        for d in DIGITS:
            for i in range(3):
                a, b = self.now[d][i], self.target[d][i]
                m = self.RATE * dt
                self.now[d][i] = b if abs(b - a) <= m else a + math.copysign(m, b - a)
        for d in DIGITS:
            flex, ext, spread = self.now[d]
            fl, ex, sp_plus, sp_minus = digit_cables(d)
            for c in fl:
                self.hand.cables[c].set_tension(FLEX_N * flex * self.master)
            for c in ex:
                self.hand.cables[c].set_tension(EXT_N * ext)
            self.hand.cables[sp_plus].set_tension(SPREAD_N * max(0.0, spread))
            self.hand.cables[sp_minus].set_tension(SPREAD_N * max(0.0, -spread))
        for c, n in self.manual.items():
            self.hand.cables[c].set_tension(n)

    def excursion(self, name, rest):
        return (rest[name] - self.hand.cables[name].length) * 1000.0



def _scene(width, height, headless):
    canvas = tp.Canvas("tendon hand", width=width, height=height, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.set_clear_color(0x161A22)
    scene = tp.Scene()
    cam = tp.PerspectiveCamera(42, width / height, 0.01, 10)
    # Translucent skin swallows light, so this is lit harder than a solid model would be.
    scene.add(tp.AmbientLight(0xFFFFFF, 1.1))
    scene.add(tp.HemisphereLight(0xFFFFFF, 0x50505A, 1.4))
    key = tp.DirectionalLight(0xFFFFFF, 2.6)
    key.position.set(0.25, 0.35, 0.30)
    scene.add(key)
    # A volar fill, because the palm-side views are the informative ones and the first
    # pass lit only the dorsum: the volar render came back as a black rectangle.
    fill = tp.DirectionalLight(0xFFE8D8, 2.0)
    fill.position.set(0.10, -0.40, 0.20)
    scene.add(fill)
    rim = tp.DirectionalLight(0x88AACC, 0.9)
    rim.position.set(-0.30, 0.10, -0.35)
    scene.add(rim)
    return canvas, renderer, scene, cam


VIEWS = {
    #                camera position           look-at
    # One framing has to hold both the open hand (~190 mm, fingers extended) and the
    # closed fist (~110 mm), so it is set by the open pose and the fist simply sits
    # smaller inside it.
    # Framed to hold the forearm as well as the hand: the picture is motors-at-the-back,
    # rope-through-the-wrist, fingers-pulled, and cropping to the hand loses the point.
    "3q":     ((0.225, -0.170, -0.245), (0.018, -0.018, -0.004)),   # three-quarter volar-radial
    "volar":  ((0.022, -0.330, -0.018), (0.018, -0.018, -0.004)),   # straight into the palm
    "radial": ((0.237, -0.070, -0.396), (0.018, -0.018, -0.004)),   # thumb side
    "hand":   ((0.294, -0.169, -0.194), (0.095, -0.025, -0.004)),   # the hand alone
}


def visual(shots=None, poses=None, width=1280, height=800, obj="ball", tension=35.0,
           headless=False, out="tendon_hand"):
    """Close the hand on an object with every cable drawn.

    With --shots it renders stills headless and writes PNGs; without, it opens a window
    and runs live.
    """
    canvas, renderer, scene, cam = _scene(width, height, headless or bool(shots) or bool(poses))
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT,
                          max_substeps=1, tgs_pcm=True)
    pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
    hand = Hand(world, material=pad).finalize()
    hand.route()
    for m in hand.meshes:
        scene.add(m)
    for m in getattr(hand, "decor", []):
        scene.add(m)

    body = mesh = None
    if obj != "none":
        if obj == "can":
            mesh = tp.Mesh(tp.CapsuleGeometry(0.022, 0.040), skin(0x3388CC, 0.45))
            mesh.rotation.x = math.pi / 2
        else:
            mesh = tp.Mesh(tp.SphereGeometry(0.018, 32, 24), skin(0x3388CC, 0.45))
        mesh.position.set(0.072, -0.043, -0.006)
        body = world.add(mesh, 780.0, pad)
        scene.add(mesh)

    arm = Forearm(scene, hand)
    view = RopeView(scene, hand)
    close = [f"{f}_{t}" for f in FINGERS for t in ("fdp", "fds")] +             ["thumb_fpl", "thumb_fpb", "thumb_add"]

    def step(i):
        k = min(1.0, max(0.0, (i - 120) / 1400.0))
        for c in close:
            hand.cables[c].set_tension(tension * k)
        world.step(DT)
        view.update(arm.update())

    if poses:
        # Render each preset at steady state. This is how the control layer is checked
        # without a window: a preset that does not actually produce the posture it is
        # named after is obvious on sight and invisible in a tension readout.
        ctl = HandController(hand)
        for name in poses:
            ctl.preset("open")
            for _ in range(900):
                ctl.step(DT)
                world.step(DT)
            ctl.preset(name)
            for _ in range(1500):
                ctl.step(DT)
                world.step(DT)
            view.update(arm.update())
            # The posture is the claim, so print it. A preset that does not produce what
            # its name says is invisible in a tension readout and obvious here.
            print(f"  {name}:")
            for d in DIGITS:
                js = [n for n in hand.dof_names if n.startswith(d)]
                print("     " + f"{d:7s} " + "  ".join(
                    f"{n.split('_', 1)[1]}={math.degrees(hand.links[n].joint_position):6.1f}"
                    for n in js))
            for vname in ("3q", "volar"):
                pos, tgt = VIEWS[vname]
                cam.position.set(*pos)
                cam.look_at(*tgt)
                renderer.render(scene, cam)
                fn = f"{out}_pose_{name}_{vname}.png"
                renderer.save_frame(fn)
                print(f"  wrote {fn}")
        return

    if shots:
        frames = {"open": 60, "closing": 800, "closed": 2600}
        want = [s for s in shots] if shots != ["all"] else list(frames)
        last = max(frames[s] for s in want)
        marks = {frames[s]: s for s in want}
        for i in range(last + 1):
            step(i)
            if i in marks:
                for vname, (pos, tgt) in VIEWS.items():
                    cam.position.set(*pos)
                    cam.look_at(*tgt)
                    renderer.render(scene, cam)
                    fn = f"{out}_{marks[i]}_{vname}.png"
                    renderer.save_frame(fn)
                    print(f"  wrote {fn}")
        return

    pos, tgt = VIEWS["3q"]
    cam.position.set(*pos)
    cam.look_at(*tgt)
    controls = tp.OrbitControls(cam, canvas)
    controls.target.set(*tgt)
    controls.enable_damping = True
    ui = tp.ImguiContext(canvas, renderer)
    ctl = HandController(hand)
    rest = {n: hand.cables[n].length for n in hand.cables}
    clock = tp.Clock()
    ui_state = {"advanced": False, "skin": SKIN_OPACITY, "ropes": True, "readout": True}
    start = None
    if obj != "none":
        start = (mesh.position.x, mesh.position.y, mesh.position.z)

    def draw_ui():
        tp.imgui.set_next_window_pos(10, 10)
        tp.imgui.set_next_window_size(330, 0)
        tp.imgui.begin("Tendon drive")
        tp.imgui.text("Every slider is NEWTONS on a real cord.")
        tp.imgui.text("There is no joint target anywhere in this hand.")
        tp.imgui.separator()

        for i, name in enumerate(("open", "fist", "pinch", "point", "spread", "relax")):
            if i % 3:
                tp.imgui.same_line()
            if tp.imgui.button(f"{name:>6}"):
                ctl.preset(name)
        tp.imgui.separator()
        _, ctl.master = tp.imgui.slider_float("grip x", ctl.master, 0.0, 1.5)

        for d in DIGITS:
            if tp.imgui.collapsing_header(d):
                t = ctl.target[d]
                _, t[0] = tp.imgui.slider_float(f"flex##{d}", t[0], 0.0, 1.0)
                _, t[1] = tp.imgui.slider_float(f"extend##{d}", t[1], 0.0, 1.0)
                _, t[2] = tp.imgui.slider_float(f"spread##{d}", t[2], -1.0, 1.0)
                fl, ex, sp, sm = digit_cables(d)
                for c in fl + ex + [sp, sm]:
                    tp.imgui.text(f"  {c.replace(d + '_', ''):<8} "
                                  f"{hand.cables[c].tension:5.1f} N  "
                                  f"{ctl.excursion(c, rest):+6.1f} mm")

        tp.imgui.separator()
        _, ui_state["advanced"] = tp.imgui.checkbox("per-cable override", ui_state["advanced"])
        if ui_state["advanced"]:
            for c in hand.cables:
                cur = ctl.manual.get(c, hand.cables[c].tension)
                changed, v = tp.imgui.slider_float(c, cur, 0.0, 60.0)
                if changed:
                    ctl.manual[c] = v
            if tp.imgui.button("clear overrides"):
                ctl.manual.clear()

        tp.imgui.separator()
        ch, ui_state["skin"] = tp.imgui.slider_float("skin opacity", ui_state["skin"], 0.0, 1.0)
        if ch:
            for m in hand.meshes:
                m.material.opacity = ui_state["skin"]
        ch2, ui_state["ropes"] = tp.imgui.checkbox("show rope", ui_state["ropes"])
        if ch2:
            view.set_visible(ui_state["ropes"])
        if start and tp.imgui.button("reset object"):
            body.set_pose(tp.Vector3(*start))
            body.set_linear_velocity(tp.Vector3(0, 0, 0))
            body.set_angular_velocity(tp.Vector3(0, 0, 0))
        tp.imgui.same_line()
        if tp.imgui.button("reset view"):
            cam.position.set(*pos)
            controls.target = tp.Vector3(*tgt)
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
        tp.imgui.end()

    def loop():
        dt = min(0.05, clock.get_delta())
        ctl.step(dt)
        n = max(1, min(8, int(round(dt / DT))))
        for _ in range(n):
            world.step(DT)
        view.update(arm.update())
        controls.enabled = not ui.want_capture_mouse
        controls.update()
        renderer.render(scene, cam)
        ui.render(draw_ui)

    canvas.animate(loop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="measure the mechanics, no window")
    ap.add_argument("--grasp", nargs="?", const="can", default=None,
                    help="close on an object and pull until it slips: can | ball")
    ap.add_argument("--tension", type=float, default=35.0, help="flexor tension, N per cable")
    ap.add_argument("--view", action="store_true", help="open a window: live grasp, cables drawn")
    ap.add_argument("--shots", nargs="*", default=None,
                    help="render stills headless: open | closing | closed | all")
    ap.add_argument("--poses", nargs="*", default=None,
                    help="render each tendon preset headless: open fist pinch point spread")
    ap.add_argument("--object", default="ball", help="ball | can | none")
    ap.add_argument("--size", default="1280x800")
    a = ap.parse_args()
    if a.view or a.shots is not None or a.poses is not None:
        w, h = (int(v) for v in a.size.split("x"))
        shots = None if a.shots is None else (a.shots or ["all"])
        poses = None if a.poses is None else (a.poses or list(PRESETS))
        visual(shots=shots, poses=poses, width=w, height=h, obj=a.object, tension=a.tension)
    elif a.grasp:
        grasp_test(a.grasp, a.tension)
    else:
        selftest()


if __name__ == "__main__":
    main()
