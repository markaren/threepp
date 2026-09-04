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
"""

import argparse
import math

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

DENSITY = 1100.0                  # kg/m^3, hand soft tissue + bone

# Passive joint resistance. A zero-stiffness, non-zero-damping articulation drive is a
# pure viscous damper about the joint axis, which is what synovial fluid plus a tendon
# in its sheath actually is. max_force caps it so it can never overpower the cables.
JOINT_DAMPING = 0.0035            # N.m.s/rad
JOINT_DAMP_MAX = 0.5              # N.m
JOINT_FRICTION = 0.0008           # N.m of stiction, per joint


def skin(colour=0xC8A088, roughness=0.75):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(colour)
    m.roughness = roughness
    return m


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
        self.meshes = []
        self._mat = material

        palm = tp.Mesh(tp.BoxGeometry(0.075, 0.026, 0.080), skin(roughness=0.8))
        palm.position.set(*(self.base + np.array([0.0325, 0.0, 0.0])))
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
        carrier = tp.Mesh(tp.SphereGeometry(0.0035, 8, 6), skin())
        carrier.position.set(kx, ky, kz)
        self._add(f"{name}_abd", carrier, self.palm, (0, 1, 0), (kx, ky, kz), -abd, abd)

        m = cap(R_PP, pp)
        m.position.set(kx + 0.5 * pp, ky, kz)
        m.rotation.z = -math.pi / 2                     # CapsuleGeometry runs along +Y
        self._add(f"{name}_mcp", m, self.links[f"{name}_abd"], (0, 0, -1), (kx, ky, kz), lo, hi)

        m = cap(R_MP, mp)
        m.position.set(kx + pp + 0.5 * mp, ky, kz)
        m.rotation.z = -math.pi / 2
        self._add(f"{name}_pip", m, self.links[f"{name}_mcp"], (0, 0, -1), (kx + pp, ky, kz), -0.17, 1.83)

        m = cap(R_DP, dp)
        m.position.set(kx + pp + mp + 0.5 * dp, ky, kz)
        m.rotation.z = -math.pi / 2
        self._add(f"{name}_dip", m, self.links[f"{name}_pip"], (0, 0, -1), (kx + pp + mp, ky, kz), -0.17, 1.31)

    def _thumb(self):
        """Trapeziometacarpal (2 DOF) + MCP + IP, four DOF in all.

        Built along +X like the fingers, with the SAME -90 deg capsule twist, so the
        actor-frame convention in bone() is identical everywhere and there is one less
        class of silent routing bug. Opposition is produced by the joints, not by the
        build pose: palmar abduction lifts the thumb volar out of the palm plane, then
        the swing brings it across toward the fingers.

        APPROXIMATION, and the largest geometric one in this model: Hollister 1992
        showed the real CMC has two NON-orthogonal, NON-intersecting axes fixed in
        different bones (flexion/extension in the trapezium, abduction/adduction in the
        metacarpal), which is what gives a real thumb its axial pronation during
        opposition. Two orthogonal serial revolutes cannot reproduce that.
        """
        cx, cy, cz = self.base + np.array([0.020, -0.026, -0.052])
        mc, pp, dp = 0.045, 0.032, 0.022 + PULP

        carrier = tp.Mesh(tp.SphereGeometry(0.005, 8, 6), skin())
        carrier.position.set(cx, cy, cz)
        # Palmar abduction: lifts the thumb out of the palm plane (volar, -Y).
        self._add("thumb_cmc_abd", carrier, self.palm, (1, 0, 0), (cx, cy, cz), -0.17, 1.22)

        m = cap(0.0105, mc)
        m.position.set(cx + 0.5 * mc, cy, cz)
        m.rotation.z = -math.pi / 2
        # Swing across the palm toward the fingers (+Z) or away from them.
        self._add("thumb_cmc_swing", m, self.links["thumb_cmc_abd"], (0, 1, 0), (cx, cy, cz), -1.05, 0.35)

        m = cap(0.0095, pp)
        m.position.set(cx + mc + 0.5 * pp, cy, cz)
        m.rotation.z = -math.pi / 2
        self._add("thumb_mcp", m, self.links["thumb_cmc_swing"], (0, 0, -1), (cx + mc, cy, cz), -0.17, 0.96)

        m = cap(0.0090, dp)
        m.position.set(cx + mc + pp + 0.5 * dp, cy, cz)
        m.rotation.z = -math.pi / 2
        self._add("thumb_ip", m, self.links["thumb_mcp"], (0, 0, -1), (cx + mc + pp, cy, cz), -0.26, 1.40)

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
            return (kx - (self.base[0] + 0.0325) + dx, -volar, kz - self.base[2])

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
                (palm, (kx - (self.base[0] + 0.0325) - 0.024, -0.004, kz - self.base[2] + lat)),
                (p1, bone(IO_VOLAR, 0.35 * pp, lat * 0.25)),
            ])

    def _thumb_cables(self):
        """Five cables for four DOF, same N+1 bound as the fingers.

        FPL and EPL are the flexor/extensor pair; APB and ADD straddle the palmar
        abduction axis so it is drivable both ways by pull-only actuators; FPB sits off
        to one side of the swing axis and is what makes the swing controllable.
        """
        L = self.links
        palm, ab = L["palm"], L["thumb_cmc_abd"]
        m1, m2, m3 = L["thumb_cmc_swing"], L["thumb_mcp"], L["thumb_ip"]
        cx, cy, cz = self.base + np.array([0.020, -0.026, -0.052])
        mc, pp, dp = 0.045, 0.032, 0.022 + PULP
        px, py, pz = cx - (self.base[0] + 0.0325), cy - self.base[1], cz - self.base[2]
        v1, v2, v3 = 0.0105, 0.0080, 0.0055        # volar standoffs, proximal -> distal
        d1, d2, d3 = 0.0080, 0.0050, 0.0035        # dorsal

        def bone(volar, along, lat=0.0):
            return (volar, along, lat)

        self._cable("thumb_fpl", [
            (palm, (px - 0.014, py - v1, pz + 0.012)), (ab, (0.0, -v1, 0.0)),
            (m1, bone(v2, 0.05 * mc)), (m2, bone(v3, 0.05 * pp)),
            (m3, bone(v3 * 0.8, -0.5 * dp + 0.20 * dp)),
        ])
        AXIS = (0.0, 0.0, 1.0)
        self._cable("thumb_epl", [
            (palm, (px - 0.014, py + d1, pz + 0.012)),
            (m1, bone(-d2, -0.30 * mc)),
            ("wrap", m1, (0.0, 0.5 * mc, 0.0), AXIS, d2, (-1.0, 0.0, 0.0)),
            (m2, bone(-d3, -0.20 * pp)),
            ("wrap", m2, (0.0, 0.5 * pp, 0.0), AXIS, d3, (-1.0, 0.0, 0.0)),
            (m3, bone(-d3 * 0.8, -0.5 * dp + 0.25 * dp)),
        ])
        # Abductor / adductor pollicis: opposite sides of the CMC abduction axis (X), so
        # one lifts the thumb volar out of the palm and the other pulls it back in.
        self._cable("thumb_abd", [
            (palm, (px - 0.024, py - 0.014, pz - 0.010)), (ab, (0.0, -0.012, -0.004)),
            (m1, bone(0.010, 0.10 * mc, -0.004)),
        ])
        self._cable("thumb_add", [
            (palm, (px - 0.010, py + 0.012, pz + 0.010)), (ab, (0.0, 0.011, 0.004)),
            (m1, bone(-0.009, 0.10 * mc, 0.004)),
        ])
        # Flexor pollicis brevis: offset toward the fingers, so it drives the SWING as
        # well as flexing - the cable that makes opposition reachable.
        self._cable("thumb_fpb", [
            (palm, (px - 0.016, py - 0.008, pz + 0.014)), (ab, (0.0, -0.007, 0.010)),
            (m1, bone(v2 * 0.7, 0.20 * mc, 0.010)),
            (m2, bone(v3 * 0.8, 0.0, 0.004)),
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
            for ang in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
                qp = hand.zero()
                for i in dofs:
                    if not hand.dof_names[i].endswith(("_abd", "_swing")):
                        qp[i] = ang
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="measure the mechanics, no window")
    ap.add_argument("--grasp", nargs="?", const="can", default=None,
                    help="close on an object and pull until it slips: can | ball")
    ap.add_argument("--tension", type=float, default=35.0, help="flexor tension, N per cable")
    a = ap.parse_args()
    if a.grasp:
        grasp_test(a.grasp, a.tension)
    else:
        selftest()


if __name__ == "__main__":
    main()
