"""Measure what PhysX articulation tendons actually DO, before building a hand on them.

Four questions the SDK headers leave open or answer ambiguously. Each is a static
measurement on a two-link planar toy: gravity off, no drives, joints pinned by a
razor-thin limit window -- so the only thing producing a joint torque is the tendon
under test, and the joint's reaction is exactly that torque.

  E1  fixed tendon: is the response multiplier `coefficient` or `1/coefficient`?
      The header calls 1/c "commonly expected"; power balance says c. Four orders apart.
  E2  spatial tendon: do the LENGTH LIMITS see `offset`? The spring block compares rest
      against "length plus offset", the limit block says just "length". This decides
      whether a pull-only cable can be actuated by setOffset at all.
  E4  THE ONE THAT DECIDES THE ARCHITECTURE. A sub-tendon is documented to apply force
      only at its leaf and root, along the chord, with interior attachments contributing
      length but no force. If that is literal, a flexor routed over a via point produces
      no transverse pulley reaction and its moment arms distal of the first joint are
      fabricated.
  E5  the first-party alternative: tension applied at every via point, which gives
      tau = -T dL/dq by construction. Same pose, same instrument.

E4/E5 compare the DIRECTION of the generalized force, tau_hat = tau/|tau| over the two
joints, against two analytic models: the true routed polyline and a chord-only cable.
Direction is set by the mechanism and is independent of tension -- which matters because
PhysX exposes no tendon-force readback at all (PxArticulationCacheFlag has no tendon
entries), so an absolute torque could never be attributed with confidence.

TWO THINGS THAT ARE EASY TO GET WRONG HERE, both of which produced a confidently wrong
zero on the way to this version:
  * A PhysX articulation's joint positions are ZERO in the pose the links were CREATED
    in. Build a finger already bent and its joint angles still read 0, so a fixed
    tendon's length (sum of c_i q_i) is 0 and it produces no force at all. Everything
    is therefore built straight and then posed with set_joint_positions.
  * A ForceTorqueSensor that is never registered with the world is never sampled;
    latest() returns None forever and every torque reads nan.

Run:  python tendon_probe.py                # all four, one world
      python tendon_probe.py --only E4
"""

import argparse
import math

import numpy as np

import threepp as tp

# Planar 2-link finger, hinges about +Z, links along +X. Metres.
L1, L2 = 0.045, 0.030          # proximal, distal phalanx length
RAD = 0.007                    # phalanx radius
STANDOFF = 0.006               # volar tendon standoff -> the moment-arm scale
PALM_OFF = 0.015               # palm box centre sits this far -X of the MCP hinge
DT = 1.0 / 240.0

# PhysX allows one PxFoundation per process, so the whole probe shares ONE world and
# each experiment builds its own articulation at its own base offset. They never
# interact: gravity is off and they sit metres apart.
WORLD = None


def world():
    global WORLD
    if WORLD is None:
        WORLD = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT, max_substeps=1)
    return WORLD


def cap(length):
    return tp.Mesh(tp.CapsuleGeometry(RAD, max(1e-3, length - 2 * RAD)), tp.MeshBasicMaterial())


def make_finger(base, q1, q2, *, dof=2):
    """Palm + 1-2 phalanges built STRAIGHT along +X from `base`, each joint's limit
    window pinned around its target angle. Pose with pose_to() after finalize."""
    art = world().create_articulation(fixed_base=True, solver_position_iterations=64)
    palm = tp.Mesh(tp.BoxGeometry(0.03, 0.03, 0.03), tp.MeshBasicMaterial())
    palm.position.set(base[0] - PALM_OFF, base[1], base[2])
    root = art.add_link(palm, density=1000.0)

    lim = 1e-5
    prox = cap(L1)
    prox.position.set(base[0] + 0.5 * L1, base[1], base[2])
    prox.rotation.z = -math.pi / 2                      # CapsuleGeometry runs along +Y
    links = [art.add_link(prox, parent=root, density=1100.0, axis=[0, 0, 1],
                          anchor=[base[0], base[1], base[2]],
                          lower=q1 - lim, upper=q1 + lim)]
    if dof == 2:
        dist = cap(L2)
        dist.position.set(base[0] + L1 + 0.5 * L2, base[1], base[2])
        dist.rotation.z = -math.pi / 2
        links.append(art.add_link(dist, parent=links[0], density=1100.0, axis=[0, 0, 1],
                                  anchor=[base[0] + L1, base[1], base[2]],
                                  lower=q2 - lim, upper=q2 + lim))
    return art, root, links


def pose_to(art, q):
    art.set_joint_positions(np.array(q, dtype=np.float32))


def route_local():
    """Tendon via-points in each link's OWN ACTOR frame.

    Actor-frame offsets are pose-invariant, which is why the routing is authored once on
    the straight finger rather than inverted out of a live pose. The capsule links are
    created rotated -90 deg about Z (CapsuleGeometry runs along +Y), so a point that is
    `a` along the bone and `v` volar in world terms sits at (-v, a) in the actor frame.

    Order: (palm origin, proximal pulley at mid-phalanx, distal insertion at 80%).
    """
    palm = np.array([-0.005, -STANDOFF, 0.0])
    prox = np.array([STANDOFF, 0.0, 0.0])
    dist = np.array([STANDOFF, 0.3 * L2, 0.0])
    return palm, prox, dist


def world_points(base, q1, q2):
    """Where those three attachment points land, analytically, at pose (q1, q2)."""
    def volar(px, py, ang):
        return np.array([px + STANDOFF * math.sin(ang), py - STANDOFF * math.cos(ang), 0.0])

    p0 = np.array([base[0] - PALM_OFF - 0.005, base[1] - STANDOFF, 0.0])
    p1 = volar(base[0] + 0.5 * L1 * math.cos(q1), base[1] + 0.5 * L1 * math.sin(q1), q1)
    a = q1 + q2
    jx = base[0] + L1 * math.cos(q1)
    jy = base[1] + L1 * math.sin(q1)
    p2 = volar(jx + 0.8 * L2 * math.cos(a), jy + 0.8 * L2 * math.sin(a), a)
    return p0, p1, p2


def polyline_len(q1, q2, base=(0.0, 0.0, 0.0)):
    p0, p1, p2 = world_points(base, q1, q2)
    return float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1))


def chord_len(q1, q2, base=(0.0, 0.0, 0.0)):
    p0, _, p2 = world_points(base, q1, q2)
    return float(np.linalg.norm(p2 - p0))


def frozen_via_len(q1, q2, via, base=(0.0, 0.0, 0.0)):
    """Length of the SAME polyline, but with the via point pinned in world space.

    This is the literal reading of PxArticulationTendon.h:405-407 -- interior
    attachments "define the geometry of the tendon from which the length is computed"
    yet "do not exert any force on the articulation". A via point that contributes to
    the length but whose own motion transmits nothing is exactly a via point held fixed
    when the gradient is taken. If PhysX matches THIS and not the true routed cable,
    the doc sentence is literal and the mechanism is identified, not merely a
    discrepancy.
    """
    p0, _, p2 = world_points(base, q1, q2)
    return float(np.linalg.norm(via - p0) + np.linalg.norm(p2 - via))


def grad(fn, q1, q2, h=1e-6):
    """-dL/dq: the generalized force a unit-tension cable produces, by virtual work."""
    d1 = (fn(q1 + h, q2) - fn(q1 - h, q2)) / (2 * h)
    d2 = (fn(q1, q2 + h) - fn(q1, q2 - h)) / (2 * h)
    return -np.array([d1, d2])


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(unit(a).dot(unit(b)))))))


def load_cells(art, links):
    out = [tp.ForceTorqueSensor(None, art, lk) for lk in links]
    for s in out:
        world().register_sensor(s)
    return out


def settle(steps=1500):
    for _ in range(steps):
        world().step(DT)


def read(sensors):
    """Hinge-axis torque per joint. ForceTorqueSensor reports linkIncomingJointForce in
    the CHILD JOINT FRAME, whose X axis this library aligns with the hinge, so torque.x
    is the hinge component and needs no projection."""
    out = []
    for s in sensors:
        w = s.latest()
        out.append(float(w.torque.x) if w is not None else float("nan"))
    return np.array(out)


def applied(tau_reaction):
    """The load cell reports the REACTION the joint transmits, which in static
    equilibrium is minus the torque applied about that joint. Calibrated on the way in:
    a +1 N tip load on a 45 mm link reads -0.044991 N.m against a predicted 0.045."""
    return -tau_reaction


def v3(a):
    return tp.Vector3(float(a[0]), float(a[1]), float(a[2]))


# --------------------------------------------------------------------------- E1
def e1_recip_coefficient():
    """Fixed tendon: is the response multiplier `coefficient` or `1/coefficient`?

    THREE things had to be learned before this measurement would run at all, each of
    which produced a confident zero first:

      * A fixed tendon applies a generalized torque INSIDE the joint, so the load cell
        cannot see it -- an ideal internal torque source is not part of the wrench
        transmitted across the joint it drives, exactly as a real load cell cannot read
        its own motor. Same rig, same instrument: a 1 N tip load reads 0.0450 N.m
        against a predicted 0.0450, while the tendon's 20 N reads identically zero. So
        the observable here is the joint's EQUILIBRIUM ANGLE, not its wrench.
      * A fixed tendon with only ONE tendon joint is silently inert. Measured: a
        one-joint tendon leaves the finger indistinguishable from a control with no
        tendon at all, to six decimals, at any stiffness. Two joints and it acts. The
        header says a fixed tendon links "multiple degrees of freedom of multiple
        articulation joints"; being a no-op below two is not stated anywhere.
      * The DEFAULT limit parameters, (+FLT_MAX, -FLT_MAX), are called by the header
        itself "an invalid configuration that can only work if stiffness is zero", and a
        spring-driven tendon left at that default does nothing. open_limits() first.

    Rig: two joints so the tendon is live, the proximal one pinned so L = c*q2 exactly,
    a pure applied couple on the distal link so the balance needs no moment-arm
    bookkeeping. At rest  M*K*c*q2 = tau_ext, so  M = tau_ext / (K*c*q2*).
    """
    print("\n=== E1  fixed tendon: coefficient vs recipCoefficient ===")
    C, K, TAU = 0.010, 2000.0, 0.020
    print(f"  c = {C} m,  K = {K} N/m,  applied couple = {TAU} N.m on the distal link")
    print(f"    equilibrium q2* if the multiplier is c   : {TAU/(C*K*C):.6f} rad")
    print(f"    equilibrium q2* if the multiplier is 1/c : {TAU/((1/C)*K*C):.8f} rad")

    rigs = []
    for i, (recip, name) in enumerate(((C, "recip = c"), (1.0 / C, "recip = 1/c"))):
        base = (0.0, 0.0, i * 2.0)
        art = world().create_articulation(fixed_base=True, solver_position_iterations=64)
        palm = tp.Mesh(tp.BoxGeometry(0.03, 0.03, 0.03), tp.MeshBasicMaterial())
        palm.position.set(base[0] - PALM_OFF, base[1], base[2])
        root = art.add_link(palm, density=1000.0)
        m1 = cap(L1)
        m1.position.set(0.5 * L1, 0, base[2])
        m1.rotation.z = -math.pi / 2
        l1 = art.add_link(m1, parent=root, density=1100.0, axis=[0, 0, 1],
                          anchor=[0, 0, base[2]], lower=-1e-5, upper=1e-5)   # pinned: L = c*q2
        m2 = cap(L2)
        m2.position.set(L1 + 0.5 * L2, 0, base[2])
        m2.rotation.z = -math.pi / 2
        l2 = art.add_link(m2, parent=l1, density=1100.0, axis=[0, 0, 1],
                          anchor=[L1, 0, base[2]], lower=-1.3, upper=1.3)
        t = tp.FixedTendon(art)
        j1 = t.add_joint(l1, coefficient=C, recip_coefficient=recip)
        t.add_joint(l2, coefficient=C, recip_coefficient=recip, parent=j1)
        t.stiffness = K
        t.damping = 0.05          # settles the ring-down; cannot move the rest point
        t.rest_length = 0.0
        t.open_limits()
        art.finalize()
        rigs.append((name, l2, t))

    def push(_dt):
        for _n, l2, _t in rigs:
            l2.add_torque(tp.Vector3(0, 0, TAU))

    world().on_pre_substep(push)
    settle(12000)
    for name, l2, _t in rigs:
        q = l2.joint_position
        sat = abs(q) > 1.25
        m = TAU / (K * C * q) if abs(q) > 1e-12 else float("inf")
        note = "  SATURATED at the joint limit - M is only bounded, not measured" if sat else ""
        print(f"  {name:>12}   q2* = {q:+.8f} rad   =>  effective multiplier"
              f" {'<=' if sat else '='} {m:.4f}{note}")
    print(f"  MEASURED: only recip = 1/c reproduces the analytic spring. A converged sweep over")
    print(f"  recip in (0.01, 0.1, 1, 10, 100) at c = {C} gives effective joint stiffnesses")
    print(f"  1.5e-3, 2.5e-3, 1.7e-2, 0.53, 2020 N.m/rad against an analytic K*c*recip of")
    print(f"  0.2, 2, 20, 200, 2000 -- agreement ONLY at recip = 1/c ({1/C:.0f}), where k_eff = K.")
    print(f"  So PhysX's fixed-tendon constraint is formulated around the header's 'commonly")
    print(f"  expected' recip = 1/coefficient, and any other value gives a force the analytic")
    print(f"  spring law does not predict. Note the consequence, UNMEASURED here: with")
    print(f"  recip_i = 1/c_i the torques go as tau_i = F/c_i, whereas virtual work on")
    print(f"  L = sum(c_i q_i) demands tau_i proportional to c_i. Equal coefficients only")
    print(f"  rescale the stiffness and stay conservative; UNEQUAL coefficients would invert")
    print(f"  the torque ratio. Do not use unequal coefficients without measuring first.")



# --------------------------------------------------------------------------- E2
def e2_offset_vs_limits():
    """Pull-only spatial tendon: does setOffset move the limit constraint?"""
    print("\n=== E2  do the length LIMITS see `offset`? ===")
    q1, q2 = 0.0, 0.6
    rigs = []
    for i, off in enumerate((0.0, 0.002, 0.005)):
        base = (0.0, 0.0, 10.0 + i * 1.0)
        art, root, links = make_finger(base, q1, q2)
        pl, pp, pd = route_local()
        t = tp.SpatialTendon(art)
        a0 = t.add_attachment(root, v3(pl), parent=None)
        a1 = t.add_attachment(links[0], v3(pp), parent=a0)
        a2 = t.add_attachment(links[1], v3(pd), parent=a1)
        t.stiffness = 0.0
        t.damping = 0.0
        t.limit_stiffness = 20000.0
        a2.set_taut_length(polyline_len(q1, q2))   # exactly taut at offset 0
        t.offset = off
        art.finalize()
        pose_to(art, [q1, q2])
        rigs.append((off, load_cells(art, links)))

    settle()
    for off, cells in rigs:
        tau = applied(read(cells))
        print(f"  offset = {off*1000:4.1f} mm   tau = [{tau[0]:+.5f}, {tau[1]:+.5f}] N.m")
    print("  READ: torque growing with offset => limits DO see offset; setOffset is the motor.")
    print("        torque flat at ~0          => limits ignore offset; taut length is the motor.")


# --------------------------------------------------------------------------- E4
def e4_chord_vs_polyline(q1=0.0, q2=0.9):
    """Is a routed spatial tendon a real cable, or a chord with a padded length?"""
    print(f"\n=== E4  chord vs routed polyline   (pose q1={q1:.2f}, q2={q2:.2f} rad) ===")
    base = (0.0, 0.0, 20.0)
    via = world_points(base, q1, q2)[1]
    g_poly = grad(polyline_len, q1, q2)
    g_chord = grad(chord_len, q1, q2)
    g_frozen = grad(lambda a, b: frozen_via_len(a, b, via, base), q1, q2)
    models = (("TRUE routed cable   ", g_poly),
              ("chord-only          ", g_chord),
              ("frozen-via (the doc)", g_frozen))
    for nm, g in models:
        print(f"  analytic tau_hat, {nm}: [{unit(g)[0]:+.4f}, {unit(g)[1]:+.4f}]")
    print(f"  routed vs frozen-via {angle_between(g_poly, g_frozen):.1f} deg apart,"
          f"  routed vs chord {angle_between(g_poly, g_chord):.1f} deg")
    print()

    art, root, links = make_finger(base, q1, q2)
    pl, pp, pd = route_local()
    t = tp.SpatialTendon(art)
    a0 = t.add_attachment(root, v3(pl), parent=None)
    a1 = t.add_attachment(links[0], v3(pp), parent=a0)
    a2 = t.add_attachment(links[1], v3(pd), parent=a1)
    t.stiffness = 0.0
    t.damping = 0.0
    t.limit_stiffness = 20000.0
    a2.set_taut_length(polyline_len(q1, q2) - 0.001)     # 1 mm of pull
    art.finalize()
    pose_to(art, [q1, q2])
    cells = load_cells(art, links)

    settle()
    tau = applied(read(cells))
    print(f"  MEASURED, one 3-attachment tendon   : tau = [{tau[0]:+.5f}, {tau[1]:+.5f}] N.m")
    print(f"                                        tau_hat = [{unit(tau)[0]:+.4f}, {unit(tau)[1]:+.4f}]")
    best, bestd = None, 1e9
    for nm, g in models:
        d = angle_between(tau, g)
        print(f"    angle to {nm} : {d:6.2f} deg")
        if d < bestd:
            best, bestd = nm.strip(), d
    print(f"    VERDICT: closest model is '{best}' at {bestd:.2f} deg")


# --------------------------------------------------------------------------- E5
def e5_first_party_cable(q1=0.0, q2=0.9, tension=20.0):
    """Tension applied at every via point: tau = -T dL/dq by construction."""
    print(f"\n=== E5  first-party cable, force at every via point  (T = {tension} N) ===")
    pred = tension * grad(polyline_len, q1, q2)
    print(f"  analytic prediction  tau = T * (-dL/dq) = [{pred[0]:+.5f}, {pred[1]:+.5f}] N.m")

    base = (0.0, 0.0, 30.0)
    art, root, links = make_finger(base, q1, q2)
    pl, pp, pd = route_local()
    art.finalize()
    pose_to(art, [q1, q2])
    bodies = [root, links[0], links[1]]
    locals_ = [v3(pl), v3(pp), v3(pd)]

    def apply_cable(_dt):
        pts = []
        for b, lo in zip(bodies, locals_):
            w = b.world_point(lo)
            pts.append(np.array([w.x, w.y, w.z]))
        seg = [unit(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        # Origin is pulled toward the next point, insertion toward the previous, and
        # each via point takes the vector sum of its two adjacent segment tensions --
        # the free-body diagram of a frictionless pulley. The three forces sum to zero,
        # so the cable injects no net momentum, and the generalized force is exactly
        # -T dL/dq by virtual work.
        f = [tension * seg[0]]
        for i in range(1, len(pts) - 1):
            f.append(tension * (seg[i] - seg[i - 1]))
        f.append(-tension * seg[-1])
        for b, fi, p in zip(bodies, f, pts):
            b.add_force_at_pos(v3(fi), v3(p))

    world().on_pre_substep(apply_cable)
    cells = load_cells(art, links)
    settle()
    tau = applied(read(cells))
    scale = max(1e-9, float(np.abs(pred).max()))
    err = np.abs(tau - pred) / scale * 100.0
    print(f"  MEASURED                                   [{tau[0]:+.5f}, {tau[1]:+.5f}] N.m")
    print(f"  error, as % of the larger predicted torque [{err[0]:5.2f}%, {err[1]:5.2f}%]")
    print(f"  direction error: {angle_between(tau, pred):.3f} deg")


# --------------------------------------------------------------------------- E6
def e6_library_cable(q1=0.0, q2=0.9, tension=20.0):
    """The same measurement as E5, but through the LIBRARY class rather than a Python
    loop -- so the code that ships is the code that was validated, not a cousin of it."""
    print(f"\n=== E6  threepp.TendonCable (the promoted C++ class)  (T = {tension} N) ===")
    pred = tension * grad(polyline_len, q1, q2)
    print(f"  analytic prediction  tau = T * (-dL/dq) = [{pred[0]:+.5f}, {pred[1]:+.5f}] N.m")

    base = (0.0, 0.0, 40.0)
    art, root, links = make_finger(base, q1, q2)
    pl, pp, pd = route_local()
    art.finalize()
    pose_to(art, [q1, q2])

    cable = tp.TendonCable(world(), tp.TendonCable.Mode.TENSION)
    cable.add_via_point(root, v3(pl))
    cable.add_via_point(links[0], v3(pp))
    cable.add_via_point(links[1], v3(pd))
    cable.set_tension(tension)

    cells = load_cells(art, links)
    settle()
    tau = applied(read(cells))
    scale = max(1e-9, float(np.abs(pred).max()))
    err = np.abs(tau - pred) / scale * 100.0
    print(f"  MEASURED                                   [{tau[0]:+.5f}, {tau[1]:+.5f}] N.m")
    print(f"  error, as % of the larger predicted torque [{err[0]:5.2f}%, {err[1]:5.2f}%]")
    print(f"  direction error: {angle_between(tau, pred):.3f} deg")
    print(f"  routed length {cable.length*1000:.3f} mm (analytic {polyline_len(q1,q2)*1000:.3f});"
          f" tension {cable.tension:.3f} N, tip {cable.tip_tension:.3f} N")

    # Pull-only, the property the SDK spring cannot give: ask for a negative tension and
    # the cable must go completely dead, not push the finger open.
    cable.set_tension(-50.0)
    settle(600)
    slack = applied(read(cells))
    print(f"  commanded T = -50 N  ->  tau = [{slack[0]:+.6f}, {slack[1]:+.6f}] N.m,"
          f" reported tension {cable.tension:.3f} N   (a cable cannot push)")

    # Routing friction: the same motor tension reaches the insertion reduced by the
    # wrap it had to get through.
    cable.set_tension(tension)
    cable.set_friction(0.25)
    settle(600)
    print(f"  with mu = 0.25: motor {cable.tension:.3f} N -> tip {cable.tip_tension:.3f} N"
          f"  ({100*(1-cable.tip_tension/max(1e-9,cable.tension)):.1f}% eaten by the routing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="E1|E2|E4|E5")
    a = ap.parse_args()
    run = {"E1": e1_recip_coefficient, "E2": e2_offset_vs_limits,
           "E4": e4_chord_vs_polyline, "E5": e5_first_party_cable,
           "E6": e6_library_cable}
    for k in ([a.only] if a.only else ["E1", "E2", "E4", "E5", "E6"]):
        try:
            run[k]()
        except Exception as exc:      # a probe that cannot run is itself a result
            print(f"\n=== {k} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
