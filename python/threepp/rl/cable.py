"""TendonCable, batched in torch, so a routed-cable robot can live in a direct-GPU batch.

WHY THIS FILE EXISTS. `threepp::TendonCable` (include/threepp/extras/physx/TendonCable.hpp)
applies its pulley forces through `PxRigidBody::addForce/addTorque`, and PhysX rejects both
under `PxSceneFlag::eENABLE_DIRECT_GPU_API`. So the CPU tendon hand cannot enter `GpuSim` --
not because the law is expensive, but because of where the forces go in. The batch already
exposes per-link pose reads and per-link world-frame force AND torque writes, so the identical
law evaluates as a torch kernel over [K envs, C cables] and goes in through those writes
instead. Nothing here is a new model: every rule below is ported line-for-line from
`resolve()` and `apply()`, including the two that took the longest to get right in C++ --
the tangent-pair sign convention and the non-sheathed DISENGAGE test.

WHAT IS PORTED, and what is not:
  + Via points on any link, in the link's ACTOR frame.
  + Wraps: planar tangent-arc-tangent around a cylinder, the side chosen by the sideHint,
    the sweep taken the short way round between the tangent points of one pair.
  + The DISENGAGE rule: a non-sheathed wrap whose chord already clears the cylinder is NOT
    in contact, and the cable is a straight chord there. This is the whole difference
    between a flexor and an extensor -- see the comment in TendonCable.hpp.
  + The free-body force law: anchor +T*s0, interior T*(s_i - s_{i-1}), insertion -T*s_{n-2},
    each at its own world point, as force-at-the-COM plus (p - com) x F.
  - TENSION mode only. Length mode needs a per-cable rate across substeps and the hand uses
    nothing else.
  - FRICTIONLESS (mu = 0). The hand sets no friction; the capstan factor would be one more
    cumulative product along the slot axis if it were ever needed.

THE ONE APPROXIMATION THE BATCH ADDS is the arc sampling. The C++ picks its step count from
the sweep (`ceil(|sweep| / 0.25)` segments, so a variable number of points); a batched layout
has to be static, so every wrap gets exactly `arc_points` points whatever its sweep. That
changes the SAMPLED length of an arc by the usual chord deficit and nothing else -- the force
law is the exact gradient of whatever polyline it is given either way, so the cable is still
self-consistent. The default of 8 is MEASURED, not guessed: over 30 poses spanning the tendon
hand's joint limits, the worst routed length disagreed with the C++ cable by 0.2332 mm at 4
arc points and by 0.0200 mm at 8; in both cases the five worst cables were exactly the five
wrapped ones (every via-only cable sits under 0.005 mm either way, which is about where the
float32 pose readback itself lands). See tendon_hand_gpu_check.py gate A1a.

COM == LINK ORIGIN is assumed, because `torque` is written about the centre of mass and the
world points are built from the link ORIGIN pose the batch reports. That holds for every link
of the tendon hand: `Articulation::addLink` infers an analytic shape centred on the actor
origin and `updateMassAndInertia` puts the COM there. The one exception is the convex-hull
palm, which is the FIXED ROOT, so no force applied to it does anything at all.
"""
import numpy as np
import torch


def quat_rotate(q, v):
    """Rotate v [...,3] by the quaternion q [...,4] = (qx,qy,qz,qw). Batched, broadcasting."""
    qv, qw = q[..., 0:3], q[..., 3:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


class CableRouting:
    """The static node lists of every cable, with link NAMES instead of link handles.

    A `TendonCable` owns its nodes privately and exposes only counts and the resolved path,
    so the routing is taken from the Python side that authored it: `Hand._cable` records its
    `nodes` argument into `hand.routing`, which is the same list it hands to the C++ object,
    so there is no second description of the routing to drift.
    """

    def __init__(self, cables):
        self.cables = dict(cables)        # name -> list of nodes, in order
        self.names = list(self.cables)

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_hand(cls, hand):
        """Read `hand.routing` and replace every ArticulationLink handle with its name.

        Matched by object identity against `hand.links`, which is where those very handles
        came from -- `add_link` returns one Python wrapper and the dict holds it.
        """
        if not getattr(hand, "routing", None):
            raise RuntimeError(
                "CableRouting.from_hand: hand.routing is empty. Call hand.route() (or "
                "hand.route(build=False) for the GPU path) before extracting the routing.")
        by_id = {id(lk): nm for nm, lk in hand.links.items()}
        out = {}
        for name, nodes in hand.routing.items():
            path = []
            for nd in nodes:
                if nd[0] == "wrap":
                    _, link, centre, axis, radius, side = nd[:6]
                    sheathed = bool(nd[6]) if len(nd) > 6 else False
                    path.append(("wrap", by_id[id(link)], tuple(float(v) for v in centre),
                                 tuple(float(v) for v in axis), float(radius),
                                 tuple(float(v) for v in side), sheathed))
                else:
                    link, off = nd
                    path.append(("via", by_id[id(link)], tuple(float(v) for v in off)))
            out[name] = path
        return cls(out)


def link_index_map(hand, link_pose_row, tol=2e-3, margin=4.0):
    """add-order link NAME -> the link's index in the direct-GPU batch's link buffers.

    The batch reports links in PhysX BREADTH-FIRST order, which is not add order and which a
    branched articulation like a hand reorders heavily. Rather than reconstructing PhysX's
    traversal (an assumption that would fail silently), the two orders are matched by REST
    POSITION: every link of the hand sits at a distinct place, so one warm-up step's worth of
    link poses is enough to identify all 21.

    `link_pose_row` is one env's [L,7] slice in PhysX layout [qx,qy,qz,qw, px,py,pz]; `hand`
    is a CPU hand built at the SAME base, or just its {name: (x,y,z)} rest positions -- PhysX
    allows one foundation per process, so a caller that has already torn its CPU world down to
    make room for the batch passes the positions it kept. Raises with the offending names if
    any match is further than `tol`, or if the runner-up is not at least `margin` times worse:
    an ambiguous match here would silently mis-address every cable force.
    """
    pose = np.asarray(link_pose_row, dtype=np.float64)
    gpu_p = pose[:, 4:7]
    if hasattr(hand, "links"):
        rest = {n: (lk.position.x, lk.position.y, lk.position.z) for n, lk in hand.links.items()}
    else:
        rest = dict(hand)
    names = list(rest)
    cpu_p = np.array([rest[n] for n in names], dtype=np.float64)
    d = np.linalg.norm(cpu_p[:, None, :] - gpu_p[None, :, :], axis=2)   # [n_names, L]
    order = np.argsort(d, axis=1)
    best, second = order[:, 0], order[:, 1]
    r_best = d[np.arange(len(names)), best]
    r_second = d[np.arange(len(names)), second]
    bad = [f"{names[i]}: nearest GPU link {best[i]} at {r_best[i]*1000:.3f} mm, "
           f"runner-up {second[i]} at {r_second[i]*1000:.3f} mm"
           for i in range(len(names))
           if r_best[i] > tol or r_second[i] < margin * max(r_best[i], 1e-9)]
    if bad:
        raise RuntimeError("link_index_map: rest-position matching is not unique:\n  "
                           + "\n  ".join(bad))
    if len(set(best.tolist())) != len(names):
        raise RuntimeError("link_index_map: two links matched the same GPU index")
    return {names[i]: int(best[i]) for i in range(len(names))}, float(r_best.max())


class BatchedCables:
    """Every cable of every env, resolved and applied in one pass.

    LAYOUT. One static [C, N] slot grid: a via node owns one slot, a wrap owns `arc_points`
    consecutive slots, and cables shorter than N are padded by REPEATING their insertion
    point. Repetition is not a special case anywhere downstream: a zero-length segment has a
    zero direction, so it adds nothing to the length and the uniform force expression
    T*(s_i - s_{i-1}) collapses to exactly the insertion's -T*s_{n-2} at the last real slot
    and to zero past it.
    """

    def __init__(self, routing, index_map, num_links, device="cuda", arc_points=8):
        if arc_points < 2:
            raise ValueError("arc_points must be at least 2 (the two tangent points)")
        self.routing, self.device = routing, torch.device(device)
        self.names = list(routing.names)
        self.C, self.L, self.W = len(self.names), int(num_links), int(arc_points)

        # ---- pass 1: hand out slots ------------------------------------------------
        node_slot, n_slots = {}, {}
        for ci, nm in enumerate(self.names):
            s = 0
            for i, nd in enumerate(routing.cables[nm]):
                node_slot[(ci, i)] = s
                s += 1 if nd[0] == "via" else self.W
            n_slots[ci] = s
        self.N = max(n_slots.values())

        # ---- pass 2: fill the static tables -----------------------------------------
        owner = np.zeros((self.C, self.N), dtype=np.int64)
        via_flat, via_link, via_off = [], [], []
        w_flat0, w_link, w_ctr, w_ax, w_rad, w_side, w_sheath, w_prev, w_next = ([] for _ in range(9))
        pad_dst, pad_src = [], []
        for ci, nm in enumerate(self.names):
            nodes = routing.cables[nm]
            if len(nodes) < 2 or nodes[0][0] != "via" or nodes[-1][0] != "via":
                raise ValueError(f"cable {nm}: must start and end with a via point")
            for i, nd in enumerate(nodes):
                s = node_slot[(ci, i)]
                if nd[0] == "via":
                    _, link, off = nd
                    owner[ci, s] = index_map[link]
                    via_flat.append(ci * self.N + s)
                    via_link.append(index_map[link])
                    via_off.append(off)
                else:
                    _, link, ctr, ax, rad, side, sheathed = nd
                    # The C++ resolve() places a wrap between the PREVIOUS RESOLVED POINT and
                    # the NEXT VIA node, and silently drops it if either is missing. Here that
                    # would be a static-layout bug rather than a quiet no-op, so it is refused.
                    if i == 0 or nodes[i - 1][0] != "via" or i + 1 >= len(nodes) or nodes[i + 1][0] != "via":
                        raise ValueError(f"cable {nm}: wrap at node {i} must sit between two via points")
                    owner[ci, s:s + self.W] = index_map[link]
                    w_flat0.append(ci * self.N + s)
                    w_link.append(index_map[link])
                    w_ctr.append(ctr); w_ax.append(ax); w_rad.append(rad); w_side.append(side)
                    w_sheath.append(bool(sheathed))
                    w_prev.append(ci * self.N + node_slot[(ci, i - 1)])
                    w_next.append(ci * self.N + node_slot[(ci, i + 1)])
            last = node_slot[(ci, len(nodes) - 1)]
            for s in range(n_slots[ci], self.N):
                owner[ci, s] = owner[ci, last]
                pad_dst.append(ci * self.N + s)
                pad_src.append(ci * self.N + last)

        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        self.owner_flat = t(owner.reshape(-1), torch.long)
        self.via_flat, self.via_link = t(via_flat, torch.long), t(via_link, torch.long)
        self.via_off = t(via_off).view(-1, 3)
        self.has_wrap = len(w_flat0) > 0
        if self.has_wrap:
            self.w_flat0, self.w_link = t(w_flat0, torch.long), t(w_link, torch.long)
            self.w_ctr, self.w_ax = t(w_ctr).view(-1, 3), t(w_ax).view(-1, 3)
            self.w_rad = t(w_rad).view(-1)
            self.w_side, self.w_sheath = t(w_side).view(-1, 3), t(w_sheath, torch.bool).view(-1)
            self.w_prev, self.w_next = t(w_prev, torch.long), t(w_next, torch.long)
            # the W sample fractions: the arc runs tangent point to tangent point inclusive,
            # the disengaged chord uses strictly interior fractions so its sub-chords sum to
            # |PQ| exactly and every one of its points has collinear neighbours (zero force).
            k = torch.arange(self.W, device=self.device, dtype=torch.float32)
            self.arc_frac = (k / (self.W - 1)).view(1, 1, self.W, 1)
            self.chord_frac = ((k + 1.0) / (self.W + 1.0)).view(1, 1, self.W, 1)
        self.pad_dst = t(pad_dst, torch.long) if pad_dst else None
        self.pad_src = t(pad_src, torch.long) if pad_dst else None

    # ---- resolution ------------------------------------------------------------------
    def resolve(self, link_pose):
        """link_pose [K, L, 7] (PhysX layout) -> the world polyline p [K, C, N, 3]."""
        K = link_pose.shape[0]
        q_all, x_all = link_pose[..., 0:4], link_pose[..., 4:7]
        p = torch.zeros(K, self.C * self.N, 3, device=link_pose.device, dtype=link_pose.dtype)

        q = q_all[:, self.via_link]                                   # [K, nv, 4]
        p[:, self.via_flat] = x_all[:, self.via_link] + quat_rotate(q, self.via_off.expand(K, -1, -1))

        if self.has_wrap:
            p[:, self._wrap_slots()] = self._arcs(p, q_all, x_all).reshape(K, -1, 3)
        if self.pad_dst is not None:
            p[:, self.pad_dst] = p[:, self.pad_src]
        return p.view(K, self.C, self.N, 3)

    def _wrap_slots(self):
        if not hasattr(self, "_wslots"):
            k = torch.arange(self.W, device=self.device, dtype=torch.long)
            self._wslots = (self.w_flat0.view(-1, 1) + k.view(1, -1)).reshape(-1)
        return self._wslots

    def _arcs(self, p, q_all, x_all):
        """Planar tangent-arc-tangent for every wrap at once -> [K, nw, W, 3].

        A straight port of TendonCable::arc(), including the branch selection: with the 2D
        basis anchored so P sits at angle 0, a cable leaving P at +ta must arrive at
        angB - tb, and the sweep between two tangent points of ONE pair is always the short
        way (a cable can never wrap more than half a circle between them). Getting either
        sign backwards makes both branches fail and the wrap silently becomes a chord.
        """
        K = p.shape[0]
        P, Q = p[:, self.w_prev], p[:, self.w_next]                    # [K, nw, 3]
        q, x = q_all[:, self.w_link], x_all[:, self.w_link]
        Cw = x + quat_rotate(q, self.w_ctr.expand(K, -1, -1))
        n = quat_rotate(q, self.w_ax.expand(K, -1, -1))
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        hw = quat_rotate(q, self.w_side.expand(K, -1, -1))
        r = self.w_rad.view(1, -1, 1)

        flat = lambda X: (X - Cw) - n * ((X - Cw) * n).sum(-1, keepdim=True)
        a, b = flat(P), flat(Q)
        ra = a.norm(dim=-1, keepdim=True)
        rb = b.norm(dim=-1, keepdim=True)
        u = a / ra.clamp_min(1e-9)
        v = torch.linalg.cross(n, u, dim=-1)
        dot = lambda A, B: (A * B).sum(-1, keepdim=True)
        angB = torch.atan2(dot(b, v), dot(b, u))
        ta = torch.acos((r / ra.clamp_min(1e-9)).clamp(-1.0, 1.0))
        tb = torch.acos((r / rb.clamp_min(1e-9)).clamp(-1.0, 1.0))
        hang = torch.atan2(dot(hw, v), dot(hw, u))

        pi = torch.pi
        wrap_pi = lambda z: torch.remainder(z + pi, 2.0 * pi) - pi     # == the C++ fmod + fixup
        best_t1 = best_sweep = best_score = None
        for sg in (1.0, -1.0):
            t1 = sg * ta
            sweep = wrap_pi(angB - sg * tb - t1)
            score = torch.cos(wrap_pi(t1 + 0.5 * sweep - hang))        # +1 on the hinted side
            if best_score is None:
                best_t1, best_sweep, best_score = t1, sweep, score
            else:
                take = score > best_score
                best_t1 = torch.where(take, t1, best_t1)
                best_sweep = torch.where(take, sweep, best_sweep)
                best_score = torch.where(take, score, best_score)

        # Contact only where the free chord would actually cut the pulley -- unless the
        # tendon is SHEATHED and cannot leave it. This is the flexor/extensor distinction,
        # not an optimisation; see TendonCable.hpp.
        d = Q - P
        dd = dot(d, d)
        tt = (dot(Cw - P, d) / dd.clamp_min(1e-12)).clamp(0.0, 1.0)
        clear = (Cw - (P + tt * d)).norm(dim=-1, keepdim=True)
        engaged = (ra > r * 1.001) & (rb > r * 1.001) & (self.w_sheath.view(1, -1, 1) | (clear < r))

        th = best_t1.unsqueeze(2) + best_sweep.unsqueeze(2) * self.arc_frac       # [K,nw,W,1]
        arc = (Cw.unsqueeze(2) + u.unsqueeze(2) * (r.unsqueeze(2) * torch.cos(th))
               + v.unsqueeze(2) * (r.unsqueeze(2) * torch.sin(th)))
        chord = P.unsqueeze(2) + d.unsqueeze(2) * self.chord_frac
        return torch.where(engaged.unsqueeze(2), arc, chord)

    # ---- the law ----------------------------------------------------------------------
    def lengths(self, link_pose):
        """Total routed length of every cable, [K, C] metres."""
        p = self.resolve(link_pose)
        return (p[:, :, 1:] - p[:, :, :-1]).norm(dim=-1).sum(-1)

    def apply(self, link_pose, tension):
        """link_pose [K,L,7], tension [K,C] (N, clamped at zero -- a cable cannot push).

        Returns (force [K,L,3], torque [K,L,3], length [K,C]): world-frame force at each
        link's centre of mass and torque about it, ready for write_link_force /
        write_link_torque, plus the routed length the observation wants for free.
        """
        K = link_pose.shape[0]
        p = self.resolve(link_pose)
        d = p[:, :, 1:] - p[:, :, :-1]                                  # [K, C, N-1, 3]
        m = d.norm(dim=-1, keepdim=True)
        s = d / m.clamp_min(1e-9)                                       # zero-length -> zero dir
        length = m.squeeze(-1).sum(-1)

        # One expression for all three cases of the free-body diagram: pad the segment
        # directions with a zero at each end and the anchor (+T*s0), every interior point
        # (T*(s_i - s_{i-1})) and the insertion (-T*s_{n-2}) all fall out of the difference.
        z = torch.zeros(K, self.C, 1, 3, device=p.device, dtype=p.dtype)
        s_ext = torch.cat([z, s, z], dim=2)                             # [K, C, N+1, 3]
        T = tension.clamp_min(0.0).unsqueeze(-1).unsqueeze(-1)
        F = T * (s_ext[:, :, 1:] - s_ext[:, :, :-1])                    # [K, C, N, 3]

        com = link_pose[..., 4:7].index_select(1, self.owner_flat).view(K, self.C, self.N, 3)
        tau = torch.linalg.cross(p - com, F, dim=-1)

        force = torch.zeros(K, self.L, 3, device=p.device, dtype=p.dtype)
        torque = torch.zeros_like(force)
        force.index_add_(1, self.owner_flat, F.reshape(K, -1, 3))
        torque.index_add_(1, self.owner_flat, tau.reshape(K, -1, 3))
        return force, torque, length
