"""The propeller half of the sea-drone co-design pilot, differentiable in Warp.

WHAT THIS IS. `python/examples/warp_prop_vortex.py` computes a Wageningen
B5-75 screw on the host in numpy, once per frame, to draw a wake. This module
computes the same chain on the device, over a BATCH of designs, on a Warp tape,
so that a gradient of an endurance objective with respect to (D, P/D, shaft
depth) exists. Same polynomials, same validity box, same Burrill criterion,
same actuator-disc momentum balance. The only thing added here is the
self-propulsion solve -- the demo is told its rps by a slider, a design is not:
its rps is whatever makes the screw's thrust equal the hull's resistance.

WHY THE KERNELS ARE SPLIT THE WAY THEY ARE. Warp 1.16 gets a silently wrong
adjoint for a variable accumulated in a kernel loop (`tot += ...`) that then
feeds a NONLINEAR op inside the same kernel. The polynomial sums are exactly
that kind of accumulator, so they leave `k_poly` through an array store and
every nonlinear step downstream -- the fade multiply, the thrust products, the
efficiency division, the Burrill power, the Newton update's division -- is a
separate kernel that READS that array. One launch per stage, one launch per
Newton iteration. `optimize.py --fd` is what decides whether that is enough;
nothing here is trusted until that table is green.

WHY float64. The finite-difference gate compares a central difference at
h = 1e-4 relative against the tape, to 1 %. In float32 the difference of two
nearly equal forwards costs three digits and the gate would be measuring
roundoff. The whole chain is float64; only the Adam parameter vector is
float32, because `warp.optim.Adam` supports float16/float32/vec3 and nothing
wider, and three normalised numbers near 1.0 do not need more.
"""

import math

import numpy as np
import warp as wp

F = wp.float64

# ── THE POLYNOMIALS ────────────────────────────────────────────────────────
# PROVENANCE: copied verbatim from python/examples/warp_prop_vortex.py
# (`_KT_TERMS`, `_KQ_TERMS`, around lines 976-1033), which states them as the
# Wageningen B-series open-water regression fitted at Rn = 2e6, a page of
# public-domain numbers. Each tuple is (coef, s, t, u, v) for
#     coef * J^s * (P/D)^t * (A_E/A_0)^u * Z^v.
# TRANSCRIPTION IS THE WHOLE RISK, and a copy is a second transcription, so
# this copy is not trusted either: `optimize.py --check-tables` parses the
# demo's source with `ast` (it is NEVER imported -- that module opens a window
# at import) and asserts these tuples are identical to its own.
KT_TERMS = (
    (+0.00880496, 0, 0, 0, 0), (-0.204554, 1, 0, 0, 0),
    (+0.166351, 0, 1, 0, 0), (+0.158114, 0, 2, 0, 0),
    (-0.147581, 2, 0, 1, 0), (-0.481497, 1, 1, 1, 0),
    (+0.415437, 0, 2, 1, 0), (+0.0144043, 0, 0, 0, 1),
    (-0.0530054, 2, 0, 0, 1), (+0.0143481, 0, 1, 0, 1),
    (+0.0606826, 1, 1, 0, 1), (-0.0125894, 0, 0, 1, 1),
    (+0.0109689, 1, 0, 1, 1), (-0.133698, 0, 3, 0, 0),
    (+0.00638407, 0, 6, 0, 0), (-0.00132718, 2, 6, 0, 0),
    (+0.168496, 3, 0, 1, 0), (-0.0507214, 0, 0, 2, 0),
    (+0.0854559, 2, 0, 2, 0), (-0.0504475, 3, 0, 2, 0),
    (+0.010465, 1, 6, 2, 0), (-0.00648272, 2, 6, 2, 0),
    (-0.00841728, 0, 3, 0, 1), (+0.0168424, 1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1), (-0.0317791, 0, 3, 1, 1),
    (+0.018604, 1, 0, 2, 1), (-0.00410798, 0, 2, 2, 1),
    (-0.000606848, 0, 0, 0, 2), (-0.0049819, 1, 0, 0, 2),
    (+0.0025983, 2, 0, 0, 2), (-0.000560528, 3, 0, 0, 2),
    (-0.00163652, 1, 2, 0, 2), (-0.000328787, 1, 6, 0, 2),
    (+0.000116502, 2, 6, 0, 2), (+0.000690904, 0, 0, 1, 2),
    (+0.00421749, 0, 3, 1, 2), (+0.0000565229, 3, 6, 1, 2),
    (-0.00146564, 0, 3, 2, 2),
)
KQ_TERMS = (
    (+0.00379368, 0, 0, 0, 0), (+0.00886523, 2, 0, 0, 0),
    (-0.032241, 1, 1, 0, 0), (+0.00344778, 0, 2, 0, 0),
    (-0.0408811, 0, 1, 1, 0), (-0.108009, 1, 1, 1, 0),
    (-0.0885381, 2, 1, 1, 0), (+0.188561, 0, 2, 1, 0),
    (-0.00370871, 1, 0, 0, 1), (+0.00513696, 0, 1, 0, 1),
    (+0.0209449, 1, 1, 0, 1), (+0.00474319, 2, 1, 0, 1),
    (-0.00723408, 2, 0, 1, 1), (+0.00438388, 1, 1, 1, 1),
    (-0.0269403, 0, 2, 1, 1), (+0.0558082, 3, 0, 1, 0),
    (+0.0161886, 0, 3, 1, 0), (+0.00318086, 1, 3, 1, 0),
    (+0.015896, 0, 0, 2, 0), (+0.0471729, 1, 0, 2, 0),
    (+0.0196283, 3, 0, 2, 0), (-0.0502782, 0, 1, 2, 0),
    (-0.030055, 3, 1, 2, 0), (+0.0417122, 2, 2, 2, 0),
    (-0.0397722, 0, 3, 2, 0), (-0.00350024, 0, 6, 2, 0),
    (-0.0106854, 3, 0, 0, 1), (+0.00110903, 3, 3, 0, 1),
    (-0.000313912, 0, 6, 0, 1), (+0.0035985, 3, 0, 1, 1),
    (-0.00142121, 0, 6, 1, 1), (-0.00383637, 1, 0, 2, 1),
    (+0.0126803, 0, 2, 2, 1), (-0.00318278, 2, 3, 2, 1),
    (+0.00334268, 0, 6, 2, 1), (-0.00183491, 1, 1, 0, 2),
    (+0.000112451, 3, 2, 0, 2), (-0.0000297228, 3, 6, 0, 2),
    (+0.000269551, 1, 0, 1, 2), (+0.00083265, 2, 0, 1, 2),
    (+0.00155334, 0, 2, 1, 2), (+0.000302683, 0, 6, 1, 2),
    (-0.0001843, 0, 0, 2, 2), (-0.000425399, 0, 3, 2, 2),
    (+0.0000869243, 3, 3, 2, 2), (-0.0004659, 0, 6, 2, 2),
    (+0.0000554194, 1, 6, 2, 2),
)

# ── CONSTANTS, ALL OF THEM THE DEMO'S OWN ──────────────────────────────────
EAR = 0.75                  # expanded area ratio A_E/A_0; the fold is for THIS
BLADES = 5                  # Z; the fold is for THIS
RHO = 1025.0                # kg/m3, seawater
P_ATM = 101325.0            # Pa at the surface
P_VAP = 2340.0              # Pa, vapour pressure of water at 20 C
G_ACC = 9.81
PD_LO, PD_HI = 0.5, 1.4     # the regression's validity box in pitch ratio
J_LO, J_HI = 0.0, 1.5       # ... and in advance ratio
BETA_0 = 2.0                # deg, the no-lift blade angle at 0.7 R
BURRILL_5PCT = 0.30         # tau_c = 0.30 sigma^0.57, Burrill's 5% back line
BURRILL_P = 0.57
BURRILL_INC = 0.610         # x that line = inception
NU_WATER = 1.19e-6          # m2/s, kinematic viscosity for the ITTC line
PI = math.pi
TWO_PI = 2.0 * math.pi
# Composite constants are folded HERE and not in a kernel: Warp types each
# module-level float it inlines on its own, so `wp.float64(RHO * G_ACC)` in
# kernel source becomes a float64 times a float32 and NVRTC cannot pick an
# overload. Every constant a kernel names is one finished number.
RHO_G = RHO * G_ACC

NEWTON_ITERS = 8            # unrolled, one launch per iteration
D_MIN = 0.15                # m, the small end of the grid and of the box


def pd_of(beta_deg):
    """P/D from the blade angle at 0.7 R. The blade IS a screw thread."""
    return 0.7 * math.pi * math.tan(math.radians(max(beta_deg, 0.0)))


PD_ZERO = pd_of(BETA_0)     # 0.0768: where K_T is faded to zero below the box
PD_FADE_DEN = PD_LO - PD_ZERO


def fold(terms):
    """(C, s, t, u, v) -> (C * EAR^u * Z^v, s, t), EAR and Z being fixed."""
    return tuple((c * EAR ** u * float(BLADES) ** v, s, t)
                 for c, s, t, u, v in terms)


KT_FOLDED = fold(KT_TERMS)
KQ_FOLDED = fold(KQ_TERMS)

# The kernel selects powers with an if-chain rather than calling pow, because
# pow(0, 0) has no usable adjoint and J_LO is exactly zero. That is only safe
# if the exponents really are the handful the chain covers, so: assert it.
_S_OK, _T_OK = {0, 1, 2, 3}, {0, 1, 2, 3, 6}
for _tab, _nm in ((KT_FOLDED, "K_T"), (KQ_FOLDED, "K_Q")):
    for _c, _s, _t in _tab:
        if _s not in _S_OK or _t not in _T_OK:
            raise RuntimeError(f"{_nm} term has exponents ({_s},{_t}) outside "
                               f"the set k_poly's if-chain covers")


def poly_np(folded, j, pd):
    """The host reference, for --check-tables. Vectorised over j and pd."""
    j = np.asarray(j, dtype=np.float64)
    pd = np.asarray(pd, dtype=np.float64)
    tot = np.zeros(np.broadcast(j, pd).shape, dtype=np.float64)
    for c, s, t in folded:
        tot = tot + c * j ** s * pd ** t
    return tot


def open_water_np(j, pd):
    """K_T, K_Q with the demo's clamp and its below-box fade, on the host."""
    jc = np.clip(j, J_LO, J_HI)
    pdc = np.clip(pd, PD_LO, PD_HI)
    kt = poly_np(KT_FOLDED, jc, pdc)
    kq = poly_np(KQ_FOLDED, jc, pdc)
    fade = np.where(np.asarray(pd) < PD_LO,
                    np.maximum((np.asarray(pd) - PD_ZERO)
                               / (PD_LO - PD_ZERO), 0.0), 1.0)
    return kt * fade, kq


# ── THE BRIEF ──────────────────────────────────────────────────────────────
# A 3 m survey drone. C_R is a RESIDUARY PLACEHOLDER: a flat 0.0020 stands in
# for the wave-making the hull sculpt will contribute as a Michell integral
# once the hull joins this graph. Until it does, the hull is fixed and the
# only thing gradient descent is allowed to move is the screw.
# C_R is a residuary placeholder until the hull sculpt's Michell term joins;
# 0.006 is a short hull at Fn ~0.37, not a slender one. The brief was chosen
# (grid sweep, 2026-09-14) so that the cavitation constraint BINDS: the step-0
# guess cavitates at sprint (Burrill 1.34), and the optimum sits at Burrill 1
# with the diameter on the clearance bound and the pitch and depth interior.
# With the earlier 4 m/s sprint / 0.40 m / C_R 0.002 nothing bound and the
# optimiser only ran to the box corner (D_max, P/D 1.4).
DEFAULT_HULL = dict(L=3.0, S_wet=3.2, k_form=0.25, C_R=0.0060)
DEFAULT_BRIEF = dict(
    V_survey=2.0, V_sprint=7.0, w=0.15, t=0.12,
    E_batt=5.0,                 # kWh
    n_max=50.0,                 # rps: the motor's limit, checked at SPRINT
                                # (the highest shaft speed; 3000 rpm)
    D_max=0.25,                 # m, tip clearance under a ~0.35 m draft
    depth_min=0.15, depth_max=0.30,     # m, shaft centre below the surface
    # Soft hinges: the penalised optimum overshoots a binding constraint by
    # ~1/lam (Burrill 1.0135 at lam 10, measured), so the weights are stiff.
    lam_cav=100.0, lam_n=100.0, lam_box=100.0,
    hull=dict(DEFAULT_HULL),
)


def resistance(v, brief):
    """R_T(V) = 0.5 rho V^2 S_wet (C_F(1+k) + C_R), ITTC-57 friction line."""
    hull = brief["hull"]
    re = max(abs(v) * hull["L"] / NU_WATER, 1.0e3)
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    ct = cf * (1.0 + hull["k_form"]) + hull["C_R"]
    return 0.5 * RHO * v * v * hull["S_wet"] * ct


# ── KERNELS ────────────────────────────────────────────────────────────────

@wp.kernel
def k_poly(jj: wp.array(dtype=F), pd: wp.array(dtype=F),
           coef: wp.array(dtype=F), sx: wp.array(dtype=wp.int32),
           tx: wp.array(dtype=wp.int32), nterm: wp.int32,
           out: wp.array(dtype=F), dout: wp.array(dtype=F)):
    """The regression and its dK/dJ, both as LOOP ACCUMULATORS that leave this
    kernel through an array store and are touched by nothing nonlinear here.
    J and P/D are clamped to the validity box on the way in (R5 in the demo:
    outside the box a polynomial does not degrade, it diverges)."""
    i = wp.tid()
    j1 = wp.clamp(jj[i], wp.float64(J_LO), wp.float64(J_HI))
    p1 = wp.clamp(pd[i], wp.float64(PD_LO), wp.float64(PD_HI))
    j2 = j1 * j1
    j3 = j2 * j1
    p2 = p1 * p1
    p3 = p2 * p1
    p6 = p3 * p3
    tot = wp.float64(0.0)
    dtot = wp.float64(0.0)
    for k in range(nterm):
        s = sx[k]
        t = tx[k]
        jp = wp.float64(1.0)
        jm = wp.float64(0.0)          # j^(s-1), zero for s = 0
        if s == 1:
            jp = j1
            jm = wp.float64(1.0)
        elif s == 2:
            jp = j2
            jm = j1
        elif s == 3:
            jp = j3
            jm = j2
        pp = wp.float64(1.0)
        if t == 1:
            pp = p1
        elif t == 2:
            pp = p2
        elif t == 3:
            pp = p3
        elif t == 6:
            pp = p6
        c = coef[k]
        tot += c * jp * pp
        dtot += c * wp.float64(s) * jm * pp
    out[i] = tot
    dout[i] = dtot


@wp.kernel
def k_fade(pd: wp.array(dtype=F), kt_in: wp.array(dtype=F),
           dkt_in: wp.array(dtype=F), kt: wp.array(dtype=F),
           dkt: wp.array(dtype=F)):
    """Below the box, fade K_T linearly to zero at the no-lift pitch. The
    demo does this inline; here it is its own launch because it MULTIPLIES the
    polynomial accumulator, and that product is exactly the shape of adjoint
    Warp 1.16 gets wrong when it happens in the accumulating kernel."""
    i = wp.tid()
    p = pd[i]
    f = wp.float64(1.0)
    if p < wp.float64(PD_LO):
        f = wp.max((p - wp.float64(PD_ZERO))
                   / wp.float64(PD_FADE_DEN), wp.float64(0.0))
    kt[i] = kt_in[i] * f
    dkt[i] = dkt_in[i] * f


@wp.kernel
def k_n_init(design: wp.array2d(dtype=F), va: F, n0: wp.array(dtype=F)):
    """The Newton seed: the rps that would put this screw at J = 0.75, which
    is inside the box and within a factor of two of every self-propulsion
    point this brief can reach."""
    i = wp.tid()
    n0[i] = wp.clamp(va / (wp.float64(0.75) * design[i, 0]),
                     wp.float64(0.5), wp.float64(400.0))


@wp.kernel
def k_advance(n: wp.array(dtype=F), design: wp.array2d(dtype=F), va: F,
              jj: wp.array(dtype=F)):
    """J = V_a / (n D). A division, so it is its own launch."""
    i = wp.tid()
    jj[i] = va / (wp.max(n[i], wp.float64(1.0e-4)) * design[i, 0])


@wp.kernel
def k_pd(design: wp.array2d(dtype=F), pd: wp.array(dtype=F)):
    i = wp.tid()
    pd[i] = design[i, 1]


@wp.kernel
def k_newton(n_in: wp.array(dtype=F), kt: wp.array(dtype=F),
             dkt: wp.array(dtype=F), jj: wp.array(dtype=F),
             design: wp.array2d(dtype=F), rt: F, t_ded: F,
             n_out: wp.array(dtype=F)):
    """ONE self-propulsion iteration, one launch.

        r(n)  = K_T(J(n)) rho n^2 D^4 (1 - t) - R_T
        r'(n) = rho D^4 (1 - t) n (2 K_T - J dK_T/dJ)      [dJ/dn = -J/n]

    K_T and dK_T/dJ arrive through arrays that k_poly wrote, so the only thing
    this kernel does with a loop accumulator is read it. The step is limited to
    half the current n, which cannot move the fixed point (the step is zero
    there) but keeps a wild cell of the brute-force grid from stepping n
    negative."""
    i = wp.tid()
    n = n_in[i]
    d = design[i, 0]
    d2 = d * d
    fac = wp.float64(RHO) * d2 * d2 * (wp.float64(1.0) - t_ded)
    r = kt[i] * fac * n * n - rt
    rp = fac * n * (wp.float64(2.0) * kt[i] - jj[i] * dkt[i])
    if wp.abs(rp) < wp.float64(1.0e-9):
        rp = wp.float64(1.0e-9)
    step = wp.clamp(r / rp, wp.float64(-0.5) * n, wp.float64(0.5) * n)
    n_out[i] = wp.clamp(n - step, wp.float64(0.05), wp.float64(400.0))


@wp.kernel
def k_residual(n: wp.array(dtype=F), kt: wp.array(dtype=F),
               design: wp.array2d(dtype=F), rt: F, t_ded: F,
               res: wp.array(dtype=F)):
    """|K_T rho n^2 D^4 (1-t) - R_T| / R_T, the convergence diagnostic."""
    i = wp.tid()
    d = design[i, 0]
    d2 = d * d
    fac = wp.float64(RHO) * d2 * d2 * (wp.float64(1.0) - t_ded)
    res[i] = wp.abs(kt[i] * fac * n[i] * n[i] - rt) / wp.abs(rt)


@wp.kernel
def k_thrust(kt: wp.array(dtype=F), kq: wp.array(dtype=F),
             jj: wp.array(dtype=F), n: wp.array(dtype=F),
             design: wp.array2d(dtype=F),
             thrust: wp.array(dtype=F), torque: wp.array(dtype=F),
             power: wp.array(dtype=F), eta: wp.array(dtype=F)):
    """T = K_T rho n^2 D^4, Q = K_Q rho n^2 D^5, P_D = 2 pi n Q,
    eta_0 = J K_T / (2 pi K_Q). The demo's PropState, one row per design."""
    i = wp.tid()
    d = design[i, 0]
    d2 = d * d
    rn2d4 = wp.float64(RHO) * n[i] * n[i] * d2 * d2
    q = kq[i] * rn2d4 * d
    thrust[i] = kt[i] * rn2d4
    torque[i] = q
    power[i] = wp.float64(TWO_PI) * n[i] * q
    e = wp.float64(0.0)
    if kq[i] > wp.float64(1.0e-9) and kt[i] > wp.float64(0.0):
        e = jj[i] * kt[i] / (wp.float64(TWO_PI) * kq[i])
    eta[i] = e


@wp.kernel
def k_burrill(thrust: wp.array(dtype=F), n: wp.array(dtype=F),
              design: wp.array2d(dtype=F), va: F,
              sigma: wp.array(dtype=F), tauc: wp.array(dtype=F),
              ratio: wp.array(dtype=F)):
    """Burrill at 0.7 R, the demo's own criterion with the shaft depth as a
    design variable rather than a CLI knob: sigma = 2 (P_ATM + rho g h -
    P_VAP)/rho / V_R^2, V_R^2 = V_a^2 + (0.7 omega R)^2, tau_c = T / (0.5 rho
    A_P V_R^2) with Taylor's A_P = EAR A_disc (1.067 - 0.229 P/D), and the 5 %
    back-cavitation line 0.30 sigma^0.57. ratio > 1 is cavitating."""
    i = wp.tid()
    d = design[i, 0]
    pd = wp.clamp(design[i, 1], wp.float64(PD_LO), wp.float64(PD_HI))
    h = design[i, 2]
    r_tip = wp.float64(0.5) * d
    om = wp.float64(TWO_PI) * n[i]
    vr2 = va * va + (wp.float64(0.7) * om * r_tip) * (wp.float64(0.7) * om
                                                      * r_tip)
    vr2 = wp.max(vr2, wp.float64(1.0e-6))
    margin = (wp.float64(P_ATM) + wp.float64(RHO_G) * h
              - wp.float64(P_VAP)) / wp.float64(RHO)
    sig = wp.float64(2.0) * margin / vr2
    a_disc = wp.float64(PI) * r_tip * r_tip
    a_p = wp.float64(EAR) * a_disc * (wp.float64(1.067)
                                      - wp.float64(0.229) * pd)
    tc = thrust[i] / (wp.float64(0.5) * wp.float64(RHO) * a_p * vr2)
    line5 = wp.float64(BURRILL_5PCT) * wp.pow(sig, wp.float64(BURRILL_P))
    sigma[i] = sig
    tauc[i] = tc
    ratio[i] = wp.max(tc, wp.float64(0.0)) / (wp.float64(BURRILL_INC) * line5)


@wp.kernel
def k_objective(design: wp.array2d(dtype=F), p_survey: wp.array(dtype=F),
                n_sprint: wp.array(dtype=F), ratio_sprint: wp.array(dtype=F),
                range_ref: wp.array(dtype=F),
                v_survey: F, e_batt_j: F, n_max: F, d_max: F,
                depth_min: F, depth_max: F,
                lam_cav: F, lam_n: F, lam_box: F,
                range_km: wp.array(dtype=F), loss_row: wp.array(dtype=F)):
    """range = V_survey E_batt / P_D(V_survey), and everything that is not
    allowed to be bought with it: cavitation at the sprint, shaft speed at the
    sprint (the motor's limit is met at the highest speed), and the box. Hinges are squared, so a constraint that is satisfied
    contributes neither value nor gradient."""
    i = wp.tid()
    rng = v_survey * e_batt_j / wp.max(p_survey[i], wp.float64(1.0e-6)) \
        / wp.float64(1000.0)
    range_km[i] = rng
    obj = -rng / range_ref[i]
    hc = wp.max(ratio_sprint[i] - wp.float64(1.0), wp.float64(0.0))
    hn = wp.max(n_sprint[i] / n_max - wp.float64(1.0), wp.float64(0.0))
    obj += lam_cav * hc * hc + lam_n * hn * hn
    d = design[i, 0]
    pd = design[i, 1]
    h = design[i, 2]
    bd = wp.max(d / d_max - wp.float64(1.0), wp.float64(0.0))
    bd2 = wp.max(wp.float64(D_MIN) / d_max - d / d_max, wp.float64(0.0))
    bp = wp.max(pd - wp.float64(PD_HI), wp.float64(0.0))
    bp2 = wp.max(wp.float64(PD_LO) - pd, wp.float64(0.0))
    bh = wp.max((h - depth_max) / depth_max, wp.float64(0.0))
    bh2 = wp.max((depth_min - h) / depth_max, wp.float64(0.0))
    obj += lam_box * (bd * bd + bd2 * bd2 + bp * bp + bp2 * bp2
                      + bh * bh + bh2 * bh2)
    loss_row[i] = obj


@wp.kernel
def k_reduce(loss_row: wp.array(dtype=F), loss: wp.array(dtype=F)):
    i = wp.tid()
    wp.atomic_add(loss, 0, loss_row[i])


# ── HOST SIDE ──────────────────────────────────────────────────────────────

class _Tables:
    """The folded polynomials on the device, built once per device."""

    _cache = {}

    def __new__(cls, device):
        key = str(device)
        hit = cls._cache.get(key)
        if hit is None:
            hit = super().__new__(cls)
            c, s, t = zip(*KT_FOLDED)
            hit.kt_c = wp.array(np.array(c, dtype=np.float64), dtype=F,
                                device=device)
            hit.kt_s = wp.array(np.array(s, dtype=np.int32),
                                dtype=wp.int32, device=device)
            hit.kt_t = wp.array(np.array(t, dtype=np.int32),
                                dtype=wp.int32, device=device)
            hit.kt_n = len(KT_FOLDED)
            c, s, t = zip(*KQ_FOLDED)
            hit.kq_c = wp.array(np.array(c, dtype=np.float64), dtype=F,
                                device=device)
            hit.kq_s = wp.array(np.array(s, dtype=np.int32),
                                dtype=wp.int32, device=device)
            hit.kq_t = wp.array(np.array(t, dtype=np.int32),
                                dtype=wp.int32, device=device)
            hit.kq_n = len(KQ_FOLDED)
            cls._cache[key] = hit
        return hit


class _OpBuffers:
    """Every array one operating point needs, including one slot per Newton
    iteration -- n is never written in place, because an adjoint cannot be
    replayed through a buffer that has been overwritten."""

    def __init__(self, n_rows, device, requires_grad):
        def mk():
            return wp.zeros(n_rows, dtype=F, device=device,
                            requires_grad=requires_grad)
        m = NEWTON_ITERS + 1
        self.n = [mk() for _ in range(m)]
        self.j = [mk() for _ in range(m)]
        self.kt_raw = [mk() for _ in range(m)]
        self.dkt_raw = [mk() for _ in range(m)]
        self.kt = [mk() for _ in range(m)]
        self.dkt = [mk() for _ in range(m)]
        self.kq_raw, self.dkq_raw = mk(), mk()
        self.kq, self.dkq = mk(), mk()
        self.thrust, self.torque, self.power, self.eta = mk(), mk(), mk(), mk()
        self.sigma, self.tauc, self.ratio = mk(), mk(), mk()
        self.res = mk()


class Buffers:
    """The whole chain's working set for a batch of `n_rows` designs."""

    def __init__(self, n_rows, device="cuda:0", requires_grad=True):
        self.n_rows = n_rows
        self.device = device
        self.requires_grad = requires_grad
        self.tab = _Tables(device)
        self.design = wp.zeros((n_rows, 3), dtype=F, device=device,
                               requires_grad=requires_grad)
        self.pd = wp.zeros(n_rows, dtype=F, device=device,
                           requires_grad=requires_grad)
        self.range_ref = wp.full(n_rows, 1.0, dtype=F, device=device)
        self.range_km = wp.zeros(n_rows, dtype=F, device=device,
                                 requires_grad=requires_grad)
        self.loss_row = wp.zeros(n_rows, dtype=F, device=device,
                                 requires_grad=requires_grad)
        self.loss = wp.zeros(1, dtype=F, device=device,
                             requires_grad=requires_grad)
        self.survey = _OpBuffers(n_rows, device, requires_grad)
        self.sprint = _OpBuffers(n_rows, device, requires_grad)

    def set_design(self, rows):
        """rows: (n_rows, 3) of (D, P/D, depth)."""
        self.design.assign(np.ascontiguousarray(rows, dtype=np.float64))


def poly_gpu(j, pd, device="cuda:0"):
    """K_T (faded) and K_Q from the kernels, for the --check-tables gate."""
    j = np.ascontiguousarray(j, dtype=np.float64)
    pd = np.ascontiguousarray(pd, dtype=np.float64)
    n = j.size
    tab = _Tables(device)
    ja = wp.array(j, dtype=F, device=device)
    pa = wp.array(pd, dtype=F, device=device)
    out = [wp.zeros(n, dtype=F, device=device) for _ in range(6)]
    wp.launch(k_poly, dim=n, device=device,
              inputs=[ja, pa, tab.kt_c, tab.kt_s, tab.kt_t, tab.kt_n],
              outputs=[out[0], out[1]])
    wp.launch(k_fade, dim=n, device=device, inputs=[pa, out[0], out[1]],
              outputs=[out[2], out[3]])
    wp.launch(k_poly, dim=n, device=device,
              inputs=[ja, pa, tab.kq_c, tab.kq_s, tab.kq_t, tab.kq_n],
              outputs=[out[4], out[5]])
    return out[2].numpy().copy(), out[4].numpy().copy()


def _poly_pair(buf, jj, pd, op, k):
    """K_T (with its below-box fade) and dK_T/dJ at slot k."""
    tab, dev, n = buf.tab, buf.device, buf.n_rows
    wp.launch(k_poly, dim=n, device=dev,
              inputs=[jj, pd, tab.kt_c, tab.kt_s, tab.kt_t, tab.kt_n],
              outputs=[op.kt_raw[k], op.dkt_raw[k]])
    wp.launch(k_fade, dim=n, device=dev,
              inputs=[pd, op.kt_raw[k], op.dkt_raw[k]],
              outputs=[op.kt[k], op.dkt[k]])


def _solve_op(buf, op, v_ship, brief):
    """The self-propulsion solve at one ship speed, then the full state."""
    dev, n = buf.device, buf.n_rows
    va = v_ship * (1.0 - brief["w"])
    rt = resistance(v_ship, brief)
    t_ded = brief["t"]
    wp.launch(k_n_init, dim=n, device=dev,
              inputs=[buf.design, va], outputs=[op.n[0]])
    for k in range(NEWTON_ITERS):
        wp.launch(k_advance, dim=n, device=dev,
                  inputs=[op.n[k], buf.design, va], outputs=[op.j[k]])
        _poly_pair(buf, op.j[k], buf.pd, op, k)
        wp.launch(k_newton, dim=n, device=dev,
                  inputs=[op.n[k], op.kt[k], op.dkt[k], op.j[k], buf.design,
                          rt, t_ded],
                  outputs=[op.n[k + 1]])
    k = NEWTON_ITERS
    wp.launch(k_advance, dim=n, device=dev,
              inputs=[op.n[k], buf.design, va], outputs=[op.j[k]])
    _poly_pair(buf, op.j[k], buf.pd, op, k)
    tab = buf.tab
    wp.launch(k_poly, dim=n, device=dev,
              inputs=[op.j[k], buf.pd, tab.kq_c, tab.kq_s, tab.kq_t, tab.kq_n],
              outputs=[op.kq_raw, op.dkq_raw])
    # K_Q is NOT faded below the box (the demo's choice: an overestimate keeps
    # a feathered shaft absorbing power, which is the conservative direction).
    wp.launch(k_thrust, dim=n, device=dev,
              inputs=[op.kt[k], op.kq_raw, op.j[k], op.n[k], buf.design],
              outputs=[op.thrust, op.torque, op.power, op.eta])
    wp.launch(k_burrill, dim=n, device=dev,
              inputs=[op.thrust, op.n[k], buf.design, va],
              outputs=[op.sigma, op.tauc, op.ratio])
    wp.launch(k_residual, dim=n, device=dev,
              inputs=[op.n[k], op.kt[k], buf.design, rt, t_ded],
              outputs=[op.res])
    op.kq = op.kq_raw
    return op


def chain(design, brief, buf=None, device="cuda:0", tape=None):
    """Launch the whole sequence for a batch of designs and return the buffers.

    `design` is (B, 3) of (D, P/D, shaft depth); `brief` is the dict above.
    If `tape` is given the launches are recorded inside it, so
    `tape.backward(loss=buf.loss)` gives d(sum of per-row objectives)/d(design)
    in `buf.design.grad` -- and the rows are independent, so row i's gradient
    is the gradient of row i's own objective.
    """
    rows = np.ascontiguousarray(design, dtype=np.float64).reshape(-1, 3)
    if buf is None:
        buf = Buffers(rows.shape[0], device=device,
                      requires_grad=tape is not None)
    buf.set_design(rows)
    if tape is not None:
        with tape:
            _chain_body(buf, brief)
    else:
        _chain_body(buf, brief)
    return buf


def _chain_body(buf, brief):
    dev, n = buf.device, buf.n_rows
    wp.launch(k_pd, dim=n, device=dev, inputs=[buf.design], outputs=[buf.pd])
    _solve_op(buf, buf.survey, brief["V_survey"], brief)
    _solve_op(buf, buf.sprint, brief["V_sprint"], brief)
    buf.loss.zero_()
    wp.launch(k_objective, dim=n, device=dev,
              inputs=[buf.design, buf.survey.power, buf.sprint.n[NEWTON_ITERS],
                      buf.sprint.ratio, buf.range_ref,
                      brief["V_survey"], brief["E_batt"] * 3.6e6,
                      brief["n_max"], brief["D_max"],
                      brief["depth_min"], brief["depth_max"],
                      brief["lam_cav"], brief["lam_n"], brief["lam_box"]],
              outputs=[buf.range_km, buf.loss_row])
    wp.launch(k_reduce, dim=n, device=dev,
              inputs=[buf.loss_row], outputs=[buf.loss])


def read_rows(buf):
    """Everything the reports and the JSON need, as numpy, one row per design."""
    k = NEWTON_ITERS
    return dict(
        D=buf.design.numpy()[:, 0].copy(),
        PD=buf.design.numpy()[:, 1].copy(),
        depth=buf.design.numpy()[:, 2].copy(),
        n_survey=buf.survey.n[k].numpy().copy(),
        n_sprint=buf.sprint.n[k].numpy().copy(),
        J_survey=buf.survey.j[k].numpy().copy(),
        J_sprint=buf.sprint.j[k].numpy().copy(),
        kt_survey=buf.survey.kt[k].numpy().copy(),
        kq_survey=buf.survey.kq.numpy().copy(),
        eta_survey=buf.survey.eta.numpy().copy(),
        eta_sprint=buf.sprint.eta.numpy().copy(),
        P_survey=buf.survey.power.numpy().copy(),
        P_sprint=buf.sprint.power.numpy().copy(),
        T_survey=buf.survey.thrust.numpy().copy(),
        T_sprint=buf.sprint.thrust.numpy().copy(),
        ratio_survey=buf.survey.ratio.numpy().copy(),
        ratio_sprint=buf.sprint.ratio.numpy().copy(),
        res_survey=buf.survey.res.numpy().copy(),
        res_sprint=buf.sprint.res.numpy().copy(),
        range_km=buf.range_km.numpy().copy(),
        loss=buf.loss_row.numpy().copy(),
    )


def binding(row, brief, tol=1.0e-3):
    """Which constraint is at its bound for one row of `read_rows`."""
    out = []
    if row["D"] >= brief["D_max"] * (1.0 - tol):
        out.append("D = D_max")
    if row["D"] <= D_MIN * (1.0 + tol):
        out.append("D = D_min")
    if row["ratio_sprint"] >= 1.0 - tol:
        out.append("Burrill(sprint) = 1")
    if row["n_sprint"] >= brief["n_max"] * (1.0 - tol):
        out.append("n_sprint = n_max")
    if row["PD"] >= PD_HI * (1.0 - tol):
        out.append("P/D = P/D_hi")
    if row["PD"] <= PD_LO * (1.0 + tol):
        out.append("P/D = P/D_lo")
    if row["depth"] >= brief["depth_max"] * (1.0 - tol):
        out.append("depth = depth_max")
    if row["depth"] <= brief["depth_min"] * (1.0 + tol):
        out.append("depth = depth_min")
    return out or ["interior"]
