"""The rotor downwash (tp.Downwash, renderer.downwash): one test per consumer.

The API and the CPU warp run on every build; the render checks need the Vulkan
backend and a GPU, and skip without them, as test_vulkan.py does.

* water: a flat MeshStandardMaterial with ripples = 1 under a hovering jet takes
  concentric rings on its G-buffer normal (read back as world normals): tilted
  along the radius, not across it, and flat with ripples = 0 or no downwash.
* grass: a ring of blades round the jet leans out from its axis (seen from the
  side, the blades' pixels spread from the centre column) and stands again when
  it leaves, though the field's distance gate has frozen it.
* particles: a Renderer-owned emitter's cloud is blown out of a hollow under
  the jet (its pixels leave the disc under the axis).
"""
import math

import numpy as np
import pytest

import threepp as tp

W, H = 320, 240


# ── API and the CPU profile ──────────────────────────────────────────────────

def test_momentum_theory_share():
    """v_i = sqrt(T / (2 rho pi R^2)); the reference rotor (an M350: 90 N through a 1 m
    disc) is share 1, and quadrupling the thrust doubles the share."""
    v = tp.Downwash.induced_velocity(90.0, 0.5)
    assert v == pytest.approx(math.sqrt(90.0 / (2 * 1.225 * math.pi * 0.25)), rel=1e-5)
    dw = tp.Downwash()
    jet = dw.add_jet(tp.Vector3(0, 2, 0))
    assert tp.Downwash.jet_share(jet) == pytest.approx(1.0, rel=1e-5)
    jet.thrust = 360.0
    assert tp.Downwash.jet_share(jet) == pytest.approx(2.0, rel=1e-5)
    jet.strength = 0.5
    assert tp.Downwash.jet_share(jet) == pytest.approx(0.5)


def test_sources_are_live_references_and_capped_at_eight():
    dw = tp.Downwash()
    jet = dw.add_jet(tp.Vector3(1, 2, 3), thrust=60.0, radius=0.4)
    rings = dw.add_rings(tp.Vector3(4, 0, 5), start_time=1.5, speed=0.5, count=3)
    assert len(dw) == 2
    assert dw[0].kind == tp.Downwash.Kind.Jet and dw[1].kind == tp.Downwash.Kind.Rings
    jet.center = tp.Vector3(7, 8, 9)
    assert dw[0].center.x == pytest.approx(7.0)
    assert rings.start_time == pytest.approx(1.5) and rings.count == 3
    for _ in range(6):
        dw.add_jet(tp.Vector3(0, 1, 0))
    with pytest.raises(Exception):
        dw.add_jet(tp.Vector3(0, 1, 0))
    dw.remove(0)
    assert len(dw) == 7 and dw[0].kind == tp.Downwash.Kind.Rings
    dw.clear()
    assert len(dw) == 0


def test_cpu_warp_blows_a_hollow_and_leaves_the_far_cloud():
    """displace() is the particle warp for host-owned positions: points under the jet
    leave a disc about hollow_radius wide, points far out stay where they were, and
    points above the rotor plane are not touched. The rim is a thinning band (each point
    takes its own share of the hollow), not a hard circle."""
    dw = tp.Downwash()
    jet = dw.add_jet(tp.Vector3(0, 1.5, 0))
    jet.surface_y = 0.0
    rng = np.random.default_rng(3)
    pts = np.zeros((4000, 4), np.float32)
    pts[:, 0] = rng.uniform(-8, 8, 4000)
    pts[:, 1] = rng.uniform(0.0, 1.0, 4000)
    pts[:, 2] = rng.uniform(-8, 8, 4000)
    before = pts.copy()
    dw.displace(pts)
    r0 = np.hypot(before[:, 0], before[:, 2])
    r1 = np.hypot(pts[:, 0], pts[:, 2])
    assert np.all(r1 >= r0 - 1e-5), "the jet never pulls a point inward"
    assert (r1 < 1.1).sum() == 0, "the disc under the jet is empty"
    # the rim thins out over 0.45..1 of the hollow (each point's own share), no hard cut
    assert 0 < (r1 < 2.0).sum() < 0.45 * (r0 < 2.0).sum()
    far = r0 > 12.0
    assert np.allclose(pts[far], before[far]), "the far cloud is untouched"
    assert np.allclose(pts[:, 3], before[:, 3]), "w is not touched"
    high = np.array([[0.2, 3.0, 0.1]], np.float32)
    dw.displace(high)
    assert np.allclose(high, [[0.2, 3.0, 0.1]]), "nothing above the rotor plane moves"
    with pytest.raises(TypeError):
        dw.displace(np.zeros((4, 3), np.float64))


def test_material_ripples_round_trip():
    m = tp.MeshStandardMaterial()
    assert m.ripples == 0.0
    m.ripples = 1.0
    assert m.ripples == 1.0


# ── Render checks ────────────────────────────────────────────────────────────

def _renderer():
    if not tp.HAS_VULKAN:
        pytest.skip("built without the Vulkan backend")
    canvas = tp.Canvas("downwash-test", width=W, height=H, headless=True, vsync=False)
    try:
        r = tp.VulkanRenderer(canvas)
    except Exception as e:  # no Vulkan GPU
        pytest.skip(f"no Vulkan renderer: {e}")
    r.starfield = 0.0
    return canvas, r


def _top_camera(height=6.0, fov=60):
    cam = tp.PerspectiveCamera(fov, W / H, 0.1, 100)
    cam.position.set(0, height, 0.001)
    cam.look_at(0, 0, 0)
    return cam


def _step(r, scene, cam, t, frames=3):
    for i in range(frames):
        r.sim_time = t + i / 60.0
        r.render(scene, cam)


def test_water_rings_on_the_gbuffer_normal():
    canvas, r = _renderer()
    scene = tp.Scene()
    scene.add(tp.DirectionalLight(tp.Color(1, 1, 1), 2.0))
    water = tp.MeshStandardMaterial()
    water.color = tp.Color(0.02, 0.03, 0.03)
    water.roughness = 0.04
    plane = tp.Mesh(tp.PlaneGeometry(12, 12), water)
    plane.rotation.x = -math.pi / 2
    scene.add(plane)
    cam = _top_camera(4.0)
    dw = tp.Downwash()
    dw.add_jet(tp.Vector3(0, 1.5, 0))

    def normals():
        _step(r, scene, cam, 2.0)
        return r.read_normals_float(scene, cam)

    flat = normals()                       # no downwash
    r.downwash = dw
    off = normals()                        # a downwash, ripples 0
    water.ripples = 1.0
    water.needs_update()                   # a material edit after its first render
    on = normals()
    r.downwash = None
    back = normals()

    def tilt(n):
        return np.hypot(n[..., 0], n[..., 2])

    assert tilt(flat).max() < 0.02 and tilt(off).max() < 0.02 and tilt(back).max() < 0.02
    # the rings: a band round the axis between the jet's peak and a few metres out
    ys, xs = np.mgrid[0:on.shape[0], 0:on.shape[1]]
    cx, cy = on.shape[1] / 2, on.shape[0] / 2
    rr = np.hypot(xs - cx, ys - cy)
    band = (rr > 0.10 * on.shape[0]) & (rr < 0.45 * on.shape[0])
    t = tilt(on)[band]
    assert np.percentile(t, 90) > 0.05, f"the rings tilt the normal (90th pct {np.percentile(t, 90):.3f})"
    # along the radius, not across it: the tilt's radial share dominates
    ux, uy = (xs - cx) / np.maximum(rr, 1), (ys - cy) / np.maximum(rr, 1)
    # image y runs down = world +z for this camera (looking down -y, up = -z)
    radial = np.abs(on[..., 0] * ux + on[..., 2] * uy)[band]
    strong = t > 0.03
    assert strong.sum() > 100
    assert np.median(radial[strong] / t[strong]) > 0.9, "the rings are concentric"


def _ring_of_blades(n_blades=240, r_in=0.4, r_out=3.0, h=0.6, w=0.03):
    rng = np.random.default_rng(1)
    pos, hf, idx = [], [], []
    for i in range(n_blades):
        a = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(rng.uniform(r_in ** 2, r_out ** 2))
        x, z = rr * math.cos(a), rr * math.sin(a)
        # a blade facing the axis, two triangles in a vertical plane across the radius
        tx, tz = -math.sin(a) * w, math.cos(a) * w
        b = len(pos)
        for s in range(5):
            f = s / 4
            pos += [(x - tx, f * h, z - tz), (x + tx, f * h, z + tz)]
            hf += [f, f]
        for s in range(4):
            a0 = b + 2 * s
            idx += [a0, a0 + 1, a0 + 3, a0, a0 + 3, a0 + 2]
    g = tp.BufferGeometry()
    g.set_attribute("position", np.array(pos, np.float32).reshape(-1, 3))
    g.set_attribute("heightFrac", np.array(hf, np.float32).reshape(-1, 1))
    g.set_index(np.array(idx, np.uint32))
    g.compute_vertex_normals()
    return g


def test_grass_leans_out_from_the_jet_and_stands_again():
    canvas, r = _renderer()
    scene = tp.Scene()
    scene.add(tp.DirectionalLight(tp.Color(1, 1, 1), 2.0))
    scene.add(tp.AmbientLight(tp.Color(1, 1, 1), 0.5))
    mat = tp.MeshStandardMaterial()
    mat.color = tp.Color(0.2, 0.5, 0.1)
    mat.side = tp.Side.Double
    grass = tp.GrassMesh(_ring_of_blades(), mat)
    grass.wind_strength = 0.0
    grass.max_anim_distance = 0.01    # frozen by the gate: only a jet near it wakes it
    scene.add(grass)
    # from the side: blades leaning out from the axis spread away from the centre column
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cam.position.set(0, 0.5, 6.0)
    cam.look_at(0, 0.15, 0)

    def spread():
        _step(r, scene, cam, 1.0)
        ids = r.read_instance_ids(scene, cam)
        ys, xs = np.nonzero(ids)
        return float(np.abs(xs + 0.5 - ids.shape[1] / 2).mean()), len(xs)

    still, n0 = spread()
    dw = tp.Downwash()
    dw.add_jet(tp.Vector3(0, 1.2, 0))
    r.downwash = dw
    bent, n1 = spread()
    dw.clear()
    back, n2 = spread()
    assert n0 > 500 and n1 > 500
    assert bent > still * 1.03, f"the blades lean out ({still:.1f} -> {bent:.1f} px)"
    assert back == pytest.approx(still, rel=0.02), "and stand again once the jet is gone"


def test_particles_are_blown_out_of_a_hollow():
    canvas, r = _renderer()
    scene = tp.Scene()
    scene.add(tp.DirectionalLight(tp.Color(1, 1, 1), 2.0))
    cfg = tp.ParticleField.Config()
    cfg.capacity = 6000
    cfg.ownership = tp.ParticleField.Ownership.Renderer
    cfg.w_semantic = tp.ParticleField.WSemantic.Radius
    cfg.uniform_radius = 0.03
    field = tp.ParticleField.create(cfg)
    e = tp.ParticleField.EmitterParams()
    e.spawn_center = tp.Vector3(0, 0.3, 0)
    e.spawn_half_extent = tp.Vector3(5, 0.25, 5)
    e.velocity = tp.Vector3(0, 0, 0)
    e.lifetime = 10.0
    e.size = 0.03
    field.set_emitter(e)
    pm = tp.MeshStandardMaterial()
    field.set_mesh_repr(tp.BoxGeometry(0.06, 0.06, 0.06), pm)
    field.frustum_culled = False
    scene.add(field)
    cam = _top_camera(7.0, fov=60)

    def disc_share():
        field.set_emitter_time(2.0, 1.0 / 60.0)
        _step(r, scene, cam, 2.0)
        ids = r.read_instance_ids(scene, cam)
        ys, xs = np.mgrid[0:ids.shape[0], 0:ids.shape[1]]
        rr = np.hypot(xs - ids.shape[1] / 2, ys - ids.shape[0] / 2)
        disc = rr < 0.12 * ids.shape[0]
        return float((ids[disc] != 0).mean()), int((ids != 0).sum())

    calm, n0 = disc_share()
    dw = tp.Downwash()
    jet = dw.add_jet(tp.Vector3(0, 1.5, 0))
    jet.surface_y = 0.0
    r.downwash = dw
    blown, n1 = disc_share()
    assert n0 > 2000 and n1 > 2000
    assert calm > 0.05, f"the cloud covers the disc before ({calm:.3f})"
    assert blown < 0.2 * calm, f"the jet empties it ({calm:.3f} -> {blown:.3f})"
