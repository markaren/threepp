"""Vulkan deferred renderer + G-buffer AOV readback.

Skips entirely on a GL-only build (threepp.HAS_VULKAN is False). On a Vulkan
build these require a Vulkan-capable GPU.
"""
import importlib.util
import math
import os
import time

import numpy as np
import pytest

import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")

# Same 4:3 aspect as the old 128x96, but wide enough that no window manager
# clamps it: a headless canvas on Windows falls back to a hidden window, and
# the WM enforces a minimum window width (~232 px), so a 128-wide canvas gets
# a 232-wide swapchain and every (H, W) shape assertion fails.
W, H = 320, 240


@pytest.fixture(scope="module")
def vk_canvas():
    return tp.Canvas("vk-test", width=W, height=H, headless=True, vsync=False)


@pytest.fixture(scope="module")
def vk_renderer(vk_canvas):
    return tp.VulkanRenderer(vk_canvas)


def make_scene():
    scene = tp.Scene()
    scene.background = 0x202830
    mat = tp.MeshStandardMaterial()
    mat.color = 0xff5533
    scene.add(tp.Mesh(tp.BoxGeometry(), mat))
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    sun = tp.DirectionalLight(0xffffff, 3.0)
    sun.position.set(3, 5, 2)
    scene.add(sun)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(1.5, 1.5, 3.0)
    cam.look_at(0, 0, 0)
    return scene, cam


def test_rgb_render(vk_renderer):
    scene, cam = make_scene()
    img = vk_renderer.render_aov(scene, cam, "rgb")
    assert img.shape == (H, W, 3) and str(img.dtype) == "uint8"
    assert int(img.max()) > int(img.min()), "shaded render is flat"


def _renderer_without_presents(title):
    """A canvas and renderer of their own, with presents suppressed. The switch
    is read once, at context creation, so it is set for the constructor only."""
    prev = os.environ.get("THREEPP_VULKAN_SUPPRESS_PRESENT")
    os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"] = "1"
    try:
        canvas = tp.Canvas(title, width=W, height=H, headless=True, vsync=False)
        return tp.VulkanRenderer(canvas)
    finally:
        if prev is None:
            del os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"]
        else:
            os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"] = prev


def _flat_scene(background):
    scene = tp.Scene()
    scene.background = background
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    return scene, cam


def _hud_patch():
    """A blue unlit square over the middle of the view, and its 2D camera."""
    hud = tp.Scene()
    patch = tp.MeshBasicMaterial()
    patch.color = 0x0000ff
    hud.add(tp.Mesh(tp.PlaneGeometry(1, 1), patch))
    cam = tp.OrthographicCamera(-W / H, W / H, 1, -1, 0.1, 10)
    cam.position.z = 1
    return hud, cam


def _centre(img):
    return _dominant(img[img.shape[0] // 2, img.shape[1] // 2][:3])


def _change_background(renderer, scene, colour):
    """Change the background and restart the temporal history. TAA weights a new
    frame by its duration (the blend is anchored at 90 fps), so without the reset
    the first frame after a full-screen change is mostly the old picture when
    frames come fast: (202, 53, 0) for red to green at 1.4 ms a frame, green
    alone from 50 ms. With the reset that frame is the new picture at any frame
    rate, and a readback of an older swapchain image still reads the old one."""
    scene.background = colour
    renderer.reset_temporal_history()


@pytest.mark.parametrize("presents", ["presenting", "suppressed"])
def test_save_frame_writes_the_frame_it_rendered(vk_renderer, presents, tmp_path):
    """save_frame renders a frame and reads it back in one call. render() only
    records: the submit and the present run from the canvas frame-end callback,
    after the animate body returns. The readback used to copy the swapchain
    image before that, so the PNG held whatever the image carried from its
    previous use: the scene as it was three frames earlier (two with presents
    suppressed, which pins one image per in-flight slot), or an image nothing
    had ever drawn into when save_frame was the first call. A still taken after
    a loop of render() calls looked right only because nothing had changed in
    between."""
    renderer = vk_renderer
    scene, cam = _flat_scene(0xff0000)

    def saved(name):
        path = tmp_path / name
        renderer.save_frame(scene, cam, str(path))
        return tp.TextureLoader(False).load(str(path), False).to_numpy()[..., :3]

    if presents == "suppressed":
        # A renderer of its own, so this is also the first-call case.
        renderer = _renderer_without_presents("vk-test-save-frame")
        first = saved("first.png")
        assert _centre(first) == "r", f"first call: {first[H // 2, W // 2]}"

    for _ in range(3):
        renderer.render(scene, cam)
    _change_background(renderer, scene, 0x00ff00)
    png = saved("changed.png")
    assert _centre(png) == "g", f"after the change: {png[H // 2, W // 2]}"
    # The file and read_pixels() are the same frame, so the same bytes.
    assert np.array_equal(png, renderer.read_pixels())


def test_readback_inside_the_frame_returns_that_frame(vk_canvas, vk_renderer):
    """The same readback from inside the animate body, which is how the C++
    examples take their stills (render(), then writeFramebuffer()). The readback
    finishes the frame to read it, so what it returns is what was recorded up to
    that point: the scene, and a HUD pass rendered before it."""
    scene, cam = _flat_scene(0xff0000)
    hud, hud_cam = _hud_patch()
    got = {}

    for _ in range(3):
        vk_renderer.render(scene, cam)
    _change_background(vk_renderer, scene, 0x00ff00)

    def scene_then_readback():
        vk_renderer.render(scene, cam)
        got["scene"] = vk_renderer.read_pixels()

    assert vk_canvas.animate_once(scene_then_readback)
    assert _centre(got["scene"]) == "g", f"scene: {got['scene'][H // 2, W // 2]}"
    assert np.array_equal(got["scene"], vk_renderer.read_pixels())

    def hud_then_readback():
        vk_renderer.render(scene, cam)
        vk_renderer.render(hud, hud_cam)
        got["hud"] = vk_renderer.read_pixels()

    assert vk_canvas.animate_once(hud_then_readback)
    assert _centre(got["hud"]) == "b", f"HUD patch: {got['hud'][H // 2, W // 2]}"
    assert _dominant(got["hud"][5, 5]) == "g", f"scene beside the HUD: {got['hud'][5, 5]}"


def test_hud_pass_after_a_readback_is_dropped_with_a_hint(vk_canvas, vk_renderer, capfd):
    """A HUD pass draws onto the frame the scene render opened. After a readback
    that frame is finished, and the pass would open one of its own: the HUD over
    a cleared image, presented right after the scene. It is dropped instead, and
    the renderer says to move it before the readback. The hint prints once per
    process."""
    scene, cam = _flat_scene(0x00ff00)
    hud, hud_cam = _hud_patch()
    got = {}

    def readback_then_hud():
        vk_renderer.render(scene, cam)
        got["scene"] = vk_renderer.read_pixels()
        vk_renderer.render(hud, hud_cam)

    capfd.readouterr()
    assert vk_canvas.animate_once(readback_then_hud)
    err = capfd.readouterr().err
    assert "before the readback" in err, "no hint was printed; stderr was:\n" + err
    assert _centre(got["scene"]) == "g"
    # The frame that was presented is the scene, not the HUD over a cleared image.
    assert _centre(vk_renderer.read_pixels()) == "g"
    # The next frame is whole again: the drop does not outlive its frame.
    assert vk_canvas.animate_once(
        lambda: (vk_renderer.render(scene, cam), vk_renderer.render(hud, hud_cam)))
    assert _centre(vk_renderer.read_pixels()) == "b"


def test_sim_time_round_trips(vk_renderer):
    """The deterministic frame clock: None on the wall clock (the default), a
    pinned value reads back exactly, and None or a negative value releases it."""
    try:
        assert vk_renderer.sim_time is None
        vk_renderer.sim_time = 4.25
        assert vk_renderer.sim_time == pytest.approx(4.25)
        vk_renderer.sim_time = -1.0
        assert vk_renderer.sim_time is None
        vk_renderer.sim_time = 4.25
        vk_renderer.sim_time = None
        assert vk_renderer.sim_time is None
    finally:
        vk_renderer.sim_time = None  # back to the wall clock


_OCEAN_PROBES = [(0.0, 0.0), (13.5, -7.25), (-41.0, 30.5)]


def _ocean_heights(renderer, *, t0, frames=4, dt=1.0 / 30.0, pace_s=0.0):
    """Render a fresh Ocean for `frames` frames with renderer.sim_time stepped
    from t0 by dt, sleeping pace_s of WALL time between frames, and return the
    CPU-mirrored wave height at a few probes."""
    scene = tp.Scene()
    scene.background = 0x202830
    ocean = tp.Ocean(size=200.0, resolution=64, wind_speed=10.0, fft_size=128)
    scene.add(ocean)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 1000)
    cam.position.set(0, 15, 40)
    cam.look_at(0, 0, 0)
    ocean.sample_height(0.0, 0.0)  # arms the GPU->CPU height readback (sticky opt-in)
    for i in range(frames):
        renderer.sim_time = t0 + i * dt
        renderer.render(scene, cam)
        if pace_s:
            time.sleep(pace_s)
    return [ocean.sample_height(x, z) for x, z in _OCEAN_PROBES]


@pytest.mark.skipif(not hasattr(tp, "Ocean"), reason="Ocean needs the Vulkan build")
def test_sim_time_pins_ocean_against_frame_pacing(vk_renderer):
    """With sim_time pinned, the same frame sequence rendered back-to-back and
    rendered with 0.1 s of wall time between frames lands the ocean on the same
    wave heights. Unpinned, the sea advances at wall speed while the app steps
    its own physics at 1/fps -- the warp_sailboat film bug, where a 1080p render
    (~90 ms/frame) got a ~5x faster ocean than the buoyancy follower expected."""
    try:
        fast = _ocean_heights(vk_renderer, t0=3.0)
        slow = _ocean_heights(vk_renderer, t0=3.0, pace_s=0.1)
        assert any(abs(h) > 1e-3 for h in fast), "height mirror never filled"
        assert slow == pytest.approx(fast, abs=1e-4)
        # The pin is what drives the field: a different pinned time is a different sea.
        later = _ocean_heights(vk_renderer, t0=4.5)
        assert later != pytest.approx(fast, abs=1e-3)
    finally:
        vk_renderer.sim_time = None


def _ocean_height_series(renderer, *, t0, frames, dt, pace_s, probe=(12.5, -7.25)):
    """Render a fresh 1400 m / 640-vertex / FFT-1024 Ocean for `frames` frames with
    renderer.sim_time stepped from t0 by dt, sleeping pace_s of WALL time after
    each frame, and return the CPU-mirrored wave height at `probe` after every
    frame (the order buoyancy code reads it in)."""
    scene = tp.Scene()
    scene.background = 0x202830
    ocean = tp.Ocean(size=1400.0, resolution=640, wind_speed=10.0, fft_size=1024, wave_scale=1.7)
    scene.add(ocean)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 5000)
    cam.position.set(0, 15, 40)
    cam.look_at(0, 0, 0)
    ocean.sample_height(0.0, 0.0)  # arms the GPU->CPU height readback (sticky opt-in)
    out = []
    for i in range(frames):
        renderer.sim_time = t0 + i * dt
        renderer.render(scene, cam)
        if pace_s:
            time.sleep(pace_s)
        out.append(ocean.sample_height(*probe))
    return out


@pytest.mark.skipif(not hasattr(tp, "Ocean"), reason="Ocean needs the Vulkan build")
def test_sample_height_mirror_is_pacing_independent():
    """DisplacedMesh.sample_height reads a CPU mirror of the GPU height field. The
    mirror must sit a FIXED number of frames behind the render, not "whatever the
    GPU had finished copying when the host looked": the same pinned sim_time
    sequence, sampled after every frame, gives the same heights flat out and with
    wall time between frames. Runs one renderer frame per render() with presents
    suppressed (the film/bench mode), where the frames in flight really overlap;
    with the binding's default 3 flush frames per render(), or a presenting hidden
    window, the pipeline drains every call and the race cannot show. Before the
    per-frame-in-flight readback ring (2026-08) this differed in ~95% of frames
    at this sea (max ~3 cm; torn mixes of two frames' fields). It then still
    differed in 1-2 frames of 40 (a few mm, run to run, fast-vs-fast too) from
    two GPU-side hazards the ring could not see: the readback copy (TRANSFER)
    racing the NEXT frame's spectrum dispatch rewriting the same image, and the
    dynamic-spectrum time living in one host-mapped UBO rewritten at record
    time (frame N's cascades evolving to frame N+1's time). Both fixed 2026-08-23
    (TRANSFER->COMPUTE barrier; push-constant params)."""
    prev = os.environ.get("THREEPP_VULKAN_SUPPRESS_PRESENT")
    os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"] = "1"  # read once, at context creation
    try:
        canvas = tp.Canvas("vk-test-height-mirror", width=W, height=H, headless=True, vsync=False)
        renderer = tp.VulkanRenderer(canvas, flush_frames=1)
    finally:
        if prev is None:
            del os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"]
        else:
            os.environ["THREEPP_VULKAN_SUPPRESS_PRESENT"] = prev
    try:
        fast = _ocean_height_series(renderer, t0=3.0, frames=40, dt=1.0 / 60.0, pace_s=0.0)
        slow = _ocean_height_series(renderer, t0=3.0, frames=40, dt=1.0 / 60.0, pace_s=0.03)
        assert any(abs(h) > 1e-3 for h in fast), "height mirror never filled"
        steps = [abs(b - a) for a, b in zip(fast[4:], fast[5:])]
        assert sum(steps) / len(steps) > 1e-3, "the sea did not advance under the stepped clock"
        assert slow == pytest.approx(fast, abs=1e-6)
    finally:
        renderer.sim_time = None


def test_auto_lod_chain_lands_on_a_fixed_frame_under_a_pinned_clock():
    """Auto-LOD simplifies each eligible geometry on a background thread, and the
    frame loop takes the chain when it is done. On the wall clock that is whenever
    the thread finished. With sim_time pinned it has to be a function of the scene:
    otherwise a replayed run switches level on a different frame whenever the
    machine is busy, and every stochastically shaded pixel near the mesh diverges
    from there (usv_ocean --cam abeam under CPU load, 2026-10-01: ~11,000 px of
    the still, boat, reflection and wake). So a pinned frame waits for the chains
    queued before it, and no render() ends with a job still in the worker. The
    sphere's ~295k triangles take many frames' worth of wall time to simplify;
    without the wait chains_queued read 1 for all of them."""
    canvas = tp.Canvas("vk-test-lod-pinned", width=W, height=H, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas, flush_frames=1)
    scene, cam = make_scene()
    ball = tp.Mesh(tp.SphereGeometry(0.5, 384, 384), tp.MeshStandardMaterial())
    ball.position.set(-1.2, 0.0, 0.0)
    scene.add(ball)
    try:
        ready_at = None
        for i in range(40):
            renderer.sim_time = i / 60.0
            renderer.render(scene, cam)
            stats = renderer.auto_lod_stats
            assert stats["chains_queued"] == 0, f"frame {i} ended with a chain still in the worker"
            if ready_at is None and stats["chains_ready"] == 1:
                ready_at = i
        assert ready_at is not None, "the sphere never got a chain"
    finally:
        renderer.sim_time = None


def test_aov_shapes(vk_renderer):
    scene, cam = make_scene()
    out = vk_renderer.render_aovs(scene, cam, ["rgb", "normals", "segmentation", "albedo"])
    assert set(out) == {"rgb", "normals", "segmentation", "albedo"}
    for arr in out.values():
        assert arr.shape == (H, W, 3) and str(arr.dtype) == "uint8"


def test_aovs_are_distinct(vk_renderer):
    scene, cam = make_scene()
    rgb = vk_renderer.render_aov(scene, cam, "rgb")
    normals = vk_renderer.render_aov(scene, cam, "normals")
    seg = vk_renderer.render_aov(scene, cam, "segmentation")
    assert not np.array_equal(rgb, normals)
    assert not np.array_equal(rgb, seg)
    assert not np.array_equal(normals, seg)


def test_segmentation_has_distinct_regions(vk_renderer):
    scene, cam = make_scene()
    seg = vk_renderer.render_aov(scene, cam, "segmentation")
    colors = {tuple(c) for c in seg.reshape(-1, 3).tolist()}
    # at least background (black) + the box's hashed id colour
    assert len(colors) >= 2


def test_convenience_aov_accessors(vk_renderer):
    scene, cam = make_scene()
    assert vk_renderer.read_normals(scene, cam).shape == (H, W, 3)
    assert vk_renderer.read_segmentation(scene, cam).shape == (H, W, 3)
    assert vk_renderer.read_albedo(scene, cam).shape == (H, W, 3)


def test_unknown_aov_raises(vk_renderer):
    scene, cam = make_scene()
    with pytest.raises(ValueError):
        vk_renderer.render_aov(scene, cam, "not_an_aov")


def test_depth_is_metric(vk_renderer):
    # A fronto-parallel wall filling the frame: view-space depth equals the
    # camera distance, so read_depth must return that distance.
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    scene.add(tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial()))
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cy, cx = H // 2, W // 2
    for dist in (3.0, 7.0, 15.0):
        cam.position.set(0, 0, dist)
        cam.look_at(0, 0, 0)
        depth = vk_renderer.read_depth(scene, cam)
        assert depth.shape == (H, W) and str(depth.dtype) == "float32"
        assert depth[cy, cx] == pytest.approx(dist, abs=0.05)


@pytest.mark.skipif(not tp.HAS_IMGUI, reason="built without imgui")
def test_imgui_over_vulkan():
    # The ImGui overlay records into the Vulkan deferred frame after the scene.
    # (Kept here, not in test_imgui.py, so a GL and a Vulkan ImGui context are
    # never alive at the same time — two live ImGui contexts crash.)
    canvas = tp.Canvas("imgui-vk", width=320, height=240, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    ui = tp.ImguiContext(canvas, renderer)  # Vulkan backend

    scene = tp.Scene()
    scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    cam = tp.PerspectiveCamera(60, 320 / 240, 0.1, 100)
    cam.position.set(1.5, 1.5, 3)
    cam.look_at(0, 0, 0)

    def draw():
        tp.imgui.set_next_window_pos(8, 8)
        tp.imgui.set_next_window_size(180, 120)
        tp.imgui.begin("Vulkan UI")
        tp.imgui.text("hello")
        tp.imgui.slider_float("v", 0.5, 0.0, 1.0)
        tp.imgui.end()

    for _ in range(3):  # flush MAILBOX
        canvas.animate_once(lambda: (renderer.render(scene, cam), ui.render(draw)))
    img = renderer.read_pixels()
    panel = img[8:128, 8:188]
    assert int(panel.max()) - int(panel.min()) > 40, "Vulkan UI panel did not draw"


@pytest.mark.skipif(not tp.HAS_IMGUI, reason="built without imgui")
def test_imgui_vulkan_canvas_needs_renderer():
    canvas = tp.Canvas("imgui-vk2", width=64, height=64, headless=True, vsync=False)
    tp.VulkanRenderer(canvas)
    # A renderer is required on every backend; omitting it must fail (pybind
    # raises TypeError for the missing arg, our own guard raises ValueError).
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        tp.ImguiContext(canvas)


def test_depth_occlusion(vk_renderer):
    # A box in front of the wall must read nearer at the centre than the wall.
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    scene.add(tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial()))  # z=0
    box = tp.Mesh(tp.BoxGeometry(2, 2, 2), tp.MeshStandardMaterial())
    box.position.set(0, 0, 3)  # front face at z=4
    scene.add(box)
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cam.position.set(0, 0, 8)  # 8 from wall, 4 from box front
    cam.look_at(0, 0, 0)
    depth = vk_renderer.read_depth(scene, cam)
    assert depth[H // 2, W // 2] == pytest.approx(4.0, abs=0.1)   # box front
    assert depth[5, 5] == pytest.approx(8.0, abs=0.1)             # wall behind


def test_orthographic_scene_rendering_defaults_on(vk_renderer):
    assert vk_renderer.orthographic_scene_rendering is True
    try:
        vk_renderer.orthographic_scene_rendering = False
        assert vk_renderer.orthographic_scene_rendering is False
    finally:
        vk_renderer.orthographic_scene_rendering = True


def _dominant(px):
    """'r', 'g' or 'b' when one channel leads the other two by a clear margin,
    'grey' when all three sit together, else None."""
    r, g, b = (int(c) for c in px)
    if max(r, g, b) - min(r, g, b) < 24:
        return "grey"
    for name, (a, o1, o2) in (("r", (r, g, b)), ("g", (g, r, b)), ("b", (b, r, g))):
        if a > o1 + 40 and a > o2 + 40:
            return name
    return None


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_top_down_view_keeps_the_nearest_surface(vk_renderer, projection):
    # A red box on a blue ground plane with a green patch on its lid, seen from
    # straight above. Under an OrthographicCamera the Vulkan backend used to
    # draw this as a 2D overlay (painter's order, no depth, black clear), so
    # the ground, added last, covered the box and the background never showed.
    # Both projections frame the ground plane at 1 m half-height, so the same
    # pixel samples land on the same surfaces.
    scene = tp.Scene()
    scene.background = 0x808080
    scene.add(tp.HemisphereLight(0xffffff, 0xffffff, 1.0))

    def mat(c):
        m = tp.MeshStandardMaterial()
        m.color = c
        m.roughness = 1.0
        return m

    box = tp.Mesh(tp.BoxGeometry(0.6, 0.6, 0.6), mat(0xff0000))
    box.position.y = 0.3
    scene.add(box)
    lid = tp.Mesh(tp.PlaneGeometry(0.2, 0.2), mat(0x00ff00))
    lid.rotation.x = -math.pi / 2
    lid.position.y = 0.601
    scene.add(lid)
    ground = tp.Mesh(tp.PlaneGeometry(1.6, 1.6), mat(0x0000ff))
    ground.rotation.x = -math.pi / 2
    scene.add(ground)

    if projection == "orthographic":
        cam = tp.OrthographicCamera(-W / H, W / H, 1, -1, 0.1, 100)
    else:  # tan(fov/2) = 1 m / 10 m puts the ground's half-height at 1 m too
        cam = tp.PerspectiveCamera(2 * math.degrees(math.atan(0.1)), W / H, 0.1, 100)
    cam.position.set(0, 10, 0)
    cam.up.set(0, 0, -1)
    cam.look_at(0, 0, 0)

    img = vk_renderer.render_aov(scene, cam, "rgb")
    px_per_m = H / 2
    cy, cx = H // 2, W // 2

    def at(metres):
        return img[cy, cx + int(metres * px_per_m)]

    assert _dominant(at(0.0)) == "g", f"lid patch: {at(0.0)}"
    assert _dominant(at(0.2)) == "r", f"box lid: {at(0.2)}"
    assert _dominant(at(0.55)) == "b", f"ground: {at(0.55)}"
    assert _dominant(at(-1.25)) == "grey", f"background: {at(-1.25)}"


def test_ortho_overlay_warns_about_a_lit_scene(vk_renderer, capfd):
    # With the flag off, a standalone ortho render is the 2D overlay, which
    # draws every mesh as a flat fill. MeshBasicMaterial art is what that path
    # is for and stays quiet; a lit material means the camera was a 3D view,
    # and the renderer says which flag to set. The hint prints once per process.
    cam = tp.OrthographicCamera(-W / H, W / H, 1, -1, 0.1, 100)
    cam.position.set(0, 0, 10)
    cam.look_at(0, 0, 0)
    flat = tp.Scene()
    flat.add(tp.Mesh(tp.PlaneGeometry(1, 1), tp.MeshBasicMaterial()))
    lit = tp.Scene()
    lit.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    try:
        vk_renderer.orthographic_scene_rendering = False
        capfd.readouterr()
        vk_renderer.render(flat, cam)
        err = capfd.readouterr().err
        assert "setOrthographicSceneRendering" not in err, err
        vk_renderer.render(lit, cam)
        err = capfd.readouterr().err
        assert "setOrthographicSceneRendering(true)" in err, \
            "no hint was printed; stderr was:\n" + err
    finally:
        vk_renderer.orthographic_scene_rendering = True


def _ocean_scene():
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2, 1, 2)
    scene.add(sun)
    floor_mat = tp.MeshStandardMaterial()
    floor_mat.color = 0x050505
    floor = tp.Mesh(tp.PlaneGeometry(200, 200), floor_mat)
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -5.0
    scene.add(floor)
    # Small grid + FFT so the per-frame displace/BLAS stays cheap in the test.
    ocean = tp.Ocean(size=200.0, resolution=128, fft_size=256, wind_speed=10.0)
    scene.add(ocean)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 400)
    cam.position.set(0, 8, 30)
    cam.look_at(0, 0, 0)
    return scene, cam, ocean


def test_ocean_is_displaced_mesh_with_knobs():
    ocean = tp.Ocean(size=500.0)
    assert isinstance(ocean, tp.DisplacedMesh)
    assert isinstance(ocean, tp.Mesh)
    # inherited Object3D API works across threepp's virtual base
    ocean.position.set(1, 2, 3)
    assert ocean.position.x == 1
    # params + warp are mutable sub-objects
    assert ocean.params.tile_size_0 == pytest.approx(500.0)
    ocean.params.wind_speed = 6.5
    assert ocean.params.wind_speed == pytest.approx(6.5)
    ocean.warp_toward(10.0, -4.0, 0.2)
    assert ocean.warp.half_range > 0 and ocean.warp.center_x == pytest.approx(10.0)
    # foam API is callable
    ocean.add_foam_disturbance(0.0, 0.0, 2.0, 1.0)
    ocean.clear_foam_disturbances()
    # the vessel pair: hull exclusion (incl. the waterline plane) + wake trail
    ocean.hull_exclusion.set_pose(4.0, -3.0, 0.7, yaw=0.5, pitch=0.04, roll=-0.2,
                                  half_length=4.8, half_beam=1.5)
    h = ocean.hull_exclusion
    assert (h.center_x, h.center_z, h.center_y) == pytest.approx((4.0, -3.0, 0.7))
    assert (h.pitch, h.roll, h.half_length) == pytest.approx((0.04, -0.2, 4.8))
    assert h.sin_yaw == pytest.approx(math.sin(0.5))
    ocean.wake.forward_speed = 3.2
    ocean.add_wake_sample(4.0, -3.0, h.sin_yaw, h.cos_yaw, 3.2)
    assert ocean.age_wake(0.5) == 1 and ocean.wake.trail[0].age == pytest.approx(0.5)
    assert ocean.age_wake(10.0) == 0          # aged past max_age -> dropped
    # sampleWakeHeight runs off the CURRENT pose even with an empty trail
    assert ocean.sample_wake_height(6.0, -3.0) > 0.0
    ocean.wake.forward_speed = 0.0            # below the speed gate -> no wake
    assert ocean.sample_wake_height(6.0, -3.0) == pytest.approx(0.0)


def test_ocean_carries_several_vessels():
    ocean = tp.Ocean(size=500.0)
    assert tp.DisplacedMesh.MAX_VESSELS == 8
    with pytest.raises(IndexError):
        ocean.vessel(tp.DisplacedMesh.MAX_VESSELS)
    # vessel 0 IS the mesh-level hull_exclusion / wake (the single-vessel API)
    boat = ocean.vessel(0)
    boat.hull_exclusion.set_pose(0.0, 0.0, half_length=3.0, half_beam=1.0)
    assert ocean.hull_exclusion.half_length == pytest.approx(3.0)
    ocean.wake.forward_speed = 4.0
    assert boat.wake.forward_speed == pytest.approx(4.0)
    assert boat.index == 0

    def bow_bump(x, z):
        return ocean.sample_wake_height(x, z)

    alone = bow_bump(0.0, 3.0)                # the boat's bow, current pose, no trail
    assert alone > 0.0
    # a second vessel far away: its own wake, its own trail, the boat's untouched
    ship = ocean.vessel(1)
    assert ship.hull_exclusion.half_length == 0.0 and "off" in repr(ship)
    ship.hull_exclusion.set_pose(200.0, 0.0, yaw=0.0, half_length=30.0, half_beam=6.0)
    ship.wake.forward_speed = 5.0
    assert bow_bump(0.0, 3.0) == pytest.approx(alone)
    assert bow_bump(200.0, 30.0) > 0.0        # the ship's own bow bump
    ship.add_wake_sample(200.0, 0.0, 0.0, 1.0, 5.0)
    assert ship.age_wake(0.5) == 1 and ship.wake.trail[0].age == pytest.approx(0.5)
    assert len(ocean.wake.trail) == 0         # the trails are separate
    ship.clear_wake()
    assert len(ship.wake.trail) == 0
    # wakes ADD where two vessels' wakes overlap (same pose twice = double)
    twin = ocean.vessel(2)
    twin.hull_exclusion.set_pose(0.0, 0.0, half_length=3.0, half_beam=1.0)
    twin.wake.forward_speed = 4.0
    assert bow_bump(0.0, 3.0) == pytest.approx(2.0 * alone)
    twin.hull_exclusion.half_length = 0.0     # half_length 0 takes a vessel off the ocean
    assert bow_bump(0.0, 3.0) == pytest.approx(alone)
    # a handle keeps its ocean alive
    del ocean, boat
    assert ship.hull_exclusion.half_beam == pytest.approx(6.0)


def test_ocean_renders_and_displaces(vk_renderer):
    scene, cam, ocean = _ocean_scene()
    # The CPU height mirror is a lazy sticky opt-in: the renderer only records
    # the GPU->host copies once something has called sample_height, and the
    # data lands the next rendered frame. Prime it BEFORE the render loop or
    # every later sample reads an (correctly) empty mirror.
    ocean.sample_height(0.0, 0.0)
    for _ in range(8):  # let the wave field evolve + the BLAS displace
        ocean.warp_toward(0.0, 0.0, 0.3)
        vk_renderer.render(scene, cam)
    img = vk_renderer.read_pixels()
    assert img.shape == (H, W, 3) and str(img.dtype) == "uint8"
    assert int(img.max()) > int(img.min()), "ocean render is flat"
    # The CPU height readback is filled by the Vulkan render -> waves vary across
    # the tile (validates the FFT/displace pipeline ran end to end).
    coords = np.linspace(-80, 80, 6)
    heights = np.array([ocean.sample_height(float(x), float(z)) for x in coords for z in coords])
    assert np.all(np.isfinite(heights))
    assert np.ptp(heights) > 1e-3, "wave height field looks flat"


def test_point_light_lights_the_ocean(vk_renderer):
    """A PointLight over the sea is the water's light too. The water shade
    looped the directional and spot lights only, so a lantern lit the hull
    beside it and left the water under it black. Night scene, no environment:
    the lamp is the only thing that can put light on the water."""
    scene = tp.Scene()
    scene.background = 0x000000
    # No choppiness, so no folded crests and no foam: foam is shaded as an
    # opaque surface and took the lamp all along, which is not the water.
    scene.add(tp.Ocean(size=200.0, resolution=128, fft_size=256, wind_speed=6.0, choppiness=0.0))
    lamp = tp.PointLight(0xffffff, 300.0, 0.0, 2.0)
    lamp.position.set(0, 4, 0)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 400)
    cam.position.set(0, 9, -24)
    cam.look_at(0, 0, 0)                                  # the water under the lamp = frame centre

    def water_under_lamp():
        vk_renderer.reset_temporal_history()
        for _ in range(4):
            vk_renderer.sim_time = 5.0                    # the same sea both times
            vk_renderer.render(scene, cam)
        img = vk_renderer.read_pixels()
        return float(img[H // 2 - 12:H // 2 + 12, W // 2 - 16:W // 2 + 16].mean())

    try:
        dark = water_under_lamp()
        scene.add(lamp)
        lit = water_under_lamp()
    finally:
        vk_renderer.sim_time = None
    assert dark < 2.0, "the unlit night sea is not black"
    assert lit > dark + 20.0, "the point light left the water under it dark"


def test_underwater_murk_takes_the_hue_of_what_is_under_it(vk_renderer):
    """A yellow slab 2 m under the sea, seen from straight over it. The scalar
    murk dims it and leaves it yellow, which is why it read as lying at the
    surface. A per-channel murk takes its red first, and the ambient falloff
    takes the sky's fill off it; neither touches the open water beside it, and
    the scalar call renders what it rendered before either existed."""
    scene = tp.Scene()
    scene.background = 0x9db4d0
    scene.add(tp.HemisphereLight(0xffffff, 0x808080, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2, 6, 1)
    scene.add(sun)
    scene.add(tp.Ocean(size=200.0, resolution=128, fft_size=256, wind_speed=2.0, choppiness=0.0))
    mat = tp.MeshStandardMaterial()
    mat.color = tp.Color(0.85, 0.65, 0.02)
    mat.roughness = 0.8
    slab = tp.Mesh(tp.BoxGeometry(8.0, 0.5, 8.0), mat)
    slab.position.set(0, -2.25, 0)                       # its top face 2 m down
    scene.add(slab)
    cam = tp.PerspectiveCamera(40, W / H, 0.1, 400)
    cam.position.set(0, 14, 0.01)
    cam.look_at(0, 0, 0)
    colour = tp.Color(0.055, 0.14, 0.125)                # a fjord's green

    def shot():
        """(the slab, the open water beside it) as mean RGB."""
        vk_renderer.reset_temporal_history()
        for _ in range(6):
            vk_renderer.sim_time = 5.0                    # the same sea every time
            vk_renderer.render(scene, cam)
        img = vk_renderer.read_pixels().astype(np.float64)
        return (img[H // 2 - 15:H // 2 + 15, W // 2 - 20:W // 2 + 20].reshape(-1, 3).mean(0),
                img[H // 2 - 15:H // 2 + 15, 4:24].reshape(-1, 3).mean(0))

    assert vk_renderer.underwater_ambient_falloff == 0.0
    assert vk_renderer.underwater_sprite_foreshortening is True
    try:
        vk_renderer.set_fog_water_surface_y(0.0)
        vk_renderer.set_underwater_murk(0.3, colour)
        e = vk_renderer.underwater_murk_extinction
        assert (e.x, e.y, e.z) == pytest.approx((0.3, 0.3, 0.3))
        slab_s, sea_s = shot()
        assert slab_s[0] > sea_s[0] + 25.0, "the slab does not show through 2 m of this murk"

        # three equal extinctions are the scalar murk
        vk_renderer.set_underwater_murk(tp.Vector3(0.3, 0.3, 0.3), colour)
        slab_v, sea_v = shot()
        assert slab_v == pytest.approx(slab_s, abs=0.5) and sea_v == pytest.approx(sea_s, abs=0.5)

        # per channel, from the colour: green keeps 0.3, red gets 0.3 * 0.14 / 0.055
        vk_renderer.set_underwater_murk(0.3, colour, chromatic=1.0)
        e = vk_renderer.underwater_murk_extinction
        assert (e.x, e.y, e.z) == pytest.approx((0.3 * 0.14 / 0.055, 0.3, 0.3 * 0.14 / 0.125), rel=1e-4)
        slab_c, sea_c = shot()
        assert slab_c[0] < slab_s[0] - 25.0, "the per-channel murk left the slab its red"
        assert abs(slab_c[1] - slab_s[1]) < 4.0, "the clearest channel moved"
        assert sea_c == pytest.approx(sea_s, abs=2.0), "the open water changed colour"

        # the sky's fill dies with depth
        vk_renderer.set_underwater_murk(0.3, colour)
        vk_renderer.underwater_ambient_falloff = 1.0
        slab_a, sea_a = shot()
        assert slab_a[1] < slab_s[1] - 6.0, "the ambient falloff left the slab its fill"
        assert sea_a == pytest.approx(sea_s, abs=2.0), "the open water changed with the ambient falloff"
    finally:
        vk_renderer.underwater_ambient_falloff = 0.0
        vk_renderer.set_underwater_murk(0.0)
        vk_renderer.set_fog_water_surface_y(1e30)
        vk_renderer.sim_time = None


def test_depthsensor_pathtraced(vk_renderer):
    # The backend-neutral tp.DepthSensor.scan must work on Vulkan (path-traced
    # through the renderer's TLAS) and reconstruct world-space heights — the same
    # call signature as on GL. Z-up scene: ground top at z=0, a 0.30 m cube on it.
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    ground = tp.Mesh(tp.BoxGeometry(20, 20, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)                       # top at z=0
    scene.add(ground)
    box = tp.Mesh(tp.BoxGeometry(0.6, 0.6, 0.3), tp.MeshStandardMaterial())
    box.position.set(0, 0, 0.15)                          # top at z=0.30
    scene.add(box)
    cam = tp.PerspectiveCamera(55, W / H, 0.05, 100)
    cam.up.set(0, 0, 1)
    cam.position.set(3, 3, 3)
    cam.look_at(0, 0, 0)
    vk_renderer.render(scene, cam)                        # build the TLAS (required before scan)

    sensor = tp.DepthSensor(fov_y=70, width=96, height=96, near=0.05, far=8.0)
    sensor.range_noise = 0.0                              # exact reconstruction for the assert
    sensor.position.set(0, 0, 3.0)                        # identity rot -> looks straight down (Z-up)
    pts = sensor.scan(vk_renderer, scene)

    assert pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 1000
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[:, 2]
    on_box = r < 0.25                                     # squarely over the cube top
    on_ground = (r > 1.0) & (r < 4.0)                     # well off the cube
    assert on_box.sum() > 5 and z[on_box].mean() == pytest.approx(0.30, abs=0.02)
    assert on_ground.sum() > 50 and z[on_ground].mean() == pytest.approx(0.0, abs=0.02)


def test_depthsensor_scan_is_reproducible(vk_renderer):
    # A generated dataset is only worth recording if it can be replayed. On
    # Vulkan the range noise is drawn by the path-traced back-end the sensor
    # delegates to, so this covers the seam a GL-only test cannot: a reset that
    # stopped at the front door would leave exactly this path unreplayable.
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    ground = tp.Mesh(tp.BoxGeometry(20, 20, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)                       # Z-up, top at z=0
    scene.add(ground)
    cam = tp.PerspectiveCamera(55, W / H, 0.05, 100)
    cam.up.set(0, 0, 1)
    cam.position.set(3, 3, 3)
    cam.look_at(0, 0, 0)
    vk_renderer.render(scene, cam)                        # build the TLAS

    sensor = tp.DepthSensor(fov_y=70, width=64, height=64, near=0.05, far=8.0)
    sensor.position.set(0, 0, 3.0)

    # A zero model is a perfect sensor: two scans of a static scene are equal.
    sensor.noise = tp.RangeNoiseModel()
    assert np.array_equal(sensor.scan(vk_renderer, scene), sensor.scan(vk_renderer, scene))

    # With noise the stream advances (consecutive scans differ)...
    sensor.noise = tp.RangeNoiseModel(stddev=0.03, seed=99)
    first = sensor.scan(vk_renderer, scene)
    second = sensor.scan(vk_renderer, scene)
    assert not np.array_equal(first, second), "noise frozen: the stream is not advancing"
    assert first.shape == second.shape
    assert np.abs(first - second).max() > 1e-4

    # ...and re-seeding replays the episode exactly.
    sensor.reset_noise()
    assert np.array_equal(sensor.scan(vk_renderer, scene), first)

    # Scans are stamped with the sensor's sim clock, not a wall clock.
    assert sensor.last_scan_time == 0.0                   # nothing has driven it
    sensor.advance_clock(0.25)
    sensor.scan(vk_renderer, scene)
    assert sensor.last_scan_time == pytest.approx(0.25)


@pytest.mark.skipif(importlib.util.find_spec("warp") is None,
                    reason="warp-lang not installed")
def test_vertex_interop_cuda_write(vk_renderer):
    """A CUDA producer writing the renderer's own vertex buffer changes the image.

    Local-verification only: no CI job has both Vulkan and CUDA (the
    python-bindings job builds a GL preset, the lavapipe job has no GPU), so
    this never runs there.
    """
    import warp as wp

    from threepp.cuda_interop import VkInteropArray

    wp.init()
    device = wp.get_preferred_device()
    if not device.is_cuda:
        pytest.skip("no CUDA device")

    # Fixed capacity, as the contract requires: one live triangle plus a
    # degenerate tail. The Vulkan mesh path honours drawRange now, so the tail is
    # no longer the only way to hide unused triangles — this gate keeps it because
    # a degenerate tail is still valid and is what it was written against.
    # 9 tris = 27 verts = 324 position bytes — deliberately NOT 16-aligned:
    # createExternalBuffer pads the export allocation up and reports the
    # PADDED size, and the repaint-transplant below must accept export >=
    # buffer (an 8-tri / 288-byte capacity is aligned and hid exactly that).
    ntri, cap = 9, 9 * 3
    # All vertices coincident, and deliberately PARKED FAR OFF-SCREEN at y=-50.
    # Zero area, so nothing rasterises — and, more to the point, the host array
    # describes a shape 50 m below the camera while the producer draws one at the
    # origin. That is the honest shape of interop (the host copy is meaningless
    # once a foreign device owns the buffer), and it is a regression gate: every
    # CPU-bounds cull in the renderer must exempt an interop mesh, or the entry is
    # culled out of the G-buffer and the CUDA-written triangle never appears. An
    # array of zeros would pass by accident, because the origin is in view.
    pos = np.zeros((cap, 3), np.float32)
    pos[:, 1] = -50.0
    geom = tp.BufferGeometry()
    geom.set_attribute("position", pos)
    geom.set_attribute("normal", np.tile(np.float32([0, 0, 1]), (cap, 1)))
    # Present BEFORE arming, so the repaint phase below is a same-shape content
    # update — the case the record transplant must survive. Adding the
    # attribute after arming is the OTHER case (deliberate loud disable).
    geom.set_attribute("color", np.tile(np.float32([0.2, 0.8, 0.2]), (cap, 1)))
    mat = tp.MeshStandardMaterial()
    mat.color = 0xffffff
    mat.side = tp.Side.Double
    mesh = tp.Mesh(geom, mat)
    mesh.frustum_culled = False             # CPU bounds never see the GPU writes

    scene = tp.Scene()
    scene.background = 0x000000
    scene.add(mesh)
    scene.add(tp.AmbientLight(0xffffff, 3.0))
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(0, 0, 3)
    cam.look_at(0, 0, 0)

    # Depth, not colour: it answers "did the geometry land" without depending on
    # how a one-triangle scene happens to shade.
    blank = vk_renderer.read_depth(scene, cam)          # also builds the record
    assert blank[H // 2, W // 2] > 50.0, "the degenerate tail drew something"
    # Poll, exactly as the API documents. The first call can legitimately return
    # None twice over: the BlasRecord does not exist until the mesh has been
    # rendered once, and an ordinary mesh's first record is built with PACKED
    # attributes, which no external producer can write — the renderer then
    # schedules an unpacked rebuild and asks to be called again. Three frames is
    # ample for both handoffs; anything more is a real failure, not a skip.
    handles = None
    for _ in range(3):
        handles = vk_renderer.enable_vertex_interop(mesh, lambda: None)
        if handles is not None:
            break
        vk_renderer.read_depth(scene, cam)              # let the rebuild land
    assert handles is not None,         "vertex interop never armed after 3 frames of polling"
    (pos_handle, pos_bytes), (nrm_handle, nrm_bytes) = handles
    assert pos_bytes >= cap * 12 and nrm_bytes >= cap * 12

    tri = pos.copy()
    tri[0], tri[1], tri[2] = (-1, -1, 0), (1, -1, 0), (0, 1, 0)
    ipos = VkInteropArray(pos_handle, pos_bytes, wp.vec3, cap, device)
    inrm = VkInteropArray(nrm_handle, nrm_bytes, wp.vec3, cap, device)
    try:
        def on_frame():
            ipos.array.assign(tri)
            inrm.array.assign(np.tile(np.float32([0, 0, 1]), (cap, 1)))
            wp.synchronize_device(device)   # MUST be synchronous — see the API doc

        # Re-arm with the real callback now that the imports exist (same
        # allocations, so the handles above stay valid).
        assert vk_renderer.enable_vertex_interop(mesh, on_frame) is not None
        drawn = vk_renderer.read_depth(scene, cam)
        assert drawn[H // 2, W // 2] == pytest.approx(3.0, abs=0.05), \
            "the CUDA-written triangle never reached the renderer's vertex buffer"

        # Repaint under interop. A same-count edit to a CPU-owned side
        # attribute (vertex color here) bumps the composite geomVersion, and
        # the eviction path used to rebuild the record from the STALE host
        # arrays — freeing exports CUDA still held imports of, dangling the
        # already-enqueued refit, and freezing the mesh at the parked
        # positions (warp_hull_sculpt's "nothing moves after the color
        # change"). The record is transplanted now: the new colors land and
        # the producer's writes keep landing. Prove the latter by moving the
        # triangle AFTER the repaint and reading its new depth.
        # The eviction only ever ran on a frame that was structural for some
        # OTHER reason (the interop refit re-stamps geomVersion every frame,
        # so a lone side-edit is deferred until one) — in warp_hull_sculpt it
        # was a material+visibility flip in the same call as the repaint.
        # Recreate that: entry churn forces the full structural pass through
        # the eviction branch on the very frame the color version moved.
        churn = tp.Mesh(tp.BoxGeometry(0.1, 0.1, 0.1), tp.MeshStandardMaterial())
        churn.position.set(6.0, 0.0, 0.0)               # out of the camera's way
        scene.add(churn)
        geom.update_attribute("color", np.tile(np.float32([0.9, 0.1, 0.1]), (cap, 1)))
        vk_renderer.read_depth(scene, cam)              # the structural frame
        tri[0], tri[1], tri[2] = (-1, -1, 1), (1, -1, 1), (0, 1, 1)
        moved = vk_renderer.read_depth(scene, cam)
        assert moved[H // 2, W // 2] == pytest.approx(2.0, abs=0.05), \
            "producer writes stopped landing after a color repaint - the " \
            "interop registration did not survive the record transplant"
    finally:
        ipos.close()
        inrm.close()
        vk_renderer.disable_vertex_interop(mesh)


def test_vertex_interop_blas_sizing_validates_clean(vk_renderer):
    """Arming vertex interop on a real-sized mesh must not overrun its BLAS.

    Interop refits build with PREFER_FAST_BUILD, and that lineage is a
    different BVH format from the PREFER_FAST_TRACE one the initial build
    uses — on NVIDIA ~5% LARGER over the same geometry (345856 vs 329600 B
    measured at 5120 triangles). Before buildBlasFor sized interop records'
    storage for the max of both lineages, every armed frame built past the
    end of the structure: VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-
    10126 with the layers on, silent heap corruption surfacing as
    VK_ERROR_DEVICE_LOST without them. test_vertex_interop_cuda_write never
    sees this because its 8-triangle capacity is too small for the two
    lineages' sizes to diverge; an icosphere at subdivision 4 is comfortably
    past the crossover.

    No CUDA needed — the overrun is in the renderer's own refit, not the
    producer's write, so a no-op on_frame reproduces it. Meaningful ONLY
    under THREEPP_VULKAN_VALIDATION=1: the assertion is "zero validation
    errors", so the test skips (rather than passing vacuously) when the
    layer is not active.
    """
    if not tp.vulkan_validation_active():
        pytest.skip("validation layer not active - run with THREEPP_VULKAN_VALIDATION=1")

    geom = tp.IcosahedronGeometry(1.0, 4)               # 5120 triangles
    mesh = tp.Mesh(geom, tp.MeshStandardMaterial())
    scene = tp.Scene()
    scene.background = 0x000000
    scene.add(mesh)
    scene.add(tp.AmbientLight(0xffffff, 3.0))
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(0, 0, 4)
    cam.look_at(0, 0, 0)

    vk_renderer.render(scene, cam)                      # builds the record
    handles = None
    for _ in range(4):                                  # poll, as the API documents
        handles = vk_renderer.enable_vertex_interop(mesh, lambda: None)
        if handles is not None:
            break
        vk_renderer.render(scene, cam)                  # let the unpacked rebuild land
    if handles is None:
        pytest.skip("vertex interop unavailable (no external-memory support)")

    err0 = tp.vulkan_validation_error_count()
    try:
        # Every armed frame records a PREFER_FAST_BUILD build into the
        # record's storage (the flags flip away from the initial FAST_TRACE
        # build forces MODE_BUILD on the first one), so a handful of frames
        # is plenty to trip an undersized allocation.
        for _ in range(6):
            vk_renderer.render(scene, cam)
    finally:
        vk_renderer.disable_vertex_interop(mesh)        # drains the device
    assert tp.vulkan_validation_error_count() == err0, \
        "BLAS refit under vertex interop raised validation errors " \
        "(storage too small for the FAST_BUILD lineage?)"


def test_pathtraced_lidar_sensor(vk_renderer):
    """The renderer's own lidar tracer, finally reachable from Python: full
    radiometric returns, live LidarParams, and deterministic replay."""
    scene, cam = make_scene()                             # unit box at origin
    vk_renderer.render(scene, cam)                        # build the TLAS

    lid = tp.PathTracedLidarSensor(60.0, 64, 48, 50.0)    # depth-camera mode
    lid.position.set(0, 0, 3)                             # default pose faces -Z; the
    #                                                       unit box must subtend >100 beams
    r = lid.scan(vk_renderer)
    hits = r["return_no"] > 0
    assert r["position"].shape == (lid.beam_count, 3) and hits.sum() > 100
    assert (r["intensity"][hits] > 0.0).all() and (r["intensity"][hits] <= 1.0).all()
    assert (r["instance_id"][hits] >= 0).all()            # stable ids on surface hits

    # params are live state, not a copy: raising the detector threshold must
    # drop the grazing-attenuated returns (this knob was UNREACHABLE from
    # Python before). Tilt the box first -- head-on inside the reference
    # range every face pixel clamps to 1.0 and no threshold can bite.
    scene.children[0].rotation.y = 0.7
    vk_renderer.render(scene, cam)
    r_low = lid.scan(vk_renderer)
    lid.params.detector_threshold = 0.9
    r2 = lid.scan(vk_renderer)
    assert 0 < (r2["return_no"] > 0).sum() < (r_low["return_no"] > 0).sum()

    # Zero-noise scans of a static scene replay exactly.
    r3 = lid.scan(vk_renderer)
    assert np.array_equal(r2["position"], r3["position"])
    assert np.array_equal(r2["intensity"], r3["intensity"])

    # The raw beam-table entry point: one dispatch, arbitrary beams. This is
    # the multi-pose scoring path (12 poses as one table instead of 12
    # scans); here 64 parallel beams straight at the box.
    org = np.zeros((64, 3), np.float32)
    org[:, 2] = 3.0
    org[:, 0] = np.linspace(-0.3, 0.3, 64)
    dirs = np.tile(np.float32([0, 0, -1]), (64, 1))
    raw = vk_renderer.scan_lidar(org, dirs, tp.LidarParams())
    raw_hits = raw["return_no"] > 0
    assert raw_hits.sum() > 32 and (raw["intensity"][raw_hits] > 0).all()
    raw2 = vk_renderer.scan_lidar(org, dirs, tp.LidarParams())
    assert np.array_equal(raw["position"], raw2["position"])


def test_lidar_intensity_monotone_at_grazing(vk_renderer):
    """The Smith G regression gate. Without shadowing-masking the back-scatter
    specular is D*F/(4cos^2) and intensity CLIMBS back up past ~80 deg
    incidence (measured 0.017 at 55 deg but >0.16 at 85 on the old shader),
    handing every closed body a bright silhouette rim no shaping can remove.
    Post-fix the curve decays monotonically; the residual grazing intensity is
    the physical Fresnel sheen, so the gate is strictly-below, not zero."""
    scene = tp.Scene()
    scene.background = 0x000000
    mat = tp.MeshStandardMaterial()
    mat.color = 0xffffff
    mat.roughness = 0.5
    panel = tp.Mesh(tp.PlaneGeometry(4.0, 4.0), mat)
    scene.add(panel)
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(0, 0, 8)
    cam.look_at(0, 0, 0)

    lid = tp.PathTracedLidarSensor(10.0, 9, 9, 50.0)   # narrow pinhole grid
    lid.position.set(0, 0, 5.0)
    lid.params.detector_threshold = 0.0                # raw curve, no gating
    curve = []
    for deg in range(0, 90, 5):
        panel.rotation.y = math.radians(deg)
        vk_renderer.render(scene, cam)
        d = lid.scan(vk_renderer)
        hit = d["return_no"] > 0
        curve.append(float(d["intensity"][hit].mean()) if hit.any() else 0.0)
    tail = curve[11:]                                  # 55..85 deg
    assert all(tail[i + 1] <= tail[i] * 1.05 for i in range(len(tail) - 1)), \
        f"intensity not monotone past 55 deg: {tail}"
    assert curve[-1] < 0.5 * curve[11], \
        f"grazing not subdued vs 55 deg: {curve[-1]} vs {curve[11]}"


def test_event_camera_visualisation(vk_renderer):
    """The GPU DVS detector: pinned sensor resolution, a mono accumulator of
    that shape, and a moving edge that actually fires events."""
    scene, cam = make_scene()
    box = scene.children[0]
    vk_renderer.set_event_camera_params(threshold=0.15, decay=0.85)
    vk_renderer.set_event_camera_resolution(64, 48)
    vk_renderer.event_camera_enabled = True
    try:
        sw, sh = vk_renderer.event_camera_resolution
        assert (sw, sh) == (64, 48)
        viz = None
        for i in range(8):                      # ring latency + reference settle
            box.rotation.y = 0.35 * i
            vk_renderer.render(scene, cam)
            viz = vk_renderer.read_event_camera_visualisation()
        assert viz.shape == (sh, sw) and str(viz.dtype) == "uint8"
        assert int(viz.max()) > 128 or int(viz.min()) < 128, \
            "a spinning box fired no events (accumulator is flat mid-grey)"
    finally:
        vk_renderer.event_camera_enabled = False


def test_event_camera_source_final(vk_renderer):
    """The 'final' source: the detector reads the presented frame (box-averaged
    to the sensor), so a STATIC scene settles to ~nothing once TAA converges,
    a moving edge still fires, and the switch itself is not a burst."""
    scene, cam = make_scene()
    box = scene.children[0]
    assert vk_renderer.event_camera_source == "shaded"
    vk_renderer.event_camera_source = "final"
    vk_renderer.set_event_camera_params(threshold=0.15, decay=0.85)
    vk_renderer.set_event_camera_resolution(64, 48)
    vk_renderer.event_camera_enabled = True
    try:
        assert vk_renderer.event_camera_source == "final"
        for _ in range(12):                     # reference latch + TAA settle
            vk_renderer.render(scene, cam)
        quiet = 0
        for _ in range(6):                      # static: the picture converged
            vk_renderer.render(scene, cam)
            ev, _ = vk_renderer.read_event_stream()
            quiet += len(ev)
        assert quiet <= 6 * 64 * 48 * 0.01, \
            f"static scene fired {quiet} events over 6 frames on the final source"
        moving = 0
        for i in range(6):                      # a spinning box: edges fire
            box.rotation.y = 0.35 * (i + 1)
            vk_renderer.render(scene, cam)
            ev, _ = vk_renderer.read_event_stream()
            moving += len(ev)
        assert moving > quiet, "a spinning box fired no events on the final source"
        with pytest.raises(ValueError):
            vk_renderer.event_camera_source = "beauty"
    finally:
        vk_renderer.event_camera_enabled = False
        vk_renderer.event_camera_source = "shaded"


def test_event_camera_subframe_timestamps(vk_renderer):
    """ESIM sub-frame stamps. With the microsecond clock driven every frame the
    detector places each threshold crossing where the log-intensity ramp between
    the previous sample and this one crosses its level, so a frame's events
    spread across the interval instead of piling onto one value -- and the
    packet comes back in time order."""
    scene, cam = make_scene()
    box = scene.children[0]
    vk_renderer.set_event_camera_resolution(64, 48)
    vk_renderer.event_camera_enabled = True
    dt_us = 16667                               # a 60 fps sim clock
    latest = 0
    try:
        stamps = []
        for i in range(16):
            latest = i * dt_us
            vk_renderer.set_event_camera_params(threshold=0.15, decay=0.85,
                                                frame_time_us=latest)
            box.rotation.y = 0.35 * i
            vk_renderer.render(scene, cam)
            if i < 8:                           # ring latency + reference settle
                continue
            ev, _ = vk_renderer.read_event_stream(max_events=200000)
            t = ev[:, 3]
            assert np.all(np.diff(t) >= 0), \
                "a read came back out of time order (the packet must be sorted)"
            stamps.append(t)
        t_all = np.concatenate(stamps)
        assert t_all.size and np.any(t_all % dt_us != 0), (
            f"{t_all.size} events, none of them stamped between two frame "
            "boundaries -- the sub-frame interpolation is not running")
        # Two frames of ring latency mean a read belongs to an EARLIER frame, so
        # the only bound worth asserting is that nothing ran ahead of the clock.
        assert int(t_all.max()) <= latest, \
            f"a stamp ({int(t_all.max())}) is ahead of the newest clock ({latest})"
    finally:
        vk_renderer.event_camera_enabled = False


# ── Zero-copy frames out (VulkanRenderer.enable_frame_interop) ───────────────
# Local-verification only, like the vertex-interop test above: no CI job has
# both Vulkan and CUDA, so these skip everywhere but a machine with an NVIDIA
# GPU and a CUDA torch.

def _cuda_torch():
    """torch, if it is installed AND has a CUDA device; else None."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    return torch if torch.cuda.is_available() else None


needs_cuda_torch = pytest.mark.skipif(_cuda_torch() is None,
                                      reason="needs torch with a CUDA device")

# A module-scoped canvas of its own so the interop fixture's exported frames
# are independent of the AOV tests' render cadence. Width must clear the
# Windows hidden-window minimum (~232 px, see W/H above) so the swapchain
# extent matches what was asked for and the exports are FW x FH exactly.
FW, FH = 320, 240


@pytest.fixture(scope="module")
def vk_frames():
    canvas = tp.Canvas("vk-frames", width=FW, height=FH, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    scene = tp.Scene()
    scene.background = 0x202830
    mat = tp.MeshStandardMaterial()
    mat.color = 0xff5533
    box = tp.Mesh(tp.BoxGeometry(), mat)
    scene.add(box)
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    sun = tp.DirectionalLight(0xffffff, 3.0)
    sun.position.set(3, 5, 2)
    scene.add(sun)
    cam = tp.PerspectiveCamera(55, FW / FH, 0.1, 100)
    cam.position.set(1.5, 1.5, 3.0)
    cam.look_at(0, 0, 0)
    renderer.render(scene, cam)          # the exports are sized from a real frame
    return renderer, scene, cam, box


def _host_aov(renderer, name, view):
    """One AOV of the last rendered frame, reinterpreted from its raw bytes."""
    raw = renderer.read_gbuffer_aov_raw(name, view)
    assert raw is not None, f"host readback of {name!r} failed"
    h, w = raw.shape[0], raw.shape[1]
    if name == "depth":
        return raw.view(np.float32).reshape(h, w)
    if name == "ids":
        return raw.view(np.uint32).reshape(h, w, 4)
    return raw.reshape(h, w, 4)          # albedo: RGBA8, already bytes


@needs_cuda_torch
@pytest.mark.parametrize("secondary", [False, True])
def test_frame_interop_is_bit_exact_with_the_host_readback(vk_frames, secondary):
    """The acceptance test: the same frame, read two ways, byte for byte.

    The zero-copy path copies the same images read_gbuffer_aov copies, so this
    is exact equality and not allclose -- any difference at all would mean the
    copy took the wrong slot, the wrong layout or the wrong stride.
    """
    from threepp.torch_frames import FrameTensors

    renderer, scene, cam, _ = vk_frames
    view = 0
    if secondary:
        cam2 = tp.PerspectiveCamera(60, 1.0, 0.1, 100)
        cam2.position.set(-2, 1, 2)
        cam2.look_at(0, 0, 0)
        view = renderer.add_view(cam2, 160, 160)
        assert view > 0
        renderer.render(scene, cam)      # the view's own G-buffer exists now

    channels = ("depth", "ids", "albedo")
    frames = FrameTensors(renderer, view, channels)
    try:
        assert set(frames.channels) == set(channels)
        renderer.render(scene, cam)
        assert frames.sync()
        # Cloned before the host readback, which is what a real consumer does
        # with a single-buffered export.
        got = {c: frames[c].clone().cpu().numpy() for c in channels}
        for c in channels:
            host = _host_aov(renderer, c, view)
            mine = got[c].astype(np.uint32) if c == "ids" else got[c]
            assert mine.shape == host.shape, c
            assert np.array_equal(mine, host), f"{c} differs from the host readback"
        # The frame really had content in it -- an all-zero comparison would
        # pass the equality above and prove nothing.
        assert int(got["ids"].astype(np.uint32).max()) > 0, "no geometry in the frame"
    finally:
        frames.close()
        if secondary:
            renderer.remove_view(view)


@needs_cuda_torch
def test_frame_interop_color_is_the_scene_capture(vk_frames):
    """The Color channel is the post-TAA, pre-overlay picture -- exactly the one
    read_scene_pixels() returns, in the swapchain's own byte order. Recorded at
    the same point in the frame, so this is byte equality, and it is what pins
    the recording site: composited overlays or a picture-in-picture view would
    show up here and nowhere else."""
    from threepp.torch_frames import FrameTensors

    renderer, scene, cam, _ = vk_frames
    renderer.scene_capture = True
    frames = FrameTensors(renderer, 0, ("color",))
    try:
        renderer.render(scene, cam)
        frames.sync()
        got = frames.color.clone().cpu().numpy()
        host = renderer.read_scene_pixels()          # (H, W, 3), RGB
        assert got.shape[:2] == host.shape[:2]
        rgb = got[..., [2, 1, 0]] if frames.bgra else got[..., :3]
        assert np.array_equal(rgb, host)
    finally:
        frames.close()
        renderer.scene_capture = False


@needs_cuda_torch
def test_frame_interop_tensors_are_live_and_untorn(vk_frames):
    """The tensors track the renderer, frame after frame, with no tearing.

    UNTORN is the sharp half: every frame, the tensor equals that frame's host
    readback exactly. A copy that raced the frame it was recorded in would show
    up here as a partial image, intermittently, over several frames.

    LIVE is the other half: the ids change when the box turns. The stability
    check is a THRESHOLD, not equality -- the raster is TAA-jittered, so a
    static scene still flips the silhouette pixels between frames, and the ids
    attachment inherits that. What separates "static" from "moved" is how MANY
    pixels move: a jitter fringe is a percent or two, a rotating box is tens.
    """
    from threepp.torch_frames import FrameTensors

    torch = _cuda_torch()
    renderer, scene, cam, box = vk_frames
    frames = FrameTensors(renderer, 0, ("depth", "ids"))
    try:
        renderer.render(scene, cam)
        frames.sync()
        assert frames.depth.device.type == "cuda"

        def changed_fraction(a, b):
            return float((a != b).any(dim=-1).float().mean())

        prev = frames.ids.clone()
        static_max = 0.0
        for _ in range(3):               # static scene, static camera
            renderer.render(scene, cam)
            frames.sync()
            # Untorn: this frame's tensor IS this frame's host readback.
            host = _host_aov(renderer, "ids", 0)
            assert np.array_equal(frames.ids.clone().cpu().numpy().astype(np.uint32), host)
            static_max = max(static_max, changed_fraction(frames.ids, prev))
            prev = frames.ids.clone()
        assert static_max < 0.05, \
            f"{static_max:.1%} of ids pixels moved with nothing moving -- that is " \
            "more than the sub-pixel jitter fringe"

        before_depth = frames.depth.clone()
        box.rotation.y = 0.9             # now move something
        for _ in range(2):
            renderer.render(scene, cam)
            frames.sync()
        moved = changed_fraction(frames.ids, prev)
        assert moved > 4 * max(static_max, 1e-3), \
            f"the box turned and only {moved:.1%} of ids pixels changed -- the " \
            "tensor is not live"
        assert not torch.equal(frames.depth, before_depth)
    finally:
        box.rotation.y = 0.0
        frames.close()


@needs_cuda_torch
def test_frame_interop_is_disabled_by_a_reallocation(vk_frames, capfd):
    """render_scale reallocates the G-buffer, and the renderer disables
    interop rather than freeing images CUDA has imported. The consumer sees a
    warning, nothing crashes, and re-arming gives fresh (smaller) exports."""
    from threepp.torch_frames import FrameTensors

    renderer, scene, cam, _ = vk_frames
    frames = FrameTensors(renderer, 0, ("depth",))
    full_h, full_w = tuple(frames.depth.shape)
    try:
        renderer.render(scene, cam)
        frames.sync()
        assert not frames.stale
        capfd.readouterr()               # drop everything up to here
        renderer.render_scale = 0.5
        renderer.render(scene, cam)      # the deferred realloc lands here
        err = capfd.readouterr().err
        assert "frame interop disabled" in err, \
            "no invalidation warning was printed; stderr was:\n" + err
        # And the consumer sees it without having to read stderr.
        assert frames.stale and not renderer.frame_interop_active(0)
        # Still no crash: the fence wait and the teardown both behave.
        assert renderer.sync_frame_interop()
    finally:
        frames.close()

    # Re-arming after the reallocation works, and reports the NEW extent.
    again = FrameTensors(renderer, 0, ("depth",))
    try:
        h, w = tuple(again.depth.shape)
        assert h < full_h and w < full_w, (h, w, full_h, full_w)
        renderer.render(scene, cam)
        assert again.sync()
    finally:
        again.close()
        renderer.render_scale = 1.0
        renderer.render(scene, cam)


@needs_cuda_torch
def test_frame_interop_rejects_a_stale_view(vk_frames):
    """A handle that never existed exports nothing -- the same rule
    read_view_gbuffer_aov follows, for the same reason: a frame attributed to
    the wrong camera is worse than no frame."""
    from threepp.torch_frames import FrameInteropUnavailable, FrameTensors

    renderer, _, _, _ = vk_frames
    assert renderer.enable_frame_interop(9999, ["depth"]) == []
    with pytest.raises(FrameInteropUnavailable):
        FrameTensors(renderer, 9999, ("depth",))
    with pytest.raises(ValueError):
        FrameTensors(renderer, 0, ("not_a_channel",))
