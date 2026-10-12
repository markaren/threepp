"""threepp.capture: the standard-library PNG round trip, the golden-test metric and its
gate, a GL still on the session renderer, and a Vulkan still with its pass timings and
text report (the Vulkan parts skip on a GL-only build)."""
import numpy as np
import pytest

import threepp as tp
from threepp import capture


def _scene(aspect):
    scene = tp.Scene()
    scene.background = 0x202830
    mat = tp.MeshStandardMaterial()
    mat.color = 0xff5533
    scene.add(tp.Mesh(tp.BoxGeometry(), mat))
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    sun = tp.DirectionalLight(0xffffff, 3.0)
    sun.position.set(3, 5, 2)
    scene.add(sun)
    cam = tp.PerspectiveCamera(55, aspect, 0.1, 100)
    cam.position.set(1.5, 1.5, 3.0)
    cam.look_at(0, 0, 0)
    return scene, cam


def test_png_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    capture.write_png(tmp_path / "a.png", rgb)
    capture.write_png(tmp_path / "b.png", rgba)
    assert np.array_equal(capture.read_png(tmp_path / "a.png"), rgb)
    assert np.array_equal(capture.read_png(tmp_path / "b.png"), rgba)


def test_read_png_undoes_adaptive_filters(tmp_path):
    """A PNG written by another encoder uses Sub/Up/Average/Paeth rows; the reader must
    undo all of them, not just the filter-0 rows our own writer emits."""
    Image = pytest.importorskip("PIL.Image")
    y, x = np.mgrid[0:64, 0:80]
    rgb = np.stack([x * 3 % 256, y * 4 % 256, (x + y) % 256], axis=2).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "pil.png", optimize=True)
    assert np.array_equal(capture.read_png(tmp_path / "pil.png"), rgb)


def test_compare_metric_and_gate():
    a = np.full((20, 30, 3), 100, np.uint8)
    d = capture.compare(a, a.copy())
    assert d.identical and d.psnr == 99.0 and d.max_delta == 0 and d.hot_pct == 0.0 and d.passes()

    b = a.copy()
    b[0, 0, 0] = 200                      # one hot channel
    d = capture.compare(a, b)
    assert d.max_delta == 100 and not d.identical
    assert d.hot_pct == pytest.approx(100.0 / (20 * 30 * 3))
    assert d.psnr == pytest.approx(10 * np.log10(255.0 ** 2 / (100.0 ** 2 / (20 * 30 * 3))))
    assert not d.passes()                 # maxD 100 > 32 even though PSNR is high
    assert d.passes(max_delta=100)

    assert not capture.compare(a, np.zeros((21, 30, 3), np.uint8)).passes()
    assert "mismatch" in str(capture.compare(a, np.zeros((21, 30, 3), np.uint8)))

    # RGBA against RGB compares the first three channels; a float image is quantised
    assert capture.compare(np.dstack([a, a[..., :1]]), a / 255.0).identical


def test_make_renderer_rejects_a_width_the_window_manager_would_widen():
    with pytest.raises(ValueError):
        capture.make_renderer("gl", (128, 96))


def test_gl_still_on_the_session_renderer(renderer, tmp_path):
    scene, cam = _scene(1.0)             # the wrong aspect: render() fixes it to the canvas
    frame = capture.render(scene, cam, backend=renderer, frames=1, path=tmp_path / "gl.png")
    assert frame.backend == "gl" and frame.image.shape == (150, 200, 3)
    assert cam.aspect == pytest.approx(200 / 150)
    assert not frame.stats()["flat"]
    assert (tmp_path / "gl.png").exists()
    assert capture.compare(frame, tmp_path / "gl.png").identical
    assert "no GPU timings" in frame.report()
    assert "Frame(gl, 200x150, 1 frames)" in str(frame)


@pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")
def test_vulkan_still_timings_and_report(tmp_path):
    W, H = 320, 240
    scene, cam = _scene(W / H)
    frame = capture.render(scene, cam, backend="vulkan", size=(W, H), frames=8,
                           keep_timings=True, path=tmp_path / "vk.png")
    assert frame.backend == "vulkan" and frame.image.shape == (H, W, 3)
    assert len(frame.per_frame) == 8 and "gpu_total_ms" in frame.frame_timings
    assert not frame.stats()["flat"]
    text = frame.report()
    assert "gpu_total=" in text
    if frame.pass_timings:                # a device without timestamps has no rows
        assert "passes (" in text and "graph.entry" in {r[0] for r in frame.pass_timings}

    # a second render of the same scene sits within the golden gate
    again = capture.render(scene, cam, backend="vulkan", size=(W, H), frames=8)
    d = capture.compare(frame, again)
    assert d.passes(), str(d)
