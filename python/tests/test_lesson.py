"""threepp.lesson, the experimental explainer-video toolkit: its clocks and cameras, the
measured-speech layout, the code tokenizer, the HUD's wrapping and its pool, the file writers
and Film, and one headless frame through Stage and Hud on the session renderer."""
import json
import math
import struct
import zlib

import numpy as np
import pytest

import threepp as tp
from threepp import lesson


def test_timing_and_speech(tmp_path):
    tl = lesson.Timeline().add("a", 0.0, 2.0).then("b", 3.0, gap=1.0)
    assert tl.span("b") == (3.0, 6.0) and tl.duration == 6.0
    assert tl.p("a", 1.0) == pytest.approx(0.5) and tl.which(4.0) == "b"

    k = lesson.Keys([(0.0, [0, 0, 0]), (2.0, [2, 4, 6])])
    assert np.allclose(k(1.0), [1, 2, 3]) and np.allclose(k(-1.0), [0, 0, 0]) and np.allclose(k(9.0), [2, 4, 6])

    # the film clock: a hold stops the script clock for d seconds of film
    tm = lesson.TimeMap([(2.0, 1.5, "x"), (5.0, 0.5, "y")])
    assert tm.film(1.0) == 1.0 and tm.film(3.0) == 4.5 and tm.film(6.0) == 8.0
    assert tm.script(2.75) == (2.0, ("x", 0.5))
    for t in (0.3, 1.9, 2.5, 4.9, 5.4, 7.0):
        assert tm.script(tm.film(t)) == (pytest.approx(t), None)

    # captions laid out from measured speech: [lead-in, a, gap, b, tail], then [c, tail]
    cache = tmp_path / "demo.speech.json"
    cache.write_text(json.dumps({"af_heart": {"a": {"s": 2.0, "said": "Alpha."}, "b": {"s": 1.0, "said": "Beta."},
                                              "c": {"s": 3.0, "said": "three p p runs at 60 frames per second."}}}))
    sp = lesson.Speech({"a": "Alpha.", "b": "Beta.", "c": "threepp runs at {n} fps.", "d": "four words, not measured"},
                       str(cache), voice_only={"b"})
    tl, cap = sp.layout([("one", 0.5, ["a", "b"], 1.0), ("two", 0.0, ["c"], 0.5)], gap=0.2, slack=0.9)
    assert cap["a"] == pytest.approx((0.5, 3.4)) and cap["b"] == pytest.approx((3.6, 5.5))
    assert cap["c"] == pytest.approx((6.5, 10.4))
    assert tl.span("one") == pytest.approx((0.0, 6.5)) and tl.span("two") == pytest.approx((6.5, 10.9))
    assert sp.seconds("d") == pytest.approx(0.42 * 4 + 0.4)          # unmeasured: estimated from its words
    caps = sp.captions(cap, fields={"n": 60})
    assert caps[1][3] is False and len(caps[0]) == 3                  # b is spoken, not drawn
    assert caps[2][2] == "threepp runs at 60 fps." and sp.stale() == []
    assert lesson.spoken("threepp runs at 60 fps.") == "three p p runs at 60 frames per second."
    with pytest.raises(TypeError):
        lesson.DEFAULT_WORDS["x"] = "y"                               # read-only: a lesson passes words=

    # a line longer than its caption: needs 5 s, gets 3.4 s plus a 0.5 s hold inside it
    extra = lesson.overruns([(0.0, 4.0), (10.0, 20.0)], [5.0, 1.0], holds=[(2.0, 0.5)])
    assert len(extra) == 1 and extra[0][:2] == (0, pytest.approx(4.0 - lesson.VOICE_TAIL))
    assert extra[0][2] == pytest.approx(5.0 - (4.0 - lesson.VOICE_TAIL - lesson.VOICE_LEAD + 0.5))


def test_timeline_redefines_a_beat_in_place():
    tl = lesson.Timeline().add("a", 0.0, 1.0).add("b", 1.0, 2.0).add("a", 0.0, 3.0)
    assert tl.order == ["a", "b"] and tl.span("a") == (0.0, 3.0) and tl.duration == 3.0


def test_speech_stretch(tmp_path):
    cache = tmp_path / "s.speech.json"
    cache.write_text(json.dumps({"af_heart": {"a": {"s": 3.0, "said": "Alpha."}, "b": {"s": 0.5, "said": "Beta."}}}))
    sp = lesson.Speech({"a": "Alpha.", "b": "Beta."}, str(cache))
    tm = sp.stretch({"a": (0.0, 2.0), "b": (2.0, 4.0)}, margin=0.3)
    # a needs 3.3 s of a 2 s caption: the film holds the shortfall where a's speech should end; b fits
    (t, d, tag), = tm.holds
    assert tag == ("speech", "a") and t == pytest.approx(2.0 - lesson.VOICE_TAIL)
    assert d == pytest.approx(3.3 - (2.0 - lesson.VOICE_LEAD - lesson.VOICE_TAIL))
    assert tm.film(1.0) == 1.0 and tm.film(2.0) == pytest.approx(2.0 + d)


def test_orbit_camera_frames():
    still = ((0.0, 0.0), (0.0, 0.0))
    keys = [(0.0, 0.0, 0.0, 2.0, [0, 0, 0], 40.0), (1.0, 90.0, 0.0, 2.0, [0, 0, 0], 40.0),
            (2.0, 0.0, 90.0, 2.0, [0, 0, 0], 40.0)]
    for frame in ("zup", "yup"):            # the same shots, whichever frame the look points are given in
        cam = lesson.OrbitCamera(keys, drift=still, frame=frame)
        eye, look, fov = cam(0.0)
        assert np.allclose(eye, [2, 0, 0]) and np.allclose(look, 0) and fov == 40.0
        assert np.allclose(cam(1.0)[0], [0, 0, -2])          # a quarter turn: Z-up +y is Y-up -z
        assert np.allclose(cam(2.0)[0], [0, 2, 0])           # straight above
    cam = lesson.OrbitCamera(keys[:1], drift=still, frame="yup", follow=lambda t: [t, 0, 0],
                             heading=lambda t: math.pi / 2)  # a chase camera
    eye, look, _ = cam(3.0)
    assert np.allclose(look, [3, 0, 0]) and np.allclose(eye, [3, 0, -2])


def test_set_pose_round_trip():
    def axis_angle(ax, ang):
        ax = np.asarray(ax, float) / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    def from_quat(q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    rng = np.random.default_rng(0)
    cases = [(a, math.pi) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3])]   # the 180-degree branches
    cases += [(rng.normal(size=3), rng.uniform(0, math.pi)) for _ in range(40)]
    o = tp.Object3D()
    for ax, ang in cases:
        M = np.eye(4)
        M[:3, :3] = axis_angle(ax, ang)
        M[:3, 3] = [0.1, -0.2, 0.3]
        lesson.set_pose(o, M)
        q = o.quaternion
        assert np.allclose(from_quat((q.x, q.y, q.z, q.w)), M[:3, :3], atol=1e-6)
        assert np.allclose([o.position.x, o.position.y, o.position.z], M[:3, 3])


def test_tokenize():
    col = lesson.CODE_COLOURS
    assert lesson.tokenize("#include <threepp/threepp.hpp>", "cpp") == [
        (0, "#include", col["pre"]), (9, "<threepp/threepp.hpp>", col["str"])]
    kinds = {s: c for _, s, c in lesson.tokenize("auto m = tp::Mesh(0x20, 1.5f); // a mesh", "cpp")}
    assert kinds["auto"] == col["kw"] and kinds["tp"] == col["mod"] and kinds["Mesh"] == col["type"]
    assert kinds["0x20"] == col["num"] and kinds["1.5f"] == col["num"] and kinds["// a mesh"] == col["com"]
    assert kinds["m"] == col["id"] and kinds["("] == col["punc"] and " " not in kinds
    kinds = {s: c for _, s, c in lesson.tokenize('x = foo("hi")  # go', "py")}
    assert kinds["foo"] == col["fn"] and kinds['"hi"'] == col["str"] and kinds["# go"] == col["com"]
    assert {s: c for _, s, c in lesson.tokenize("void f()", "cpp")}["void"] == col["id"]
    assert {s: c for _, s, c in lesson.tokenize("void f()", "cpp", lesson.KEYWORDS["cpp"] | {"void"})}["void"] == col["kw"]
    with pytest.raises(TypeError):
        lesson.CODE_COLOURS["kw"] = 0                                  # read-only: tokenize(palette=)


def test_hud_wrap_and_pool():
    ov = lesson.Hud(400, 300, pool_limit=8)
    text = "the quick brown fox jumps over the lazy dog"
    lines = ov.wrap(text, 20, 140)
    assert len(lines) > 1 and " ".join(lines) == text and all(ov.text_width(l, 20) <= 140 for l in lines)
    balanced = ov.wrap_balanced(text, 20, 140)
    assert len(balanced) == len(lines) and " ".join(balanced) == text
    assert max(ov.text_width(l, 20) for l in balanced) <= 141          # never wider than asked (lo < hi)
    # a string that changes every frame: the pool releases old ones at end(), never this frame's
    for f in range(30):
        ov.begin()
        ov.text(10, 10, f"{f:03d}", size=20)
        ov.end()
        assert ov._pool[("text", f"{f:03d}", 20, "regular")][0].visible
        assert sum(len(v) for v in ov._pool.values()) <= 8 and len(ov.scene.children) <= 8
    # live text draws per glyph: the ten digits serve every number, and lay out as the string does
    live = lesson.Hud(400, 300)
    for f in range(100):
        live.begin()
        live.text(10, 10, f"{f:04d}", size=20, live=True)
        live.end()
    assert len(live._pool) <= 10
    assert live.text_width("0123", 20) == pytest.approx(sum(live.text_width(c, 20) for c in "0123"), abs=1e-3)


def test_film(tmp_path, monkeypatch):
    from threepp.lesson import media
    monkeypatch.setattr(media, "ffmpeg_exe", lambda prefer="bundled": None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        lesson.Film(str(tmp_path / "none.mp4"), 32, 32)
    monkeypatch.undo()
    if media.ffmpeg_exe() is None:
        pytest.skip("no ffmpeg (imageio-ffmpeg or PATH)")
    frame = np.zeros((48, 64, 3), np.uint8)
    out = tmp_path / "a.mp4"
    with lesson.Film(str(out), 64, 48, fps=10, preset="ultrafast") as film:
        for k in range(5):
            frame[:, :, 0] = 50 * k
            film.write(frame)
    assert out.stat().st_size > 0 and not (tmp_path / "a.mp4.part.mp4").exists()
    with pytest.raises(ZeroDivisionError):
        with lesson.Film(str(tmp_path / "b.mp4"), 64, 48, fps=10, preset="ultrafast") as film:
            film.write(frame)
            1 / 0
    assert not (tmp_path / "b.mp4").exists() and not (tmp_path / "b.mp4.part.mp4").exists()


def test_media(tmp_path):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    lesson.write_png(str(p), img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + body) & 0xFFFFFFFF
        chunks[tag] = body
        pos += 12 + n
    w, h = struct.unpack(">II", chunks[b"IHDR"][:8])
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + w * 3)
    assert (w, h) == (5, 7) and not raw[:, 0].any()                   # filter type 0 on every row
    assert np.array_equal(raw[:, 1:].reshape(h, w, 3), img)

    s = tmp_path / "a.srt"
    lesson.write_srt(str(s), [(0.5, 2.25, "Hello."), (61.0, 3725.5, "Spoken, not drawn.", False)])
    assert s.read_text(encoding="utf-8") == ("1\n00:00:00,500 --> 00:00:02,250\nHello.\n\n"
                                             "2\n00:01:01,000 --> 01:02:05,500\nSpoken, not drawn.\n\n")


def test_stage_and_hud_render(renderer):
    kept = (renderer.tone_mapping, renderer.tone_mapping_exposure, renderer.shadow_map_enabled)
    try:
        st = lesson.Stage(200, 150, renderer=renderer)
        assert st.design == (200, 150)                                 # the Hud below is built at this size
        ball = tp.Mesh(tp.SphereGeometry(0.12, 24, 16), lesson.standard(0xff2020, emissive=0xff2020,
                                                                        emissive_intensity=2.0))
        ball.position.set(0.35, 0.6, 0.0)
        st.scene.add(ball)
        st.look([0.0, 1.0, 3.0], [0.0, 0.5, 0.0], fov=40)
        bare = st.frame(0.0).astype(int)
        assert bare.shape == (150, 200, 3)

        x, y, depth = st.project([0.35, 0.6, 0.0])
        assert depth > 0 and 0 <= x < 200 and 0 <= y < 150
        r, g, b = bare[int(y), int(x)]
        assert r > 150 and r > g + 60 and r > b + 60, f"the ball is not at its projected pixel: {(r, g, b)}"

        ov = lesson.Hud(200, 150)
        ov.begin()
        ov.panel(10, 10, 60, 30, radius=4, fill=0x20ff20, alpha=1.0)
        ov.text(20, 110, "Hi", size=24, color=0xffffff)
        ov.end()
        hud = st.frame(0.0, hud=ov).astype(int)
        assert hud[25, 40, 1] > 200 and abs(hud[25, 40] - bare[25, 40]).max() > 100      # the panel
        assert abs(hud[100:140, 15:60] - bare[100:140, 15:60]).max() > 100                 # the text
        assert np.array_equal(hud[140:, 180:], bare[140:, 180:])                           # the far corner
        assert renderer.auto_clear                                                         # as it was
    finally:
        renderer.tone_mapping, renderer.tone_mapping_exposure, renderer.shadow_map_enabled = kept
