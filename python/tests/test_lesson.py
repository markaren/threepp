"""threepp.lesson, the experimental explainer-video toolkit: its clocks, its file writers,
and one headless frame through Stage and Hud on the session renderer."""
import json
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
        st = lesson.Stage(200, 150, renderer=renderer, design=(200, 150))
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
    finally:
        renderer.tone_mapping, renderer.tone_mapping_exposure, renderer.shadow_map_enabled = kept
