import math

import threepp as tp


def test_animation_enums_exist():
    assert tp.Loop.ONCE != tp.Loop.REPEAT
    assert tp.AnimationBlendMode.NORMAL != tp.AnimationBlendMode.ADDITIVE
    assert tp.Interpolation.LINEAR != tp.Interpolation.DISCRETE


def test_keyframe_track_properties():
    track = tp.VectorKeyframeTrack(".position", [0.0, 1.0], [0, 0, 0, 0, 5, 0])
    assert track.name == ".position"
    assert track.value_type_name == "vector"
    assert list(track.times) == [0.0, 1.0]
    assert len(track.values) == 6


def test_animation_clip_construction():
    track = tp.VectorKeyframeTrack(".position", [0.0, 2.0], [0, 0, 0, 0, 5, 0])
    clip = tp.AnimationClip("bounce", 2.0, [track])
    assert clip.name == "bounce"
    assert clip.duration == 2.0
    assert clip.uuid  # non-empty
    # find_by_name resolves out of a list, three.js-style
    assert tp.AnimationClip.find_by_name([clip], "bounce") is not None
    assert tp.AnimationClip.find_by_name([clip], "nope") is None


def _bounce_setup():
    """A cube whose .position.y is keyed 0 -> 5 -> 0 over 2 seconds."""
    cube = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    track = tp.VectorKeyframeTrack(".position", [0.0, 1.0, 2.0],
                                   [0, 0, 0, 0, 5, 0, 0, 0, 0])
    clip = tp.AnimationClip("bounce", 2.0, [track])
    mixer = tp.AnimationMixer(cube)
    return cube, clip, mixer


def test_mixer_drives_position_over_time():
    cube, clip, mixer = _bounce_setup()
    action = mixer.clip_action(clip)
    action.play()
    assert action.is_running()

    # t=0 -> y≈0
    mixer.update(0.0)
    assert abs(cube.position.y) < 1e-3

    # advance to t=1.0 -> y≈5 (mid keyframe)
    mixer.update(1.0)
    assert cube.position.y > 4.5

    # advance to t=2.0 -> back to y≈0
    mixer.update(1.0)
    assert cube.position.y < 0.5


def test_clip_action_is_cached():
    _, clip, mixer = _bounce_setup()
    a = mixer.clip_action(clip)
    b = mixer.clip_action(clip)
    # Same clip on the same root resolves to the same underlying C++ action
    # (the mixer caches it), so state set through one is visible through the other.
    a.play()
    assert b.is_running()


def test_fluent_api_chains():
    _, clip, mixer = _bounce_setup()
    # set_loop / play return the action so calls chain, three.js-style.
    action = mixer.clip_action(clip).set_loop(tp.Loop.ONCE, 1)
    assert action.play() is action
    assert action.is_running()


def test_time_scale_slows_playback():
    cube, clip, mixer = _bounce_setup()
    mixer.time_scale = 0.5
    mixer.clip_action(clip).play()
    # With half speed, after 1s of wall time we are at clip-time 0.5 -> y≈2.5
    mixer.update(1.0)
    assert 2.0 < cube.position.y < 3.0


def test_quaternion_track_animates_rotation():
    cube = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    # keyframe 0: identity; keyframe 1: 180° about Y -> (0, 1, 0, 0)
    track = tp.QuaternionKeyframeTrack(".quaternion", [0.0, 1.0],
                                       [0, 0, 0, 1, 0, 1, 0, 0])
    clip = tp.AnimationClip("spin", 1.0, [track])
    mixer = tp.AnimationMixer(cube)
    mixer.clip_action(clip).play()
    # Advance to the midpoint (not the full duration — default Repeat would wrap
    # t=1.0 back to t=0). Slerping identity -> 180°Y lands near 90°Y: y ≈ sin45.
    mixer.update(0.5)
    assert cube.quaternion.y > 0.6


def test_mixer_outlives_root_reference():
    # keep_alive must pin the root: drop the local name, force GC, still update.
    import gc
    _, clip, mixer = _bounce_setup()
    action = mixer.clip_action(clip)
    action.play()
    gc.collect()
    mixer.update(1.0)  # must not crash on a freed root
    assert action.is_running()
