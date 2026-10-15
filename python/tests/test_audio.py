"""Audio bindings: the reverb bus, the reverb send, occlusion and the playback rate.

Nothing here plays a sound: the calls set parameters on stopped sounds, at master
volume 0. Skipped when the module was built without audio.
"""
import numpy as np
import pytest
import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_AUDIO, reason="built without audio")


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.arange(4410) / 44100.0
    tp.write_wav(str(path), list(0.1 * np.sin(2 * np.pi * 440.0 * t)), 44100)
    return str(path)


@pytest.fixture
def listener():
    try:
        lis = tp.AudioListener()
    except RuntimeError as e:  # no audio device on this machine
        pytest.skip(f"no audio device: {e}")
    lis.master_volume = 0.0
    return lis


def test_listener_reverb_is_bound(listener):
    listener.set_reverb(1.2, 0.3)
    listener.set_reverb(rt60=0.4, wet=0.0)


def test_audio_reverb_send_and_playback_rate(listener, wav):
    a = tp.Audio(listener, wav)
    a.set_reverb_send(0.5)
    a.set_playback_rate(1.5)
    a.set_playback_rate(0.0)  # ignored, not an error
    assert not a.is_playing


def test_positional_audio_reverb_send_and_occlusion(listener, wav):
    p = tp.PositionalAudio(listener, wav)
    p.set_reverb_send(send=0.25)
    p.set_occlusion(0.7)
    p.set_playback_rate(0.8)
    assert not p.is_playing
