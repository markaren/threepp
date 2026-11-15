"""The library's log messages from Python: thresholds and capture (tp.set_log_level, tp.capture_log)."""
import pytest

import threepp as tp


@pytest.fixture
def captured():
    tp.take_log()
    tp.capture_log()
    yield
    tp.capture_log(False)
    tp.set_log_level("*", tp.LogLevel.Info)
    tp.take_log()


def _add_to_itself():
    # Object3D.add refuses to parent an object under itself and says so: a
    # core warning that needs no renderer.
    node = tp.Object3D()
    node.name = "loop"
    node.add(node)
    assert node.parent is None


def test_capture_takes_the_message_off_stderr(captured, capfd):
    _add_to_itself()
    records = tp.take_log()
    assert len(records) == 1
    r = records[0]
    assert r.level == tp.LogLevel.Warn
    assert r.category == "core.object"
    assert "loop" in r.message
    assert "core.object" in repr(r)
    assert capfd.readouterr().err == ""
    assert tp.take_log() == []


def test_echo_keeps_the_stderr_line(capfd):
    tp.take_log()
    tp.capture_log(True, echo=True)
    try:
        _add_to_itself()
    finally:
        tp.capture_log(False)
    assert len(tp.take_log()) == 1
    assert "[core.object] warning:" in capfd.readouterr().err


def test_without_capture_the_message_goes_to_stderr(capfd):
    tp.take_log()
    _add_to_itself()
    assert "[core.object] warning:" in capfd.readouterr().err
    assert tp.take_log() == []


def test_a_threshold_silences_a_category_and_what_is_below_it(captured):
    tp.set_log_level("core", tp.LogLevel.Off)
    _add_to_itself()
    assert tp.take_log() == []
    tp.set_log_level("core.object", tp.LogLevel.Warn)
    _add_to_itself()
    assert len(tp.take_log()) == 1


def test_levels_spec(captured):
    assert tp.set_log_levels("core.object:error,vulkan:debug")
    _add_to_itself()
    assert tp.take_log() == []
    assert not tp.set_log_levels("core.object:loud")
