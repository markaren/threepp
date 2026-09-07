// The coroutine scheduler behind threepp.editor.start_coroutine.
//
// A script method has to return before the frame can continue, so anything that
// takes TIME — "lower the arm, wait half a second, open the gripper, drive to
// the next marker" — has to be written as a state machine over update(), with
// the interesting part (the sequence) spread across an enum and the boring part
// (the bookkeeping) repeated in every script that does it. A coroutine is the
// same sequence written as a sequence: a generator that yields a CONDITION, and
// a scheduler that resumes it once per frame when the condition is met.
//
// Why the scheduler is Python rather than C++
// -------------------------------------------
// It is sixty lines of generator plumbing — send(), close(), StopIteration.value,
// a stack of nested generators — every one of which is one line here and five
// error-prone lines through the C API. It is exec'd into the threepp.editor
// submodule while the embedded module's body runs, so the names exist from the
// first `import threepp` and there is no ordering to get right later.
//
// C++ owns exactly three things, and no more:
//
//   * start_coroutine itself, because it is the one that must REFUSE. A task
//     belongs to the instance that is being dispatched (see
//     scripting::dispatchingScript), and a generator handed over with nobody
//     being dispatched has no owner to disable when it raises;
//   * the task COUNT, mirrored on this side so ScriptPlaySession::update can ask
//     "is there anything to pump" before it acquires the GIL — the one question
//     that decides whether a frame pays for Python at all;
//   * the three pump/drop/clear hooks, which return their failures rather than
//     raising, so a broken coroutine takes the ordinary one-report one-disable
//     path every other script error takes.
//
// Where the pump sits in a frame is a decision, not an accident: AFTER the
// update() sweep, inside the same GIL acquisition. So an until() predicate reads
// a world that has been stepped by physics and then had every update() run on
// it — the settled state of the frame, not a half-built one. Documented in
// doc/editor.md, tested in EditorScriptCoroutine_test.

#include "ScriptTasks.hpp"

#include "Scripting.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    namespace py = pybind11;

    // The scheduler. Kept as one string for the reason ScriptHost.cpp's helpers
    // are: there is no file to find at runtime, and the editor must work from
    // any working directory.
    constexpr const char* kTaskSource = R"PY(
_tasks = []

# True while a task is being wound UP - _shed running its finally: blocks. The
# attribution window is open there on purpose, so a finally: that raises is
# reported against its owner; the same open window would let a finally: START
# work, and both teardown paths get that wrong in their own way (_clear snapshots
# before shedding, so the newcomer survives into the NEXT session; _drop_for
# rebuilds the list after, so it vanishes silently). One rule instead of two
# accidents: a finally: may tidy, not begin - _start refuses while this is set.
_shedding = False


def _describe():
    """The exception being handled, as a traceback the console can print.

    Imported lazily on purpose: this module body is exec'd while `threepp`
    itself is still being imported, and there is no reason to drag another
    import into that window for a path that only runs when a script breaks.
    """
    import traceback
    return traceback.format_exc()


class _Wait:
    """What wait() yields. A value, not a timer: the deadline is computed when
    the scheduler sees it, against the clock of that pump."""

    __slots__ = ("seconds",)

    def __init__(self, seconds):
        self.seconds = float(seconds)


class _Until:
    """What until() yields: a predicate the scheduler polls once per pump."""

    __slots__ = ("predicate",)

    def __init__(self, predicate):
        if not callable(predicate):
            raise TypeError(
                "threepp.editor.until wants something to CALL - "
                "until(lambda: self.landed), not until(self.landed). Got %s."
                % (type(predicate).__name__,))
        self.predicate = predicate


def wait(seconds):
    """Yield this to suspend for `seconds` of SIMULATED time.

    `yield threepp.editor.wait(0.5)` resumes at the first pump where
    threepp.editor.time.sim_time has advanced half a second past the moment the
    yield was seen. Simulated, not wall: a mission written this way freezes when
    physics is paused or starved, and takes the same number of substeps on a
    machine that renders half as fast. Without a physics world sim_time degrades
    to wall time, so this still measures something honest there.
    """
    return _Wait(seconds)


def until(predicate):
    """Yield this to suspend until `predicate()` is truthy.

    The truthy value is SENT BACK IN, so `hit = yield
    threepp.editor.until(lambda: threepp.editor.raycast(o, d, 2.0))` both waits
    for the hit and hands you the hit. Polled once per frame, after physics has
    stepped and after every script's update() has run.
    """
    return _Until(predicate)


class Task:
    """A running coroutine, as start_coroutine hands it back.

    Keep it to cancel(), or to ask whether it is done. Dropping it does NOT stop
    the coroutine - the scheduler holds it until it finishes, its owning script
    is disabled, or Play stops.
    """

    __slots__ = ("_owner", "_stack", "_done", "_mode", "_deadline", "_predicate")

    def __init__(self, generator, owner):
        self._owner = owner
        # Innermost generator last. A yielded generator is pushed here rather
        # than driven by a second Task, so `yield child()` costs one frame of
        # nothing and the parent's resume value is the child's return value.
        self._stack = [generator]
        self._done = False
        # "frame" before anything has run at all: the first pump enters the body
        # and runs it up to its first yield. Nothing of a coroutine executes
        # inside start_coroutine, which is what keeps a task registered from
        # fixed_update (mid-substep) from running frame-paced code there.
        self._mode = "frame"
        self._deadline = 0.0
        self._predicate = None

    @property
    def done(self):
        """True once the coroutine returned, raised, or was cancelled."""
        return self._done

    def cancel(self):
        """Stop the coroutine now. Not an error, and harmless if already done.

        The generator is close()d, so a `finally:` inside it runs - which is
        where anything a half-finished mission has to put back belongs.
        """
        if self._done:
            return
        self._close()

    # --- internals, driven by the pump -------------------------------------

    def _close(self):
        self._done = True
        self._predicate = None
        stack, self._stack = self._stack, []
        # Innermost first: a child's finally: runs before its parent's, the
        # order they would have unwound in had the code simply returned.
        for generator in reversed(stack):
            generator.close()

    def _step(self, now):
        mode = self._mode
        if mode == "wait":
            if now < self._deadline:
                return
            send = None
        elif mode == "until":
            hit = self._predicate()
            if not hit:
                return
            # The truthy value is the yield's result: the predicate that found
            # the thing hands the thing over.
            send = hit
        else:
            send = None
        self._advance(send, now)

    def _advance(self, send, now):
        # Runs until the task is actually SUSPENDED (or finished): pushing a
        # nested generator and popping a finished one both continue the same
        # pump, so `yield child()` does not cost a frame of its own and neither
        # does the child returning.
        while self._stack:
            generator = self._stack[-1]
            try:
                yielded = generator.send(send)
            except StopIteration as stop:
                self._stack.pop()
                # A child's `return value` is what the parent's yield evaluates
                # to; None for a bare return, which is what it means anyway.
                send = stop.value
                continue
            if yielded is None:
                self._mode = "frame"
                return
            if isinstance(yielded, _Wait):
                self._mode = "wait"
                self._deadline = now + yielded.seconds
                return
            if isinstance(yielded, _Until):
                self._mode = "until"
                self._predicate = yielded.predicate
                return
            if hasattr(yielded, "send") and hasattr(yielded, "close"):
                self._stack.append(yielded)
                send = None
                continue
            raise TypeError(
                "a coroutine may yield nothing (resume next frame), "
                "threepp.editor.wait(seconds), threepp.editor.until(predicate), "
                "or another generator to run to completion - got %s."
                % (type(yielded).__name__,))
        self._done = True


def _start(generator, owner):
    """Register `generator` as a task owned by the script instance `owner`."""
    if _shedding:
        raise RuntimeError(
            "threepp.editor.start_coroutine: this coroutine is being wound up - "
            "its owner was disabled or Play is stopping - and a finally: block "
            "may tidy up, not start new work.")
    if not (hasattr(generator, "send") and hasattr(generator, "throw")
            and hasattr(generator, "close")):
        raise TypeError(
            "threepp.editor.start_coroutine wants a generator - CALL the "
            "generator function and pass the result: start_coroutine(self.run()), "
            "not start_coroutine(self.run). Got %s."
            % (type(generator).__name__,))
    task = Task(generator, owner)
    _tasks.append(task)
    return task


def _shed(task, failures):
    """Close `task`, attributing anything its finally: blocks raise."""
    global _shedding
    _attribute(task._owner)
    _shedding = True
    try:
        task._close()
    except BaseException:
        failures.append((task._owner, _describe()))
        # Already reported; do not let a raising finally leave a live task
        # behind for the next pump to trip over.
        task._done = True
        task._stack = []
    finally:
        _shedding = False
        _attribute("")


def _pump(now):
    """Advance every registered task once. Returns (failures, remaining)."""
    failures = []
    reap = False
    # A snapshot: a coroutine may start another task, and that one belongs to
    # the NEXT pump rather than to the middle of this one's iteration.
    for task in list(_tasks):
        if task._done:
            reap = True
            continue
        # Attribution for the duration of the step, so start_coroutine called
        # from inside a coroutine belongs to the same instance.
        _attribute(task._owner)
        try:
            task._step(now)
        except BaseException:
            failures.append((task._owner, _describe()))
            try:
                task._close()
            except BaseException:
                task._done = True
                task._stack = []
        finally:
            _attribute("")
        if task._done:
            reap = True
    if reap:
        _tasks[:] = [t for t in _tasks if not t._done]
    return failures, len(_tasks)


def _drop_for(uuid):
    """Drop every task owned by `uuid`. Returns (failures, remaining)."""
    failures = []
    kept = []
    for task in _tasks:
        if task._done:
            continue
        if task._owner != uuid:
            kept.append(task)
            continue
        _shed(task, failures)
    _tasks[:] = kept
    return failures, len(_tasks)


def _clear():
    """Drop everything. Returns (failures, remaining)."""
    failures = []
    pending = list(_tasks)
    del _tasks[:]
    for task in pending:
        if task._done:
            continue
        _shed(task, failures)
    return failures, 0
)PY";

    // The scheduler's entry points, resolved once at module init.
    //
    // Leaked, exactly as the interpreter itself is (see ScriptHost.cpp): CPython
    // is never finalized here, so a static py::object's destructor would run at
    // process exit, with no GIL held, against an interpreter nobody is looking
    // after any more. Null until the module body has run, which is what makes
    // the "no interpreter ever started" paths below trivial.
    py::object* g_start = nullptr;
    py::object* g_pump = nullptr;
    py::object* g_dropFor = nullptr;
    py::object* g_clear = nullptr;
    // The registry itself, so start_coroutine can refresh the mirror below
    // without a name lookup. Mutated in place by the scheduler (`_tasks[:] =`,
    // `del _tasks[:]`), never rebound, so this reference stays the live list.
    py::object* g_tasks = nullptr;

    // The GIL-free mirror of len(_tasks). See taskCount().
    std::size_t g_count = 0;

    // Unpack a (failures, remaining) answer: the count lands in the mirror, the
    // failures go back to the caller. GIL held.
    py::object take(const py::object& answer) {

        const auto pair = py::cast<py::tuple>(answer);
        g_count = py::cast<std::size_t>(pair[1]);
        return py::reinterpret_borrow<py::object>(pair[0]);
    }

}// namespace


namespace threepp::editor::scripting {

    std::size_t taskCount() {

        return g_count;
    }

    void initTasks(py::module_& m) {

        // The submodule init_editor already made — never a second one, or the
        // names would land somewhere no script imports.
        auto sub = m.def_submodule("editor");

        // The scheduler's hold on the attribution seam. Private, and the only
        // reason it exists at all: the pump loops in Python, so the "whose task
        // is this" window has to be opened and closed from there.
        sub.def(
                "_attribute", [](const std::string& uuid) { dispatchingScript() = uuid; },
                py::arg("uuid"),
                "Internal: set the script instance coroutine ownership is attributed to.");

        py::exec(kTaskSource, sub.attr("__dict__"));

        g_start = new py::object(sub.attr("_start"));
        g_pump = new py::object(sub.attr("_pump"));
        g_dropFor = new py::object(sub.attr("_drop_for"));
        g_clear = new py::object(sub.attr("_clear"));
        g_tasks = new py::object(sub.attr("_tasks"));

        sub.def(
                "start_coroutine", [](const py::object& generator) -> py::object {
                    const auto& owner = dispatchingScript();
                    if (owner.empty()) {
                        throw std::runtime_error(
                                "threepp.editor.start_coroutine: no script is being run right "
                                "now. A coroutine belongs to the script instance that started "
                                "it - a raise inside it disables that instance, and its tasks "
                                "die with it - so there has to be one. Call this from a "
                                "script's own methods (start, update, fixed_update, the "
                                "collision, trigger and break callbacks, stop), or from inside "
                                "another coroutine of one.");
                    }
                    auto task = (*g_start)(generator, owner);
                    // The mirror, refreshed here rather than at the next pump:
                    // a task registered from start() must make the very first
                    // frame take the GIL, and update()'s early-out reads this.
                    g_count = py::len(*g_tasks);
                    return task;
                },
                py::arg("generator"),
                "Run `generator` as a coroutine, and return the Task driving it.\n\n"
                "The generator is resumed once per frame, AFTER physics has stepped and "
                "after every script's update() has run - so what it sees is the settled "
                "state of the frame. Nothing of it executes inside this call; the body "
                "runs up to its first yield at the next pump.\n\n"
                "What it may yield:\n"
                "  yield                     - resume next frame\n"
                "  yield editor.wait(0.5)    - resume after half a second of SIMULATED "
                "time (see threepp.editor.time; a paused or starved physics world "
                "genuinely pauses the wait)\n"
                "  yield editor.until(pred)  - resume once pred() is truthy, and receive "
                "that value: `hit = yield editor.until(...)`\n"
                "  yield another_gen()       - run it to completion, and receive its "
                "return value\n\n"
                "The task belongs to the script instance whose method is running. A raise "
                "inside it is reported once and disables that instance whole, exactly as a "
                "raise in update() does; the instance's other tasks go with it. Every task "
                "is dropped at Stop, so the next Play starts from nothing. Start as many "
                "as you like.");
    }

    py::object pumpTasks(double simTime) {

        if (!g_pump) return py::list();
        return take((*g_pump)(simTime));
    }

    py::object dropTasksFor(const std::string& uuid) {

        if (!g_dropFor) return py::list();
        return take((*g_dropFor)(uuid));
    }

    py::object clearTasks() {

        if (!g_clear) {
            g_count = 0;
            return py::list();
        }
        return take((*g_clear)());
    }

}// namespace threepp::editor::scripting
