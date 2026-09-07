// threepp.editor.start_coroutine — the coroutine scheduler a play session pumps.
//
// Internal to apps/editor/scripting, like ScriptHost.hpp: the scheduler itself
// is Python (see ScriptTasks.cpp) and everything here speaks py::object.
//
// The four entry points below are the WHOLE surface ScriptPlaySession uses. All
// of them require the GIL, and all of them tolerate an interpreter that never
// started — a Play on a scene with no scripts must not be the thing that brings
// CPython up, and a teardown after such a Play must not be the thing that
// dereferences a module that was never imported.

#ifndef THREEPP_EDITOR_SCRIPTTASKS_HPP
#define THREEPP_EDITOR_SCRIPTTASKS_HPP

#include "ScriptHost.hpp"

#include <cstddef>
#include <string>

namespace threepp::editor::scripting {

    // Registers threepp.editor.start_coroutine, wait, until and Task into the
    // submodule init_editor created, and execs the scheduler into it. Called
    // from the embedded module's body, after the submodule exists.
    void initTasks(py::module_& m);

    // How many tasks are registered, as of the last call into the scheduler.
    //
    // GIL-FREE on purpose, and the reason the three calls below report a count
    // at all: ScriptPlaySession::update decides whether to acquire the GIL
    // before it has one, and "are there coroutines to pump" must be answerable
    // there. The number can only be stale HIGH — a script cancelling its own
    // task between two pumps leaves it counted until the next pump reaps it,
    // which costs one needless sweep and never a missed one.
    [[nodiscard]] std::size_t taskCount();

    // What every scheduler call hands back: the tasks that raised, as
    // (owner uuid, traceback) pairs, ready for ScriptPlaySession::Impl::fail.
    // Empty is the normal answer.

    // Advance every registered task once, with `simTime` as the clock wait()
    // measures against. GIL must be held.
    py::object pumpTasks(double simTime);

    // Drop every task owned by `uuid`, running its finally: blocks. Called when
    // an instance is disabled: its tasks go with it, exactly as its methods do.
    // GIL must be held.
    py::object dropTasksFor(const std::string& uuid);

    // Drop everything, running finally: blocks. Tasks are session state and do
    // not survive a Stop. GIL must be held.
    py::object clearTasks();

}// namespace threepp::editor::scripting

#endif//THREEPP_EDITOR_SCRIPTTASKS_HPP
