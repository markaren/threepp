
#include "Scripting.hpp"

#include "ScriptHost.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/scenes/Scene.hpp"

// fixed_update rides the physics world's substep loop, so this session has to
// reach the world the PhysicsPlaySession is playing. Same seam
// bind_editor_physics.cpp uses — the static active() on the session, plus its
// lifetime token — and gated on the same SDK: without PhysX there is no fixed
// clock to run on, and the session says so once instead of inventing one.
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#endif

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace py = pybind11;

namespace {

    // What a message calls an object. The name if it has one — a uuid in the
    // console tells nobody anything.
    std::string labelOf(const Object3D& object) {

        return object.name.empty() ? ("<" + object.type() + ">") : object.name;
    }

    // What a message calls the code. A file has a name; inline source has
    // nothing but the object it belongs to, which the message already carries.
    std::string originOf(const ScriptConfig& config) {

        if (config.isInline()) return "inline script";
        return std::filesystem::path(config.path).filename().string();
    }

    // A script declares start(self, obj) or start(self, obj, scene); the scene
    // is passed only when asked for, so every existing script keeps working.
    // `start` arrives as a bound method, so `self` is already hidden and the
    // counts here are one positional versus two. *args reads as wanting it.
    bool startWantsScene(const py::object& start) {

        try {
            const auto inspect = py::module_::import("inspect");
            const auto parameters =
                    inspect.attr("signature")(start).attr("parameters").attr("values")();
            std::size_t positional = 0;
            for (auto parameter : parameters) {
                const auto kind = parameter.attr("kind");
                if (kind.equal(parameter.attr("VAR_POSITIONAL"))) return true;
                if (kind.equal(parameter.attr("POSITIONAL_ONLY")) ||
                    kind.equal(parameter.attr("POSITIONAL_OR_KEYWORD"))) {
                    ++positional;
                }
            }
            return positional >= 2;
        } catch (py::error_already_set&) {
            // signature() balks at some exotic callables; the safe reading is
            // the one every script had before this feature existed.
            return false;
        }
    }

}// namespace


struct ScriptPlaySession::Impl {

    struct Instance {
        std::string uuid;
        std::string label;
        // "spinner.py" or "inline script" — what the console calls the code.
        std::string origin;
        // The script object itself. Created and destroyed with the GIL held.
        py::object self;
        bool hasUpdate = false;
        bool hasStop = false;
        // Cached like the others, and for the same reason: hasattr is a dict
        // lookup down the MRO, and this one would otherwise be paid per SUBSTEP.
        bool hasFixedUpdate = false;
        // Set the first time this script raises. It stays set for the rest of
        // the session: a script that throws every frame must not fill the
        // console with the same traceback sixty times a second.
        bool failed = false;
    };

    std::vector<Instance> instances;
    // Kept across stop() so the inspector can still show what went wrong.
    std::unordered_map<std::string, std::string> errors;
    std::function<void(const std::string&)> logger;

    void log(const std::string& message) const {

        if (logger) logger(message);
    }

    // One report, one disable. Never rethrows: the editor keeps playing.
    void fail(Instance& instance, const std::string& what, const std::string& detail) {

        instance.failed = true;
        instance.self = py::object();
        errors[instance.uuid] = detail;
        log("script error in " + instance.origin + " on " + instance.label +
            " (" + what + "): " + detail);
    }

    // Applies the authored overrides to a fresh instance. Fields the document
    // has no value for keep the class attribute, which is what the inspector
    // showed as the default.
    void applyFields(const py::object& cls, const py::object& self, const ScriptConfig& config) {

        const auto discovered = py::cast<py::list>(scripting::exposedFields(cls));
        for (const auto& entry : discovered) {
            const auto tuple = py::cast<py::tuple>(entry);
            const auto name = py::cast<std::string>(tuple[0]);
            const auto stored = config.field(name);
            if (!stored) continue;

            const auto kind = py::cast<std::string>(tuple[1]);
            if (kind == "bool") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toBool(*stored)));
            } else if (kind == "int") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toInt(*stored)));
            } else if (kind == "float") {
                py::setattr(self, name.c_str(), py::cast(ScriptConfig::toFloat(*stored)));
            } else {
                py::setattr(self, name.c_str(), py::cast(*stored));
            }
        }
    }

    // --- the physics clock -------------------------------------------------
    //
    // fixed_update() is not a frame callback. It runs from inside
    // PhysxWorld::step(), once per fixed substep, with the constant substep dt —
    // which is the whole point: a controller that applies forces or drive
    // targets from update(dt) is frame-rate dependent, and a frame-rate
    // dependent controller is not a controller.
    //
    // ONE callback for the whole session, registered only when at least one live
    // instance defines the method: a scene whose scripts never ask must not pay
    // a per-substep call, nor a GIL acquisition, for the privilege.
    //
    // onPreSubstep, NOT onPostSubstep. Inside a substep PhysxWorld does:
    //     pre callbacks -> simulate -> fetchResults -> post callbacks -> sensors tick
    // so the registered sensors are sampled AFTER both hooks; neither one can see
    // the sample belonging to the substep it sits in. The pre hook of substep k+1
    // stands one tick later than the post hook of substep k, and both feed the
    // same simulate() — so pre is never staler and puts the write immediately
    // before the solve that consumes it, which is the ordering "read the sensor,
    // write the drive target" wants.
    //
    // (What a script's sensor HANDLE reads is one step removed from either: the
    // sensor session drains its rings once per frame, since drain() empties them
    // and there may be only one drainer. So a handle here reports the batch that
    // existed when the frame began, and two substeps of one frame see the same
    // reading. Documented in doc/editor.md rather than worked around — a second
    // drainer would starve the Sensors panel.)

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // The world is BORROWED. Physics is registered first and therefore stops
    // first, so by the time stop() runs here the PhysxWorld is usually already
    // gone — the token (which PhysicsPlaySession drops before the world) is what
    // makes "is my registration still there to remove" answerable without ever
    // dereferencing a dead pointer.
    PhysxWorld* world = nullptr;
    std::weak_ptr<const void> worldLife;
    PhysxWorld::SubstepHandle substep = 0;
    bool registered = false;
#endif

    // Live instances that define fixed_update.
    [[nodiscard]] std::size_t fixedUpdateWanted() const {

        std::size_t wanted = 0;
        for (const auto& instance : instances) {
            if (!instance.failed && instance.hasFixedUpdate) ++wanted;
        }
        return wanted;
    }

    // Hook the session onto the playing world's substep loop, or say once why
    // there is nothing to hook onto. Called at the end of start().
    void attachToPhysics() {

        const auto wanted = fixedUpdateWanted();
        if (wanted == 0) return;

#ifdef THREEPP_EDITOR_WITH_PHYSX
        // PhysicsPlaySession::start() always builds a world, even for a scene
        // with nothing physical in it — so "a session is playing" is the whole
        // condition. Sessions start in registration order and physics is first,
        // so it is already up by the time we get here.
        if (auto* physics = PhysicsPlaySession::active(); physics && physics->world()) {
            world = physics->world();
            worldLife = physics->lifetime();
            substep = world->onPreSubstep([this](float dt) { fixedUpdate(dt); });
            registered = true;
            return;
        }
#endif
        // No world, no fixed clock. Faking one — a timer of our own, ticking at
        // some nominal rate — would be a lie about the one thing the name
        // promises, so the method simply never fires and the console says so.
        log("fixed_update needs a playing physics world - it never runs on " +
            std::to_string(wanted) + (wanted == 1 ? " script" : " scripts"));
    }

    // Give the registration back, but only while there is provably something to
    // give it back to. A world that died first took the callback with it, and
    // the handle we are still holding is then just a number.
    void detachFromPhysics() {

#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (registered && world && !worldLife.expired()) {
            world->removeSubstepCallback(substep);
        }
        registered = false;
        world = nullptr;
        worldLife.reset();
#endif
    }

    // One substep, every live instance that asked for it. Same error semantics
    // as the frame sweep: first raise disables the instance for the session, so
    // a script throwing at 60 Hz reports once and then stops costing anything.
    void fixedUpdate(float dt) {

        bool any = false;
        for (const auto& instance : instances) {
            if (!instance.failed && instance.hasFixedUpdate) any = true;
        }
        if (!any) return;

        // Inside the callback, not around the step: the physics session holds no
        // GIL, and a session whose scripts all failed pays nothing here.
        py::gil_scoped_acquire gil;

        for (auto& instance : instances) {
            if (instance.failed || !instance.hasFixedUpdate) continue;
            try {
                instance.self.attr("fixed_update")(dt);
            } catch (py::error_already_set& e) {
                fail(instance, "fixed_update", scripting::describeError(e));
            } catch (const std::exception& e) {
                fail(instance, "fixed_update", e.what());
            }
        }
    }

    // Drops every interpreter-side object. GIL is acquired here rather than by
    // the caller so this is safe from the destructor too.
    void release() {

        if (instances.empty()) return;
        if (!scripting::interpreterStarted()) {
            // Nothing was ever created; the py::objects are all null.
            instances.clear();
            return;
        }
        py::gil_scoped_acquire gil;
        instances.clear();
    }
};


ScriptPlaySession::ScriptPlaySession()
    : impl_(std::make_unique<Impl>()) {}

ScriptPlaySession::~ScriptPlaySession() {

    // The editor torn down mid-Play never gets a stop(), and a substep callback
    // outliving the object it captured is exactly what PhysxWorld's "anything
    // that registers must unregister in its destructor" rule is about.
    impl_->detachFromPhysics();
    impl_->release();
}

void ScriptPlaySession::setLogger(std::function<void(const std::string&)> logger) {

    impl_->logger = std::move(logger);
}

std::string ScriptPlaySession::errorFor(const std::string& uuid) const {

    const auto it = impl_->errors.find(uuid);
    return it == impl_->errors.end() ? std::string{} : it->second;
}

std::size_t ScriptPlaySession::errorCount() const {

    return impl_->errors.size();
}

void ScriptPlaySession::clearErrors() {

    impl_->errors.clear();
}

std::size_t ScriptPlaySession::instanceCount() const {

    return impl_->instances.size();
}

void ScriptPlaySession::start(Scene& scene) {

    // A previous session's registration goes before its instances do. Replaying
    // without it would leave the old callback in a world that has since been
    // rebuilt — and the handle stale for the new one.
    impl_->detachFromPhysics();
    impl_->release();
    impl_->errors.clear();

    // Collect first, so nothing Python-shaped happens on a scene with no
    // scripts — starting an interpreter costs a hundred milliseconds and every
    // Play would pay it.
    struct Attached {
        Object3D* object;
        ScriptConfig config;
    };
    std::vector<Attached> attached;
    scene.traverse([&attached](Object3D& object) {
        if (auto config = ScriptConfig::read(object)) {
            attached.push_back({&object, std::move(*config)});
        }
    });
    if (attached.empty()) return;

    std::string error;
    if (!scripting::ensureInterpreter(&error)) {
        impl_->log("scripts skipped - the Python interpreter did not start: " + error);
        return;
    }

    py::gil_scoped_acquire gil;

    for (auto& entry : attached) {

        // Recorded before anything can go wrong, so a failure has somewhere to
        // be recorded against.
        const auto label = labelOf(*entry.object);
        impl_->instances.push_back(
                Impl::Instance{entry.object->uuid, label, originOf(entry.config)});
        auto& instance = impl_->instances.back();

        std::error_code ec;
        if (entry.config.isFile() && !std::filesystem::exists(entry.config.path, ec)) {
            impl_->fail(instance, "load", "file not found: " + entry.config.path);
            continue;
        }

        try {
            std::string className;
            // Inline source is compiled fresh here, exactly as a file is read
            // and compiled fresh — inline scripts are therefore inherently hot:
            // Apply in the editor, press Play, and that is what runs. The
            // module is named after the object so a traceback says which one.
            auto cls = entry.config.isInline()
                               ? scripting::loadInlineScriptClass(entry.config.source,
                                                                  entry.object->uuid, label, className)
                               : scripting::loadScriptClass(entry.config.path, className);
            auto self = cls();
            impl_->applyFields(cls, self, entry.config);

            instance.hasUpdate = py::hasattr(self, "update");
            instance.hasStop = py::hasattr(self, "stop");
            instance.hasFixedUpdate = py::hasattr(self, "fixed_update");
            instance.self = self;

            // Every method is optional; a script that only wants update() — or
            // only fixed_update() — is a perfectly good script.
            if (py::hasattr(self, "start")) {
                const auto start = self.attr("start");
                if (startWantsScene(start)) {
                    start(scripting::handleFor(*entry.object), scripting::handleFor(scene));
                } else {
                    start(scripting::handleFor(*entry.object));
                }
            }
        } catch (py::error_already_set& e) {
            impl_->fail(instance, "start", scripting::describeError(e));
        } catch (const std::exception& e) {
            impl_->fail(instance, "start", e.what());
        }
    }

    std::size_t live = 0;
    for (const auto& instance : impl_->instances) {
        if (!instance.failed) ++live;
    }
    if (live > 0) {
        impl_->log("scripts running: " + std::to_string(live) +
                   (live == 1 ? " instance" : " instances"));
    }

    // Last, so it only ever considers instances that actually came up: a script
    // whose start() raised is disabled, and its fixed_update with it.
    impl_->attachToPhysics();
}

void ScriptPlaySession::update(float dt) {

    if (impl_->instances.empty()) return;

    bool any = false;
    for (const auto& instance : impl_->instances) {
        if (!instance.failed && instance.hasUpdate) any = true;
    }
    if (!any) return;

    // Once for the whole sweep, not once per script: acquiring the GIL is the
    // expensive part, and there is nothing else running that wants it.
    py::gil_scoped_acquire gil;

    for (auto& instance : impl_->instances) {
        if (instance.failed || !instance.hasUpdate) continue;
        try {
            instance.self.attr("update")(dt);
        } catch (py::error_already_set& e) {
            impl_->fail(instance, "update", scripting::describeError(e));
        } catch (const std::exception& e) {
            impl_->fail(instance, "update", e.what());
        }
    }
}

void ScriptPlaySession::stop() {

    // Before anything Python-shaped, and unconditionally: the instance list
    // being empty says nothing about whether a callback is still registered.
    impl_->detachFromPhysics();

    if (impl_->instances.empty()) return;

    if (scripting::interpreterStarted()) {
        py::gil_scoped_acquire gil;
        for (auto& instance : impl_->instances) {
            if (instance.failed || !instance.hasStop) continue;
            try {
                instance.self.attr("stop")();
            } catch (py::error_already_set& e) {
                impl_->fail(instance, "stop", scripting::describeError(e));
            } catch (const std::exception& e) {
                impl_->fail(instance, "stop", e.what());
            }
        }
        // Inside the same GIL scope: dropping the last reference to a script
        // instance runs its __del__ and frees Python memory.
        impl_->instances.clear();
        return;
    }
    impl_->instances.clear();
}
