
#include "Scripting.hpp"

#include "ScriptHost.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/scenes/Scene.hpp"

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
            instance.self = self;

            // All three methods are optional; a script that only wants update()
            // is a perfectly good script.
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
