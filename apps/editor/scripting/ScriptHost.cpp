
#include "ScriptHost.hpp"

#include "Scripting.hpp"

#include "threepp/extras/editor/ScriptConfig.hpp"

#include <memory>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    namespace py = pybind11;

    // The loader and the discovery rules, in Python because that is where the
    // reflection is. Kept as one string so there is no file to find at runtime —
    // the editor must work from any working directory.
    constexpr const char* kHelperSource = R"PY(
import inspect, os, sys, types

_PREFIX = "_threepp_editor_script_"

def load(path):
    """Compile and run `path` in a brand-new module, and return (module, stem).

    Deliberately NOT importlib: the source loader validates its __pycache__
    entry against (mtime, size), both of which survive a same-second edit that
    happens not to change the length. Editing a script and pressing Play again
    would then re-run the OLD bytecode, which is the one failure this feature
    cannot have. Reading and compiling the file ourselves has no cache to be
    stale.
    """
    path = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    name = _PREFIX + stem
    with open(path, "rb") as handle:
        source = handle.read()
    # compile() before touching sys.modules, so a syntax error leaves nothing
    # half-registered behind it.
    code = compile(source, path, "exec")

    mod = types.ModuleType(name)
    mod.__file__ = path
    # Registered while it runs: a class defined here has to be picklable, and
    # anything doing `sys.modules[__name__]` has to find itself.
    sys.modules[name] = mod
    folder = os.path.dirname(path)
    added = bool(folder) and folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    try:
        exec(code, mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        if added:
            try:
                sys.path.remove(folder)
            except ValueError:
                pass
    return mod, stem

def find_class(mod, stem):
    """The script class: named after the file, else the one defining update()."""
    own = [v for v in vars(mod).values()
           if inspect.isclass(v) and getattr(v, "__module__", None) == mod.__name__]
    for c in own:
        if c.__name__.lower() == stem.lower():
            return c
    updaters = [c for c in own if callable(getattr(c, "update", None))]
    if len(updaters) == 1:
        return updaters[0]
    if not own:
        raise LookupError("%s.py defines no class" % stem)
    if not updaters:
        raise LookupError("%s.py: no class named '%s', and none defines update()" % (stem, stem))
    raise LookupError("%s.py: several classes define update(); name one '%s'" % (stem, stem))

def load_class(path):
    mod, stem = load(path)
    return find_class(mod, stem)

# Exact types only. `type(x) is bool` before int matters (bool IS an int), and
# rejecting subclasses keeps an IntEnum or a numpy scalar out of the inspector,
# where it would round-trip through text as something else.
_KINDS = ((bool, "bool"), (int, "int"), (float, "float"), (str, "str"))

def fields(cls):
    """[(name, kind, value)] for the plain data attributes of `cls`."""
    out = []
    index = {}
    for base in reversed(getattr(cls, "__mro__", (cls,))):
        if base is object:
            continue
        for name, value in vars(base).items():
            if name.startswith("_"):
                continue
            kind = None
            for t, k in _KINDS:
                if type(value) is t:
                    kind = k
                    break
            if kind is None:
                continue
            if name in index:
                out[index[name]] = (name, kind, value)
            else:
                index[name] = len(out)
                out.append((name, kind, value))
    return out
)PY";

    // The interpreter, the module scripts see, and the loader helpers.
    //
    // Never destroyed — see the leak below. It therefore has no destructor and
    // no teardown order to get wrong.
    struct Interpreter {

        py::scoped_interpreter interpreter{};
        py::object helpers;
        std::unique_ptr<py::gil_scoped_release> release;

        Interpreter() {

            // Import it here, not lazily. PYBIND11_EMBEDDED_MODULE only adds an
            // inittab entry — the class_ registrations that make py::cast work
            // run when the module body executes, i.e. on first import. Without
            // this, handing a script an object fails with "the bound type does
            // not use std::shared_ptr as its holder type", which is pybind11's
            // way of saying it has never heard of the type at all.
            py::module_::import("threepp");

            // A real module object rather than a bare dict: exec() then gives
            // the helper functions a proper __globals__, so they can see each
            // other and the builtins.
            py::object host = py::module_::import("types").attr("ModuleType")("_threepp_editor_scripting");
            py::exec(kHelperSource, host.attr("__dict__"));
            helpers = host;
            // The editor is not a Python program: it holds the GIL only inside
            // a sweep, and never between frames.
            release = std::make_unique<py::gil_scoped_release>();
        }
    };

    // Deliberately leaked, and it stays that way.
    //
    // Finalizing CPython 3.14 from pybind11 3.0.4 access-violates on Windows at
    // process exit — measured with a bare py::scoped_interpreter and nothing
    // else, so it is not the threepp bindings. Nothing is gained by shutting the
    // interpreter down at exit anyway (the OS reclaims the process), and a
    // crash-on-quit is exactly the kind of thing a user notices and reports.
    // One interpreter, started on the first Play that needs it, alive until the
    // process ends.
    Interpreter* g_interpreter = nullptr;
    // Sticky: a failed start is not retried every frame, and the reason is
    // still available to report.
    bool g_startAttempted = false;
    std::string g_startError;

}// namespace


namespace threepp::editor::scripting {

    bool ensureInterpreter(std::string* error) {

        if (g_interpreter) return true;
        if (g_startAttempted) {
            if (error) *error = g_startError;
            return false;
        }
        g_startAttempted = true;

        // Force the PYBIND11_EMBEDDED_MODULE object file into the link. It
        // registers `threepp` as a built-in from a static initializer, which has
        // already run by now — but only if the linker kept the TU.
        registerEmbeddedModule();

        try {
            g_interpreter = new Interpreter();
        } catch (const std::exception& e) {
            g_startError = e.what();
            if (error) *error = g_startError;
            return false;
        } catch (...) {
            g_startError = "unknown failure starting the Python interpreter";
            if (error) *error = g_startError;
            return false;
        }
        return true;
    }

    bool interpreterStarted() {

        return g_interpreter != nullptr;
    }

    py::object helpers() {

        return g_interpreter->helpers;
    }

    py::object loadScriptClass(const std::filesystem::path& path, std::string& className) {

        auto cls = helpers().attr("load_class")(path.string());
        className = py::cast<std::string>(cls.attr("__name__"));
        return cls;
    }

    py::object exposedFields(const py::object& cls) {

        return helpers().attr("fields")(cls);
    }

    std::string describeError(py::error_already_set& error) {

        // what() carries type, message and the traceback, and it is the one
        // formatter that cannot itself raise into our lap.
        try {
            return error.what();
        } catch (...) {
            return "unreportable Python error";
        }
    }

    Inspection inspect(const std::filesystem::path& path) {

        Inspection result;

        if (!ensureInterpreter(&result.error)) return result;

        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            result.error = "file not found: " + path.string();
            return result;
        }

        py::gil_scoped_acquire gil;
        try {
            auto cls = loadScriptClass(path, result.className);
            const auto discovered = py::cast<py::list>(exposedFields(cls));
            for (const auto& entry : discovered) {
                const auto tuple = py::cast<py::tuple>(entry);
                ScriptField field;
                field.name = py::cast<std::string>(tuple[0]);
                const auto kind = py::cast<std::string>(tuple[1]);
                if (kind == "bool") {
                    field.type = ScriptField::Type::Bool;
                    field.defaultValue = ScriptConfig::toText(py::cast<bool>(tuple[2]));
                } else if (kind == "int") {
                    field.type = ScriptField::Type::Int;
                    field.defaultValue = ScriptConfig::toText(py::cast<int>(tuple[2]));
                } else if (kind == "float") {
                    field.type = ScriptField::Type::Float;
                    field.defaultValue = ScriptConfig::toText(py::cast<float>(tuple[2]));
                } else {
                    field.type = ScriptField::Type::String;
                    field.defaultValue = ScriptConfig::sanitized(py::cast<std::string>(tuple[2]));
                }
                result.fields.push_back(std::move(field));
            }
        } catch (py::error_already_set& e) {
            result.error = describeError(e);
            result.fields.clear();
            result.className.clear();
        } catch (const std::exception& e) {
            result.error = e.what();
            result.fields.clear();
            result.className.clear();
        }
        return result;
    }

}// namespace threepp::editor::scripting
