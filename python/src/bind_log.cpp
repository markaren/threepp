// threepp's log messages from Python: thresholds per category, and capture.
//
//   tp.set_log_level("vulkan", tp.LogLevel.Warn)      # or tp.set_log_levels("vulkan.probe:debug,core:off")
//   tp.capture_log()                                  # records queue up instead of going to stderr
//   ...
//   for r in tp.take_log(): print(r.level, r.category, r.message, r.frame)
//
// Capture is a queue, not a Python callback, on purpose. The library logs on
// whichever thread reached the message, worker threads included, and it does
// so while the calling Python thread may hold the GIL and wait for exactly
// that worker. A sink that took the GIL there would deadlock; a sink that
// only appends to a C++ queue cannot. Python drains the queue when it likes.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/utils/Log.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;

namespace threepp_py {

    namespace {

        struct PyLogRecord {
            log::Level level;
            std::string category;
            std::string message;
            uint64_t frame;
        };

        struct Capture {
            std::mutex mutex;
            std::vector<PyLogRecord> records;
            size_t dropped = 0;
        };

        // Records kept between two take_log() calls; past it the oldest go.
        constexpr size_t kMaxRecords = 4096;

        Capture& capture() {
            // Never destroyed: the sink may run while the interpreter exits.
            static Capture* c = new Capture;
            return *c;
        }

        const char* levelWord(log::Level l) {
            return l == log::Level::Warn ? "warning: " : l == log::Level::Error ? "error: " : "";
        }

    }// namespace

    void init_log(py::module_& m) {

        py::enum_<log::Level>(m, "LogLevel", "Severity of a threepp log message, and the threshold of a category.")
                .value("Trace", log::Level::Trace)
                .value("Debug", log::Level::Debug)
                .value("Info", log::Level::Info)
                .value("Warn", log::Level::Warn)
                .value("Error", log::Level::Error)
                .value("Off", log::Level::Off);

        py::class_<PyLogRecord>(m, "LogRecord", "One captured log message.")
                .def_readonly("level", &PyLogRecord::level)
                .def_readonly("category", &PyLogRecord::category, "Dotted source name, e.g. 'vulkan.overlay'.")
                .def_readonly("message", &PyLogRecord::message)
                .def_readonly("frame", &PyLogRecord::frame, "Renderer frame number when it was logged.")
                .def("__repr__", [](const PyLogRecord& r) {
                    return "LogRecord([" + r.category + "] " + levelWord(r.level) + r.message + ")";
                });

        m.def(
                "set_log_level",
                [](const std::string& category, log::Level level) { log::setLevel(category, level); },
                py::arg("category"), py::arg("level"),
                "Threshold for `category` and every category below it ('vulkan' covers 'vulkan.overlay'; "
                "'*' covers all). Messages under the threshold are not produced. The default is Info. "
                "The THREEPP_LOG environment variable, when set, wins over this call.");

        m.def(
                "set_log_levels",
                [](const std::string& spec) { return log::setLevels(spec); },
                py::arg("spec"),
                "Several thresholds at once, in the THREEPP_LOG form: 'vulkan.probe:debug,core:warn'. "
                "Returns False if an entry was not understood (the others still apply).");

        m.def(
                "capture_log",
                [](bool enabled, bool echo) {
                    if (!enabled) {
                        log::setSink({});
                        return;
                    }
                    log::setSink([echo](const log::Record& r) {
                        if (echo) {
                            std::string line;
                            line.append("[").append(r.category).append("] ").append(levelWord(r.level)).append(r.message).append("\n");
                            std::fwrite(line.data(), 1, line.size(), stderr);
                            std::fflush(stderr);
                        }
                        auto& c = capture();
                        std::lock_guard lock(c.mutex);
                        if (c.records.size() >= kMaxRecords) {
                            c.records.erase(c.records.begin());
                            ++c.dropped;
                        }
                        c.records.push_back({r.level, std::string(r.category), std::string(r.message), r.frame});
                    });
                },
                py::arg("enabled") = true, py::arg("echo") = false,
                "Queue log messages for take_log() instead of writing them to stderr; `echo` keeps the "
                "stderr line as well. capture_log(False) restores the stderr output. Records already "
                "queued stay until taken.");

        m.def(
                "take_log",
                [] {
                    auto& c = capture();
                    std::lock_guard lock(c.mutex);
                    c.dropped = 0;
                    return std::exchange(c.records, {});
                },
                "The messages captured since the last call, oldest first, as LogRecord objects. The queue "
                "keeps the newest 4096.");
    }

}// namespace threepp_py
