// threepp.editor, authoring half — what a GENERATOR script uses to put the
// content it builds into the document.
//
// Editor-only, like bind_editor_physics.cpp: the wheel has no document to author
// into, so these names exist only in the editor's embedded module. Must be
// registered AFTER init_editor, which creates the submodule.
//
// The contract is deliberately narrow. A generator does not get the document to
// mutate however it likes; it gets ONE verb, add(), which appends to the sink the
// editor opened for this run (see scripting::authoringSink). The editor then
// commits that finished subtree as a single undoable step. Consequences worth
// knowing:
//
//   * outside a generator run there is no sink, and add() raises rather than
//     silently doing nothing — a behaviour script calling it during Play is a
//     mistake, and the document is exactly what Play must not touch;
//   * a script that raises halfway has added nothing to the document, because
//     nothing was in the document yet;
//   * the objects are plain threepp objects, so everything else the module can
//     already build (geometries, materials, InstancedMesh, lights) composes
//     without any of it knowing about the editor.
//
// scene() and selected() are READ access, for scripts that place content
// relative to what is already there — scatter onto that terrain, one crate per
// marker. They hand back the live graph, so a script CAN reach in and mutate it
// directly; that is not authored content and will not be saved as such, and the
// docstrings say so rather than pretending the door is locked.
//
// scene() is the one name here that a BEHAVIOUR script uses too: during Play it
// answers with the scene the play session is running (scripting::playScene),
// which is how a script reaches a neighbour by name. It is the same verb asking
// the same question — "what is around me" — so it is one binding rather than
// two spellings of it, and the fallback below is the whole of the difference.

#include "bindings.hpp"

#include "Scripting.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>

using namespace threepp;

namespace threepp_py {
namespace {

    // The type behind threepp.editor.time. Deliberately EMPTY: it carries no
    // reading of its own, and every property on it reads the live
    // scripting::scriptClock() when asked. A struct holding a copy would be a
    // snapshot, and a script that stashed one in start() would read the same
    // stale numbers forever.
    struct ScriptClockView {};

    // Same handle-not-typed-pointer rule the rest of threepp.editor follows:
    // threepp's virtual bases make a typed shared_ptr<Object3D> parameter
    // unusable, so every editor entry point takes py::handle and converts.
    std::shared_ptr<Object3D> sinkRelative(const py::handle& parent) {

        auto* sink = editor::scripting::authoringSink();
        if (!sink) {
            throw std::runtime_error(
                    "threepp.editor.add: nothing is being generated right now. This is for a "
                    "scene generator script, run from the editor's Generator section; a "
                    "behaviour script must not add to the document while playing.");
        }
        if (parent.is_none()) return {};
        auto object = threepp_py::as_object3d(parent);
        if (!object) {
            throw std::runtime_error("threepp.editor.add: `parent` is not an Object3D");
        }
        return object;
    }

    // The colour of a debug draw call: a hex int, a threepp.Color, or nothing
    // (white). Wrong types raise with the two accepted spellings named, rather
    // than the cast's own message.
    Color drawColor(const py::handle& value) {

        if (value.is_none()) return Color(0xffffff);
        try {
            return py::cast<Color>(value);
        } catch (const py::cast_error&) {
        }
        try {
            return Color(py::cast<int>(value));
        } catch (const py::cast_error&) {
        }
        throw std::runtime_error(
                "draw color must be a hex int (0xff0000) or a threepp.Color");
    }

}// namespace

    void init_editor_authoring(py::module_& m) {

        auto sub = m.def_submodule("editor");

        sub.def(
                "add", [](const py::handle& object, const py::handle& parent) -> py::object {
                    auto explicitParent = sinkRelative(parent);
                    auto child = as_object3d(object);
                    if (!child) {
                        throw std::runtime_error("threepp.editor.add: not an Object3D");
                    }
                    Object3D* target = explicitParent ? explicitParent.get()
                                                      : editor::scripting::authoringSink();
                    target->add(child);
                    // Hand the object straight back so a script can keep building
                    // on it: parent = editor.add(group).
                    return py::cast(child);
                },
                py::arg("object"), py::arg("parent") = py::none(),
                "Add `object` to what this generator is building, and return it. With no "
                "`parent` it goes at the generator's root; pass one of your own earlier adds "
                "to nest. Raises outside a generator run.");

        sub.def(
                "scene", []() -> py::object {
                    // A generator run first, because during one that IS the
                    // scene the caller means — and the two never overlap anyway:
                    // the editor does not generate into a document it is
                    // playing.
                    auto* scene = editor::scripting::authoringScene();
                    if (!scene) scene = editor::scripting::playScene();
                    if (!scene) {
                        throw std::runtime_error(
                                "threepp.editor.scene: no scene right now. This answers "
                                "during a generator run and from a behaviour script's "
                                "start/update/stop while playing — nowhere else.");
                    }
                    return py::cast(scene->shared_from_this());
                },
                "The live scene: what a generator is authoring into, or what a behaviour "
                "script is playing in.\n\n"
                "READ it to reach what you did not author — `scene.get_object_by_name(\"Ground\")`, "
                "`scene.children`. A generator places content relative to what already exists (a "
                "marker to put a crate on, a surface to scatter over); objects reached that way "
                "are NOT its output and are not replaced when it re-runs, only what you pass to "
                "add() is.\n\n"
                "During Play this answers from start() onwards, including update(), "
                "fixed_update() and the collision and trigger callbacks — so a script that needs "
                "the scene later does not have to stash it. Raises when nothing is generating "
                "and nothing is playing.");

        // Input, for a BEHAVIOUR script during Play — the other half of "a play session is a
        // thing you can drive". Poll it from update(); it never sticks.
        sub.def(
                "is_key_down", [](const std::string& key) {
                    auto& provider = editor::scripting::keyStateProvider();
                    // No window (headless, or a front end that installed no provider) reads as
                    // "nothing is held" rather than raising: a script that steers with the
                    // arrow keys should still RUN in a headless pass, just uncommanded.
                    return provider ? provider(key) : false;
                },
                py::arg("key"),
                "Poll whether a key is currently held — 'W', 'SPACE', 'UP', 'LEFT', 'KP8', the "
                "same names Canvas.is_key_down takes. Answers False while the user is typing "
                "into a field, so driving a robot cannot eat somebody's rename, and False in a "
                "build or a pass with no window. Query it every update() for continuous "
                "control; it never sticks.");

        // --- the clock -------------------------------------------------------
        //
        // threepp.editor.time — ONE object, bound once, whose every property
        // reads the live ScriptClock at the moment it is asked. So a script may
        // keep `t = editor.time` in start() and still read fresh numbers from it
        // a thousand frames later; there is no snapshot to go stale.
        //
        // Why it exists at all: update(dt) is wall time and fixed_update(dt) is
        // simulated time, the two diverge under load, and a script had no way to
        // ask which one it was holding (see ScriptClock).

        py::class_<ScriptClockView>(
                sub, "Time",
                "The play session's clocks, live - read `threepp.editor.time`.\n\n"
                "There are two, and they do not agree. WALL time is what update(dt) "
                "rides: real seconds, however many the last frame took. SIM time is what "
                "the physics world advances in fixed substeps, and what fixed_update(dt) "
                "and every sensor timestamp are stamped with. A frame that hitches "
                "advances wall time in full but simulates at most a few substeps and "
                "drops the remainder, so the two drift apart for good - which is why "
                "anything integrating toward a physics quantity should be reading "
                "sim_time (or living in fixed_update) rather than summing update's dt.\n\n"
                "Every field is zero outside Play.")
                .def_property_readonly(
                        "frame_dt", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().frameDt;
                        },
                        "Wall-clock seconds the last frame took - the same number update(dt) "
                        "is handed, readable from the methods that are not handed it.")
                .def_property_readonly(
                        "wall_time", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().wallTime;
                        },
                        "Real seconds since Play started.")
                .def_property_readonly(
                        "sim_time", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().simTime;
                        },
                        "Simulated seconds since Play started: the physics world's own clock, "
                        "which advances only when substeps run. This is the SAME clock that "
                        "stamps sensor samples, so comparing a sample's timestamp against it is "
                        "meaningful. Inside fixed_update it reads the time at the START of the "
                        "substep about to be solved.")
                .def_property_readonly(
                        "sim_dt", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().simDt;
                        },
                        "The fixed substep in seconds - constant for the run, and exactly what "
                        "fixed_update(dt) is handed.")
                .def_property_readonly(
                        "steps", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().steps;
                        },
                        "Fixed substeps completed since Play started. Advances by 0, 1 or more "
                        "per frame depending on how long the frame took.")
                .def_property_readonly(
                        "fixed_clock", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().fixedClock;
                        },
                        "True when sim_time and sim_dt come from a playing physics world. False "
                        "in a build or a pass without one, where sim_time falls back to wall_time "
                        "and sim_dt to frame_dt - still an elapsed-time answer, but not a "
                        "simulated one, and this is how to tell.")
                .def_property_readonly(
                        "playing", [](const ScriptClockView&) {
                            return editor::scripting::scriptClock().active;
                        },
                        "True between the start of a play session and its stop.")
                .def("__repr__", [](const ScriptClockView&) {
                    const auto& clock = editor::scripting::scriptClock();
                    if (!clock.active) return std::string("<threepp.editor.time (not playing)>");
                    return "<threepp.editor.time sim=" + std::to_string(clock.simTime) +
                           "s wall=" + std::to_string(clock.wallTime) +
                           "s steps=" + std::to_string(clock.steps) + ">";
                });

        // The singleton. An instance rather than a module attribute per field,
        // because a module attribute would be evaluated once at import and freeze
        // the numbers at zero.
        sub.attr("time") = ScriptClockView{};

        // --- debug draw ------------------------------------------------------
        //
        // A behaviour script's instrument panel: world-space lines, drawn over
        // the scene for ONE frame and gone. Everything decomposes to segments
        // into one shared list (scripting::debugDraw) that the editor drains
        // into a single overlay LineSegments per rendered frame — one primitive,
        // one draw path. Immediate mode: a line that should persist is redrawn
        // every update(), which is exactly when a script is called anyway.
        //
        // No-ops outside a play session, same reasoning as is_key_down above: a
        // script that draws must still RUN headless, just unseen. And the lines
        // are editor furniture, invisible to the sensors — a lidar must not
        // range against somebody's debug arrow.

        constexpr const char* kDrawSeconds =
                "Draws for the CURRENT frame only - call it every update() to keep it visible. "
                "No-op outside Play.";

        sub.def(
                "draw_line",
                [](const Vector3& a, const Vector3& b, const py::handle& color) {
                    const auto c = drawColor(color);
                    editor::scripting::debugDraw().push(a.x, a.y, a.z, b.x, b.y, b.z,
                                                        c.r, c.g, c.b);
                },
                py::arg("a"), py::arg("b"), py::arg("color") = py::none(),
                (std::string("Draw a world-space line from `a` to `b`. `color` is a hex int or "
                             "a threepp.Color; default white. ") +
                 kDrawSeconds)
                        .c_str());

        sub.def(
                "draw_ray",
                [](const Vector3& origin, const Vector3& direction, float length,
                   const py::handle& color) {
                    const auto c = drawColor(color);
                    const Vector3 end{origin.x + direction.x * length,
                                      origin.y + direction.y * length,
                                      origin.z + direction.z * length};
                    editor::scripting::debugDraw().push(origin.x, origin.y, origin.z,
                                                        end.x, end.y, end.z, c.r, c.g, c.b);
                },
                py::arg("origin"), py::arg("direction"), py::arg("length") = 1.f,
                py::arg("color") = py::none(),
                (std::string("Draw `origin` plus `direction` times `length` - the shape of a "
                             "raycast, so `draw_ray(origin, direction, hit.distance)` shows "
                             "exactly the ray that hit. `direction` is used as given, not "
                             "normalised. ") +
                 kDrawSeconds)
                        .c_str());

        sub.def(
                "draw_point",
                [](const Vector3& p, float size, const py::handle& color) {
                    const auto c = drawColor(color);
                    const float h = size * 0.5f;
                    auto& list = editor::scripting::debugDraw();
                    list.push(p.x - h, p.y, p.z, p.x + h, p.y, p.z, c.r, c.g, c.b);
                    list.push(p.x, p.y - h, p.z, p.x, p.y + h, p.z, c.r, c.g, c.b);
                    list.push(p.x, p.y, p.z - h, p.x, p.y, p.z + h, c.r, c.g, c.b);
                },
                py::arg("point"), py::arg("size") = 0.25f, py::arg("color") = py::none(),
                (std::string("Draw a small axis-aligned cross at `point` - a position made "
                             "visible. ") +
                 kDrawSeconds)
                        .c_str());

        sub.def(
                "draw_box",
                [](const Vector3& center, const Vector3& size, const py::handle& color) {
                    const auto c = drawColor(color);
                    const float hx = size.x * 0.5f, hy = size.y * 0.5f, hz = size.z * 0.5f;
                    auto& list = editor::scripting::debugDraw();
                    // 12 edges of the axis-aligned box, bottom ring / top ring /
                    // verticals.
                    const float x0 = center.x - hx, x1 = center.x + hx;
                    const float y0 = center.y - hy, y1 = center.y + hy;
                    const float z0 = center.z - hz, z1 = center.z + hz;
                    const auto edge = [&](float ax, float ay, float az, float bx, float by,
                                          float bz) {
                        list.push(ax, ay, az, bx, by, bz, c.r, c.g, c.b);
                    };
                    edge(x0, y0, z0, x1, y0, z0);
                    edge(x1, y0, z0, x1, y0, z1);
                    edge(x1, y0, z1, x0, y0, z1);
                    edge(x0, y0, z1, x0, y0, z0);
                    edge(x0, y1, z0, x1, y1, z0);
                    edge(x1, y1, z0, x1, y1, z1);
                    edge(x1, y1, z1, x0, y1, z1);
                    edge(x0, y1, z1, x0, y1, z0);
                    edge(x0, y0, z0, x0, y1, z0);
                    edge(x1, y0, z0, x1, y1, z0);
                    edge(x1, y0, z1, x1, y1, z1);
                    edge(x0, y0, z1, x0, y1, z1);
                },
                py::arg("center"), py::arg("size"), py::arg("color") = py::none(),
                (std::string("Draw the 12 edges of an axis-aligned box: `center` and `size` as "
                             "full extents - an AABB, a spawn region, a sensor's reach. ") +
                 kDrawSeconds)
                        .c_str());

        sub.def(
                "draw_sphere",
                [](const Vector3& center, float radius, const py::handle& color) {
                    const auto c = drawColor(color);
                    auto& list = editor::scripting::debugDraw();
                    // Three great circles, one per plane - the wireframe idiom
                    // that reads as a sphere from any angle.
                    constexpr int kSteps = 24;
                    constexpr float kTau = 6.2831853f;
                    for (int i = 0; i < kSteps; ++i) {
                        const float t0 = kTau * static_cast<float>(i) / kSteps;
                        const float t1 = kTau * static_cast<float>(i + 1) / kSteps;
                        const float c0 = std::cos(t0) * radius, s0 = std::sin(t0) * radius;
                        const float c1 = std::cos(t1) * radius, s1 = std::sin(t1) * radius;
                        list.push(center.x + c0, center.y + s0, center.z,
                                  center.x + c1, center.y + s1, center.z, c.r, c.g, c.b);
                        list.push(center.x + c0, center.y, center.z + s0,
                                  center.x + c1, center.y, center.z + s1, c.r, c.g, c.b);
                        list.push(center.x, center.y + c0, center.z + s0,
                                  center.x, center.y + c1, center.z + s1, c.r, c.g, c.b);
                    }
                },
                py::arg("center"), py::arg("radius") = 1.f, py::arg("color") = py::none(),
                (std::string("Draw a wireframe sphere as three great circles - a trigger "
                             "radius, a sensor range, a clearance. ") +
                 kDrawSeconds)
                        .c_str());

        sub.def(
                "draw_axes",
                [](const py::handle& h, float size) {
                    auto object = as_object3d(h);
                    if (!object) {
                        throw std::runtime_error("threepp.editor.draw_axes: not an Object3D");
                    }
                    Vector3 origin;
                    Quaternion rotation;
                    object->getWorldPosition(origin);
                    object->getWorldQuaternion(rotation);
                    auto& list = editor::scripting::debugDraw();
                    const auto axis = [&](float x, float y, float z, float r, float g,
                                          float b) {
                        Vector3 dir{x, y, z};
                        dir.applyQuaternion(rotation).multiplyScalar(size);
                        list.push(origin.x, origin.y, origin.z, origin.x + dir.x,
                                  origin.y + dir.y, origin.z + dir.z, r, g, b);
                    };
                    axis(1, 0, 0, 1.f, 0.25f, 0.25f);
                    axis(0, 1, 0, 0.35f, 1.f, 0.35f);
                    axis(0, 0, 1, 0.3f, 0.5f, 1.f);
                },
                py::arg("object"), py::arg("size") = 1.f,
                (std::string("Draw `object`'s world-space frame: X red, Y green, Z blue - the "
                             "one question every attitude bug comes down to. ") +
                 kDrawSeconds)
                        .c_str());
    }

}// namespace threepp_py
