// Animation: keyframe tracks, clips, the mixer that drives a scene-graph root,
// and the per-clip actions (play/stop/fade/loop). Mirrors three.js' animation
// system. Clips usually come from GLTFLoader (see bind_loaders.cpp), but can
// also be built procedurally from KeyframeTracks here.
//
//     mixer  = tp.AnimationMixer(model)            # root Object3D
//     action = mixer.clip_action(clips[0]).play()  # fluent, three.js-style
//     ...
//     mixer.update(clock.get_delta())              # per frame
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/animation/AnimationAction.hpp"
#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/animation/KeyframeTrack.hpp"
#include "threepp/animation/tracks/ColorKeyframeTrack.hpp"
#include "threepp/animation/tracks/NumberKeyframeTrack.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/core/Object3D.hpp"

#include <optional>

using namespace threepp;

namespace threepp_py {

    // Every keyframe-track subclass shares the same (name, times, values,
    // interpolation?) constructor. Register it as a shared_ptr-held subclass of
    // KeyframeTrack so a Python list of mixed tracks up-casts cleanly to
    // vector<shared_ptr<KeyframeTrack>> for AnimationClip (KeyframeTrack is a
    // plain, non-virtual base — no Object3D-style pointer-adjustment hazard).
    template<class Track>
    static void bind_track(py::module_& m, const char* name) {
        py::class_<Track, KeyframeTrack, std::shared_ptr<Track>>(m, name)
                .def(py::init([](const std::string& trackName, std::vector<float> times,
                                 std::vector<float> values, std::optional<Interpolation> interp) {
                         return std::make_shared<Track>(trackName, times, values, interp);
                     }),
                     py::arg("name"), py::arg("times"), py::arg("values"),
                     py::arg("interpolation") = std::nullopt);
    }

    void init_animation(py::module_& m) {

        // ---- Enums -----------------------------------------------------------
        py::enum_<Loop>(m, "Loop", "Looping mode for an AnimationAction.")
                .value("ONCE", Loop::Once)
                .value("REPEAT", Loop::Repeat)
                .value("PING_PONG", Loop::PingPong);

        py::enum_<AnimationBlendMode>(m, "AnimationBlendMode")
                .value("NORMAL", AnimationBlendMode::Normal)
                .value("ADDITIVE", AnimationBlendMode::Additive);

        py::enum_<Interpolation>(m, "Interpolation", "Keyframe interpolation mode.")
                .value("DISCRETE", Interpolation::Discrete)
                .value("LINEAR", Interpolation::Linear)
                .value("SMOOTH", Interpolation::Smooth);

        // ---- KeyframeTrack (abstract base) -----------------------------------
        // Track names follow three.js: "<node>.<property>" (e.g.
        // "mixamorigHips.position", "Cube.quaternion"); a leading "." (empty
        // node) targets the mixer's root object directly (".position").
        py::class_<KeyframeTrack, std::shared_ptr<KeyframeTrack>>(m, "KeyframeTrack")
                .def_property_readonly("name", &KeyframeTrack::getName)
                .def_property_readonly("value_type_name", &KeyframeTrack::ValueTypeName)
                .def_property_readonly("times", &KeyframeTrack::getTimes)
                .def_property_readonly("values", &KeyframeTrack::getValues)
                .def_property_readonly("interpolation", &KeyframeTrack::getInterpolation)
                .def("set_interpolation", [](KeyframeTrack& t, Interpolation i) { t.setInterpolation(i); }, py::arg("interpolation"))
                .def("make_additive", &KeyframeTrack::makeAdditive,
                     "Convert this track's values to deltas from its first frame (for additive layering).");

        // Concrete tracks. vector → position/scale, quaternion → rotation,
        // number → scalars (opacity, morph weights, material floats), color → RGB.
        bind_track<VectorKeyframeTrack>(m, "VectorKeyframeTrack");
        bind_track<QuaternionKeyframeTrack>(m, "QuaternionKeyframeTrack");
        bind_track<NumberKeyframeTrack>(m, "NumberKeyframeTrack");
        bind_track<ColorKeyframeTrack>(m, "ColorKeyframeTrack");

        // ---- AnimationClip ---------------------------------------------------
        py::class_<AnimationClip, std::shared_ptr<AnimationClip>>(m, "AnimationClip")
                .def(py::init<std::string, float, const std::vector<std::shared_ptr<KeyframeTrack>>&, AnimationBlendMode>(),
                     py::arg("name"), py::arg("duration") = 1.f,
                     py::arg("tracks") = std::vector<std::shared_ptr<KeyframeTrack>>{},
                     py::arg("blend_mode") = AnimationBlendMode::Normal)
                .def_property_readonly("name", &AnimationClip::name)
                .def_property_readonly("duration", &AnimationClip::getDuration)
                .def_property_readonly("uuid", &AnimationClip::uuid)
                .def_readwrite("blend_mode", &AnimationClip::blendMode)
                .def("reset_duration", &AnimationClip::resetDuration,
                     "Recompute duration as the maximum track end time.")
                .def("make_additive", &AnimationClip::makeAdditive,
                     "Convert every track to additive form and mark the clip Additive.")
                .def_static("find_by_name", [](const std::vector<std::shared_ptr<AnimationClip>>& clips, const std::string& name) {
                    return AnimationClip::findByName(clips, name);
                }, py::arg("clips"), py::arg("name"))
                .def("__repr__", [](const AnimationClip& c) {
                    return "<threepp.AnimationClip name='" + c.name() + "' duration=" + std::to_string(c.getDuration()) + ">";
                });

        // ---- AnimationAction -------------------------------------------------
        // Created via AnimationMixer.clip_action; owned by the mixer (never
        // constructed directly). The fluent setters return self for chaining.
        // Held by shared_ptr because AnimationAction derives from
        // enable_shared_from_this (pybind requires the matching holder); the
        // mixer's reservoir keeps a shared owner, so the raw pointer we hand back
        // resolves to that same instance via shared_from_this.
        py::class_<AnimationAction, std::shared_ptr<AnimationAction>>(m, "AnimationAction")
                .def_readwrite("blend_mode", &AnimationAction::blendMode)
                .def("play", &AnimationAction::play, py::return_value_policy::reference)
                .def("stop", &AnimationAction::stop, py::return_value_policy::reference)
                .def("reset", &AnimationAction::reset, py::return_value_policy::reference)
                .def("is_running", &AnimationAction::isRunning)
                .def("is_scheduled", &AnimationAction::isScheduled)
                .def("start_at", &AnimationAction::startAt, py::arg("time"), py::return_value_policy::reference)
                .def("set_loop", &AnimationAction::setLoop, py::arg("mode"), py::arg("repetitions") = -1, py::return_value_policy::reference)
                .def("set_effective_weight", &AnimationAction::setEffectiveWeight, py::arg("weight"), py::return_value_policy::reference)
                .def("get_effective_weight", &AnimationAction::getEffectiveWeight)
                .def("fade_in", &AnimationAction::fadeIn, py::arg("duration"), py::return_value_policy::reference)
                .def("fade_out", &AnimationAction::fadeOut, py::arg("duration"), py::return_value_policy::reference)
                .def("cross_fade_to", &AnimationAction::crossFadeTo, py::arg("other"), py::arg("duration"), py::arg("warp") = false, py::return_value_policy::reference)
                .def("stop_fading", &AnimationAction::stopFading, py::return_value_policy::reference)
                .def("set_effective_time_scale", &AnimationAction::setEffectiveTimeScale, py::arg("time_scale"), py::return_value_policy::reference)
                .def("get_effective_time_scale", &AnimationAction::getEffectiveTimeScale)
                .def("set_duration", &AnimationAction::setDuration, py::arg("duration"), py::return_value_policy::reference)
                .def("sync_with", &AnimationAction::syncWith, py::arg("action"), py::return_value_policy::reference)
                .def("halt", &AnimationAction::halt, py::arg("duration"), py::return_value_policy::reference)
                .def("warp", &AnimationAction::warp, py::arg("start_time_scale"), py::arg("end_time_scale"), py::arg("duration"), py::return_value_policy::reference)
                .def("stop_warping", &AnimationAction::stopWarping, py::return_value_policy::reference);

        // ---- AnimationMixer --------------------------------------------------
        // Drives a root Object3D's animation. keep_alive<1,2> ties the root's
        // lifetime to the mixer: the mixer stores a bare Object3D* (it does not
        // own the root), so the Python root must outlive it.
        py::class_<AnimationMixer>(m, "AnimationMixer")
                .def(py::init([](const py::object& root) {
                         return std::make_unique<AnimationMixer>(*as_object3d(root));
                     }),
                     py::arg("root"), py::keep_alive<1, 2>())
                .def_readwrite("time", &AnimationMixer::time)
                .def_readwrite("time_scale", &AnimationMixer::timeScale)
                .def("clip_action", [](AnimationMixer& mx, const std::shared_ptr<AnimationClip>& clip, std::optional<AnimationBlendMode> blendMode) {
                    return mx.clipAction(clip, nullptr, blendMode);
                }, py::arg("clip"), py::arg("blend_mode") = std::nullopt, py::return_value_policy::reference_internal,
                     "Return (creating if needed) the AnimationAction for a clip on this mixer's root.")
                .def("stop_all_action", &AnimationMixer::stopAllAction)
                .def("update", &AnimationMixer::update, py::arg("dt"),
                     "Advance all active actions by dt seconds and write the result into the scene graph.");
    }

}// namespace threepp_py
