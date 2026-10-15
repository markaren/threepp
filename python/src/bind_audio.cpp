// Audio bindings: AudioListener (3D listener in scene graph), Audio (positional-
// independent playback), PositionalAudio (spatialized sound as Object3D).
// No-op when built without audio support (THREEPP_WITH_AUDIO=OFF).
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_AUDIO

#include <pybind11/stl.h>

#include "threepp/audio/Audio.hpp"
#include "threepp/audio/WavFile.hpp"
#include "threepp/core/Object3D.hpp"

using namespace threepp;

#endif

namespace threepp_py {

    void init_audio(py::module_& m) {

#ifdef THREEPP_PY_HAS_AUDIO

        // ---- AudioListener ---------------------------------------------------
        py::class_<AudioListener, Object3D, std::shared_ptr<AudioListener>>(m, "AudioListener")
                .def(py::init([]() { return std::make_shared<AudioListener>(); }))
                .def_property("master_volume",
                              &AudioListener::getMasterVolume,
                              &AudioListener::setMasterVolume)
                .def("set_reverb", &AudioListener::setReverb, py::arg("rt60"), py::arg("wet"),
                     "Retune the shared reverb bus every sound sends to: rt60 is the decay time in "
                     "seconds, wet the bus output gain [0, 1]. Each sound sets its own share with "
                     "set_reverb_send.");

        // ---- Audio -----------------------------------------------------------
        // Non-copyable, non-movable: uses unique_ptr holder (pybind11 default).
        // Must be kept alive as long as it is in use; caller owns it.
        py::class_<Audio>(m, "Audio")
                .def(py::init([](AudioListener& listener, const std::string& file) {
                         return std::make_unique<Audio>(listener, file);
                     }),
                     py::arg("listener"), py::arg("file"), py::keep_alive<1, 2>())
                .def_property_readonly("is_playing", &Audio::isPlaying)
                .def("play", &Audio::play)
                .def("stop", &Audio::stop)
                .def("set_volume", &Audio::setVolume, py::arg("volume"))
                .def("set_playback_rate", &Audio::setPlaybackRate, py::arg("rate"),
                     "Playback rate / pitch multiplier (1 = as recorded). Applies from the next audio "
                     "period, a few ms, playing or not: ramp it to bend a loop's pitch, or re-roll it per "
                     "trigger. A rate <= 0 is ignored.")
                .def("set_reverb_send", &Audio::setReverbSend, py::arg("send"),
                     "How much of this sound goes to the listener's reverb bus [0, 1] "
                     "(AudioListener.set_reverb). A send: the dry signal keeps its own path.")
                .def("toggle_play", &Audio::togglePlay)
                .def("seek_to_start", &Audio::seekToStart)
                .def("set_looping", &Audio::setLooping, py::arg("loop"));

        // ---- PositionalAudio -------------------------------------------------
        // Inherits from both Audio and Object3D. Registered with Object3D as the
        // pybind11 base (so scene.add() works); Audio methods exposed directly.
        py::enum_<PositionalAudio::DistanceModel>(m, "AudioDistanceModel")
                .value("NONE", PositionalAudio::DistanceModel::None)
                .value("INVERSE", PositionalAudio::DistanceModel::Inverse)
                .value("LINEAR", PositionalAudio::DistanceModel::Linear)
                .value("EXPONENTIAL", PositionalAudio::DistanceModel::Exponential)
                .export_values();

        py::class_<PositionalAudio, Object3D, std::shared_ptr<PositionalAudio>>(m, "PositionalAudio")
                .def(py::init([](AudioListener& listener, const std::string& file) {
                         return std::make_shared<PositionalAudio>(listener, file);
                     }),
                     py::arg("listener"), py::arg("file"), py::keep_alive<1, 2>())
                // Audio methods
                .def_property_readonly("is_playing", [](PositionalAudio& self) { return self.isPlaying(); })
                .def("play", [](PositionalAudio& self) { self.play(); })
                .def("stop", [](PositionalAudio& self) { self.stop(); })
                .def("set_volume", [](PositionalAudio& self, float v) { self.setVolume(v); }, py::arg("volume"))
                .def("set_playback_rate", [](PositionalAudio& self, float r) { self.setPlaybackRate(r); }, py::arg("rate"),
                     "Playback rate / pitch multiplier (1 = as recorded). Applies from the next audio "
                     "period, a few ms, playing or not. A rate <= 0 is ignored.")
                .def("set_reverb_send", [](PositionalAudio& self, float s) { self.setReverbSend(s); }, py::arg("send"),
                     "How much of this sound goes to the listener's reverb bus [0, 1].")
                .def("toggle_play", [](PositionalAudio& self) { self.togglePlay(); })
                .def("seek_to_start", [](PositionalAudio& self) { self.seekToStart(); })
                .def("set_looping", [](PositionalAudio& self, bool loop) { self.setLooping(loop); }, py::arg("loop"))
                // PositionalAudio-specific
                .def("set_min_distance", &PositionalAudio::setMinDistance, py::arg("distance"))
                .def("set_max_distance", &PositionalAudio::setMaxDistance, py::arg("distance"))
                .def("set_rolloff_factor", &PositionalAudio::setRolloffFactor, py::arg("rolloff"))
                .def("set_distance_model", &PositionalAudio::setDistanceModel, py::arg("model"))
                .def("set_occlusion", &PositionalAudio::setOcclusion, py::arg("occlusion"),
                     "0 = unobstructed, 1 = fully occluded: a low-pass filter and a gain dip on this "
                     "sound. Call at most once per frame with a smoothed value.");

        // ---- write_wav -------------------------------------------------------
        m.def("write_wav", [](const std::string& path, const std::vector<float>& samples, int sample_rate) {
                  audio::writeWav(path, samples, sample_rate);
              },
              py::arg("path"), py::arg("samples"), py::arg("sample_rate") = 44100,
              "Write a mono 16-bit PCM WAV file from normalised float samples in [-1, 1].");

        m.attr("HAS_AUDIO") = true;

#else
        m.attr("HAS_AUDIO") = false;
#endif
    }

}// namespace threepp_py
