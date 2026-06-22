// Texture and model loaders. ModelLoader is the one-call entry point: it
// dispatches by file extension (.obj/.gltf/.glb/.stl/.dae) to first-party
// loaders — no Assimp/FBX/USD required.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/STLLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;

namespace threepp_py {

    void init_loaders(py::module_& m) {

        // ---- TextureLoader ---------------------------------------------------
        // Pass color_space=ColorSpace.SRGB for albedo/emissive maps; leave the
        // default (NoColorSpace) for data maps (normal, roughness, ao, ...).
        py::class_<TextureLoader>(m, "TextureLoader")
                .def(py::init<bool>(), py::arg("use_cache") = true)
                .def("load", [](TextureLoader& l, const std::string& path, bool flip_y) { return l.load(path, flip_y); },
                     py::arg("path"), py::arg("flip_y") = true)
                .def("load", [](TextureLoader& l, const std::string& path, ColorSpace cs, bool flip_y) { return l.load(path, cs, flip_y); },
                     py::arg("path"), py::arg("color_space"), py::arg("flip_y") = true)
                .def("clear_cache", &TextureLoader::clearCache);

        // ---- RGBELoader (Radiance .hdr -> float equirect Texture) ------------
        // The returned texture (float RGBA, EquirectangularReflection mapping,
        // linear color space) is ready to assign to scene.environment for IBL on
        // standard/physical materials, or to scene.background for an HDR backdrop.
        // The GL renderer PMREM-prefilters it into a real image-based light.
        py::class_<RGBELoader>(m, "RGBELoader")
                .def(py::init<>())
                .def("load", [](RGBELoader& l, const std::string& path, bool flip_y) { return l.load(path, flip_y); },
                     py::arg("path"), py::arg("flip_y") = true,
                     "Load a Radiance .hdr equirectangular environment as a float Texture.");

        // ---- ModelLoader (dispatch-by-extension) -----------------------------
        py::class_<ModelLoader>(m, "ModelLoader")
                .def(py::init<>())
                .def("load", [](ModelLoader& l, const std::string& path) { return l.load(path); }, py::arg("path"),
                     "Load a model (.obj/.gltf/.glb/.stl/.dae) as a Group.")
                .def("set_ignore_up_direction", [](ModelLoader& l, bool ignore) { l.setIgnoreUpDirection(ignore); }, py::arg("ignore"));

        // ---- OBJLoader -------------------------------------------------------
        py::class_<OBJLoader>(m, "OBJLoader")
                .def(py::init<>())
                .def("load", [](OBJLoader& l, const std::string& path, bool try_load_mtl) { return l.load(path, try_load_mtl); },
                     py::arg("path"), py::arg("try_load_mtl") = true);

        // ---- STLLoader (geometry only) ---------------------------------------
        py::class_<STLLoader>(m, "STLLoader")
                .def(py::init<>())
                .def("load", [](const STLLoader& l, const std::string& path) { return l.load(path); }, py::arg("path"));

        // ---- GLTFResult ------------------------------------------------------
        // Mirrors three.js' `gltf` object: the loaded scene plus its animation
        // clips. `scene` is the convenient root; `animations` feeds an
        // AnimationMixer (see bind_animation.cpp). Returned by GLTFLoader.load.
        py::class_<GLTFResult>(m, "GLTFResult")
                .def_readonly("scene", &GLTFResult::scene, "Root Group of the loaded model.")
                .def_readonly("scenes", &GLTFResult::scenes, "All scenes in the file.")
                .def_readonly("animations", &GLTFResult::animations, "All AnimationClips in the file.")
                .def("__repr__", [](const GLTFResult& r) {
                    return "<threepp.GLTFResult scenes=" + std::to_string(r.scenes.size()) +
                           " animations=" + std::to_string(r.animations.size()) + ">";
                });

        // ---- GLTFLoader (returns scene + animations) -------------------------
        // Unlike ModelLoader (geometry only, returns a Group), GLTFLoader hands
        // back the full result so animations survive. Raises RuntimeError if the
        // file can't be loaded.
        py::class_<GLTFLoader>(m, "GLTFLoader")
                .def(py::init<>())
                .def("load", [](GLTFLoader& l, const std::string& path) -> GLTFResult {
                    auto result = l.load(path);
                    if (!result) throw std::runtime_error("GLTFLoader: failed to load '" + path + "'");
                    return std::move(*result);
                }, py::arg("path"));
    }

}// namespace threepp_py
