// Texture and model loaders. ModelLoader is the one-call entry point: it
// dispatches by file extension (.obj/.gltf/.glb/.stl/.dae) to first-party
// loaders — no Assimp/FBX/USD required.
#include "bindings.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/OBJLoader.hpp"
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

        // ---- GLTFLoader (returns the scene Group) ----------------------------
        py::class_<GLTFLoader>(m, "GLTFLoader")
                .def(py::init<>())
                .def("load", [](GLTFLoader& l, const std::string& path) -> std::shared_ptr<Group> {
                    auto result = l.load(path);
                    return result ? result->scene : nullptr;
                }, py::arg("path"));
    }

}// namespace threepp_py
