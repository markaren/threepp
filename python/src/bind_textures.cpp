// Textures and their enums. Texture derives from EventDispatcher (no virtual
// base), so it binds straightforwardly. Registered before materials so their
// texture-map slots resolve.
#include "bindings.hpp"

#include "threepp/constants.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;

namespace threepp_py {

    void init_textures(py::module_& m) {

        py::enum_<TextureWrapping>(m, "TextureWrapping")
                .value("Repeat", TextureWrapping::Repeat)
                .value("ClampToEdge", TextureWrapping::ClampToEdge)
                .value("MirroredRepeat", TextureWrapping::MirroredRepeat);

        py::enum_<Filter>(m, "Filter")
                .value("Nearest", Filter::Nearest)
                .value("NearestMipmapNearest", Filter::NearestMipmapNearest)
                .value("NearestMipmapLinear", Filter::NearestMipmapLinear)
                .value("Linear", Filter::Linear)
                .value("LinearMipmapNearest", Filter::LinearMipmapNearest)
                .value("LinearMipmapLinear", Filter::LinearMipmapLinear);

        py::enum_<Mapping>(m, "Mapping")
                .value("UV", Mapping::UV)
                .value("CubeReflection", Mapping::CubeReflection)
                .value("CubeRefraction", Mapping::CubeRefraction)
                .value("EquirectangularReflection", Mapping::EquirectangularReflection)
                .value("EquirectangularRefraction", Mapping::EquirectangularRefraction);

        py::enum_<ColorSpace>(m, "ColorSpace")
                .value("NoColorSpace", ColorSpace::NoColorSpace)
                .value("Linear", ColorSpace::Linear)
                .value("SRGB", ColorSpace::sRGB);

        py::class_<Texture, std::shared_ptr<Texture>>(m, "Texture")
                .def(py::init([] { return Texture::create(); }))
                .def_readwrite("name", &Texture::name)
                .def_readwrite("wrap_s", &Texture::wrapS)
                .def_readwrite("wrap_t", &Texture::wrapT)
                .def_readwrite("mag_filter", &Texture::magFilter)
                .def_readwrite("min_filter", &Texture::minFilter)
                .def_readwrite("anisotropy", &Texture::anisotropy)
                .def_readwrite("offset", &Texture::offset)
                .def_readwrite("repeat", &Texture::repeat)
                .def_readwrite("center", &Texture::center)
                .def_readwrite("rotation", &Texture::rotation)
                .def_readwrite("mapping", &Texture::mapping)
                .def_readwrite("generate_mipmaps", &Texture::generateMipmaps)
                .def_readwrite("color_space", &Texture::colorSpace)
                .def("needs_update", &Texture::needsUpdate)
                .def("update_matrix", &Texture::updateMatrix)
                .def("dispose", &Texture::dispose)
                .def("__repr__", [](const Texture& t) { return "<threepp.Texture name='" + t.name + "'>"; });
    }

}// namespace threepp_py
