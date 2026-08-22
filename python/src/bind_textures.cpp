// Textures and their enums. Texture derives from EventDispatcher (no virtual
// base), so it binds straightforwardly. Registered before materials so their
// texture-map slots resolve.
#include "bindings.hpp"

#include "threepp/constants.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"

#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>

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
                // In-place re-bake of a float_texture's pixels: same allocation,
                // same Texture object, new radiance. The point is the ping-pong
                // a moving sun needs — write the BACK texture, then swap
                // scene.environment onto it — without churning two allocations
                // per bake. Shape must match what the texture was created at.
                .def("update_float",
                     [](Texture& t, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
                         if (t.images().empty() || !t.image().isFloat())
                             throw std::runtime_error("update_float: not a float texture (see tp.float_texture)");
                         if (data.ndim() != 3)
                             throw std::runtime_error("update_float: expected a 3-D (height, width, channels) array");
                         auto& img = t.image();
                         const auto h = static_cast<unsigned int>(data.shape(0));
                         const auto w = static_cast<unsigned int>(data.shape(1));
                         const auto c = static_cast<unsigned int>(data.shape(2));
                         if (w != img.width() || h != img.height())
                             throw std::runtime_error("update_float: shape does not match the texture "
                                                      "(it is never resized — make a new one instead)");
                         if (c != 3 && c != 4)
                             throw std::runtime_error("update_float: channels must be 3 (RGB) or 4 (RGBA)");
                         auto& dst = img.data<float>();
                         dst.assign(static_cast<size_t>(w) * h * 4, 1.f);
                         const float* src = data.data();
                         for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                             dst[i * 4 + 0] = src[i * c + 0];
                             dst[i * 4 + 1] = src[i * c + 1];
                             dst[i * 4 + 2] = src[i * c + 2];
                             if (c == 4) dst[i * 4 + 3] = src[i * c + 3];
                         }
                         t.needsUpdate();
                     },
                     py::arg("data"),
                     "Rewrite a float texture's pixels in place from a (height, width, 3|4) float32 "
                     "array of the SAME size, and mark it dirty. For ping-ponging a procedural sky.")
                .def("needs_update", &Texture::needsUpdate)
                .def("update_matrix", &Texture::updateMatrix)
                .def("dispose", &Texture::dispose)
                .def("__repr__", [](const Texture& t) { return "<threepp.Texture name='" + t.name + "'>"; });

        // ── Textures authored in Python ───────────────────────────────────────
        // The counterpart to BufferGeometry.set_attribute: geometry could already
        // be built from numpy, but every texture had to arrive through a LOADER or
        // a purpose-built baker (TerrainGenerator.bake_splat_texture,
        // TreeTextures), so a procedural material was not expressible from Python
        // at all. RGB or RGBA uint8, row-major, row 0 is v = 0.
        m.def("data_texture",
              [](py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> data,
                 bool srgb) -> std::shared_ptr<Texture> {
                  if (data.ndim() != 3)
                      throw std::runtime_error("data_texture: expected a 3-D (height, width, channels) array");
                  const auto h = static_cast<unsigned int>(data.shape(0));
                  const auto w = static_cast<unsigned int>(data.shape(1));
                  const auto c = static_cast<unsigned int>(data.shape(2));
                  if (c != 3 && c != 4)
                      throw std::runtime_error("data_texture: channels must be 3 (RGB) or 4 (RGBA)");
                  if (w == 0 || h == 0)
                      throw std::runtime_error("data_texture: zero-sized image");
                  // The image is stored RGBA regardless: a 3-channel source is
                  // widened here rather than at every sampling site downstream.
                  std::vector<unsigned char> buf(static_cast<size_t>(w) * h * 4, 255);
                  const std::uint8_t* src = data.data();
                  for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                      buf[i * 4 + 0] = src[i * c + 0];
                      buf[i * 4 + 1] = src[i * c + 1];
                      buf[i * 4 + 2] = src[i * c + 2];
                      if (c == 4) buf[i * 4 + 3] = src[i * c + 3];
                  }
                  auto tex = DataTexture::create(ImageData(std::move(buf)), w, h);
                  // sRGB for anything the eye reads as a colour (albedo, emissive);
                  // Linear for data maps (roughness, metalness, AO) -- tagging a
                  // roughness map sRGB silently gamma-curves the values.
                  tex->colorSpace = srgb ? ColorSpace::sRGB : ColorSpace::Linear;
                  tex->magFilter = Filter::Linear;
                  tex->minFilter = Filter::LinearMipmapLinear;
                  tex->wrapS = TextureWrapping::Repeat;
                  tex->wrapT = TextureWrapping::Repeat;
                  tex->needsUpdate();
                  return tex;
              },
              py::arg("data"), py::arg("srgb") = true,
              "Build a Texture from a (height, width, 3|4) uint8 numpy array. "
              "srgb=True for colour maps (map, emissive_map); srgb=False for data "
              "maps (roughness_map, metalness_map, ao_map).");

        // ── HDR environments authored in Python ───────────────────────────────
        // data_texture above is uint8, which cannot carry a sky: an equirect env
        // spans a sun disc at ~10^4 and a night horizon at ~10^-3, and 8 bits
        // quantises both to nothing. This is the float sibling — the same buffer
        // C++ builds by hand for a procedural sky (examples/vulkan/vulkan_fjord),
        // so a numpy sky model can drive scene.environment / scene.background
        // and, on Vulkan, the IBL that lights the scene.
        //
        // Re-assigning scene.environment is NOT cheap (re-upload + PMREM +
        // device idle + a cold ReSTIR start), so a day/night cycle should
        // ping-pong two of these and re-bake on a throttle, not every frame.
        m.def("float_texture",
              [](py::array_t<float, py::array::c_style | py::array::forcecast> data) -> std::shared_ptr<Texture> {
                  if (data.ndim() != 3)
                      throw std::runtime_error("float_texture: expected a 3-D (height, width, channels) array");
                  const auto h = static_cast<unsigned int>(data.shape(0));
                  const auto w = static_cast<unsigned int>(data.shape(1));
                  const auto c = static_cast<unsigned int>(data.shape(2));
                  if (c != 3 && c != 4)
                      throw std::runtime_error("float_texture: channels must be 3 (RGB) or 4 (RGBA)");
                  if (w == 0 || h == 0)
                      throw std::runtime_error("float_texture: zero-sized image");
                  // Stored RGBA regardless, like data_texture: the upload path
                  // wants four channels and widening here beats widening at
                  // every sampling site.
                  std::vector<float> buf(static_cast<size_t>(w) * h * 4, 1.f);
                  const float* src = data.data();
                  for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                      buf[i * 4 + 0] = src[i * c + 0];
                      buf[i * 4 + 1] = src[i * c + 1];
                      buf[i * 4 + 2] = src[i * c + 2];
                      if (c == 4) buf[i * 4 + 3] = src[i * c + 3];
                  }
                  auto tex = Texture::create(Image(std::move(buf), w, h));
                  tex->format = Format::RGBA;
                  tex->type = Type::Float;
                  // Linear, always: an HDR buffer IS radiance. Tagging it sRGB
                  // would gamma-curve values that are already linear.
                  tex->colorSpace = ColorSpace::Linear;
                  // The one mapping a sky wants; harmless on a plain map slot,
                  // where the material's own UVs are used instead.
                  tex->mapping = Mapping::EquirectangularReflection;
                  tex->needsUpdate();
                  return tex;
              },
              py::arg("data"),
              "Build a float (HDR) Texture from a (height, width, 3|4) float32 numpy array: "
              "RGBA, linear, equirectangular. This is the sky/IBL texture type — assign it to "
              "scene.environment and scene.background. Row 0 is v = 0 (the zenith end of an "
              "equirect). Use Texture.update_float to re-bake one in place.");
    }

}// namespace threepp_py
