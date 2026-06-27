// Texture and model loaders. ModelLoader is the one-call entry point: it
// dispatches by file extension (.obj/.gltf/.glb/.stl/.dae) to first-party
// loaders — no Assimp/FBX/USD required.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#ifdef THREEPP_PY_HAS_FBX
#include "threepp/loaders/FBXLoader.hpp"
#endif
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

        // ---- rotate_equirect -------------------------------------------------
        // Resample an equirectangular (lat/long) float texture by a 3D rotation
        // (Euler XYZ, degrees), returning a NEW float-RGBA Texture. The renderer
        // samples env maps with a Y-up convention (latitude = asin(dir.y)), so a
        // Z-up scene needs its HDRI pitched: rotate_equirect(env, -90, 0, 0).
        // y_deg then spins the map about the new up axis to aim the sun.
        m.def("rotate_equirect",
              [](const std::shared_ptr<Texture>& src, float xDeg, float yDeg, float zDeg) -> std::shared_ptr<Texture> {
                  if (!src) throw std::runtime_error("rotate_equirect: null texture");
                  const Image& img = src->image();
                  if (!img.isFloat() || img.channels() != 4)
                      throw std::runtime_error("rotate_equirect: expected a float RGBA equirect (e.g. from RGBELoader)");
                  const int W = static_cast<int>(img.width());
                  const int H = static_cast<int>(img.height());
                  const std::vector<float>& s = img.data<float>();

                  // R = Rx·Ry·Rz, applied to each output texel's Y-up direction;
                  // the rotated direction is looked up in the source equirect.
                  const float d2r = 0.01745329252f;
                  const float cx = std::cos(xDeg * d2r), sx = std::sin(xDeg * d2r);
                  const float cy = std::cos(yDeg * d2r), sy = std::sin(yDeg * d2r);
                  const float cz = std::cos(zDeg * d2r), sz = std::sin(zDeg * d2r);
                  const float r00 = cy * cz,               r01 = -cy * sz,              r02 = sy;
                  const float r10 = sx * sy * cz + cx * sz, r11 = -sx * sy * sz + cx * cz, r12 = -sx * cy;
                  const float r20 = -cx * sy * cz + sx * sz, r21 = cx * sy * sz + sx * cz, r22 = cx * cy;

                  const float TWO_PI = 6.28318530718f, PI_ = 3.14159265359f;
                  std::vector<float> out(static_cast<size_t>(W) * H * 4);

                  auto sample = [&](float u, float v, int c) -> float {
                      u -= std::floor(u);                            // wrap azimuth
                      v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);       // clamp poles
                      const float fx = u * W - 0.5f, fy = v * H - 0.5f;
                      const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
                      const float tx = fx - x0, ty = fy - y0;
                      auto px = [&](int x, int y) -> float {
                          x = ((x % W) + W) % W;
                          y = y < 0 ? 0 : (y >= H ? H - 1 : y);
                          return s[(static_cast<size_t>(y) * W + x) * 4 + c];
                      };
                      const float a = px(x0, y0) * (1 - tx) + px(x0 + 1, y0) * tx;
                      const float b = px(x0, y0 + 1) * (1 - tx) + px(x0 + 1, y0 + 1) * tx;
                      return a * (1 - ty) + b * ty;
                  };

                  for (int iy = 0; iy < H; ++iy) {
                      const float v = (iy + 0.5f) / H, lat = (v - 0.5f) * PI_;
                      const float cl = std::cos(lat), sl = std::sin(lat);
                      for (int ix = 0; ix < W; ++ix) {
                          const float u = (ix + 0.5f) / W, az = (u - 0.5f) * TWO_PI;
                          const float Dx = cl * std::cos(az), Dy = sl, Dz = cl * std::sin(az);
                          const float ex = r00 * Dx + r01 * Dy + r02 * Dz;
                          const float ey = r10 * Dx + r11 * Dy + r12 * Dz;
                          const float ez = r20 * Dx + r21 * Dy + r22 * Dz;
                          const float su = 0.5f + std::atan2(ez, ex) / TWO_PI;
                          const float sv = 0.5f + std::asin(ey < -1.f ? -1.f : (ey > 1.f ? 1.f : ey)) / PI_;
                          const size_t o = (static_cast<size_t>(iy) * W + ix) * 4;
                          out[o + 0] = sample(su, sv, 0);
                          out[o + 1] = sample(su, sv, 1);
                          out[o + 2] = sample(su, sv, 2);
                          out[o + 3] = 1.0f;
                      }
                  }

                  Image outImg{ImageData(std::move(out)), static_cast<unsigned int>(W), static_cast<unsigned int>(H)};
                  auto tex = Texture::create(outImg);
                  tex->name = src->name + "_rot";
                  tex->format = Format::RGBA;
                  tex->type = Type::Float;
                  tex->colorSpace = ColorSpace::Linear;
                  tex->mapping = Mapping::EquirectangularReflection;
                  tex->wrapS = src->wrapS;
                  tex->wrapT = src->wrapT;
                  tex->needsUpdate();
                  return tex;
              },
              py::arg("texture"), py::arg("x_deg") = 0.f, py::arg("y_deg") = 0.f, py::arg("z_deg") = 0.f,
              "Resample an equirect float texture by a 3D rotation (Euler XYZ degrees). "
              "For a Z-up scene with a Y-up HDRI: rotate_equirect(env, -90, 0, 0).");

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

#ifdef THREEPP_PY_HAS_FBX
        // ---- FBXLoader (OpenFBX-backed) --------------------------------------
        // Only present when the threepp lib was built with -DTHREEPP_WITH_FBX=ON.
        // Returns a Group, like the other model loaders.
        {
            py::class_<FBXLoader> fbx(m, "FBXLoader");
            py::enum_<FBXLoader::MaterialMode>(fbx, "MaterialMode")
                    .value("Auto", FBXLoader::MaterialMode::Auto)
                    .value("Phong", FBXLoader::MaterialMode::Phong)
                    .value("PBR", FBXLoader::MaterialMode::PBR);
            fbx.def(py::init<>())
                    .def_readwrite("material_mode", &FBXLoader::materialMode,
                                   "How the FBX SPECULAR slot is interpreted (Auto/Phong/PBR).")
                    .def_readwrite("emissive_scale", &FBXLoader::emissiveScale,
                                   "Multiplier on every emissive material's intensity (1.0 = file values).")
                    .def("load", [](FBXLoader& l, const std::string& path) { return l.load(path); },
                         py::arg("path"), "Load an .fbx file as a Group.");
        }
#endif
    }

}// namespace threepp_py
