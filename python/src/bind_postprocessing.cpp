// The GL studio kit: RoomEnvironment + PMREMGenerator (image-based light
// without an HDR file) and the EffectComposer post-processing chain
// (RenderPass, OutputPass, UnrealBloomPass, BokehPass, OutlinePass, GTAOPass).
// All of it is OpenGL only: the Vulkan renderer owns its own post chain.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/postprocessing/BokehPass.hpp"
#include "threepp/postprocessing/EffectComposer.hpp"
#include "threepp/postprocessing/GTAOPass.hpp"
#include "threepp/postprocessing/OutlinePass.hpp"
#include "threepp/postprocessing/OutputPass.hpp"
#include "threepp/postprocessing/Pass.hpp"
#include "threepp/postprocessing/RenderPass.hpp"
#include "threepp/postprocessing/UnrealBloomPass.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace threepp_py {

    namespace {

        // OutlinePass keeps its selection as RAW Object3D pointers. The Python
        // binding therefore owns the selection too: this subclass holds a
        // shared_ptr to every selected object, so an object selected from
        // Python cannot be freed while the pass still points at it. Plain C++
        // ownership, so it needs no GIL whenever the pass is destroyed.
        struct PyOutlinePass: OutlinePass {
            using OutlinePass::OutlinePass;
            std::vector<std::shared_ptr<Object3D>> held;

            void select(const py::iterable& objects) {
                std::vector<std::shared_ptr<Object3D>> next;
                for (const auto& h : objects) next.emplace_back(as_object3d(h));
                held = std::move(next);
                selectedObjects.clear();
                for (const auto& o : held) selectedObjects.emplace_back(o.get());
            }
        };

        Type parse_type(const std::string& s) {
            if (s == "HalfFloat") return Type::HalfFloat;
            if (s == "Float") return Type::Float;
            if (s == "UnsignedByte") return Type::UnsignedByte;
            throw std::invalid_argument("EffectComposer: type must be 'HalfFloat', 'Float' or 'UnsignedByte'");
        }

        py::array_t<uint8_t> rgb_to_numpy(const std::vector<unsigned char>& buf, int w, int h, bool flip) {
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w), static_cast<py::ssize_t>(3)});
            const size_t rowBytes = static_cast<size_t>(w) * 3;
            auto* dst = arr.mutable_data();
            if (buf.size() >= rowBytes * static_cast<size_t>(h)) {
                for (int row = 0; row < h; ++row) {
                    const auto srcRow = static_cast<size_t>(flip ? (h - 1 - row) : row);
                    std::memcpy(dst + static_cast<size_t>(row) * rowBytes, buf.data() + srcRow * rowBytes, rowBytes);
                }
            }
            return arr;
        }

    }// namespace

    void init_postprocessing(py::module_& m) {

        // ---- RoomEnvironment -------------------------------------------------
        py::class_<RoomEnvironment, Scene, std::shared_ptr<RoomEnvironment>>(
                m, "RoomEnvironment",
                "A neutral studio (a white room with a few boxes and six light panels) that is not "
                "looked at but turned into an environment map, for image-based light without an HDR "
                "file:\n\n"
                "    pmrem = tp.PMREMGenerator(renderer)\n"
                "    scene.environment = pmrem.from_scene(tp.RoomEnvironment(), 0.04)\n")
                .def(py::init([] { return RoomEnvironment::create(); }))
                .def("dispose", &RoomEnvironment::dispose, "Free the geometry and materials the room owns.");

        // ---- PMREMGenerator (GL) ---------------------------------------------
        py::class_<PMREMGenerator>(m, "PMREMGenerator",
                                   "Turns a Scene into an environment map (GLRenderer only). Keeps the renderer alive.")
                .def(py::init([](GLRenderer& r) { return std::make_unique<PMREMGenerator>(r); }),
                     py::arg("renderer"), py::keep_alive<1, 2>())
                .def("from_scene",
                     [](PMREMGenerator& g, Scene& scene, float sigma, float near, float far,
                        int size, const std::optional<Vector3>& position) {
                         PMREMFromSceneOptions opts;
                         opts.size = size;
                         if (position) opts.position = *position;
                         return g.fromScene(scene, sigma, near, far, opts);
                     },
                     py::arg("scene"), py::arg("sigma") = 0.f, py::arg("near") = 0.1f, py::arg("far") = 100.f,
                     py::arg("size") = 256, py::arg("position") = py::none(),
                     "Render `scene` into a cube from `position` and return it as an equirectangular "
                     "HalfFloat environment Texture, for scene.environment (or scene.background). "
                     "`sigma` is a Gaussian blur in radians (0.04 suits RoomEnvironment); `size` is the "
                     "cube face edge. The texture lives on the GPU of the renderer that made it and "
                     "stays valid while referenced, independent of this generator.");

        // ---- Pass (abstract base) -------------------------------------------
        py::class_<Pass, std::shared_ptr<Pass>>(m, "Pass", "One step of an EffectComposer chain.")
                .def_readwrite("enabled", &Pass::enabled, "A disabled pass is skipped, including its buffer swap.")
                .def_readwrite("need_swap", &Pass::needsSwap, "Whether the composer swaps read/write buffers after this pass.")
                .def_readwrite("clear", &Pass::clear, "Whether the pass clears its target before drawing.")
                .def_readwrite("render_to_screen", &Pass::renderToScreen,
                               "Set by the composer; the composer always ends its chain with its own output pass.")
                .def("set_size", &Pass::setSize, py::arg("width"), py::arg("height"));

        // ---- EffectComposer --------------------------------------------------
        py::class_<EffectComposer>(m, "EffectComposer",
                                   "A chain of full-screen passes (GLRenderer only). Every pass works on linear HDR; "
                                   "the composer tone-maps once at the end through an implicit OutputPass.")
                .def(py::init([](GLRenderer& r, unsigned int samples, const std::optional<std::string>& type,
                                 bool depthBuffer, bool stencilBuffer) {
                         EffectComposer::Options o;
                         o.samples = samples;
                         o.depthBuffer = depthBuffer;
                         o.stencilBuffer = stencilBuffer;
                         if (type) o.type = parse_type(*type);
                         return std::make_unique<EffectComposer>(r, o);
                     }),
                     py::arg("renderer"), py::arg("samples") = 0u, py::arg("type") = py::none(),
                     py::arg("depth_buffer") = true, py::arg("stencil_buffer") = true, py::keep_alive<1, 2>(),
                     "samples: MSAA samples for the internal targets (0 = off). type: 'HalfFloat' (default), "
                     "'Float' or 'UnsignedByte' (cheaper, but bands in dark gradients).")
                .def_readwrite("render_to_screen", &EffectComposer::renderToScreen,
                               "False keeps the result offscreen (read it with read_pixels()).")
                .def("add_pass", &EffectComposer::addPass, py::arg("pass_"))
                .def("insert_pass", &EffectComposer::insertPass, py::arg("pass_"), py::arg("index"))
                .def("remove_pass", [](EffectComposer& c, const std::shared_ptr<Pass>& p) { c.removePass(p.get()); },
                     py::arg("pass_"))
                .def_property_readonly("passes", [](const EffectComposer& c) { return c.passes(); },
                                       "A fresh list of the passes (editing it does not change the chain).")
                .def("render", &EffectComposer::render, py::arg("delta_time") = 0.f)
                .def("set_size", &EffectComposer::setSize, py::arg("width"), py::arg("height"),
                     "Resize the internal targets (framebuffer pixels; the pixel ratio is applied here).")
                .def("set_pixel_ratio", &EffectComposer::setPixelRatio, py::arg("pixel_ratio"))
                .def("get_pixel_ratio", &EffectComposer::getPixelRatio)
                .def(
                        "read_pixels",
                        [](EffectComposer& c, GLRenderer& r, bool flip) {
                            std::vector<unsigned char> buf;
                            {
                                py::gil_scoped_release release;
                                buf = c.readRGBPixels();
                            }
                            // readBuffer() is sized in framebuffer pixels.
                            const auto s = r.size();
                            const float pr = c.getPixelRatio();
                            const int w = static_cast<int>(static_cast<float>(s.width()) * pr);
                            const int h = static_cast<int>(static_cast<float>(s.height()) * pr);
                            return rgb_to_numpy(buf, w, h, flip);
                        },
                        py::arg("renderer"), py::arg("flip") = true,
                        "The chain's finished image as an (H, W, 3) uint8 array (use with render_to_screen=False; "
                        "linear HDR clamped unless the chain ends in an OutputPass).");

        // ---- RenderPass ------------------------------------------------------
        py::class_<RenderPass, Pass, std::shared_ptr<RenderPass>>(m, "RenderPass", "Draws the scene: the head of nearly every chain.")
                .def(py::init([](const py::handle& scene, Camera& camera) {
                         return std::make_shared<RenderPass>(*as_object3d(scene), camera);
                     }),
                     py::arg("scene"), py::arg("camera"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
                .def_property(
                        "override_material",
                        [](RenderPass& p) { return material_to_py(p.overrideMaterial); },
                        [](RenderPass& p, const py::object& mat) { p.overrideMaterial = as_material(mat); })
                .def_readwrite("clear_color", &RenderPass::clearColor, "Color to clear to instead of the renderer's, or None.")
                .def_readwrite("clear_alpha", &RenderPass::clearAlpha)
                .def_readwrite("clear_depth", &RenderPass::clearDepth);

        // ---- OutputPass ------------------------------------------------------
        py::class_<OutputPass, Pass, std::shared_ptr<OutputPass>>(
                m, "OutputPass",
                "Tone mapping + output colour space. The composer appends one implicitly; add it "
                "explicitly only when a later pass needs display-ready input.")
                .def(py::init([] { return std::make_shared<OutputPass>(); }));

        // ---- UnrealBloomPass -------------------------------------------------
        py::class_<UnrealBloomPass, Pass, std::shared_ptr<UnrealBloomPass>>(m, "UnrealBloomPass")
                .def(py::init([](const Vector2& resolution, float strength, float radius, float threshold) {
                         return std::make_shared<UnrealBloomPass>(resolution, strength, radius, threshold);
                     }),
                     py::arg("resolution") = Vector2(256, 256), py::arg("strength") = 1.f,
                     py::arg("radius") = 0.f, py::arg("threshold") = 0.f)
                .def_readwrite("strength", &UnrealBloomPass::strength)
                .def_readwrite("radius", &UnrealBloomPass::radius, "0..1; higher spreads the glow further.")
                .def_readwrite("threshold", &UnrealBloomPass::threshold, "Luminance a pixel must exceed to bloom.");

        // ---- BokehPass -------------------------------------------------------
        py::class_<BokehPass, Pass, std::shared_ptr<BokehPass>>(m, "BokehPass", "Depth of field.")
                .def(py::init([](Scene& scene, Camera& camera, float focus, float aperture, float maxblur) {
                         return std::make_shared<BokehPass>(scene, camera, focus, aperture, maxblur);
                     }),
                     py::arg("scene"), py::arg("camera"), py::arg("focus") = 1.f, py::arg("aperture") = 0.025f,
                     py::arg("maxblur") = 0.01f, py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
                .def_readwrite("focus", &BokehPass::focus, "Distance to the focus plane along the view axis.")
                .def_readwrite("aperture", &BokehPass::aperture, "Larger is shallower (less in focus).")
                .def_readwrite("maxblur", &BokehPass::maxblur, "Ceiling on the blur radius, in UV units.")
                .def_readwrite("aspect", &BokehPass::aspect);

        // ---- OutlinePass -----------------------------------------------------
        py::class_<OutlinePass, Pass, std::shared_ptr<OutlinePass>>(
                m, "OutlinePass",
                "A glowing outline around the selected objects (visible_edge_color where in view, "
                "hidden_edge_color where something hides them). Only meshes and sprites take part.")
                .def(py::init([](const Vector2& resolution, Scene& scene, Camera& camera, const py::object& selected) {
                         auto p = std::make_shared<PyOutlinePass>(resolution, scene, camera);
                         if (!selected.is_none()) p->select(selected);
                         return std::shared_ptr<OutlinePass>(p);
                     }),
                     py::arg("resolution"), py::arg("scene"), py::arg("camera"), py::arg("selected_objects") = py::none(),
                     py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
                .def_property(
                        "selected_objects",
                        [](OutlinePass& p) {
                            py::list out;
                            if (auto* pp = dynamic_cast<PyOutlinePass*>(&p)) {
                                for (const auto& o : pp->held) out.append(py::cast(o));
                            }
                            return out;
                        },
                        [](OutlinePass& p, const py::iterable& objects) {
                            auto* pp = dynamic_cast<PyOutlinePass*>(&p);
                            if (!pp) throw std::runtime_error("OutlinePass: not created from Python");
                            pp->select(objects);
                        },
                        "The objects to outline (a Group outlines every mesh under it). The getter returns a "
                        "FRESH list: `p.selected_objects.append(m)` changes nothing. Assign a whole list "
                        "instead: `p.selected_objects = [m]`. The pass keeps the objects alive.")
                .def_readwrite("visible_edge_color", &OutlinePass::visibleEdgeColor)
                .def_readwrite("hidden_edge_color", &OutlinePass::hiddenEdgeColor)
                .def_readwrite("edge_strength", &OutlinePass::edgeStrength, "Multiplies the edge (linear light) before it is added.")
                .def_readwrite("edge_glow", &OutlinePass::edgeGlow, "A wider, softer copy of the edge; 0 is a crisp line.")
                .def_readwrite("edge_thickness", &OutlinePass::edgeThickness, "Edge blur width in downsampled pixels, up to 4.")
                .def_readwrite("pulse_period", &OutlinePass::pulsePeriod, "Seconds-ish period of a brightness pulse; 0 = off.")
                .def_readwrite("use_pattern_texture", &OutlinePass::usePatternTexture)
                .def_readwrite("pattern_texture", &OutlinePass::patternTexture)
                .def_readwrite("down_sample_ratio", &OutlinePass::downSampleRatio, "Takes effect on the next set_size.")
                .def_readwrite("resolution", &OutlinePass::resolution);

        // ---- GTAOPass --------------------------------------------------------
        auto gtao = py::class_<GTAOPass, Pass, std::shared_ptr<GTAOPass>>(
                m, "GTAOPass",
                "Ground-truth ambient occlusion: darkens creases, corners and contact points. Renders the "
                "scene once more per frame for normals + depth, then denoises and multiplies the AO in.");
        py::enum_<GTAOPass::Output>(gtao, "Output")
                .value("Off", GTAOPass::Output::Off)
                .value("Default", GTAOPass::Output::Default)
                .value("Diffuse", GTAOPass::Output::Diffuse)
                .value("Depth", GTAOPass::Output::Depth)
                .value("Normal", GTAOPass::Output::Normal)
                .value("AO", GTAOPass::Output::AO)
                .value("Denoise", GTAOPass::Output::Denoise);
        gtao.def(py::init([](Scene& scene, Camera& camera, unsigned int width, unsigned int height) {
                     return std::make_shared<GTAOPass>(scene, camera, width, height);
                 }),
                 py::arg("scene"), py::arg("camera"), py::arg("width") = 512u, py::arg("height") = 512u,
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
                .def_readwrite("output", &GTAOPass::output, "What reaches the next pass: the lit image with AO, or a buffer.")
                .def_readwrite("blend_intensity", &GTAOPass::blendIntensity, "0 = no AO, 1 = full AO.")
                .def(
                        "update_gtao_material",
                        [](GTAOPass& p, std::optional<float> radius, std::optional<float> distanceExponent,
                           std::optional<float> thickness, std::optional<float> distanceFallOff,
                           std::optional<float> scale, std::optional<int> samples, std::optional<bool> screenSpaceRadius) {
                            GTAOParameters g;
                            g.radius = radius;
                            g.distanceExponent = distanceExponent;
                            g.thickness = thickness;
                            g.distanceFallOff = distanceFallOff;
                            g.scale = scale;
                            g.samples = samples;
                            g.screenSpaceRadius = screenSpaceRadius;
                            p.updateGtaoMaterial(g);
                        },
                        py::kw_only(), py::arg("radius") = py::none(), py::arg("distance_exponent") = py::none(),
                        py::arg("thickness") = py::none(), py::arg("distance_fall_off") = py::none(),
                        py::arg("scale") = py::none(), py::arg("samples") = py::none(),
                        py::arg("screen_space_radius") = py::none(),
                        "Change the AO estimate; only the keywords given are touched. radius is the world-space "
                        "reach (0.25), thickness the depth gap beyond which a sample stops occluding (1), "
                        "scale an exponent on the AO (1), samples the depth taps per pixel (16).")
                .def(
                        "update_pd_material",
                        [](GTAOPass& p, std::optional<float> lumaPhi, std::optional<float> depthPhi,
                           std::optional<float> normalPhi, std::optional<float> radius,
                           std::optional<float> radiusExponent, std::optional<float> rings, std::optional<int> samples) {
                            PoissonDenoiseParameters d;
                            d.lumaPhi = lumaPhi;
                            d.depthPhi = depthPhi;
                            d.normalPhi = normalPhi;
                            d.radius = radius;
                            d.radiusExponent = radiusExponent;
                            d.rings = rings;
                            d.samples = samples;
                            p.updatePdMaterial(d);
                        },
                        py::kw_only(), py::arg("luma_phi") = py::none(), py::arg("depth_phi") = py::none(),
                        py::arg("normal_phi") = py::none(), py::arg("radius") = py::none(),
                        py::arg("radius_exponent") = py::none(), py::arg("rings") = py::none(),
                        py::arg("samples") = py::none(),
                        "Change the Poisson denoise that follows the AO; only the keywords given are touched.")
                .def("set_scene_clip_box", &GTAOPass::setSceneClipBox, py::arg("box"),
                     "Fade AO out beyond `box` (a Box3), so a small scene does not shade a distant backdrop. None removes it.");
    }

}// namespace threepp_py
