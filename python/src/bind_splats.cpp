// 3D Gaussian Splatting: the .ply loader and the SplatCloud scene object.
//
// NOT gated on the Vulkan backend: SplatCloud renders on the GL backend (it is
// a Mesh over an instanced unit quad; see SplatCloud.hpp), and the Vulkan
// deferred renderer draws the same object through its own SplatPass — so a
// GL-only build gets splats too, unlike Ocean/DisplacedMesh whose pipeline
// exists only under Vulkan.
//
// Surface for v1 is deliberately the consumption path: load a 3DGS optimiser's
// .ply, put the cloud in a scene, render it, and read what it costs. The
// LodTable / loadSogWithLod machinery and raw per-splat data access stay
// unbound until their API has settled.
#include "bindings.hpp"

#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"

#include <utility>

using namespace threepp;

namespace threepp_py {

    void init_splats(py::module_& m) {

        // Opaque carrier between the loader and the cloud. count() is enough
        // introspection for v1; the mutation API (outlier removal, Morton
        // reorder) waits until someone needs it from Python.
        py::class_<SplatData>(m, "SplatData")
                .def_property_readonly("count", &SplatData::count,
                                       "Number of splats held.")
                .def("__len__", &SplatData::count)
                .def("__repr__", [](const SplatData& d) {
                    return "<SplatData count=" + std::to_string(d.count()) + ">";
                });

        py::class_<SplatLoader>(m, "SplatLoader")
                .def(py::init<>())
                .def_static("load_ply", &SplatLoader::loadPly, py::arg("path"),
                            "Load a 3D-Gaussian-Splatting .ply (the format 3DGS optimisers emit) "
                            "into a SplatData. Header-driven: files with/without normals, extra "
                            "per-splat properties or any SH degree all parse. Raises RuntimeError "
                            "with the offending property in the message on anything unrepresentable.")
                .def_static("is_splat_ply", py::overload_cast<const std::filesystem::path&>(&SplatLoader::isSplatPly),
                            py::arg("path"),
                            "Does this .ply hold Gaussian splats rather than a mesh? Reads only the "
                            "header and never raises — a missing or malformed file is simply False.")
                .def_static("write_ply",
                            [](const SplatData& data, const std::filesystem::path& path) {
                                SplatLoader::writePly(data, path);
                            },
                            py::arg("data"), py::arg("path"),
                            "Write a SplatData as a 3DGS .ply (the INRIA layout load_ply reads: "
                            "channel-major f_rest, log scales, logit opacity, w-first rotation). "
                            "Raises RuntimeError if the file cannot be written.")
                .def_static("is_point_cloud_ply",
                            py::overload_cast<const std::filesystem::path&>(&SplatLoader::isPointCloudPly),
                            py::arg("path"),
                            "Does this .ply hold a colour-only point cloud (x/y/z, no f_dc_0, no "
                            "faces)? Header only, never raises.")
                .def_static(
                        "load_point_cloud_ply",
                        [](const std::filesystem::path& path, float sigma, float sigma_per_spacing,
                           float opacity, bool use_normals, float normal_thickness) {
                            SplatLoader::PointCloudOptions o;
                            o.sigma = sigma;
                            o.sigmaPerSpacing = sigma_per_spacing;
                            o.opacity = opacity;
                            o.useNormals = use_normals;
                            o.normalThickness = normal_thickness;
                            return SplatLoader::loadPointCloudPly(path, o);
                        },
                        py::arg("path"), py::arg("sigma") = 0.f, py::arg("sigma_per_spacing") = 1.0f,
                        py::arg("opacity") = 1.f, py::arg("use_normals") = true,
                        py::arg("normal_thickness") = 0.15f,
                        "Load a colour-only point-cloud .ply (binary or ascii; red/green/blue, "
                        "nx/ny/nz and intensity honoured) as a SplatData of degree-0 Gaussians, "
                        "one per point. sigma 0 sizes them from the cloud's median neighbour "
                        "spacing times sigma_per_spacing; a point with normals becomes a disc "
                        "facing them. Render with SplatCloud.point_mix = 1 for dots, 0 for a "
                        "closed surface.");

        // Bound as a subclass of the already-registered Mesh: Mesh is a
        // non-virtual base of SplatCloud, so the concrete Object3D API bound on
        // Mesh (position/rotation/add/...) is inherited safely — the same
        // pattern as DisplacedMesh and InstancedMesh.
        py::class_<SplatCloud, Mesh, std::shared_ptr<SplatCloud>>(m, "SplatCloud")
                .def(py::init([](SplatData& data) {
                         // CONSUMES data (leaves it empty): a scan is hundreds of
                         // bytes per splat and millions of splats, and copying
                         // 1.4 GB to preserve a Python handle nobody reuses is
                         // the wrong default. Load again if you need two clouds.
                         return SplatCloud::create(std::move(data));
                     }),
                     py::arg("data"),
                     "Create a renderable splat cloud. CONSUMES `data` (it is left empty) "
                     "to avoid copying gigabyte-scale scans; load the file again if you "
                     "need a second cloud.")
                .def("update", &SplatCloud::update, py::arg("camera"),
                     "Sort splats back-to-front for this camera and refresh per-frame state. "
                     "Cheap when nothing moved. The object's own pre-render hook also calls "
                     "this, but one frame LATE for single-shot captures — call it explicitly "
                     "before render() when grabbing a single frame headless.")
                .def_property_readonly("splat_count", &SplatCloud::splatCount,
                                       "Number of splats in the cloud.")
                .def_property_readonly("cpu_bytes", &SplatCloud::cpuBytes,
                                       "Host memory this cloud holds (bytes): splat data + sort scratch, "
                                       "plus the GL data textures once a GL frame has built them. "
                                       "~423 B/splat at SH degree 3 after a GL draw.")
                .def("set_viewport_size", &SplatCloud::setViewportSize,
                     py::arg("width"), py::arg("height"),
                     "Override the framebuffer pixel size used for splat scaling — only needed "
                     "when rendering into a target whose size differs from the renderer's own.")
                .def_property("point_mix", &SplatCloud::pointMix, &SplatCloud::setPointMix,
                              "0 (default) renders Gaussians; 1 renders every splat as an opaque "
                              "disc of point_size pixels at its centre, nearest wins — the point "
                              "cloud view. Values between dissolve one into the other. Same depth "
                              "sort and mesh occlusion on both backends.")
                .def_property("point_size", &SplatCloud::pointSize, &SplatCloud::setPointSize,
                              "Disc diameter in pixels at point_mix 1. Floored at 1; default 2.")
                .def_property("debug_non_finite",
                              &SplatCloud::debugNonFinite,
                              &SplatCloud::setDebugNonFinite,
                              "Draw non-finite colour fragments magenta instead of discarding them, "
                              "so a cloud quietly producing NaNs stops looking identical to a healthy "
                              "one. Default off.")
                .def_property("submit_ranges",
                              &SplatCloud::submitRanges,
                              [](SplatCloud& c, std::vector<std::pair<uint32_t, uint32_t>> r) {
                                  c.setSubmitRanges(std::move(r));
                              },
                              "Partial submission: a list of (offset, count) ranges into this cloud's "
                              "splats to draw this frame, in order; empty (default) draws all. The "
                              "chunk-LOD/culling mechanism. VULKAN ONLY today — the GL path draws "
                              "every splat regardless. At most 64 ranges are honoured.");
    }

}// namespace threepp_py
