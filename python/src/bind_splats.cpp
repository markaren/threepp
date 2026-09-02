// 3D Gaussian Splatting: the .ply loader and the SplatCloud scene object.
//
// NOT gated on the Vulkan backend: SplatCloud renders on the GL backend (it is
// a Mesh over an instanced unit quad; see SplatCloud.hpp), and the Vulkan
// deferred renderer draws the same object through its own SplatPass — so a
// GL-only build gets splats too, unlike Ocean/DisplacedMesh whose pipeline
// exists only under Vulkan.
//
// Surface for v1 was deliberately the consumption path: load a 3DGS optimiser's
// .ply, put the cloud in a scene, render it, and read what it costs. v2 adds
// the SCAN path a robot demo needs: the SOG container (SogLoader), dynamic LOD
// (loadSogWithLod + selectLod) and the depth-fusion surface bake
// (splats::bakeSurface) that gives a scan collision and sensor geometry. Raw
// per-splat data access stays unbound until its API has settled.
//
// bake_surface is VULKAN ONLY — the median-depth AOV it fuses is a Vulkan
// G-buffer attachment. In a GL-only build the def still exists and raises
// RuntimeError, so a script fails with a sentence rather than an AttributeError.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"
#include "threepp/splats/SplatLod.hpp"
#include "threepp/splats/SplatSurface.hpp"

#ifdef THREEPP_PY_HAS_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cstdint>
#include <stdexcept>
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

        // ── SOG / SSOG: the compressed scan container ────────────────────────
        // Same consumption shape as SplatLoader, one level at a time. `path` is
        // a directory holding lod-meta.json (a whole asset) or meta.json (one
        // chunk), either of those files by name, or a ZIP/.sog archive of them.
        py::class_<SogLoader>(m, "SogLoader")
                .def(py::init<>())
                .def_static("is_sog", &SogLoader::isSog, py::arg("path"),
                            "Is this a SOG / SuperSplat SSOG asset? Answers by CONTENT (the "
                            "meta.json / lod-meta.json inside), not by name — a SOG asset is "
                            "usually an extensionless directory. Never raises: a missing or "
                            "malformed asset is simply False.")
                .def_static(
                        "describe",
                        [](const std::filesystem::path& path) {
                            const auto info = SogLoader::describe(path);
                            py::list levels;
                            py::list counts;
                            for (const auto& lv : info.levels) {
                                py::dict d;
                                d["lod"] = lv.lod;
                                d["count"] = lv.count;
                                d["chunks"] = lv.chunks.size();
                                levels.append(std::move(d));
                                counts.append(lv.count);
                            }
                            py::dict out;
                            out["lod_levels"] = info.lodLevels;
                            out["sh_degree"] = info.shDegree;
                            out["levels"] = std::move(levels);
                            out["counts"] = std::move(counts);
                            out["bound_min"] = info.bound.min();
                            out["bound_max"] = info.bound.max();
                            return out;
                        },
                        py::arg("path"),
                        "What the asset holds, WITHOUT decoding a plane — json only, "
                        "milliseconds against the gigabyte load() of level 0 costs. Returns a "
                        "dict: lod_levels, sh_degree, counts (splats per level, finest first), "
                        "levels (per-level dicts with lod/count/chunks) and bound_min/max. "
                        "Levels are ALTERNATIVES, not a residual pyramid: each covers the whole "
                        "scene at its own density, so the counts do not add up to a total.")
                .def_static(
                        "load",
                        [](const std::filesystem::path& path, int level) {
                            SogLoader::Options o;
                            o.lod = level;
                            return SogLoader::load(path, o);
                        },
                        py::arg("path"), py::arg("level") = 0,
                        py::call_guard<py::gil_scoped_release>(),
                        "Read ONE detail level of a SOG asset into a SplatData. 0 is the finest "
                        "(and the only legal value for a lone chunk, which declares no levels). "
                        "Seconds and gigabytes at level 0 on a real scan — call describe() first "
                        "and pick a coarser level if that is not what you want. Raises "
                        "RuntimeError naming the offending member, file and numbers.\n\n"
                        "FRAME: the file's own coordinates, untouched. A SOG re-encoded from a "
                        "COLMAP .ply is +Y DOWN like its .ply, so the caller flips it the same "
                        "way — cloud.rotation.x = math.pi is what the editor and the "
                        "gaussian_splats example do.");

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
                              "every splat regardless. At most 64 ranges are honoured.")
                .def_static(
                        "from_sog_lod",
                        [](const std::filesystem::path& path) {
                            splats::SogLodResult loaded;
                            {
                                py::gil_scoped_release release;// seconds of parsing
                                loaded = splats::loadSogWithLod(path);
                            }
                            auto cloud = SplatCloud::create(std::move(loaded.data));
                            cloud->setLodTable(std::move(loaded.table));
                            return cloud;
                        },
                        py::arg("path"),
                        "Load a multi-level SOG asset for DYNAMIC LOD: every other level "
                        "(0, 2, 4, ...) resident in one cloud, with the LOD table set on it. "
                        "Every other because residency is the sum of the levels and it is paid "
                        "twice (GL textures + the Vulkan pass); adjacent levels differ by only "
                        "2x, and keeping level 0 is what the close-up invariant needs.\n\n"
                        "Then call select_lod(cloud, camera, viewport_h) once per frame: it "
                        "writes submit_ranges for the level and chunks that frame should draw. "
                        "A single-level asset comes back with an EMPTY table — it renders "
                        "plainly and select_lod is a no-op returning 0.")
                .def_property_readonly(
                        "lod_levels",
                        [](const SplatCloud& c) {
                            py::list out;
                            for (const auto& lv : c.lodTable().levels) {
                                py::dict d;
                                d["lod"] = lv.lod;
                                d["base"] = lv.base;
                                d["count"] = lv.count;
                                d["chunks"] = lv.chunks.size();
                                out.append(std::move(d));
                            }
                            return out;
                        },
                        "The resident detail levels as dicts (lod, base, count, chunks), finest "
                        "first — empty unless the cloud came from from_sog_lod on a multi-level "
                        "asset. `base` is the level's first splat index in this cloud.")
                .def_property_readonly(
                        "lod_held_level",
                        [](const SplatCloud& c) { return c.lodTable().heldLevel; },
                        "Index into lod_levels of the level select_lod is currently holding "
                        "(hysteresis state lives on the cloud's own table). 0 without LOD. On "
                        "the per-node path this is the FINEST level in use this frame, since "
                        "there is no single level any more.")
                .def_property_readonly(
                        "lod_node_count",
                        [](const SplatCloud& c) { return c.lodTable().nodes.size(); },
                        "Leaves of the asset's SSOG tree resident in this cloud. 0 when the "
                        "asset carries no tree (or no per-node offsets), which is exactly when "
                        "select_lod falls back to whole-cloud selection.")
                .def_property_readonly(
                        "lod_node_levels",
                        [](const SplatCloud& c) {
                            py::list out;
                            for (const auto& n : c.lodTable().nodes) out.append(n.frameLevel);
                            return out;
                        },
                        "The level index each tree node was submitted at by the last per-node "
                        "select_lod, -1 for a node the frustum culled. Length lod_node_count; "
                        "all -1 before the first per-node selection. A histogram of this is the "
                        "readable form of 'what did LOD actually do this frame'.");

        // ── Dynamic LOD policy ───────────────────────────────────────────────
        m.def(
                "select_lod",
                [](SplatCloud& cloud, const Camera& camera, int viewport_height_px,
                   float target_splats_per_pixel, float hysteresis, bool per_node) {
                    if (per_node) {
                        return splats::selectLod(cloud, cloud.lodTable(), camera, viewport_height_px,
                                                 target_splats_per_pixel, hysteresis);
                    }
                    return splats::selectLodWholeCloud(cloud, cloud.lodTable(), camera,
                                                       viewport_height_px, target_splats_per_pixel,
                                                       hysteresis);
                },
                py::arg("cloud"), py::arg("camera"), py::arg("viewport_height_px"),
                py::arg("target_splats_per_pixel") = 1.f, py::arg("hysteresis") = 1.25f,
                py::arg("per_node") = true,
                "Choose what this frame draws, and write it into the cloud's submit_ranges. "
                "Returns a level index into cloud.lod_levels (not the asset's own lod number).\n\n"
                "PER NODE by default, whenever the asset carried an SSOG tree "
                "(cloud.lod_node_count > 0): every leaf of that tree gets its own level, so a "
                "near wall stays fine while the far end of the canyon coarsens. Without a tree "
                "this is the whole-cloud rule — one level for everything, then its chunks "
                "against the frustum — which is also what per_node=False forces, for A/B.\n\n"
                "target_splats_per_pixel means the same thing on both paths: splats per SCREEN "
                "pixel for the whole visible cloud. The per-node rule derives its own per-leaf "
                "threshold by dividing it by the frame's overdraw factor (the visible leaves' "
                "footprints summed over the screen area), so the total submitted count lands "
                "near target_splats_per_pixel * screen pixels by construction and the SAME "
                "argument value keeps its meaning across the switch. Raise it when the camera "
                "stands INSIDE the scan; the calico demo runs 8.\n\n"
                "Call once per frame, before render(), with the RENDER resolution's height. The "
                "cloud carries its own hysteresis state (per node on the per-node path), so "
                "nothing has to be kept on the Python side. A cloud with no LOD table is left "
                "alone and 0 comes back.");

        // ── Surface bake: a scan becomes triangles ───────────────────────────
        // Where the generated capture cameras stand, which is the difference
        // between baking a room and baking the block it is inside.
        py::enum_<splats::SurfaceBakeOptions::PoseSet>(m, "SplatPoseSet")
                .value("Orbit", splats::SurfaceBakeOptions::PoseSet::Orbit,
                       "Outside the fit sphere looking IN. Right for an object, a facade, an "
                       "outdoor site seen from outside. The default.")
                .value("Interior", splats::SurfaceBakeOptions::PoseSet::Interior,
                       "Inside the scan looking OUT, from the fit centre plus jittered "
                       "stations. Right for a room, a canyon, anything the camera stands IN. "
                       "Orbiting such a scan reconstructs the OUTSIDE of its walls and never "
                       "observes the walkable volume at all.");

        py::class_<splats::BakePose>(m, "BakePose",
                                     "One capture viewpoint for bake_surface. `fov` is vertical "
                                     "degrees; the bake renders at the renderer's own framebuffer "
                                     "extent, so the horizontal field follows from its aspect.")
                .def(py::init([](const Vector3& position, const Vector3& target, const Vector3& up, float fov) {
                         splats::BakePose p;
                         p.position = position;
                         p.target = target;
                         p.up = up;
                         p.fov = fov;
                         return p;
                     }),
                     py::arg("position"), py::arg("target"),
                     py::arg("up") = Vector3(0.f, 1.f, 0.f), py::arg("fov") = 55.f)
                .def_readwrite("position", &splats::BakePose::position)
                .def_readwrite("target", &splats::BakePose::target)
                .def_readwrite("up", &splats::BakePose::up)
                .def_readwrite("fov", &splats::BakePose::fov);

        py::class_<splats::SurfaceMesh>(
                m, "SurfaceMesh",
                "The triangle surface bake_surface fused out of a splat cloud, in WORLD space "
                "(the cloud's transform is already in the vertices). Three consumers: "
                "make_sensor_mesh() for the renderer's sensors, PhysxWorld.add_static_trimesh "
                "for collision, and to_geometry() for anything else.")
                .def_property_readonly(
                        "positions",
                        [](const splats::SurfaceMesh& s) {
                            const auto n = static_cast<py::ssize_t>(s.vertexCount());
                            py::array_t<float> out({n, static_cast<py::ssize_t>(3)});
                            if (n > 0) std::copy(s.positions.begin(), s.positions.end(), out.mutable_data());
                            return out;
                        },
                        "Vertices as an (N, 3) float32 array, world space. A copy.")
                .def_property_readonly(
                        "indices",
                        [](const splats::SurfaceMesh& s) {
                            const auto n = static_cast<py::ssize_t>(s.triangleCount());
                            py::array_t<std::uint32_t> out({n, static_cast<py::ssize_t>(3)});
                            if (n > 0) std::copy(s.indices.begin(), s.indices.end(), out.mutable_data());
                            return out;
                        },
                        "Triangles as an (M, 3) uint32 array of vertex indices. A copy.")
                .def_property_readonly("triangle_count",
                                       [](const splats::SurfaceMesh& s) { return s.triangleCount(); })
                .def_property_readonly("vertex_count",
                                       [](const splats::SurfaceMesh& s) { return s.vertexCount(); })
                .def_property_readonly("empty", [](const splats::SurfaceMesh& s) { return s.empty(); })
                .def_property_readonly(
                        "stats",
                        [](const splats::SurfaceMesh& s) {
                            const auto& st = s.stats;
                            py::dict d;
                            d["poses"] = st.poses;
                            d["voxel_size"] = st.voxelSize;
                            d["truncation"] = st.truncation;
                            d["max_depth"] = st.maxDepth;
                            d["depth_samples"] = st.depthSamples;
                            d["skipped_fringe"] = st.skippedFringe;
                            d["skipped_outlier"] = st.skippedOutlier;
                            d["skipped_far"] = st.skippedFar;
                            d["blocks"] = st.blocks;
                            d["peak_block_bytes"] = st.peakBlockBytes;
                            d["refused_blocks"] = st.refusedBlocks;
                            d["observed_voxels"] = st.observedVoxels;
                            d["beyond_centre_samples"] = st.beyondCentreSamples;
                            d["carve_skipped_blocks"] = st.carveSkippedBlocks;
                            d["carve_bulk_blocks"] = st.carveBulkBlocks;
                            d["carve_voxel_blocks"] = st.carveVoxelBlocks;
                            d["components"] = st.components;
                            d["culled_components"] = st.culledComponents;
                            d["culled_triangles"] = st.culledTriangles;
                            d["aabb_min"] = st.aabbMin;
                            d["aabb_max"] = st.aabbMax;
                            d["render_ms"] = st.renderMs;
                            d["fuse_ms"] = st.fuseMs;
                            d["mesh_ms"] = st.meshMs;
                            return d;
                        },
                        "What the bake did and what it dropped, as a dict. The ones that "
                        "diagnose an empty or wrong bake: observed_voxels 0 means no voxel ever "
                        "met weight_floor (too few poses agreed); refused_blocks > 0 means "
                        "max_block_bytes bit; beyond_centre_samples a large fraction of "
                        "depth_samples under PoseSet.Orbit means YOUR SCAN MAY BE AN INTERIOR — "
                        "re-bake with PoseSet.Interior. render_ms/fuse_ms/mesh_ms split the cost.")
                .def(
                        "to_geometry",
                        [](const splats::SurfaceMesh& s) {
                            auto g = BufferGeometry::create();
                            g->setAttribute("position",
                                            FloatBufferAttribute::create(s.positions, 3));
                            g->setIndex(std::vector<unsigned int>(s.indices.begin(), s.indices.end()));
                            g->computeVertexNormals();
                            return g;
                        },
                        "The surface as a BufferGeometry (position + index + computed normals), "
                        "ready for tp.Mesh(geom, material). The hand route is the same thing "
                        "spelled out: BufferGeometry().set_attribute('position', s.positions) "
                        "then .set_index(s.indices.reshape(-1)). A tp.Mesh built either way is "
                        "what PhysxWorld.add_static_trimesh takes — or skip both and hand it "
                        "make_sensor_mesh(s), which is already a Mesh over these triangles.")
                .def("__repr__", [](const splats::SurfaceMesh& s) {
                    return "<SurfaceMesh vertices=" + std::to_string(s.vertexCount()) +
                           " triangles=" + std::to_string(s.triangleCount()) + ">";
                });

        m.def(
                "bake_surface",
                [](const py::handle& renderer, SplatCloud& cloud, float voxel_size, float truncation,
                   float truncation_voxels, float max_weight, float weight_floor, float max_depth,
                   bool carve_fast_paths, std::uint64_t max_block_bytes,
                   splats::SurfaceBakeOptions::PoseSet pose_set,
                   const std::vector<splats::BakePose>& poses, int pose_count, float pose_distance,
                   int min_component_voxels, int fringe_erode, float outlier_tolerance) {
#ifdef THREEPP_PY_HAS_VULKAN
                    Renderer* base = py_vulkan_native_renderer(renderer);
                    auto* vk = dynamic_cast<VulkanRenderer*>(base);
                    if (!vk) throw std::runtime_error("bake_surface: needs a VulkanRenderer (the median-depth AOV it fuses is a Vulkan G-buffer attachment)");

                    splats::SurfaceBakeOptions o;
                    o.voxelSize = voxel_size;
                    o.truncation = truncation;
                    o.truncationVoxels = truncation_voxels;
                    o.maxWeight = max_weight;
                    o.weightFloor = weight_floor;
                    o.maxDepth = max_depth;
                    o.carveFastPaths = carve_fast_paths;
                    o.maxBlockBytes = max_block_bytes;
                    o.poseSet = pose_set;
                    o.poses = poses;
                    o.poseCount = pose_count;
                    o.poseDistance = pose_distance;
                    o.minComponentVoxels = min_component_voxels;
                    o.fringeErode = fringe_erode;
                    o.outlierTolerance = outlier_tolerance;

                    splats::SurfaceMesh out;
                    {
                        py::gil_scoped_release release;// tens of GPU frames + fusion
                        out = splats::bakeSurface(*vk, cloud, o);
                    }
                    return out;
#else
                    (void) renderer; (void) cloud; (void) voxel_size; (void) truncation;
                    (void) truncation_voxels; (void) max_weight; (void) weight_floor;
                    (void) max_depth; (void) carve_fast_paths; (void) max_block_bytes;
                    (void) pose_set; (void) poses; (void) pose_count; (void) pose_distance;
                    (void) min_component_voxels; (void) fringe_erode; (void) outlier_tolerance;
                    throw std::runtime_error("bake_surface: this build has no Vulkan backend (the median-depth AOV it fuses is a Vulkan G-buffer attachment)");
#endif
                },
                py::arg("renderer"), py::arg("cloud"), py::arg("voxel_size") = 0.f,
                py::arg("truncation") = 0.f, py::arg("truncation_voxels") = 4.f,
                py::arg("max_weight") = 32.f, py::arg("weight_floor") = 2.f,
                py::arg("max_depth") = 0.f, py::arg("carve_fast_paths") = true,
                py::arg("max_block_bytes") = static_cast<std::uint64_t>(1) << 30,
                py::arg("pose_set") = splats::SurfaceBakeOptions::PoseSet::Orbit,
                py::arg("poses") = std::vector<splats::BakePose>{}, py::arg("pose_count") = 26,
                py::arg("pose_distance") = 0.f, py::arg("min_component_voxels") = 256,
                py::arg("fringe_erode") = 1, py::arg("outlier_tolerance") = 1.f,
                "Fuse a Gaussian-splat cloud into a triangle SurfaceMesh, by rendering it from "
                "a set of poses and integrating the median-depth AOV into a TSDF. VULKAN ONLY. "
                "Deterministic: the same cloud baked twice gives the same vertices and indices, "
                "bit for bit.\n\n"
                "Bake AFTER the cloud has its final transform — the vertices come out in WORLD "
                "space, so a later rotation of the cloud does not move them.\n\n"
                "The knobs that matter first:\n"
                "  pose_set    Orbit (default) stands OUTSIDE looking in; Interior stands "
                "inside looking out. A canyon, a room, anything the camera is IN needs "
                "Interior — Orbit reconstructs the outside of its walls instead.\n"
                "  poses       an explicit list of BakePose overrides pose_set entirely; the "
                "answer for replaying a real capture trajectory.\n"
                "  voxel_size  0 derives it from the cloud's robust fit (radius / 256, clamped "
                "to 5 mm .. 10 cm).\n"
                "  weight_floor is COUNTED IN POSES: 2 means two viewpoints had to agree. It "
                "can never be met by fewer poses than its own value.\n"
                "  min_component_voxels drops islands smaller than this many surface CELLS "
                "(voxel_size^2 each) — photogrammetry floaters that survived carving.\n\n"
                "An empty result is diagnosed from .stats: see its docstring.");

        m.def(
                "make_sensor_mesh",
                [](const splats::SurfaceMesh& surface) { return splats::makeSensorMesh(surface); },
                py::arg("surface"),
                "The baked surface as a Mesh that ONLY THE SENSORS perceive. Add it at the "
                "SCENE ROOT, not under the cloud: the vertices are world space already.\n\n"
                "It is inert until the scene opts in with "
                "VulkanRenderer.set_sensor_only_surfaces(True); after that a lidar scan hits "
                "it, and so does any secondary view that also asks "
                "(set_view_sensor_surfaces(handle, True)). The primary camera NEVER draws it "
                "and no radiance trace sees it — the real splats render there, which is the "
                "whole point. Returns None for an empty surface.\n\n"
                "It is also an ordinary triangle Mesh, so PhysxWorld.add_static_trimesh(mesh) "
                "takes it directly: one bake is the ground for the feet and the ground for the "
                "sensors.");
    }

}// namespace threepp_py
