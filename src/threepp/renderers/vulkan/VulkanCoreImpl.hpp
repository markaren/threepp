// Private implementation header — the shared Impl for VulkanRenderer.cpp.
// Never include from public API headers.
// VMA_IMPLEMENTATION must be #defined in VulkanRenderer.cpp BEFORE including this file.
#pragma once

#include "CpuPar.hpp"
#include "EnvKnob.hpp"
#include "VulkanContext.hpp"
#include "VulkanResources.hpp"
#include "VulkanRetireQueue.hpp"
// Types split out of this header. Each one is aliased back into
// VulkanRenderer::Impl at the spot its definition used to occupy, so every
// reference site — Impl's own methods and the VulkanCore*.cpp TUs — is
// unchanged.
#include "VulkanImplCommon.hpp"
#include "VulkanGpuLayouts.hpp"
#include "VulkanGeometryState.hpp"
#include "VulkanSceneTypes.hpp"
#include "EdgeRefine.hpp"
#include "VulkanViewContext.hpp"
#include "EnvPrefilter.hpp"
#include "SkyBake.hpp"
#include "SceneEnvKnobs.hpp"
#include "VulkanLog.hpp"
#include "EventCameraDetector.hpp"
#include "LidarScanner.hpp"
#include "SkinningPipeline.hpp"
#include "TetSkinningPipeline.hpp"
#include "GpuTimings.hpp"
#include "OverlayPass.hpp"
#include "VulkanFrameTypes.hpp"
#include "RenderGraph.hpp"
#include "TransientPool.hpp"
#include "TaaResolve.hpp"
#if defined(THREEPP_WITH_FSR)
#include "FsrUpscaler.hpp"
#endif
#if defined(THREEPP_WITH_DLSS)
#include "DlssUpscaler.hpp"
#endif
#include "AutoExposure.hpp"
#include "GbufResolve.hpp"
#include "BillboardGlowPass.hpp"
#include "BillboardHdrLayer.hpp"
#include "BillboardSoftDepth.hpp"
#include "BloomPass.hpp"
#include "PostComposite.hpp"
#include "DofPass.hpp"
#include "SensorPass.hpp"
#include "SplatPass.hpp"
#include "DeferredShade.hpp"
#include "HiZPyramid.hpp"
#include "InstanceExpand.hpp"
#include "DrawGen.hpp"
#include "TlasGen.hpp"
#include "OcclusionCull.hpp"
#include "ParticleFieldPass.hpp"
#include "PathTracer.hpp"
#include "ProbeGI.hpp"
#include "WaterDisplacePipeline.hpp"
#include "WakeFieldPipeline.hpp"
#include "GrassWindPipeline.hpp"
#include "VertexSanitizePipeline.hpp"
#include "shaders/downwash_shared.h"// DownwashGpu (uploadDownwash)
#include "shaders/vulkan_shared.h"// MaterialDesc + kMaxMaterialTextures — same source the shaders read

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/AttributeView.hpp"
#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/scenes/Downwash.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

// (RT-pipeline / photon / denoise shader SPVs removed with the path tracer)
// (prefilter_env shader SPV moved into vulkan/EnvPrefilter.cpp)
// (skinning shader SPV moved into vulkan/SkinningPipeline.cpp)
// (water_displace shader SPV moved into vulkan/WaterDisplacePipeline.cpp)
// (gbuffer/gbuffer_indirect shader SPVs moved into vulkan/VulkanCorePipelines.cpp)
// (taa_resolve.comp.spv moved into vulkan/TaaResolve.cpp)
// (overlay/overlay_depth/overlay_color/overlay_point/overlay_sprite/particle/
//  sprite3d shader SPVs moved into vulkan/VulkanCorePipelines.cpp)
// (debug_resolve.comp.spv moved into vulkan/VulkanCoreIndirect.cpp)

#include "threepp/renderers/common/BC7Encode.hpp"
#include "threepp/renderers/common/BCnDecode.hpp"
#include "threepp/utils/GeometryLod.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <future>
#include <mutex>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The state groups Impl inherits (one header each, see the map at Impl).
#include "state/CoreState.hpp"
#include "state/GeometryState.hpp"
#include "state/TlasState.hpp"
#include "state/EnvironmentState.hpp"
#include "state/OceanState.hpp"
#include "state/LodState.hpp"
#include "state/SceneState.hpp"
#include "state/SensorState.hpp"
#include "state/RasterState.hpp"
#include "state/DeformState.hpp"
#include "state/ViewState.hpp"
#include "state/FieldBillboardState.hpp"
#include "state/PostChainState.hpp"
#include "state/SplatState.hpp"
#include "state/LensState.hpp"
#include "state/AtmosphereState.hpp"
#include "state/LightingState.hpp"
#include "state/GIState.hpp"
#include "state/OcclusionState.hpp"
#include "state/ParticleState.hpp"
#include "state/GpuDrivenState.hpp"
#include "state/DebugState.hpp"
#include "state/OverlayState.hpp"
#include "state/FrameLoopState.hpp"
#include "state/UploadState.hpp"
#include "state/IndirectDrawState.hpp"
#include "state/MaterialState.hpp"
#include "state/ExposureState.hpp"
#include "state/ShadowState.hpp"
#include "state/RenderGraphState.hpp"

namespace threepp {

    using vulkan::VulkanContext;
    // Resource helpers moved to vulkan/VulkanResources.{hpp,cpp} to start
    // peeling apart the 11k-line VulkanRenderer.cpp monolith. Type names +
    // free functions imported into the local scope so the rest of this file
    // (which references them as unqualified Buffer / Image2D / createBuffer
    // / destroyBuffer / check / alignUp / etc.) keeps compiling unchanged.
    using vulkan::Buffer;
    using vulkan::Image2D;
    using vulkan::check;
    using vulkan::alignUp;
    using vulkan::createBuffer;
    using vulkan::destroyBuffer;
    using vulkan::createAsScratchBuffer;
    using vulkan::destroyImage2D;
    using vulkan::uploadHostVisible;
    using vulkan::flushHostWrites;
    using vulkan::invalidateHostReads;
    using vulkan::TimingPass;
    using vulkan::TP_RasterGbuf;
    using vulkan::TP_OverlayDepth;
    using vulkan::TP_DeferredShade;
    using vulkan::TP_Denoise;
    using vulkan::TP_TAA;
    using vulkan::TP_OverlayDraw;
    using vulkan::TP_GbufResolve;
    using vulkan::TP_ShadeB;
    using vulkan::TP_Dof;
    using vulkan::TP_Froxel;
    using vulkan::TP_SensorImage;
    using vulkan::TP_Splat;
    using vulkan::TP_ProbeGI;
    using vulkan::TP_ShadowMaps;
    using vulkan::TP_Rtao;
    using vulkan::TP_FogMarch;
    using vulkan::TP_BillboardLayer;

    namespace {
        // kFramesInFlight — and the long note on why it MUST stay EVEN —
        // moved to VulkanImplCommon.hpp (the extracted types size arrays with
        // it). Forwarded here so this header and the VulkanCore*.cpp TUs keep
        // referring to it unqualified.
        constexpr uint32_t kFramesInFlight = vulkan::impl::kFramesInFlight;
    }// namespace

    // ── State groups ─────────────────────────────────────────────────────
    // Impl's data members live in the state structs listed here, one header each
    // under vulkan/state/ (namespace vulkan::impl); Impl inherits them, so
    // its methods and the VulkanCore*.cpp TUs name every member unqualified.
    // The base order is the members' original declaration order where it
    // matters: bases are constructed first to last, before Impl's own
    // members, and destroyed last to first, after them, so a group that owns a pass or a
    // thread sits where that owner used to be declared (ctx in CoreState
    // outlives everything). Keep it that way when adding a member: an owner
    // of device objects goes in the group, and at the place, its lifetime
    // needs. Impl itself keeps the methods and the few members whose
    // initializers call its static methods (the shadow settings).
    //
    //   CoreState.hpp             the canvas and its size, the clear colour, viewport, scissor and
    //                             split-screen pane, and the Vulkan context with the frame-local
    //                             transient memory pools everything else allocates from
    //   GeometryState.hpp         per-geometry GPU records (the BLAS cache) and the deformable meshes'
    //                             states (skinned, tet, morphed, displaced, grass) with the shared
    //                             compute pipelines that deform them, plus the displacement-map specs
    //   TlasState.hpp             the top-level acceleration structure: its buffers, refit scratch,
    //                             instance staging and dirty rows, and the per-entry ray-flag
    //                             classification (cutout and ray-shadow distance)
    //   EnvironmentState.hpp      the environment map and what feeds it: background colour, the baked
    //                             Preetham sky, the prefiltered env and its extracted sun, the scene's
    //                             env knobs and the env-sun policy
    //   OceanState.hpp            the ocean surface bindings the shade reads: fine height and foam
    //                             fields, crest foam, climate, wake and ripple views, and the foam
    //                             detail and crest-pattern images
    //   LodState.hpp              auto-LOD: the background worker thread and its job and result queues,
    //                             the level budget, the landing batches, and the per-entry LOD fade and
    //                             level buffers
    //   SceneState.hpp            the scene walk's snapshot and delta, and the entry list it produces:
    //                             spans, slots, admission, per-entry moved/placed bits, motion matrices
    //                             and stable ids
    //   SensorState.hpp           the lidar scanner and the event camera (its detector, luma buffer and
    //                             shade pipeline), and the per-scene opt-in for sensor-only surfaces
    //   RasterState.hpp           the G-buffer raster: its render passes, pipelines and layouts (single-
    //                             and multisampled), the MSAA resolve and dummies, the G-buffer sampler
    //                             and the material-raster toggles
    //   DeformState.hpp           GPU deformation recorded into the frame: the skinning and tet-skinning
    //                             pipelines and the per-frame deform, refit and resync queues
    //   ViewState.hpp             the views (the primary and the secondaries), the current view, the
    //                             primary's frame-prep output, pending view changes and the view
    //                             composites
    //   FieldBillboardState.hpp   ParticleField billboards: the billboard pipeline layout and the sun
    //                             snapshot, the glow pass, the HDR layer and its pipelines, and the
    //                             soft-depth pass
    //   PostChainState.hpp        the upscalers (FSR, DLSS with Ray Reconstruction), TAA, the render
    //                             scale, and the post knobs: depth of field, bloom, sharpen, motion blur
    //   SplatState.hpp            Gaussian splats: the splat pass, this frame's visible and parked
    //                             clouds, the depth stamp, the splat volume (reflection) table and the
    //                             depth AOV mode
    //   LensState.hpp             the camera sensor image: the sensor pass, lens distortion and sensor
    //                             noise, and the lens rig (its face cameras and views)
    //   AtmosphereState.hpp       the participating media: air fog and its march, underwater murk, the
    //                             medium, clouds, height fog, the deferred volumetric fog knobs and the
    //                             rotor downwash
    //   LightingState.hpp         what the deferred shade lights with: the lights UBO and clusters, the
    //                             emissive triangles and their cache, the sun's size and softness, and
    //                             the shade toggles (AO, ReSTIR DI, denoise, firefly clamp)
    //   GIState.hpp               global illumination: the probe GI and path tracer passes and the probe
    //                             grid's camera, light signature and fast-blend state
    //   OcclusionState.hpp        occlusion culling: the cull pass and its Hi-Z pyramid, the occlusion
    //                             render passes, and the per-mesh cull-bit ranges
    //   ParticleState.hpp         particles: ParticleSystem billboards (pipelines, textures, lit-
    //                             particle buffers), world-space sprites, and the ParticleField pass
    //                             with its density volume
    //   GpuDrivenState.hpp        GPU-driven work generation: instance expansion, the draw generator and
    //                             the TLAS generator, with their spans, work counts and self-checks
    //   DebugState.hpp            diagnostics and stats: GPU timings, draw/dynamic-geometry/TLAS stats,
    //                             ray counts, the hybrid debug views and their resolve pipeline
    //   OverlayState.hpp          the post-TAA overlay: the overlay pass, its pipelines (lines, points,
    //                             meshes, depth prepass and import, MSAA inject), overlay fog, colour
    //                             grade, and the line and morphed-point geometry caches
    //   FrameLoopState.hpp        the frame loop: swapchain images, command pool and buffers, semaphores
    //                             and fences, the frame state machine and clocks, the retire queue, the
    //                             overlay callback, the deferred-apply requests, and the frame outputs
    //                             (scene capture, frame interop)
    //   UploadState.hpp           one-shot submits and async work: the one-shot batch, async one-shots,
    //                             BLAS compaction and its retired storage, and the async grass prep
    //   IndirectDrawState.hpp     the indirect draw builder's state: the per-span draw cache and its
    //                             mutex, the scene build serial, and the stable, instance and class ids
    //                             it assigns
    //   MaterialState.hpp         materials and textures: the material and geometry descriptor tables
    //                             with their dirty ranges, the deduplicated material table, the bindless
    //                             texture array, its samplers and the async texture prep
    //   ExposureState.hpp         exposure and tone mapping: the tone-mapping operator, the physical
    //                             camera and photometric light units, the pre-exposure history, white
    //                             balance, and auto exposure
    //   ShadowState.hpp           shadow maps: the distance, the dummy bindings, the static-caster
    //                             cache, the shadow pipelines and the sun and lamp slot assignments
    //   RenderGraphState.hpp      recording: the frame and compute render graphs, the async-compute
    //                             pilot, the per-view record states, CPU-parallel recording (record-
    //                             apart buffers, the secondary views' prep) and the graph diagnostics
    struct VulkanRenderer::Impl : vulkan::impl::CoreState,
                                  vulkan::impl::GeometryState,
                                  vulkan::impl::TlasState,
                                  vulkan::impl::EnvironmentState,
                                  vulkan::impl::OceanState,
                                  vulkan::impl::LodState,
                                  vulkan::impl::SceneState,
                                  vulkan::impl::SensorState,
                                  vulkan::impl::RasterState,
                                  vulkan::impl::DeformState,
                                  vulkan::impl::ViewState,
                                  vulkan::impl::FieldBillboardState,
                                  vulkan::impl::PostChainState,
                                  vulkan::impl::SplatState,
                                  vulkan::impl::LensState,
                                  vulkan::impl::AtmosphereState,
                                  vulkan::impl::LightingState,
                                  vulkan::impl::GIState,
                                  vulkan::impl::OcclusionState,
                                  vulkan::impl::ParticleState,
                                  vulkan::impl::GpuDrivenState,
                                  vulkan::impl::DebugState,
                                  vulkan::impl::OverlayState,
                                  vulkan::impl::FrameLoopState,
                                  vulkan::impl::UploadState,
                                  vulkan::impl::IndirectDrawState,
                                  vulkan::impl::MaterialState,
                                  vulkan::impl::ExposureState,
                                  vulkan::impl::ShadowState,
                                  vulkan::impl::RenderGraphState {

        [[nodiscard]] float physicalExposure() const {
            const float ev100 = std::log2(camAperture_ * camAperture_ / camShutter_ *
                                          100.f / camIso_) -
                                camEvComp_;
            return 1.f / (1.2f * std::exp2(ev100));
        }
        // The factor already baked into this frame's sceneHdr stores (1.0 in
        // legacy mode). Hoisted into preExpBits_ once per frame in
        // recordCommandBuffer; preExpHist_ remembers each frame-in-flight's
        // value so cross-frame sceneHdr consumers (the exposure meter) can
        // un-bake the right slot's exposure.
        [[nodiscard]] float preExposure() const {
            return physicalCamera_ ? currentExposure() : 1.f;
        }

        // Definition moved to VulkanGeometryState.hpp.
        using LodGeomSel = vulkan::impl::LodGeomSel;
        static LodGeomSel selectLodGeom(const BlasRecord& rec, uint8_t lodLevel) {
            if (lodLevel > 0 && static_cast<size_t>(lodLevel - 1) < rec.lodLevels.size()) {
                const auto& lvl = rec.lodLevels[lodLevel - 1];
                // asAddress is 0 for a level without a structure (its record
                // has none either, so no TLAS instance reads it).
                if (lvl.index.handle != VK_NULL_HANDLE) {
                    return {lvl.address, lvl.index.address, lvl.indexCount, true};
                }
            }
            return {rec.address, rec.index.address, rec.indexCount,
                    rec.index.handle != VK_NULL_HANDLE};
        }

        // The displacement m's material asks for (false: none, or not for a
        // mesh like this one; see VulkanCoreScene.cpp).
        bool displacementSpecOf(Mesh& m, bool morphed, DisplacementSpec& out) const;
        void captureDisplacementFocus(Camera& camera);
        // geom refined around d's focus (a new geometry), or nullptr when d
        // does not refine, the geometry can't be, or no edge splits.
        static std::shared_ptr<BufferGeometry> refineForDisplacement(const BufferGeometry& geom,
                                                                     const DisplacementSpec& d);
        // Registers m's displacement under its geometry; the spec, or nullptr
        // when its material displaces nothing. Render thread (the scene walks).
        const DisplacementSpec* noteDisplacement(Mesh& m);
        // After m's material changed: re-reads its displacement and returns
        // whether the geometry's spec moved (the record then needs a rebuild).
        bool displacementChanged(Mesh& m);
        [[nodiscard]] uint64_t displacementSignatureOf(const BufferGeometry* g) const {
            const auto it = geomDisplacement_.find(g);
            return (it != geomDisplacement_.end() && it->second.geom.lock().get() == g) ? it->second.signature : 0u;
        }
        // The object-space box geom's record draws: its own bounding box, or
        // the displaced positions' (BlasRecord::dispMin/Max), or, for a
        // displacementMap whose record is not built yet, its own grown by the
        // spec's reach. False when geom has no box at all.
        bool boundsOf(BufferGeometry& geom, Box3& out) const {
            // No displacementMap anywhere (the usual scene): no lookups.
            const uint64_t sig = geomDisplacement_.empty() ? 0u : displacementSignatureOf(&geom);
            if (const auto it = blasCache.find(&geom);
                sig != 0u && it != blasCache.end() && it->second->dispSig == sig) {
                const auto& r = *it->second;
                out.set(Vector3(r.dispMin[0], r.dispMin[1], r.dispMin[2]), Vector3(r.dispMax[0], r.dispMax[1], r.dispMax[2]));
                return true;
            }
            if (!geom.boundingBox) geom.computeBoundingBox();
            if (!geom.boundingBox) return false;
            out = *geom.boundingBox;
            if (sig != 0u) out.expandByScalar(geomDisplacement_.at(&geom).reach());
            return true;
        }

        // Definition moved to VulkanGeometryState.hpp.
        using GeometryDesc = vulkan::impl::GeometryDesc;
        // The word view of it that the shaders reading only the material index
        // and the instance colour declare.
        static_assert(sizeof(GeometryDesc) == kGeomDescWords * 4u &&
                              offsetof(GeometryDesc, flags) == kGeomDescFlagsWord * 4u &&
                              offsetof(GeometryDesc, instanceColor) == kGeomDescColorWord * 4u,
                      "GeometryDesc changed shape: update kGeomDescWords / kGeomDescColorWord / kGeomDescFlagsWord "
                      "in vulkan_shared.h and geometry_desc.glsl.");
        // The material record an entry's GeometryDesc names (its flags' top bits).
        [[nodiscard]] static uint32_t geomMatIdx(const GeometryDesc& gd) {
            return gd.flags >> kGeomFlagMaterialShift;
        }
        static void setGeomMatIdx(GeometryDesc& gd, uint32_t idx) {
            gd.flags = (gd.flags & ~kGeomFlagMaterialMask) | (idx << kGeomFlagMaterialShift);
        }
        [[nodiscard]] static bool geomFlag(const GeometryDesc& gd, uint32_t bit) {
            return (gd.flags & bit) != 0u;
        }
        static void setGeomFlag(GeometryDesc& gd, uint32_t bit, bool on) {
            gd.flags = on ? (gd.flags | bit) : (gd.flags & ~bit);
        }

        // TLAS instance flags for an entry, from the MaterialDesc its shaders
        // read (mats[instanceCustomIndex]). Every BLAS is built non-opaque so
        // alpha-cutout casters yield candidates, which made EVERY solid
        // triangle a shadow, emitter-NEE, ReSTIR or probe-shadow ray met a
        // round trip to the shader to read its material and confirm it. Those
        // five queries are all terminate-on-first-hit visibility tests, and for
        // a material that none of them can reject each one confirms
        // unconditionally — so FORCE_OPAQUE hands the RT core the same answer
        // without the round trip. "None can reject" is exactly: not an
        // emitter (the emitter loops skip emissive >= 0.05; shadowOccludes
        // skips >= 1), not transmissive, not a blend (alphaCutoff < 0), and no
        // live cutout (alphaCutoff > 0 with an albedo texture). KEEP IN SYNC
        // with shadowOccludes (deferred_shade_10_lighting_utils.glsl,
        // probe_update.comp) and the emitter-skip tests. Rays that already
        // pass gl_RayFlagsOpaqueEXT are unaffected. THREEPP_NO_FORCE_OPAQUE=1
        // turns it off (A/B and bisecting).
        [[nodiscard]] static VkGeometryInstanceFlagsKHR tlasInstanceFlags(const MaterialDesc& m) {
            static const bool disabled = vulkan::env::flag("THREEPP_NO_FORCE_OPAQUE");
            VkGeometryInstanceFlagsKHR f = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            if (disabled) return f;
            const float emMax = std::max({m.emissive[0] * m.emissiveIntensity,
                                          m.emissive[1] * m.emissiveIntensity,
                                          m.emissive[2] * m.emissiveIntensity});
            const bool alwaysOccludes = emMax < 0.05f && !(m.transmission > 0.0f) &&
                                        !(m.alphaCutoff < 0.0f) &&
                                        (m.albedoTexIndex < 0 || m.alphaCutoff == 0.0f);
            if (alwaysOccludes) f |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
            return f;
        }
        // A material whose ONLY reason not to be force-opaque is a live cutout
        // (alphaCutoff > 0 with an albedo texture): foliage cards, grids. The
        // class setCutoutRayDistance may force opaque beyond its distance.
        [[nodiscard]] static bool cutoutCaster(const MaterialDesc& m) {
            const float emMax = std::max({m.emissive[0] * m.emissiveIntensity,
                                          m.emissive[1] * m.emissiveIntensity,
                                          m.emissive[2] * m.emissiveIntensity});
            return emMax < 0.05f && !(m.transmission > 0.0f) &&
                   m.alphaCutoff > 0.0f && m.albedoTexIndex >= 0;
        }
        // Is the whole of an entry's bounding sphere farther from `cam` than
        // `dist`? Unknown bounds (lodRadius 0) are never far.
        [[nodiscard]] static bool entryBeyond(const vulkan::impl::MeshEntry& en, const float cam[3], float dist) {
            if (en.lodRadius <= 0.f) return false;
            const float* M = en.worldMatrix.data();
            const float cx = M[0] * en.lodCenter[0] + M[4] * en.lodCenter[1] + M[8] * en.lodCenter[2] + M[12];
            const float cy = M[1] * en.lodCenter[0] + M[5] * en.lodCenter[1] + M[9] * en.lodCenter[2] + M[13];
            const float cz = M[2] * en.lodCenter[0] + M[6] * en.lodCenter[1] + M[10] * en.lodCenter[2] + M[14];
            const float s0 = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
            const float s1 = M[4] * M[4] + M[5] * M[5] + M[6] * M[6];
            const float s2 = M[8] * M[8] + M[9] * M[9] + M[10] * M[10];
            const float reach = dist + en.lodRadius * std::sqrt(std::max(s0, std::max(s1, s2)));
            const float dx = cam[0] - cx, dy = cam[1] - cy, dz = cam[2] - cz;
            return dx * dx + dy * dy + dz * dz > reach * reach;
        }
        // The ray-mask half of the same per-material classification (see the
        // visibility groups in vulkan_shared.h). Water (displaced) stays in the
        // opaque group whatever its material says.
        [[nodiscard]] static bool alphaMaskGroup(const MaterialDesc& m) {
            return m.transmission > 0.0f || m.alphaCutoff < 0.0f;
        }
        // An entry's TLAS ray mask before the LOD crossfade has its say (the
        // side lodFadeCasts turns down is hit by no ray; the callers apply
        // that per entry). Visibility group (see vulkan_shared.h): blend/
        // transmissive surfaces move to the alpha mask so pure-visibility
        // occlusion queries (env gather, GI, emissive-NEE) cull them out — a
        // text decal's transparent quad must not block IBL light. Water
        // (DisplacedMesh) stays opaque-mask: underwater light transport is
        // handled by its volumetrics, not pass-through. `hidden` is the
        // caller's (the lean refit reads its span's); no `md` (the lean
        // refit's entry past the desc cache) means no alpha group.
        [[nodiscard]] uint8_t tlasRayMask(const vulkan::impl::MeshEntry& en, const MaterialDesc* md, bool hidden) const {
            if (hidden) return 0u;// MeshEntry::hidden: no ray hits it
            // Sensors only, and only once opted in: mask 0 is hit by no ray at all.
            if (en.sensorOnly) return sensorOnlySurfaces_ ? kRayMaskSensorOnly : 0u;
            // Object3D::rayCastShadow overrides the viewmodel rule either way.
            const uint8_t rs = en.rayShadowState;
            if ((rs & vulkan::impl::kRayCastOff) || (en.camAttached && !(rs & vulkan::impl::kRayCastOn)))
                return kRayMaskNoShadow;// FP viewmodel: visible, never occludes
            if (!en.isDisplaced && md && alphaMaskGroup(*md)) return kRayMaskAlpha;
            return kRayMaskOpaque;
        }
        // The per-instance half: `group` is tlasRayMask's answer, fadeCasts
        // lodFadeCasts' for the entry. The ray shadow LOD (rayShadowState)
        // turns an opaque caster into a no-shadow one; a proxy past its
        // rayShadowFrom keeps its mask on the side of a crossfade that would
        // otherwise drop out. tlas_gen.comp's instanceMask is the GPU copy.
        [[nodiscard]] static uint8_t entryRayMask(uint8_t group, const vulkan::impl::MeshEntry& en, bool fadeCasts) {
            if (group == 0u) return 0u;
            const uint8_t rs = en.rayShadowState;
            if ((rs & vulkan::impl::kRayShadowProxy) && group != kRayMaskNoShadow) return group;
            if (!fadeCasts) return 0u;
            if (group == kRayMaskOpaque && (rs & (vulkan::impl::kRayShadowFar | vulkan::impl::kRayShadowNear)))
                return kRayMaskNoShadow;
            return group;
        }
        // A TLAS instance's transform: VkTransformMatrixKHR is row-major 3x4;
        // threepp Matrix4 is column-major 4x4 (elements[c*4 + r]). For
        // InstancedMesh the entry's worldMatrix already incorporates the
        // per-instance transform.
        static void copyTlasTransform(VkTransformMatrixKHR& dst, const std::array<float, 16>& e) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    dst.matrix[r][c] = e[c * 4 + r];
                }
            }
        }

        // Definition moved to state/StateCommon.hpp.
        using DescDirtyRanges = vulkan::impl::DescDirtyRanges;

        // Start a pass over the table: records are re-derived on their first
        // use in it.
        void matTableBeginPass() {
            ++matTableEpoch_;
            matTableMemoSlot_ = 0u;
        }
        // Empty but for record 0, for a structural rebuild to fill.
        void matTableReset();
        // The record for `mat`, created when new; `md` is its desc. Written and
        // dirtied only when its content changes. Counts no reference: the
        // caller hands the index to setEntryMat / assignEntryGeom. `owner` is
        // a mesh drawing `mat`, for the record's weak reference to it.
        uint32_t matTableSlot(const Material* mat, const MaterialDesc& md, const Mesh& owner);
        // A record for `key`: a new one while the GPU buffers have room, else
        // the longest-dormant one, else a new one after a growth.
        uint32_t matTableAcquire(const MatTableKey& key);
        void matTableGrow(uint32_t capacity);
        void matTableRef(uint32_t idx) {
            // A record's first entry: its material joins the live-edit watch.
            if (idx != 0u && matTableRefs_[idx]++ == 0u) matTableTouched_.push_back(matTableKeys_[idx]);
        }
        void matTableUnref(uint32_t idx);
        // The record entry `i` draws with.
        [[nodiscard]] const MaterialDesc& entryMat(size_t i) const {
            return matTable_[geomMatIdx(geomDescsCached_[i])];
        }
        // Point entry `i` at record `idx`, dirtying its GeometryDesc on a change.
        void setEntryMat(size_t i, uint32_t idx) {
            const uint32_t was = geomMatIdx(geomDescsCached_[i]);
            if (was == idx) return;
            matTableRef(idx);
            setGeomMatIdx(geomDescsCached_[i], idx);
            matTableUnref(was);
            markGeomDescsDirty(static_cast<uint32_t>(i));
        }
        // Replace entry `i`'s whole GeometryDesc with `gd` naming record
        // `idx` (the caller marks the range dirty).
        void assignEntryGeom(size_t i, const GeometryDesc& gd, uint32_t idx) {
            const uint32_t was = geomMatIdx(geomDescsCached_[i]);
            matTableRef(idx);
            geomDescsCached_[i] = gd;
            setGeomMatIdx(geomDescsCached_[i], idx);
            matTableUnref(was);
        }

        // Stage a desc change for every slot at once (a host-side patch applies
        // to the scene, not to one frame). Named per array so the call sites
        // read as what they patched.
        void markMatDescsDirty(uint32_t first, uint32_t count = 1) {
            for (auto& d : matDescsDirty_) d.mark(first, count);
        }
        void markMatDescsWhole() {
            for (auto& d : matDescsDirty_) d.markWhole();
        }
        void markGeomDescsDirty(uint32_t first, uint32_t count = 1) {
            for (auto& d : geomDescsDirty_) d.mark(first, count);
        }
        void markGeomDescsWhole() {
            for (auto& d : geomDescsDirty_) d.markWhole();
        }

        // Let go of envImage: retired when this Impl owns it, forgotten when it
        // is the sky bake's, which keeps its image for the renderer's life.
        void dropEnvImage() {
            if (envIsSky) {
                envImage = {};
                envIsSky = false;
            } else {
                retire(std::move(envImage));
            }
        }

        // (prevCamBufData_ / prevCameraValid moved to ViewContext.)

        [[nodiscard]] bool entryPlaced(size_t i) const {
            const size_t w = i >> 5;
            return w < entryPlacedBits_.size() && (entryPlacedBits_[w] & (1u << (i & 31u))) != 0u;
        }
        void markDrawRowFlags(size_t i) {
            if ((i >> 5) >= drawRowFlagBits_.size()) drawRowFlagBits_.resize((i >> 5) + 1u, 0u);
            drawRowFlagBits_[i >> 5] |= 1u << (i & 31u);
        }

        // Definition moved to VulkanSceneTypes.hpp.
        using MeshEntry = vulkan::impl::MeshEntry;

        // Classification-routing flags for a mesh — shared by the snapshot
        // build and the replay walk so the two can't drift.
        uint32_t snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const;

        // Replay the last expansion's traversal against the snapshot. true ⇒
        // tree shape, visibility, classification routing and all mesh/geom/mat
        // pointers are unchanged since the last full expansion. Also re-runs
        // LOD level selection (camera-driven visibility mutation) on kSnapLod
        // nodes — a level switch makes the walk diverge from the snapshot ⇒
        // false ⇒ the full pass re-expands with the new level. A mesh with
        // entries that was hidden or shown (Object3D::visible, material
        // visible) is no divergence: its span flips MeshEntry::hidden in place.
        bool sceneSnapshotMatches(Object3D& scene, Camera& camera);
        // The replay's test of one node against its record: false = the node
        // changed. A mesh with entries hidden or shown is no change: `flip`
        // says so, applied once the walk is accepted.
        struct SnapFlip {
            size_t snIdx = 0;
            bool hidden = false;   // Object3D-hidden (own flag or an ancestor's)
            bool matHidden = false;// material()->visible == false
        };
        bool snapNodeUnchanged(const SnapNode& sn, Object3D& o, bool vis, bool underLod, Camera& camera,
                               bool& flip, SnapFlip& f);

        // The expansion walk's record of one node (VulkanCoreScene.cpp), shared
        // by the full expansion and the edit-proportional diff walk, so both
        // derive a node, its span and its entries the same way.
        struct SceneWalkOut {
            std::vector<MeshEntry>& built;
            std::vector<vulkan::impl::LineEntry>& lines;
            std::vector<vulkan::impl::EntrySpan>& spans;
            std::vector<std::pair<ParticleField*, uint32_t>>& fields;
            std::vector<SnapNode>& snap;
            // Lazy rows: an InstancedMesh that had entries gets its first only
            // (the slot path derives the rest).
            bool lazyRows = false;
            bool anyRowsDeferred = false;
            // The meshes with spans before this walk (by address and id),
            // built at the first question.
            std::unordered_map<const Mesh*, unsigned int> shownBefore;
            bool shownBeforeBuilt = false;
        };
        bool hadEntriesBefore(SceneWalkOut& w, const Mesh& m);
        void recordSceneNode(SceneWalkOut& w, Object3D& o, bool vis, bool underLod, Camera& camera);

        // ── The edit-proportional path ──────────────────────────────────
        // A frame whose replay walk fails used to walk the whole scene again
        // (the full expansion) and match every span of it to the live ones
        // (trySlotEdit): ~6 ms an edit frame on the pine forest's ~13k nodes,
        // whether the edit added one cell or several. diffSceneWalk walks once
        // more against the last snapshot, matching each node among its old
        // parent's old children by subtree range (SnapNode::subtreeEnd): kept
        // nodes are copied, new or changed ones recorded anew, removed
        // subtrees collected. trySlotDelta then frees the removed spans in
        // place and places the new ones, with no copy, match or sort of the
        // spans that stay. Anything else (a line, a LOD, a field, a Sky, an
        // overlay, an instanced count change) takes the full path as before.
        // THREEPP_VULKAN_NO_DELTA=1 turns it off (the A/B lane).
        struct SceneDelta {
            std::vector<SnapNode> snap;      // the new snapshot; kept nodes keep their live spanIndex
            std::vector<MeshEntry> built;    // the entries of the spans recorded anew
            std::vector<vulkan::impl::EntrySpan> spans;    // those spans (first: into built)
            std::vector<vulkan::impl::LineEntry> lines;    // stays empty (a line takes the full path)
            std::vector<std::pair<ParticleField*, uint32_t>> fields;// stays empty
            std::vector<uint32_t> spanSnap;  // per new span: its node in snap
            std::vector<uint32_t> removed;   // live spans whose nodes are gone or recorded anew
            std::vector<SnapFlip> flips;     // in-place visibility flips (snIdx into snap)
            uint32_t swaps = 0;              // meshes placed again with a geometry, material or map swapped
            bool changed = false;
            bool fallback = false;
        };
        static bool deltaDisabled();
        // useVersions: a child whose Object3D::structureVersion is the one its
        // node recorded is copied with its whole subtree, unvisited (its flags
        // are then settled by flatSnapshotCheck).
        bool diffSceneWalk(Object3D& scene, Camera& camera, SceneDelta& d, bool useVersions);
        void diffSceneNode(SceneDelta& d, SceneWalkOut& w, Object3D& o, bool parentVis, bool underLod,
                           int64_t oldIdx, int32_t parentNewIdx, bool useVersions, Camera& camera);
        // ── Walks proportional to the edit (Object3D::structureVersion) ─────
        // A frame whose scene's structure version is the snapshot's had nothing
        // added, removed or swapped anywhere in it, so every node the snapshot
        // names is alive and in place: flatSnapshotCheck tests only the nodes
        // whose state a plain field can change (meshes, lines, points and LOD
        // nodes; flatWatch_), in pre-order, each with its effective visibility
        // from its ancestors (memoised; flatVis_), by the replay's own per-node
        // test, and applies the visibility flips. Anything else returns false
        // and the replay walk decides. THREEPP_VULKAN_NO_VERSIONS=1: never.
        static bool versionsDisabled();
        bool flatSnapshotCheck(Object3D& scene, Camera& camera);
        void diffSceneRemoved(SceneDelta& d, size_t first, size_t end);
        bool trySlotDelta(SceneDelta& d);
        // THREEPP_VULKAN_WALK_CHECK=1: after a delta, a full recording compared
        // node for node (mismatches to walkCheckMismatches_ and stderr).
        void deltaWalkCheck(Object3D& scene, Camera& camera);
        // Hide or show span `spanIndex`'s entries in place (MeshEntry::hidden),
        // flag their pixels' history and set hiddenFlippedThisFrame_.
        void setSpanHidden(size_t spanIndex, bool hidden);

        // This view's white balance and colour grade for the overlay passes
        // (OverlayGradeGpu), from its PostComposite; zero flags when the
        // overlay grade is off or the view grades nothing.
        vulkan::OverlayGradeGpu overlayGrade() const;
        // What a PostComposite applies (its white balance and grade) as that
        // block, whatever the overlay switch; the path trace's resolve too.
        static vulkan::OverlayGradeGpu gradeOf(const vulkan::PostComposite& post);

        // Lean frames: re-derive MeshEntry::rayFarOpaque when due. Reads
        // the material table, so call after the material patch. True when any
        // entry's bit flipped (the TLAS must then be refilled and rebuilt).
        bool classifyCutoutFar(const float camPos[3]);
        // Ray shadow LOD (Object3D::setRayShadowDistance / setRayShadowFrom /
        // setRayCastShadow): resolves each span's knobs (only once any object
        // has set one: Object3D::rayShadowEpoch) and re-derives the entries'
        // MeshEntry::rayShadowState when the knobs or spans changed or the
        // camera moved 5 % of the smallest distance from where they were last
        // derived. True when any entry's state changed (a TLAS refill and
        // full build). Needs no materials, so it runs before either fill.
        bool classifyRayShadow(const float camPos[3]);
        // An in-place edit of a record whose auto-LOD chain is still landing:
        // the chain goes to its batch's garbage (freed after that batch's
        // fence) instead of costing a device-wide wait. False: no such batch.
        bool detachLandingLod(BlasRecord& rec);
        // The lean TLAS fill's per-span signature (EntrySpan::tlasSig).
        [[nodiscard]] uint64_t spanTlasSig(const vulkan::impl::EntrySpan& sp, uint8_t mask,
                                           VkGeometryInstanceFlagsKHR flags, bool cutout, const BlasRecord* rec,
                                           VkDeviceAddress blas0, bool perEntryLod) const;
        // Where the build running on this thread counts.
        DrawStats& drawStatsOut() {
            ViewContext& v = view();
            return v.prepIdsReadOnly ? v.prepDrawStats : drawStats_;
        }
        void drawStatsTick();
        void markTlasRowDirty(size_t i) {
            if ((i >> 5) < tlasRowDirty_.size()) tlasRowDirty_[i >> 5] |= 1u << (i & 31u);
        }
        [[nodiscard]] bool tlasRowsChangedIn(uint32_t first, uint32_t count) const {
            if (count == 0u) return false;
            const size_t w0 = first >> 5, w1 = (size_t(first) + count - 1u) >> 5;
            for (size_t w = w0; w <= w1; ++w) {
                const uint32_t bits = (w < meshMovedBits_.size() ? meshMovedBits_[w] : 0u) |
                                      (w < tlasRowDirty_.size() ? tlasRowDirty_[w] : 0u);
                if (bits != 0u) return true;// a neighbour's bit in an edge word: a false positive, harmless
            }
            return false;
        }
        [[nodiscard]] bool tlasRowChanged(size_t i) const {
            const size_t w = i >> 5;
            const uint32_t bit = 1u << (i & 31u);
            return (w < meshMovedBits_.size() && (meshMovedBits_[w] & bit)) ||
                   (w < tlasRowDirty_.size() && (tlasRowDirty_[w] & bit));
        }
        void readRayCounts(uint32_t frame);
        void recordRayCountClear(VkCommandBuffer cb);
        void recordRayCountCopy(VkCommandBuffer cb);

        void ensureLodWorkerStarted() {
            if (lodWorker_.joinable()) return;
            lodWorker_ = std::thread([this] { lodWorkerMain(); });
        }
        void lodWorkerMain();
        // Snapshots geom's position/index (or, for non-indexed soup, its
        // position/normal/uv) arrays and hands them to the worker. Called
        // from the selection pass the first time an eligible entry
        // references a record whose lodState is None (never attempted, or
        // reset after a stale-result discard in drainLodResults). Returns
        // false when nothing was enqueued (no position attribute) so the
        // caller never strands the record in Queued with no result coming.
        // Simplifier normal-weight from the enqueueing mesh's material:
        // w = 0.5·glossiness² (glossiness = 1−roughness). Matte foliage
        // (r≈0.85 → w≈0.01) simplifies essentially position-only — the far-
        // vegetation LOD win depends on it; glossy paint (r≈0.25 → w≈0.28)
        // weights normals enough that panels never flatten visibly.
        // Unlit renders no shading at all → 0.
        // Unknown/absent material stays conservative at 0.5. dynamic_cast is
        // fine here: once per geometry lifetime, not per frame.
        // A clearcoat is the glossier of the two lobes when it is on: bathroom2's
        // tub is matte ceramic (roughness 1, w = 0) under a 0.03 mirror coat, so it
        // simplified position-only, its rolled rim became one coarse bevel, and the
        // bevel's interpolated normals mirrored the wall above in a cream band
        // along the tub's back wall.
        static float lodNormalWeightFor(const Mesh& mesh) {
            auto mat = mesh.material();
            if (!mat) return 0.5f;
            if (dynamic_cast<MeshBasicMaterial*>(mat.get())) return 0.f;
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat.get())) {
                float rough = r->roughness;
                if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get()); cc && cc->clearcoat > 0.f)
                    rough = std::min(rough, cc->clearcoatRoughness);
                const float gloss = 1.f - std::clamp(rough, 0.f, 1.f);
                return 0.5f * gloss * gloss;
            }
            return 0.5f;
        }

        bool enqueueLodJob(const BufferGeometry* geomPtr, unsigned int geomVersion, BufferGeometry& geom,
                           float normalWeight);
        // Per-frame chain-finalization budget (drainLodResults, called once
        // per frame from ensureSceneBuilt): up to 2 geometries OR 1 MiB of
        // new level resources (index buffers + BLAS storage), whichever hits
        // first. On the wall clock it takes what the worker has finished; on
        // a pinned clock (setSimTime) it waits for the worker until the
        // budget is spent or no job is outstanding, so the frame each chain
        // lands on is the same in every run, unless setReplayableLodSchedule
        // turned that wait off.
        // All of a frame's level BLAS builds are recorded into ONE
        // one-shot submit (fenced and unwaited, see LodLandingBatch below) —
        // a submit per level (or even per geometry) at Bistro-scale entry
        // counts costs more in queue round-trips than the batching saves.
        // Stale results (record
        // evicted / geomVersion moved on) and failed chains are processed
        // outside the budget — they create no resources and cost only a
        // hash lookup + state write each.
        void drainLodResults();
        // Definition moved to VulkanGeometryState.hpp.
        using LodPendingBuild = vulkan::impl::LodPendingBuild;
        // Creates one level's index buffer + AS handle/storage/address
        // (PREFER_FAST_TRACE, no ALLOW_UPDATE — LOD levels are immutable
        // once built) against `rec`'s EXISTING vertex buffer, and appends
        // the deferred build to `pending`. Mirrors buildBlasFor's BLAS-build
        // shape for the subset that differs (index-only geometry input; the
        // AS device address is queried at creation — it's a property of the
        // storage binding, valid before the build executes). Returns false
        // (leaves `out` default, appends nothing) on a degenerate/zero-
        // primitive level.
        bool buildLodLevelFor(BlasRecord& rec, const geometrylod::Level& level,
                              BlasRecord::LodLevel& out, std::vector<LodPendingBuild>& pending);
        // The level builds of one drain as cmdBuildAccelerationStructures
        // into `cb`.
        void recordLodLevelBuilds(VkCommandBuffer cb, std::vector<LodPendingBuild>& pending);
        void submitLodLevelBuilds(std::vector<LodPendingBuild>& pending,
                                  const std::vector<const BufferGeometry*>& pendingGeoms,
                                  const std::vector<const BufferGeometry*>& chainGeoms);
        void landLodBatches();
        // Submits `cb` (ended here) with a fresh fence and files the batch.
        void submitLodBatch(LodLandingBatch&& b, VkCommandBuffer cb, std::vector<LodLandingBatch>& into);
        void destroyLodLandingBatches();// teardown, device idle
        // Destroys every level's AS/storage/index and decrements the running
        // byte totals — called from every blasCache erase site (destructor,
        // eviction prune, geomVersion-changed rebuild) so a record's LOD
        // resources never outlive the record itself.
        void destroyBlasLodLevels(BlasRecord& rec);
        // The same for a record with no chain still landing (lodLanding
        // empty), without a device wait: the level BLASes and index buffers
        // go to the retire queue, since only earlier frames still in flight
        // reference them. The in-place refresh's frame-cb route uses it.
        void retireBlasLodLevels(BlasRecord& rec);

        // Free every GPU resource a BlasRecord owns. One shared helper for all
        // six BlasRecord holders (blasCache, skinned / tet / displaced / grass
        // / morphed states), so adding a Buffer to BlasRecord can only ever be
        // missed in one place.
        // NB: LOD levels are NOT freed here — blasCache is the only owner of a
        // LOD chain and calls destroyBlasLodLevels() itself, which also keeps
        // the lodBlasBytes_ accounting straight.
        void destroyBlasRecord(BlasRecord& rec) {
            if (!blasCompactPending_.empty()) {
                blasCompactPending_.erase(
                        std::remove(blasCompactPending_.begin(), blasCompactPending_.end(), &rec),
                        blasCompactPending_.end());
            }
            if (rec.as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec.as, nullptr);
            destroyBuffer(ctx->allocator(), rec.storage);
            destroyBuffer(ctx->allocator(), rec.vertex);
            destroyBuffer(ctx->allocator(), rec.index);
            destroyBuffer(ctx->allocator(), rec.rtIndex);
            destroyBuffer(ctx->allocator(), rec.normal);
            destroyBuffer(ctx->allocator(), rec.uv);
            destroyBuffer(ctx->allocator(), rec.uv2);
            destroyBuffer(ctx->allocator(), rec.color);
            destroyBuffer(ctx->allocator(), rec.prevVertex);
            destroyBuffer(ctx->allocator(), rec.blasScratch);
            destroyBuffer(ctx->allocator(), rec.dynStaging);
            // Zero-copy vertex interop exports, freed here and only here.
            // Note the asymmetry with the buffers above — a foreign API may
            // still hold an import of this memory. Nothing in Vulkan can wait
            // for that; the contract (documented on enableVertexInterop) is that
            // the application stops its producer before dropping the geometry,
            // as it must for the soft-body and ParticleField exports too.
            if (rec.sanitizeDS != VK_NULL_HANDLE && vertexSanitize_) {
                vertexSanitize_->freeRecordDescriptorSet(rec.sanitizeDS);
                rec.sanitizeDS = VK_NULL_HANDLE;
            }
            vulkan::destroyExternalBuffer(ctx->device(), rec.posExt);
            vulkan::destroyExternalBuffer(ctx->device(), rec.nrmExt);
            rec.externalCopy = nullptr;
            rec.interop = false;
        }

        [[nodiscard]] float emissiveGateK() const;
        [[nodiscard]] bool isGlowOnly(const Object3D& o) const {
            return !glowOnlyObjects_.empty() && glowOnlyObjects_.count(o.id) != 0;
        }
        // (prevWorldMats hash map replaced by the entry-aligned
        //  prevWorldByEntry_ / prevWorldValidByEntry_ vectors — identity-
        //  remapped across full rebuilds alongside meshMovedSticky_.)

        void markMotionDirty(uint32_t first, uint32_t count) {
            for (auto& d : motionDirty_) d.mark(first, count);
        }
        void markMotionWhole() {
            for (auto& d : motionDirty_) d.markWhole();
        }

        Mesh& slotTombstone();
        // A free span over [first, first + count), and the entry every slot
        // of one holds.
        EntrySpan slotFreeSpan(uint32_t first, uint32_t count, uint64_t parkedUntil);
        MeshEntry slotFreeEntry();
        // Slots an InstancedMesh span is laid out with past its count, so a
        // setCount grows in place: none for a plain mesh or a full instance
        // buffer, else min(capacity - count, max(count / 2, 64)). Both the
        // structural layout and the slot path leave it.
        static uint32_t slotSlackFor(const EntrySpan& sp, uint32_t count);
        // THREEPP_VULKAN_NO_SLOTS=1: every content change is a structural
        // rebuild in today's dense layout, the A/B lever against the slots.
        static bool slotsDisabled();
        uint32_t slotMaterialAssetIdx(const Material* mat);
        // Called at the end of the full expansion with what it built. Edits
        // lastVisibleEntries_, entrySpans_ and every entry-indexed row in
        // place and returns true, or touches nothing the full rebuild reads
        // and returns false (the change is structural: the caller rebuilds).
        bool trySlotEdit(std::vector<MeshEntry>& built, std::vector<EntrySpan>& builtSpans,
                         const std::vector<std::pair<ParticleField*, uint32_t>>& builtFields);
        // The rows both slot paths (trySlotEdit, trySlotDelta) write: a span's
        // record (its geometry's in blasCache, or a grass field's own), the
        // fingerprint the full path's slow branch derives for a mesh, the
        // tombstone's row, the row arrays grown to `newSize`, and one entry's
        // row written as new or as a tombstone.
        const BlasRecord* slotRecordOf(const MeshEntry& e);
        MeshFingerprint slotFingerprintOf(Mesh& m);
        void slotTombRows(MeshEntry& e, MeshFingerprint& fp);
        void slotGrowRows(size_t oldSize, size_t newSize, const MeshEntry& tombEntry, const MeshFingerprint& tombFp);
        void slotFillRow(size_t i, const MeshEntry& be, const MeshFingerprint& fp0, const BlasRecord* rec,
                         const GeometryDesc& gd, const MaterialDesc& md0, uint32_t stableId);
        void slotTombRow(size_t i, const MeshEntry& tombEntry, const MeshFingerprint& tombFp);
        // The lean path's per-span world-matrix refresh (scene.3), shared with
        // the frame trySlotEdit edited.
        void refreshSpanMatrices();
        // A mesh's MaterialDesc before its instance color, with every map
        // admitted to the bindless array; materialAssetIdx left for the
        // caller. One function for the structural rebuild, the material
        // patch and the slot path.
        MaterialDesc buildMaterialDesc(Mesh& m);
        // An entry's GeometryDesc for the selected BLAS level, as the
        // structural rebuild and the slot path both write it.
        GeometryDesc buildGeometryDesc(const MeshEntry& en, const BlasRecord& rec, const LodGeomSel& lodSel) const;

        // The source the frame actually runs. Final needs a final frame
        // (eventsOnlyMode skips producing one) and a device that can imageLoad
        // the BGRA8 swapchain without a format qualifier; otherwise Shaded.
        [[nodiscard]] EventCameraSource effectiveEventCamSource() const {
            if (eventCamSource_ == EventCameraSource::Final && !eventsOnlyMode_ &&
                ctx && ctx->storageImageReadWithoutFormat()) {
                return EventCameraSource::Final;
            }
            return EventCameraSource::Shaded;
        }
        // True when the event camera reads the raw raster G-buffer — the one
        // consumer for which per-frame raster jitter is not resolved later but
        // leaks straight into a sensor as false motion (jittered silhouette
        // coverage flips → a static scene fires ~6.7k events/frame).
        // The Final source reads the post-TAA frame, where jitter has already
        // been resolved, so it keeps jitter on. Every raster-jitter gate keys
        // off this, not off eventCamEnabled_.
        [[nodiscard]] bool eventCamReadsGbuf() const {
            return eventCamEnabled_ &&
                   effectiveEventCamSource() == EventCameraSource::Shaded;
        }

        // Per-view state moved to VulkanViewContext.hpp (the hybrid-prepass
        // note and the per-view ownership rationale travel with it).
        using RasterGbufImages = vulkan::impl::RasterGbufImages;

        // Total bytes VMA currently has allocated across every heap. Sampled
        // before/after a view's allocation to report what that view cost —
        // measured rather than computed from a table of formats that would rot
        // the moment someone adds an accumulator to the G-buffer.
        [[nodiscard]] VkDeviceSize vmaAllocatedBytes() const {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice(), &mp);
            std::vector<VmaBudget> budgets(mp.memoryHeapCount);
            vmaGetHeapBudgets(ctx->allocator(), budgets.data());
            VkDeviceSize total = 0;
            for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
                total += budgets[i].statistics.allocationBytes;
            return total;
        }
        [[nodiscard]] VkDeviceSize vmaReservedBytes() const {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice(), &mp);
            std::vector<VmaBudget> budgets(mp.memoryHeapCount);
            vmaGetHeapBudgets(ctx->allocator(), budgets.data());
            VkDeviceSize total = 0;
            for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
                total += budgets[i].statistics.blockBytes;
            return total;
        }
        // A view's frame prep running on a pool thread (THREEPP_VK_CPU_VIEWS)
        // names its view in the thread's context (CpuPar.hpp), and so do the
        // forks it makes; everywhere else the context is null and the view is
        // curView_.
        [[nodiscard]] ViewContext* viewPtr() const {
            auto* v = static_cast<ViewContext*>(vulkan::par::threadContext());
            return v ? v : curView_;
        }
        ViewContext&       view()       { return *viewPtr(); }
        const ViewContext& view() const { return *viewPtr(); }
        ViewPrepOut& prepOut() {
            auto* v = static_cast<ViewContext*>(vulkan::par::threadContext());
            return v ? v->prepStage : prepOut_;
        }
        // The primary view, by name, for the (many) places that mean "the
        // swapchain view" rather than "whichever view is being recorded".
        ViewContext&       primaryView()       { return *views_[0]; }
        const ViewContext& primaryView() const { return *views_[0]; }
        // Did entry `i` survive THIS view's frustum cull? Out-of-range reads
        // answer "yes", which is the conservative direction: a view that has
        // not culled yet (or an entry list that grew since) draws everything
        // rather than dropping geometry.
        [[nodiscard]] bool viewCulled(size_t i) const {
            const auto& c = curView_->inFrustum;
            return i >= c.size() || c[i] != 0;
        }
        // <<< VIEWCTX_END

        // Host mirror of particlefield_gbuf.vert's push_constant block
        // (scalar layout; 64-bit members first so nothing needs padding).
        struct ParticleFieldPC {
            uint64_t posAddr;
            uint64_t prevPosAddr;
            uint64_t oriAddr;
            uint32_t drawIdx;
            uint32_t wSemantic;
            float    invUniformRadius;
            // F4 distance LOD. camPos is the WORLD position of the view being
            // recorded, which is why it rides the push constant rather than the
            // FieldDesc: the same field is drawn once per view and the gate has
            // to move with the camera.
            float    lodFar;
            float    camPos[3];
            float    lodFade;
            float    nearCull;
            float    _pad;
        };
        static_assert(sizeof(ParticleFieldPC) == 64,
                      "ParticleFieldPC drifted from particlefield_gbuf.vert");
        // (rasterDescPool / rasterDescSets / drawInfoBuffers /
        //  indirectCmdBuffers and their capacities moved to ViewContext. Only
        //  the LAYOUT stays shared — every view's set has the same shape, but
        //  a pool sized for one view's frames-in-flight cannot serve two.)

        // ── Splat depth stamp ──────────────────────────────────────────────
        // Writes the Gaussian-splat depth AOV into the overlay's depth
        // attachment between the splat composite and the overlay draw, so a
        // wireframe / line / sprite BEHIND a cloud is occluded by it. See
        // shaders/splat_overlay_depth.frag and recordSplatOverlayDepthStamp.
        // One descriptor set per frame in flight (each names that frame's AOV
        // image and is rewritten only when the view handle changes, so the
        // update never lands on a set the GPU is still reading).
        // Mirrors the push_constant block of splat_overlay_depth.frag — the
        // two move together.
        struct SplatStampPC {
            float    aovScale[2];
            float    paneOrigin[2];
            float    aovLimit[2];
            float    projA;
            float    projB;
            uint32_t ortho;
        };

        // Canvas antialiasing rounded DOWN to a power of two the device can
        // actually use for BOTH a color and a depth framebuffer attachment,
        // capped at 8. Resolved once (the answer can't change: it depends only
        // on the Canvas parameter and the physical device) and cached, because
        // the sample count must be known before createOverlayPipeline builds
        // the pass's pipelines AND before createRasterGbufImages decides
        // whether to allocate the single-sample unjitDepth at all.
        uint32_t overlaySamples() {
            if (overlaySamples_ != 0) return overlaySamples_;
            uint32_t want = canvas.samples();
            if (want < 2) {
                overlaySamples_    = 1;
                overlaySampleBits_ = VK_SAMPLE_COUNT_1_BIT;
                return overlaySamples_;
            }
            if (want > 8) want = 8;
            uint32_t p = 8;
            while (p > want) p >>= 1;// round DOWN to a power of two
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            // VK_SAMPLE_COUNT_N_BIT == N numerically, so the count doubles as
            // its own flag bit.
            const VkSampleCountFlags supported = props.limits.framebufferColorSampleCounts &
                                                 props.limits.framebufferDepthSampleCounts;
            while (p > 1 && !(supported & static_cast<VkSampleCountFlags>(p))) p >>= 1;
            overlaySamples_    = p;
            overlaySampleBits_ = static_cast<VkSampleCountFlagBits>(p);
            return overlaySamples_;
        }

        // Same answer as overlaySamples(), as the VkSampleCountFlagBits the
        // pipeline / image create infos want.
        VkSampleCountFlagBits overlaySampleBits() {
            overlaySamples();
            return overlaySampleBits_;
        }

        // Lazily (re)size the multisampled overlay targets + the 1-sample
        // inject scratch to the swapchain extent and point overlayInjectSet_
        // at the scratch. Single images (NOT per-frame-in-flight): each is
        // written and read inside one command buffer, and cross-frame WAR is
        // handled by UNDEFINED-discard barriers at the point of use. Extent
        // only changes across a swapchain recreate (device drained) and the
        // first call precedes any submit that references the set, so the
        // descriptor update never races the GPU. Old images go through the
        // frame-serial retire queue. No-op when MSAA is off.
        void ensureOverlayMsaaImages(VkExtent2D ext);

        // Host mirror of particlefield_billboard.{vert,frag}'s push block
        // (scalar layout). Exactly 128 B — the range every Vulkan
        // implementation guarantees — so anything per-field lives in the
        // device-address record instead and only the per-view camera is
        // here. proj is unjittered: this pass runs after the temporal resolve.
        struct FieldBillboardPC {
            float    proj[16];  //   0
            float    mv[3][4];  //  64  rows of the affine view * model
            uint64_t paramsAddr;// 112
            // Per-view record (exposure, toneMapMode, fog terms) rides behind
            // this address — the push block has no room past 128 B.
            uint64_t viewAddr;  // 120
        };
        static_assert(sizeof(FieldBillboardPC) == 128,
                      "FieldBillboardPC drifted from particlefield_billboard.vert");
        struct FieldGlowPC {
            float    intensity;
            float    exposure;
            uint32_t toneMapMode;
            uint32_t _pad;
            float    invDisplay[2];
            vulkan::OverlayGradeGpu grade;// setOverlayColorGrade; zero flags when off
        };
        static_assert(sizeof(FieldGlowPC) == 120,
                      "FieldGlowPC drifted from particlefield_glow.frag");

        // Create particle_light.comp's IO sets (set 1: centers in, results out)
        // once both their layout owner (deferredShade_, lazily constructed on
        // the first scene build) and the particle buffers (createParticlePipeline)
        // exist. Idempotent; buffers are fixed at creation so the sets are
        // written exactly once.
        void ensureParticleIoSets();

        // Gather every visible ParticleSystem's particle centers (model → world,
        // every 4th vertex of the coincident-quad layout) into this frame's
        // centers buffer and assign each mesh its base index, in the SAME order
        // the overlay loop draws them. Runs in recordCommandBuffer BEFORE the
        // scene-dispatch hook; the hook dispatches particle_light.comp over the
        // result. No-op (count 0) when the scene has no particles, the billboard
        // pipeline isn't initialised yet, or the IO sets can't exist (no
        // deferred shade → nothing would ever light the result buffer).
        void prepareParticleLighting();

        // (currVPunjit_ / currViewUnjit_ / currProjUnjit_ moved to ViewContext.)

        // Definition moved to VulkanViewContext.hpp.
        using RasterCameraData = vulkan::impl::RasterCameraData;
        // (rasterCameraUbos, the prev-VP / prev-jitter / sky-reproject /
        //  depth-linearization TAA bookkeeping, and the prev camera pose all
        //  moved to ViewContext — see the RasterCameraData note there; the
        //  camera's rotation since last frame is prepOut_.deferredCamRotAngle.)

        void   resolveUpscalers();
        // DLSS can run: the feature exists, or it has not been asked for yet
        // and NGX reports it (the public getters' answer; waits for NGX's
        // init, creates nothing).
        [[nodiscard]] bool dlssAvailableOrPending();
        [[nodiscard]] bool dlssRrAvailableOrPending();// the same for Ray Reconstruction
        [[nodiscard]] bool dlssOnOrPending();         // and DLSS is switched on (the public dlss())
        void   revealWindow() {
            if (windowRevealed_) return;
            windowRevealed_ = true;
            canvas.showWindow();// a headless canvas stays hidden
        }
        // An external upscaler (FSR/DLSS) compiled in AND created (available).
        // Gates the HDR-mode plumbing both reuse — hdrOut_ stays allocated
        // regardless of the runtime toggles, so setFsr/setDlss never need a
        // descriptor rewrite. Always compiled (false without either option) to
        // keep #ifs out of the shared code.
        [[nodiscard]] bool fsrActiveForHdrPlumbing() const {
            bool active = false;
#if defined(THREEPP_WITH_FSR)
            active = active || fsrActive_;
#endif
#if defined(THREEPP_WITH_DLSS)
            active = active || dlssActive_;
#endif
            return active;
        }
        // Per-upscaler availability (compiled in + context/feature created),
        // independent of the runtime toggles — the public *Available() getters.
        [[nodiscard]] bool fsrAvailable() const {
#if defined(THREEPP_WITH_FSR)
            return fsrActive_;
#else
            return false;
#endif
        }
        [[nodiscard]] bool dlssAvailable() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssActive_;
#else
            return false;
#endif
        }
        // DLSS is the ACTIVE upscaler this frame: available AND runtime-enabled.
        // Outranks FSR (see useFsr) and forces the projection jitter on, like FSR.
        [[nodiscard]] bool useDlss() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssActive_ && dlssEnabled_;
#else
            return false;
#endif
        }
        // FSR is the ACTIVE upscaler this frame: available AND runtime-enabled,
        // and DLSS is not running (DLSS outranks FSR when both are on).
        // Drives the record branch and the jitter source (FSR needs jitter, so it
        // also forces the projection jitter on — see cameraJitterOn).
        [[nodiscard]] bool useFsr() const {
#if defined(THREEPP_WITH_FSR)
            return fsrActive_ && fsrEnabled_ && !useDlss();
#else
            return false;
#endif
        }
        // Runtime FSR on/off. No reallocation — the FSR and TAA paths both keep
        // their resources, so this is frame-to-frame switchable. Resets the
        // temporal history so the switched-to path doesn't inherit the other's
        // accumulation (FSR re-primes via fsrResetNext_).
        void setFsr(bool enabled) {
#if defined(THREEPP_WITH_FSR)
            if (fsrEnabled_ == enabled) return;
            fsrEnabled_ = enabled;
            if (view().taa_) view().taa_->invalidateHistory();
            markMaterialSamplerDirty();// jitter gate flips → sampler policy flips
            fsrResetNext_ = true;
#if defined(THREEPP_WITH_DLSS)
            dlssResetNext_ = true;// path hand-off re-primes whichever runs next
#endif
#else
            (void) enabled;
#endif
        }
        // Runtime DLSS on/off, same switching semantics as setFsr. Turning DLSS
        // off hands the frame to FSR (if available + enabled) or the TAA path.
        void setDlss(bool enabled) {
#if defined(THREEPP_WITH_DLSS)
            if (dlssEnabled_ == enabled) return;
            dlssEnabled_ = enabled;
            if (view().taa_) view().taa_->invalidateHistory();
            markMaterialSamplerDirty();// jitter gate flips → sampler policy flips
            dlssResetNext_ = true;
#if defined(THREEPP_WITH_FSR)
            fsrResetNext_ = true;// path hand-off re-primes whichever runs next
            // The FSR context this hands the frame to, if there is none yet,
            // is made at the start of the next frame (resolveUpscalers), with
            // the HDR output it writes through.
#endif
#else
            (void) enabled;
#endif
        }
        // Runtime DLSS Ray Reconstruction on/off. Honoured only while DLSS
        // runs (useDlss) and the denoiser is on; renderFrame creates or
        // releases the DLSSD feature on the next frame
        // (syncDlssRayReconstruction). Flipping resets the temporal history
        // like setDlss: RR and SR keep separate histories.
        void setDlssRayReconstruction(bool enabled) {
#if defined(THREEPP_WITH_DLSS)
            if (dlssRrEnabled_ == enabled) return;
            dlssRrEnabled_ = enabled;
            dlssResetNext_ = true;
            if (view().taa_) view().taa_->invalidateHistory();
            if (enabled && dlssActive_ && dlss_ && !dlss_->rayReconstructionAvailable()) {
                THREEPP_LOGF(vulkan::kLogUpscale, Warn, "DLSS Ray Reconstruction is requested, but NGX reports it unavailable "
                                                        "(nvngx_dlssd.dll missing, or the GPU or driver refused it); DLSS SR runs");
            }
#else
            (void) enabled;
#endif
        }
        // RR is the DLSS evaluate this frame: knob on, DLSS running, the
        // denoiser on, the DLSSD feature live and not in a failure streak.
        [[nodiscard]] bool useDlssRayReconstruction() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssRrEnabled_ && denoiseEnabled_ && useDlss() && dlss_ && dlss_->rayReconstructionValid() &&
                   !dlss_->rayReconstructionFailing();
#else
            return false;
#endif
        }
        // This view's frame goes through Ray Reconstruction: the primary, the
        // DLSSD feature live, the guide images at this frame's render extent.
        // The shade's permutation, the filters' mode and the record path's
        // evaluate all key on it, so they agree within a frame. Defined in
        // VulkanRenderer.cpp beside the shade dispatch.
        [[nodiscard]] bool rrShadeActive();
        // The knob is on and RR can run (DLSS runs, the denoiser is on, NGX
        // reports RR): what the DLSSD feature and the shade's guide images follow.
        [[nodiscard]] bool dlssRrWanted() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssRrEnabled_ && denoiseEnabled_ && useDlss() && dlss_ && dlss_->rayReconstructionAvailable();
#else
            return false;
#endif
        }
        [[nodiscard]] bool dlssRayReconstructionAvailable() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssActive_ && dlss_ && dlss_->rayReconstructionAvailable();
#else
            return false;
#endif
        }
        // Brings the DLSSD feature in line with the knob, the DLSS state and
        // the render extent; renderFrame calls it once a frame before the
        // frame records. Defined in VulkanCoreFrame.cpp.
        void syncDlssRayReconstruction();
        // tan(fovY/2), stashed in updateCameraUbo: prepOut_.tanHalfFovY.
        // Sensor height in METRES, stashed in updateCameraUbo from
        // PerspectiveCamera::getFilmHeight() (filmGauge is the film WIDTH in
        // mm; the height follows from the camera's aspect). This is the one
        // sensor the whole camera model shares: FOV comes from it via
        // setFocalLength, and the DoF CoC + cameraIntrinsics() read it back.
        // 0.0197 = the default 35 mm gauge at 16:9. Non-perspective cameras
        // keep whatever the last perspective camera left here:
        // prepOut_.filmHeightM.
        // Unjittered projection terms stashed in updateCameraUbo, consumed by
        // cameraIntrinsics(): [0]/[5] = 2n/(r-l), 2n/(t-b); [8]/[9] = frustum
        // skew (non-zero under filmOffset / setViewOffset): prepOut_.projP*.

        // Is there any image formation left to do after the overlay pass?
        [[nodiscard]] bool sensorStageActive() const {
            return lens_.active() || sensorNoise_.enabled;
        }
        // Overscan only means anything with a lens: without one the warp is
        // identity and a widened frustum would just silently crop the view.
        // A fisheye never overscans: its field beyond the pinhole's comes from
        // the rig's side views instead (LensRig.hpp), so the primary keeps the
        // camera's own frustum and its full resolution.
        [[nodiscard]] float effectiveOverscan() const {
            return lens_.active() && lens_.model != LensModel::Fisheye ? lensOverscan_ : 1.f;
        }

        // ── The fisheye rig (LensRig.hpp) ───────────────────────────────────
        // A fisheye frame = the primary (the camera's own pinhole) plus up to
        // four side views, addView views that look along the camera's right,
        // left, down and up axes, resampled together in the sensor pass. The
        // views are internal: created, re-aimed every frame and removed here.
        //
        // Called by render() for every perspective frame before it opens; a
        // no-op (and the side views are removed) unless the lens is a fisheye.
        void updateLensRig(Camera& camera);
        // Is a face's view rendering this frame (resources exist, not going
        // away)? Same predicate addSecondaryViewPasses uses to skip one.
        [[nodiscard]] ViewContext* lensRigFaceView(int face);
        void addSensorPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // The side views' own DLSS SR (+ RR) features, brought in line with
        // the primary's at the top of each frame: created when the primary's
        // DLSS runs, recreated on a size or preset change, released when it
        // stops. Their RR guides follow the same way (ensureRrGuides).
        void syncSideViewDlss();
        // Does the CURRENT view feed its own DLSS this frame (the primary, or
        // a side view whose feature is live)?
        [[nodiscard]] bool viewTakesDlss() const {
#if defined(THREEPP_WITH_DLSS)
            if (!view().secondary) return true;
            return view().lensRigFace && view().dlss.active && view().dlss.up && view().dlss.up->valid() &&
                   !view().dlss.up->failing();
#else
            return !view().secondary;
#endif
        }

        // Pack this frame's lens + sensor state for the final SensorPass.
        // Intrinsics go over NORMALIZED (fx/W, fy/H, cx/W, cy/H) so the same
        // numbers are correct in that pass, which runs at DISPLAY extent, as
        // they are at the render extent they were measured on. Advances the
        // noise frame counter, so call exactly once per recorded frame.
        [[nodiscard]] vulkan::SensorPass::Params buildSensorParams();
        void uploadDownwash(uint32_t frame);
        // Whether a live jet of the downwash reaches a grass field whose world
        // REST box is `box` (the wind gate and the cull bound): a conservative
        // test against the jet's farthest reach over that box's heights.
        [[nodiscard]] bool downwashReachesBox(const Box3& box) const;

        bool probeLightingStepped();

        // Phase 1. Per visible field, the index of its ONE DrawInfo record in
        // this view's concatenated draw array — the push constant the particle
        // vertex stage reads. Parallel to particleFields_ /
        // particleFieldPass_->drawStates(), rebuilt by buildIndirectDrawData
        // (and restored from the view cache on its skip path, since the field's
        // DrawInfo is as static as every other one): prepOut_.particleDrawSlots.
        // Definition moved to VulkanViewContext.hpp.
        using ParticleDrawSlot = vulkan::impl::ParticleDrawSlot;

        void prepareDrawGen(uint32_t frame);

        void prepareTlasGen(uint32_t frame);
        void markLodFadeDirty(uint32_t first, uint32_t count) {
            for (auto& d : lodFadeDirty_) d.mark(first, count);
        }
        void markLodFadeWhole() {
            for (auto& d : lodFadeDirty_) d.markWhole();
        }
        VkBuffer uploadLodFades(uint32_t frame);
        VkBuffer uploadLodLevels(uint32_t frame);
        void recordTlasGen(VkCommandBuffer cb, uint32_t frame);
        bool verifyTlasGen(VulkanRenderer::TlasGenCheck& out);
        // The span's record when draw_gen.comp may write its draws (the
        // primary view's eligibility, minus the per-frame conditions), else
        // null. One predicate for buildIndirectDrawData and the frustum cull.
        const BlasRecord* drawGenEligibleRecord(const EntrySpan& sp);
        // The primary view's frame hands draw_gen.comp spans to cull (stage 4):
        // the CPU cull keeps only its span-level answer for them.
        // Not while the overlay depth prepass DRAWS THE SCENE: it draws one
        // entry at a time off this view's CPU cull, which would then hold only
        // the span-level answer for these spans and draw every instance of a
        // span that is partly in view (the pine forest under its HUD: +2.5 ms
        // of GPU). The prepass draws the scene for overlay meshes, lines and
        // sprites only; for field billboards alone it imports the G-buffer's
        // depth and touches no entry (overlayDepthImportsGbuffer).
        [[nodiscard]] bool drawGenCullsOnGpu() const {
            return !view().secondary && gpuDrawGen_ && gpuInstanceExpand_ && drawGen_ && instExpand_ &&
                   !gpuDrawGenCheck_ && !occlusionCullingEnabled_ && !overlayDepthNeedsSceneDraw();
        }
        void recordDrawGen(VkCommandBuffer cb, uint32_t frame);
        bool verifyDrawGen(VulkanRenderer::DrawGenCheck& out);

        void advanceHeadlessClock();
        [[nodiscard]] double frameNowSec() const {
            if (simTimeSec_ >= 0.0) return simTimeSec_;
            if (headlessClockSec_ >= 0.0) return headlessClockSec_;
            return glfwGetTime();
        }
        // The app's or the headless clock, or negative on the wall clock.
        [[nodiscard]] double pinnedClockSec() const {
            return simTimeSec_ >= 0.0 ? simTimeSec_ : headlessClockSec_;
        }
        // The clock the cloud layer and the height fog animate on: the pinned
        // clock when there is one (setSimTime starts at the app's clock,
        // typically 0), else wall seconds since the first call, so the float
        // stays small.
        [[nodiscard]] float mediumClockSec() const {
            if (pinnedClockSec() >= 0.0) return static_cast<float>(pinnedClockSec());
            static const auto epoch = std::chrono::steady_clock::now();
            return std::chrono::duration<float>(std::chrono::steady_clock::now() - epoch).count();
        }

        // (rasterMatTexValid_ — the binding-3 texture-table gate — lives on
        // ViewContext: the raster sets it guards are per-view.)

        // (renderScale < 1 is upsampled by TAA straight to the swapchain.)

        // Deferred render extent: swapchain extent × renderScale_, each axis
        // clamped to ≥ 1px. Exactly equal to the swapchain extent when
        // renderScale_ is 1 (the unscaled fast path).
        VkExtent2D primaryRenderExtent() const {
            const VkExtent2D s = ctx->swapchainExtent();
            if (renderScale_ >= 0.999f) return s;
            const auto px = [](uint32_t v, float k) -> uint32_t {
                const auto r = static_cast<uint32_t>(static_cast<float>(v) * k + 0.5f);
                return r < 1u ? 1u : r;
            };
            return {px(s.width, renderScale_), px(s.height, renderScale_)};
        }
        // ── The two extents every pass in the render chain is written against ──
        // renderExtent()  — where the G-buffer rasterizes and the shade
        //                   dispatches (the TAA resolve's INPUT).
        // viewOutExtent() — where the temporal history lives and the frame is
        //                   finally written (the TAA resolve's OUTPUT).
        //
        // The primary DERIVES both from the swapchain on every call, exactly as
        // before. That is deliberate: caching them would introduce a staleness
        // window on resize / renderScale change that the old code could not
        // have, and the primary path must stay provably identical. Only a
        // secondary reads its stored pair, which is fixed at addView time and
        // never tracks the window.
        //
        // Public addView views are native-res: renderExt == outExt, so their
        // TAA resolve is a plain 1:1 temporal filter. The fisheye rig's side
        // views render at the primary's render scale and upsample like it.
        VkExtent2D renderExtent() const {
            const ViewContext* v = viewPtr();
            if (v && v->secondary) return v->renderExt;
            return primaryRenderExtent();
        }
        VkExtent2D viewOutExtent() const {
            const ViewContext* v = viewPtr();
            if (v && v->secondary) return v->outExt;
            return ctx->swapchainExtent();
        }
        // True when the scene has any content the post-TAA overlay pass will
        // draw this frame: an overlay-tagged mesh, or any Line/LineSegments/
        // Points entry (those always render via the overlay path). The
        // unjittered-depth prepass and the overlay draw BOTH gate on this same
        // current-frame answer, so the prepass that fills + transitions
        // unjitDepth runs in lockstep with the draw that reads it. (Previously
        // the prepass keyed off the *previous* frame's result, so the first
        // frame an overlay appeared — including frame 1 after a resize — the
        // prepass was skipped and the overlay draw read unjitDepth while it was
        // still UNDEFINED: VUID-vkCmdBeginRendering-pRenderingInfo-09588.)
        // Cheap: both lists are already populated by the visibility pass before
        // any command recording, so this is just two size/flag checks.
        // The overlay content that is hard-edged and tested against the
        // prepass's UNJITTERED depth: lines and points, world sprites, and the
        // wireframe / overlay-layer / particle-system entries. These are what
        // the prepass rasterizes the scene for.
        bool sceneHasOverlayMeshes() const {
            if (!lastVisibleLines_.empty()) return true;
            // World-space sprites are drawn in the overlay pass and depth-test
            // against unjitDepth, so they need the prepass + overlay pass too.
            if (!lastVisibleSprites_.empty()) return true;
            // Per SPAN, not per entry — isOverlay is uniform across an
            // InstancedMesh expansion, and isOverlay covers particle
            // billboards too (kSnapParticle folds into isOverlay), so a scene
            // with only particles still triggers the depth prepass + overlay
            // pass the billboard loop needs.
            for (const auto& sp : entrySpans_) {
                if (lastVisibleEntries_[sp.first].isOverlay) return true;
            }
            return false;
        }
        bool sceneHasOverlayContent() const {
            if (sceneHasOverlayMeshes()) return true;
            // ParticleField billboards draw in the overlay pass too, and they
            // are NOT entries flagged isOverlay — a field is one ordinary
            // MeshEntry whose billboard representation is a second, independent
            // draw. A scene of nothing but a campfire's embers would otherwise
            // skip the depth prepass, leaving unjitDepth UNDEFINED for a pass
            // that reads it (VUID-vkCmdBeginRendering-pRenderingInfo-09588).
            // For them alone the prepass takes its cheap form, the G-buffer
            // depth import (overlayDepthImportsGbuffer).
            return sceneHasFieldBillboards();
        }
        // The prepass fills the overlay depth from the G-buffer's depth (one
        // fullscreen triangle) rather than by drawing the scene: nothing but
        // field billboards wants the attachment, and the import pipeline
        // exists. A full-scene depth pass at display resolution cost 0.9 ms of
        // GPU and 1.2 ms of CPU on a 15k-entry forest for a few dozen dust
        // sprites — the jitter-exact depth it buys matters for hard-edged
        // wireframes and lines, not for a soft sprite drawn outside TAA.
        [[nodiscard]] bool overlayDepthImportsGbuffer() const {
            return overlayDepthImportPipeline_ != VK_NULL_HANDLE && !overlayDepthNeedsSceneDraw();
        }
        // The overlay content for which the prepass must draw the scene itself.
        // By default any overlay mesh, line or world sprite. With
        // setOverlayDepthFromGbuffer the import serves them all but lines and
        // wireframe materials: those sit ON a surface, and against the
        // resampled (render-scale, jittered) depth they would flicker in and
        // out of it; a pane or a sprite in front of the scene does not.
        bool overlayDepthNeedsSceneDraw() const {
            if (!overlayDepthFromGbuffer_) return sceneHasOverlayMeshes();
            if (!lastVisibleLines_.empty()) return true;
            for (const auto& sp : entrySpans_) {
                const MeshEntry& e0 = lastVisibleEntries_[sp.first];
                if (!e0.isOverlay || e0.isParticle || !e0.mesh) continue;
                const auto mat = e0.mesh->material();
                if (auto* wf = dynamic_cast<const MaterialWithWireframe*>(mat.get()); wf && wf->wireframe) return true;
            }
            return false;
        }

        // (cameraUbos moved to ViewContext.)

        // Hand a resource to the retire queue, stamped with this frame's serial.
        void retire(Buffer&& b)   { retireQueue_.retire(std::move(b), frameSerial_); }
        void retire(Image2D&& i)  { retireQueue_.retire(std::move(i), frameSerial_); }
        void retireAS(VkAccelerationStructureKHR as) { retireQueue_.retireAS(as, frameSerial_); }
        // Frame-start reclaim: destroys everything whose referencing frame has
        // provably completed (fence waited). Call right after the
        // vkWaitForFences(inFlight[currentFrame]) at each frame-begin path.
        void drainRetireQueue() {
            retireQueue_.drain(*ctx, frameSerial_, kFramesInFlight);
            reapRetiredRecords(false);
        }
        // Post-vkDeviceWaitIdle flush: whole device idle ⇒ destroy all queued
        // resources now, or they leak at device destroy. Call after EVERY
        // remaining vkDeviceWaitIdle that precedes resource destruction.
        void flushRetireQueue() {
            retireQueue_.flushAll(*ctx);
            reapRetiredRecords(true);
        }
        // On the slot path nothing rebuilds, so nothing runs the structural
        // rebuild's prune: a streamed world (terrain tiles, cells) frees a
        // geometry and its textures with every mesh it drops, and their
        // records and bindless texture slots stayed until the next rebuild,
        // which never came (a long GeoScene flight ran out of the 2048 texture
        // slots and drew its new tiles white). pruneFreedAssets, at the end of
        // each slot edit, retires the records of geometries that died and the
        // images of textures that died, with this frame's serial; the slots
        // go back to the free list at once (each frame-in-flight descriptor
        // set is rewritten before the retired image is destroyed, as
        // refreshDirtyMaterialTextures does). VulkanCoreSlots.cpp.
        void pruneFreedAssets();
        void reapRetiredRecords(bool all);
        // A GrassMesh on the slot path (VulkanCoreSlots.cpp). Its state is the
        // field's own record plus its rest pose and height fractions: a field
        // the slot edit admits gets one built in the edit's batch (or, added
        // hidden, over the frames after) and primed by its first deform in the
        // frame; one whose mesh or geometry died, or whose mesh took another
        // geometry, is retired: the record with retiredRecords_, the two
        // input buffers through the retire queue.
        GrassMeshState* admitGrassState(GrassMesh& gm);
        void retireGrassState(GrassMeshState& st);
        static bool grassInputsValid(const BufferGeometry& g);
        // Retire (rather than inline-destroy) a particle geometry record's
        // buffers on an in-flight topology change. Mirrors destroyParticleGeomRec.
        void retireParticleGeomRec(ParticleGeomRec& rec) {
            for (auto& a : rec.anim) retireParticleAnimSet(a);
            retire(std::move(rec.uv));
            retire(std::move(rec.index));
        }
        void retireParticleAnimSet(vulkan::impl::ParticleAnimSet& a) {
            retire(std::move(a.position));
            retire(std::move(a.normal));
            retire(std::move(a.color));
        }

        // (orthoFrame_ moved to ViewContext — under multi-view the projection
        //  kind is a property of the VIEW, not of the frame: a perspective
        //  primary and an orthographic secondary coexist in one submission.
        //  Distinct from orthoSceneRendering_, which is the user's standing
        //  permission rather than any view's projection.)

        // Does THIS render() call mean "shade the scene through a parallel
        // projection"? Only a standalone one (nothing in flight) can — a second
        // render() with an ortho camera over an open frame is the HUD pattern
        // and stays overlay-only whatever the flag says.
        [[nodiscard]] bool orthoSceneRender(const Camera& camera) const {
            return orthoSceneRendering_ && frameState_ == FrameState::Idle &&
                   camera.is<OrthographicCamera>();
        }

        // Where one channel's pixels come from THIS frame: the image, the
        // aspect and the layout it rests in at the record tail. The table is
        // readViewGBufferAOVs' verbatim, plus the Color row (the swapchain for
        // the primary, the view's colorTarget for a secondary), and it is
        // resolved twice — once at enable to size the export, once per frame to
        // record the copy — so the two can never disagree about a format.
        struct FrameInteropSource {
            VkImage            image      = VK_NULL_HANDLE;
            VkImageAspectFlags aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
            VkImageLayout      restLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t width = 0, height = 0, bpp = 0;
            bool    bgra = false;
        };
        bool frameInteropSource(uint32_t viewHandle, VulkanRenderer::FrameChannel channel,
                                uint32_t gbufSlot, uint32_t imageIndex,
                                FrameInteropSource& out);
        std::vector<VulkanRenderer::FrameInteropExport>
        enableFrameInterop(uint32_t viewHandle,
                           const std::vector<VulkanRenderer::FrameChannel>& channels);
        void disableFrameInterop(uint32_t viewHandle);
        [[nodiscard]] bool frameInteropActive(uint32_t viewHandle) const {
            for (const auto& s : frameInterops_)
                if (s.viewHandle == viewHandle) return true;
            return false;
        }
        // Invalidation (see the header): the source images are about to be
        // reallocated or destroyed, so the exports the foreign API imported are
        // torn down with a warning rather than left pointing at freed images.
        void invalidateFrameInterop(uint32_t viewHandle, const char* why);
        bool syncFrameInterop();
        // A pass of the post-view tail graph, after the scene capture: one copy
        // per armed channel. Adds nothing when no view is armed.
        void addFrameInteropPass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        void destroyFrameInterops();

        explicit Impl(Canvas& c);

        // Print a VMA memory-usage summary to stderr: the allocator's reserved-vs-
        // live block totals and per-heap usage/budget. Cheap; for manual/debug use
        // (there is otherwise no memory introspection in the renderer). Reflects
        // all VMA allocations — G-buffer, history, denoiser scratch, upscalers,
        // and any scene AS/geometry live at the call site.
        void dumpMemoryStats(const char* tag = "") const;
        // Prints the one "deforming geometry landed in system memory" line
        // when noteDeformPlacement counted any bytes since the last call.
        void reportDeformPlacement();

        ~Impl();

        void createCommandResources();

        // (Re)create the per-swapchain-image present-wait semaphores. Called
        // from createCommandResources and again on every swapchain recreation
        // (the image count can change and retired semaphores may be left in
        // an indeterminate signalled state). Requires an idle device.
        void createRenderFinishedSemaphores();

        // The frame's swapchain image. Normally a straight
        // vkAcquireNextImageKHR; when presents are suppressed (headless canvas)
        // the slot's image is acquired once and reused from then on, because a
        // never-presented image is never released back to the presentation
        // engine and so can never be re-acquired. Sets pinnedSwapImage_ /
        // acquireSemPending_ and returns the acquire's VkResult (VK_SUCCESS on
        // the reuse path) for the caller's OUT_OF_DATE / SUBOPTIMAL handling.
        VkResult acquireOrReuseSwapchainImage(uint32_t& imageIndex);

        // Allocate, begin, return a one-shot command buffer.
        VkCommandBuffer beginOneShot();

        // The optional `label` is folded into the error message when the
        // submit / wait throws — handy for distinguishing which one-shot site
        // hit a device-lost during runtime debugging.
        void endAndSubmitOneShot(VkCommandBuffer cb, const char* label = "one-shot");

        void beginOneShotBatch() { oneShotBatch_ = true; }
        void flushOneShotBatch();
        // The batch submitted WITHOUT a host wait, for work the frame about
        // to be recorded consumes (the slot path's BLAS builds and texture
        // uploads): the frame's command buffer is submitted after it on the
        // same queue, and the render graph's entry barrier (ALL_COMMANDS,
        // every prior submission in its first scope) orders and publishes it.
        // Its transients go to the frame-serial retire queue, its command
        // buffer to asyncOneShots_ until its fence signals. Its static BLASes
        // are compacted after that (compactAsync): their compacted-size
        // queries ride the batch, in a pool of its own.
        void flushOneShotBatchAsync();
        // Free the command buffers and fences of async batches that finished
        // (all of them when the device is known idle), and compact what a
        // finished batch queried.
        void reapAsyncOneShots(bool deviceIdle);
        // A finished async batch's queried BLASes: each copied with
        // COPY_MODE_COMPACT into storage of its compacted size, in one more
        // async batch (ahead of the frame in queue order, as the build was),
        // the record swapped to the copy at once and the build-size pair
        // retired with the frame about to be recorded, whose TLAS is then
        // built anew (blasRereferenced_): the frames still in flight trace
        // the old pair, this one the new.
        void compactAsync(const AsyncCompaction& c);
        // A slot edit that needs more slots than the pool holds: every
        // pool-sized GPU buffer (the geometry and material desc rings, the
        // motion matrices and moved bits, the TLAS and its instance buffers)
        // is made anew for `capacity` entries and refilled from its host copy;
        // the old ones are retired with this frame (the frames in flight keep
        // reading them), and each frame-in-flight's sets that name them are
        // rewritten after its own fence. No device wait; the entries, spans
        // and every temporal history stay. VulkanCoreSlots.cpp.
        void growSlotPool(uint32_t capacity);

        // The flags a static record, and an auto-LOD level, is built with.
        VkBuildAccelerationStructureFlagsKHR staticBlasBuildFlags() const {
            VkBuildAccelerationStructureFlagsKHR f = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            if (blasCompact_) f |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            return f;
        }
        // Records, after builds already recorded into `cb`, one barrier and the
        // compacted-size query of ases[k] into slot k.
        void recordCompactedSizeQueries(VkCommandBuffer cb, const std::vector<VkAccelerationStructureKHR>& ases);
        // One compaction round over structures whose sizes were queried into
        // slots 0..n-1 by recordCompactedSizeQueries and whose build has
        // completed: creates each compacted pair, copies all of them in ONE
        // one-shot (one wait), then swaps the handles in. `retire` keeps the
        // replaced pairs alive and records address remaps (batch mode);
        // otherwise they are destroyed at once. A job whose compacted size is
        // not smaller than its storage is left as it is. Returns the bytes
        // freed per job (0 for a job left as it was).
        struct BlasCompactJob {
            VkAccelerationStructureKHR* as = nullptr;
            Buffer* storage = nullptr;
            VkDeviceAddress* address = nullptr;
            bool* compacted = nullptr;// may be null (auto-LOD levels)
        };
        std::vector<VkDeviceSize> compactBlases(const std::vector<BlasCompactJob>& jobs, bool retire);
        // Rewrites TLAS instance references built from batch-compacted records
        // and frees the replaced pairs. Called right before buildTlas.
        void applyBlasCompactionRemap(std::vector<VkAccelerationStructureInstanceKHR>& instances);
        void destroyBlasCompactionResources();
        // Free `buf` now, or defer it to the next flush when a batch is open.
        void destroyBufferMaybeBatched(Buffer& buf) {
            if (oneShotBatch_) {
                oneShotBatchGarbage_.push_back(buf);
                buf = {};// caller's handle retired; the garbage copy owns it now
            } else {
                destroyBuffer(ctx->allocator(), buf);
            }
        }

        bool hostMappable(const Buffer& b) const;
        // deviceLocal: AUTO_PREFER_DEVICE, no host access (TRANSFER_DST is
        // added for the staging copy); otherwise AUTO + sequential host writes.
        Buffer createGeomBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, bool deviceLocal);
        // Writes `bytes` from `src` at `dstOffset` of `dst`: at once when dst
        // is host-mappable, else queued in `up` for recordStagedUploads.
        void stageUpload(StagedUploads& up, const Buffer& dst, const void* src,
                         VkDeviceSize bytes, VkDeviceSize dstOffset = 0);
        // stageUpload without the host-write shortcut: always queued for the
        // staging copy. For writes recorded into a FRAME command buffer, where
        // an earlier frame still in flight may read `dst` (a host-mappable
        // device-local buffer in the BAR is no exception): the copy is ordered
        // behind those reads on the queue, a host memcpy is not.
        void stageCopy(StagedUploads& up, const Buffer& dst, const void* src,
                       VkDeviceSize bytes, VkDeviceSize dstOffset = 0);
        // Queues the uv, color and index buffers of `rec` whose attribute
        // version moved since rec.attrVersions, whole, in the record's own
        // (possibly packed) formats: through stageCopy when `viaCopy` (frame
        // command buffer), stageUpload otherwise. Returns true when the index
        // was among them, which rules out a BLAS refit. Stamps nothing.
        bool stageChangedSideAttributes(StagedUploads& up, const BlasRecord& rec,
                                        const BufferGeometry& geom, bool viaCopy);
        // Records `up`'s queued copies into `cb` from one staging buffer, then
        // a barrier that makes them visible to every later command (this cb's
        // BLAS build and prevVertex seed, and later frames). `dstUntouched`:
        // the destinations are fresh, so no barrier is needed before the
        // copies. Returns the staging buffer (empty when nothing was queued);
        // the caller frees it once cb has executed (destroyBufferMaybeBatched
        // right after endAndSubmitOneShot). Clears `up`. `keepCapacity`: `up`
        // is a scratch its caller fills again next frame, so its byte vector
        // keeps its allocation; otherwise that is released with the upload.
        Buffer recordStagedUploads(StagedUploads& up, VkCommandBuffer cb, bool dstUntouched,
                                   bool keepCapacity = false);
        // recordStagedUploads in two halves, the first for any thread:
        // makeStaging copies up's bytes into a new host-visible staging buffer
        // (clears up.data, keeps up.copies; the allocator and nothing of the
        // renderer's state), recordStagedCopies records the copies and their
        // barriers from it into cb (clears up.copies).
        Buffer makeStaging(StagedUploads& up, bool keepCapacity = false);
        void recordStagedCopies(StagedUploads& up, const Buffer& staging, VkCommandBuffer cb, bool dstUntouched);
        // The one-shot form: begin, record, submit (or join the open batch),
        // release the staging buffer.
        void submitStagedUploads(StagedUploads& up, const char* label);

        // Who rewrites an updatable record's vertex and normal after its
        // build. Host: a per-frame host path writes them (morph blend, the
        // displaced ocean's existing placement), so they stay host-visible.
        // Gpu: only compute shaders write them (skinning, tet skinning, grass
        // wind), so they are device-local like a static record's: a
        // host-visible deformer buffer that spills out of the 256 MiB BAR
        // window lands in system memory and its deform pass crosses PCIe
        // every frame.
        enum class VertexWriter : uint8_t { Host, Gpu };
        // Adds `b`'s bytes to deformHostBytes_ when it landed outside device
        // memory (the dumpMemoryStats "host" bucket). Called once per deformer
        // record or input buffer at creation; ensureSceneBuilt prints one line
        // per structural build when the sum is non-zero, then clears it.
        void noteDeformPlacement(const Buffer& b);

        // The composite version of the tracked attributes (GeomAttrRefs).
        static unsigned int geomVersionOf(const BufferGeometry& g) {
            return GeomAttrRefs::of(g).versionSum();
        }

        // The parts of geomVersionOf, kept apart.
        static AttrVersions attrVersionsOf(const BufferGeometry& g) {
            return GeomAttrRefs::of(g).versions();
        }

        // Stamp both version snapshots after the record's buffers were filled
        // from `g` (a build or a refresh).
        static void stampVersions(BlasRecord& rec, const BufferGeometry& g) {
            rec.geomVersion = geomVersionOf(g);
            rec.attrVersions = attrVersionsOf(g);
        }

        // An in-place refresh writes uv and color in the format the record
        // was BUILT with. A packed format was picked because every value then
        // fit it (uv in [0,1] for unorm16, LDR color for unorm8); an edit that
        // leaves that range would be clamped. False when a CHANGED uv or
        // color attribute no longer fits, so the caller re-admits the record
        // through the structural rebuild, which picks the format afresh.
        static bool refreshFitsPacking(const BlasRecord& rec, const BufferGeometry& g);

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        // allowPacked: pack normal/uv/color into narrow device formats
        // (BlasRecord::packedMask). Static geometry only — the deforming
        // callers (skinned / tet / displaced / grass / morphed) keep the
        // default false because their per-frame compute rewrites assume
        // tightly-packed float layouts.
        // `updatable`: the record is refit in place later (deformers), so it is
        // built ALLOW_UPDATE and never compacted. Static records pass false and
        // are compacted (see staticBlasBuildFlags); an interop-marked geometry
        // is always updatable.
        // `rtIndices`: when given, the AS is built over these uint32 indices
        // instead of the geometry's own (BlasRecord::rtIndex); the record's
        // `index` still carries the geometry's for the raster.
        // `writer`: for an updatable record, VertexWriter::Gpu places vertex
        // and normal device-local (see VertexWriter); ignored for static and
        // interop records.
        // traced = false builds the record as it is built without ray query:
        // its buffers, no acceleration structure. For a geometry that only
        // overlay meshes draw (a wireframe, an overlay-layer panel): no ray
        // meets an overlay, it is never a TLAS instance, and its structure was
        // build time at admission, a refit per edit and device memory for
        // nothing (ensureSceneBuilt decides per geometry, see tracedGeoms).
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom, bool allowPacked, bool updatable,
                                                 const std::vector<uint32_t>* rtIndices = nullptr,
                                                 VertexWriter writer = VertexWriter::Host,
                                                 bool traced = true);
        // `disp`: positions displaced by a material displacementMap (see
        // DisplacementSpec), read by the caller like interopTarget.
        bool prepareBlas(const BufferGeometry& geom, bool allowPacked, bool updatable,
                         const std::vector<uint32_t>* rtIndices, VertexWriter writer, bool traced,
                         bool interopTarget, BlasPrep& p, const DisplacementSpec* disp = nullptr);
        // geom's positions moved along its normals by d's map: false (and a
        // one-time warning) when the geometry or the image can't be read.
        static bool displacePositions(const BufferGeometry& geom, const DisplacementSpec& d,
                                      std::vector<float>& out);
        std::unique_ptr<BlasRecord> finishBlas(const BufferGeometry& geom, BlasPrep& p);
        void discardBlasPrep(BlasPrep& p);
        std::unique_ptr<GrassPrepData> prepareGrassData(const BufferGeometry& geom, bool interopTarget);
        void beginGrassPrep(GrassMesh& gm);
        [[nodiscard]] bool grassPrepPending(const GrassMesh& gm) const;
        void discardGrassPrep(GrassPrepData& d);
        // Drops preps whose field died or changed geometry (all: every one,
        // at shutdown); waits for a worker still on one.
        void sweepGrassPreps(bool all);

        // Allocate or look up the per-SkinnedMesh BLAS state. Builds the BLAS
        // once with the current pose; subsequent dirty frames go through
        // refreshSkinnedBlas which only re-skins + rebuilds in-place. Returns
        // null if the geometry is unsupported (no position/normal/skin attrs).
        SkinnedMeshState* ensureSkinnedBlas(SkinnedMesh& sm);

        // Re-skin the SkinnedMesh's vertices/normals on the CPU, copy them
        // into the host-mapped vertex/normal buffers, and rebuild the BLAS in
        // place (same AS handle and storage so the device address — and the
        // TLAS reference to it — stay valid). The TLAS doesn't need refit
        // for pose-only changes; the instance's transform is unchanged.
        void refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st);

        // Allocate or look up the per-tet-skinned-mesh BLAS state (PhysX soft body).
        // Mirrors ensureSkinnedBlas: build the BLAS once at rest pose, upload the
        // static tet bindings + rest normals + baked Dr^-1 columns, allocate the
        // per-frame tet-position buffer + descriptor set, then queue a first refresh.
        // Returns null if the geometry/material lack the tet attributes that
        // SoftBody::enableGpuSkinning() sets.
        TetMeshState* ensureTetBlas(Mesh& m);

        // Per-frame: upload the soft body's current collision-tet positions into the
        // tet-position buffer (read from the material's tet texture image, which
        // PhysxWorld::syncSoftBodies refreshes each frame), then queue the GPU skin +
        // BLAS rebuild — recorded in recordCommandBuffer next to the skinned path.
        void refreshTetBlas(Mesh& m, TetMeshState& st);

        // Rewrite binding 6 (tetPos — see tet_skinning.comp) of every ring slot's
        // descriptor set to point at `buf` (the single exported interop buffer).
        // Used by the interop enable swap; the other 8 bindings are untouched.
        // The caller vkDeviceWaitIdles first, so no set is in a pending cb.
        void rewriteTetPosBinding(TetMeshState& st, VkBuffer buf);

        // Zero-copy interop enable: swap the mesh's tet-position buffer for an
        // EXPORTED dedicated device allocation and register the CUDA copy that
        // fills it each frame. Once-per-body; drains the device (the in-flight
        // frames' dispatches read the buffer being replaced, and this is a
        // registration-time call, not a per-frame one).
        VulkanRenderer::SoftBodyInteropHandle
        enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy) {
            auto it = tetMeshStates.find(&mesh);
            if (it == tetMeshStates.end() || !ctx->externalMemorySupported()) return {};
            auto& st = *it->second;
            if (st.tetPosExt.handle != VK_NULL_HANDLE) {// already enabled — same allocation
                st.tetPosExternalCopy = std::move(deviceCopy);
                return {vulkan::takeOsHandle(ctx->device(), st.tetPosExt),
                        static_cast<size_t>(st.tetPosExt.size)};
            }
            if (st.tetPosBytes == 0 || st.tetDescSet[0] == VK_NULL_HANDLE) return {};
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop enable)");
            st.tetPosExt = vulkan::createExternalBuffer(
                    ctx->physicalDevice(), ctx->device(), st.tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            rewriteTetPosBinding(st, st.tetPosExt.handle);
            for (auto& slot : st.tetPos)// CPU-path ring no longer read
                destroyBuffer(ctx->allocator(), slot);
            st.tetPosExternalCopy = std::move(deviceCopy);
            // takeOsHandle, not .osHandle: on POSIX the fd's ownership passes to
            // the importer, so the record must stop believing it owns it.
            return {vulkan::takeOsHandle(ctx->device(), st.tetPosExt),
                    static_cast<size_t>(st.tetPosExt.size)};
        }

        // Interop disable / CUDA-import-failure fallback: restore the host-visible
        // VMA buffer and the CPU upload path. The caller must have stopped (or
        // never started) the CUDA writes into the exported memory.
        void disableSoftBodyInterop(const Mesh& mesh);

        // F6: the same idea one abstraction up — a whole FIELD's positions,
        // exported once and filled device-to-device every frame. Thin: the pass
        // owns the state, the gate and the fallback (ParticleFieldPass::
        // enableInterop); this converts its record to the public handle type.
        // Valid BEFORE the first frame: the pass (and, inside it, the field's
        // device state and its exported allocation) is brought up on demand, so
        // an application arms interop at setup and no frame ever runs with an
        // Interop field whose copy is not registered.
        VulkanRenderer::ParticleFieldInteropHandle
        enableParticleFieldInterop(ParticleField& field, std::function<void()> deviceCopy) {
            ensureParticleFieldPass();
            const auto e = particleFieldPass_->enableInterop(field, std::move(deviceCopy));
            return {e.osHandle, e.sizeBytes, e.attrHandle, e.attrSizeBytes};
        }

        // Zero-copy MESH VERTEX interop — the same idea again, now on a plain
        // mesh's position/normal attributes. Definitions in VulkanCoreGeometry.cpp
        // beside the dynamic-refit machinery they hook into; the design notes
        // live on BlasRecord::posExt (why a copy and not a buffer swap) and on
        // VulkanRenderer::enableVertexInterop (the caller-facing contract).
        VulkanRenderer::VertexInteropHandle
        enableVertexInterop(const Mesh& mesh, std::function<void()> deviceCopy, bool validate,
                            bool stableCorrespondence);
        void disableVertexInterop(const Mesh& mesh);
        void setStableCorrespondence(const Mesh& mesh, bool stable);
        void setEmissiveCastsLight(const Object3D& obj, bool castsLight);
        // The record backing `mesh` for interop purposes, or null. Interop is a
        // plain-mesh feature: skinned / tet / displaced / grass / morphed meshes
        // already have a per-frame GPU producer of their own that would overwrite
        // whatever the foreign API wrote, so they are refused rather than
        // silently fighting over the buffer.
        BlasRecord* interopRecordFor(const Mesh& mesh);

        // Definition moved to VulkanGeometryState.hpp.
        using GeomRefreshOp = vulkan::impl::GeomRefreshOp;

        // Batched per-frame BLAS refresh. For each op:
        //   (1) snapshot rec.vertex → rec.prevVertex (chit motion-vector channel)
        //   (2) host memcpy of new positions/normals/uv/index into rec.vertex etc.
        //   (3) refit rec.as in place (MODE_UPDATE, periodic MODE_BUILD every
        //       kBlasFullRebuildInterval frames to keep the BVH balanced)
        //
        // Phase (1) and (3) record into one shared command buffer each, so the
        // whole batch costs 2 vkQueueSubmit + 2 vkQueueWaitIdle pairs regardless
        // of N. Previously, looping refreshGeomBlas paid that pair per geometry
        // — at ~30 µs queue overhead per submit, 5 soft bodies = 300 µs/frame
        // wasted on submit round-trips alone.
        //
        // Caller must (a) gate on equal vertex/index counts (topology must not
        // change — fall through to fullRebuild if it does) and (b) have already
        // drained in-flight GPU work; phase (2)'s host memcpys race in-flight
        // closest_hit reads of rec.vertex otherwise.
        void refreshGeomBlasBatch(const std::vector<GeomRefreshOp>& ops);

        // Frame-cb twin of refreshGeomBlasBatch, for every in-place edit of a
        // record with no auto-LOD chain: graduated records
        // (BlasRecord::perFrameDynamic) CPU-pack new positions/normals into
        // this frame's slot of their staging ring, the others (an occasional
        // edit) into one staging buffer retired with the frame, honouring
        // updateRange. Then records vertex→prevVertex snapshot, staging→vertex/
        // normal copies and the batched BLAS refit into `cb` with barriers:
        // zero extra submits, zero waits. Also records the one-frame
        // prevVertex re-sync for pendingDynamicPrevResyncs_.
        // Consumes (clears) both pending lists.
        void recordDynamicGeomRefits(VkCommandBuffer cb);

        // Size of `rec`'s structure in the updatable lineage (PREFER_FAST_TRACE
        // | ALLOW_UPDATE) over `geom`'s live span.
        // An occasional edit's positions and normals (the updateRange part) as
        // copies in `up`, in the record's formats; consumes both ranges.
        void stageEditedVertexRange(StagedUploads& up, const BlasRecord& rec,
                                    const FloatBufferAttribute& posAttr,
                                    const FloatBufferAttribute& nrmAttr);
        VkDeviceSize updatableBlasSize(const BlasRecord& rec, const BufferGeometry& geom) const;
        // A static record turning dynamic lives in storage sized for its static
        // build or its compacted copy: too small for the updatable lineage.
        // Moves it to new storage of `asSize` (no-op, false, when it already
        // fits) and flags the TLAS instance refill (lodChangedThisFrame_). The
        // old pair is retired (`retireOld`: an earlier frame still in flight
        // may trace it) or freed at once (the caller drained the device). The
        // new structure is empty: the caller's next build must be MODE_BUILD.
        bool moveToUpdatableBlasStorage(BlasRecord& rec, VkDeviceSize asSize, bool retireOld);

        // ── Morph-target helpers ─────────────────────────────────────────

        static bool isMorphedMesh(const Mesh& m) {
            return m.geometry()->getMorphAttributes().count("position") > 0;
        }

        static void cpuMorphBlend(Mesh& mesh,
                                  std::vector<float>& outPos,
                                  std::vector<float>& outNorm);
        // The blend itself, for any object's influences; outNorm == nullptr
        // skips the normals (Points).
        static void cpuMorphBlend(const BufferGeometry& geom,
                                  const std::vector<float>& influences,
                                  std::vector<float>& outPos,
                                  std::vector<float>* outNorm);

        MorphedMeshState* ensureMorphedBlas(Mesh& mesh) {
            auto it = morphedMeshStates.find(&mesh);
            if (it != morphedMeshStates.end()) return it->second.get();

            auto rec = buildBlasFor(*mesh.geometry(), /*allowPacked=*/false, /*updatable=*/true);
            if (!rec) return nullptr;
            rec->liveCheck = mesh.geometry();

            auto state = std::make_unique<MorphedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = mesh.geometry();
            state->owner.bind(mesh);

            auto* raw = state.get();
            morphedMeshStates.emplace(&mesh, std::move(state));

            refreshMorphedBlas(mesh, *raw);
            return raw;
        }

        void refreshMorphedBlas(Mesh& mesh, MorphedMeshState& st);

        // ── DisplacedMesh helpers ────────────────────────────────────────
        // Lazy create + initialize the per-DisplacedMesh state. Builds the
        // BLAS from the rest geometry (will get overwritten by the displace
        // compute pass before the first ray-trace) and stands up the FFT
        // cascade. Returns nullptr if the geometry is unsupported (must be a
        // square indexed plane with N×N vertices for some power-of-two N).
        DisplacedMeshState* ensureDisplacedState(DisplacedMesh& dm);
        // Chop normals: builds their resources the first time the mesh asks,
        // resolves st.chopOn and points binding 13 to match (see the definition).
        void syncOceanChop(DisplacedMesh& dm, DisplacedMeshState& st);
        void syncOceanCrestFoam(DisplacedMesh& dm, DisplacedMeshState& st);// called by syncOceanChop
        void syncOceanClimate(DisplacedMesh& dm, DisplacedMeshState& st);  // called by syncOceanChop

        // Per-frame: run the FFT chain → water_displace → BLAS rebuild for
        // one DisplacedMesh. Mirrors refreshSkinnedBlas's structure.
        // Record the full per-frame water update — FFT chain → water_displace →
        // world foam → in-place BLAS rebuild — into `cb`. NO submit and NO CPU
        // wait: the caller either batches this into the frame command buffer
        // (recordCommandBuffer draining pendingDisplacedDeforms_, the per-frame
        // path) or wraps it in a one-shot for the rare structural first build
        // (refreshDisplacedBlas below).
        // timed: write the TP_Ocean* timestamp brackets. Only the per-frame
        // path may pass true, and only for the FIRST displaced mesh of the
        // frame — the query pool has one slot pair per pass, and the one-shot
        // first-build cb never reset those slots at all.
        void recordDisplacedDeform(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds, bool timed = false);
        // Its two halves, which recordDisplacedDeform runs back to back: the
        // update (FFT, displace, foam, wake, readback copies) and the BLAS
        // rebuild over the displaced vertices. The async-compute pilot records
        // them into different command buffers on different queues.
        void recordDisplacedUpdate(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds, bool timed);
        // The patch mesh's frame (DisplacedMesh::Lod): selects the patches,
        // assigns the slots, uploads the table; returns its device address.
        VkDeviceAddress updateOceanLod(DisplacedMesh& dm, DisplacedMeshState& st);
        void recordDisplacedBlasRebuild(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, bool timed);

        // Mirror cascade height fields into DisplacedMesh for CPU sampling
        // (boat hydrodynamics etc.). On the batched per-frame path this reads
        // what the last COMPLETED frame's copy wrote — one/two frames of
        // latency, which wave sampling tolerates (same class as the skinned
        // path's single-buffered bone uploads).
        void mirrorDisplacedHeightfields(DisplacedMesh& dm, DisplacedMeshState& st);

        // Synchronous variant for the structural (re)build path: the TLAS /
        // geometry descriptors built right after need the displaced result in
        // place. Rare (scene restructure), so the one-shot block is fine here.
        void refreshDisplacedBlas(DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds);

        // ── GrassMesh helpers ────────────────────────────────────────────
        // Conservative world-space AABB for a wind-swayed grass field, shared by
        // the frustum cull, the occlusion-cull meta build, and the distance-gated
        // freeze so all three test the SAME box. The CPU-side "position" attribute
        // IS the rest pose (the GPU displaces a private copy into the BLAS), and
        // grass_wind.comp bends each vertex by windDir·(gust·windStrength·hf²) with
        // |gust| ≤ 0.85 and hf² ≤ 1, so the swayed vertex never leaves the rest
        // AABB dilated by windStrength (windDir is ~unit — |component| ≤ 1). That
        // makes the dilation a provably conservative bound, which is the whole
        // reason grass can be culled at all. A rotor downwash jet that reaches
        // the field adds its lean: at most the blades' height sideways and
        // never below the base (downwash.glsl, dwGrassOffset), so the box grows
        // by that height. Returns false (→ caller treats the
        // field as always-visible / always-animate) if the geometry has no bounds.
        // en.mesh is known to be a GrassMesh (cached en.isGrass flag), so the
        // static_cast avoids the per-entry dynamic_cast the hot loops must not pay.
        bool grassSwayWorldAabb(const MeshEntry& en, Box3& out) const {
            auto* gm = static_cast<GrassMesh*>(en.mesh);
            auto geom = gm->geometry();
            if (!geom) return false;
            if (!geom->boundingBox) geom->computeBoundingBox();
            if (!geom->boundingBox) return false;
            out = *geom->boundingBox;
            Matrix4 w;
            std::memcpy(w.elements.data(), en.worldMatrix.data(), 64);
            out.applyMatrix4(w);
            float grow = gm->params.windStrength;
            if (downwash_ && !downwash_->empty() && downwashReachesBox(out)) {
                float h = gm->params.bladeHeight;
                if (!(h > 0.f)) {
                    const auto it = grassStates.find(gm);
                    h = it != grassStates.end() ? it->second->bladeHeight : 0.f;
                }
                grow += h * w.getMaxScaleOnAxis();
            }
            out.expandByScalar(grow);
            return true;
        }

        // Distance-gated wind/refit freeze decision (see GrassMesh::Params::
        // maxAnimDistance). 0 ⇒ always animate. Otherwise animate only while the
        // camera is within maxAnimDistance of the field's sway-dilated AABB;
        // beyond that the field is frozen (skips the per-frame dispatch + refit,
        // holds its last pose). Distance is measured to the nearest point of the
        // box so a large field animates whenever the camera is anywhere near it.
        // Whether a field with a state deforms this frame. Hidden: no (no ray
        // hits it, no raster draws it), except an admission's prime, which
        // gives its BLAS a pose before any show; the skip is remembered, and
        // the first frame shown again is a prime whatever the distance gate
        // says (GrassMeshState::skippedHidden).
        static bool grassWantsDeform(GrassMeshState& st, bool hidden, bool animate) {
            if (hidden) {
                if (st.primePending) return true;
                st.skippedHidden = true;
                return false;
            }
            if (st.skippedHidden) {
                st.skippedHidden = false;
                st.primePending = true;
            }
            return st.primePending || animate;
        }

        bool grassShouldAnimate(const MeshEntry& en, const Vector3& camPos) const {
            auto* gm = static_cast<GrassMesh*>(en.mesh);
            const float maxD = gm->params.maxAnimDistance;
            if (maxD <= 0.f) return true;
            Box3 box;
            if (!grassSwayWorldAabb(en, box)) return true;// no bounds → animate
            if (box.distanceToPoint(camPos) <= maxD) return true;
            // A downwash jet near the field keeps it awake, and so does the
            // bend the last one left (one deform to stand back up).
            if (downwash_ && !downwash_->empty() && downwashReachesBox(box)) return true;
            const auto it = grassStates.find(gm);
            return it != grassStates.end() && it->second->downwashBent;
        }

        // Lazy create the per-GrassMesh state: a BLAS over the rest geometry
        // (built ALLOW_UPDATE, refit each frame) plus an immutable copy of the
        // rest positions and the per-vertex height fractions the wind shader
        // reads. Returns nullptr if the geometry lacks position / heightFrac.
        GrassMeshState* ensureGrassState(GrassMesh& gm);

        // Record the grass wind deform (rest → displaced positions in the BLAS
        // vertex buffer) + in-place BLAS refit into `cb`. No submit — the caller
        // batches these into the main frame command buffer (recordCommandBuffer),
        // exactly like the skinned-mesh path, so there is NO mid-frame
        // vkQueueWaitIdle. Uses the persistent per-BLAS scratch (grass geometry
        // is fixed-size, so it's allocated once on first use). Normals are left
        // static, so only the position buffer is written/barriered. Before the
        // dispatch it copies the current pose into blas->prevVertex, the motion
        // vectors' previous pose, as the skinned and displaced paths do.
        void recordGrassDeform(VkCommandBuffer cb, GrassMesh& gm, GrassMeshState& st);

        // Grass fields that swayed last frame and are not deformed this frame
        // (the distance-gated freeze): copy vertex -> prevVertex once so their
        // motion collapses to zero. grassPrevSettlePending() says whether any
        // is due, so the deform.grass pass exists on a frame with no deform.
        [[nodiscard]] bool grassPrevSettlePending() const;
        void recordGrassPrevSettles(VkCommandBuffer cb);

        // One-shot wrapper — used only for the initial prime during scene build
        // (rare). The per-frame path records into the frame cb via the pending
        // queue, so it never drains. Leaves prevVertex == vertex (no motion on
        // the field's first frame).
        void refreshGrassBlas(GrassMesh& gm, GrassMeshState& st, float /*elapsedSeconds*/);

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses.
        // buildNow false: the structure and its buffers only, sized for the
        // pool; the caller's frame records its first build (recordTlasRefit
        // promotes the refit to one, nothing having been built).
        void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances, bool buildNow = true);

        // Record a per-frame TLAS refit/rebuild into `cb` — NO blocking submit.
        // The host instance write goes to tlasInstancesBuffers[currentFrame]
        // (safe: recordCommandBuffer runs after this slot's fence wait), and the
        // build uses the persistent scratch. Leaves the TLAS handle unchanged so
        // descriptor binding 0 keeps pointing at it. Must be recorded AFTER all
        // deformable BLAS rebuilds in the same cb.
        void recordTlasRefit(VkCommandBuffer cb,
                             const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                             bool fullBuild);

        // Sized for `capacity` descs (at least descs.size()): the slot path
        // writes rows past the structural build's count into the same buffer.
        template<typename DescT>
        void uploadDescBuffer(Buffer& target, const std::vector<DescT>& descs, size_t capacity = 0) {
            const VkDeviceSize bytes = std::max<VkDeviceSize>(
                    std::max(descs.size(), capacity) * sizeof(DescT), sizeof(DescT));
            target = createBuffer(
                    ctx->allocator(), ctx->device(), bytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            if (!descs.empty()) {
                uploadHostVisible(ctx->allocator(), target, descs.data(),
                                  descs.size() * sizeof(DescT));
            }
        }

        // Flush the cached host-side MaterialDescs into this frame's slot of
        // the per-frame ring. Called from renderFrame after the fence wait —
        // the in-flight signal guarantees the previous use of this slot has
        // retired, so the memcpy races nothing. Replaces the old shared-buffer
        // path that called vkDeviceWaitIdle on every animated-pbr update.
        void flushMaterialDescsIfDirty(uint32_t frame);
        // Same fence-gated ring flush for GeometryDescs (auto-LOD level
        // switches repoint indexAddress/indexed — see geomDescsCached_).
        void flushGeometryDescsIfDirty(uint32_t frame);

        // Per-frame frustum cull: tag every entry with `inFrustum` so raster
        // passes (gbuf prepass, overlay depth prepass) can skip off-screen
        // geometry. Each draw on the raster gbuf pass costs ~15 µs of GPU
        // command-processor time regardless of whether anything actually
        // rasterizes — at 1500-mesh scenes that's ~22 ms eaten by draws
        // whose vertex shader transforms all land outside the clip cube.
        //
        // Deformable entries (skinned / displaced / morphed) keep
        // inFrustum=true unconditionally — their local AABB is the rest-
        // pose extent, not the deformed one, so a tight test would clip
        // out poses that bulge past the bind silhouette. Re-fitting a
        // tight bound every frame on the CPU isn't worth the cost; the
        // GPU pays a small constant for these. Overlay entries also stay
        // on (debug viz should always render).
        void cullEntriesAgainstFrustum(Camera& camera);
        // A span's world bounds for the cull, derived when cleared; false when
        // it has none (VulkanCoreScene.cpp).
        bool refreshSpanCullBounds(EntrySpan& sp);
        // The rows a span's cull boxes cover (EntrySpan::chunkAabbs, blockAabbs).
        static constexpr uint32_t kCullChunk = 512;
        static constexpr uint32_t kCullBlock = 64;
        // Every span a cull walk would box, boxed now, ahead of culls and
        // shadow fits that run at once.
        void boxSpansForCull();
        // Whether the current view's cull would walk the spans this frame.
        [[nodiscard]] bool cullWouldWalk(const Matrix4& vp, bool gpuCull) const;
        [[nodiscard]] bool cullWouldWalk(Camera& camera) const;

        // Pull per-mesh PBR material params off the threepp Material chain.
        // Anything not satisfying the relevant interface gets a sensible
        // default so meshes lacking a material still render. albedoTexIndex
        // stays at -1 here — caller patches it after `ensureMaterialTexture`.
        static MaterialDesc materialFromMesh(const Mesh& m);
        // The same desc from a material's typed views (materialViewsOf, the
        // dynamic_casts taken once); a null material gives the defaults.
        static MaterialDesc materialDescOf(const Material* mat, const MaterialViews& v);
        static MaterialViews materialViewsOf(const Material& mat);

        // Live material edits (MaterialState::materialWatch_): read every
        // watched material and needsUpdate() the ones that changed since the
        // last frame, ahead of the scene diffs that act on versions; and, at
        // the end of every build, settle the watch list with the material
        // table's records that gained their first entry or lost their last.
        void pollMaterialEdits(Object3D& scene);
        void watchTableMaterials(Object3D& scene);
        void addMaterialWatch(uint32_t record);
        void removeMaterialWatch(uint32_t watch);
        static void readMaterialWatch(MaterialWatch& w, const Material& mat, bool& changed);

        // three.js parity: InstancedMesh::instanceColor multiplies the
        // material's diffuse color per instance (color_vertex.glsl). Carried
        // in the entry's GeometryDesc, which every consumer (raster G-buffer,
        // traced hits, lidar) reaches per entry, and multiplied into the
        // material record's albedo there (entryMat). (1,1,1) for plain meshes
        // and for instanced meshes that never set a color: a multiply by one
        // leaves the albedo's bits as they are.
        static void setGeomInstanceColor(GeometryDesc& gd, const MeshEntry& en);

        struct LodFadeChange {
            bool words   = false;// some entry's word changed: DrawInfo is stale
            bool casters = false;// some entry's lodFadeCasts flipped
        };
        // Polls every span's fade source (the instanceFade attribute's
        // pointer + version, else the mesh-wide fade) and re-packs the entries
        // of the spans that changed; `all` re-packs every span (structural
        // rebuild). O(spans) on a frame where nothing changed.
        LodFadeChange refreshLodFades(bool all);
        static uint32_t packLodFade(float fade, bool complement);
        // Whether shadows and rays see this entry. Exactly one side of a
        // plain-f / complement-(1-f) pair: plain while its fade is at least
        // 0.5, complement while above 0.5 (the odd step count makes the cut
        // of a pair never sit exactly on the half).
        static bool lodFadeCasts(uint32_t word) {
            if (word == 0u) return true;
            const uint32_t cut = (word & kLodFadeCutMask) >> kLodFadeCutShift;
            return (word & kLodFadeComplement) ? 2u * cut < kLodFadeSteps : 2u * cut > kLodFadeSteps;
        }
        // Whether the dither draws none of the entry's pixels: a plain fade
        // of 0, or a complement whose share is 0 (packLodFade's cut ends).
        static bool lodFadeDrawsNone(uint32_t word) {
            if ((word & kLodFadeActive) == 0u) return false;
            const uint32_t cut = (word & kLodFadeCutMask) >> kLodFadeCutShift;
            return (word & kLodFadeComplement) ? cut >= kLodFadeSteps : cut == 0u;
        }

        static std::shared_ptr<Texture> emissiveTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* em = dynamic_cast<MaterialWithEmissive*>(mat.get())) {
                return em->emissiveMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> occlusionTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* ao = dynamic_cast<MaterialWithAoMap*>(mat.get())) {
                return ao->aoMap;
            }
            return nullptr;
        }

        static void copyTexUvTransform(float (&dst)[9], const std::shared_ptr<Texture>& tex) {
            if (!tex) return;
            if (tex->matrixAutoUpdate) tex->updateMatrix();
            std::copy(tex->matrix.elements.begin(), tex->matrix.elements.end(), dst);
        }

        // Tiled world-anchored detail albedo (MaterialWithDetailMap; terrain
        // etc.). Returns null when absent or strength is zero.
        static std::shared_ptr<Texture> detailTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* dm = dynamic_cast<MaterialWithDetailMap*>(mat.get())) {
                if (dm->detailMap && dm->detailStrength > 0.f) return dm->detailMap;
            }
            return nullptr;
        }

        // Detail NORMAL + ROUGHNESS map (MaterialWithDetailMap). Independent of
        // the detail albedo: a terrain can carry relief/roughness breakup with
        // no albedo layer, or vice-versa. Gated on the map plus a nonzero
        // strength on either the normal or the roughness term.
        static std::shared_ptr<Texture> detailNormalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* dm = dynamic_cast<MaterialWithDetailMap*>(mat.get())) {
                if (dm->detailNormalMap && (dm->detailNormalScale > 0.f || dm->detailRoughStrength > 0.f))
                    return dm->detailNormalMap;
            }
            return nullptr;
        }

        // Terrain splat maps (MaterialWithTerrainMaps). The weight map is the
        // path's on-switch: bound only when a band set exists to select.
        static std::shared_ptr<Texture> terrainWeightTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get())) {
                if (tm->terrainMapsActive()) return tm->terrainWeightMap;
            }
            return nullptr;
        }

        // World-space normal map — independent of the band sets (LOD-seam fix
        // is worthwhile even for a macro-colour-only terrain).
        static std::shared_ptr<Texture> terrainNormalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainNormalMap;
            return nullptr;
        }

        static std::shared_ptr<Texture> terrainBandAlbedoTexOf(const Mesh& m, int band) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainBandAlbedo[static_cast<size_t>(band)];
            return nullptr;
        }

        static std::shared_ptr<Texture> terrainBandNormalTexOf(const Mesh& m, int band) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainBandNormalRough[static_cast<size_t>(band)];
            return nullptr;
        }

        // Walk a material's chain for the albedo (`map`) texture.
        // Returns null if the material doesn't carry one.
        static std::shared_ptr<Texture> albedoTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* mm = dynamic_cast<MaterialWithMap*>(mat.get())) {
                return mm->map;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> roughnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat.get())) {
                return r->roughnessMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> metalnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* mt = dynamic_cast<MaterialWithMetalness*>(mat.get())) {
                return mt->metalnessMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> normalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* nm = dynamic_cast<MaterialWithNormalMap*>(mat.get())) {
                return nm->normalMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> transmissionTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tr = dynamic_cast<MaterialWithTransmission*>(mat.get())) {
                return tr->transmissionMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> clearcoatTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get())) {
                return cc->clearcoatMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> clearcoatRoughnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get())) {
                return cc->clearcoatRoughnessMap;
            }
            return nullptr;
        }

        // The maps a mesh fingerprint compares (MeshFingerprint::tex).
        static MaterialTexSet texSetOf(const Mesh& m) {
            return {albedoTexOf(m), roughnessTexOf(m), metalnessTexOf(m),
                    normalTexOf(m), transmissionTexOf(m), clearcoatTexOf(m),
                    clearcoatRoughnessTexOf(m), emissiveTexOf(m), occlusionTexOf(m)};
        }

        // The material values a mesh fingerprint compares (MeshFingerprint::pbr):
        // albedo, roughness, metalness, emissive + intensity, normalScale,
        // transmission + ior, clearcoat + its roughness.
        static std::array<float, 15> pbrKeyOf(const MaterialDesc& md) {
            return {md.albedo[0], md.albedo[1], md.albedo[2],
                    md.roughness, md.metalness,
                    md.emissive[0], md.emissive[1], md.emissive[2],
                    md.emissiveIntensity,
                    md.normalScale[0], md.normalScale[1],
                    md.transmission, md.ior,
                    md.clearcoat, md.clearcoatRoughness};
        }

        // Walk the scene each frame, fingerprint the meshes, and rebuild the
        // TLAS + per-instance desc tables when anything that affects ray
        // tracing changes (mesh added/removed, transform animated, material
        // slider tweaked). BLAS records persist across rebuilds — keyed by
        // BufferGeometry pointer — so static geometry isn't re-traced.
        //
        // Per-frame cost when nothing changes is: traverse + N MeshFingerprint
        // compares. Cheap. The rare rebuild path waits the GPU idle, retires
        // the TLAS + scene desc buffers, and rewrites bindings 0/3/4 across
        // every descriptor set without re-allocating from the pool.
        void ensureSceneBuilt(Object3D& scene, Camera& camera);

        // ── ensureSceneBuilt's phases (VulkanCoreScene.cpp), in call order ──
        // ensureSceneBuilt is a driver over these; each holds the code its phase
        // ran inside it, unchanged. The locals they share are one SceneBuildFrame
        // (SceneBuildFrame.hpp) the driver owns. A bool phase returns true when the
        // build is finished (updateSceneInPlace) or, for refreshDirtyGeometry, when
        // updateSceneInPlace must fall through to the full rebuild.
        struct SceneBuildFrame;
        void prepareSceneBuild(SceneBuildFrame& sb, Object3D& scene, Camera& camera);
        void matchSceneSnapshot(SceneBuildFrame& sb, Object3D& scene, Camera& camera);
        void refreshSceneEntries(SceneBuildFrame& sb, Object3D& scene, Camera& camera);
        void expandSceneEntries(Object3D& scene, Camera& camera);
        void settleLeanEntries(SceneBuildFrame& sb);
        void selectAutoLodLevels(Camera& camera);
        void captureRayCamera(SceneBuildFrame& sb, Camera& camera);
        void diffLeanEntries(SceneBuildFrame& sb);
        void fingerprintEntries(SceneBuildFrame& sb);
        void scanDeformers(SceneBuildFrame& sb, Camera& camera);
        bool updateSceneInPlace(SceneBuildFrame& sb);
        void compareFingerprints(SceneBuildFrame& sb);
        void pollInstanceColors(SceneBuildFrame& sb);
        void queueDeformerRefits(SceneBuildFrame& sb);
        void refreshTetMeshes(SceneBuildFrame& sb);
        void enqueueVertexInterop(SceneBuildFrame& sb);
        bool refreshDirtyGeometry(SceneBuildFrame& sb);
        void queueSkinnedPrevResyncs(SceneBuildFrame& sb);
        void resyncPrevVertices(SceneBuildFrame& sb);
        void collectInstanceClassChanges(SceneBuildFrame& sb);
        void patchMaterialDescs(SceneBuildFrame& sb);
        void refitTlasInPlace(SceneBuildFrame& sb);
        void rebuildScene(SceneBuildFrame& sb, Camera& camera);
        void pruneDeadSceneResources();
        void admitRebuildEntries(SceneBuildFrame& sb, Camera& camera);
        void buildRebuildTlas(SceneBuildFrame& sb);
        void uploadRebuildDescs(SceneBuildFrame& sb);
        void remapEntryHistory(SceneBuildFrame& sb);
        void resizeEntryBuffers(SceneBuildFrame& sb);

        void createCameraUbos();

        // Grow emissiveTriBuffers[frame] in-place if the current frame's
        // emissive count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 14. 2× headroom
        // matches motionMatBuffers.
        bool ensureEmissiveTriCapacity(uint32_t frame, uint32_t needed);

        // Clears both ReSTIR DI reservoir ping-pong image pairs
        // (reservoirPosImagesPP + reservoirWImagesPP) to (0,0,0,0), so the
        // deferred shade's temporal-reuse read sees M=0 (no prior history)
        // on the next frame instead of garbage. A stale tap that *does* slip
        // past the reservoir's own validity guard (e.g. by undefined memory
        // aliasing a real mesh-ID after image creation) then returns a
        // harmless 0 mean / 0 M reservoir instead of garbage. Caller must
        // hold the GPU idle (we don't issue our own barrier-into-TRANSFER_DST
        // since the only legal layout transition path for an image being
        // cleared is GENERAL → TRANSFER_DST → GENERAL).
        void clearGbufImages();

        // Compute per-instance motion matrices = prevWorld * inverse(curWorld)
        // and upload to motionMatBuffers[frame]. Identity for first-seen
        // entries (cold-start frame after a topology rebuild) so the reproject
        // is a no-op until prevWorldMats picks up real history. Keying by
        // (Mesh*, instanceIndex) so each InstancedMesh sub-instance carries
        // its own motion delta. Caller must have already waited the
        // inFlight[frame] fence — we write a buffer the GPU may have been
        // reading on the previous use of `frame`.
        // Reset the entry-aligned motion state (scratch identity, prev =
        // current worlds). Called at every full rebuild (before the identity
        // remap restores surviving history) and by the accumulation resets.
        void seedMotionState(const std::vector<MeshEntry>& entries);

        void computeAndUploadMotionMatrices(uint32_t frame,
                                            const std::vector<MeshEntry>& entries);

        // ── GPU instance expansion (stage 1) ────────────────────────────────
        // Rebuild the GPU-path span list, grow this slot's pools, and upload
        // the SpanDescs + every span whose instance matrices the CPU re-read
        // since this slot last saw them. Caller must have already waited the
        // inFlight[frame] fence: both pools and the frame's descriptor set are
        // written here, and a descriptor written while its frame is in flight
        // is the VUID-03047 zone.
        void prepareInstanceExpansion(uint32_t frame);

        // ── ParticleField (phase 0) ─────────────────────────────────────────
        // Grow this frame's per-field position ring, copy each field's host
        // staging into THIS frame's ring slot (version-gated), and rewrite the
        // FieldDesc SSBO. Same window and same reason as
        // prepareInstanceExpansion: post-fence, pre-record.
        void prepareParticleFields(uint32_t frame);
        // Bring up the ParticleField pass if it does not exist yet. Called from
        // ensureHybridResources on the ordinary path and from
        // enableParticleFieldInterop, which must work before the first frame.
        void ensureParticleFieldPass();

        // Publish each field's device-side live count into its draw command.
        // Head of the frame command buffer, before any consumer reads it.
        void recordParticleFieldCounts(VkCommandBuffer cb);

        // ── ParticleField device emitter (F2) ───────────────────────────────
        // One dispatch per Ownership::Renderer field, at the head of the frame
        // command buffer and before the density scatter, the counts copy and
        // every view's raster pass — all of which read the positions it writes.
        // Bracketed by TP_ParticleEmit for timing. No-op without a
        // Renderer-owned field.
        void recordParticleFieldEmit(VkCommandBuffer cb, uint32_t frame);

        // ── ParticleField density (phase 2) ─────────────────────────────────
        // Clear + splat every density field into its world-anchored volume,
        // once for all views. Bracketed by TP_ParticleDensity for timing.
        // No-op without a density field.
        void recordParticleDensityScatter(VkCommandBuffer cb, uint32_t frame);
        // Per-FIF ParticleDensityUbo (bindings 68) + the 1x1x1 dummy volume.
        // Both created once and never resized.
        void ensureParticleDensityResources();
        // Rewrite the frame's ParticleDensityUbo from the pass's current volume
        // list. Post-fence, pre-record; also refreshes the deferred descriptor
        // sets when the volume LIST changed (generation bump).
        void updateParticleDensityUbo(uint32_t frame);

        // ── Splat reflection volumes (plans/splat-volume-reflections.md) ────
        // Per-FIF SplatVolumeUbo (binding 71) + the 1x1x1 dummy volume for the
        // unused slots of binding 70. Both created once and never resized —
        // the two functions above, mirrored.
        void ensureSplatVolumeResources();
        // The eight image views binding 70 names: live volumes first, the
        // 1x1x1 dummy for the rest. ONE producer, used by both the descriptor
        // write and the staleness key below, so the two cannot disagree about
        // what "the current list" is.
        void splatVolumeBindViews(std::array<VkImageView, vulkan::kMaxSplatVolumes>& out);
        // Identifies that exact list. Keyed on the view handles, not on
        // SplatPass::volumeGeneration() alone: the generation bumps when a
        // volume is baked or freed, and the free happens inside syncClouds —
        // by which time a descriptor set that still named the view has already
        // been recorded into a command buffer that may still be executing
        // (VUID-vkDestroyImageView-imageView-01026). The handle list changes
        // the moment a cloud stops being visible, which is framesInFlight+1
        // syncs before retireStale destroys anything — the margin retireStale
        // assumes every consumer has. The generation is mixed in anyway, so a
        // rebake that happens to be handed the same handle back still counts
        // as a change. The cost is a rewrite whenever a cloud is shown or
        // hidden; the density table accepts the same churn.
        [[nodiscard]] uint64_t splatVolumeBindKey();
        // Rewrite the frame's SplatVolumeUbo from SplatPass::volumeEntries():
        // world→UVW and the conservative world AABB composed on the HOST from
        // each cloud's model matrix and local bake box. Post-fence, pre-record,
        // and AFTER collectSplatClouds' syncClouds (so a cloud uploaded this
        // frame is baked before the UBO names it); also refreshes the deferred
        // descriptor sets when the volume LIST changed (generation bump).
        void updateSplatVolumeUbo(uint32_t frame);

        // One vkCmdDrawIndirect per visible ParticleField, inside the G-buffer
        // render pass and after every ordinary bucket. Separate pipeline, and
        // deliberately so: see particlefield_gbuf.vert's header.
        void recordParticleFieldDraws(VkCommandBuffer cb, bool useMsaa);

        // Upload + BLAS-build a geometry that is nobody's Mesh geometry (the
        // ParticleField MeshRepr proxy). Same blasCache, same liveCheck-based
        // eviction as every ordinary static geometry.
        const BlasRecord* ensureCachedBlas(const std::shared_ptr<BufferGeometry>& geom) {
            if (!geom) return nullptr;
            auto it = blasCache.find(geom.get());
            if (it != blasCache.end()) return it->second.get();
            auto rec = buildBlasFor(*geom, /*allowPacked=*/true, /*updatable=*/false);
            if (!rec) return nullptr;
            rec->liveCheck = geom;
            return blasCache.emplace(geom.get(), std::move(rec)).first->second.get();
        }

        // One dispatch, into the frame's already-open command buffer.
        void recordInstanceExpansion(VkCommandBuffer cb, uint32_t frame);

        // Copy the GPU world matrices back and compare them against
        // MeshEntry::worldMatrix over the instanced spans. DRAINS THE DEVICE;
        // backs VulkanRenderer::instanceExpandCheck.
        bool verifyInstanceExpansion(VulkanRenderer::InstanceExpandCheck& out);

        // Upload meshMovedBits_ to meshMovedBitsBuffers[frame]. Caller must have
        // already waited the inFlight[frame] fence.
        void uploadMeshMovedBits(uint32_t frame);

        // Walk visible entries, gather emissive triangles in world space, and
        // upload to emissiveTriBuffers[frame]. Per-tri 64-byte record:
        //   v0.xyz = world pos0,    v0.w = triangle area
        //   v1.xyz = world pos1,    v1.w = running cumPower (CDF)
        //   v2.xyz = world pos2,    v2.w = per-tri power (lum * area)
        //   emission.xyz = emissive*intensity, emission.w = unused
        // then a 64-byte header (v0.x = emissive-instance count, v0.y = the
        // range gate's exposure term) and, under kEmissiveGateMaxLights, one
        // 64-byte record per emissive instance for the shader's range gate
        // and coverage mode (emissive_lights.glsl).
        //
        // Uniform-by-area within each tri × power-weighted picking across tris
        // gives a constant area-weighted-luminance pdf for closest_hit's NEE.
        //
        // Returns true when the buffer handle changed (capacity grew); caller
        // must then rewrite descriptor binding 14 for this frame's sets.
        bool buildAndUploadEmissiveTris(uint32_t frame,
                                        const std::vector<MeshEntry>& entries);

        // Grow motionMatBuffers[frame] in-place if the current scene's
        // instance count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 10. We grow with
        // 2× headroom to avoid thrashing on incremental scene growth.
        bool ensureMotionMatCapacity(uint32_t frame, uint32_t needed);

        // Same dance as ensureMotionMatCapacity, but for the per-mesh
        // moved-bitmask SSBO at binding 21. `neededWords` is the number of
        // 32-bit words required to address every visible TLAS instance.
        bool ensureMeshMovedBitsCapacity(uint32_t frame, uint32_t neededWords);



        // REVERSED-Z Vulkan projection from a GL-convention (NDC z∈[-1,1]) proj.
        // threepp's projection is GL-style; fed to Vulkan as-is it (a) clips the
        // [-1,0] near-slab so HALF the depth precision is wasted on never-visible
        // geometry, and (b) bunches precision near the camera → distant z-fighting.
        // Reverse-Z maps near→1, far→0 in Vulkan [0,1] → reclaims the lost half AND
        // gives near-uniform precision (with D32F, kills distance z-fighting). The
        // new z-output row = 0.5*w_row - 0.5*z_row. Pair with depth clear 0.0 +
        // GREATER compare. Shaders are convention-agnostic (use the uploaded
        // matrices), so this is host-side only.
        static Matrix4 reverseZVk(const Matrix4& glProj) {
            Matrix4 p = glProj;
            auto& e = p.elements;// column-major: z-output = row 2, w-output = row 3
            for (int c = 0; c < 4; ++c)
                e[c * 4 + 2] = 0.5f * e[c * 4 + 3] - 0.5f * e[c * 4 + 2];
            return p;
        }

        void updateCameraUbo(uint32_t frame, Camera& camera);

        // This frame's sub-pixel jitter (texels, [-0.5, +0.5] per axis) and the
        // gate that decides whether the camera UBOs apply it. updateCameraUbo
        // and uploadRasterCameraUbo each call both, so the deferred and raster
        // cameras jitter identically (VulkanCoreUploads.cpp).
        void cameraJitter(VkExtent2D ext, Camera& camera, float& jx, float& jy);
        [[nodiscard]] bool cameraJitterOn() const;

        // ── Hybrid raster runtime helpers ───────────────────────────────────
        // (halton_ is also used by cameraJitter above, which mirrors the
        // raster's jitter into the deferred shade's camera UBO — keep both
        // helpers reachable.)
        // Halton(2,3) sub-pixel jitter for raster TAA. (1, base) skips the
        // zero entry so frame 0 already gets a non-zero offset.
        static float halton_(uint32_t i, uint32_t base) {
            float f = 1.f, r = 0.f;
            while (i > 0u) {
                f /= float(base);
                r += f * float(i % base);
                i /= base;
            }
            return r;
        }

        // FSR2-style jitter phase count: the Halton(2,3) sequence length must
        // scale with the upscale ratio so the sub-pixel samples fully cover the
        // higher-res OUTPUT grid — 8 at native, 8·ratio² when upsampling (≈18 at
        // 1.5×, 32 at 2×, 72 at 3×). A fixed period under-samples the grid the
        // moment renderScale < 1, leaving residual aliasing the temporal resolve
        // can't reconstruct. Mirrors ffxFsr2GetJitterPhaseCount. ratio = max over
        // both axes (they're equal under uniform renderScale); clamped to [8,128].
        static uint32_t jitterPhaseCount_(VkExtent2D render, VkExtent2D display) {
            const float rx = render.width  > 0u ? float(display.width)  / float(render.width)  : 1.f;
            const float ry = render.height > 0u ? float(display.height) / float(render.height) : 1.f;
            const float ratio = std::max(rx, ry);
            const uint32_t n = static_cast<uint32_t>(8.f * ratio * ratio + 0.5f);
            return std::clamp(n, 8u, 128u);
        }

        // Cache per-entry cull mode from each entry's material record.
        // Called wherever the table is patched so the gbuffer draw loop can
        // pick BACK-cull (Front, default fast path), FRONT-cull (Back), or
        // NONE-cull (Double).
        //
        // Side::Front + transmissive materials with proper outward winding
        // (ocean, most glass viewed from outside) render correctly under
        // BACK culling. Camera-inside-glass cases (windshield from cabin)
        // are an artist content concern: mark the glass as Side::Double if
        // you need interior viewing — chit's BVH path doesn't cull, but the
        // raster prepass does, so unmarked single-sided glass shows the
        // surface BEHIND it in the gbuffer.
        void cacheCullFlags();

        // Resolve the BlasRecord backing a given visible entry. The same
        // physical buffers feed BLAS and the raster prepass (VERTEX_BUFFER_BIT
        // was added at allocation), so this is a pure lookup, no upload.
        // Branches off the cached type flags so we don't dynamic_cast every
        // entry on every raster draw call.
        const BlasRecord* resolveBlasForEntry(const MeshEntry& en) const {
            if (en.isParticleField) {
                // A field's raster draw pulls the MeshRepr PROXY, not the
                // zero-area placeholder the field carries as its Mesh geometry
                // (the placeholder is what keeps the field's own BLAS/TLAS
                // instance harmless, and what keeps it valid on GL). Nothing
                // else names the proxy, so ensureCachedBlas uploads it during
                // scene expansion and this is the read side.
                const auto& repr = static_cast<const ParticleField*>(en.mesh)->meshRepr();
                if (!repr.enabled || !repr.geometry) return nullptr;
                auto it = blasCache.find(repr.geometry.get());
                return it != blasCache.end() ? it->second.get() : nullptr;
            }
            if (en.isSkinned) {
                auto* sm = static_cast<SkinnedMesh*>(en.mesh);
                auto it = skinnedMeshStates.find(sm);
                if (it != skinnedMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isDisplaced) {
                auto* dm = static_cast<DisplacedMesh*>(en.mesh);
                auto it = displacedStates.find(dm);
                if (it != displacedStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isGrass) {
                auto* gm = static_cast<GrassMesh*>(en.mesh);
                auto it = grassStates.find(gm);
                if (it != grassStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isTet) {
                auto it = tetMeshStates.find(en.mesh);
                if (it != tetMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isMorphed) {
                auto it = morphedMeshStates.find(en.mesh);
                if (it != morphedMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
            }
            auto* geom = en.mesh->geometry().get();
            auto it = blasCache.find(geom);
            if (it != blasCache.end()) return it->second.get();
            return nullptr;
        }

        // Per-frame raster camera UBO upload + descriptor set rewrite.
        // Must run AFTER ensureSceneBuilt (motionMatBuffers[frame] populated)
        // and AFTER the inFlight fence wait (safe to write a slot the GPU
        // was reading on the previous use of `frame`).
        void uploadRasterCameraUbo(uint32_t frame, Camera& camera);

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using DrawInfoGpu = vulkan::impl::DrawInfoGpu;

        uint32_t stableIdForObject(const Object3D& o) {
            // Both id maps are empty unless the app opts in (setInstanceId /
            // setClassId), which is the common case — skip the probe entirely.
            if (!instanceIdOverride_.empty()) {
                if (const auto it = instanceIdOverride_.find(o.id); it != instanceIdOverride_.end()) {
                    return it->second;
                }
            }
            if (view().prepIdsReadOnly) {
                // A view preparing beside the others: numbering an object here
                // would hand out the next id in whichever order the views got
                // here. Its build is redone on the frame thread instead.
                const auto it = autoStableIds_.find(o.id);
                if (it != autoStableIds_.end()) return it->second;
                view().prepIdMiss = true;
                return 0u;
            }
            const auto [it, inserted] = autoStableIds_.try_emplace(o.id, 0u);
            if (inserted) {
                it->second = nextAutoStableId_;
                if (nextAutoStableId_ < kMaxStableId) ++nextAutoStableId_;// saturate
            }
            return it->second;
        }
        // Read-only sibling of stableIdForObject: same answer, but never
        // ASSIGNS an auto id. The lidar's entry->id table (entryStableIds_) is
        // filled from the TLAS build loop, which runs before the indirect draw
        // builder in a frame; calling the assigning form there would renumber
        // the auto ids into TLAS-entry order and silently change what the
        // raster Ids AOV reports for unlabelled objects. So the assignment
        // stays where it has always been and this only reads the result.
        //
        // 0 is returned for an object that has no user id AND has not been
        // auto-numbered yet — which is the documented meaning of 0 (sky /
        // unassigned), not a new sentinel. It self-heals after one frame for
        // anything the draw builder visits.
        [[nodiscard]] uint32_t stableIdIfAssigned(const Object3D& o) const {
            if (!instanceIdOverride_.empty()) {
                if (const auto it = instanceIdOverride_.find(o.id); it != instanceIdOverride_.end()) {
                    return it->second;
                }
            }
            const auto it = autoStableIds_.find(o.id);
            return it == autoStableIds_.end() ? 0u : it->second;
        }
        uint16_t classIdForObject(const Object3D& o) const {
            if (classIds_.empty()) return 0;
            const auto it = classIds_.find(o.id);
            return it == classIds_.end() ? uint16_t(0) : it->second;
        }

        uint32_t occlCullBitFor(const MeshEntry& en) {
            const uint32_t need = en.instanceIndex + 1u;
            if (en.mesh->id == occlBitMemoMeshId_ && need <= occlBitMemoRange_.capacity)
                return occlBitMemoRange_.base + en.instanceIndex;
            auto [it, inserted] = occlBitRanges_.try_emplace(en.mesh->id, OcclBitRange{0u, 0u});
            if (inserted || it->second.capacity < need) {
                uint32_t cap = need;
                if (en.isInstanced)// flag guarantees the static type — no dynamic_cast per record
                    cap = std::max(cap, static_cast<uint32_t>(
                                                static_cast<const InstancedMesh*>(en.mesh)->count()));
                it->second = {occlBitDomain_, cap};
                occlBitDomain_ += cap;
            }
            occlBitMemoMeshId_ = en.mesh->id;
            occlBitMemoRange_  = it->second;
            return it->second.base + en.instanceIndex;
        }

        // Definition moved to VulkanSceneTypes.hpp.
        using DrawGroup = vulkan::impl::DrawGroup;
        // [0] Front (BACK cull), [1] Back (FRONT cull), [2] Double (NONE cull),
        // [3] blend decals (NONE cull, drawn LAST with rasterGbufDecalPipeline
        // so their albedo lerps over the already-rasterized receivers).
        // The static order means recordRasterGbufPass can issue at most 4
        // vkCmdDrawIndirect calls and skip the empty ones. The groups and the
        // total of the view being built or recorded: prepOut_.indirectGroups
        // and .indirectTotalDraws.

        // Definition moved to VulkanViewContext.hpp.
        using SpanDrawPlan   = vulkan::impl::SpanDrawPlan;
        using SpanEmit       = vulkan::impl::SpanEmit;
        using SpanRun        = vulkan::impl::SpanRun;
        using SpanRecordSlot = vulkan::impl::SpanRecordSlot;
        using SpanRecordPool = vulkan::impl::SpanRecordPool;
        static constexpr uint32_t kSpanStampBlockShift = vulkan::impl::kSpanStampBlockShift;
        static constexpr uint32_t kSpanStampBlock      = vulkan::impl::kSpanStampBlock;
        // Spans this short take the per-entry path (their entries are few).
        static constexpr uint32_t kSpanDrawCacheMinRows = 64;

        bool ensureDrawInfoCapacity(uint32_t frame, VkDeviceSize neededBytes);

        bool ensureIndirectCmdCapacity(uint32_t frame, VkDeviceSize neededBytes);

        // Build the per-frame DrawInfo + indirect-command buffers for the
        // hybrid raster G-buffer pass. Called from renderFrame right after
        // cullEntriesAgainstFrustum (which sets MeshEntry::inFrustum) so we
        // can skip culled draws here. The actual GPU dispatch happens in
        // recordRasterGbufPass via 1-4 vkCmdDrawIndirect calls partitioned
        // by cull mode (+ a trailing blend-decal bucket).
        //
        // Draw partitioning: walk entries once, sort each visible draw
        // into one of four buckets (Side::Front/Back/Double, then blend
        // decals). Buckets are concatenated [Front | Back | Double | Decal]
        // in the device buffers, and each VkDrawIndirectCommand's firstInstance carries the
        // global DrawInfo index — surfaced to the VS as gl_InstanceIndex
        // so the shader fetches `draws[gl_InstanceIndex]`. This trick
        // sidesteps the gl_DrawIDARB-resets-per-call issue without
        // needing dynamic-offset descriptors or per-call push constants.
        void buildIndirectDrawData(uint32_t frame);

        // Begin the raster G-buffer render pass and ship the prebuilt
        // indirect-draw groups via 1-4 vkCmdDrawIndirect calls (one per
        // active cull mode, plus a trailing blend-decal group on the
        // albedo-blend pipeline). Replaces the prior per-mesh draw loop —
        // see buildIndirectDrawData above for how the GPU buffers are
        // populated.
        void recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame);

        // Shared body: `renderPass` must be COMPATIBLE with the pipelines'
        // creation pass (the occlusion-culling load/store variants are), and
        // `indirectBuffer` supplies the VkDrawIndirectCommand records the
        // bucket groups index into (the two-phase path swaps in the compute-
        // written phase buffers; offsets/counts are identical by design).
        // `particles`: issue the ParticleField draws in THIS pass. The
        // two-phase occlusion path calls this twice over the same attachments,
        // and a field is always-draw, so without the flag every particle would
        // rasterize twice. Phase A gets them, so the HiZ built from its depth
        // already sees the bed.
        void recordRasterGbufPassInternal(VkCommandBuffer cb, uint32_t frame,
                                          VkRenderPass renderPass, VkFramebuffer fb,
                                          bool useMsaa, VkBuffer indirectBuffer,
                                          bool clear, bool particles = true);

        // Lazily create the debug_resolve compute pipeline + descriptor set.
        void createDebugResolvePipeline();

        // Debug visualization: sample one G-buffer channel and write a
        // readable RGB image straight to the swapchain via debug_resolve.comp.
        // Replaces the old raw vkCmdBlitImage, which could only show the
        // world-normal attachment: a blit cannot bias the SIGNED motion vector
        // (it clamped to near-black) and is INVALID from the integer ids
        // attachment (R32G32B32A32_UINT) into the UNORM swapchain (the Vulkan
        // spec forbids integer<->non-integer blits). The gbuf attachments are
        // in SHADER_READ_ONLY_OPTIMAL here (the gbuffer render pass declares a
        // COMPUTE consumer dependency, same as the deferred / event-shade
        // consumers), so no gbuf barrier is needed — we only move the
        // freshly-acquired swapchain image to GENERAL for the storage write
        // and leave it there, the exact exit contract the full deferred path
        // uses so endFrame()'s shared overlay/present finalize is unchanged.
        // A frame-graph pass: the set is written at build time and declared
        // from the reflected shader (the swapchain comes from
        // importSwapchain).
        void addHybridDebugResolvePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex, uint32_t frame);


        void createLightsUbos();

        // Walk the scene each frame for AmbientLight + DirectionalLight; pack
        // into the per-frame lights UBO. Direction is computed from the
        // light's world-space position toward its (possibly defaulted) target,
        // mirroring three.js's DirectionalLight.target convention. The shader
        // expects the L vector (toward the light), so we negate.
        void updateLightsUbo(uint32_t frame, Object3D& scene, Camera& camera);

        void createFogUbos();

        // Pack scene.fog (Fog/FogExp2 variant) into the per-frame fog UBO.
        // FogExp2.density maps directly
        // to sigma_t; linear Fog reaches ~63% extinction at farPlane via
        // sigma = 1 / (far - near). Hash detect changes so the per-pixel motion
        // path halves FC and the new fog state converges quickly.
        void updateFogUbo(uint32_t frame, Object3D& scene, Camera& camera);

        void createCloudUbos();
        // Pack the setClouds state into the per-frame cloud UBO (binding 58 of
        // the deferred shade set). Disabled → enabled=0 → the march is a no-op.
        void updateCloudUbo(uint32_t frame);

        // Allocate, transition, and upload an Image2D from a tightly-
        // packed CPU buffer. Pixel layout matches `format`. Caller owns the
        // returned Image2D and must call destroyImage2D() on shutdown.
        // One-shot variant: submits + waits — the queue drain is fine at
        // load time but a measurable hitch mid-frame.
        Image2D createSampledImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     const void* pixels, VkDeviceSize byteSize,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr) {
            VkCommandBuffer cb = beginOneShot();
            Buffer staging{};
            Image2D out = buildSampledImage2D(cb, w, h, format, pixels, byteSize,
                                              filter, addrU, addrV, debugName, staging);
            endAndSubmitOneShot(cb);
            // Deferred to the batch flush when a batch is open (staging is still
            // referenced by the not-yet-submitted shared cb); immediate freeing
            // otherwise. The image view/sampler are host-side and valid at
            // once; only the pixel content lands when the batch submit runs,
            // which happens before any shader samples it.
            destroyBufferMaybeBatched(staging);
            return out;
        }

        // Block-compressed sampled image from PRE-ENCODED mip levels (level 0
        // first, every level present down to 1x1). BC images cannot be blit-
        // downsampled, so unlike buildSampledImage2D there is no GPU mip-gen:
        // the caller encodes each level (bcn::buildMipChainRGBA8Area +
        // bcn::bc7EncodeMode6) and this uploads the chain verbatim — one
        // staging buffer, one copy region per level. View/sampler setup
        // matches buildSampledImage2D (aniso + full LOD range). Nothing here
        // is specific to block formats: an uncompressed material texture
        // whose size is not a power of two comes this way too, with RGBA8
        // levels, because the blit chain is only exact on even levels.
        Image2D createSampledImageBC(uint32_t w, uint32_t h, VkFormat format,
                                     const std::vector<std::vector<std::uint8_t>>& levels,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr);

        // In-frame upload of CPU-built levels (level 0 first; one level is an
        // unmipped image): records it into the CALLER's command buffer (must
        // be outside a render-pass instance) and retires the staging buffer
        // through the frame-serial queue — no submit, no queue drain. A
        // mid-record one-shot's vkQueueWaitIdle blocks on every in-flight
        // frame (~2 frames of GPU time); HUD TextSprites re-rasterizing their
        // atlas per ammo-counter change made that a 40-50 ms hitch on every
        // shot. Same-queue execution order + the final SHADER_READ_ONLY
        // barrier make samples in this and later frames safe without any host
        // wait. The caller builds the mips because the sprite atlases that
        // come this way are odd-sized, where buildSampledImage2D's blit chain
        // skips rows (bcn::buildMipChainRGBA8Area).
        Image2D createSampledImageLevelsInFrame(VkCommandBuffer cb,
                                                uint32_t w, uint32_t h, VkFormat format,
                                                const std::vector<std::vector<std::uint8_t>>& levels,
                                                VkFilter magFilter, VkFilter minFilter,
                                                VkSamplerMipmapMode mipmapMode,
                                                VkSamplerAddressMode addrU,
                                                VkSamplerAddressMode addrV,
                                                const char* debugName = nullptr) {
            Buffer staging{};
            Image2D out = buildSampledImageLevels(cb, w, h, format, levels,
                                                  magFilter, minFilter, mipmapMode,
                                                  addrU, addrV, debugName, staging);
            retire(std::move(staging));
            return out;
        }

        // Shared body of createSampledImageBC and the in-frame variant above:
        // image + view + sampler creation and the recorded upload of every
        // level (one staging buffer, one copy region per level, final
        // transition) into `cb`. `stagingOut` is still referenced by cb when
        // this returns — the caller covers its lifetime (drain+free for the
        // one-shot, frame-serial retire for the in-frame path).
        Image2D buildSampledImageLevels(VkCommandBuffer cb,
                                        uint32_t w, uint32_t h, VkFormat format,
                                        const std::vector<std::vector<std::uint8_t>>& levels,
                                        VkFilter magFilter, VkFilter minFilter,
                                        VkSamplerMipmapMode mipmapMode,
                                        VkSamplerAddressMode addrU,
                                        VkSamplerAddressMode addrV,
                                        const char* debugName,
                                        Buffer& stagingOut);

        // Shared body of createSampledImage2D: image + view + sampler
        // creation and the recorded upload (staging copy + mip-blit chain +
        // final transition) into `cb`. `stagingOut` is still referenced by
        // cb when this returns — the caller covers its lifetime (drain+free).
        Image2D buildSampledImage2D(VkCommandBuffer cb,
                                    uint32_t w, uint32_t h, VkFormat format,
                                    const void* pixels, VkDeviceSize byteSize,
                                    VkFilter filter, VkSamplerAddressMode addrU,
                                    VkSamplerAddressMode addrV,
                                    const char* debugName,
                                    Buffer& stagingOut);

        // ── ParticleSystem billboard resources ──────────────────────────────

        // Lazily build the 1×1 white texel bound for untextured particle
        // systems (matches the GL path, where an unset `tex` sampler reads
        // white). Created once; freed in deinit.
        void ensureParticleWhiteTexture();

        // Upload/refresh a particle texture, keyed on the raw Texture* (the
        // ShaderMaterial uniform holds no shared_ptr). Returns the cached
        // Image2D, or nullptr (caller falls back to the white default). Mirrors
        // OverlayPass::ensureSpriteAtlasTexture, with the texture's dispose
        // event in place of its weak_ptr liveCheck (see ParticleTexRec).
        const Image2D* ensureParticleTexture(const Texture* tex);
        // Retires the images of particle textures that have been disposed of
        // (destroyed, or dispose() called), once per frame.
        void sweepParticleTextures();

        void destroyParticleGeomRec(ParticleGeomRec& rec);

        // Ensure the per-geometry particle vertex/index buffers exist and the
        // animated attributes (position/normal/color) are current. uv + index
        // are static (uploaded once). Returns nullptr on malformed geometry.
        // Float-typed reads on purpose: the animated attributes re-upload on
        // every version bump, so narrow (compressAttributes) storage would
        // re-widen per update — particle geometry must stay float.
        ParticleGeomRec* ensureParticleGeom(const std::shared_ptr<BufferGeometry>& geomSp);

        // Storage-image (GENERAL layout) factory used for the ReSTIR DI
        // reservoir ping-pong images (createReservoirImages). No staging
        // upload — contents are initialised the first frame after
        // sampleIndex resets to 0. The deferred shade reads/writes these
        // every frame, so we transition once at creation and keep them in
        // GENERAL forever after.
        Image2D createStorageImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     VkImageUsageFlags extraUsage = 0,
                                     const char* debugName = nullptr);

        void createReservoirImages();


        // Manual accumulation reset. Mirrors the post-create reset block above:
        // wipes gbuf + accum + ReSTIR DI reservoirs, rewinds sampleIndex, and
        // invalidates reproject state so the next frame cold-starts from
        // sample 1. Issues a vkDeviceWaitIdle since clearGbufImages requires
        // the GPU idle before its TRANSFER_DST layout transition. When
        // called mid-frame (frameState_ != Idle) the GPU-side clear is
        // deferred to the next beginDeferredFrame; CPU-side counters reset
        // immediately so the user-visible behaviour matches.
        void resetAccumulation();

        // Path-traced LIDAR scan entry-point. Re-uses the main TLAS and the
        // last submitted frame's geom/mat descriptors via a private RT pipeline
        // owned by `lidar_`. Synchronous: blocks the calling thread until
        // per-beam results land in `outResults`.
        // `cleanResults`, when non-null AND params.pairedCleanTrace is set,
        // receives the second (particle-dust-free) leg at the same layout.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params,
                       std::vector<LidarReturn>* cleanResults = nullptr);

        // The same scan, pipelined: fire on one frame, take delivery on a
        // later one. Blocking readback costs every frame already queued behind
        // the fence (~28 ms at two frames in flight), which is a hitch a 10 Hz
        // sensor would inflict ten times a second — see LidarScanner.
        int scanLidarBegin(const std::vector<LidarBeam>& beams, const LidarParams& params);
        [[nodiscard]] bool scanLidarReady(int handle) const;
        bool scanLidarCollect(int handle, std::vector<LidarReturn>& outResults,
                              std::vector<LidarReturn>* cleanResults = nullptr);

        // ── Hybrid raster G-buffer prepass implementation ───────────────────
        // Lazy-initialized on first render().
        // All resources owned by Impl; cleanup in dtor + destroyRasterGbufImages
        // is also called on swapchain resize.

        Image2D createAttachmentImage2D(uint32_t w, uint32_t h, VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkImageAspectFlags aspect,
                                        const char* debugName = nullptr,
                                        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                                        vulkan::TransientPool* pool = nullptr, vulkan::TransientSpan poolSpan = 0,
                                        uint32_t poolSlots = vulkan::kTransientAllSlots);

        // 3D storage image (froxel volumetrics) — the volume sibling of
        // createAttachmentImage2D (OPTIMAL tiling, single mip, 3D view).
        Image2D createImage3D(uint32_t w, uint32_t h, uint32_t depth, VkFormat format,
                              VkImageUsageFlags usage, const char* debugName = nullptr,
                              vulkan::TransientPool* pool = nullptr, vulkan::TransientSpan poolSpan = 0,
                              uint32_t poolSlots = vulkan::kTransientAllSlots);

        void destroyRasterGbufImages();

        // Every MSAA-only render pass / pipeline the hybrid raster path owns
        // (the MS *images* live in rasterGbufs and go through
        // destroyRasterGbufImages instead). ONE owner for the whole set, called
        // both when MSAA is switched off at runtime and from ~Impl, so a future
        // teardown path cannot silently forget half of it — the destructor used
        // to miss the four rasterGbuf*MS handles entirely. Handles are nulled,
        // so calling it twice is safe. Caller must guarantee the device is idle.
        void destroyRasterGbufMsObjects();

        void createRasterGbufRenderPass();

        // Two-phase occlusion-culling variants of the G-buffer pass (1× and
        // MSAA flavours). Same attachments/formats/samples as the source
        // pass → render-pass COMPATIBLE, so the framebuffer and graphics
        // pipelines are shared; only load/store ops and layouts differ:
        //   A: CLEAR + STORE everything; colors end COLOR_ATTACHMENT (pass B
        //      loads them), depth ends DEPTH_STENCIL_READ_ONLY (the farthest
        //      HiZ + cull compute sample it between the passes — under MSAA
        //      the pyramid reduces the raw MS attachment's samples directly).
        //   B: LOAD everything; final layouts identical to the single pass,
        //      so every downstream consumer (incl. gbuf_resolve at MSAA) is
        //      untouched.
        void createOcclRenderPasses(VkSampleCountFlagBits samples,
                                    VkRenderPass& outA, VkRenderPass& outB);

        // MSAA sibling of createRasterGbufRenderPass — mirrors the 1× pass
        // exactly except attachments[*].samples. Consumed only by
        // gbuf_resolve.comp (COMPUTE reads all live samples via texelFetch);
        // nothing else ever touches the MS attachments, so dstStageMask only
        // needs COMPUTE_SHADER (no ray-query stage reads the MS attachments
        // directly — only gbuf_resolve.comp does).
        void createRasterGbufRenderPassMS(VkSampleCountFlagBits samples);

        void createRasterGbufImages(uint32_t w, uint32_t h);
        // DLSS Ray Reconstruction's guide images for this view (RasterGbufImages
        // rr*): the render extent while the view feeds RR, 1x1 otherwise.
        // Returns true when it (re)created them, so the caller rewrites the
        // shade's descriptors. Idles nothing itself: the caller does.
        bool ensureRrGuides();

        void createRasterCameraUbos();

        void createRasterDsLayoutAndPool();

        void createRasterGbufPipeline();

        // MSAA sibling of createRasterGbufPipeline — same shaders/vertex
        // layout/descriptor layout/push-constant range (reuses
        // rasterPipelineLayout + rasterDsLayout, already created by the 1×
        // path in ensureHybridResources), only ms.rasterizationSamples and
        // the target render pass differ. Kept as a full duplicate rather
        // than parametrizing createRasterGbufPipeline so the proven 1× path
        // is never touched (msaa=1 byte-identical guarantee).
        void createRasterGbufPipelineMS(VkSampleCountFlagBits samples);

        // ── Hybrid raster overlay pipeline (wireframe variant) ──────────────
        // Dynamic-rendering pipeline targeting the swapchain (B8G8R8A8_UNORM)
        // + the existing G-buffer depth (D32_SFLOAT, read-only). Pushes
        // mat4 mvp + vec4 color (80B). Triangle topology + polygon-mode
        // line draws each visible mesh as a wireframe; the host gates
        // per-draw on material.wireframe / overlayLayer membership.
        // Line/LineSegments get their own pipeline variants + cached
        // vertex buffer, also built below.
        void createOverlayPipeline();

        // ParticleSystem billboard pipelines. Two variants (alpha-blended /
        // additive) that differ only in blend + depth-test state. Both: 4 vertex
        // bindings (pos/normal/uv/color), a combined-image-sampler set 0, a 128B
        // push constant (modelView + proj), dynamic viewport/scissor, dynamic
        // rendering onto the swapchain + unjitDepth (read-only). Modeled on
        // createOverlayPipeline + OverlayPass::createSpriteOverlayPipeline.
        void createParticlePipeline();

        // World-space Sprite billboard pipeline. Perspective billboard
        // (sprite3d.vert) + the shared overlay_sprite.frag, alpha-blended and
        // depth-tested against unjitDepth (occluded by scene geometry,
        // depth-write off). Reuses particlePipelineLayout_ (128B SpritePC + set-0
        // sampler) and builds the shared static quad. Called from
        // createParticlePipeline so the shared layout already exists.
        void createSpriteWorldPipeline();

        // ParticleField billboard pipeline (F-D). Vertex-less: ZERO vertex
        // bindings, a triangle strip of 4, one indirect draw per field with the
        // instance count coming off the device. Additive, depth-TESTED against
        // the same read-only unjittered depth the legacy path uses (which the
        // legacy ADDITIVE variant deliberately does not do — a fireball is
        // meant to draw over the scene, a rain streak behind a wall is not).
        // Called from createParticlePipeline, so the shared texture set layout
        // and the 1x1 white default already exist.
        void createFieldBillboardPipeline();
        // F4: the fullscreen additive draw that folds the billboard glow
        // pyramid into the swapchain inside the overlay pass. Needs
        // billboardGlow_'s set layout, so it is created with it.
        void createFieldGlowCompositePipeline();

        // Draw every visible field's billboard quads into the render pass the
        // caller has already begun. Used by BOTH the primary's post-TAA overlay
        // pass and a secondary view's own composite — a field is scene content,
        // not a primary-view garnish, so a CameraSensor must see it.
        // The caller passes the pipeline (the MSAA overlay variant, the
        // 1-sample sibling a secondary view needs, or F4's linear-HDR glow
        // variant), and `glowPass` selects which fields are drawn and which
        // value domain they are written in.
        // `pipeAlpha` is the alpha-over sibling, bound for fields that ask for
        // it; VK_NULL_HANDLE (the glow leg) draws every field additively.
        void recordFieldBillboards(VkCommandBuffer cb, VkPipeline pipe, bool glowPass,
                                   VkPipeline pipeAlpha = VK_NULL_HANDLE);

        // F4: build this view's BillboardViewGpu record — the display transform
        // and the fog medium the quads are attenuated by — and publish it into
        // the pass's per-frame block. Returns 0 when there is no room, which
        // the caller must treat as "skip the draw": a null buffer_reference
        // dereference is undefined, not a no-op.
        [[nodiscard]] VkDeviceAddress pushBillboardViewRecord(const Matrix4& viewM,
                                                              bool linearOut);

        // F4: render the glow-enabled fields into BillboardGlowPass's offscreen
        // target and run the pyramid over it. Recorded AFTER the upscale/post
        // passes and BEFORE recordHybridOverlay, because the composite draw that
        // consumes it lives inside that overlay pass and compute cannot run
        // inside a render-pass instance. Added to the graph only when
        // fieldBillboardGlowActive().
        void recordFieldBillboardGlow(VkCommandBuffer cb);
        // R8/R9: (T_cam, T_sun) once per particle for the view whose draws come
        // next. Outside any render-pass instance; no-op when nothing marches.
        void recordFieldTransmittance(VkCommandBuffer cb);
        // BillboardRepr::softDistance: when a visible field asks for a soft
        // edge, add the pass that turns `depth` (the depth this view's sprites
        // are tested against, in DEPTH_STENCIL_READ_ONLY_OPTIMAL) into this
        // view's linear depth buffer, and return that buffer so the billboard
        // passes after it can declare their read. VK_NULL_HANDLE (and no pass)
        // when no field asks or the view cannot provide one.
        VkBuffer addFieldSoftDepthPass(vulkan::rg::RenderGraph& g, VkImage depth, VkImageView depthView,
                                       bool multisampled, VkExtent2D ext);
        // Lazily create the glow pass + its two graphics pipelines + its images.
        // Called from the PREPARE window (never mid-recording), so the pipeline
        // compile lands where every other lazy creation in this renderer does.
        void ensureFieldBillboardGlow();
        // The billboard HDR layer: create and size its targets, in the
        // same PREPARE window, when the switch is on and a field draws quads.
        void ensureFieldBillboardHdrLayer();
        // Draw the fields into the layer's two linear targets, against the
        // overlay pass's depth (or the layer's own copy of it, with the splat
        // stamp, when `ownDepth`). Its own graph pass.
        void recordFieldBillboardHdrLayer(VkCommandBuffer cb, bool ownDepth, bool stamp);
        // The primary's transmittance prepass, soft depth and HDR layer, once
        // per frame: ahead of the post composite on the upscaler path, at the
        // head of the tail on the built-in TAA path.
        void addFieldBillboardFrontPasses(vulkan::rg::RenderGraph& g);

        // Any visible ParticleField will draw billboards this frame. Cheap (a
        // handful of fields at most) and used by sceneHasOverlayContent(), so
        // the overlay depth prepass and the overlay pass itself run for a scene
        // whose ONLY overlay content is a field of embers.
        [[nodiscard]] bool sceneHasFieldBillboards() const;
        // A visible field has an enabled billboard representation, live
        // particles or not: what the billboard HDR layer's images are
        // allocated for, so the first live particle does not pay for them.
        [[nodiscard]] bool sceneHasFieldBillboardRepr() const;

        // The scene's SceneCensus for this render(): one walk on the first ask,
        // a per-type answer for each node, so a scene of plain meshes pays no
        // failed cast per node per collector (about 75 ns a cast on MSVC, and
        // there were four such walks a frame).
        const SceneCensus& sceneCensus(Object3D& scene);

        // Snapshot the census's world-space Sprites (screenSpace == false) with
        // a texture map, their world transform + material, into
        // lastVisibleSprites_. Run every perspective frame (sprites move / spawn
        // / expire constantly — no snapshot caching). Mirrors OverlayPass's
        // sprite collection, minus the screen-space branch.
        void collectWorldSprites(Object3D& scene);

        // Sidecar collector for SplatClouds — same shape and the same reason as
        // collectWorldSprites: a SplatCloud is not a kind the snapshot/BLAS
        // machinery has anything to say about, so it is gathered from the
        // census and nothing else has to change. Also estimates each cloud's
        // p1/p99 view distance here, where the camera is in hand, from the same
        // fixed-stride sample SplatCloud::sortByDepth uses on the GL path.
        void collectSplatClouds(Object3D& scene, Camera& camera);

        // Composite this frame's splat clouds into sceneHdr. Called between the
        // deferred shade and the depth of field. Reads splatParams_, which the
        // collector filled while it still had the camera — everything except
        // the raster's sub-pixel jitter, which is only decided later (in
        // uploadRasterCameraUbo) and is folded in here.
        void addSplatPasses(vulkan::rg::RenderGraph& g);
        void addSecondaryViewSplatPasses(vulkan::rg::RenderGraph& g);

        // Stamp the splat depth AOV into the overlay's depth attachment, so
        // the post-resolve overlay draw (wireframe, lines, world sprites,
        // particle billboards) is occluded by a cloud in front of it — split
        // in two so the stamp can be a DRAW inside the hybrid overlay pass,
        // ordered after the overlays kSplatUnoccludedOverlayLayer exempts.
        // splatStampPrepare gates (clouds AND overlay content this frame —
        // see splatOverlayDepth_ for the latch that turns the AOV on for it)
        // and points the stamp's set at this frame's AOV; true obliges the
        // overlay pass to declare and attach its depth WRITABLE and record
        // recordSplatStampDraw at the exempt/occluded boundary. Called while
        // the frame graph is built.
        bool splatStampPrepare();
        void recordSplatStampDraw(VkCommandBuffer cb, VkPipeline pipe = VK_NULL_HANDLE);

        // The vertex buffer holding `points`' blended positions for this
        // frame, or VK_NULL_HANDLE when it does not morph (no morphTargets on
        // its material, no position targets, or every influence zero), and
        // the caller draws the geometry's rest positions.
        VkBuffer ensureMorphedPointsUploaded(Points& points, const BufferGeometry& geom);

        // Evict entries untouched for longer than the in-flight window. The
        // margin (> kFramesInFlight) guarantees no evicted buffer is still
        // referenced by an in-flight command buffer; the retire queue then holds
        // them until their frame serial has provably passed, so this is safe to
        // call from endFrame while earlier frames are still executing.
        static constexpr uint64_t kLineGeomEvictAge = 8;
        void sweepLineGeomCache() {
            if (lineGeomCache_.empty() && morphedPointsCache_.empty()) return;
            if (overlayFrameCounter_ <= kLineGeomEvictAge) return;
            const uint64_t cutoff = overlayFrameCounter_ - kLineGeomEvictAge;
            for (auto it = lineGeomCache_.begin(); it != lineGeomCache_.end();) {
                if (it->second.lastTouch < cutoff) {
                    for (auto& rec : it->second.slots) {
                        retire(std::move(rec.vertex));
                        retire(std::move(rec.index));
                        retire(std::move(rec.color));
                    }
                    it = lineGeomCache_.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = morphedPointsCache_.begin(); it != morphedPointsCache_.end();) {
                if (it->second.lastTouch < cutoff) {
                    for (auto& b : it->second.vertex) retire(std::move(b));
                    it = morphedPointsCache_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Lazy-upload Line / LineSegments / Points geometry into host-visible
        // vertex (+ optional index and color) buffers and return the set to
        // bind while recording frame slot currentFrame. Returns nullptr if the
        // geometry has no usable position attribute. Cached by raw pointer;
        // the BufferAttribute::version counters gate re-uploads, and a
        // geometry that changes gets one buffer set per frame in flight (see
        // vulkan::LineCacheEntry).
        const vulkan::LineRec* ensureLineGeometryUploaded(const BufferGeometry* geom);

        // ── TAA resources ───────────────────────────────────────────────────
        // Pipeline + images + descriptor sets now live in vulkan::TaaResolve.
        // This helper packs the external view sources (raster gbuffer +
        // swapchain) into the args struct TaaResolve expects, then asks
        // TaaResolve to rewrite all per-(frame, swapchain-image) descriptor
        // sets. Call after createTaaImages OR after a swapchain resize OR
        // after the raster gbuffer is reallocated.
        void rewriteTaaDescriptors();

        // The bloom pyramid's internal sets + the PostComposite (reads
        // sceneHdr + bloom level 0 + the per-frame G-buffer for the solid-bg
        // sky bypass, writes the per-frame TAA input). Call after bloom_->
        // createImages OR after the gbuffer / TAA images are reallocated.
        void rewriteBloomDescriptors();
        void ensureSplatTarget();

        // Raster-first deferred shade reads the camera + lights UBOs, the env
        // PMREM, the raster material G-buffer (normal+rough / albedo+metal /
        // depth / ids) and writes bloom_->sceneHdr. Call after the raster
        // gbuffer or sceneHdr is reallocated (resize) and after the env image
        // is rebuilt. The UBO buffers are stable; rewriting them is harmless.
        void rewriteDeferredDescriptors(int onlyFrame = -1);

        // Fit the probe grid to the scene's world AABB (called when probe GI
        // is enabled and the scene structure changed). Walks the canonical
        // entry list with the same boundingBox·worldMatrix union the frustum
        // cull uses; overlay/particle entries don't contribute geometry.
        void fitProbeGridToScene();


        // Lazy bring-up: called at the start of each render() when hybrid is on.
        // Idempotent — handles both initial creation and post-resize reallocation.
        // 1x1 multisample dummy images (2x, or 4x where the device lacks 2x)
        // for deferred_shade.comp's dispatch-B sampler2DMS/usampler2DMS
        // bindings when MSAA is off — see gbufDummyMS_'s declaration for why
        // a single-sample dummy can't stand in here. Formats mirror
        // normal/depth/ids/uv/albedo exactly
        // (SAMPLED usage only; never rasterized into, never resolved from).
        void ensureGbufDummyMS();

        void ensureHybridResources();

        // 1×1 black RGBA32F dummy so the env-map binding is always populated.
        // Replaced with the real scene environment by uploadEnvFromTexture()
        // the first time render() sees a non-empty scene.environment / .background.
        void createDefaultEnvImage();

        // Single shared sampler for every material texture in binding 8.
        // Trilinear + 16× anisotropy: minified surfaces fetch from the proper
        // mip level (set by createSampledImage2D's blit-chain) and glancing
        // angles get aniso-filtered, both of which kill the per-frame texture
        // shimmer that TAA was trying to absorb.
        void createTextureSampler();

        // Slot 0 of the bindless array is a 1×1 white texel — used to fill the
        // descriptor array padding so every slot has a valid view, and as the
        // descriptor for materials whose albedoTexIndex stays at -1 (the
        // shader still indexes safely; the multiply by 1.0 is a no-op).
        void createDefaultMaterialTexture();

        // GPU skinning pipeline moved into vulkan/SkinningPipeline.{hpp,cpp}.
        // Renderer holds a unique_ptr `skinning_` and delegates allocation +
        // dispatch through it.

        // Water displace pipeline moved into vulkan/WaterDisplacePipeline.{hpp,cpp}.
        // Renderer holds a unique_ptr `waterDisplace_` and delegates allocation
        // + dispatch through it.

        // createPmremEnvImage moved into vulkan/EnvPrefilter.{hpp,cpp}.
        // Call envPrefilter_->buildPmrem(w, h, pixels, byteSize) instead.

        static VkSamplerAddressMode wrapToVk(TextureWrapping w) {
            switch (w) {
                case TextureWrapping::ClampToEdge:    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                case TextureWrapping::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                default:                              return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            }
        }

        // Upload `tex` to the next bindless slot if not already cached. Returns
        // the slot index; -1 if upload fails or capacity is exhausted.
        // uv2Ok: the caller samples it with the second UV set (the aoMap), so
        // a Texture.channel of 1 is honoured and not warned about.
        int32_t ensureMaterialTexture(const std::shared_ptr<Texture>& texSp, bool uv2Ok = false);

        // Build a tightly-packed RGBA8 sampled image from a material texture's
        // CPU image (BCn decompress / channel pad / float quantize). Returns a
        // null Image2D ({}) on unsupported/degenerate input. Slot-agnostic so the
        // initial upload (ensureMaterialTexture) and the in-place re-upload
        // (refreshDirtyMaterialTextures) share one code path.
        Image2D buildMaterialImage2D(const Texture* tex);
        PreparedMaterialImage prepareMaterialImage(const Texture* tex, bool bc7Ok) const;
        static PreparedMaterialImage preparedLevels(uint32_t w, uint32_t h, VkFormat fmt,
                                                    std::vector<std::vector<std::uint8_t>>&& levels,
                                                    VkSamplerAddressMode u, VkSamplerAddressMode v, const char* name);
        static PreparedMaterialImage preparedRgba(uint32_t w, uint32_t h, VkFormat fmt, std::vector<unsigned char>&& rgba,
                                                  VkSamplerAddressMode u, VkSamplerAddressMode v, const char* name);
        Image2D createPreparedImage(const PreparedMaterialImage& p);
        // ── Hidden admission off the edit frame ──────────────────────────────
        // A mesh that arrives hidden (resident, visible false) is neither drawn
        // nor traced until it shows, so its admission (the BLAS build, the
        // textures' encode and upload) need not be paid on the frame it was
        // added: its slot edit places it with default rows, and
        // pumpHiddenAdmissions, after each frame's scene walk, starts its
        // textures' CPU half (prepareMaterialImage) and its field's prep on
        // worker threads, a few a frame (EntrySpan::prepStarted), then lands
        // the spans whose jobs are ready, a few a frame: record, images, rows,
        // and a full TLAS build as for any first show. Nothing visible
        // changes. A mesh shown before its admission landed is placed again
        // by that frame's slot edit, with the synchronous admission a mesh
        // arriving visible gets (a texture still being prepared is waited
        // for, not encoded twice; one not started yet is encoded there).
        // THREEPP_VULKAN_SYNC_ADMIT=1 admits hidden meshes
        // synchronously too, and encodes a slot edit's textures one after the
        // other on the render thread (the A/B lane: the admission as it was).
        static bool syncAdmission();
        // Starts a pending span's worker jobs (its mesh's textures' CPU half,
        // a field's prep) and returns how many it started; pumpHiddenAdmissions
        // calls it for a few spans a frame.
        uint32_t beginHiddenAdmission(Mesh& m);
        // The lidar's entry -> stable id table (entryStableIds_) is written
        // read-only (stableIdIfAssigned) where entries are made, and a mesh's
        // auto id is assigned by the indirect draw builder the first time it
        // draws: an entry made before that reads 0. The rebuild used to heal it
        // at the next rebuild; the slot path never makes one, so every mesh
        // streamed in after the first frame reported lidar id 0. On a frame
        // that ends lean after entries were made or shown, this fills the
        // visible spans still at 0 from their meshes' assigned ids, until none
        // is left.
        void refreshPendingStableIds();
        // Start the CPU half of every texture of `m` that is not resident and
        // not already being prepared (worker threads); ensureMaterialTexture
        // takes the result. Returns how many it started. The slot edit calls
        // it for all a visible edit's new meshes before building their descs,
        // so their encodes run side by side; a hidden span's start from
        // pumpHiddenAdmissions, a few a frame.
        uint32_t prepareTexturesAsync(Mesh& m);
        // Every map buildMaterialDesc would admit for `m` (null where unset).
        static std::vector<std::shared_ptr<Texture>> texturesOf(const Mesh& m);
        void pumpHiddenAdmissions();

        // GL compressed-format enum (Image::compressedFormat, as the DDS
        // loader stores it) → Vulkan BC format. The COLOR-SPACE TAG on the
        // Texture wins over the GL enum's own sRGB-ness, matching the old
        // decode path (which uploaded RGBA8_SRGB/UNORM purely by tag): FBX
        // materials tag albedo sRGB and data maps linear regardless of how
        // the DDS was authored. UNDEFINED = no pass-through (e.g. BC6H).
        static VkFormat glCompressedToVk(unsigned int glFmt, bool srgb);

        // BC7 sampled-image support, queried once. Universal on desktop GPUs;
        // the check exists for headless/software drivers (lavapipe et al).
        bool bc7SampledSupported() {
            if (bc7Supported_ < 0) {
                VkFormatProperties fp{};
                vkGetPhysicalDeviceFormatProperties(ctx->physicalDevice(),
                                                    VK_FORMAT_BC7_SRGB_BLOCK, &fp);
                bc7Supported_ = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? 1 : 0;
            }
            return bc7Supported_ == 1;
        }

        // Re-upload any cached material texture whose Texture::version() changed
        // since last upload (e.g. a DataTexture edited via setData + needsUpdate),
        // in place at its existing bindless slot. ensureMaterialTexture caches by
        // pointer and the bindless image array is only rewritten on a full scene
        // rebuild, so without this a live texture edit would never reach the GPU.
        // Cheap version scan each frame; only drains + rewrites when dirty.
        void refreshDirtyMaterialTextures();

        // Generate a 64×64 blue-noise tile via void-and-cluster (Ulichney 1993).
        // One-time CPU work at startup; the resulting tile has:
        //   - Spatially smooth correlation between adjacent cells (silhouette
        //     stability — neighbors agree on object/background membership)
        //   - Globally well-distributed (no coherent shake possible)
        //   - Suppressed low-frequency content (variance lives in high
        //     frequencies that denoise + TAA absorb)
        //
        // ~50ms on a single core for the 64×64 case (M=4096). Acceptable as a
        // one-shot startup cost. The output is uint8 ranks normalized to
        // [0, 255]; shader maps via R8_UNORM sampler to [0, 1) jitter.
        std::vector<uint8_t> generateBlueNoiseTile_();

        // Generate the tile and upload as an R8_UNORM image. Called once at
        // ctor time (after createDescriptorPool) so the descriptor write below
        // can include it.
        void createBlueNoiseImage_();

        // Tileable foam detail tile — R = micro bubble brightness, G = ridged
        // "lace" filament pattern. All lattice frequencies are integers so
        // every octave wraps seamlessly across the tile edges. Channel content
        // matches the procedural noise it replaces in the foam shading:
        //   R ≈ vnoise·0.55 + vnoise(×2.3)·0.30 + vnoise(×5.1)·0.15 with
        //       0.154 m base features over the shader's 4 m world mapping;
        //   G ≈ 1 − 2·|nf − 0.5| (filament cores at nf = 0.5) with ~1.2 m
        //       cells over the 12 m mapping.
        std::vector<unsigned char> generateFoamDetailTile_(int res);

        void createFoamDetailImage_();
        // Crest foam's detail tile (deferred binding 92), uploaded the first
        // time an ocean turns crest foam on (syncOceanCrestFoam).
        void createCrestFoamPattern_();

        // 1×1 R32F dummy ocean height used when no DisplacedMesh is in the
        // scene. deferred_shade.comp gates on oceanFineTileSize > 0 so the
        // sampler result is unread; the descriptor still needs a valid
        // view/sampler to keep the layout populated. Replaced (view only) with
        // the active ocean's cascade-2 height image via rewriteDeferredDescriptors().
        void createOceanFineDummy_();

        // 1×1 R32F stand-in for the ocean's sampled inputs (crest foam,
        // wave climate) while no DisplacedMesh provides them: the descriptor
        // still needs a valid view/sampler.
        void createOceanFoamDummy_();

        // Detect the active env texture (scene.environment, falling back to
        // scene.background.texture()) and upload it if it differs from the
        // currently bound one. Returns true when descriptors must be rewritten.
        bool refreshEnvTextureFromScene(Object3D& scene);

        // Populate `infos` with the current bindless material-texture array,
        // padding any unused tail slots — and any slots reclaimed by prune —
        // with the slot-0 white default. Reclaimed slots are detected via a
        // null view (Image2D is reset to {} on destroy).
        template <std::size_t N>
        void fillMaterialTextureInfos(std::array<VkDescriptorImageInfo, N>& infos) {
            const VkImageView fallbackView = materialTextures[0].view;
            // The policy samplers (see the textureSampler_ member comment) —
            // the per-image samplers are deliberately not bound here. Slots
            // whose texture asked for ClampToEdge get the clamp flavour of
            // the SAME filter policy (materialTexClampUV_).
            const VkSampler samplerRepeat = materialSampler(false);
            const VkSampler samplerClamp  = materialSampler(true);
            for (std::size_t i = 0; i < N; ++i) {
                const bool hasSlot = i < materialTextures.size() && materialTextures[i].view;
                const bool clampUV = hasSlot && i < materialTexClampUV_.size() && materialTexClampUV_[i];
                infos[i].sampler     = clampUV ? samplerClamp : samplerRepeat;
                infos[i].imageView   = hasSlot ? materialTextures[i].view : fallbackView;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }

        // The sampler the material-texture bindings use this frame: AUTO
        // (16×, jittered or not) unless
        // setTextureAnisotropy forced a level, in the REPEAT (default) or
        // CLAMP_TO_EDGE wrap flavour. May lazily create the custom sampler
        // pair for forced levels other than 1/16. Implemented in
        // VulkanCoreTextures.cpp with the sampler creation code.
        [[nodiscard]] VkSampler materialSampler(bool clampUV);
        // Invalidate every descriptor write that binds material textures so
        // the next per-frame lazy rewrite picks up a sampler-policy change.
        // Same (fence-idle-safe) pattern refreshDirtyMaterialTextures uses.
        void markMaterialSamplerDirty() {
            deferredDescDirty_.fill(true);
            // Every view: the flag is per-view (it guards per-view sets).
            for (auto& v : views_) { v->rasterMatTexValid_.fill(0); v->shadowMatTexValid_.fill(0); }
        }
        void setTextureAnisotropy(float aniso);// 0 = auto; VulkanCoreTextures.cpp

        // Rebuild every resource sized to the deferred shade's render extent
        // — the gbuf / reservoir / GI ping-pongs, the denoiser + TAA images,
        // the raster G-buffer, the upscale intermediates — plus the
        // descriptor sets that bind them. Shared by swapchain recreation and
        // runtime renderScale changes. createReservoirImages resets
        // accumulation; the caller must have the GPU idle before this runs.
        void reallocateRenderExtentResources();

        void recreateSwapchainAndDescriptors();

        // Runtime render-scale change. Reallocates every render-extent
        // resource and resets accumulation. Issues a vkDeviceWaitIdle.
        // Safe to call from inside the user's animate lambda — when a
        // frame is already mid-record (frameState_ != Idle) the
        // reallocation is deferred to the next beginDeferredFrame, after
        // that frame's fence has signalled. Otherwise (Idle) applies
        // immediately. No-op when the clamped value already matches.
        void setRenderScale(float scale);

        // Raster G-buffer MSAA sample count. 1 (default) = today's single-
        // sample path, byte-identical output. 2/4 rasterize the G-buffer at
        // that sample count and resolve with a dominant-sample pick (see
        // GbufResolve) instead of a box/average resolve — averaging normals/
        // ids/depth across a silhouette produces nonsense; picking the
        // majority-covering surface is what a G-buffer needs. Reallocates
        // render-extent resources (render pass + pipelines + MS images),
        // same idle-gate as setRenderScale — safe to call from inside the
        // user's animate lambda.
        void setGbufferMsaa(uint32_t samples);

        [[nodiscard]] uint32_t gbufferMsaa() const { return gbufMsaaSamples_; }

        // VulkanRenderer::setProbeCascades. Growing the probe stores rewrites
        // the deferred sets of every view, so it shares setRenderScale's idle
        // gate: mid-frame the request waits for the next beginDeferredFrame.
        void setProbeCascades(uint32_t count);
        void applyProbeCascades(uint32_t count);// device idle

        // Whether the DEVICE can rasterize + sample the G-buffer at this
        // count. The MS G-buffer spans color, depth AND integer (ids) formats,
        // each used as both attachment and sampled image, so the usable counts
        // are the intersection of all five limit masks. The spec only
        // guarantees 1x and 4x in each of them — 2x is optional (lavapipe
        // advertises exactly 1|4), and vkCreateImage with an unsupported
        // sample count is a spec violation
        // (VUID-VkImageCreateInfo-samples-02258 family) and a crash risk, not
        // a graceful fallback. Callers that want N>1 fall back to 4, the
        // mandatory multisample count. Same clamping idea as overlaySamples()
        // above.
        [[nodiscard]] bool gbufMsaaCountSupported(uint32_t samples) const {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const auto& lim = props.limits;
            const VkSampleCountFlags mask = lim.framebufferColorSampleCounts &
                                            lim.framebufferDepthSampleCounts &
                                            lim.sampledImageColorSampleCounts &
                                            lim.sampledImageDepthSampleCounts &
                                            lim.sampledImageIntegerSampleCounts;
            // VK_SAMPLE_COUNT_N_BIT == N numerically (see overlaySamples).
            return (mask & static_cast<VkSampleCountFlags>(samples)) != 0;
        }

        void setSplatDepthAov(SplatDepthMode mode);

        // What the APP asked for — the answer VulkanRenderer::splatDepthAov()
        // and the readGBufferAOV(SplatDepth) gate hand back, unchanged by the
        // renderer's own use of the same image below.
        [[nodiscard]] bool splatDepthAov() const { return splatDepthMode_ != SplatDepthMode::Off; }
        // Whether the AOV image is REAL (full-res, written) this frame.
        // splatOverlayDepth_ is the renderer's own reason to want it — the
        // overlay depth stamp reads it — and it feeds the same image, so every
        // allocate/clear/write decision has to consult both. A caller who
        // never asked still gets "you never asked" from the public getter.
        [[nodiscard]] bool splatDepthAovAllocated() const {
            return splatDepthAov() || splatOverlayDepth_;
        }
        // The statistic the raster exports. Median is the front of the cloud,
        // which is what an occlusion test wants; the expected value sits a
        // cloud-thickness behind it. So when the AOV exists ONLY for the stamp
        // the mode is Median, and when the app asked for it the app's choice
        // wins — one image, and a sensor's answer is not ours to redefine.
        [[nodiscard]] bool splatDepthMedian() const {
            return splatDepthMode_ == SplatDepthMode::Median ||
                   (splatDepthMode_ == SplatDepthMode::Off && splatOverlayDepth_);
        }
        [[nodiscard]] SplatDepthMode splatDepthAovMode() const { return splatDepthMode_; }

        // Deferred scene dispatch: shades the raster material G-buffer into
        // bloom_->sceneHdr (direct analytic lights + split-sum specular IBL +
        // approximate diffuse IBL + ray-query accents). Adds its passes to
        // the frame graph between the G-buffer/AS head and the splats + post
        // tail (recordCommandBuffer); defined in VulkanRenderer.cpp.
        // The async-compute pilot builds it in two calls: AsyncOnly (probeGI,
        // clusterBuild, cloudShadow, froxels, into computeGraph_) and
        // SkipAsync (everything else, into the frame graph).
        enum class DispatchPart { All, AsyncOnly, SkipAsync };
        void addSceneDispatchPasses(vulkan::rg::RenderGraph& g, DispatchPart part = DispatchPart::All);

        // Called once after bloom_->createImages(); wires sceneHdr views.
        void onAfterBloomCreateImages() {
            // Auto-exposure meters the PRIMARY's sceneHdr — it drives the
            // display's exposure, and a secondary must not re-point it at its
            // own (differently exposed, differently sized) image.
            if (!autoExposure_ || view().secondary) return;
            VkImageView views[kFramesInFlight];
            for (uint32_t f = 0; f < kFramesInFlight; ++f)
                views[f] = view().bloom_->sceneHdrView(f);
            autoExposure_->rewriteDescriptors(views);
        }

        // CPU per-frame tick: lazy-init, then read histogram + advance EMA.
        void onBeginDeferredFrame(uint32_t frame, float dt) {
            if (!autoExposureEnabled_) return;
            if (!autoExposure_) {
                autoExposure_ = std::make_unique<vulkan::AutoExposure>(*ctx, kFramesInFlight);
                autoExposure_->adaptSpeed = autoExpSpeed_;
                autoExposure_->minEV      = autoExpMinEV_;
                autoExposure_->maxEV      = autoExpMaxEV_;
                VkImageView views[kFramesInFlight];
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    views[f] = view().bloom_->sceneHdrView(f);
                autoExposure_->rewriteDescriptors(views);
            }
            // Physical camera: the EMA adapts an EV COMPENSATION around the
            // EV100-derived exposure instead of an absolute multiplier
            // around 1.0 (the histogram/EMA machinery is reused unchanged).
            autoExposure_->baseExposure = physicalCamera_ ? physicalExposure() : 1.f;
            autoExposure_->tick(frame, dt);
        }

        // Exposure value used for this frame's composite push constant.
        // Auto-exposure wins when active (AutoExposure::exposure() composes
        // the physical-camera base it was handed in onBeginDeferredFrame);
        // otherwise physical camera mode derives it from aperture/shutter/ISO.
        [[nodiscard]] float currentExposure() const {
            if (autoExposureEnabled_ && autoExposure_)
                return autoExposure_->exposure();
            return physicalCamera_ ? physicalExposure() : toneMappingExposure_;
        }

        [[nodiscard]] bool envSunExtractionWanted() const {
            return envSunPolicy_ != VulkanRenderer::EnvSunPolicy::Off;
        }
        [[nodiscard]] bool envSunDefersToSceneSun() const {
            return envSunPolicy_ == VulkanRenderer::EnvSunPolicy::Auto;
        }

        // ── Shadows (setShadowMode; VulkanCoreShadows.cpp) ─────────────────
        // The request as set (THREEPP_VK_SHADOW_MODE seeds it) and the knobs of
        // the map mode; activeShadowMode() resolves Auto, and RayTraced on a
        // device without ray query, to what runs.
        VulkanRenderer::ShadowMode shadowModeReq_ = shadowModeFromEnv();
        // THREEPP_VK_SHADOW_CASCADES=<1..4> and THREEPP_VK_SHADOW_LAMPS=<spots>,<points>
        // seed these (ablation knobs for the map pass's cost); the setters win.
        uint32_t shadowCascades_   = shadowCascadesFromEnv();
        uint32_t shadowedSpots_    = shadowedLampsFromEnv(0);
        uint32_t shadowedPoints_   = shadowedLampsFromEnv(1);
        uint32_t shadowMapSizeCap_ = shadowMapSizeCapFromEnv();// THREEPP_VK_SHADOW_MAPSIZE, 0 = none
        static uint32_t shadowCascadesFromEnv();
        static uint32_t shadowedLampsFromEnv(int which);
        static VulkanRenderer::ShadowMode shadowModeFromEnv();
        static uint32_t shadowMapSizeCapFromEnv();
        [[nodiscard]] VulkanRenderer::ShadowMode activeShadowMode() const;
        [[nodiscard]] bool shadowMapsActive() const {
            return activeShadowMode() == VulkanRenderer::ShadowMode::ShadowMaps;
        }
        void ensureShadowDummies();
        void destroyShadowResources();
        // What bindings 80-83 of the current view's deferred set slot `frame`
        // should name now: the view's maps while its pass is live, else the
        // dummies.
        struct ShadowBinding {
            VkImageView cascades = VK_NULL_HANDLE;
            VkImageView spots    = VK_NULL_HANDLE;
            VkImageView points   = VK_NULL_HANDLE;
            VkBuffer    ubo      = VK_NULL_HANDLE;
            uint64_t    gen      = 0;// 0 = the dummies
        };
        [[nodiscard]] ShadowBinding shadowBindingFor(uint32_t frame) const;
        // Per view, per frame, after the slot's fence: rewrite bindings 80-83
        // of the view's deferred set slot `frame` when they name something
        // else than shadowBindingFor(frame) says.
        void syncShadowDescriptors(uint32_t frame);
        // Per view, after its cull and G-buffer draw build: decide whether the
        // view draws shadow maps this frame (view().shadowLive) and, if so, fit
        // them to `camera`, build the caster list and fill the UBO; then sync
        // the deferred set's bindings 80-83.
        void prepareShadowMaps(Camera& camera);
        // The fit and fill half of prepareShadowMaps; false when the view has
        // nothing to draw a map for (no sun, no slotted lamp).
        bool fitShadowMaps(Camera& camera);
        // The maps' sizes this frame's sun and lamp slots ask for (all zero:
        // none), as fitShadowMaps allocates them.
        [[nodiscard]] vulkan::ShadowMapPass::Sizes shadowMapSizes() const;
        // The caster list of the current view's maps: every entry a shadow
        // ray could hit (shadowOccludes' set: no blend, no transmission, no
        // strong emitter, no viewmodel), kept when its world sphere meets a
        // cascade's box or a slotted lamp's range. Same DrawInfo records as
        // the G-buffer's.
        // One volume per layer of ShadowMapPass::layers, in the same order:
        // a cascade's box (centre, the sun's frame, half extent; every box
        // shares the frame), a spot's cone (the lamp, its axis, tan of the
        // half angle, far) or a point lamp's cube face (the lamp, the face's
        // axis, far). A layer the view cannot see is `skip`: cleared, drawn
        // into by nothing.
        struct ShadowCullVolumes {
            struct Volume {
                enum Kind : uint32_t { Box, Cone, Face };
                Kind  kind = Box;
                float c[3]{};
                float x[3]{}, y[3]{}, l[3]{};
                float d[3]{};
                float r       = 0.f;
                float tanHalf = 0.f;
                bool  skip    = false;
                bool  cached  = false;// a cascade served from the static cache: only moving casters reach it
            };
            std::vector<Volume> layers;
        };
        // The static caches of the cascades and the lamp layers (ShadowMapPass::cache, spotCache, pointCache):
        // THREEPP_VK_SHADOW_CACHE=0 turns it off; THREEPP_VK_SHADOW_CACHE_MARGIN
        // (1..2, default 1.15) is how much wider than the slice a cascade's box
        // is frozen: 1 = the slice itself, so the image is the uncached one but
        // a camera that moves redraws every frame (the twin's drone at 30 m/s:
        // 5 ms); 1.15 = a box 15 % wider, kept while the camera stays inside
        // it, at texels 15 % coarser (0.4 ms, a 2-4 ms rebuild frame every
        // second or so). A caster is static unless it is a deformer, an interop
        // mesh, grass, or recently moved (meshMovedSticky_); the hash of that
        // set is what a frozen box is valid for.
        bool  shadowCacheOn_     = shadowCacheFromEnv();
        float shadowCacheMargin_ = shadowCacheMarginFromEnv();
        static bool  shadowCacheFromEnv();
        static float shadowCacheMarginFromEnv();
        [[nodiscard]] bool     shadowCasterDynamic(uint32_t entry) const;
        [[nodiscard]] uint64_t shadowStaticHash();
        void buildShadowDrawData(uint32_t frame, const ShadowCullVolumes& vol, const Vector3& camPos);
        void     updateShadowCasterSnap(vulkan::ShadowMapPass& smp, uint32_t cascades);
        void createShadowPipeline();
        // Writes GpuClusterLight::shadowSlot into `recs` (src: each record's
        // light) and refreshes the two slot tables; -1 everywhere when the maps
        // are not in use.
        void assignShadowSlots(std::vector<GpuClusterLight>& recs, const std::vector<Light*>& src, Camera& camera);

        // Frame recording, split along the frame's own stage seams (bodies in
        // VulkanCoreRecord.cpp). recordCommandBuffer is the narrative: it adds
        // each stage's passes to the frame graph in order, and the graph
        // derives the barriers between them from what each pass declares.
        void updatePaneRegion();
        // The frame's head, as graph passes ahead of the deformers: the GPU
        // instance expansion, the ParticleField emitter (+ surface bake), the
        // interop snapshot and indirect-count copies, and the density scatter.
        // Once for all views; every view's G-buffer and billboard draws, the
        // froxels and the shade read what they write.
        void addHeadPasses(vulkan::rg::RenderGraph& g);
        // Every ParticleField's position ring, prevPositions, counts block,
        // surface-bake map and draw-indirect records: what the emitter, the
        // interop snapshot and the counts copy write, and the density scatter,
        // the G-buffer / billboard vertex stages and the transmittance prepass
        // read through buffer device addresses and indirect draws.
        static constexpr const char* kParticleFields = "particles.fields";
        // A pass that reads kParticleFields: `draws` = the G-buffer and
        // billboard draws (indirect records + vertex pulls), otherwise the
        // transmittance prepass (compute). No-op without a ParticleField pass.
        void useParticleFields(vulkan::rg::RenderGraph& g, vulkan::rg::PassBuilder& pass, bool draws);
        // The deformers (dynamic plain meshes, skinned, tet, displaced water,
        // grass) and the per-frame TLAS refit, as graph passes. Each deformer
        // pass keeps the barriers between its own copy / dispatch / BLAS build;
        // all of them write kSceneGeometry.
        void addDeformAndTlasPasses(vulkan::rg::RenderGraph& g);
        void recordSkinnedDeforms(VkCommandBuffer cb);
        void recordTetDeforms(VkCommandBuffer cb);
        // Every deformable mesh's vertex / normal / prevVertex / index buffers,
        // its BLAS, and the ocean's FFT outputs: the graph memory resource
        // (RenderGraph::importMemory) the deformers write and the TLAS refit,
        // the raster passes and every ray query read. Ray queries declare it
        // through the TLAS's registration (registerAccelerationStructure).
        static constexpr const char* kSceneGeometry = "scene.geometry";
        // The G-buffer stage as graph passes (VulkanCoreRecord.cpp).
        void addGbufferPasses(vulkan::rg::RenderGraph& g);
        void recordGbufferRaster(VkCommandBuffer cb);
        void recordGbufferMsaaResolve(VkCommandBuffer cb);
        void recordOverlayDepthPrepass(VkCommandBuffer cb);
        // The scene form's draws of entries [first, last) into `cb`, inside the
        // prepass's rendering instance: pipeline, set, one draw per entry.
        void recordOverlayDepthDraws(VkCommandBuffer cb, size_t first, size_t last);

        VkCommandBuffer acquireSecondaryCommandBuffer();
        void            destroyRecordApartBuffers();
        [[nodiscard]] bool gbufferMsaaResolveActive();
        [[nodiscard]] bool overlayDepthPrepassActive();
        // The import form of the prepass: the descriptor refresh (graph-build
        // time, this slot's post-fence window) and the draw (inside the
        // prepass's render-pass instance).
        void prepareOverlayDepthImport();
        void recordOverlayDepthImport(VkCommandBuffer cb);
        // The overlay pass's entry walks, per span (overlaySpanEntries): the
        // indices of the entries whose span is flagged isOverlay (or, with
        // `particle`, isParticle), in entry order. One scratch, reused.
        void overlaySpanEntries(bool particle);
        // Events-only mode: the swapchain cleared to black (the canvas the
        // event accumulator's sprite overlay draws on), as a graph pass.
        void addEventsOnlyClearPass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // The acquired swapchain image, imported UNDEFINED (the entry
        // barrier's ALL_COMMANDS source chains to the acquire wait) with exit
        // layout GENERAL, where the post-graph overlay and endFrame expect it.
        // Called before any pass names it, so these layouts are the ones the
        // graph keeps.
        vulkan::rg::ImageHandle importSwapchain(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // importSwapchain plus the split-screen clear.
        void addSwapchainPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // Depth of field, bloom, the temporal resolve / upscaler and the post
        // composite, as passes of the frame graph.
        // `secondary`: a secondary view's reduced chain — no DoF, no DLSS/FSR,
        // no sharpening or motion blur, the authored TAA alpha, and its own
        // colour target in place of the swapchain.
        void addUpscaleAndPostPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex,
                                     VkExtent2D ext, VkExtent2D ptExt,
                                     uint32_t exposureBits, float preExp,
                                     bool secondary = false);

        // The command buffer this frame's recording continues in after the
        // graph (overlay, HUD render() calls, the present transition).
        VkCommandBuffer frameCb() const {
            return asyncFrame_ ? asyncTailCmdBuffers_[currentFrame] : cmdBuffers[currentFrame];
        }
        void destroyAsyncPilotResources();

        void saveViewRecordState(ViewRecordState& s);
        // Also silences the GPU timestamps for a secondary view: one query
        // pool per frame-in-flight, one slot pair per pass, and a secondary
        // re-running those passes would overwrite the primary's.
        void applyViewRecordState(const ViewRecordState& s);
        // The "view" pass that applies viewRecordStates_[index] at record time.
        void addViewEnterPass(vulkan::rg::RenderGraph& g, size_t index);

        // ── The water's mirror view (no ray query; VulkanCoreFrame.cpp) ──────
        // OceanState::waterMirrorView_. The UBOs of deferred binding 94, made
        // once; the view's lifecycle, at the frame boundary (adds, re-sizes or
        // removes it); and its G-buffer passes, inside the primary's run of
        // passes and ahead of the shade that reads them.
        void ensureWaterMirrorUbos();
        void updateWaterMirror(Camera& camera);
        void addWaterMirrorPasses(vulkan::rg::RenderGraph& g);
        // Execute a graph; with THREEPP_LOG=vulkan.rendergraph:trace also print its passes and
        // barriers to stderr whenever they differ from what was last printed
        // under the same name (a pass appearing, a layout changing).
        void executeGraph(VkCommandBuffer cb, vulkan::rg::RenderGraph& graph, const char* name) {
            executeGraph(&cb, nullptr, 0, graph, name);
        }
        // The same over consecutive command buffers of one queue
        // (RenderGraph::execute with splits; the async-compute pilot).
        void executeGraph(const VkCommandBuffer* cbs, const uint32_t* splits, uint32_t splitCount,
                          vulkan::rg::RenderGraph& graph, const char* name) {
            // Which imported images share memory (TransientPool); the graph
            // orders the hand-over between them and checks their lifetimes.
            if (transientPool_) transientPool_->declare(graph);
            if (gbufMsPool_) gbufMsPool_->declare(graph);
            if (splitCount == 0) graph.execute(cbs[0]);
            else graph.execute(cbs, splits, splitCount);
            // Diagnostics (two-layout declarations, overlapping aliases): every
            // occurrence is counted, each distinct message printed and kept once.
            for (auto& e : graph.diagnostics()) {
                ++rgDiagnosticCount_;
                if (rgDiagnosticsReported_.insert(e).second) {
                    THREEPP_LOGF(vulkan::kLogRenderGraph, Warn, "'%s': %s", name, e.c_str());
                    rgDiagnostics_.push_back(std::move(e));
                }
            }
            if (!vulkan::kLogRenderGraph.enabled(log::Level::Trace)) return;
            std::string text = graph.dump();
            auto& last = rgDumped_[name];
            if (text == last) return;
            // One record per line of the dump.
            THREEPP_LOGF(vulkan::kLogRenderGraph, Trace, "graph '%s' (frame serial %llu):", name,
                         static_cast<unsigned long long>(frameSerial_));
            for (size_t b = 0; b < text.size();) {
                const size_t e = std::min(text.find('\n', b), text.size());
                THREEPP_LOG(vulkan::kLogRenderGraph, Trace) << std::string_view(text).substr(b, e - b);
                b = e + 1;
            }
            last = std::move(text);
        }
        // The primary's tail as graph passes: the field transmittance prepass,
        // the billboard glow, the hybrid overlay and the sensor image stage.
        void addTailPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // A secondary view's share of the tail: its field billboards (and their
        // transmittance prepass, for its own eye) onto its colour target.
        void addSecondaryFieldBillboardPasses(vulkan::rg::RenderGraph& g);
        [[nodiscard]] bool hybridOverlayActive() const {
            return overlayWireframePipeline != VK_NULL_HANDLE && sceneHasOverlayContent();
        }
        [[nodiscard]] bool fieldBillboardGlowActive() const;
        // The overlay depth attachment: the MSAA overlay depth, or unjitDepth.
        [[nodiscard]] const Image2D& overlayDepthImage();
        void recordHybridOverlay(VkCommandBuffer cb, uint32_t imageIndex, bool stampActive);
        // The primary view's passes, from the frame head to the sensor stage
        // (or, in the hybrid debug view / events-only mode, the head, the
        // G-buffer and that mode's swapchain pass).
        void addPrimaryViewPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);

        // Records the ImGui (or any overlay) callback inside a dynamic render
        // pass, then transitions the swapchain image GENERAL → PRESENT_SRC.
        // When no overlay callback is set, just emits the GENERAL → PRESENT_SRC
        // barrier directly. Called from endFrame() immediately before
        // vkEndCommandBuffer + submit.
        // The frame's record tail after the secondary views, as one graph:
        // scene capture, frame interop, the event camera, the view composite.
        // The swapchain enters and leaves it in GENERAL, where the screen-space
        // sprite overlay, HUD render() calls and endFrame's ImGui + present
        // transition (all self-synchronising, outside any graph) expect it.
        void addPostViewTailPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // Snapshot the post-TAA swapchain image into the host-visible
        // sceneCaptureBuf_ before any overlay (sprites, ImGui) composites.
        // Allocates / resizes the buffer lazily on first use or swapchain
        // resize, at build time.
        void addSceneCapturePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);

        // ── Event-camera shade compute ─────────────────────────────────
        // Creates the event_shade compute pipeline + descriptor set
        // layout. Called once, lazily, on first setEventCameraEnabled.
        void createEventShadePipeline();

        // (Re-)allocate eventLumaBuf_ at the swapchain dimensions.
        // Called from setEventCameraEnabled and on resize.
        void allocateEventLumaBuffer(uint32_t w, uint32_t h);

        // The event_shade dispatch and the detector, as two passes of the
        // post-view tail graph (after the gbuf prepass AND after the swapchain
        // holds the final frame). The event_shade set is rewritten at build
        // time, so its declaration (reflected) names this frame's gbuf images
        // (SHADER_READ_ONLY_OPTIMAL) and the acquired swapchain image
        // (GENERAL, the Final source's storage image).
        void addEventCameraPasses(vulkan::rg::RenderGraph& g, uint32_t frame, uint32_t imageIndex);

        void recordOverlayAndPresentTransition(VkCommandBuffer cb, uint32_t imageIndex);

        // Common cmd-buffer-begin epilogue used by both the deferred-render
        // and ortho-only frame starts: opens the command buffer and resets
        // the per-frame timestamp pool. Caller must have already acquired
        // the swap image, reset the fence, and reset the cmd buffer.
        void beginCommandRecording(VkCommandBuffer cb);

        // Acquire next swapchain image, run all per-frame UBO / descriptor
        // uploads, open the cmd buffer, and record the full deferred-render
        // body. On exit, the swapchain image is in GENERAL and the cmd
        // buffer is open; endFrame() finishes it (overlay + present-src +
        // submit + present). Returns false on swapchain OUT_OF_DATE —
        // caller leaves state Idle.
        bool beginDeferredFrame(Object3D& scene, Camera& camera);

        // ── Multi-view API (see ViewContext) ───────────────────────────────
        // addView stands up a COMPLETE second deferred chain — G-buffer,
        // history, camera UBOs, descriptor sets, its own TaaResolve /
        // BloomPass / PostComposite / DeferredShade, and the colour target it
        // resolves into. Eagerly, all of it: a view is a persistent object, so
        // the cost is paid once here rather than as a stall on the frame that
        // first happens to need it. Returns a handle, or 0 on failure.
        //
        // Views are NOT churned per frame. Adding one drains the device and
        // allocates; rendering an existing one costs a second pass over the
        // shared scene and nothing else.
        // renderWidth/Height: where the view rasterizes and shades, when it
        // is to be upsampled to width x height by its temporal resolve (the
        // fisheye rig's side views follow the primary's render scale). 0 =
        // native, as every public addView view is.
        uint32_t addViewImpl(Camera& camera, uint32_t width, uint32_t height,
                             uint32_t renderWidth = 0u, uint32_t renderHeight = 0u);
        bool     removeViewImpl(uint32_t handle);
        bool     setViewCameraImpl(uint32_t handle, Camera& camera);
        ViewContext* findView(uint32_t handle);
        // TAA switched off for the CURRENT view (setViewTaa). A secondary
        // never upscales, so its flag is final; the primary can only drop TAA
        // while no upscaler runs, because DLSS and FSR reconstruct from the
        // jitter. Read by both camera UBO uploads (the raster jitter gate)
        // and by the two resolve calls (blend alpha 1 = passthrough).
        [[nodiscard]] bool viewTaaOff() const {
            return !view().taaEnabled && (view().secondary || !(useFsr() || useDlss()));
        }
        // Allocate / free everything a SECONDARY view owns. Both assume the
        // device is idle.
        void createSecondaryViewResources(ViewContext& v);
        void destroySecondaryViewResources(ViewContext& v);
        // Drain, then service every view marked pendingCreate / pendingDestroy.
        // Called at the frame boundary; a no-op (and harmlessly retried next
        // frame) until the shared render pass exists.
        void applyPendingViewChanges();
        // Every secondary view's chain, as passes after the primary's in the
        // frame graph. Shares the frame's TLAS/BLAS, lights, materials and
        // textures; re-runs only what is genuinely per-camera. Returns whether
        // any view added passes (the tail then re-enters the primary).
        bool addSecondaryViewPasses(vulkan::rg::RenderGraph& g);
        // THREEPP_VK_CPU_VIEWS: the secondary views' frame prep (camera UBOs,
        // cull, draw records, shadow maps), started on the pool for every
        // view at once, beside the primary's graph build; joined, with their
        // outputs taken in view order, by addSecondaryViewPasses. False when
        // this frame's views must prepare one after the other there instead
        // (a view's first frame, a shadow map size change: work that creates
        // resources, which is the frame thread's to do).
        bool startSecondaryViewPrep(Camera& primaryCamera);
        void joinSecondaryViewPrep();
        // Run `fn` once per LIVE view — the primary and every secondary that
        // already owns its resources — with curView_ pointed at it.
        //
        // The per-view helpers (rewriteDeferredDescriptors, clearGbufImages,
        // ...) all read view(), which made them silently primary-only at the
        // call sites that are actually SCENE-wide: a rebuilt TLAS, a swapped
        // material texture, a new environment map. A secondary left out of
        // those keeps a descriptor pointing at freed memory — it does not
        // crash, it just renders the world as it was before, or as nothing.
        //
        // Assumes the device is idle, or that the caller knows the slot it is
        // rewriting is fence-proven idle: same precondition the single-view
        // call always had.
        template<class F>
        void forEachLiveView(F&& fn) {
            ViewContext* saved = curView_;
            for (auto& v : views_) {
                if (v->pendingCreate || v->pendingDestroy) continue;
                curView_ = v.get();
                fn();
            }
            curView_ = saved;
        }
        // Copy every DISPLAYED secondary view's colour target into the primary's
        // swapchain image, at that view's rect. A pass after the secondaries
        // have resolved and after the scene capture (which must stay a clean
        // picture of the primary alone), and before the overlay, so ImGui and
        // sprites still draw on top.
        void addViewCompositePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        bool setViewDisplayRectImpl(uint32_t handle, int x, int y, int w, int h);
        // Per-view permission to rasterize sensor-only surfaces. Raster state
        // only — read while the draw list is built, so it takes effect on the
        // next frame with nothing to invalidate. The lidar's TLAS masks are a
        // separate, scene-level decision (sensorOnlySurfaces_).
        bool setViewSensorSurfacesImpl(uint32_t handle, bool enabled);
        bool setViewSplatsImpl(uint32_t handle, bool enabled);
        // Copy a secondary view's finished colour target to host memory as
        // tightly-packed RGB8, top-down (matching readRGBPixels — the Vulkan
        // readback is already top-down and must NOT be flipped).
        std::vector<unsigned char> readViewPixelsImpl(uint32_t handle);

        // Ortho-only frame start: no deferred render, no per-frame uploads.
        // Acquires the swap image, opens the cmd buffer, and clears the
        // swapchain to the configured clearColor (leaves layout = GENERAL).
        // recordOrthoOverlay() then draws the HUD scene atop this cleared image.
        bool beginFrameOrthoOnly();

        // Finish the open frame: record overlay/ImGui pass + PRESENT_SRC
        // transition, close the cmd buffer, submit and present, advance the
        // frame slot, bump sampleIndex. No-op when no frame is open.
        // Registered as Canvas frame-end callback (fires after the user's
        // animate lambda returns); also invoked from render() when the user
        // unexpectedly starts a second perspective render mid-frame.
        void endFrame();

        // State-machine dispatcher for VulkanRenderer::render(). First call
        // of a user animate iteration runs beginFrame*, subsequent calls
        // either append HUD overlay draws (ortho path) or finalize
        // the prior frame and restart (defensive — second perspective call).
        // endFrame() runs from the Canvas frame-end callback at the tail of
        // animateOnce().
        void renderFrame(Object3D& scene, Camera& camera);
    };

} // namespace threepp
