// StateCommon: the shared prelude of the VulkanRenderer::Impl state groups (state/*.hpp).
//
// The headers the state groups' member types come from. The same list
// VulkanCoreImpl.hpp includes (it includes this one first), spelled from this
// folder, so each state header compiles on its own.
#pragma once

#include "../CpuPar.hpp"
#include "../EnvKnob.hpp"
#include "../VulkanContext.hpp"
#include "../VulkanResources.hpp"
#include "../VulkanRetireQueue.hpp"
#include "../VulkanImplCommon.hpp"
#include "../VulkanGpuLayouts.hpp"
#include "../VulkanGeometryState.hpp"
#include "../VulkanSceneTypes.hpp"
#include "../EdgeRefine.hpp"
#include "../VulkanViewContext.hpp"
#include "../EnvPrefilter.hpp"
#include "../SkyBake.hpp"
#include "../SceneEnvKnobs.hpp"
#include "../EventCameraDetector.hpp"
#include "../LidarScanner.hpp"
#include "../SkinningPipeline.hpp"
#include "../TetSkinningPipeline.hpp"
#include "../GpuTimings.hpp"
#include "../OverlayPass.hpp"
#include "../VulkanFrameTypes.hpp"
#include "../RenderGraph.hpp"
#include "../TransientPool.hpp"
#include "../TaaResolve.hpp"
#if defined(THREEPP_WITH_FSR)
#include "../FsrUpscaler.hpp"
#endif
#if defined(THREEPP_WITH_DLSS)
#include "../DlssUpscaler.hpp"
#endif
#include "../AutoExposure.hpp"
#include "../GbufResolve.hpp"
#include "../BillboardGlowPass.hpp"
#include "../BillboardHdrLayer.hpp"
#include "../BillboardSoftDepth.hpp"
#include "../BloomPass.hpp"
#include "../PostComposite.hpp"
#include "../DofPass.hpp"
#include "../SensorPass.hpp"
#include "../SplatPass.hpp"
#include "../DeferredShade.hpp"
#include "../HiZPyramid.hpp"
#include "../InstanceExpand.hpp"
#include "../DrawGen.hpp"
#include "../TlasGen.hpp"
#include "../OcclusionCull.hpp"
#include "../ParticleFieldPass.hpp"
#include "../PathTracer.hpp"
#include "../ProbeGI.hpp"
#include "../WaterDisplacePipeline.hpp"
#include "../FoamWorldPipeline.hpp"
#include "../WakeFieldPipeline.hpp"
#include "../GrassWindPipeline.hpp"
#include "../VertexSanitizePipeline.hpp"
#include "../shaders/downwash_shared.h"
#include "../shaders/vulkan_shared.h"
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

namespace threepp::vulkan::impl {

        // ── Per-slot dirty state for the entries-indexed desc rings ──────────
        // What one frame-in-flight slot still owes the GPU: either the whole
        // array, or a set of half-open element ranges. Ranges matter because
        // the GeometryDesc array is entries-indexed and large (until the
        // material table, the MaterialDescs were too: 47.7 MB at 78.4k
        // entries), so a whole-array flush for a small per-frame change costs
        // milliseconds of CPU. The material table's ranges are record ranges.
        //
        // Per slot, not global: a change dirtied on frame N must reach every
        // slot's buffer, but each slot may only be written after its own fence
        // has signalled, so each slot accumulates independently and is cleared
        // alone when its turn comes.
        struct DescDirtyRanges {
            // Half-open [first, last) entry ranges, kept sorted and coalesced.
            std::vector<std::pair<uint32_t, uint32_t>> ranges;
            bool whole = false;// structural rebuild / unidentifiable source
            uint32_t maxRanges = 64;// past this many, one whole-array flush

            bool any() const { return whole || !ranges.empty(); }
            void clear() {
                whole = false;
                ranges.clear();
            }
            void markWhole() {
                whole = true;
                ranges.clear();// subsumed
            }
            // The scene walk marks in increasing entry order, so coalescing
            // against the last range collapses a contiguous instanced span into
            // one range. Marks also arrive out of order: a slot not flushed last
            // frame still holds that frame's ranges when the next walk starts
            // again from entry 0. A mark that overlaps or touches the last range
            // widens it at either end; any other is a range of its own (the
            // upload copies each range, so an overlap with an older one is
            // harmless). Until 2026-10-15 a mark below the last range only
            // raised its end and was lost: an edit to an earlier entry the frame
            // after an edit to a later one never showed (VulkanInstanceColor_test,
            // phase 3).
            void mark(uint32_t first, uint32_t count = 1) {
                if (whole || count == 0) return;
                const uint32_t last = first + count;
                if (!ranges.empty() && first <= ranges.back().second && last >= ranges.back().first) {
                    ranges.back().first  = std::min(ranges.back().first, first);
                    ranges.back().second = std::max(ranges.back().second, last);
                    return;
                }
                // Many ranges suggest a scene-wide change; fall back to one
                // whole-array flush (broader than necessary, never wrong).
                if (ranges.size() >= maxRanges) {
                    markWhole();
                    return;
                }
                ranges.emplace_back(first, last);
            }
        };

}// namespace threepp::vulkan::impl
