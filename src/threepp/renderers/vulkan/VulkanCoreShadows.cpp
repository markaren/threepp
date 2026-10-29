// Shadows: the mode knob (VulkanRenderer::setShadowMode) and the shadow-map
// path's host side — the always-bound stand-ins for bindings 80-83 of the
// deferred set, and the per-view descriptor upkeep.

#include "VulkanCoreImpl.hpp"

#include "CpuPar.hpp"
#include "VulkanCpuPhaseProf.hpp"

#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/Sphere.hpp"

#include "threepp/renderers/vulkan/shaders/shadow_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_depth.vert.spv.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace threepp {

    VulkanRenderer::ShadowMode VulkanRenderer::Impl::shadowModeFromEnv() {
        const char* e = std::getenv("THREEPP_VK_SHADOW_MODE");
        if (!e) return VulkanRenderer::ShadowMode::Auto;
        const std::string v = e;
        if (v == "rt") return VulkanRenderer::ShadowMode::RayTraced;
        if (v == "sm") return VulkanRenderer::ShadowMode::ShadowMaps;
        if (v == "off") return VulkanRenderer::ShadowMode::Off;
        if (v != "auto") {
            std::fprintf(stderr, "[VulkanRenderer] THREEPP_VK_SHADOW_MODE=%s: expected auto, rt, sm or off; "
                                 "using auto\n", e);
        }
        return VulkanRenderer::ShadowMode::Auto;
    }

    uint32_t VulkanRenderer::Impl::shadowMapSizeCapFromEnv() {
        return static_cast<uint32_t>(vulkan::env::integer("THREEPP_VK_SHADOW_MAPSIZE").value_or(0));
    }

    // Ablation knobs for the map pass's cost: THREEPP_VK_SHADOW_CASCADES=<1..4>,
    // THREEPP_VK_SHADOW_LAMPS=<spots>,<points>. The setters still win.
    uint32_t VulkanRenderer::Impl::shadowCascadesFromEnv() {
        const auto n = vulkan::env::integer("THREEPP_VK_SHADOW_CASCADES");
        if (!n) return 4u;
        return std::clamp(static_cast<uint32_t>(*n), 1u, static_cast<uint32_t>(kMaxShadowCascades));
    }

    uint32_t VulkanRenderer::Impl::shadowedLampsFromEnv(int which) {
        const char* e = std::getenv("THREEPP_VK_SHADOW_LAMPS");
        if (!e || !*e) return which == 0 ? 4u : 8u;
        char*          end   = nullptr;
        const uint32_t spots = static_cast<uint32_t>(std::strtoul(e, &end, 10));
        uint32_t       pts   = 8u;
        if (end && *end == ',') pts = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
        return which == 0 ? std::min(spots, static_cast<uint32_t>(kMaxShadowSpots))
                          : std::min(pts, static_cast<uint32_t>(kMaxShadowPoints));
    }

    bool VulkanRenderer::Impl::shadowCacheFromEnv() {
        return vulkan::env::flag("THREEPP_VK_SHADOW_CACHE", true);
    }

    float VulkanRenderer::Impl::shadowCacheMarginFromEnv() {
        const auto v = vulkan::env::number("THREEPP_VK_SHADOW_CACHE_MARGIN");
        if (!v) return 1.15f;
        return std::clamp(static_cast<float>(*v), 1.f, 2.f);
    }

    // A caster that may move between two frames: a deformer or interop mesh
    // (its bounds are stale too), grass (it sways), or a mesh moved within the
    // last kMovedStickyFrames (the G-buffer's own notion of "moving").
    bool VulkanRenderer::Impl::shadowCasterDynamic(uint32_t entry) const {
        const MeshEntry& e = lastVisibleEntries_[entry];
        if (e.isSkinned || e.isDisplaced || e.isMorphed || e.isTet || e.isVertexInterop || e.isGrass) return true;
        return entry < meshMovedSticky_.size() && meshMovedSticky_[entry] > 0u;
    }

    // The static caster set as a number: every span's hidden flag and every
    // entry's static/dynamic bit, FNV-1a over the entry walk. A frozen cascade
    // box is valid for one value of it: a mesh that starts or stops moving, is
    // hidden or shown, or a scene rebuild, changes it and redraws the cache.
    uint64_t VulkanRenderer::Impl::shadowStaticHash() {
        uint64_t h = 1469598103934665603ull;
        const auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix(static_cast<uint64_t>(lastVisibleEntries_.size()));
        // A crossfade handing the shadow to its other level changes the
        // static casters without moving anything.
        mix(static_cast<uint64_t>(lodFadeCastEpoch_));
        shadowDynamicIdx_.clear();
        shadowSpanOfEntry_.assign(lastVisibleEntries_.size(), 0u);
        for (size_t s = 0; s < entrySpans_.size(); ++s) {
            const auto& sp = entrySpans_[s];
            mix(sp.hidden ? 0x9e3779b97f4a7c15ull : 0x7f4a7c159e3779b9ull);
            if (sp.hidden) continue;
            for (uint32_t j = 0; j < sp.count; ++j) {
                const uint32_t i   = sp.first + j;
                const bool     dyn = shadowCasterDynamic(i);
                mix(static_cast<uint64_t>(i) * 2ull + (dyn ? 1ull : 0ull));
                if (i < shadowSpanOfEntry_.size()) {
                    shadowSpanOfEntry_[i] = static_cast<uint32_t>(s);
                    if (dyn) shadowDynamicIdx_.push_back(i);
                }
            }
        }
        return h;
    }

    // The stale-tolerant cascades' per-entry bits (ShadowMapPass::casterSnap),
    // brought up to this frame before the cascades are fitted: a caster that
    // left the static set since its cascade was drawn clears its bits, and the
    // cascades whose box it reached are stale from now; the casters to draw
    // over the caches are listed in span order.
    void VulkanRenderer::Impl::updateShadowCasterSnap(vulkan::ShadowMapPass& smp, uint32_t cascades) {
        const size_t   n   = lastVisibleEntries_.size();
        const uint64_t now = std::max<uint64_t>(frameSerial_, 1u);
        const auto     leave = [&](uint8_t s) {
            for (uint32_t c = 0; c < cascades; ++c)
                if (((s >> (4u + c)) & 1u) && smp.staleSince[c] == 0u) smp.staleSince[c] = now;
        };
        // A structural rebuild numbers the entries anew: the bits would name
        // other meshes. Slot edits keep the numbering.
        if (smp.snapLayout != tlasStats_.structuralRebuilds) {
            smp.snapLayout = tlasStats_.structuralRebuilds;
            smp.casterSnap.assign(n, 0u);
            smp.staleSince.fill(0u);
            for (auto& c : smp.cascadeCache) c.valid = false;
        }
        if (smp.casterSnap.size() > n) {
            for (size_t i = n; i < smp.casterSnap.size(); ++i) leave(smp.casterSnap[i]);
            smp.casterSnap.resize(n);
        } else {
            smp.casterSnap.resize(n, 0u);
        }
        const uint8_t full = static_cast<uint8_t>((1u << cascades) - 1u);
        smp.onTop.clear();
        for (const auto& sp : entrySpans_) {
            for (uint32_t j = 0; j < sp.count; ++j) {
                const uint32_t i = sp.first + j;
                if (i >= n) break;
                uint8_t&   s     = smp.casterSnap[i];
                const bool dyn   = shadowCasterDynamic(i);
                const bool casts = !sp.hidden && (i >= entryLodFade_.size() || lodFadeCasts(entryLodFade_[i]));
                if (s != 0u && (dyn || !casts)) {
                    leave(s);
                    s = 0u;
                }
                if (!sp.hidden && (dyn || (casts && (s & full) != full))) smp.onTop.push_back(i);
            }
        }
    }

    VulkanRenderer::ShadowMode VulkanRenderer::Impl::activeShadowMode() const {
        using M = VulkanRenderer::ShadowMode;
        switch (shadowModeReq_) {
            case M::Off: return M::Off;
            case M::ShadowMaps: return M::ShadowMaps;
            default: return ctx->rayQuerySupported() ? M::RayTraced : M::ShadowMaps;
        }
    }

    void VulkanRenderer::Impl::ensureShadowDummies() {
        if (shadowDummyUbo_.handle != VK_NULL_HANDLE) return;
        shadowDummyArray_ = vulkan::createShadowDepthArray(*ctx, 1, 1, false, false, "shadowDummy.array");
        shadowDummyCube_  = vulkan::createShadowDepthArray(*ctx, 1, 6, true, false, "shadowDummy.cube");
        shadowDummyUbo_   = createBuffer(ctx->allocator(), ctx->device(), sizeof(GpuShadowMapsUbo),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        const GpuShadowMapsUbo zero{};
        uploadHostVisible(ctx->allocator(), shadowDummyUbo_, &zero, sizeof(zero));
        // Cleared to 0 (the far plane: nothing cast) and left where the
        // deferred set's descriptors say they rest.
        VkCommandBuffer cb = beginOneShot();
        for (VkImage img : {shadowDummyArray_.image, shadowDummyCube_.image}) {
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            range.levelCount = 1;
            range.layerCount = VK_REMAINING_ARRAY_LAYERS;
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = img;
            b.subresourceRange    = range;
            b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
            const VkClearDepthStencilValue clear{0.f, 0u};
            vkCmdClearDepthStencilImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
        }
        endAndSubmitOneShot(cb, "shadow map dummies");
    }

    void VulkanRenderer::Impl::destroyShadowResources() {
        if (shadowPipeline_) vkDestroyPipeline(ctx->device(), shadowPipeline_, nullptr);
        if (shadowDepthPipeline_) vkDestroyPipeline(ctx->device(), shadowDepthPipeline_, nullptr);
        shadowPipeline_      = VK_NULL_HANDLE;
        shadowDepthPipeline_ = VK_NULL_HANDLE;
        vulkan::destroyShadowDepthArray(*ctx, shadowDummyArray_);
        vulkan::destroyShadowDepthArray(*ctx, shadowDummyCube_);
        destroyBuffer(ctx->allocator(), shadowDummyUbo_);
    }

    VulkanRenderer::Impl::ShadowBinding VulkanRenderer::Impl::shadowBindingFor(uint32_t frame) const {
        ShadowBinding b;
        b.cascades = shadowDummyArray_.view;
        b.spots    = shadowDummyArray_.view;
        b.points   = shadowDummyCube_.view;
        b.ubo      = shadowDummyUbo_.handle;
        const auto& v = *curView_;
        if (!v.shadowLive || !v.shadowMaps_) return b;
        const auto& s = v.shadowMaps_->slot(frame);
        if (s.gen == 0) return b;
        if (s.cascades.view) b.cascades = s.cascades.view;
        if (s.spots.view) b.spots = s.spots.view;
        if (s.points.view) b.points = s.points.view;
        b.ubo = s.ubo.handle;
        b.gen = s.gen;
        return b;
    }

    void VulkanRenderer::Impl::syncShadowDescriptors(uint32_t frame) {
        if (!view().deferredShade_) return;
        ensureShadowDummies();
        const ShadowBinding b = shadowBindingFor(frame);
        if (view().shadowBoundGen[frame] == b.gen) return;
        view().deferredShade_->rewriteShadowMaps(frame, b.cascades, b.spots, b.points, b.ubo);
        view().shadowBoundGen[frame] = b.gen;
    }

    void VulkanRenderer::Impl::prepareShadowMaps(Camera& camera) {
        view().shadowLive = shadowMapsActive() && view().deferredShade_ && rasterDsLayout != VK_NULL_HANDLE &&
                            fitShadowMaps(camera);
        syncShadowDescriptors(currentFrame);
    }

    // ── Lamp slots ───────────────────────────────────────────────────────────
    // Once a frame from the primary camera, while the maps are in use: per
    // type, the lamps with castShadow == true whose range reaches the camera's
    // frustum and that are lit, by power over the squared distance to the
    // camera. A lamp keeps its slot (and its layer) while it stays eligible,
    // unless a newcomer is twice as strong: no slot flicker between two
    // near-equal lamps.
    //
    // castShadow is the gate, as in three.js, not merely a priority. The
    // slots used to be handed out to every lamp in range, castShadow first,
    // which gave a shadowless lamp a map whenever a slot was free — and a map
    // is the scene drawn again into six cube faces every frame it is lit. A
    // muzzle flash (PointLight, castShadow false, 12 m) in a 15k-entry forest
    // cost 10 ms of GPU and 1 ms of CPU per lit frame that way; the game had
    // asked for no shadow at all. A dark lamp (intensity 0, or a flash
    // between rounds) is skipped too: a map of nothing lights nothing.
    void VulkanRenderer::Impl::assignShadowSlots(std::vector<GpuClusterLight>& recs, const std::vector<Light*>& src,
                                                 Camera& camera) {
        const bool active = shadowMapsActive();
        Frustum    frustum;
        if (active) {
            Matrix4 vp;
            vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            frustum.setFromProjectionMatrix(vp);
        }
        const auto&   mw = camera.matrixWorld->elements;
        const Vector3 camPos(mw[12], mw[13], mw[14]);
        struct Cand {
            uint32_t rec;
            float    score;
            bool     taken;
        };
        const auto better = [](const Cand& a, const Cand& b) { return a.score > b.score; };
        for (int spot = 0; spot < 2; ++spot) {
            auto&          slots = spot ? shadowSpotSlots_ : shadowPointSlots_;
            const uint32_t cap   = active ? (spot ? shadowedSpots_ : shadowedPoints_) : 0u;
            std::vector<Cand> cands;
            for (uint32_t r = 0; r < recs.size() && cap > 0u; ++r) {
                const auto& g = recs[r];
                if ((g.type > 0.5f) != (spot == 1) || g.cullRadius <= 0.f) continue;
                if (!src[r]->castShadow) continue;
                const float lum = 0.2126f * g.color[0] + 0.7152f * g.color[1] + 0.0722f * g.color[2];
                if (lum <= 0.f) continue;
                const Vector3 p(g.position[0], g.position[1], g.position[2]);
                if (!frustum.intersectsSphere(Sphere(p, g.cullRadius))) continue;
                cands.push_back({r, lum / std::max(p.distanceToSquared(camPos), 1e-2f), false});
            }
            std::stable_sort(cands.begin(), cands.end(), better);

            std::vector<ShadowLamp> next(cap);
            std::vector<Cand>       held(cap);
            // Holders that are still eligible keep their slot.
            for (uint32_t k = 0; k < cap && k < slots.size(); ++k) {
                if (!slots[k].used) continue;
                for (auto& c : cands) {
                    if (c.taken || src[c.rec]->id != slots[k].id) continue;
                    c.taken = true;
                    held[k] = c;
                    next[k].used = true;
                    break;
                }
            }
            // Newcomers fill the free slots, then may displace the weakest holder.
            for (auto& c : cands) {
                if (c.taken) continue;
                int k = -1;
                for (uint32_t i = 0; i < cap; ++i) {
                    if (!next[i].used) {
                        k = static_cast<int>(i);
                        break;
                    }
                }
                if (k < 0) {
                    int weak = -1;
                    for (uint32_t i = 0; i < cap; ++i) {
                        if (weak < 0 || better(held[weak], held[i])) weak = static_cast<int>(i);
                    }
                    const Cand& w = held[weak];
                    if (!(c.score > 2.f * w.score)) break;
                    k = weak;
                }
                c.taken      = true;
                held[k]      = c;
                next[k].used = true;
            }
            for (uint32_t k = 0; k < cap; ++k) {
                if (!next[k].used) continue;
                auto&       g     = recs[held[k].rec];
                Light*      light = src[held[k].rec];
                ShadowLamp& L     = next[k];
                g.shadowSlot      = static_cast<float>(k);
                L.id              = light->id;
                std::memcpy(L.pos, g.position, sizeof(L.pos));
                std::memcpy(L.dir, g.direction, sizeof(L.dir));
                float range = g.range;
                L.fov       = 0.f;
                if (auto* sl = dynamic_cast<SpotLight*>(light)) {
                    const auto  ss    = std::dynamic_pointer_cast<SpotLightShadow>(sl->shadow);
                    const float focus = ss ? ss->focus : 1.f;
                    L.fov             = std::clamp(2.f * sl->angle * focus, 0.02f, 3.1f);
                }
                L.far = range > 0.f ? range : g.cullRadius;
                if (auto* ws = dynamic_cast<LightWithShadow*>(light); ws && ws->shadow) {
                    L.bias       = ws->shadow->bias;
                    L.normalBias = ws->shadow->normalBias;
                    L.radius     = ws->shadow->radius;
                    L.intensity  = ws->shadow->intensity;
                }
            }
            slots = std::move(next);
        }
    }

    namespace {

        float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

        // world -> clip of a light camera whose matrixWorld is `world`, with
        // the GL projection `glProj` made reverse-Z (the raster camera's own
        // conversion, reverseZVk).
        void lightClip(float* out, const Matrix4& world, const Matrix4& glProj) {
            Matrix4 view = world;
            view.invert();
            Matrix4 proj = glProj;
            auto&   e    = proj.elements;
            for (int c = 0; c < 4; ++c) e[c * 4 + 2] = 0.5f * e[c * 4 + 3] - 0.5f * e[c * 4 + 2];
            Matrix4 vp;
            vp.multiplyMatrices(proj, view);
            std::memcpy(out, vp.elements.data(), 64);
        }

    }// namespace

    bool VulkanRenderer::Impl::fitShadowMaps(Camera& camera) {
        const uint32_t f   = currentFrame;
        auto&          v   = view();
        const uint32_t cap = shadowMapSizeCap_ ? std::min(shadowMapSizeCap_, 8192u) : 8192u;
        const auto     clampSize = [cap](uint32_t s) { return std::clamp(s, 16u, cap); };

        vulkan::ShadowMapPass::Sizes sz{};
        if (shadowSun_.valid) {
            sz.cascadeSize = clampSize(shadowSun_.mapSize);
            sz.cascades    = shadowCascades_;
        }
        const auto anyUsed = [](const std::vector<ShadowLamp>& v) {
            for (const auto& L : v)
                if (L.used) return true;
            return false;
        };
        if (anyUsed(shadowSpotSlots_)) {
            sz.spotSize = clampSize(1024u);
            sz.spots    = static_cast<uint32_t>(shadowSpotSlots_.size());
        }
        if (anyUsed(shadowPointSlots_)) {
            sz.pointSize = clampSize(512u);
            sz.points    = static_cast<uint32_t>(shadowPointSlots_.size());
        }
        if (sz.cascades == 0u && sz.spots == 0u && sz.points == 0u) return false;

        if (!v.shadowMaps_) {
            v.shadowMaps_ = std::make_unique<vulkan::ShadowMapPass>(
                    *ctx, kFramesInFlight, rasterDsLayout, transientPool_.get(), v.id,
                    [this](Image2D&& i) { retire(std::move(i)); });
        }
        if (shadowPipeline_ == VK_NULL_HANDLE) createShadowPipeline();
        auto& smp = *v.shadowMaps_;
        smp.ensure(f, sz);
        smp.layers.clear();

        GpuShadowMapsUbo  u{};
        ShadowCullVolumes vol;
        // The view's frustum: a lamp layer none of it can see is cleared and
        // drawn into by nothing (cone and face volumes below).
        Frustum viewFrustum;
        {
            camera.updateMatrixWorld();
            Matrix4 vp;
            vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            viewFrustum.setFromProjectionMatrix(vp);
        }

        // ── The sun's cascades ───────────────────────────────────────────────
        // Practical split (lambda 0.75) of [near, distance] into N slices; each
        // slice's bounding sphere (its size depends on the camera's projection
        // alone, so it holds still while the camera turns) is an ortho box of
        // half extent r around its centre in the sun's frame, the centre
        // snapped to the texel grid so the map does not shimmer as the camera
        // moves. The box starts at the sphere's sunward rim; casters beyond it
        // are pancaked onto the near plane by depth clamp (or, without it, the
        // near plane is pulled 1 km toward the sun).
        // The static caster set this frame as a generation, for the caches.
        if (shadowCacheOn_) {
            const uint64_t h = shadowStaticHash();
            if (h != shadowStaticHashPrev_) {
                shadowStaticHashPrev_ = h;
                ++shadowStaticGen_;
                if (shadowCacheDebug_) {
                    const size_t n = lastVisibleEntries_.size();
                    size_t hidden = 0, dyn = 0, flips = 0;
                    std::string first;
                    for (const auto& sp : entrySpans_) hidden += sp.hidden ? 1u : 0u;
                    std::vector<uint8_t> cur(n, 0u);
                    for (size_t i = 0; i < n; ++i) {
                        cur[i] = shadowCasterDynamic(static_cast<uint32_t>(i)) ? 1u : 0u;
                        dyn += cur[i];
                        const uint8_t prev = i < shadowDynPrev_.size() ? shadowDynPrev_[i] : 0u;
                        if (cur[i] != prev) {
                            if (flips < 8) {
                                const MeshEntry& e = lastVisibleEntries_[i];
                                first += " " + std::to_string(i) + (cur[i] ? "->dyn" : "->static") +
                                         (e.isSkinned ? "(skin)" : e.isDisplaced ? "(disp)" : e.isGrass ? "(grass)" : e.isMorphed ? "(morph)" : "") +
                                         "(span " + std::to_string(i < shadowSpanOfEntry_.size() ? shadowSpanOfEntry_[i] : 0u) + ")";
                            }
                            ++flips;
                        }
                    }
                    std::fprintf(stderr, "[shadowCache] gen %llu: entries %zu (prev %zu), hidden spans %zu, dynamic %zu, flips %zu:%s\n",
                                 static_cast<unsigned long long>(shadowStaticGen_), n, shadowDynPrev_.size(), hidden, dyn, flips, first.c_str());
                    shadowDynPrev_ = std::move(cur);
                }
            }
        }
        const uint64_t staticGen = shadowStaticGen_;
        // setShadowCacheStaleFrames: a cascade outlives a changed static set
        // for that many frames, and at most one is redrawn a frame.
        const bool staleTolerant = shadowCacheOn_ && shadowCacheStaleFrames_ > 0u && sz.cascades > 0u;
        if (staleTolerant) updateShadowCasterSnap(smp, sz.cascades);
        bool cascadeRedrawn = false;
        if (sz.cascades > 0u) {
            camera.updateMatrixWorld();
            float camNear = 0.1f, camFar = 1000.f;
            if (auto* pc = camera.as<PerspectiveCamera>()) {
                camNear = pc->nearPlane;
                camFar  = pc->farPlane;
            } else if (auto* oc = camera.as<OrthographicCamera>()) {
                camNear = oc->nearPlane;
                camFar  = oc->farPlane;
            }
            camNear    = std::max(camNear, 1e-3f);
            float dist = shadowDistance_ > 0.f ? shadowDistance_ : std::clamp(camFar, 50.f, 2000.f);
            // ...and no farther than the scene reaches from the camera (the
            // farthest corner of a span's world box), so a room is not
            // shadowed with cascades sized for a kilometre of terrain. A span
            // the frustum cull has not boxed (a deformer) does not count.
            {
                const auto& mw  = camera.matrixWorld->elements;
                float       ext = 0.f;
                bool        any = false;
                for (const auto& sp : entrySpans_) {
                    if (sp.hidden || !sp.aabbValid) continue;
                    float d2 = 0.f;
                    for (int a = 0; a < 3; ++a) {
                        const float e = std::max(std::abs(sp.aabbMin[a] - mw[12 + a]), std::abs(sp.aabbMax[a] - mw[12 + a]));
                        d2 += e * e;
                    }
                    ext = std::max(ext, std::sqrt(d2));
                    any = true;
                }
                if (any) dist = std::min(dist, std::max(ext, 1.f));
            }
            dist = std::max(std::min(dist, camFar), camNear * 2.f);
            const uint32_t n = sz.cascades;
            // The sun's size for the PCF kernel below: the penumbra a caster
            // this high above the receiver gets from it.
            constexpr float kSunPenumbraHeight = 1.f;// metres
            const float     sunTan = std::tan(sunAngularRadiusDeg_ * 0.017453292519943295f);
            float          split[kMaxShadowCascades + 1];
            for (uint32_t i = 0; i <= n; ++i) {
                const float t   = float(i) / float(n);
                const float lg  = camNear * std::pow(dist / camNear, t);
                const float lin = camNear + (dist - camNear) * t;
                split[i]        = 0.75f * lg + 0.25f * lin;
            }
            const auto& P    = camera.projectionMatrix.elements;
            const auto  ndcZ = [&](float d) {// GL NDC z of the view-space point (0, 0, -d)
                const float cz = P[10] * -d + P[14];
                const float cw = P[11] * -d + P[15];
                return cz / cw;
            };
            const Matrix4& projInv  = camera.projectionMatrixInverse;
            const Matrix4& camWorld = *camera.matrixWorld;

            const float* L       = shadowSun_.dir;
            const bool   steep   = std::abs(L[1]) >= 0.99f;
            const float  up0[3]  = {steep ? 1.f : 0.f, steep ? 0.f : 1.f, 0.f};
            // The sun's frame: z toward the sun (the light camera looks along -L).
            float x[3] = {up0[1] * L[2] - up0[2] * L[1], up0[2] * L[0] - up0[0] * L[2],
                          up0[0] * L[1] - up0[1] * L[0]};
            const float xl = std::sqrt(dot3(x, x));
            for (float& c : x) c /= xl;
            const float y[3]  = {L[1] * x[2] - L[2] * x[1], L[2] * x[0] - L[0] * x[2], L[0] * x[1] - L[1] * x[0]};
            const float extra = ctx->depthClampSupported() ? 0.f : 1000.f;
            const float size  = float(sz.cascadeSize);

            for (uint32_t i = 0; i < n; ++i) {
                float corners[8][3];
                int   k = 0;
                for (float d : {split[i], split[i + 1]}) {
                    const float z = ndcZ(d);
                    for (float cy : {-1.f, 1.f}) {
                        for (float cx : {-1.f, 1.f}) {
                            Vector3 p(cx, cy, z);
                            p.applyMatrix4(projInv).applyMatrix4(camWorld);
                            corners[k][0] = p.x;
                            corners[k][1] = p.y;
                            corners[k][2] = p.z;
                            ++k;
                        }
                    }
                }
                float c[3] = {0.f, 0.f, 0.f};
                for (auto& q : corners)
                    for (int a = 0; a < 3; ++a) c[a] += q[a] / 8.f;
                float r = 0.f;
                for (auto& q : corners) {
                    const float d[3] = {q[0] - c[0], q[1] - c[1], q[2] - c[2]};
                    r                = std::max(r, std::sqrt(dot3(d, d)));
                }
                r = std::ceil(r * 16.f) / 16.f;// a stable size, off the float noise
                // The static cache (shadowCacheOn_, ShadowMapPass::cache): the
                // cascade's box is frozen shadowCacheMargin_ times the slice
                // and kept while the slice stays inside it, the sun holds (to
                // 0.05 degrees) and the static caster set is unchanged. Its
                // static casters are then drawn once, copied in each frame,
                // and only the moving casters drawn on top. A camera that
                // leaves the box, a sun that moves or a mesh that starts
                // moving redraws it: one frame at the uncached cost.
                auto&      cc        = smp.cascadeCache[i];
                const bool cacheable = shadowCacheOn_ && smp.cache.image != VK_NULL_HANDLE && i < smp.cache.layers;
                bool genOk = cc.staticGen == staticGen;
                if (staleTolerant) {
                    const uint64_t since = smp.staleSince[i];
                    genOk = since == 0u || frameSerial_ - since < shadowCacheStaleFrames_ || cascadeRedrawn;
                }
                const bool sunOk     = dot3(cc.sunDir, L) >= 0.9999996f;
                bool       boxIn     = false;
                if (cacheable && cc.valid) {
                    // The slice inside the frozen box, to one texel: the box's
                    // centre was snapped to the texel grid, so a still camera's
                    // slice sits up to a texel off it (the receivers pick by
                    // the frozen sphere, so the rim they lose is that texel).
                    const float dc[3] = {c[0] - cc.c[0], c[1] - cc.c[1], c[2] - cc.c[2]};
                    const float dl    = dot3(dc, L);
                    const float lim   = cc.r + cc.texel;
                    boxIn = std::abs(dot3(dc, x)) + r <= lim && std::abs(dot3(dc, y)) + r <= lim && dl - r >= -lim &&
                            dl + r <= lim;
                }
                const bool reuse = cacheable && cc.valid && genOk && sunOk && boxIn;
                if (staleTolerant && cacheable && !reuse) cascadeRedrawn = true;
                if (shadowCacheDebug_ && cacheable && !reuse) {
                    std::fprintf(stderr, "[shadowCache] cascade %u rebuilt:%s%s%s%s slice r %.2f, box r %.2f\n", i,
                                 cc.valid ? "" : " invalid",
                                 cc.valid && !genOk ? (staleTolerant ? " stale" : " casters") : "",
                                 cc.valid && !sunOk ? " sun" : "", cc.valid && !boxIn ? " box" : "",
                                 static_cast<double>(r), static_cast<double>(cc.r));
                }
                float                        texel, far, rBox;
                vulkan::ShadowMapPass::Layer layer{};
                if (reuse) {
                    std::memcpy(c, cc.c, sizeof(c));
                    rBox  = cc.r;
                    texel = cc.texel;
                    far   = cc.far;
                    std::memcpy(layer.vp, cc.vp, 64);
                    layer.mode = vulkan::ShadowMapPass::Mode::Cached;
                } else {
                    rBox  = cacheable ? std::ceil(r * shadowCacheMargin_ * 16.f) / 16.f : r;
                    texel = 2.f * rBox / size;
                    const float cx = dot3(c, x), cy = dot3(c, y);
                    const float sx = std::floor(cx / texel) * texel - cx;
                    const float sy = std::floor(cy / texel) * texel - cy;
                    for (int a = 0; a < 3; ++a) c[a] += x[a] * sx + y[a] * sy;

                    far = 2.f * rBox + extra;
                    Matrix4 world;
                    world.makeBasis(Vector3(x[0], x[1], x[2]), Vector3(y[0], y[1], y[2]), Vector3(L[0], L[1], L[2]));
                    world.setPosition(c[0] + L[0] * (rBox + extra), c[1] + L[1] * (rBox + extra),
                                      c[2] + L[2] * (rBox + extra));
                    Matrix4 proj;
                    proj.makeOrthographic(-rBox, rBox, rBox, -rBox, 0.f, far);
                    lightClip(layer.vp, world, proj);
                    layer.mode = cacheable ? vulkan::ShadowMapPass::Mode::Rebuild : vulkan::ShadowMapPass::Mode::Draw;
                    if (cacheable) {
                        cc.valid      = true;
                        cc.staticGen  = staticGen;
                        cc.r          = rBox;
                        cc.texel      = texel;
                        cc.far        = far;
                        std::memcpy(cc.c, c, sizeof(cc.c));
                        std::memcpy(cc.sunDir, L, sizeof(cc.sunDir));
                        std::memcpy(cc.vp, layer.vp, 64);
                        smp.staleSince[i] = 0u;// its bits are drawn anew (buildShadowDrawData)
                    }
                }
                layer.target     = vulkan::ShadowMapPass::Target::Cascade;
                layer.layer      = i;
                layer.cacheLayer = i;
                smp.layers.push_back(layer);

                std::memcpy(u.cascadeVP[i], layer.vp, 64);
                u.cascadeSphere[i][0] = c[0];
                u.cascadeSphere[i][1] = c[1];
                u.cascadeSphere[i][2] = c[2];
                u.cascadeSphere[i][3] = rBox;
                u.cascadeParams[i][0] = texel;
                u.cascadeParams[i][1] = 1.f / far;
                // The PCF radius in this cascade's texels: LightShadow::radius,
                // plus the sun's size (setSunAngularRadius) as the penumbra it
                // gives a caster kSunPenumbraHeight above the receiver. A fixed
                // kernel: the rays' penumbra grows with the caster's height,
                // this one does not (that takes a blocker search). 0 degrees,
                // the default, adds nothing.
                u.cascadeParams[i][2] = shadowSun_.radius + sunTan * kSunPenumbraHeight / texel;
                ShadowCullVolumes::Volume b{};
                b.kind = ShadowCullVolumes::Volume::Box;
                std::memcpy(b.c, c, sizeof(c));
                std::memcpy(b.x, x, sizeof(x));
                std::memcpy(b.y, y, sizeof(y));
                std::memcpy(b.l, L, sizeof(b.l));
                b.r      = rBox;
                b.cached = layer.mode == vulkan::ShadowMapPass::Mode::Cached;
                vol.layers.push_back(b);
            }
            u.cascadeCount    = n;
            u.flags          |= 1u;
            u.sunBias         = shadowSun_.bias;
            u.sunNormalBias   = shadowSun_.normalBias;
            u.sunIntensity    = shadowSun_.intensity;
            std::memcpy(u.sunDir, L, sizeof(u.sunDir));
            u.cascadeMapSize = size;
        }

        // ── The lamps' maps ──────────────────────────────────────────────────
        // A spot: one perspective map along its cone (fov = 2 angle focus, as
        // SpotLightShadow), a point: six cube faces of 90 degrees. Far = the
        // lamp's distance, or its influence radius when it has none; near
        // 5 cm. A cube face is drawn without the raster's y flip, with the GL
        // cube-capture cameras below, so its rows land where the hardware cube
        // lookup reads them (the Vulkan cube face table).
        constexpr float kLampNear = 0.05f;
        const auto      lampCamera = [](const float* pos, const Vector3& fwd, const Vector3& up) {
            Matrix4 world;
            const Vector3 eye(pos[0], pos[1], pos[2]);
            world.lookAt(eye, eye + fwd, up);
            world.setPosition(eye);
            return world;
        };
        for (uint32_t k = 0; k < sz.spots && k < shadowSpotSlots_.size(); ++k) {
            const ShadowLamp& L = shadowSpotSlots_[k];
            if (!L.used) continue;
            const Vector3 fwd(L.dir[0], L.dir[1], L.dir[2]);
            const Vector3 up = std::abs(fwd.y) < 0.99f ? Vector3(0.f, 1.f, 0.f) : Vector3(1.f, 0.f, 0.f);
            const float   far = std::max(L.far, kLampNear * 4.f);
            const float   t   = kLampNear * std::tan(0.5f * L.fov);
            Matrix4       proj;
            proj.makePerspective(-t, t, t, -t, kLampNear, far);
            // Its cone, and whether the view sees any of it: a spot lighting
            // nothing in the frustum has no visible shadow, so its layer is
            // cleared and left empty (a reflected surface it lights would then
            // read it as unshadowed; the rays' hits do not use the maps).
            ShadowCullVolumes::Volume cone{};
            cone.kind    = ShadowCullVolumes::Volume::Cone;
            cone.tanHalf = std::tan(0.5f * L.fov);
            cone.r       = far;
            std::memcpy(cone.c, L.pos, sizeof(cone.c));
            cone.d[0] = fwd.x; cone.d[1] = fwd.y; cone.d[2] = fwd.z;
            {
                const float   rad = 0.5f * far * std::sqrt(1.f + 4.f * cone.tanHalf * cone.tanHalf);
                const Vector3 mid(L.pos[0] + 0.5f * far * fwd.x, L.pos[1] + 0.5f * far * fwd.y,
                                  L.pos[2] + 0.5f * far * fwd.z);
                cone.skip = !viewFrustum.intersectsSphere(Sphere(mid, rad));
            }
            vulkan::ShadowMapPass::Layer layer{};
            lightClip(layer.vp, lampCamera(L.pos, fwd, up), proj);
            layer.target     = vulkan::ShadowMapPass::Target::Spot;
            layer.layer      = k;
            layer.cacheLayer = k;
            layer.skip       = cone.skip;
            // Its cache (ShadowMapPass::spotCache): served while the clip
            // matrix is the same bits and the static set holds, else rebuilt;
            // a layer the view cannot see keeps its entry.
            if (!layer.skip && shadowCacheOn_ && k < smp.spotCacheState.size()) {
                auto& lc = smp.spotCacheState[k];
                if (lc.valid && lc.staticGen == staticGen && std::memcmp(lc.vp, layer.vp, 64) == 0) {
                    layer.mode = vulkan::ShadowMapPass::Mode::Cached;
                } else {
                    layer.mode   = vulkan::ShadowMapPass::Mode::Rebuild;
                    lc.valid     = true;
                    lc.staticGen = staticGen;
                    std::memcpy(lc.vp, layer.vp, 64);
                }
            }
            cone.cached = layer.mode == vulkan::ShadowMapPass::Mode::Cached;
            smp.layers.push_back(layer);
            vol.layers.push_back(cone);
            std::memcpy(u.spotVP[k], layer.vp, 64);
            u.lampParams[k][0] = L.bias;
            u.lampParams[k][1] = L.normalBias;
            u.lampParams[k][2] = L.radius;
            u.lampParams[k][3] = L.intensity;
            u.lampTexel[k][0]  = 2.f * std::tan(0.5f * L.fov) / float(sz.spotSize);
            u.lampTexel[k][1]  = kLampNear;
            u.lampTexel[k][2]  = far;
            u.lampTexel[k][3]  = float(sz.spotSize);
        }
        for (uint32_t k = 0; k < sz.points && k < shadowPointSlots_.size(); ++k) {
            const ShadowLamp& L = shadowPointSlots_[k];
            if (!L.used) continue;
            static const Vector3 kFwd[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            static const Vector3 kUp[6]  = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
            const float          far     = std::max(L.far, kLampNear * 4.f);
            Matrix4              proj;
            proj.makePerspective(-kLampNear, kLampNear, kLampNear, -kLampNear, kLampNear, far);
            for (uint32_t face = 0; face < 6; ++face) {
                // The face's 90 degree wedge out to far, and whether the view
                // sees any of it (its bounding sphere against the frustum): a
                // face the camera cannot see is cleared and drawn into by
                // nothing.
                ShadowCullVolumes::Volume fv{};
                fv.kind = ShadowCullVolumes::Volume::Face;
                fv.r    = far;
                std::memcpy(fv.c, L.pos, sizeof(fv.c));
                fv.d[0] = kFwd[face].x; fv.d[1] = kFwd[face].y; fv.d[2] = kFwd[face].z;
                {
                    const Vector3 mid(L.pos[0] + 0.5f * far * fv.d[0], L.pos[1] + 0.5f * far * fv.d[1],
                                      L.pos[2] + 0.5f * far * fv.d[2]);
                    fv.skip = !viewFrustum.intersectsSphere(Sphere(mid, 1.5f * far));
                }
                vulkan::ShadowMapPass::Layer layer{};
                lightClip(layer.vp, lampCamera(L.pos, kFwd[face], kUp[face]), proj);
                layer.target     = vulkan::ShadowMapPass::Target::PointFace;
                layer.layer      = 6u * k + face;
                layer.cacheLayer = layer.layer;
                layer.skip       = fv.skip;
                // Its cache (ShadowMapPass::pointCache), as the spot's.
                if (!layer.skip && shadowCacheOn_ && layer.layer < smp.pointFaceCacheState.size()) {
                    auto& lc = smp.pointFaceCacheState[layer.layer];
                    if (lc.valid && lc.staticGen == staticGen && std::memcmp(lc.vp, layer.vp, 64) == 0) {
                        layer.mode = vulkan::ShadowMapPass::Mode::Cached;
                    } else {
                        layer.mode   = vulkan::ShadowMapPass::Mode::Rebuild;
                        lc.valid     = true;
                        lc.staticGen = staticGen;
                        std::memcpy(lc.vp, layer.vp, 64);
                    }
                }
                fv.cached = layer.mode == vulkan::ShadowMapPass::Mode::Cached;
                smp.layers.push_back(layer);
                vol.layers.push_back(fv);
            }
            const uint32_t pi = kMaxShadowSpots + k;
            u.pointPosFar[k][0] = L.pos[0];
            u.pointPosFar[k][1] = L.pos[1];
            u.pointPosFar[k][2] = L.pos[2];
            u.pointPosFar[k][3] = far;
            u.lampParams[pi][0] = L.bias;
            u.lampParams[pi][1] = L.normalBias;
            u.lampParams[pi][2] = L.radius;
            u.lampParams[pi][3] = L.intensity;
            u.lampTexel[pi][0]  = 2.f / float(sz.pointSize);
            u.lampTexel[pi][1]  = kLampNear;
            u.lampTexel[pi][2]  = far;
            u.lampTexel[pi][3]  = float(sz.pointSize);
        }
        u.spotCount  = sz.spots;
        u.pointCount = sz.points;
        if (shadowCacheDebug_) {
            size_t cached = 0, rebuilt = 0, drawn = 0, skipped = 0;
            for (const auto& l : smp.layers) {
                if (l.skip) ++skipped;
                else if (l.mode == vulkan::ShadowMapPass::Mode::Cached) ++cached;
                else if (l.mode == vulkan::ShadowMapPass::Mode::Rebuild) ++rebuilt;
                else ++drawn;
            }
            std::fprintf(stderr, "[shadowCache] frame: layers cached %zu rebuilt %zu drawn %zu skipped %zu\n", cached, rebuilt, drawn, skipped);
        }

        // Raster side: a small slope-scaled bias away from the light (reverse-
        // Z: negative); the receivers' normal offset does most of the work.
        smp.depthBiasConstant = 0.f;
        smp.depthBiasSlope    = -1.f;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        buildShadowDrawData(f, vol, camPos);

        auto& slot = smp.slot(f);
        std::memcpy(slot.uboMapped, &u, sizeof(u));
        flushHostWrites(ctx->allocator(), slot.ubo.alloc);
        return true;
    }

    void VulkanRenderer::Impl::buildShadowDrawData(uint32_t frame, const ShadowCullVolumes& vol,
                                                   const Vector3& camPos) {
        THREEPP_CPUPROF("frame.D_buildShadowDraws");
        constexpr uint32_t kGroups = vulkan::ShadowMapPass::kGroups;
        using Volume               = ShadowCullVolumes::Volume;
        auto&          smp         = *view().shadowMaps_;
        const auto&    entries     = lastVisibleEntries_;
        const float    cam[3]      = {camPos.x, camPos.y, camPos.z};
        const uint32_t nLayers     = static_cast<uint32_t>(std::min<size_t>(vol.layers.size(), 128));

        // Every caster once (a DrawInfo record), then one indirect command per
        // layer it reaches, bucketed per layer by cull mode and alpha test: the
        // near cascade draws the metres around the camera, a cube face what
        // lies in its 90 degree wedge within the lamp's range, a skipped layer
        // nothing. A caster whose bounds are unknown (a deformer) reaches every
        // layer, as the G-buffer keeps it.
        // The walk runs in chunks of spans (CpuPar.hpp), each emitting into
        // its own lists; the upload below lays the chunks out in span order,
        // so the bytes do not depend on the chunk count.
        auto&      chunks     = shadowChunks_;
        uint32_t   nChunks    = 1u;
        const auto resetChunk = [&](ShadowChunk& ch) {
            ch.draws.clear();
            ch.cmds.resize(size_t(nLayers) * 2u * kGroups);
            for (auto& v : ch.cmds) v.clear();
            ch.staleMask = 0u;
        };
        if (chunks.empty()) chunks.resize(1);

        // The cascades share the sun's frame, so a caster's coordinates in it
        // are taken once; each box keeps its centre's coordinates.
        struct BoxProj {
            float bx = 0.f, by = 0.f, bl = 0.f;
        };
        std::vector<BoxProj> boxProj(nLayers);
        const float*         sx = nullptr;
        const float*         sy = nullptr;
        const float*         sl = nullptr;
        uint64_t             allMask[2] = {0u, 0u};
        for (uint32_t li = 0; li < nLayers; ++li) {
            const Volume& v = vol.layers[li];
            if (v.kind == Volume::Box) {
                if (!sx) {
                    sx = v.x;
                    sy = v.y;
                    sl = v.l;
                }
                boxProj[li] = {dot3(v.c, v.x), dot3(v.c, v.y), dot3(v.c, v.l)};
            }
            if (!v.skip) allMask[li >> 6] |= 1ull << (li & 63u);
        }

        // The stale-tolerant cascades (setShadowCacheStaleFrames): the
        // cascades drawn anew this frame take every static caster's bit again;
        // a cached one is drawn over by the static casters without their bit.
        // The cascades are the first layers, the only Box volumes.
        const bool tolerant   = shadowCacheOn_ && shadowCacheStaleFrames_ > 0u;
        uint8_t    redrawCasc = 0u, cachedCasc = 0u;
        if (tolerant) {
            for (uint32_t li = 0; li < nLayers && li < 4u && vol.layers[li].kind == Volume::Box; ++li) {
                if (vol.layers[li].skip || li >= smp.layers.size()) continue;
                if (smp.layers[li].mode == vulkan::ShadowMapPass::Mode::Rebuild) redrawCasc |= uint8_t(1u << li);
                if (smp.layers[li].mode == vulkan::ShadowMapPass::Mode::Cached) cachedCasc |= uint8_t(1u << li);
            }
            if (redrawCasc != 0u) {
                const auto keep = static_cast<uint8_t>(~(redrawCasc | (redrawCasc << 4u)));
                for (auto& s : smp.casterSnap) s &= keep;
            }
        }
        const uint64_t staleNow = std::max<uint64_t>(frameSerial_, 1u);

        // The layers a world sphere (centre e, radius er) can cast into. A
        // static caster never reaches a cached cascade: the cache holds it
        // (stale-tolerant: one whose bit `snap` holds for that cascade).
        const auto reachMask = [&](const float* e, float er, bool dynamic, uint8_t snap, uint64_t* m) {
            m[0] = m[1] = 0u;
            float px = 0.f, py = 0.f, pl = 0.f;
            if (sx) {
                px = dot3(e, sx);
                py = dot3(e, sy);
                pl = dot3(e, sl);
            }
            for (uint32_t li = 0; li < nLayers; ++li) {
                const Volume& v = vol.layers[li];
                if (v.skip) continue;
                if (v.cached && !dynamic &&
                    (!tolerant || li >= 4u || v.kind != Volume::Box || ((snap >> li) & 1u) != 0u))
                    continue;
                bool in = false;
                if (v.kind == Volume::Box) {
                    // Sideways inside the box; along the sun anything in front
                    // of its far plane (casters sunward of the box are pancaked).
                    const float    lim = v.r + er;
                    const BoxProj& b   = boxProj[li];
                    in = std::abs(px - b.bx) <= lim && std::abs(py - b.by) <= lim && pl - b.bl >= -lim;
                } else {
                    const float d[3]  = {e[0] - v.c[0], e[1] - v.c[1], e[2] - v.c[2]};
                    const float along = dot3(d, v.d);
                    if (along + er < 0.f || along - er > v.r) continue;
                    if (v.kind == Volume::Cone) {
                        // Within the cone widened by the sphere's radius.
                        const float perp2 = std::max(dot3(d, d) - along * along, 0.f);
                        const float lim   = std::max(along + er, 0.f) * v.tanHalf +
                                          er * std::sqrt(1.f + v.tanHalf * v.tanHalf);
                        in = perp2 <= lim * lim;
                    } else {
                        // The face's wedge: within er of both of its 45 degree
                        // planes (the axis component is `along`, the rest `dd`).
                        const float dd[3] = {d[0] - v.d[0] * along, d[1] - v.d[1] * along, d[2] - v.d[2] * along};
                        const float u     = std::max({std::abs(dd[0]), std::abs(dd[1]), std::abs(dd[2])});
                        in                = along >= u - 1.4142136f * er;
                    }
                }
                if (in) m[li >> 6] |= 1ull << (li & 63u);
            }
        };

        // One span's constants (its record, bounds, cull bucket, LOD, draw
        // range), then one of its entries' record and commands.
        struct SpanInfo {
            const BlasRecord* rec       = nullptr;
            bool              bounded   = false;
            bool              lodStatic = false;
            int               bucket    = 2;
            LodGeomSel        sel0{};
            uint32_t          drawStart = 0u, drawCount = 0xFFFFFFFFu;
        };
        const auto spanInfo = [&](const EntrySpan& sp, SpanInfo& si) -> bool {
            const MeshEntry& e0 = entries[sp.first];
            // Overlays and particle sprites are not in the ray-traced scene, nor
            // are ParticleFields (particle_density.glsl shadows those).
            if (e0.isOverlay || e0.isParticle || e0.isParticleField) return false;
            si.rec = resolveBlasForEntry(e0);
            if (!si.rec || si.rec->vertex.handle == VK_NULL_HANDLE) return false;
            // Deformers' and interop meshes' bounds are stale: always kept, as
            // the G-buffer keeps them.
            const bool stale = e0.isSkinned || e0.isDisplaced || e0.isMorphed || e0.isTet || e0.isVertexInterop ||
                               e0.isGrass;
            si.bounded = !stale && sp.localBoundsValid;
            // Both faces cast, as a shadow ray sees both; Material::shadowSide
            // narrows it when the material sets it.
            si.bucket      = 2;
            const auto mat = e0.mesh->material();
            if (mat && mat->shadowSide) {
                si.bucket = *mat->shadowSide == Side::Front ? 0 : *mat->shadowSide == Side::Back ? 1 : 2;
            }
            si.lodStatic = si.rec->lodLevels.empty();
            si.sel0      = selectLodGeom(*si.rec, 0);
            si.drawStart = 0u;
            si.drawCount = 0xFFFFFFFFu;
            if (auto geom = e0.mesh->geometry()) {
                const auto& dr = geom->drawRange;
                si.drawStart   = dr.start > 0 ? static_cast<uint32_t>(dr.start) : 0u;
                si.drawCount   = dr.count > 0 ? static_cast<uint32_t>(dr.count) : 0u;
            }
            return true;
        };
        const auto emit = [&](const EntrySpan& sp, const SpanInfo& si, uint32_t i, ShadowChunk& ch) {
            const MeshEntry& en = entries[i];
            if (en.camAttached) return;
            if (en.sensorOnly && !(sensorOnlySurfaces_ && view().secondary && view().sensorSurfaces)) return;
            // One side of a LOD crossfade casts, the same one the rays see.
            if (i < entryLodFade_.size() && !lodFadeCasts(entryLodFade_[i])) return;
            if (i >= geomDescsCached_.size()) return;
            // shadowOccludes' policy (deferred_shade_10): transmission and
            // blend never block, strong emitters never block.
            const auto& md = entryMat(i);
            if (md.transmission > 0.f || md.alphaCutoff < 0.f) return;
            if (std::max({md.emissive[0], md.emissive[1], md.emissive[2]}) * md.emissiveIntensity >= 1.f) return;
            // Moving casters go to every layer's `dyn` list (a cached cascade
            // draws only those); static ones to `stat`.
            const bool dynamic = shadowCasterDynamic(i);
            uint8_t*   snapAt  = tolerant && i < smp.casterSnap.size() ? &smp.casterSnap[i] : nullptr;
            const uint8_t snap = snapAt ? *snapAt : 0u;
            uint64_t   mask[2] = {allMask[0], allMask[1]};
            if (si.bounded) {
                const float* M     = en.worldMatrix.data();
                const float* lc    = sp.localCenter;
                const float  wc[3] = {M[0] * lc[0] + M[4] * lc[1] + M[8] * lc[2] + M[12],
                                      M[1] * lc[0] + M[5] * lc[1] + M[9] * lc[2] + M[13],
                                      M[2] * lc[0] + M[6] * lc[1] + M[10] * lc[2] + M[14]};
                const float  s0    = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
                const float  s1    = M[4] * M[4] + M[5] * M[5] + M[6] * M[6];
                const float  s2    = M[8] * M[8] + M[9] * M[9] + M[10] * M[10];
                const float  wr    = sp.localRadius * std::sqrt(std::max({s0, s1, s2}));
                reachMask(wc, wr, dynamic, snap, mask);
            } else if (snapAt && !dynamic) {
                mask[0] &= ~uint64_t(snap & cachedCasc);// unbounded: what its bit already holds
            }
            if (snapAt && !dynamic) {
                // A cascade drawn anew holds it from now (and its box if it
                // reached it); one it is drawn over without its bit is stale.
                if (redrawCasc != 0u) {
                    uint8_t s = static_cast<uint8_t>(snap | redrawCasc);
                    for (uint32_t c = 0; c < 4u; ++c)
                        if (((redrawCasc >> c) & 1u) && ((mask[0] >> c) & 1u)) s |= uint8_t(1u << (4u + c));
                    *snapAt = s;
                }
                // Marked stale after the walk, in chunk order (the value is
                // this frame's either way).
                for (uint32_t c = 0; c < 4u; ++c)
                    if (((cachedCasc >> c) & 1u) && ((mask[0] >> c) & 1u) && smp.staleSince[c] == 0u)
                        ch.staleMask |= static_cast<uint8_t>(1u << c);
            }
            if ((mask[0] | mask[1]) == 0u) return;
            const BlasRecord* rec    = si.rec;
            const auto        sel    = si.lodStatic ? si.sel0 : selectLodGeom(*rec, en.lodLevel);
            uint32_t          vcount = sel.indexed ? sel.indexCount : rec->vertexCount;
            uint32_t          vfirst = 0u;
            if (en.lodLevel == 0 || si.lodStatic) {
                vfirst = std::min(si.drawStart, vcount);
                vcount = std::min(vcount - vfirst, si.drawCount);
            }
            if (vcount == 0u) return;
            // shadow_depth.frag's alpha test, for a cutout within the cutout
            // ray distance (setCutoutRayDistance) of this view's camera. Beyond
            // it the caster is drawn solid, as the rays then take it
            // (FORCE_OPAQUE): its shadow is the cards' outline. The rest are
            // depth only, with no fragment stage.
            const bool tested = md.alphaCutoff > 0.f && md.albedoTexIndex >= 0 &&
                                !(cutoutRayDistance_ > 0.f && cutoutCaster(md) && entryBeyond(en, cam, cutoutRayDistance_));
            const uint32_t group = (tested ? 3u : 0u) + static_cast<uint32_t>(si.bucket);
            DrawInfoGpu    di{};
            std::memcpy(di.model, en.worldMatrix.data(), 64);
            di.posAddr             = rec->vertex.address;
            di.nrmAddr             = rec->normal.address;
            di.uvAddr              = rec->uv.handle != VK_NULL_HANDLE ? rec->uv.address : 0ull;
            di.prevPosAddr         = rec->vertex.address;
            di.indexAddr           = sel.indexed ? sel.indexAddress : 0ull;
            di.instanceCustomIndex = i;
            di.indexed             = sel.indexed ? 1u : 0u;
            di.packedAttrs         = rec->packedMask;
            VkDrawIndirectCommand cmd{};
            cmd.vertexCount   = vcount;
            cmd.instanceCount = 1u;
            cmd.firstVertex   = vfirst;
            cmd.firstInstance = static_cast<uint32_t>(ch.draws.size());// its DrawInfo record, chunk-local
            ch.draws.push_back(di);
            for (uint32_t li = 0; li < nLayers; ++li) {
                // A cached layer draws only its `dyn` list: a static caster
                // drawn over a stale cascade goes there too.
                const bool over = dynamic || (tolerant && vol.layers[li].cached);
                if (mask[li >> 6] & (1ull << (li & 63u)))
                    ch.cmds[(size_t(li) * 2u + (over ? 1u : 0u)) * kGroups + group].push_back(cmd);
            }
        };

        // Steady state: every layer is skipped or served from its cache, so
        // only the moving casters need commands this frame (the caches hold
        // the static ones); their indices came out of the hash walk, in span
        // order. Any layer drawn or rebuilt this frame needs the full walk.
        bool anyStatic = false;
        for (uint32_t li = 0; li < nLayers && !anyStatic; ++li) {
            const Volume& v = vol.layers[li];
            anyStatic       = !v.skip && !v.cached;
        }
        if (shadowCacheOn_ && !anyStatic) {
            uint32_t curSpan = 0xFFFFFFFFu;
            bool     spanOk  = false;
            SpanInfo si{};
            resetChunk(chunks[0]);
            // Stale-tolerant: the static casters without their bit too.
            for (const uint32_t i : tolerant ? smp.onTop : shadowDynamicIdx_) {
                if (i >= shadowSpanOfEntry_.size() || i >= entries.size()) continue;
                const uint32_t s = shadowSpanOfEntry_[i];
                if (s != curSpan) {
                    curSpan = s;
                    spanOk  = s < entrySpans_.size() && !entrySpans_[s].hidden && entrySpans_[s].count > 0u &&
                             spanInfo(entrySpans_[s], si);
                }
                if (spanOk) emit(entrySpans_[s], si, i, chunks[0]);
            }
        } else {
            // Chunks of spans weighted by entry count (a stand of pines is
            // one span of thousands): one chunk when the knob is off.
            auto& bounds = shadowChunkBounds_;
            vulkan::par::splitByWeight(
                    entrySpans_.size(), vulkan::par::chunksFor(entries.size()),
                    [&](size_t s) -> uint64_t {
                        const auto& sp = entrySpans_[s];
                        return sp.hidden ? 0u : sp.count;
                    },
                    bounds);
            nChunks = static_cast<uint32_t>(bounds.size() - 1);
            if (chunks.size() < nChunks) chunks.resize(nChunks);
            vulkan::par::run(nChunks, [&](unsigned c) {
                ShadowChunk& ch = chunks[c];
                resetChunk(ch);
                for (uint32_t s = bounds[c]; s < bounds[c + 1]; ++s) {
                    const auto& sp = entrySpans_[s];
                    if (sp.hidden || sp.count == 0u) continue;
                    SpanInfo si{};
                    if (!spanInfo(sp, si)) continue;
                    for (uint32_t j = 0; j < sp.count; ++j) emit(sp, si, sp.first + j, ch);
                }
            });
        }
        for (uint32_t c = 0; c < nChunks; ++c)
            for (uint32_t k = 0; k < 4u; ++k)
                if (((chunks[c].staleMask >> k) & 1u) && smp.staleSince[k] == 0u) smp.staleSince[k] = staleNow;

        uint32_t totalCmds = 0, totalDraws = 0;
        for (uint32_t c = 0; c < nChunks; ++c) {
            totalDraws += static_cast<uint32_t>(chunks[c].draws.size());
            for (const auto& v : chunks[c].cmds) totalCmds += static_cast<uint32_t>(v.size());
        }
        smp.ensureDrawCapacity(frame, totalDraws, totalCmds);
        auto& slot = smp.slot(frame);
        void* dm   = nullptr;
        void* cm   = nullptr;
        check(vmaMapMemory(ctx->allocator(), slot.drawInfo.alloc, &dm), "vmaMapMemory(shadow drawInfo)");
        check(vmaMapMemory(ctx->allocator(), slot.cmds.alloc, &cm), "vmaMapMemory(shadow cmds)");
        // The chunks' records one after the other; a chunk's commands then
        // name theirs from the chunk's base.
        for (uint32_t c = 0, at = 0; c < nChunks; ++c) {
            const auto& d = chunks[c].draws;
            if (!d.empty()) std::memcpy(static_cast<DrawInfoGpu*>(dm) + at, d.data(), d.size() * sizeof(DrawInfoGpu));
            at += static_cast<uint32_t>(d.size());
        }
        smp.layerGroups.assign(nLayers, {});
        uint32_t offset = 0;
        for (uint32_t li = 0; li < nLayers; ++li) {
            for (uint32_t d = 0; d < 2u; ++d) {
                auto& groups = d == 0u ? smp.layerGroups[li].stat : smp.layerGroups[li].dyn;
                for (uint32_t g = 0; g < kGroups; ++g) {
                    const size_t idx      = (size_t(li) * 2u + d) * kGroups + g;
                    uint32_t     n        = 0u;
                    uint32_t     drawBase = 0u;
                    for (uint32_t c = 0; c < nChunks; ++c) {
                        const auto& list = chunks[c].cmds[idx];
                        auto*       out  = reinterpret_cast<VkDrawIndirectCommand*>(cm) + offset + n;
                        if (drawBase == 0u) {
                            if (!list.empty()) std::memcpy(out, list.data(), list.size() * sizeof(VkDrawIndirectCommand));
                        } else {
                            // Built in a register and written once: the
                            // mapped memory is write-combined, a read-back
                            // from it costs more than the whole walk.
                            for (const auto& cmd : list) {
                                VkDrawIndirectCommand c = cmd;
                                c.firstInstance += drawBase;
                                *out++ = c;
                            }
                        }
                        n += static_cast<uint32_t>(list.size());
                        drawBase += static_cast<uint32_t>(chunks[c].draws.size());
                    }
                    groups[g] = {offset, n};
                    offset += n;
                }
            }
        }
        smp.totalDraws = totalDraws;
        smp.totalCmds  = totalCmds;
        flushHostWrites(ctx->allocator(), slot.drawInfo.alloc);
        flushHostWrites(ctx->allocator(), slot.cmds.alloc);
        vmaUnmapMemory(ctx->allocator(), slot.cmds.alloc);
        vmaUnmapMemory(ctx->allocator(), slot.drawInfo.alloc);

        // The alpha test's inputs: this slot's material table every frame (it
        // can grow), the texture table when it changed.
        const bool texNeeded = !slot.matTexWritten || view().shadowMatTexValid_[frame] == 0;
        std::unique_ptr<std::array<VkDescriptorImageInfo, kMaxMaterialTextures>> tex;
        if (texNeeded) {
            tex = std::make_unique<std::array<VkDescriptorImageInfo, kMaxMaterialTextures>>();
            fillMaterialTextureInfos(*tex);
        }
        smp.writeRasterMaterials(frame, materialDescsBuffers[frame].handle, geometryDescsBuffers[frame].handle,
                                 texNeeded ? tex->data() : nullptr, kMaxMaterialTextures);
        view().shadowMatTexValid_[frame] = 1;
    }

    void VulkanRenderer::Impl::createShadowPipeline() {
        const auto module = [&](const uint32_t* code, size_t bytes, const char* what) {
            VkShaderModuleCreateInfo ci{};
            ci.sType         = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize      = bytes;
            ci.pCode         = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &ci, nullptr, &m), what);
            return m;
        };
        VkShaderModule vert = module(kShadowDepthVertSpv, sizeof(kShadowDepthVertSpv),
                                     "vkCreateShaderModule(shadow_depth.vert)");
        VkShaderModule frag = module(kShadowDepthFragSpv, sizeof(kShadowDepthFragSpv),
                                     "vkCreateShaderModule(shadow_depth.frag)");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName  = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.depthClampEnable = ctx->depthClampSupported() ? VK_TRUE : VK_FALSE;
        rs.polygonMode      = VK_POLYGON_MODE_FILL;
        rs.cullMode         = VK_CULL_MODE_NONE;
        rs.frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.depthBiasEnable  = VK_TRUE;
        rs.lineWidth        = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z: nearer the light wins
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType                   = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE,
                                      VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dy{};
        dy.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dy.dynamicStateCount = static_cast<uint32_t>(std::size(dyn));
        dy.pDynamicStates    = dyn;
        VkPipelineRenderingCreateInfo rci{};
        rci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkGraphicsPipelineCreateInfo gp{};
        gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.pNext               = &rci;
        gp.stageCount          = 2;
        gp.pStages             = stages;
        gp.pVertexInputState   = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState      = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState   = &ms;
        gp.pDepthStencilState  = &ds;
        gp.pColorBlendState    = &cb;
        gp.pDynamicState       = &dy;
        gp.layout              = rasterPipelineLayout;
        check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gp, nullptr, &shadowPipeline_),
              "vkCreateGraphicsPipelines(shadow maps)");
        // The same without the fragment stage: casters that need no alpha
        // test keep early depth.
        gp.stageCount = 1;
        check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gp, nullptr, &shadowDepthPipeline_),
              "vkCreateGraphicsPipelines(shadow maps, depth only)");
        vkDestroyShaderModule(ctx->device(), vert, nullptr);
        vkDestroyShaderModule(ctx->device(), frag, nullptr);
    }

}// namespace threepp
