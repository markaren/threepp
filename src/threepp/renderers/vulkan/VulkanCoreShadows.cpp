// Shadows: the mode knob (VulkanRenderer::setShadowMode) and the shadow-map
// path's host side — the always-bound stand-ins for bindings 80-83 of the
// deferred set, and the per-view descriptor upkeep.

#include "VulkanCoreImpl.hpp"

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
        const char* e = std::getenv("THREEPP_VK_SHADOW_MAPSIZE");
        return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
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
        shadowPipeline_ = VK_NULL_HANDLE;
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
    // type, the lamps whose range reaches the camera's frustum, castShadow ==
    // true first, then by power over the squared distance to the camera. A
    // lamp keeps its slot (and its layer) while it stays eligible, unless a
    // newcomer outranks it by castShadow or is twice as strong: no slot
    // flicker between two near-equal lamps.
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
            bool     prio;
            float    score;
            bool     taken;
        };
        const auto better = [](const Cand& a, const Cand& b) {
            return a.prio != b.prio ? a.prio : a.score > b.score;
        };
        for (int spot = 0; spot < 2; ++spot) {
            auto&          slots = spot ? shadowSpotSlots_ : shadowPointSlots_;
            const uint32_t cap   = active ? (spot ? shadowedSpots_ : shadowedPoints_) : 0u;
            std::vector<Cand> cands;
            for (uint32_t r = 0; r < recs.size() && cap > 0u; ++r) {
                const auto& g = recs[r];
                if ((g.type > 0.5f) != (spot == 1) || g.cullRadius <= 0.f) continue;
                const Vector3 p(g.position[0], g.position[1], g.position[2]);
                if (!frustum.intersectsSphere(Sphere(p, g.cullRadius))) continue;
                const float lum = 0.2126f * g.color[0] + 0.7152f * g.color[1] + 0.0722f * g.color[2];
                cands.push_back({r, src[r]->castShadow, lum / std::max(p.distanceToSquared(camPos), 1e-2f), false});
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
                    if (!((c.prio && !w.prio) || (c.prio == w.prio && c.score > 2.f * w.score))) break;
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
            v.shadowMaps_ = std::make_unique<vulkan::ShadowMapPass>(*ctx, kFramesInFlight, rasterDsLayout,
                                                                     transientPool_.get(), v.id);
        }
        if (shadowPipeline_ == VK_NULL_HANDLE) createShadowPipeline();
        auto& smp = *v.shadowMaps_;
        smp.ensure(f, sz);
        smp.layers.clear();

        GpuShadowMapsUbo  u{};
        ShadowCullVolumes vol;

        // ── The sun's cascades ───────────────────────────────────────────────
        // Practical split (lambda 0.75) of [near, distance] into N slices; each
        // slice's bounding sphere (its size depends on the camera's projection
        // alone, so it holds still while the camera turns) is an ortho box of
        // half extent r around its centre in the sun's frame, the centre
        // snapped to the texel grid so the map does not shimmer as the camera
        // moves. The box starts at the sphere's sunward rim; casters beyond it
        // are pancaked onto the near plane by depth clamp (or, without it, the
        // near plane is pulled 1 km toward the sun).
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
                r                 = std::ceil(r * 16.f) / 16.f;// a stable size, off the float noise
                const float texel = 2.f * r / size;
                const float cx    = dot3(c, x), cy = dot3(c, y);
                const float sx    = std::floor(cx / texel) * texel - cx;
                const float sy    = std::floor(cy / texel) * texel - cy;
                for (int a = 0; a < 3; ++a) c[a] += x[a] * sx + y[a] * sy;

                const float far = 2.f * r + extra;
                Matrix4     world;
                world.makeBasis(Vector3(x[0], x[1], x[2]), Vector3(y[0], y[1], y[2]), Vector3(L[0], L[1], L[2]));
                world.setPosition(c[0] + L[0] * (r + extra), c[1] + L[1] * (r + extra), c[2] + L[2] * (r + extra));
                Matrix4 proj;
                proj.makeOrthographic(-r, r, r, -r, 0.f, far);
                vulkan::ShadowMapPass::Layer layer{};
                lightClip(layer.vp, world, proj);
                layer.target = vulkan::ShadowMapPass::Target::Cascade;
                layer.layer  = i;
                smp.layers.push_back(layer);

                std::memcpy(u.cascadeVP[i], layer.vp, 64);
                u.cascadeSphere[i][0] = c[0];
                u.cascadeSphere[i][1] = c[1];
                u.cascadeSphere[i][2] = c[2];
                u.cascadeSphere[i][3] = r;
                u.cascadeParams[i][0] = texel;
                u.cascadeParams[i][1] = 1.f / far;
                ShadowCullVolumes::Box b{};
                std::memcpy(b.c, c, sizeof(c));
                std::memcpy(b.x, x, sizeof(x));
                std::memcpy(b.y, y, sizeof(y));
                std::memcpy(b.l, L, sizeof(b.l));
                b.r = r;
                vol.boxes.push_back(b);
            }
            u.cascadeCount    = n;
            u.flags          |= 1u;
            u.sunBias         = shadowSun_.bias;
            u.sunNormalBias   = shadowSun_.normalBias;
            u.sunRadiusTexels = shadowSun_.radius;
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
            vulkan::ShadowMapPass::Layer layer{};
            lightClip(layer.vp, lampCamera(L.pos, fwd, up), proj);
            layer.target = vulkan::ShadowMapPass::Target::Spot;
            layer.layer  = k;
            smp.layers.push_back(layer);
            std::memcpy(u.spotVP[k], layer.vp, 64);
            u.lampParams[k][0] = L.bias;
            u.lampParams[k][1] = L.normalBias;
            u.lampParams[k][2] = L.radius;
            u.lampParams[k][3] = L.intensity;
            u.lampTexel[k][0]  = 2.f * std::tan(0.5f * L.fov) / float(sz.spotSize);
            u.lampTexel[k][1]  = kLampNear;
            u.lampTexel[k][2]  = far;
            u.lampTexel[k][3]  = float(sz.spotSize);
            vol.balls.push_back({{L.pos[0], L.pos[1], L.pos[2]}, far});
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
                vulkan::ShadowMapPass::Layer layer{};
                lightClip(layer.vp, lampCamera(L.pos, kFwd[face], kUp[face]), proj);
                layer.target = vulkan::ShadowMapPass::Target::PointFace;
                layer.layer  = 6u * k + face;
                smp.layers.push_back(layer);
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
            vol.balls.push_back({{L.pos[0], L.pos[1], L.pos[2]}, far});
        }
        u.spotCount  = sz.spots;
        u.pointCount = sz.points;

        // Raster side: a small slope-scaled bias away from the light (reverse-
        // Z: negative); the receivers' normal offset does most of the work.
        smp.depthBiasConstant = 0.f;
        smp.depthBiasSlope    = -1.f;

        buildShadowDrawData(f, vol);

        auto& slot = smp.slot(f);
        std::memcpy(slot.uboMapped, &u, sizeof(u));
        flushHostWrites(ctx->allocator(), slot.ubo.alloc);
        return true;
    }

    void VulkanRenderer::Impl::buildShadowDrawData(uint32_t frame, const ShadowCullVolumes& vol) {
        THREEPP_CPUPROF("frame.D_buildShadowDraws");
        auto&                                             smp = *view().shadowMaps_;
        std::array<std::vector<DrawInfoGpu>, 3>           draws;
        std::array<std::vector<VkDrawIndirectCommand>, 3> cmds;
        const auto&                                       entries = lastVisibleEntries_;

        const auto reaches = [&](const float* e, float er) {
            for (const auto& b : vol.boxes) {
                const float d[3] = {e[0] - b.c[0], e[1] - b.c[1], e[2] - b.c[2]};
                const float lim  = b.r + er;
                // Sideways inside the box; along the sun anything in front of
                // its far plane (casters sunward of the box are pancaked).
                if (std::abs(dot3(d, b.x)) <= lim && std::abs(dot3(d, b.y)) <= lim && dot3(d, b.l) >= -lim)
                    return true;
            }
            for (const auto& s : vol.balls) {
                const float d[3] = {e[0] - s.c[0], e[1] - s.c[1], e[2] - s.c[2]};
                const float lim  = s.r + er;
                if (dot3(d, d) <= lim * lim) return true;
            }
            return false;
        };

        for (const auto& sp : entrySpans_) {
            if (sp.hidden || sp.count == 0u) continue;
            const MeshEntry& e0 = entries[sp.first];
            // Overlays and particle sprites are not in the ray-traced scene, nor
            // are ParticleFields (particle_density.glsl shadows those).
            if (e0.isOverlay || e0.isParticle || e0.isParticleField) continue;
            const BlasRecord* rec = resolveBlasForEntry(e0);
            if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;
            // Deformers' and interop meshes' bounds are stale: always kept, as
            // the G-buffer keeps them.
            const bool stale = e0.isSkinned || e0.isDisplaced || e0.isMorphed || e0.isTet || e0.isVertexInterop ||
                               e0.isGrass;
            const bool bounded = !stale && sp.localBoundsValid;
            // Both faces cast, as a shadow ray sees both; Material::shadowSide
            // narrows it when the material sets it.
            int        bucket = 2;
            const auto mat    = e0.mesh->material();
            if (mat && mat->shadowSide) {
                bucket = *mat->shadowSide == Side::Front ? 0 : *mat->shadowSide == Side::Back ? 1 : 2;
            }
            const bool lodStatic = rec->lodLevels.empty();
            const auto sel0      = selectLodGeom(*rec, 0);
            uint32_t   drawStart = 0u, drawCount = 0xFFFFFFFFu;
            if (auto geom = e0.mesh->geometry()) {
                const auto& dr = geom->drawRange;
                drawStart      = dr.start > 0 ? static_cast<uint32_t>(dr.start) : 0u;
                drawCount      = dr.count > 0 ? static_cast<uint32_t>(dr.count) : 0u;
            }
            for (uint32_t j = 0; j < sp.count; ++j) {
                const uint32_t   i  = sp.first + j;
                const MeshEntry& en = entries[i];
                if (en.camAttached) continue;
                if (en.sensorOnly && !(sensorOnlySurfaces_ && view().secondary && view().sensorSurfaces)) continue;
                if (i >= matDescsCached_.size()) continue;
                // shadowOccludes' policy (deferred_shade_10): transmission and
                // blend never block, strong emitters never block.
                const auto& md = matDescsCached_[i];
                if (md.transmission > 0.f || md.alphaCutoff < 0.f) continue;
                if (std::max({md.emissive[0], md.emissive[1], md.emissive[2]}) * md.emissiveIntensity >= 1.f)
                    continue;
                if (bounded) {
                    const float* M     = en.worldMatrix.data();
                    const float* lc    = sp.localCenter;
                    const float  wc[3] = {M[0] * lc[0] + M[4] * lc[1] + M[8] * lc[2] + M[12],
                                          M[1] * lc[0] + M[5] * lc[1] + M[9] * lc[2] + M[13],
                                          M[2] * lc[0] + M[6] * lc[1] + M[10] * lc[2] + M[14]};
                    const float  s0    = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
                    const float  s1    = M[4] * M[4] + M[5] * M[5] + M[6] * M[6];
                    const float  s2    = M[8] * M[8] + M[9] * M[9] + M[10] * M[10];
                    const float  wr    = sp.localRadius * std::sqrt(std::max({s0, s1, s2}));
                    if (!reaches(wc, wr)) continue;
                }
                const auto sel    = lodStatic ? sel0 : selectLodGeom(*rec, en.lodLevel);
                uint32_t   vcount = sel.indexed ? sel.indexCount : rec->vertexCount;
                uint32_t   vfirst = 0u;
                if (en.lodLevel == 0 || lodStatic) {
                    vfirst = std::min(drawStart, vcount);
                    vcount = std::min(vcount - vfirst, drawCount);
                }
                if (vcount == 0u) continue;
                DrawInfoGpu di{};
                std::memcpy(di.model, en.worldMatrix.data(), 64);
                di.posAddr             = rec->vertex.address;
                di.nrmAddr             = rec->normal.address;
                di.uvAddr              = rec->uv.handle != VK_NULL_HANDLE ? rec->uv.address : 0ull;
                di.prevPosAddr         = rec->vertex.address;
                di.indexAddr           = sel.indexed ? sel.indexAddress : 0ull;
                di.instanceCustomIndex = i;
                di.indexed             = sel.indexed ? 1u : 0u;
                di.packedAttrs         = rec->packedMask;
                draws[bucket].push_back(di);
                VkDrawIndirectCommand cmd{};
                cmd.vertexCount   = vcount;
                cmd.instanceCount = 1u;
                cmd.firstVertex   = vfirst;
                cmds[bucket].push_back(cmd);
            }
        }

        uint32_t total = 0;
        for (const auto& d : draws) total += static_cast<uint32_t>(d.size());
        smp.ensureDrawCapacity(frame, total);
        auto& slot = smp.slot(frame);
        void* dm   = nullptr;
        void* cm   = nullptr;
        check(vmaMapMemory(ctx->allocator(), slot.drawInfo.alloc, &dm), "vmaMapMemory(shadow drawInfo)");
        check(vmaMapMemory(ctx->allocator(), slot.cmds.alloc, &cm), "vmaMapMemory(shadow cmds)");
        uint32_t offset = 0;
        for (int b = 0; b < 3; ++b) {
            const uint32_t nb = static_cast<uint32_t>(draws[b].size());
            smp.groups[b]     = {offset, nb};
            for (uint32_t k = 0; k < nb; ++k) cmds[b][k].firstInstance = offset + k;
            if (nb > 0u) {
                std::memcpy(static_cast<uint8_t*>(dm) + offset * sizeof(DrawInfoGpu), draws[b].data(),
                            nb * sizeof(DrawInfoGpu));
                std::memcpy(static_cast<uint8_t*>(cm) + offset * sizeof(VkDrawIndirectCommand), cmds[b].data(),
                            nb * sizeof(VkDrawIndirectCommand));
            }
            offset += nb;
        }
        smp.totalDraws = total;
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
        smp.writeRasterMaterials(frame, materialDescsBuffers[frame].handle, texNeeded ? tex->data() : nullptr,
                                 kMaxMaterialTextures);
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
        vkDestroyShaderModule(ctx->device(), vert, nullptr);
        vkDestroyShaderModule(ctx->device(), frag, nullptr);
    }

}// namespace threepp
