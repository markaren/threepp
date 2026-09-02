#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

#include "threepp/lights/HemisphereLight.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>// std::getenv / std::strtof (THREEPP_VK_SPLATVOL_SIGMA)
#include <cstring>
#include <limits>

namespace threepp {

    void VulkanRenderer::Impl::createCameraUbos() {
        for (auto& b : view().cameraUbos) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ 2 * 16 * sizeof(float) + 8 * sizeof(float),
                    // viewInverse + projInverse + jitter (.xy = clip-space
                    // sub-pixel offset matching raster's Halton, .zw = 1/res)
                    // + camAux (.x = parallel projection, .yzw = world forward).
                    // Shaders that don't need camAux declare the block without
                    // it — a trailing member only the readers have to know about.
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
        // Initial cap = 1 identity matrix. ensureMotionMatCapacity grows
        // and triggers a descriptor rewrite the first time a scene is
        // built with > 1 instance.
        //
        // PRIMARY ONLY: motion matrices are per-object world-space state,
        // camera-independent, so motionMatBuffers lives on Impl and is shared
        // by every view. A secondary running this block overwrote the shared
        // handles with fresh 1-capacity buffers — leaking the primary's (one
        // ring per addView, reported at vkDestroyDevice), resetting the
        // capacity, and leaving every already-written descriptor pointing at
        // the orphaned ring.
        if (view().secondary) return;
        for (uint32_t f = 0; f < kFramesInFlight; ++f) {
            motionMatBuffers[f] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ 16 * sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            motionMatBufferCapacity[f] = 1;
            // Seed an identity so reads on the first frame (with descriptors
            // already wired) get a sane answer even before any scene build.
            static const float identity[16] = {
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1};
            uploadHostVisible(ctx->allocator(), motionMatBuffers[f], identity, sizeof(identity));
        }
        // Seed mesh-moved-bits buffers with capacity 1 word (32 meshes worth)
        // so descriptor writes have a valid handle before any scene build.
        // ensureMeshMovedBitsCapacity grows in place when scenes need more.
        for (uint32_t f = 0; f < kFramesInFlight; ++f) {
            meshMovedBitsBuffers[f] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            meshMovedBitsBufferCapacity[f] = 1;
            const uint32_t zero = 0u;
            uploadHostVisible(ctx->allocator(), meshMovedBitsBuffers[f], &zero, sizeof(zero));
        }
        // Seed emissive-tri buffers with capacity 1 so descriptor writes have
        // a valid handle even before any emissive geometry exists. The shader
        // reads it only when pc.emissiveCount > 0, so the dummy bytes are
        // never sampled — but Vulkan requires valid resource bindings.
        for (uint32_t f = 0; f < kFramesInFlight; ++f) {
            emissiveTriBuffers[f] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ 64,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            emissiveTriBufferCapacity[f] = 1;
        }
    }

    // Grow emissiveTriBuffers[frame] in-place if the current frame's
    // emissive count exceeds capacity. Returns true when the buffer
    // handle changed so the caller can rewrite binding 14. 2× headroom
    // matches motionMatBuffers.
    bool VulkanRenderer::Impl::ensureEmissiveTriCapacity(uint32_t frame, uint32_t needed) {
        if (needed <= emissiveTriBufferCapacity[frame]) return false;
        const uint32_t newCap = std::max<uint32_t>(needed, emissiveTriBufferCapacity[frame] * 2u);
        destroyBuffer(ctx->allocator(), emissiveTriBuffers[frame]);
        emissiveTriBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                /*size*/ static_cast<VkDeviceSize>(newCap) * 64u,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        emissiveTriBufferCapacity[frame] = newCap;
        return true;
    }

    // Compute per-instance motion matrices = prevWorld * inverse(curWorld)
    // and upload to motionMatBuffers[frame]. The prev matrices live in the
    // entry-aligned prevWorldByEntry_ (identity-remapped across full rebuilds
    // by (Mesh*, instanceIndex), so each InstancedMesh sub-instance keeps its
    // own motion delta and a topology rebuild reprojects as a no-op only for
    // genuinely new entries). Caller must have already waited the
    // inFlight[frame] fence — we write a buffer the GPU may have been
    // reading on the previous use of `frame`.
    void VulkanRenderer::Impl::computeAndUploadMotionMatrices(uint32_t frame,
                                        const std::vector<MeshEntry>& entries) {
        // SPLIT, not nested: this phase used to be function-scope and therefore
        // silently included the upload below, so its number was compute+copy and
        // no upload cost was separable. stop()ped after the math so A means
        // "derive the matrices" and frame.G_uploadMotion means "move the bytes",
        // and the two can be summed.
        THREEPP_CPUPROF_NAMED(profA, "frame.A_motionMats");
        const uint32_t count = static_cast<uint32_t>(entries.size());
        if (count == 0) return;

        // Defensive (re)size — the full rebuild seeds these; a mismatch here
        // means a code path skipped the seeding, fall back to identity.
        if (motionScratch_.size() != size_t(count) * 16u ||
            prevWorldByEntry_.size() != count) {
            seedMotionState(entries);
        }

        // Per-instance "settled" threshold. Physics solvers (Bullet,
        // PhysX, etc.) often leave bodies with sub-millimeter / sub-
        // milliradian residual jitter even at rest — a parked car still
        // ticks every frame as the constraint solver re-converges. The
        // deferred shade pipeline faithfully reflects that motion
        // (motionMat → motion vec → FC reset/reproject) and the user sees
        // a wobble that doesn't actually exist in the asset.
        //
        // Threshold the per-element matrix delta: if the largest absolute
        // change is below kSettledEps, treat motion as identity AND
        // don't update the prev matrix. The body then locks to its prior
        // pose persistently — sub-eps accumulation can't drift past the
        // threshold because the reference doesn't update.
        //
        // 1e-4 covers ~0.1mm translation / ~0.0057° rotation. Tighter
        // than typical solver residual, looser than visible motion.
        // Tune up if your physics still wobbles; tune down if real slow
        // motion gets frozen.
        constexpr float kSettledEps = 1e-4f;

        static constexpr float kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                                0, 0, 1, 0, 0, 0, 0, 1};

        // Per SPAN: a span whose matrices did not change this frame
        // (ensureSceneBuilt's span refresh) has identity motion by
        // definition — its scratch blocks are touched at most once (to
        // clear the previous motion) and its prev matrices stay put. Only
        // moved spans pay per-entry work. This replaced a per-entry hash
        // lookup + full 4x4 inverse per instance per frame (~11 ms at 115k
        // static, ~55 ms at 100k moving).
        bool changedAny = false;
        for (auto& sp : entrySpans_) {
            if (!sp.movedThisFrame) {
                if (sp.motionNonIdentity) {
                    // Span stopped moving: collapse its blocks back to
                    // identity once. prev already equals cur (unchanged).
                    for (uint32_t j = 0; j < sp.count; ++j)
                        std::memcpy(&motionScratch_[(size_t(sp.first) + j) * 16u],
                                    kIdentity, 64);
                    sp.motionNonIdentity = false;
                    changedAny = true;
                }
                continue;
            }
            bool spanNonIdentity = false;
            for (uint32_t j = 0; j < sp.count; ++j) {
                const size_t i = size_t(sp.first) + j;
                const float* cur = entries[i].worldMatrix.data();
                float* prev = prevWorldByEntry_[i].data();
                float* dst = &motionScratch_[i * 16u];
                if (!prevWorldValidByEntry_[i]) {
                    // First-seen (cold-start): identity motion, seed prev.
                    std::memcpy(dst, kIdentity, 64);
                    std::memcpy(prev, cur, 64);
                    prevWorldValidByEntry_[i] = 1u;
                    changedAny = true;
                    continue;
                }
                float maxDelta = 0.0f;
                for (int e = 0; e < 16; ++e) {
                    const float d = std::abs(cur[e] - prev[e]);
                    if (d > maxDelta) maxDelta = d;
                }
                if (maxDelta < kSettledEps) {
                    // Settled: motion identity, prev stays anchored.
                    if (std::memcmp(dst, kIdentity, 64) != 0) {
                        std::memcpy(dst, kIdentity, 64);
                        changedAny = true;
                    }
                    continue;
                }
                // motion = prev * inverse(cur). World matrices are affine
                // (TRS compositions; bottom row 0,0,0,1), so a 3x3-cofactor
                // affine inverse is exact and ~3x cheaper than the general
                // 4x4 cofactor expansion Matrix4::invert runs.
                float inv[16];
                {
                    const float a00 = cur[0], a01 = cur[4], a02 = cur[8],  tx = cur[12];
                    const float a10 = cur[1], a11 = cur[5], a12 = cur[9],  ty = cur[13];
                    const float a20 = cur[2], a21 = cur[6], a22 = cur[10], tz = cur[14];
                    const float c00 = a11 * a22 - a12 * a21;
                    const float c01 = a02 * a21 - a01 * a22;
                    const float c02 = a01 * a12 - a02 * a11;
                    const float det = a00 * c00 + a10 * c01 + a20 * c02;
                    const float id  = det != 0.f ? 1.f / det : 0.f;
                    const float b00 = c00 * id;
                    const float b01 = c01 * id;
                    const float b02 = c02 * id;
                    const float b10 = (a12 * a20 - a10 * a22) * id;
                    const float b11 = (a00 * a22 - a02 * a20) * id;
                    const float b12 = (a02 * a10 - a00 * a12) * id;
                    const float b20 = (a10 * a21 - a11 * a20) * id;
                    const float b21 = (a01 * a20 - a00 * a21) * id;
                    const float b22 = (a00 * a11 - a01 * a10) * id;
                    inv[0] = b00; inv[4] = b01; inv[8]  = b02;
                    inv[1] = b10; inv[5] = b11; inv[9]  = b12;
                    inv[2] = b20; inv[6] = b21; inv[10] = b22;
                    inv[12] = -(b00 * tx + b01 * ty + b02 * tz);
                    inv[13] = -(b10 * tx + b11 * ty + b12 * tz);
                    inv[14] = -(b20 * tx + b21 * ty + b22 * tz);
                    inv[3] = 0.f; inv[7] = 0.f; inv[11] = 0.f; inv[15] = 1.f;
                }
                // dst = prev * inv (column-major 4x4; both affine).
                for (int c = 0; c < 4; ++c) {
                    const float i0 = inv[c * 4 + 0], i1 = inv[c * 4 + 1],
                                i2 = inv[c * 4 + 2], i3 = inv[c * 4 + 3];
                    dst[c * 4 + 0] = prev[0] * i0 + prev[4] * i1 + prev[8] * i2 + prev[12] * i3;
                    dst[c * 4 + 1] = prev[1] * i0 + prev[5] * i1 + prev[9] * i2 + prev[13] * i3;
                    dst[c * 4 + 2] = prev[2] * i0 + prev[6] * i1 + prev[10] * i2 + prev[14] * i3;
                    dst[c * 4 + 3] = prev[3] * i0 + prev[7] * i1 + prev[11] * i2 + prev[15] * i3;
                }
                std::memcpy(prev, cur, 64);
                spanNonIdentity = true;
                changedAny = true;
            }
            if (spanNonIdentity) sp.motionNonIdentity = true;
        }
        if (changedAny) ++motionScratchVersion_;
        profA.stop();

        // Upload only when this FIF slot doesn't already hold the current
        // scratch contents (a fully static scene uploads once per slot and
        // then never again). 64 B/entry, whole-array — a single moved span
        // re-copies every entry's block.
        if (motionUploadedVersion_[frame] != motionScratchVersion_) {
            THREEPP_CPUPROF("frame.G_uploadMotion");
            uploadHostVisible(ctx->allocator(), motionMatBuffers[frame],
                              motionScratch_.data(),
                              motionScratch_.size() * sizeof(float));
            motionUploadedVersion_[frame] = motionScratchVersion_;
        }
    }

    // Seed the entry-aligned motion state to "no motion": scratch all
    // identity, prev = the entries' CURRENT world matrices (so the first
    // real move produces a correct one-frame delta, not a cold start).
    // Callers that can preserve cross-rebuild history overwrite prev slots
    // afterwards (see the identity remap in the full rebuild).
    void VulkanRenderer::Impl::seedMotionState(const std::vector<MeshEntry>& entries) {
        const size_t count = entries.size();
        motionScratch_.assign(count * 16u, 0.f);
        for (size_t i = 0; i < count; ++i) {
            motionScratch_[i * 16u + 0]  = 1.f;
            motionScratch_[i * 16u + 5]  = 1.f;
            motionScratch_[i * 16u + 10] = 1.f;
            motionScratch_[i * 16u + 15] = 1.f;
        }
        prevWorldByEntry_.resize(count);
        prevWorldValidByEntry_.assign(count, 1u);
        for (size_t i = 0; i < count; ++i)
            prevWorldByEntry_[i] = entries[i].worldMatrix;
        ++motionScratchVersion_;
        // New layout ⇒ every FIF slot's contents are stale regardless of
        // version equality (the buffers may have been reallocated too).
        motionUploadedVersion_.fill(motionScratchVersion_ - 1u);
    }

    // Upload meshMovedBits_ to meshMovedBitsBuffers[frame]. Caller must have
    // already waited the inFlight[frame] fence.
    void VulkanRenderer::Impl::uploadMeshMovedBits(uint32_t frame) {
        if (meshMovedBits_.empty()) return;
        const VkDeviceSize bytes = meshMovedBits_.size() * sizeof(uint32_t);
        const VkDeviceSize cap   = meshMovedBitsBufferCapacity[frame] * sizeof(uint32_t);
        uploadHostVisible(ctx->allocator(), meshMovedBitsBuffers[frame],
                          meshMovedBits_.data(), std::min(bytes, cap));
    }

    // Walk visible entries, gather emissive triangles in world space, and
    // upload to emissiveTriBuffers[frame]. Per-tri 64-byte record:
    //   v0.xyz = world pos0,    v0.w = triangle area
    //   v1.xyz = world pos1,    v1.w = running cumPower (CDF)
    //   v2.xyz = world pos2,    v2.w = per-tri power (lum * area)
    //   emission.xyz = emissive*intensity, emission.w = unused
    // followed by a 64-byte HEADER (v0.x = emissive-instance count L) and,
    // when L <= kEmissiveCoverMaxLights, one 64-byte record per emissive
    // INSTANCE (centre/radius, area-weighted normal sum, CDF start, tri
    // range, area, power, Le) — the shader's coverage mode, which samples
    // every light instead of letting a few strata pick among them
    // (emissive_lights.glsl). Capacity is sized on the whole record count.
    //
    // Uniform-by-area within each tri × power-weighted picking across tris
    // gives a constant area-weighted-luminance pdf for closest_hit's NEE.
    //
    // Returns true when the buffer handle changed (capacity grew); caller
    // must then rewrite descriptor binding 14 for this frame's sets.
    bool VulkanRenderer::Impl::buildAndUploadEmissiveTris(uint32_t frame,
                                    const std::vector<MeshEntry>& entries) {
        THREEPP_CPUPROF("frame.E_emissiveTris");
        emissiveTriCountThisFrame_ = 0;
        emissiveTotalPowerThisFrame_ = 0.0f;

        // Fast path: nothing that affects the world-space emissive CDF
        // has changed since the last rebuild. World-space tri positions
        // depend on mesh world matrices + emissive material values; both
        // are tracked by meshMovedBits_ (set on xfm OR mat OR bone change).
        // Camera motion does NOT invalidate. Bistro / Sponza static
        // frames hit this path and skip the per-tri walk entirely; the
        // walk is O(visible-emissive-meshes × tris) per frame and was
        // CPU-bound on Bistro before this cache.
        const bool anyMeshMoved =
                std::any_of(meshMovedBits_.begin(), meshMovedBits_.end(),
                            [](uint32_t v) { return v != 0u; });
        const bool entriesUnchanged = (cachedEmissiveEntryCount_ == entries.size());
        if (!anyMeshMoved && entriesUnchanged) {
            emissiveTriCountThisFrame_   = cachedEmissiveTriCount_;
            emissiveTotalPowerThisFrame_ = cachedEmissiveTotalPower_;
            if (cachedEmissiveTriCount_ == 0) {
                return false;// no emissives → nothing to upload
            }
            if (emissiveBufferVersion_[frame] == cachedEmissiveVersion_) {
                // This frame's GPU buffer already holds the cached data.
                return false;
            }
            const bool grew = ensureEmissiveTriCapacity(
                    frame, static_cast<uint32_t>(cachedEmissiveData_.size() / 16));
            uploadHostVisible(ctx->allocator(), emissiveTriBuffers[frame],
                              cachedEmissiveData_.data(),
                              cachedEmissiveData_.size() * sizeof(float));
            emissiveBufferVersion_[frame] = cachedEmissiveVersion_;
            return grew;
        }

        std::vector<float> data;// 16 floats per tri
        data.reserve(64 * 16);
        float cumPower = 0.0f;
        // Per-light records for the shader's COVERAGE mode (emissive_lights.glsl):
        // one per emissive INSTANCE, appended behind a header after the triangles.
        std::vector<float> lights;// 16 floats per light, kept only while under the cap
        lights.reserve(kEmissiveCoverMaxLights * 16);
        uint32_t lightCount = 0;// every emissive instance, capped or not

        // Per SPAN: the emissive verdict is a per-MESH fact, read off the
        // expansion-cached MaterialWithEmissive* (material pointer swaps force
        // a full expansion, so it can't dangle). The old per-entry walk paid a
        // material() shared_ptr + a cross-hierarchy dynamic_cast per INSTANCE
        // whenever anything in the scene moved — ~220 ms/frame on a 100k-
        // instance moving field with zero emissives.
        for (const auto& sp : entrySpans_) {
            const auto& e0 = entries[sp.first];
            if (e0.isOverlay) continue;// raster-overlay only — no emissive contribution to the traced scene
            if (e0.sensorOnly) continue;// sensor target — lights nothing
            if (!e0.mesh) continue;
            const MaterialWithEmissive* em = e0.lodEmissive;
            if (!em) continue;
            const float emR = em->emissive.r * em->emissiveIntensity;
            const float emG = em->emissive.g * em->emissiveIntensity;
            const float emB = em->emissive.b * em->emissiveIntensity;
            const float emLum = 0.2126f * emR + 0.7152f * emG + 0.0722f * emB;
            if (emLum < 1e-6f) continue;
            // emissiveMap modulates per-texel; we don't sample textures here,
            // so use the constant tint for power. Slightly under-samples
            // bright textured emissives but keeps the build lightweight.

            auto geomPtr = e0.mesh->geometry();
            if (!geomPtr) continue;
            auto* posAttr = geomPtr->getAttribute<float>("position");
            if (!posAttr) continue;
            const auto& positions = posAttr->array();
            const uint32_t vcount = static_cast<uint32_t>(posAttr->count());
            if (vcount < 3) continue;

            const auto* idxAttr = geomPtr->getIndex();
            const bool indexed = idxAttr != nullptr;
            const uint32_t triCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vcount / 3;
            if (triCount == 0) continue;

            for (uint32_t sj = 0; sj < sp.count; ++sj) {
            const MeshEntry& en = entries[sp.first + sj];
            const float* M = en.worldMatrix.data();// column-major 4x4
            auto xform = [&](float x, float y, float z, float& wx, float& wy, float& wz) {
                wx = M[0] * x + M[4] * y + M[8]  * z + M[12];
                wy = M[1] * x + M[5] * y + M[9]  * z + M[13];
                wz = M[2] * x + M[6] * y + M[10] * z + M[14];
            };
            // This instance's light record: its tri range in the CDF, total area,
            // area-weighted (outward) normal sum and bounding sphere.
            const uint32_t triBegin = static_cast<uint32_t>(data.size() / 16);
            const float    cumStart = cumPower;
            float areaSum = 0.0f, nsx = 0.0f, nsy = 0.0f, nsz = 0.0f;
            constexpr float kInf = std::numeric_limits<float>::max();
            float bbMin[3] = {kInf, kInf, kInf}, bbMax[3] = {-kInf, -kInf, -kInf};
            auto bbAdd = [&](float x, float y, float z) {
                bbMin[0] = std::min(bbMin[0], x); bbMax[0] = std::max(bbMax[0], x);
                bbMin[1] = std::min(bbMin[1], y); bbMax[1] = std::max(bbMax[1], y);
                bbMin[2] = std::min(bbMin[2], z); bbMax[2] = std::max(bbMax[2], z);
            };

            const auto* indices = indexed ? idxAttr->array().data() : nullptr;
            for (uint32_t t = 0; t < triCount; ++t) {
                uint32_t i0, i1, i2;
                if (indexed) {
                    i0 = indices[t * 3 + 0];
                    i1 = indices[t * 3 + 1];
                    i2 = indices[t * 3 + 2];
                } else {
                    i0 = t * 3 + 0;
                    i1 = t * 3 + 1;
                    i2 = t * 3 + 2;
                }
                if (i0 >= vcount || i1 >= vcount || i2 >= vcount) continue;
                float w0x, w0y, w0z, w1x, w1y, w1z, w2x, w2y, w2z;
                xform(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2],
                      w0x, w0y, w0z);
                xform(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2],
                      w1x, w1y, w1z);
                xform(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2],
                      w2x, w2y, w2z);
                const float ex = w1x - w0x, ey = w1y - w0y, ez = w1z - w0z;
                const float fx = w2x - w0x, fy = w2y - w0y, fz = w2z - w0z;
                const float cx = ey * fz - ez * fy;
                const float cy = ez * fx - ex * fz;
                const float cz = ex * fy - ey * fx;
                const float area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
                if (!(area > 1e-8f)) continue;
                const float power = emLum * area;
                cumPower += power;
                areaSum += area;
                nsx += 0.5f * cx; nsy += 0.5f * cy; nsz += 0.5f * cz;// area-weighted face normal
                bbAdd(w0x, w0y, w0z); bbAdd(w1x, w1y, w1z); bbAdd(w2x, w2y, w2z);

                data.push_back(w0x); data.push_back(w0y); data.push_back(w0z);
                data.push_back(area);
                data.push_back(w1x); data.push_back(w1y); data.push_back(w1z);
                data.push_back(cumPower);
                data.push_back(w2x); data.push_back(w2y); data.push_back(w2z);
                data.push_back(power);
                data.push_back(emR); data.push_back(emG); data.push_back(emB);
                data.push_back(0.0f);
            }
            const uint32_t triCnt = static_cast<uint32_t>(data.size() / 16) - triBegin;
            if (triCnt > 0) {
                const float ex = bbMax[0] - bbMin[0], ey = bbMax[1] - bbMin[1], ez = bbMax[2] - bbMin[2];
                const float lf[16] = {0.5f * (bbMin[0] + bbMax[0]), 0.5f * (bbMin[1] + bbMax[1]),
                                      0.5f * (bbMin[2] + bbMax[2]), 0.5f * std::sqrt(ex * ex + ey * ey + ez * ez),
                                      nsx, nsy, nsz, cumStart,
                                      static_cast<float>(triBegin), static_cast<float>(triCnt),
                                      areaSum, cumPower - cumStart,
                                      emR, emG, emB, 0.0f};
                if (++lightCount <= kEmissiveCoverMaxLights) lights.insert(lights.end(), lf, lf + 16);
            }
            }// per-entry (emissive spans only)
        }

        const uint32_t triCount = static_cast<uint32_t>(data.size() / 16);
        emissiveTriCountThisFrame_ = triCount;
        emissiveTotalPowerThisFrame_ = cumPower;
        // Tail: the header (light count), then the per-light table — only under the
        // cap the shader reads it at; bigger scenes use the global pick and skip the
        // bytes. pc.emissiveCount stays the TRIANGLE count (the CDF search range).
        if (triCount > 0) {
            const float hdr[16] = {static_cast<float>(lightCount)};// rest zero
            data.insert(data.end(), hdr, hdr + 16);
            if (lightCount <= kEmissiveCoverMaxLights) data.insert(data.end(), lights.begin(), lights.end());
        }

        // Update cache regardless — non-emissive scenes still want
        // entriesUnchanged + 0-tri to short-circuit out of the walk.
        cachedEmissiveData_           = std::move(data);
        cachedEmissiveTriCount_       = triCount;
        cachedEmissiveTotalPower_     = cumPower;
        cachedEmissiveEntryCount_     = entries.size();
        cachedEmissiveVersion_++;
        // Force per-frame upload below; mark this slot as up-to-date
        // after the memcpy, leaving the other slot stale until its turn.
        for (auto& v : emissiveBufferVersion_) v = 0;

        if (triCount == 0) return false;

        const bool grew = ensureEmissiveTriCapacity(
                frame, static_cast<uint32_t>(cachedEmissiveData_.size() / 16));

        uploadHostVisible(ctx->allocator(), emissiveTriBuffers[frame],
                          cachedEmissiveData_.data(),
                          cachedEmissiveData_.size() * sizeof(float));
        emissiveBufferVersion_[frame] = cachedEmissiveVersion_;
        return grew;
    }

    // Grow motionMatBuffers[frame] in-place if the current scene's
    // instance count exceeds capacity. Returns true when the buffer
    // handle changed so the caller can rewrite binding 10. We grow with
    // 2× headroom to avoid thrashing on incremental scene growth.
    bool VulkanRenderer::Impl::ensureMotionMatCapacity(uint32_t frame, uint32_t needed) {
        if (needed <= motionMatBufferCapacity[frame]) return false;
        const uint32_t newCap = std::max<uint32_t>(needed, motionMatBufferCapacity[frame] * 2u);
        destroyBuffer(ctx->allocator(), motionMatBuffers[frame]);
        motionMatBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                /*size*/ newCap * 16 * sizeof(float),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        motionMatBufferCapacity[frame] = newCap;
        // New buffer is undefined; force the next upload to actually run.
        motionUploadedVersion_[frame] = motionScratchVersion_ - 1u;
        return true;
    }

    // Same dance as ensureMotionMatCapacity, but for the per-mesh
    // moved-bitmask SSBO at binding 21. `neededWords` is the number of
    // 32-bit words required to address every visible TLAS instance.
    bool VulkanRenderer::Impl::ensureMeshMovedBitsCapacity(uint32_t frame, uint32_t neededWords) {
        if (neededWords <= meshMovedBitsBufferCapacity[frame]) return false;
        const uint32_t newCap = std::max<uint32_t>(
                neededWords,
                static_cast<uint32_t>(meshMovedBitsBufferCapacity[frame] * 2u));
        destroyBuffer(ctx->allocator(), meshMovedBitsBuffers[frame]);
        meshMovedBitsBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                /*size*/ newCap * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        meshMovedBitsBufferCapacity[frame] = newCap;
        return true;
    }

    void VulkanRenderer::Impl::updateCameraUbo(uint32_t frame, Camera& camera) {
        camera.updateMatrixWorld(true);

        // Stash tan(fovY/2) for the DoF focal-length derivation —
        // extracted from the projection matrix (proj[1][1] = 1/tan) so
        // it works for any camera type without a PerspectiveCamera cast.
        // Except an orthographic one, where proj[1][1] is 2/(top-bottom) — a
        // length, not a tangent — and the derivation is meaningless. The DoF
        // pass is skipped for that frame (a parallel projection has no lens),
        // so the last perspective value is simply left standing.
        if (!view().orthoFrame_ && std::abs(camera.projectionMatrix.elements[5]) > 1e-6f)
            tanHalfFovY_ = 1.f / std::abs(camera.projectionMatrix.elements[5]);

        // ...and the SENSOR that FOV was derived from. threepp's
        // PerspectiveCamera already carries the film gauge (width, mm) and
        // derives the height from the aspect, so a real camera is specified
        // the real way — `cam.filmGauge = 6.3f; cam.setFocalLength(4.8f);`
        // for a 1/2.3" sensor with a 4.8 mm lens — and the DoF CoC then
        // agrees with the projection instead of assuming full frame.
        if (auto* pcam = dynamic_cast<PerspectiveCamera*>(&camera)) {
            const float filmH = pcam->getFilmHeight();// mm
            if (filmH > 1e-4f) filmHeightM_ = filmH * 1e-3f;
        }

        // Pinhole intrinsics for cameraIntrinsics() — taken from the UNJITTERED
        // projection (the TAA sub-pixel offset is applied to the UBO copy
        // below, never to camera.projectionMatrix, so this is the lens the
        // dataset should be labelled with, not this frame's dithered one).
        // elements[8]/[9] are the frustum skew, non-zero under filmOffset /
        // setViewOffset — carrying them keeps the principal point correct
        // for an off-centre sensor.
        // Multiplying the overscan factor back out recovers the OUTPUT camera:
        // render() widened proj[0]/proj[5] by 1/overscan for this frame, and
        // the intrinsics must describe the lens the user configured, not the
        // wider frustum we rendered to fill its corners. Overscan does not move
        // the principal point, so the skew terms need no correction.
        const float osc = effectiveOverscan();
        projP0_ = camera.projectionMatrix.elements[0] * osc;
        projP5_ = camera.projectionMatrix.elements[5] * osc;
        projP8_ = camera.projectionMatrix.elements[8];
        projP9_ = camera.projectionMatrix.elements[9];

        float data[40];
        std::memcpy(data + 0,  camera.matrixWorld->elements.data(), 64);
        // Reverse-Z projInverse (matches the reverse-Z VP the raster uses) so
        // depth reconstruction (deferred worldFromDepth, hybrid primary) is
        // consistent with the reverse-Z depth buffer.
        Matrix4 vkProjInv = reverseZVk(camera.projectionMatrix);
        vkProjInv.invert();
        std::memcpy(data + 16, vkProjInv.elements.data(), 64);

        // Per-frame Halton(2,3) jitter — must match what uploadRasterCameraUbo
        // computes for THIS frame so the deferred shade's hybrid primary
        // direction lands on the same surface the raster jittered to.
        // uploadRasterCameraUbo runs after this and uses the same
        // `haltonFrame_` value before incrementing, so both ubos see
        // identical jitter. Without this, the shade reconstructs V from
        // pixel-center every frame → reflection direction frozen →
        // low-roughness metals show "lines" because TAA accumulates the
        // same env tap each frame instead of integrating.
        // Render extent: the jitter is a sub-pixel offset, so the pixel
        // size in clip space (2/width) must use the resolution the deferred
        // shade + the raster gbuffer actually run at, not the swapchain extent.
        const VkExtent2D ext = renderExtent();
        uint32_t phaseCount = jitterPhaseCount_(ext, viewOutExtent());
        float jx, jy;
#if defined(THREEPP_WITH_FSR)
        // When FSR is the active upscaler, drive the jitter from FSR's own
        // sequence (its phase count tracks the render/display ratio). The raster
        // and deferred cameras both read the same haltonFrame_ index this frame,
        // so overriding both keeps their sub-pixel offset identical — and matches
        // the dispatch jitterOffset (VulkanCoreRecord). Also stash the sub-pixel
        // offset + camera near/far/vertical-FOV for the dispatch (no camera there).
        if (useFsr() && fsr_) {
            phaseCount = static_cast<uint32_t>(
                    fsr_->jitterPhaseCount(ext.width, viewOutExtent().width));
            if (phaseCount == 0u) phaseCount = 1u;
            fsr_->jitterOffset(static_cast<int>(haltonFrame_ % phaseCount),
                               static_cast<int>(phaseCount), jx, jy);
            fsrJitterX_ = jx;
            fsrJitterY_ = jy;
            if (auto* pcam = dynamic_cast<PerspectiveCamera*>(&camera)) {
                fsrCamNear_ = pcam->nearPlane;
                fsrCamFar_  = pcam->farPlane;
                fsrCamFovY_ = pcam->fov * 3.14159265f / 180.f;// vertical FOV, radians
            } else if (auto* ocam = dynamic_cast<OrthographicCamera*>(&camera)) {
                // A parallel projection has no field of view. FSR only feeds
                // fovY into its disocclusion/reactivity heuristics, so hand it
                // the angle a perspective camera would need to cover the same
                // frustum height at the far plane — the closest equivalent,
                // and one that at least scales with the zoom instead of holding
                // whatever the last perspective frame left behind.
                fsrCamNear_ = ocam->nearPlane;
                fsrCamFar_  = ocam->farPlane;
                const float halfH = 0.5f * std::abs(ocam->top - ocam->bottom);
                fsrCamFovY_ = 2.f * std::atan(halfH / std::max(ocam->farPlane, 1e-3f));
            }
        } else
#endif
        {
            const uint32_t hi = (haltonFrame_ % phaseCount) + 1u;
            jx = halton_(hi, 2) - 0.5f;
            jy = halton_(hi, 3) - 0.5f;
        }
#if defined(THREEPP_WITH_DLSS)
        // DLSS consumes the built-in Halton sequence (jitterPhaseCount_'s
        // FSR2-style scaling matches DLSS's recommended phase count); stash the
        // sub-pixel offset for the dispatch (no camera at the record site).
        // When DLSS is active useFsr() is false, so jx/jy came from the built-in
        // branch above.
        if (useDlss()) {
            dlssJitterX_ = jx;
            dlssJitterY_ = jy;
        }
#endif
        // MSAA G-buffer rasterizes UNJITTERED (see uploadRasterCameraUbo) —
        // this UBO's jitter must match or every consumer reconstructing
        // with cam.jitter (deferred_shade worldPos, hybrid V) lands
        // off the rasterized surface.
        // Event camera on its G-buffer (Shaded) source also forces UNJITTERED:
        // event_shade reads the raw raster gbuf, and per-frame Halton jitter
        // dithers silhouette coverage so a STATIC scene fires spurious +/-
        // events every frame (the "event view flickers with no motion" bug). A
        // physical DVS never sees TAA jitter. The Final source reads the
        // post-TAA frame, where jitter is already resolved, and keeps it on.
        // Must match uploadRasterCameraUbo's identical gate (eventCamReadsGbuf).
        // FSR requires jitter to reconstruct — force it on whenever FSR is the
        // active upscaler (even under MSAA, which otherwise rasterizes unjittered),
        // so the dispatch jitterOffset matches what was rendered. The gbuf-reading
        // event camera still wins. Must match uploadRasterCameraUbo.
        const bool rasterJitterOn = !eventCamReadsGbuf() &&
                                    (useFsr() || useDlss() || gbufMsaaSamples_ <= 1);
        const float jClipX = rasterJitterOn ? 2.f * jx / float(ext.width)  : 0.f;
        const float jClipY = rasterJitterOn ? 2.f * jy / float(ext.height) : 0.f;
        data[32] = jClipX;
        data[33] = jClipY;
        // .zw = previous frame's jitter so the deferred shade's hybrid
        // reproject can correct the bilinear tap by (prev - curr) pixels.
        // The raster path tracks this in rasterPrevJitter_; this camera UBO
        // upload runs before the raster upload, so rasterPrevJitter_ here
        // still holds the PREVIOUS frame's value (which is exactly what we
        // want). First frame: self-seed to curr so the delta is zero.
        data[34] = view().rasterPrevJitterValid_ ? view().rasterPrevJitter_[0] : jClipX;
        data[35] = view().rasterPrevJitterValid_ ? view().rasterPrevJitter_[1] : jClipY;

        // camAux — the two things a shader cannot get from the matrices alone
        // cheaply. .x flags a PARALLEL projection: under one the primary rays
        // share a direction and each pixel has its OWN origin, so every ray the
        // shade builds from the eye point (view vector, fog leg, sky ray, cloud
        // march) has to be rebuilt from that pixel's ray instead. .yzw is the
        // camera's world-space forward, which IS that shared direction — the
        // -Z column of matrixWorld (col2 points backward). Zero-length only for
        // a degenerate camera matrix; the shaders never read it unless .x is 1.
        {
            const auto& wme = camera.matrixWorld->elements;
            float fx = -wme[8], fy = -wme[9], fz = -wme[10];
            const float len = std::sqrt(fx * fx + fy * fy + fz * fz);
            if (len > 1e-12f) { fx /= len; fy /= len; fz /= len; }
            data[36] = view().orthoFrame_ ? 1.f : 0.f;
            data[37] = fx;
            data[38] = fy;
            data[39] = fz;
        }

        // Build camera basis-vector buffer matching PrevCameraUbo layout.
        // matrixWorld column-major: col0=right, col1=up, col2=backward(-fwd), col3=pos.
        const auto& wm = camera.matrixWorld->elements;
        const auto& pm = camera.projectionMatrix.elements;
        float curBuf[16];
        curBuf[0] = wm[12]; curBuf[1] = wm[13]; curBuf[2] = wm[14]; curBuf[3] = pm[0];
        curBuf[4] =-wm[ 8]; curBuf[5] =-wm[ 9]; curBuf[6] =-wm[10]; curBuf[7] = pm[5];
        curBuf[8] = wm[ 0]; curBuf[9] = wm[ 1]; curBuf[10]= wm[ 2]; curBuf[11]= 0.0f;
        curBuf[12]= wm[ 4]; curBuf[13]= wm[ 5]; curBuf[14]= wm[ 6]; curBuf[15]= 0.0f;

        // Camera-motion detection: position [0..2], forward [4..6], and
        // projection scale [3]=projScaleX / [7]=projScaleY. The projection
        // terms catch FOV and aspect-ratio changes that don't move the camera.
        if (view().prevCameraValid) {
            const float dx = curBuf[0] - view().prevCamBufData_[0];
            const float dy = curBuf[1] - view().prevCamBufData_[1];
            const float dz = curBuf[2] - view().prevCamBufData_[2];
            const float fx = curBuf[4] - view().prevCamBufData_[4];
            const float fy = curBuf[5] - view().prevCamBufData_[5];
            const float fz = curBuf[6] - view().prevCamBufData_[6];
            const float sx = curBuf[3] - view().prevCamBufData_[3];// projScaleX
            const float sy = curBuf[7] - view().prevCamBufData_[7];// projScaleY
            // Position threshold lowered 1e-6 → 1e-10 (|Δpos| 1e-3 → 1e-5
            // world units). The old 1e-3 was far too coarse for PURE
            // TRANSLATION: a slow pan moves the camera only ~1e-4 units/frame,
            // which fell under 1e-3 → cameraMoved stayed false → the
            // accumulator took the SELF-TAP path (reuse prev accum at the
            // SAME pixel, full FC). But the camera HAD moved, so that reused
            // image sat slightly off and dragged with the camera — the "old
            // image follows the camera" smear, textured-surface-only because
            // uniform albedo hides a misplaced reuse. Orbit never hit this:
            // rotation moves the forward vector, which trips the far more
            // sensitive 1e-8 rotation threshold, so orbit always took the
            // correct reproject path. 1e-5 units is still well above static
            // float noise (an untouched camera has an identical matrix →
            // exactly zero delta) but catches the slowest deliberate pan.
            if (dx*dx + dy*dy + dz*dz > 1e-10f ||
                fx*fx + fy*fy + fz*fz > 1e-8f ||
                sx*sx + sy*sy > 1e-10f) {
                // sampleIndex must keep advancing — see comment in
                // ensureSceneBuilt's matrix-changed path. Per-pixel FC
                // halving handles convergence; freezing the seed kills
                // Monte Carlo variance reduction.
            }
        }

        // Update prev camera data (CPU-side only — read by the live
        // prev-camera path via prevCamBufData_, not a GPU buffer).
        std::memcpy(view().prevCamBufData_.data(), curBuf, 16 * sizeof(float));
        view().prevCameraValid = true;

        uploadHostVisible(ctx->allocator(), view().cameraUbos[frame], data, sizeof(data));
    }

    void VulkanRenderer::Impl::uploadRasterCameraUbo(uint32_t frame, Camera& camera) {
        camera.updateMatrixWorld(true);

        // VP_unjittered = projection * viewMat, viewMat = matrixWorldInverse.
        Matrix4 viewMat, proj;
        std::memcpy(viewMat.elements.data(),
                    camera.matrixWorldInverse.elements.data(), 64);
        // Reverse-Z projection (near→1, far→0) — feeds the gbuffer VP AND
        // currVPunjit_ (overlay depth prepass). Depth clear + compares flipped
        // to match (see recordRasterGbufPass clears + pipeline depthCompareOp).
        proj = reverseZVk(camera.projectionMatrix);
        Matrix4 vpUnj;
        vpUnj.multiplyMatrices(proj, viewMat);

        // Render extent — jitter + the .zw = 1/resolution this writes
        // into the raster camera UBO must match the resolution the
        // gbuffer rasterizes at (see updateCameraUbo for the rationale).
        const VkExtent2D ext = renderExtent();
        // Sub-pixel offset in [-0.5, +0.5] per axis. Halton(2,3) is the
        // industry-standard low-discrepancy sequence for primary AA.
        // Phase count scales with the upscale ratio (FSR2-style) so the
        // sequence covers the output grid when renderScale < 1; matches
        // updateCameraUbo's value this frame (same extents → same count).
        uint32_t phaseCount = jitterPhaseCount_(ext, viewOutExtent());
        float jx, jy;
#if defined(THREEPP_WITH_FSR)
        // When FSR is the active upscaler, drive the jitter from FSR's own
        // sequence (its phase count tracks the render/display ratio). The raster
        // and deferred cameras both read the same haltonFrame_ index this frame,
        // so overriding both keeps their sub-pixel offset identical — and matches
        // the dispatch jitterOffset (VulkanCoreRecord). Also stash the sub-pixel
        // offset + camera near/far/vertical-FOV for the dispatch (no camera there).
        if (useFsr() && fsr_) {
            phaseCount = static_cast<uint32_t>(
                    fsr_->jitterPhaseCount(ext.width, viewOutExtent().width));
            if (phaseCount == 0u) phaseCount = 1u;
            fsr_->jitterOffset(static_cast<int>(haltonFrame_ % phaseCount),
                               static_cast<int>(phaseCount), jx, jy);
            fsrJitterX_ = jx;
            fsrJitterY_ = jy;
            if (auto* pcam = dynamic_cast<PerspectiveCamera*>(&camera)) {
                fsrCamNear_ = pcam->nearPlane;
                fsrCamFar_  = pcam->farPlane;
                fsrCamFovY_ = pcam->fov * 3.14159265f / 180.f;// vertical FOV, radians
            } else if (auto* ocam = dynamic_cast<OrthographicCamera*>(&camera)) {
                // A parallel projection has no field of view. FSR only feeds
                // fovY into its disocclusion/reactivity heuristics, so hand it
                // the angle a perspective camera would need to cover the same
                // frustum height at the far plane — the closest equivalent,
                // and one that at least scales with the zoom instead of holding
                // whatever the last perspective frame left behind.
                fsrCamNear_ = ocam->nearPlane;
                fsrCamFar_  = ocam->farPlane;
                const float halfH = 0.5f * std::abs(ocam->top - ocam->bottom);
                fsrCamFovY_ = 2.f * std::atan(halfH / std::max(ocam->farPlane, 1e-3f));
            }
        } else
#endif
        {
            const uint32_t hi = (haltonFrame_ % phaseCount) + 1u;
            jx = halton_(hi, 2) - 0.5f;
            jy = halton_(hi, 3) - 0.5f;
        }
#if defined(THREEPP_WITH_DLSS)
        // DLSS consumes the built-in Halton sequence; stash the sub-pixel offset
        // for the dispatch. Matches updateCameraUbo's identical stash (same
        // haltonFrame_ this frame → same offset).
        if (useDlss()) {
            dlssJitterX_ = jx;
            dlssJitterY_ = jy;
        }
#endif
        // Map sub-pixel offset to clip-space: one pixel spans 2/width of
        // NDC (NDC ∈ [-1, +1]), so a 1-pixel jitter is 2/width in clip x.
        //
        // Raster jitter trade-off: ON gives sub-pixel-offset texture AA
        // on interior surfaces via the deferred shade's temporal blend, but
        // the per-frame Halton coverage flip at silhouettes shows up as
        // black-pixel stippling on moving objects (1-spp env taps on the
        // "uncovered" half-frame). Without MSAA on the gbuffer pass, that
        // aliasing costs more than the interior AA gains. Disabled by
        // default; sub-pixel jitter still happens in the deferred shade's
        // hybrid primary via the blue-noise tile (see primaryDirHybrid), so
        // interior AA doesn't disappear — only the coverage jitter is gone.
        // Raster jitter ON: sub-pixel Halton coverage +
        // temporal accumulation = proper TAA/TSR (replaces FXAA). The earlier
        // "flicker" was silhouette coverage-flip on MOVING objects with a loose
        // RGB AABB history clamp; the resolve now uses a YCoCg variance clip
        // (tighter history rejection) to suppress it. Static views converge to
        // clean AA over the 16-frame Halton cycle.
        constexpr bool kRasterJitterEnabled = true;
        // MSAA G-buffer (gbufMsaaSamples_ > 1): rasterize UNJITTERED. The
        // hardware sample coverage replaces jitter as the geometry AA, and
        // an unjittered raster makes the dominant-sample resolve — and so
        // every edge pixel's surface classification and reconstruction
        // position — CONSTANT on a static viewMat. With jitter on, the whole
        // MSAA sample cloud translates as a unit each frame, so the
        // majority vote flips nearly as often as a point sample (measured:
        // dominant-of-4 under jitter cut edge flicker only ~7 %).
        // (Measured: keeping the jitter at msaa>1 for the renderScale<1
        // upsampler does NOT work — the upsampler re-amplifies the
        // jittered coverage flips and edge flicker returns at full
        // strength, 6.7k px/frame vs 5.1k at msaa=1. Unjittered MSAA
        // upsampling is STABLE; its trade is spatial softness, not noise.)
        // Event camera on its G-buffer (Shaded) source forces UNJITTERED (must
        // match updateCameraUbo's gate, eventCamReadsGbuf): event_shade reads
        // this raw gbuf, and the per-frame Halton coverage flip at silhouettes
        // makes a STATIC scene emit spurious events every frame — the DVS
        // "flickers with no motion". A real event camera sees no jitter;
        // transform/camera motion still flows through motionMat, so genuine
        // motion events are unaffected. The Final source reads the post-TAA
        // frame (jitter resolved) and leaves it on.
        // useFsr()/useDlss() force jitter on: both need it to reconstruct, even
        // under MSAA (which otherwise rasterizes unjittered), so the dispatch
        // jitterOffset matches the render. The gbuf-reading event camera still
        // wins (a real DVS sees no jitter).
        const bool rasterJitterOn =
                kRasterJitterEnabled && !eventCamReadsGbuf() &&
                (useFsr() || useDlss() || gbufMsaaSamples_ <= 1);
        const float jClipX = rasterJitterOn ? 2.f * jx / float(ext.width)  : 0.f;
        const float jClipY = rasterJitterOn ? 2.f * jy / float(ext.height) : 0.f;
        // Stash the raw texel-unit jitter for the TAA resolve's current-
        // sample jitter cancellation (recordCommandBuffer passes it into
        // recordResolve; see taaJitterTexels_'s member comment). Zero when
        // unjittered so the resolve collapses to its exact pre-fix math.
        view().taaJitterTexels_[0] = rasterJitterOn ? jx : 0.f;
        view().taaJitterTexels_[1] = rasterJitterOn ? jy : 0.f;

        // Apply jitter by shifting the projection matrix's m02/m12 (the
        // entries that translate the projected NDC). For a column-major
        // 4x4 stored in elements[c*4 + r], that's elements[8] (col=2,row=0)
        // and elements[9] (col=2,row=1). Those entries multiply the VIEW-space
        // z on the way into clip.x/clip.y, and clip.w is -z_view under a
        // perspective projection — so the shear lands as a constant -jClip in
        // NDC at every depth (the sign the deferred shade undoes by ADDING
        // cam.jitter.xy back to the pixel's NDC).
        //
        // A PARALLEL projection has clip.w == 1, so those same entries would
        // scale with z_view instead: a sub-pixel offset that grows with
        // distance. The translation entries m03/m13 — elements[12] and [13] —
        // are the ones that shift NDC by a constant there, and they land it
        // with the OPPOSITE sign (they add straight into clip, no negated w to
        // divide by), so the offset is negated to keep the NDC shift identical
        // in both. Getting this sign wrong is not subtle: the shade then
        // reconstructs worldPos a pixel off the surface it shades, its ambient
        // rays start under the geometry and every one of them is occluded —
        // black bands across a lit ground.
        Matrix4 projJ;
        std::memcpy(projJ.elements.data(), proj.elements.data(), 64);
        if (view().orthoFrame_) {
            projJ.elements[12] -= jClipX;
            projJ.elements[13] -= jClipY;
        } else {
            projJ.elements[8]  += jClipX;
            projJ.elements[9]  += jClipY;
        }
        Matrix4 vpJ;
        vpJ.multiplyMatrices(projJ, viewMat);

        RasterCameraData ubo{};
        std::memcpy(ubo.currVPjittered,   vpJ.elements.data(),  64);
        std::memcpy(ubo.currVPunjittered, vpUnj.elements.data(), 64);
        // Mirror to Impl-level cache for recordOverlayPass — the overlay
        // pass runs in recordCommandBuffer which doesn't have direct
        // access to the camera; it needs the same unjittered VP that
        // the raster prepass + TAA used so wireframes register pixel-
        // exact with the post-TAA rendered silhouette.
        std::memcpy(view().currVPunjit_.data(), vpUnj.elements.data(), 64);
        // Mirror the unjittered viewMat + reverse-Z projection separately for
        // the particle billboard pass (it can't use the combined VP — see
        // currViewUnjit_/currProjUnjit_).
        std::memcpy(view().currViewUnjit_.data(), viewMat.elements.data(), 64);
        std::memcpy(view().currProjUnjit_.data(), proj.elements.data(), 64);
        // First frame: self-seed prevVP so motion vectors are zero. The
        // following frame picks up the real history.
        std::memcpy(ubo.prevVP,
                    view().rasterPrevVPValid_ ? view().rasterPrevVP_ : vpUnj.elements.data(),
                    64);
        ubo.jitter[0] = jClipX;
        ubo.jitter[1] = jClipY;
        ubo.jitter[2] = 1.f / float(ext.width);
        ubo.jitter[3] = 1.f / float(ext.height);
        // First frame: self-seed prev jitter to curr so motion vec for
        // static surfaces is exactly zero (no spurious offset on cold start).
        ubo.prevJitter[0] = view().rasterPrevJitterValid_ ? view().rasterPrevJitter_[0] : jClipX;
        ubo.prevJitter[1] = view().rasterPrevJitterValid_ ? view().rasterPrevJitter_[1] : jClipY;
        // .z: normal-map Toksvig spec-AA toggle (gbuffer.frag), packed here
        // because prevJitter.zw was otherwise unread by any gbuffer shader —
        // see normalMapToksvig_. .w: frame counter for the alpha-blend
        // screen-door's per-frame decorrelation (alphaHash). The hash used to
        // fold cam.jitter alone, but rasterJitterOn above zeroes the jitter in
        // exactly the msaa>1 non-upscaler case — the dither froze bit-identical
        // every frame and the screen-door never converged under TAA. Wrapped at
        // 1024 so the float is always exact.
        ubo.prevJitter[2] = normalMapToksvig_ ? 1.f : 0.f;
        ubo.prevJitter[3] = static_cast<float>(haltonFrame_ & 1023u);

        uploadHostVisible(ctx->allocator(), view().rasterCameraUbos[frame], &ubo, sizeof(ubo));

        // Far-plane reprojection for the TAA sky path — must be built
        // BEFORE rasterPrevVP_ rolls over to this frame's VP.
        {
            Matrix4 invCurr;
            invCurr.copy(vpUnj).invert();
            Matrix4 prevVPm;
            std::memcpy(prevVPm.elements.data(),
                        view().rasterPrevVPValid_ ? view().rasterPrevVP_ : vpUnj.elements.data(), 64);
            Matrix4 sky;
            sky.multiplyMatrices(prevVPm, invCurr);
            std::memcpy(view().taaSkyReproj_.data(), sky.elements.data(), 64);
        }
        // Reverse-Z viewMat-depth linearization (A,B,C,D) for the TAA depth
        // disocclusion gate: viewZ = (A·d + B)/(C·d + D), from the inverse
        // reverse-Z projection's z/w rows. Mirrors deferred_shade's
        // cam.projInverse Z unprojection so both passes gate identically.
        {
            Matrix4 projInv;
            projInv.copy(proj).invert();
            const auto& pe = projInv.elements;// column-major [col*4 + row]
            view().taaDepthLin_ = {pe[10], pe[14], pe[11], pe[15]};
        }
        std::memcpy(view().rasterPrevVP_, vpUnj.elements.data(), 64);
        view().rasterPrevVPValid_ = true;
        view().rasterPrevJitter_[0] = jClipX;
        view().rasterPrevJitter_[1] = jClipY;
        view().rasterPrevJitterValid_ = true;
        // Camera world motion (translation + forward rotation) for the
        // deferred reflection policy — see the member comment.
        {
            const auto& mw = camera.matrixWorld->elements;
            const float pos[3] = {mw[12], mw[13], mw[14]};
            const float fwd[3] = {-mw[8], -mw[9], -mw[10]};// -Z column = viewMat dir
            if (view().deferredCamPrevValid_) {
                const float dx = pos[0] - view().deferredCamPrevPos_[0];
                const float dy = pos[1] - view().deferredCamPrevPos_[1];
                const float dz = pos[2] - view().deferredCamPrevPos_[2];
                view().deferredCamDeltaLen_ = std::sqrt(dx * dx + dy * dy + dz * dz);
                const float c = std::clamp(fwd[0] * view().deferredCamPrevFwd_[0] +
                                                   fwd[1] * view().deferredCamPrevFwd_[1] +
                                                   fwd[2] * view().deferredCamPrevFwd_[2],
                                           -1.f, 1.f);
                deferredCamRotAngle_ = std::acos(c);
            } else {
                view().deferredCamDeltaLen_ = 0.f;
                deferredCamRotAngle_ = 0.f;
            }
            std::memcpy(view().deferredCamPrevPos_, pos, sizeof(pos));
            std::memcpy(view().deferredCamPrevFwd_, fwd, sizeof(fwd));
            view().deferredCamPrevValid_ = true;
        }
        // Free-running jitter index. The active Halton period is derived
        // per-read from the upscale ratio (jitterPhaseCount_) and applied as
        // a modulo, so the sequence length tracks renderScale instead of a
        // fixed 16. uint32 wrap is harmless — the modulo re-bases each frame.
        haltonFrame_ = haltonFrame_ + 1u;

        // Refresh the per-frame descriptor set. Bindings 0-2 (UBO,
        // motionMat, matDescs) are tiny, and motionMat/matDescs handles can
        // grow (handle change) when the scene instance/material count
        // increases — rewriting them every frame absorbs that automatically.
        VkDescriptorBufferInfo ubInfo{};
        ubInfo.buffer = view().rasterCameraUbos[frame].handle;
        ubInfo.offset = 0;
        ubInfo.range  = sizeof(RasterCameraData);

        VkDescriptorBufferInfo mmInfo{};
        mmInfo.buffer = motionMatBuffers[frame].handle;
        mmInfo.offset = 0;
        mmInfo.range  = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo matsInfo{};
        matsInfo.buffer = materialDescsBuffers[frame].handle;
        matsInfo.offset = 0;
        matsInfo.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = view().rasterDescSets[frame];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo     = &ubInfo;
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = view().rasterDescSets[frame];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &mmInfo;
        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = view().rasterDescSets[frame];
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &matsInfo;
        uint32_t writeCount = 3;

        // Binding 3 — the bindless material-texture array (same VkImage/
        // VkSampler handles the deferred shade sees at its own binding 8,
        // white-default padded). Its contents are identical frame-to-frame and only
        // change when the scene texture table is rebuilt, which sets
        // rasterMatTexValid_ to 0 (see the member declaration). Skip the
        // 2048-element write + the ~48 KB fillMaterialTextureInfos array
        // fill on every frame the table is unchanged for this slot (the
        // common case). matTexInfos must outlive the vkUpdateDescriptorSets
        // call below, so it is declared here in function scope.
        std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
        if (view().rasterMatTexValid_[frame] == 0) {
            fillMaterialTextureInfos(matTexInfos);
            writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet          = view().rasterDescSets[frame];
            writes[3].dstBinding      = 3;
            writes[3].dstArrayElement = 0;
            writes[3].descriptorCount = kMaxMaterialTextures;
            writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo      = matTexInfos.data();
            writeCount = 4;
            view().rasterMatTexValid_[frame] = 1;
        }
        vkUpdateDescriptorSets(ctx->device(), writeCount, writes, 0, nullptr);
    }

    void VulkanRenderer::Impl::createLightsUbos() {
        for (auto& b : lightsUbos) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(GpuLightsUbo),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
        // Clustered-light buffers: the full point/spot list (CPU-filled)
        // + the per-cell index grid (cluster_build.comp writes it).
        for (auto& b : clusterLightsBuffers) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(GpuClusterLight) * kMaxClusterLights,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
        for (auto& b : clusterGridBuffers) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(uint32_t) * kClusterCells * (kClusterMaxPerCell + 1),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    0);// device-local: GPU write (cull) → GPU read (shade)
        }
    }

    // Walk the scene each frame for AmbientLight + DirectionalLight; pack
    // into the per-frame lights UBO. Direction is computed from the
    // light's world-space position toward its (possibly defaulted) target,
    // mirroring three.js's DirectionalLight.target convention. The shader
    // expects the L vector (toward the light), so we negate.
    void VulkanRenderer::Impl::updateLightsUbo(uint32_t frame, Object3D& scene) {
        // force=false: ensureSceneBuilt already brought the graph current
        // this frame — the forced variant here was a SECOND full
        // world-multiply pass over every node, every frame.
        scene.updateMatrixWorld();

        GpuLightsUbo ubo{};

        // Uncapped point/spot collection for the clustered path. The cull
        // radius bounds each light's influence for the cluster test: the
        // authored range when set, else the distance where the 1/d^decay
        // falloff drops below ~0.5% of the peak (range 0 = infinite in
        // three.js — a literal infinity would put every light in every
        // cell). Zero-power lights get cullRadius 0 → in no cell.
        std::vector<GpuClusterLight> clusterCollect;
        clusterCollect.reserve(16);
        const auto clusterCullRadius = [](const GpuClusterLight& r) {
            if (r.range > 0.f) return r.range;
            const float L = std::max({r.color[0], r.color[1], r.color[2]});
            if (L <= 0.f) return 0.f;
            const float d = std::max(r.decay, 0.25f);
            return std::min(std::pow(L / 0.005f, 1.f / d), 500.f);
        };

        // Physical light units (setPhysicalLightUnits): point/spot
        // intensities are luminous FLUX (lumens) and convert to the
        // intensity the falloff math expects — candela Φ/4π for points,
        // Φ/π for spots (Frostbite's convention: brightness is invariant
        // under cone-angle edits). Dir lights are lux and rect lights
        // nits — both already the unit the shading math consumes, so
        // they pass through. Legacy mode: all scales 1.0, byte-identical.
        const float pointUnitScale = physicalLightUnits_ ? 1.f / (4.f * math::PI) : 1.f;
        const float spotUnitScale  = physicalLightUnits_ ? 1.f / math::PI : 1.f;

        // traverseVisible so a hidden parent prunes its child lights too.
        scene.traverseVisible([&](Object3D& o) {
            // One RTTI probe instead of five for the ~99% of nodes that are not
            // lights: every type the chain below handles derives from Light, so
            // a failed cast here cannot skip a branch that would have matched.
            // (LightProbe passes this gate and falls through the chain
            // unhandled, exactly as before.)
            if (!dynamic_cast<Light*>(&o)) return;
            if (auto* a = dynamic_cast<AmbientLight*>(&o)) {
                ubo.ambient[0] += a->color.r * a->intensity;
                ubo.ambient[1] += a->color.g * a->intensity;
                ubo.ambient[2] += a->color.b * a->intensity;
            } else if (auto* hl = dynamic_cast<HemisphereLight*>(&o)) {
                // GL parity (Lights.cpp / lights_pars_begin.glsl): irradiance
                // = mix(ground, sky, 0.5*dot(N,up)+0.5), up = the light's
                // world position direction (three.js puts a hemi at (0,1,0)
                // by default). Split as mean + zero-mean remainder: the MEAN
                // rides ubo.ambient so every isotropic ambient consumer
                // (fog, clouds, water, particles, probes) sees the energy
                // through its existing prefix-declared block; the remainder
                // 0.5*(sky-ground)*dot(N,up) goes into the hemiDelta rows
                // (per channel, so several hemis fold exactly) and only the
                // deferred surface shade evaluates it.
                Vector3 up;
                hl->getWorldPosition(up);
                if (up.lengthSq() < 1e-12f) up.set(0.f, 1.f, 0.f);
                up.normalize();
                const float upv[3] = {up.x, up.y, up.z};
                const float sky[3] = {hl->color.r * hl->intensity,
                                      hl->color.g * hl->intensity,
                                      hl->color.b * hl->intensity};
                const float gnd[3] = {hl->groundColor.r * hl->intensity,
                                      hl->groundColor.g * hl->intensity,
                                      hl->groundColor.b * hl->intensity};
                for (int c = 0; c < 3; ++c) {
                    ubo.ambient[c] += 0.5f * (sky[c] + gnd[c]);
                    ubo.hemiDeltaR[c] += 0.5f * (sky[0] - gnd[0]) * upv[c];
                    ubo.hemiDeltaG[c] += 0.5f * (sky[1] - gnd[1]) * upv[c];
                    ubo.hemiDeltaB[c] += 0.5f * (sky[2] - gnd[2]) * upv[c];
                }
            } else if (auto* dl = dynamic_cast<DirectionalLight*>(&o)) {
                if (ubo.dirCount >= kMaxDirLights) return;
                Vector3 lp, tp;
                dl->getWorldPosition(lp);
                const_cast<Object3D&>(dl->target()).getWorldPosition(tp);
                Vector3 toLight = lp.sub(tp);
                if (toLight.lengthSq() < 1e-12f) toLight.set(0.f, 1.f, 0.f);
                toLight.normalize();
                auto& g = ubo.dirLights[ubo.dirCount++];
                g.direction[0] = toLight.x; g.direction[1] = toLight.y; g.direction[2] = toLight.z;
                g.color[0] = dl->color.r * dl->intensity;
                g.color[1] = dl->color.g * dl->intensity;
                g.color[2] = dl->color.b * dl->intensity;
            } else if (auto* pl = dynamic_cast<PointLight*>(&o)) {
                // Point/spot lights are COLLECTED (uncapped) for the
                // clustered path; the UBO's 8-per-type slots are filled
                // from the power-sorted list after the traverse.
                GpuClusterLight rec{};
                Vector3 wp; pl->getWorldPosition(wp);
                rec.position[0] = wp.x; rec.position[1] = wp.y; rec.position[2] = wp.z;
                rec.range = pl->distance;
                rec.color[0] = pl->color.r * pl->intensity * pointUnitScale;
                rec.color[1] = pl->color.g * pl->intensity * pointUnitScale;
                rec.color[2] = pl->color.b * pl->intensity * pointUnitScale;
                rec.decay = pl->decay;
                rec.direction[2] = -1.f;
                rec.cosAngleOuter = -1.1f;// cone sentinel: smoothstep(-1.1,-1.05,cos≥-1) = 1
                rec.cosAngleInner = -1.05f;
                rec.radius = pl->radius;
                rec.cullRadius = clusterCullRadius(rec);
                rec.type = 0.f;
                clusterCollect.push_back(rec);
            } else if (auto* sl = dynamic_cast<SpotLight*>(&o)) {
                Vector3 lp, tp;
                sl->getWorldPosition(lp);
                const_cast<Object3D&>(sl->target()).getWorldPosition(tp);
                Vector3 emDir = tp - lp;
                if (emDir.lengthSq() < 1e-12f) emDir.set(0.f, -1.f, 0.f);
                emDir.normalize();
                GpuClusterLight rec{};
                rec.position[0] = lp.x; rec.position[1] = lp.y; rec.position[2] = lp.z;
                rec.range = sl->distance;
                rec.color[0] = sl->color.r * sl->intensity * spotUnitScale;
                rec.color[1] = sl->color.g * sl->intensity * spotUnitScale;
                rec.color[2] = sl->color.b * sl->intensity * spotUnitScale;
                rec.decay = sl->decay;
                rec.direction[0] = emDir.x; rec.direction[1] = emDir.y; rec.direction[2] = emDir.z;
                rec.cosAngleOuter = std::cos(sl->angle);
                rec.cosAngleInner = std::cos(sl->angle * (1.0f - sl->penumbra));
                rec.radius = sl->radius;
                rec.cullRadius = clusterCullRadius(rec);
                rec.type = 1.f;
                clusterCollect.push_back(rec);
            } else if (auto* rl = dynamic_cast<RectAreaLight*>(&o)) {
                if (ubo.rectCount >= kMaxRectLights) return;
                Vector3 wp; rl->getWorldPosition(wp);
                const auto& el = rl->matrixWorld->elements;
                // Column-major: col0=localX, col1=localY, col2=localZ (each with scale).
                Vector3 worldX(el[0], el[1], el[2]); worldX.normalize();
                Vector3 worldY(el[4], el[5], el[6]); worldY.normalize();
                Vector3 worldZ(el[8], el[9], el[10]); worldZ.normalize();
                auto& g = ubo.rectLights[ubo.rectCount++];
                g.position[0] = wp.x; g.position[1] = wp.y; g.position[2] = wp.z;
                const float hw = rl->width  * 0.5f;
                const float hh = rl->height * 0.5f;
                g.halfU[0] = worldX.x * hw; g.halfU[1] = worldX.y * hw; g.halfU[2] = worldX.z * hw;
                g.halfV[0] = worldY.x * hh; g.halfV[1] = worldY.y * hh; g.halfV[2] = worldY.z * hh;
                // RectAreaLight emits toward -Z in local space.
                g.normal[0] = -worldZ.x; g.normal[1] = -worldZ.y; g.normal[2] = -worldZ.z;
                g.color[0] = rl->color.r * rl->intensity;
                g.color[1] = rl->color.g * rl->intensity;
                g.color[2] = rl->color.b * rl->intensity;
            }
        });

        // Power-sort the point/spot list (strongest first) so BOTH the
        // UBO's 8-per-type slots (legacy paths: PT, reflection/GI hits,
        // volumetric beams, probes) and an overflowing cluster cell keep
        // the most important lights. Then mirror the top 8 of each type
        // into the UBO exactly as the pre-cluster path did.
        const auto lightLum = [](const GpuClusterLight& r) {
            return 0.2126f * r.color[0] + 0.7152f * r.color[1] + 0.0722f * r.color[2];
        };
        std::stable_sort(clusterCollect.begin(), clusterCollect.end(),
                         [&](const GpuClusterLight& a, const GpuClusterLight& b) {
                             return lightLum(a) > lightLum(b);
                         });
        if (clusterCollect.size() > kMaxClusterLights)
            clusterCollect.resize(kMaxClusterLights);
        for (const auto& rec : clusterCollect) {
            if (rec.type < 0.5f) {
                if (ubo.pointCount >= kMaxPointLights) continue;
                auto& g = ubo.pointLights[ubo.pointCount++];
                std::memcpy(g.position, rec.position, sizeof(g.position));
                g.range = rec.range;
                std::memcpy(g.color, rec.color, sizeof(g.color));
                g.decay  = rec.decay;
                g.radius = rec.radius;
            } else {
                if (ubo.spotCount >= kMaxSpotLights) continue;
                auto& g = ubo.spotLights[ubo.spotCount++];
                std::memcpy(g.position, rec.position, sizeof(g.position));
                g.range = rec.range;
                std::memcpy(g.color, rec.color, sizeof(g.color));
                g.decay = rec.decay;
                std::memcpy(g.direction, rec.direction, sizeof(g.direction));
                g.cosAngleOuter = rec.cosAngleOuter;
                g.cosAngleInner = rec.cosAngleInner;
                g.radius = rec.radius;
            }
        }
        clusterLightCountThisFrame_ = static_cast<uint32_t>(clusterCollect.size());
        if (!clusterCollect.empty()) {
            uploadHostVisible(ctx->allocator(), clusterLightsBuffers[frame],
                              clusterCollect.data(),
                              sizeof(GpuClusterLight) * clusterCollect.size());
        }

        // Extracted HDRI sun → analytic directional light (deferred leaf
        // only). The PMREM's mips 1+ were built sun-free, so this light
        // carries the disc's exact energy (colorE = Σ L·dΩ) instead:
        // correct sharp GGX highlight, RT shadows (jittered soft via
        // sunAngularRadiusDeg_), GI bounce, water glints, volumetric
        // shafts — all through the existing dir-light paths. Injected
        // BEFORE the hash below so toggling extraction resets accumulation.
        //
        // ONE-SUN POLICY: at this point ubo.dirCount holds exactly the
        // scene's own visible DirectionalLights (the traverse above). If
        // the scene provides one, IT is the sun — scenes authored for
        // raster renderers carry a stand-in sun light, and injecting the
        // env's sun on top lit and shadowed everything TWICE (the double
        // directional shadow report). Auto defers; Always still injects.
        // Hiding/showing a scene sun re-evaluates here every frame, and
        // the UBO hash below resets accumulation on the change.
        const bool sceneHasSun = ubo.dirCount > 0;
        if (envSunExtractionWanted() && envSun_.found && ubo.dirCount < kMaxDirLights &&
            !(envSunDefersToSceneSun() && sceneHasSun)) {
            auto& g = ubo.dirLights[ubo.dirCount++];
            g.direction[0] = envSun_.dir[0];
            g.direction[1] = envSun_.dir[1];
            g.direction[2] = envSun_.dir[2];
            g.color[0] = envSun_.colorE[0];
            g.color[1] = envSun_.colorE[1];
            g.color[2] = envSun_.colorE[2];
        }

        // ── 4c: the billboard slice's copy of the sun ───────────────────────
        // ParticleFieldPass has no descriptor set by design, so a `lit`
        // BillboardRepr cannot read the UBO above — it gets three floats in its
        // per-view record instead. Taken from the FINISHED ubo, after the
        // one-sun policy has decided between the scene's own DirectionalLight
        // and the extracted HDRI sun, so the sprites are lit by exactly the sun
        // the deferred path shades with. Brightest by luminance-ish max, which
        // for the single-sun scenes this serves is simply "the sun".
        {
            bbAmbient_[0] = ubo.ambient[0];
            bbAmbient_[1] = ubo.ambient[1];
            bbAmbient_[2] = ubo.ambient[2];
            float best = -1.f;
            bbSunRadiance_[0] = bbSunRadiance_[1] = bbSunRadiance_[2] = 0.f;
            for (std::uint32_t i = 0; i < ubo.dirCount; ++i) {
                const auto& g = ubo.dirLights[i];
                const float p = std::max({g.color[0], g.color[1], g.color[2]});
                if (p <= best) continue;
                best = p;
                bbSunDirWorld_[0] = g.direction[0];
                bbSunDirWorld_[1] = g.direction[1];
                bbSunDirWorld_[2] = g.direction[2];
                bbSunRadiance_[0] = g.color[0];
                bbSunRadiance_[1] = g.color[1];
                bbSunRadiance_[2] = g.color[2];
            }
        }

        uploadHostVisible(ctx->allocator(), lightsUbos[frame], &ubo, sizeof(ubo));
    }

    void VulkanRenderer::Impl::createFogUbos() {
        for (auto& b : fogUbos) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(GpuFogUbo),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    // Pack scene.fog (Fog/FogExp2 variant) into the per-frame fog UBO.
    // FogExp2.density maps directly
    // to sigma_t; linear Fog reaches ~63% extinction at farPlane via
    // sigma = 1 / (far - near). Hash detect changes so the per-pixel motion
    // path halves FC and the new fog state converges quickly.
    void VulkanRenderer::Impl::updateFogUbo(uint32_t frame, Object3D& scene, Camera& camera) {
        GpuFogUbo ubo{};
        // World up (for the sky aerial-perspective fog band) — these scenes set
        // camera.up to the world up (Z-up for the Spot stack, Y-up elsewhere).
        ubo.worldUp[0] = camera.up.x;
        ubo.worldUp[1] = camera.up.y;
        ubo.worldUp[2] = camera.up.z;

        // scene.fog → the homogeneous UBO fields (medium beam-σ / albedo /
        // present-flag) that the volumetric consumers read. These are the AIR
        // medium's params; the froxel hetero path (hf* below) carries the actual
        // extinction + in-scatter, but the spot-beam march + sky band still read
        // sigmaT/color/enabled here.
        bool  fogPresent = false;
        float sigma = 0.f;
        if (auto* sc = dynamic_cast<Scene*>(&scene); sc && sc->fog.has_value()) {
            Color tint{1.f, 1.f, 1.f};
            if (std::holds_alternative<FogExp2>(*sc->fog)) {
                const auto& f = std::get<FogExp2>(*sc->fog);
                sigma = f.density;
                tint  = f.color;
            } else if (std::holds_alternative<Fog>(*sc->fog)) {
                const auto& f = std::get<Fog>(*sc->fog);
                const float span = std::max(1e-4f, f.farPlane - f.nearPlane);
                sigma = 1.f / span;
                tint  = f.color;
            }
            if (sigma > 0.f) {
                fogPresent    = true;
                ubo.sigmaT[0] = sigma;
                ubo.sigmaT[1] = sigma;
                ubo.sigmaT[2] = sigma;
                ubo.enabled   = 1.f;
                ubo.color[0]  = tint.r;
                ubo.color[1]  = tint.g;
                ubo.color[2]  = tint.b;
            }
        }
        // HG anisotropy is a property of the MEDIUM, not of scene.fog — set it
        // unconditionally so a heightFog-only medium (panel Fog density, no
        // scene.fog) honours setFogAnisotropy too. Consumers gate on their own
        // medium-active conditions, so this is inert when no medium exists.
        ubo.anisotropy = fogAnisotropy_;

        // ── Resolve the ONE air medium (Phase 2 unification) ─────────────────
        // scene.fog is the primary one-knob density. setHeightFog is the ADVANCED
        // control, layered so an app that drives scene.fog per frame keeps the
        // panel's Fog-density slider live:
        //   • PROFILE (baseY/falloff/noise) — always from setHeightFog when set,
        //     so scene.fog's density can be shaped into a ground layer.
        //   • DENSITY precedence — an EXPLICIT setHeightFog density > 0 is the
        //     deliberate advanced OVERRIDE and WINS over scene.fog. A heightFog
        //     with density <= 0 is "profile-only": scene.fog supplies the density
        //     (the panel slider / FogExp2), heightFog only shapes the profile.
        //   • scene.fog otherwise drives it; neither present → no medium.
        const bool hfExplicitDensity = heightFogEnabled_ && heightFogDensity_ > 0.0f;
        mediumDensityThisFrame_ = hfExplicitDensity ? heightFogDensity_
                                : (fogPresent ? sigma : 0.0f);
        mediumActiveThisFrame_  = mediumDensityThisFrame_ > 0.0f;
        mediumBaseYThisFrame_   = heightFogEnabled_ ? heightFogBaseY_   : 0.0f;
        mediumFalloffThisFrame_ = heightFogEnabled_ ? heightFogFalloff_ : kUniformFogFalloff;
        mediumNoiseThisFrame_   = heightFogEnabled_ ? heightFogNoiseAmount_ : 0.0f;
        // Air-fog tint (= scene.fog colour when present, else white — mirrors the
        // shade pass's medAlbedo). The overlay particle draw fades smoke toward
        // this; ubo.color already holds it for the fogPresent case.
        mediumTintThisFrame_[0] = fogPresent ? ubo.color[0] : 1.0f;
        mediumTintThisFrame_[1] = fogPresent ? ubo.color[1] : 1.0f;
        mediumTintThisFrame_[2] = fogPresent ? ubo.color[2] : 1.0f;

        // Mirror the resolved medium into the fog UBO's hf* (the FILTER recombines
        // bind only this UBO; the shade/froxel read the CloudUbo copy). 0 = off →
        // the filter's fogTransmittance short-circuits the hetero branch exactly.
        ubo.hfDensity = mediumActiveThisFrame_ ? mediumDensityThisFrame_ : 0.0f;
        ubo.hfBaseY   = mediumBaseYThisFrame_;
        ubo.hfFalloff = mediumFalloffThisFrame_;

        // Underwater murk (setUnderwaterMurk) — a SEPARATE homogeneous medium
        // clipped to below waterSurfaceY, decoupled from the air fog above.
        ubo.waterSurfaceY = fogWaterSurfaceY_;
        ubo.murkDensity   = murkDensity_;
        ubo.murkColor[0]  = murkColor_[0];
        ubo.murkColor[1]  = murkColor_[1];
        ubo.murkColor[2]  = murkColor_[2];

        // Froxel-volumetrics gate: the deferred leaf records the froxel
        // passes only when a medium exists this frame (fog, or the
        // explicit clear-air beam density).
        fogEnabledThisFrame_ = ubo.enabled > 0.5f;

        uploadHostVisible(ctx->allocator(), fogUbos[frame], &ubo, sizeof(ubo));
    }

    void VulkanRenderer::Impl::createCloudUbos() {
        for (auto& b : cloudUbos) {
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(GpuCloudUbo),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    // Pack the setClouds state into the per-frame cloud UBO. timeSec is
    // wall-clock (frame-rate-independent) so wind drift + shape evolution run
    // at a fixed real-world speed. Disabled → enabled=0 → cloudMarch() no-ops.
    void VulkanRenderer::Impl::updateCloudUbo(uint32_t frame) {
        GpuCloudUbo ubo{};
        ubo.enabled     = cloudsEnabled_ ? 1.0f : 0.0f;
        ubo.coverage    = cloudCoverage_;
        ubo.density     = cloudDensity_;
        ubo.bottomY     = cloudBottomY_;
        ubo.topY        = cloudTopY_;
        ubo.evolveSpeed = cloudEvolveSpeed_;
        // Sim-time override first (setSimTime starts at whatever the app's
        // clock says, typically 0 — small floats, no epoch needed). The wall
        // fallback keeps the process-start epoch so the float stays small.
        if (simTimeSec_ >= 0.0) {
            ubo.timeSec = static_cast<float>(simTimeSec_);
        } else {
            static const auto cloudEpoch = std::chrono::steady_clock::now();
            ubo.timeSec     = std::chrono::duration<float>(
                                      std::chrono::steady_clock::now() - cloudEpoch).count();
        }
        // Heterogeneous near-field froxels run whenever an AIR medium exists this
        // frame — Phase 2: scene.fog alone is enough (the resolved medium below),
        // not only the explicit setHeightFog. The froxel medium is the height-fog
        // profile ONLY — the far cloud march already integrates the cloud over the
        // whole 0→far ray (including below 512 m), so folding cloudDensity into the
        // froxels too would double-count it (see mediumExtinction in cloud_density.glsl).
        // updateFogUbo ran first (VulkanCoreFrame.cpp) and resolved the medium.
        //
        // PLUS: a live ParticleField density volume (plan §3.3). Dust IS a
        // heterogeneous near-field medium, and froxel_inject/froxel_integrate
        // only call mediumExtinction inside `heteroActive > 0.5` — so a dust
        // cloud in a scene with no fog at all would be evaluated by nobody.
        // The predicate is the CPU half of ParticleFieldPass's: DensityRepr on
        // and at least one live particle. It is computed HERE, not read back
        // off the pass, because this UBO is written before prepareParticleFields
        // rebuilds the pass's volume list — and the two agree by construction,
        // since the pass derives its list from exactly these two facts.
        particleDensityActiveThisFrame_ = false;
        for (const auto& [field, entryIndex] : particleFields_) {
            if (field && field->densityRepr().enabled && field->liveCount() > 0) {
                particleDensityActiveThisFrame_ = true;
                break;
            }
        }
        ubo.heteroActive  = (mediumActiveThisFrame_ || particleDensityActiveThisFrame_) ? 1.0f : 0.0f;
        ubo.wind[0]       = cloudWind_[0];
        ubo.wind[1]       = cloudWind_[1];
        ubo.wind[2]       = cloudWind_[2];
        ubo.hfDensity     = mediumActiveThisFrame_ ? mediumDensityThisFrame_ : 0.0f;
        ubo.hfBaseY       = mediumBaseYThisFrame_;
        ubo.hfFalloff     = mediumFalloffThisFrame_;
        ubo.hfNoiseAmount = mediumNoiseThisFrame_;
        // Cloud shadow map is generated + sampled only when clouds are on (it's
        // the cloud's own transmittance projected to the ground).
        ubo.shadowActive  = cloudsEnabled_ ? 1.0f : 0.0f;
        ubo.epoch         = static_cast<float>(cloudEpoch_);
        uploadHostVisible(ctx->allocator(), cloudUbos[frame], &ubo, sizeof(ubo));
    }

    void VulkanRenderer::Impl::createRasterCameraUbos() {
        for (auto& b : view().rasterCameraUbos) {
            if (b.handle != VK_NULL_HANDLE) continue;
            b = createBuffer(
                    ctx->allocator(), ctx->device(),
                    sizeof(RasterCameraData),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        }
    }


VulkanRenderer::Impl::ParticleGeomRec* VulkanRenderer::Impl::ensureParticleGeom(const std::shared_ptr<BufferGeometry>& geomSp) {
            BufferGeometry* geom = geomSp.get();
            if (!geom) return nullptr;
            auto* posAttr  = geom->getAttribute<float>("position");
            auto* normAttr = geom->getAttribute<float>("normal");
            auto* uvAttr   = geom->getAttribute<float>("uv");
            auto* colAttr  = geom->getAttribute<float>("color");
            auto* idxAttr  = geom->getIndex();
            if (!posAttr || !normAttr || !uvAttr || !colAttr || !idxAttr) return nullptr;

            const uint32_t vtx = static_cast<uint32_t>(posAttr->count());
            const uint32_t idxCount = static_cast<uint32_t>(idxAttr->count());
            if (vtx == 0 || idxCount == 0) return nullptr;
            // Particle attributes are vec3/vec3/vec2/vec3 — bail if a custom
            // geometry doesn't match (the billboard pipeline assumes this layout).
            if (normAttr->count() != static_cast<int>(vtx) ||
                uvAttr->count()   != static_cast<int>(vtx) ||
                colAttr->count()  != static_cast<int>(vtx)) return nullptr;

            const unsigned int ver = geomVersionOf(*geom);

            auto uploadAnim = [&](ParticleGeomRec& rec) {
                uploadHostVisible(ctx->allocator(), rec.position, posAttr->array().data(), vtx * 3 * sizeof(float));
                uploadHostVisible(ctx->allocator(), rec.normal, normAttr->array().data(), vtx * 3 * sizeof(float));
                uploadHostVisible(ctx->allocator(), rec.color, colAttr->array().data(), vtx * 3 * sizeof(float));
            };

            auto it = particleGeomCache_.find(geom);
            if (it != particleGeomCache_.end()) {
                ParticleGeomRec& rec = it->second;
                const bool stale = rec.vertexCount != vtx || rec.indexCount != idxCount ||
                                   rec.liveCheck.expired() || rec.liveCheck.lock().get() != geom;
                if (!stale) {
                    if (rec.animVersion != ver) {
                        uploadAnim(rec);
                        rec.animVersion = ver;
                    }
                    return &rec;
                }
                // Topology change / recycled address — rebuild from scratch.
                // Retire the old buffers (in-flight frames may still draw from
                // them) instead of a full device drain. VulkanRetireQueue.hpp.
                retireParticleGeomRec(rec);
                particleGeomCache_.erase(it);
            }

            // Fresh build. All buffers host-visible; pos/normal/color re-uploaded
            // each frame, uv + index written once here.
            const VkBufferUsageFlags vbUsage =
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ParticleGeomRec rec{};
            rec.vertexCount = vtx;
            rec.indexCount  = idxCount;
            rec.liveCheck   = geomSp;
            auto mkBuf = [&](VkDeviceSize bytes) {
                return createBuffer(ctx->allocator(), ctx->device(), bytes, vbUsage,
                                    VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            };
            rec.position = mkBuf(vtx * 3 * sizeof(float));
            rec.normal   = mkBuf(vtx * 3 * sizeof(float));
            rec.uv       = mkBuf(vtx * 2 * sizeof(float));
            rec.color    = mkBuf(vtx * 3 * sizeof(float));
            rec.index    = createBuffer(ctx->allocator(), ctx->device(),
                                        idxCount * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec.uv, uvAttr->array().data(), vtx * 2 * sizeof(float));
            uploadHostVisible(ctx->allocator(), rec.index, idxAttr->array().data(), idxCount * sizeof(uint32_t));
            auto [ins, _] = particleGeomCache_.emplace(geom, std::move(rec));
            uploadAnim(ins->second);
            ins->second.animVersion = ver;
            return &ins->second;
        }

const vulkan::LineRec* VulkanRenderer::Impl::ensureLineGeometryUploaded(const BufferGeometry* geom) {
            if (!geom) return nullptr;
            auto posAttr = geom->getAttribute<float>("position");
            if (!posAttr || posAttr->count() == 0) return nullptr;
            auto* idxAttr = geom->getIndex();
            // Optional per-vertex color, used by AxesHelper-style overlays.
            // Untyped so a narrowed (compressAttributes) color still counts;
            // the upload below widens it through a FloatAttributeView.
            const auto* colAttr = geom->getAttribute("color");

            const uint32_t posVer = posAttr->version;
            const uint32_t idxVer = (idxAttr && idxAttr->count() > 0) ? idxAttr->version : 0u;
            const uint32_t colVer = (colAttr && colAttr->count() > 0) ? colAttr->version : 0u;

            auto it = lineGeomCache_.find(geom);
            if (it != lineGeomCache_.end() && it->second.geomId != geom->id) {
                // Recycled pointer: this address was a DIFFERENT geometry whose
                // buffers we still hold. Retire them and re-upload from scratch
                // (the version fields would otherwise alias — both at 0).
                // Through the retire queue, not destroyBuffer: this runs during
                // record, so the previous frame's line draw may still be reading
                // them. retire() no-ops on null handles.
                retire(std::move(it->second.vertex));
                retire(std::move(it->second.index));
                retire(std::move(it->second.color));
                lineGeomCache_.erase(it);
                it = lineGeomCache_.end();
            }
            if (it != lineGeomCache_.end()) {
                auto& rec = it->second;
                rec.lastTouch = overlayFrameCounter_;
                if (rec.positionVersion == posVer &&
                    rec.indexVersion    == idxVer &&
                    rec.colorVersion    == colVer) {
                    return &rec;
                }
                // Re-upload paths.
                const auto& posArr = posAttr->array();
                const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
                if (vbBytes > rec.vertex.size) {
                    destroyBuffer(ctx->allocator(), rec.vertex);
                    rec.vertex = createBuffer(
                            ctx->allocator(), ctx->device(), vbBytes,
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                }
                uploadHostVisible(ctx->allocator(), rec.vertex, posArr.data(), vbBytes);
                rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
                rec.positionVersion = posVer;

                if (idxAttr && idxAttr->count() > 0) {
                    const auto& indices = idxAttr->array();
                    const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                    if (rec.index.handle == VK_NULL_HANDLE || ibBytes > rec.index.size) {
                        if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.index);
                        rec.index = createBuffer(
                                ctx->allocator(), ctx->device(), ibBytes,
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    }
                    uploadHostVisible(ctx->allocator(), rec.index, indices.data(), ibBytes);
                    rec.indexCount   = static_cast<uint32_t>(indices.size());
                    rec.indexVersion = idxVer;
                } else if (rec.index.handle != VK_NULL_HANDLE) {
                    destroyBuffer(ctx->allocator(), rec.index);
                    rec.index        = {};
                    rec.indexCount   = 0;
                    rec.indexVersion = 0;
                }

                // Color buffer follows the same in-place / recreate logic.
                if (colAttr && colAttr->count() > 0) {
                    FloatAttributeView colView(colAttr);
                    const VkDeviceSize cbBytes = colView.size() * sizeof(float);
                    if (rec.color.handle == VK_NULL_HANDLE || cbBytes > rec.color.size) {
                        if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.color);
                        rec.color = createBuffer(
                                ctx->allocator(), ctx->device(), cbBytes,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    }
                    uploadHostVisible(ctx->allocator(), rec.color, colView.data(), cbBytes);
                    rec.colorVersion = colVer;
                } else if (rec.color.handle != VK_NULL_HANDLE) {
                    destroyBuffer(ctx->allocator(), rec.color);
                    rec.color        = {};
                    rec.colorVersion = 0;
                }
                return &rec;
            }

            // First-time upload.
            const auto& posArr = posAttr->array();
            vulkan::LineRec rec{};
            rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
            rec.positionVersion = posVer;
            rec.geomId          = geom->id;
            rec.lastTouch       = overlayFrameCounter_;

            const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
            rec.vertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec.vertex, posArr.data(), vbBytes);

            if (idxAttr && idxAttr->count() > 0) {
                const auto& indices = idxAttr->array();
                rec.indexCount   = static_cast<uint32_t>(indices.size());
                rec.indexVersion = idxVer;
                const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                rec.index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), rec.index, indices.data(), ibBytes);
            }

            if (colAttr && colAttr->count() > 0) {
                FloatAttributeView colView(colAttr);
                rec.colorVersion = colVer;
                const VkDeviceSize cbBytes = colView.size() * sizeof(float);
                rec.color = createBuffer(
                        ctx->allocator(), ctx->device(), cbBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), rec.color, colView.data(), cbBytes);
            }

            return &lineGeomCache_.emplace(geom, std::move(rec)).first->second;
        }

// ── GPU instance expansion (stage 1 of plans/gpu-driven-instances.md) ────────
//
// The producer half of the GPU-driven instance work, with no consumer yet. It
// mirrors, on the GPU, the loop at VulkanCoreScene.cpp's lean matrix refresh:
// for every InstancedMesh span, world = mesh->matrixWorld * instanceMatrix[i].
//
// Two contracts hold this together, and both are the ones EntrySpan already
// documents:
//   * The matrix the entries were baked from is `EntrySpan::meshWorld`, NOT a
//     live read of mesh->matrixWorld. On a lean frame where neither the mesh
//     transform nor the instance-matrix version moved, the CPU skips the span
//     and its entries keep last frame's matrices — which came from meshWorld.
//     Uploading the live matrix instead would make the GPU "more current" than
//     the CPU and the equality check would fail on exactly the frames where
//     nothing is wrong.
//   * The instance matrices are re-sent only when the CPU re-read them, i.e.
//     when EntrySpan::movedThisFrame said so. That keeps
//     instanceMatrix()->needsUpdate() load-bearing (invariant 6) instead of
//     quietly uploading 5 MB a frame for a static field, and it keeps the two
//     sides reading the attribute array at the SAME instants.

void VulkanRenderer::Impl::prepareInstanceExpansion(uint32_t frame) {
            if (!gpuInstanceExpand_ || !instExpand_) return;
            THREEPP_CPUPROF_NAMED(phSpans, "frame.M_instSpanDesc");

            // 1. The GPU-path span list + its two prefix sums. Instanced spans
            //    only; a plain mesh's single entry is already its mesh matrix
            //    and there is nothing to expand.
            instExpandScratch_.clear();
            uint32_t matBase = 0, workBase = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(entrySpans_.size()); ++i) {
                const auto& sp = entrySpans_[i];
                if (!sp.inst || sp.count == 0u) continue;
                // Clamp to what the attribute actually holds. InstancedMesh
                // allocates instanceMatrix for maxCount and setCount() can move
                // count below it, so count <= capacity always — but reading past
                // the array on a future API change would be a silent OOB, and the
                // clamped span is still a valid (smaller) comparison.
                const size_t avail = sp.inst->instanceMatrix()->array().size() / 16u;
                const uint32_t n = static_cast<uint32_t>(std::min<size_t>(sp.count, avail));
                if (n == 0u) continue;
                instExpandScratch_.push_back({i, n, matBase, workBase});
                matBase  += n;
                workBase += n;
            }
            instExpandMatrixTotal_ = matBase;
            instExpandWorkTotal_   = workBase;

            // 2. A layout change invalidates every per-slot "already uploaded"
            //    stamp: slot k of the list no longer describes the same span.
            if (instExpandScratch_ != instExpandSpans_) {
                instExpandSpans_ = instExpandScratch_;
                instExpandSerial_.assign(instExpandSpans_.size(), 1ull);
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    instExpandFifSerial_[f].assign(instExpandSpans_.size(), 0ull);
            }
            if (instExpandSpans_.empty()) return;

            instExpand_->prepareFrame(frame,
                                      static_cast<uint32_t>(instExpandSpans_.size()),
                                      instExpandMatrixTotal_,
                                      static_cast<uint32_t>(lastVisibleEntries_.size()));
            // A reallocated pool holds garbage — every span is stale in it.
            if (instExpand_->takeMatrixPoolFresh(frame))
                instExpandFifSerial_[frame].assign(instExpandSpans_.size(), 0ull);

            // 3. SpanDescs: rewritten whole, every frame. O(spans).
            auto* dst = instExpand_->spanPtr(frame);
            for (size_t k = 0; k < instExpandSpans_.size(); ++k) {
                const auto& gs = instExpandSpans_[k];
                const auto& sp = entrySpans_[gs.spanIdx];
                auto& d = dst[k];
                std::memcpy(d.world, sp.meshWorld.data(), 64);
                d.firstEntry = sp.first;
                d.count      = gs.count;
                d.matBase    = gs.matBase;
                d.workBase   = gs.workBase;
                // Content serial: bumped on the frames the CPU re-baked this
                // span, so both frames-in-flight eventually catch up.
                if (sp.movedThisFrame) ++instExpandSerial_[k];
            }
            instExpand_->flushSpans(frame, static_cast<uint32_t>(instExpandSpans_.size()));
            phSpans.stop();

            // 4. The instance matrices themselves — the new per-frame upload
            //    this stage pays for. Version-gated per span per slot; the
            //    dirty range is flushed once rather than per span.
            {
                THREEPP_CPUPROF("frame.M2_instMatUpload");
                float* pool = instExpand_->matrixPtr(frame);
                uint32_t dirtyLo = ~0u, dirtyHi = 0u;
                for (size_t k = 0; k < instExpandSpans_.size(); ++k) {
                    if (instExpandFifSerial_[frame][k] == instExpandSerial_[k]) continue;
                    instExpandFifSerial_[frame][k] = instExpandSerial_[k];
                    const auto& gs = instExpandSpans_[k];
                    const auto& src = entrySpans_[gs.spanIdx].inst->instanceMatrix()->array();
                    const size_t n = size_t(gs.count) * 16u;
                    std::memcpy(pool + size_t(gs.matBase) * 16u, src.data(), n * sizeof(float));
                    dirtyLo = std::min(dirtyLo, gs.matBase);
                    dirtyHi = std::max(dirtyHi, gs.matBase + gs.count);
                }
                if (dirtyHi > dirtyLo) instExpand_->flushMatrices(frame, dirtyLo, dirtyHi - dirtyLo);
            }
        }

// ParticleField, phase 0. The cost claim the whole entity rests on is visible
// right here: this function is O(fields) plus ONE memcpy per field whose sim
// advanced. There is no per-particle loop anywhere in it, and nothing it writes
// grows the entry list, the draw list or the TLAS.
void VulkanRenderer::Impl::prepareParticleFields(uint32_t frame) {
            if (!particleFieldPass_) return;
            // Nothing to do and nothing to sweep: the common scene. Skip
            // before the profiler scope so it only counts frames that
            // actually paid.
            if (particleFields_.empty() &&
                particleFieldPass_->liveFieldCount() == 0) return;
            THREEPP_CPUPROF("frame.P_particleFields");
            // F5: the acceleration structure the surface bake traces. Handed
            // over every frame and acted on only when the handle moved — which
            // is a structural rebuild, and therefore vkDeviceWaitIdle-guarded,
            // which is what makes writing that descriptor here safe. Given here
            // rather than in rewriteDeferredDescriptors because the pass is
            // created lazily and may not exist when that last ran.
            particleFieldPass_->setTlas(tlas);
            // Beside it, and for the sun-occlusion query that shares its set:
            // this frame's slot of the GeometryDesc ring, whose foamAddress is
            // how that query tells a water hit (walk past it) from a hull hit
            // (that is the shadow). Per frame, because the ring rotates.
            particleFieldPass_->setSceneGeomAddress(
                    geometryDescsBuffers[frame].address);
            particleFieldRecs_.clear();
            particleFieldRecs_.reserve(particleFields_.size());
            for (const auto& [field, entryIndex] : particleFields_) {
                if (!field) continue;
                // Vertices per proxy instance, from the SAME record and the
                // SAME LOD level buildIndirectDrawData puts in the DrawInfo —
                // a mismatch would draw the proxy's index buffer against the
                // wrong vertex count. 0 when MeshRepr is off or the proxy has
                // not uploaded, which parks the field's draw for the frame.
                uint32_t vcount = 0u;
                if (entryIndex < lastVisibleEntries_.size()) {
                    const auto& en = lastVisibleEntries_[entryIndex];
                    if (const BlasRecord* rec = resolveBlasForEntry(en)) {
                        if (rec->vertex.handle != VK_NULL_HANDLE) {
                            const auto sel = selectLodGeom(*rec, 0);
                            vcount = sel.indexed ? sel.indexCount : rec->vertexCount;
                        }
                    }
                }
                // classId read live, not cached at expansion: setObjectClassId
                // is a per-frame-editable label and a field is one lookup.
                particleFieldRecs_.push_back({field, entryIndex,
                                              classIdForObject(*field), vcount});
            }
            particleFieldPass_->prepareFrame(frameSerial_, frame, particleFieldRecs_);
            updateParticleDensityUbo(frame);
        }

// ── ParticleField density volumes (phase 2, plan §3.3) ───────────────────────
// The two handles bindings 67/68 always name. Created once, never resized, and
// deliberately owned HERE rather than by ParticleFieldPass: the deferred
// descriptor sets are written before the pass is lazily constructed, and go on
// being written on every scene that never gets a field.
void VulkanRenderer::Impl::ensureParticleDensityResources() {
            if (particleDensityUbos_[0].handle == VK_NULL_HANDLE) {
                vulkan::ParticleDensityUboGpu zero{};
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    particleDensityUbos_[f] = createBuffer(
                            ctx->allocator(), ctx->device(), sizeof(zero),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    // counts.x == 0 ⇒ particleDensity() returns 0.0 before a
                    // single field exists, so the very first frame is already
                    // the dust-free answer rather than whatever VMA handed us.
                    uploadHostVisible(ctx->allocator(), particleDensityUbos_[f], &zero, sizeof(zero));
                }
            }
            if (particleDensityDummy_.image == VK_NULL_HANDLE) {
                particleDensityDummy_ = createImage3D(
                        1u, 1u, 1u, VK_FORMAT_R32_UINT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "particleDensityDummy (1x1x1, binding 67 placeholder)");
                // GENERAL, matching what the real volumes declare, and cleared
                // to 0 so a slot that is bound but never written still reads as
                // "no dust" rather than as uninitialised device memory. The
                // shader never samples an unused slot (counts.x gates it), but
                // a descriptor whose image has never been transitioned is a
                // validation error the moment the set is bound.
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = 0;
                imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imb.image = particleDensityDummy_.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &imb);
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, particleDensityDummy_.image,
                                     VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
                endAndSubmitOneShot(cb);
            }
            // The r16f twin, for binding 69 (the shade's linear-sampled
            // mirrors). Same GENERAL + cleared-to-zero contract, same reason.
            if (particleDensityLinDummy_.image == VK_NULL_HANDLE) {
                particleDensityLinDummy_ = createImage3D(
                        1u, 1u, 1u, VK_FORMAT_R16_SFLOAT,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "particleDensityLinDummy (1x1x1, binding 69 placeholder)");
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = 0;
                imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imb.image = particleDensityLinDummy_.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &imb);
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, particleDensityLinDummy_.image,
                                     VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
                endAndSubmitOneShot(cb);
            }
        }

// Publish this frame's volume boxes. O(volumes), post-fence / pre-record — the
// same window every other ParticleField write lands in (R6).
void VulkanRenderer::Impl::updateParticleDensityUbo(uint32_t frame) {
            if (!particleFieldPass_) return;
            ensureParticleDensityResources();

            vulkan::ParticleDensityUboGpu ubo{};
            const auto& vols = particleFieldPass_->densityVolumes();
            const uint32_t n = std::min<uint32_t>(
                    static_cast<uint32_t>(vols.size()), vulkan::kMaxDensityFields);
            // Unbound slots keep the zero-initialised albedo/emission of the
            // struct: the shader never indexes past counts.x, but a NaN or a
            // stale temperature in a slot a future frame promotes to live is
            // exactly the class of bug that only shows up under churn.
            uint32_t emissive = 0;
            for (uint32_t i = 0; i < n; ++i) {
                ubo.boxMin[i][0] = vols[i].boxMin[0];
                ubo.boxMin[i][1] = vols[i].boxMin[1];
                ubo.boxMin[i][2] = vols[i].boxMin[2];
                ubo.boxMin[i][3] = vols[i].resolution;
                ubo.boxInvSize[i][0] = vols[i].boxInvSize[0];
                ubo.boxInvSize[i][1] = vols[i].boxInvSize[1];
                ubo.boxInvSize[i][2] = vols[i].boxInvSize[2];
                // PER FIELD since plans/particle-atmosphere.md F-A — this used
                // to be one shared vec4 filled from the first enabled field.
                ubo.albedoAniso[i][0] = vols[i].albedo[0];
                ubo.albedoAniso[i][1] = vols[i].albedo[1];
                ubo.albedoAniso[i][2] = vols[i].albedo[2];
                ubo.albedoAniso[i][3] = vols[i].anisotropy;
                for (uint32_t c = 0; c < 4; ++c) ubo.emission[i][c] = vols[i].emission[c];
                if (vols[i].emission[0] > 0.f) emissive = 1;
            }
            ubo.counts[0] = n;
            // ONE uniform branch for the whole march: the emissive path (the
            // blackbody term and the 32-step base count) is skipped wholesale
            // when this is 0, so a dust-only scene never pays for fire. (The
            // hash dither and the world-step raise apply to every march now —
            // see the march-grid rules in deferred_shade_60_fog_volumetrics.)
            ubo.counts[1] = emissive;
            uploadHostVisible(ctx->allocator(), particleDensityUbos_[frame], &ubo, sizeof(ubo));

            // The bound volume LIST changed (a field gained or lost its volume):
            // every view's set names a stale view. Refresh THIS slot now — its
            // fence has signaled, which is what makes the write legal — and mark
            // the others so they refresh at the top of their own frames.
            const uint64_t gen = particleFieldPass_->densityGeneration();
            if (particleDensityDescGen_[frame] != gen) {
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    if (f != frame) deferredDescDirty_[f] = true;
                forEachLiveView([&] { rewriteDeferredDescriptors(static_cast<int>(frame)); });
            }
        }

// ── Splat reflection volumes (plans/splat-volume-reflections.md, Part 2) ─────
// The two handles bindings 70/71 always name. The particle-density pair above,
// copied — created once, never resized, and owned HERE rather than by SplatPass
// for the same reason: the deferred descriptor sets are written before any
// cloud exists and go on being written on every scene that never gets one.
void VulkanRenderer::Impl::ensureSplatVolumeResources() {
            if (splatVolumeUbos_[0].handle == VK_NULL_HANDLE) {
                vulkan::SplatVolumeUboGpu zero{};
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    splatVolumeUbos_[f] = createBuffer(
                            ctx->allocator(), ctx->device(), sizeof(zero),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    // counts.x == 0 ⇒ svLeg returns EXACTLY 1.0 / vec3(0) before
                    // a single cloud exists, so the very first frame is already
                    // the splat-free answer rather than whatever VMA handed us.
                    uploadHostVisible(ctx->allocator(), splatVolumeUbos_[f], &zero, sizeof(zero));
                }
            }
            if (splatVolumeDummy_.image == VK_NULL_HANDLE) {
                // rgba16f, matching the baked volumes — the array is one
                // descriptor type and one format for live and unused slots
                // alike. SAMPLED only: nothing ever writes this one.
                splatVolumeDummy_ = createImage3D(
                        1u, 1u, 1u, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "splatVolumeDummy (1x1x1, binding 70 placeholder)");
                // GENERAL, matching what SplatPass::Cloud::volume declares for
                // its whole life, and cleared to 0 so a slot that is bound but
                // never written reads as "no cloud" rather than as uninitialised
                // device memory. The shader never samples an unused slot
                // (counts.x gates it), but a descriptor whose image has never
                // been transitioned is a validation error the moment the set is
                // bound — the particleDensityDummy_ contract, copied.
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = 0;
                imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imb.image = splatVolumeDummy_.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &imb);
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, splatVolumeDummy_.image,
                                     VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
                endAndSubmitOneShot(cb);
            }
        }

// The eight views binding 70 names right now. ONE producer for both consumers
// (the descriptor write and the staleness key), so they cannot disagree.
void VulkanRenderer::Impl::splatVolumeBindViews(
        std::array<VkImageView, vulkan::kMaxSplatVolumes>& out) {
            ensureSplatVolumeResources();
            uint32_t n = 0;
            if (splat_) {
                for (const auto& v : splat_->volumeEntries()) {
                    if (n >= vulkan::kMaxSplatVolumes) break;
                    out[n++] = v.view;
                }
            }
            for (uint32_t i = n; i < vulkan::kMaxSplatVolumes; ++i)
                out[i] = splatVolumeDummy_.view;
        }

// FNV-1a over (generation, the eight bound handles). See the declaration for
// why the handles and not the generation alone.
uint64_t VulkanRenderer::Impl::splatVolumeBindKey() {
            std::array<VkImageView, vulkan::kMaxSplatVolumes> views{};
            splatVolumeBindViews(views);
            uint64_t h = 0xcbf29ce484222325ull;
            auto mix = [&h](uint64_t v) { h ^= v; h *= 0x100000001b3ull; };
            mix(splat_ ? splat_->volumeGeneration() : 0ull);
            for (VkImageView v : views) {
                // memcpy, not a cast: a non-dispatchable handle is a pointer on
                // 64-bit and a uint64_t on 32-bit, and only one of those casts
                // compiles on each.
                std::uint64_t raw = 0;
                static_assert(sizeof(VkImageView) <= sizeof(raw));
                std::memcpy(&raw, &v, sizeof(v));
                mix(raw);
            }
            return h;
        }

// Publish this frame's splat volumes. O(volumes), post-fence / pre-record, and
// after collectSplatClouds' syncClouds — which is what guarantees a cloud
// uploaded this frame is already BAKED before the UBO names it.
//
// Everything the shader needs is composed HERE rather than there: worldToUvw is
// an inverse, the world AABB is eight transformed corners, and both belong on
// the host where they are paid once per volume per frame instead of once per
// step per pixel.
void VulkanRenderer::Impl::updateSplatVolumeUbo(uint32_t frame) {
            splatVolumeActiveThisFrame_ = false;
            if (!splat_) return;
            ensureSplatVolumeResources();

            // THREEPP_VK_SPLATVOL_SIGMA — the ONE calibration knob (an A/B
            // lever, not authored data). Read once: it multiplies sigma at
            // SAMPLE time, so a run must not be able to change it mid-flight and
            // leave two frames disagreeing about the same cloud.
            static const float sigmaScale = [] {
                const char* e = std::getenv("THREEPP_VK_SPLATVOL_SIGMA");
                if (!e || !*e) return 1.f;
                const float v = std::strtof(e, nullptr);
                return (std::isfinite(v) && v > 0.f) ? v : 1.f;
            }();

            vulkan::SplatVolumeUboGpu ubo{};
            const auto entries = splat_->volumeEntries();
            const uint32_t n = std::min<uint32_t>(
                    static_cast<uint32_t>(entries.size()), vulkan::kMaxSplatVolumes);
            // Unbound slots keep the zero-initialised matrices/boxes of the
            // struct: the shader never indexes past counts.x, but a stale matrix
            // in a slot a future frame promotes to live is exactly the class of
            // bug that only shows up under churn (the density table's reason,
            // verbatim).
            for (uint32_t i = 0; i < n; ++i) {
                const auto& e = entries[i];
                Matrix4 model;
                std::memcpy(model.elements.data(), e.model, 64);

                // world -> UVW = boxNormalise(localBoxMin, localBoxSize) *
                // inverse(model). The normalisation is folded in on the host so
                // the per-step transform is one mat4 multiply.
                Matrix4 inv;
                inv.copy(model).invert();
                Matrix4 norm;// identity by construction
                for (int a = 0; a < 3; ++a) {
                    const float s = (e.localBoxSize[a] > 0.f) ? e.localBoxSize[a] : 1.f;
                    norm.elements[static_cast<size_t>(a * 5)]   = 1.f / s;// diag(a,a)
                    norm.elements[static_cast<size_t>(12 + a)]  = -e.localBoxMin[a] / s;
                }
                Matrix4 worldToUvw;
                worldToUvw.multiplyMatrices(norm, inv);
                std::memcpy(ubo.worldToUvw[i], worldToUvw.elements.data(), 64);

                // Conservative world AABB of the OBB: the 8 transformed corners.
                // Only the interval clip uses it; the exact membership test is
                // the in-UVW one the shader runs after the matrix, so "too big"
                // costs steps outside the cloud and never a wrong pixel.
                float mn[3] = {1e30f, 1e30f, 1e30f};
                float mx[3] = {-1e30f, -1e30f, -1e30f};
                const auto& m = model.elements;
                for (int c = 0; c < 8; ++c) {
                    const float lx = e.localBoxMin[0] + ((c & 1) ? e.localBoxSize[0] : 0.f);
                    const float ly = e.localBoxMin[1] + ((c & 2) ? e.localBoxSize[1] : 0.f);
                    const float lz = e.localBoxMin[2] + ((c & 4) ? e.localBoxSize[2] : 0.f);
                    const float wx = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
                    const float wy = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
                    const float wz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
                    mn[0] = std::min(mn[0], wx); mx[0] = std::max(mx[0], wx);
                    mn[1] = std::min(mn[1], wy); mx[1] = std::max(mx[1], wy);
                    mn[2] = std::min(mn[2], wz); mx[2] = std::max(mx[2], wz);
                }
                for (int a = 0; a < 3; ++a) {
                    ubo.worldBoxMin[i][a] = mn[a];
                    ubo.worldBoxMax[i][a] = mx[a];
                }

                // sigma is baked per LOCAL metre, so a scaled cloud needs it
                // re-expressed per WORLD metre: divide by the uniform-equivalent
                // scale s = cbrt(|det model3x3|). Exact for uniform scale, a
                // stated approximation otherwise (the plan says so out loud).
                const float det = m[0] * (m[5] * m[10] - m[9] * m[6]) -
                                  m[4] * (m[1] * m[10] - m[9] * m[2]) +
                                  m[8] * (m[1] * m[6] - m[5] * m[2]);
                const float s = std::cbrt(std::fabs(det));
                // A degenerate (zero-scale, mirrored-to-flat) matrix would make
                // this Inf; 1.0 keeps the volume readable instead of poisoning
                // every reflection that taps it.
                const float sSafe = (std::isfinite(s) && s > 1e-6f) ? s : 1.f;
                ubo.params[i][0] = sigmaScale / sSafe;
                // The voxel edge goes the OTHER way through the same scale:
                // sigma per local metre shrinks by s, a local length grows by
                // it. The march reads this as its resolution floor
                // (splat_volume.glsl, adaptive step count).
                ubo.params[i][1] = e.voxelLocal * sSafe;
            }
            ubo.counts[0] = n;
            uploadHostVisible(ctx->allocator(), splatVolumeUbos_[frame], &ubo, sizeof(ubo));
            splatVolumeActiveThisFrame_ = n > 0;

            // The bound volume LIST changed (a bake completed, a cloud stopped
            // being visible, a cloud was freed): every view's set names the
            // wrong views. Refresh this slot now — its fence has signaled,
            // which is what makes the write legal — and mark the others so they
            // refresh at the top of their own frames. The churn is the design:
            // a cloud that appears and disappears rewrites two sets rather
            // than stalling the device.
            //
            // Keyed on the HANDLES, not on volumeGeneration() alone — see
            // splatVolumeBindKey. Dropping a hidden cloud's view here is what
            // gives retireStale the framesInFlight+1 margin its own timing
            // argument assumes; on the generation alone the set still named the
            // view at the moment SplatPass destroyed it, which the validation
            // gate reports as VUID-vkDestroyImageView-imageView-01026.
            const uint64_t key = splatVolumeBindKey();
            if (splatVolumeDescKey_[frame] != key) {
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    if (f != frame) deferredDescDirty_[f] = true;
                forEachLiveView([&] { rewriteDeferredDescriptors(static_cast<int>(frame)); });
            }
        }

// The whole density representation's per-frame GPU cost: one clear + one
// dispatch per field, at the head of the frame command buffer and BEFORE any
// view's froxel pass. Once for all views — the volume is world-anchored (R9).
void VulkanRenderer::Impl::recordParticleDensityScatter(VkCommandBuffer cb, uint32_t frame) {
            if (!particleFieldPass_ || !particleFieldPass_->densityActive()) return;
            gpuTimings_->begin(cb, vulkan::TP_ParticleDensity, frame);
            particleFieldPass_->recordDensityScatter(cb);
            gpuTimings_->end(cb, vulkan::TP_ParticleDensity, frame);
        }

// The device-side liveCount -> instanceCount publish. Head of the frame command
// buffer, beside recordInstanceExpansion, and for the same structural reason:
// no dependants, its own phase, and it must precede every consumer.
void VulkanRenderer::Impl::recordParticleFieldCounts(VkCommandBuffer cb) {
            if (!particleFieldPass_) return;
            // F6: the interop snapshot shares this call site rather than
            // earning a third one. Both are head-of-frame device copies that
            // must precede every consumer of a field, and this site is already
            // the one that guarantees that — it runs before the density
            // scatter and before any view's raster pass. The snapshot goes
            // FIRST of the two because the counts copy publishes how much of
            // what it just wrote is live.
            particleFieldPass_->recordInteropSnapshot(cb);
            particleFieldPass_->recordCounts(cb);
        }

// The device emitter. FIRST of the ParticleField block, because everything else
// in it reads what this writes: the density scatter samples the positions, the
// G-buffer draw pulls them per vertex, and the counts copy — while independent
// — belongs after so the block reads in dependency order.
void VulkanRenderer::Impl::recordParticleFieldEmit(VkCommandBuffer cb, uint32_t frame) {
            if (!particleFieldPass_ || !particleFieldPass_->emitActive()) return;
            gpuTimings_->begin(cb, vulkan::TP_ParticleEmit, frame);
            // F5: the surface height bake, immediately before the emitter that
            // reads it — and INSIDE the same timestamp bracket, deliberately. A
            // bake is recorded on a handful of frames out of thousands (a
            // structural change, a follow-centre snap), so a timer of its own
            // would read zero almost always and would hide the spike among the
            // frames that did nothing; folded in, TP_ParticleEmit reports what
            // the emitter block actually cost on the frame it happened. A bake
            // is only ever queued alongside an emit dispatch, so the early-out
            // above cannot skip one that was scheduled.
            particleFieldPass_->recordSurfaceBake(cb);
            particleFieldPass_->recordEmit(cb);
            gpuTimings_->end(cb, vulkan::TP_ParticleEmit, frame);
        }

void VulkanRenderer::Impl::recordInstanceExpansion(VkCommandBuffer cb, uint32_t frame) {
            if (!gpuInstanceExpand_ || !instExpand_ || instExpandSpans_.empty()) return;
            gpuTimings_->begin(cb, vulkan::TP_InstanceExpand, frame);
            instExpand_->record(cb, frame,
                                static_cast<uint32_t>(instExpandSpans_.size()),
                                instExpandWorkTotal_);
            gpuTimings_->end(cb, vulkan::TP_InstanceExpand, frame);
        }

bool VulkanRenderer::Impl::verifyInstanceExpansion(VulkanRenderer::InstanceExpandCheck& out) {
            out = {};
            if (!gpuInstanceExpand_ || !instExpand_ || instExpandSpans_.empty()) return false;
            // The dispatch for the frame just rendered is RECORDED but not
            // submitted — render() leaves the command buffer open and the
            // Canvas frame-end callback (or the next render()) closes it. Read
            // it now and we would be comparing this frame's CPU matrices against
            // last frame's GPU output, which is only equal when nothing moved,
            // i.e. exactly when the check proves nothing. Close the frame first.
            if (frameState_ != FrameState::Idle) endFrame();
            std::vector<float> gpu;
            if (!instExpand_->readWorldMatrices(cmdPool, ctx->graphicsQueue(),
                                                static_cast<uint32_t>(lastVisibleEntries_.size()),
                                                gpu))
                return false;
            const size_t haveEntries = gpu.size() / 16u;
            out.spans = instExpandSpans_.size();
            for (const auto& gs : instExpandSpans_) {
                const auto& sp = entrySpans_[gs.spanIdx];
                for (uint32_t j = 0; j < gs.count; ++j) {
                    const size_t e = size_t(sp.first) + j;
                    if (e >= haveEntries || e >= lastVisibleEntries_.size()) continue;
                    const float* g = gpu.data() + e * 16u;
                    const float* c = lastVisibleEntries_[e].worldMatrix.data();
                    ++out.entriesCompared;
                    if (std::memcmp(g, c, 64) == 0) continue;
                    ++out.mismatches;
                    for (int k = 0; k < 16; ++k) {
                        out.maxAbsDiff = std::max(out.maxAbsDiff, std::abs(g[k] - c[k]));
                        // ULP distance of two same-sign finites is the distance
                        // between their bit patterns read as integers — the only
                        // scale-free way to say "one step of float apart".
                        int32_t gi, ci;
                        std::memcpy(&gi, g + k, 4);
                        std::memcpy(&ci, c + k, 4);
                        if ((gi < 0) != (ci < 0)) {
                            out.maxUlpDiff = 0xFFFFFFFFu;// sign straddle: not an ULP story
                        } else {
                            const uint32_t d = static_cast<uint32_t>(std::abs(gi - ci));
                            out.maxUlpDiff = std::max(out.maxUlpDiff, d);
                        }
                    }
                }
            }
            return true;
        }
}// namespace threepp
