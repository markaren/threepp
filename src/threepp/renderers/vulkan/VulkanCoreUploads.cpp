#include "VulkanCoreImpl.hpp"

#include <chrono>

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
    // and upload to motionMatBuffers[frame]. Identity for first-seen
    // entries (cold-start frame after a topology rebuild) so the reproject
    // is a no-op until prevWorldMats picks up real history. Keying by
    // (Mesh*, instanceIndex) so each InstancedMesh sub-instance carries
    // its own motion delta. Caller must have already waited the
    // inFlight[frame] fence — we write a buffer the GPU may have been
    // reading on the previous use of `frame`.
    void VulkanRenderer::Impl::computeAndUploadMotionMatrices(uint32_t frame,
                                        const std::vector<MeshEntry>& entries) {
        const uint32_t count = static_cast<uint32_t>(entries.size());
        if (count == 0) return;

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
        // don't update prevWorldMats. The body then locks to its prior
        // pose persistently — sub-eps accumulation can't drift past the
        // threshold because the reference doesn't update.
        //
        // 1e-4 covers ~0.1mm translation / ~0.0057° rotation. Tighter
        // than typical solver residual, looser than visible motion.
        // Tune up if your physics still wobbles; tune down if real slow
        // motion gets frozen.
        constexpr float kSettledEps = 1e-4f;

        std::vector<float> data(count * 16);
        std::vector<uint8_t> settled(count, 0);
        bool anyNonIdentity = false;
        for (uint32_t i = 0; i < count; ++i) {
            Matrix4 cur;
            std::memcpy(cur.elements.data(), entries[i].worldMatrix.data(), 64);

            Matrix4 motion;// identity by default
            EntryKey key{entries[i].mesh, entries[i].instanceIndex};
            auto it = prevWorldMats.find(key);
            if (it != prevWorldMats.end()) {
                Matrix4 prev;
                std::memcpy(prev.elements.data(), it->second.data(), 64);

                // Per-element max-abs delta. Cheap, catches both
                // translation (cols 12..14) and rotation/scale (rest).
                float maxDelta = 0.0f;
                for (int e = 0; e < 16; ++e) {
                    const float d = std::abs(cur.elements[e] - prev.elements[e]);
                    if (d > maxDelta) maxDelta = d;
                }
                if (maxDelta < kSettledEps) {
                    settled[i] = 1u;// keep motion as identity
                } else {
                    Matrix4 curInv;
                    curInv.copy(cur).invert();
                    motion.multiplyMatrices(prev, curInv);
                    anyNonIdentity = true;
                }
            }
            std::memcpy(&data[i * 16], motion.elements.data(), 64);
        }

        // Fast path: if every entry's motion is identity AND the buffer
        // slot was already all-identity from a previous frame, skip the
        // upload. mmap+memcpy of an all-zero scene's per-frame motionMat
        // is a real cost on heavy scenes (1500 entries × 64B = 96KB
        // mapped+written every frame for nothing). Sub-millisecond
        // individually, but it adds up on CPU-bound paths.
        if (!anyNonIdentity && motionMatBufferAllIdentity_[frame]) {
            // Buffer slot already holds identities; skip the upload.
        } else {
            uploadHostVisible(ctx->allocator(), motionMatBuffers[frame],
                              data.data(), data.size() * sizeof(float));
            motionMatBufferAllIdentity_[frame] = !anyNonIdentity;
        }

        // Record this frame's matrices for next frame's motion delta —
        // BUT skip settled entries so prev stays anchored to its
        // pre-jitter pose. Re-evaluating against the same frozen prev
        // each frame keeps the body locked in render space until real
        // motion actually crosses the eps threshold.
        for (uint32_t i = 0; i < count; ++i) {
            if (settled[i]) continue;
            EntryKey key{entries[i].mesh, entries[i].instanceIndex};
            prevWorldMats[key] = entries[i].worldMatrix;
        }
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
    //
    // Uniform-by-area within each tri × power-weighted picking across tris
    // gives a constant area-weighted-luminance pdf for closest_hit's NEE.
    //
    // Returns true when the buffer handle changed (capacity grew); caller
    // must then rewrite descriptor binding 14 for this frame's sets.
    bool VulkanRenderer::Impl::buildAndUploadEmissiveTris(uint32_t frame,
                                    const std::vector<MeshEntry>& entries) {
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
            const bool grew = ensureEmissiveTriCapacity(frame, cachedEmissiveTriCount_);
            uploadHostVisible(ctx->allocator(), emissiveTriBuffers[frame],
                              cachedEmissiveData_.data(),
                              cachedEmissiveData_.size() * sizeof(float));
            emissiveBufferVersion_[frame] = cachedEmissiveVersion_;
            return grew;
        }

        std::vector<float> data;// 16 floats per tri
        data.reserve(64 * 16);
        float cumPower = 0.0f;

        for (const auto& en : entries) {
            if (en.isOverlay) continue;// raster-overlay only — no emissive contribution to the traced scene
            if (!en.mesh) continue;
            auto matPtr = en.mesh->material();
            if (!matPtr) continue;
            auto* em = dynamic_cast<MaterialWithEmissive*>(matPtr.get());
            if (!em) continue;
            const float emR = em->emissive.r * em->emissiveIntensity;
            const float emG = em->emissive.g * em->emissiveIntensity;
            const float emB = em->emissive.b * em->emissiveIntensity;
            const float emLum = 0.2126f * emR + 0.7152f * emG + 0.0722f * emB;
            if (emLum < 1e-6f) continue;
            // emissiveMap modulates per-texel; we don't sample textures here,
            // so use the constant tint for power. Slightly under-samples
            // bright textured emissives but keeps the build lightweight.

            auto geomPtr = en.mesh->geometry();
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

            const float* M = en.worldMatrix.data();// column-major 4x4
            auto xform = [&](float x, float y, float z, float& wx, float& wy, float& wz) {
                wx = M[0] * x + M[4] * y + M[8]  * z + M[12];
                wy = M[1] * x + M[5] * y + M[9]  * z + M[13];
                wz = M[2] * x + M[6] * y + M[10] * z + M[14];
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

                data.push_back(w0x); data.push_back(w0y); data.push_back(w0z);
                data.push_back(area);
                data.push_back(w1x); data.push_back(w1y); data.push_back(w1z);
                data.push_back(cumPower);
                data.push_back(w2x); data.push_back(w2y); data.push_back(w2z);
                data.push_back(power);
                data.push_back(emR); data.push_back(emG); data.push_back(emB);
                data.push_back(0.0f);
            }
        }

        const uint32_t triCount = static_cast<uint32_t>(data.size() / 16);
        emissiveTriCountThisFrame_ = triCount;
        emissiveTotalPowerThisFrame_ = cumPower;

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

        const bool grew = ensureEmissiveTriCapacity(frame, triCount);

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
        motionMatBufferAllIdentity_[frame] = false;
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
        // setViewOffset — carrying them is what keeps the principal point
        // honest for an off-centre sensor.
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
                // frustum height at the far plane — the closest honest answer,
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
        // Event camera ON also forces UNJITTERED: event_shade reads the raw
        // raster gbuf, and per-frame Halton jitter dithers silhouette coverage
        // so a STATIC scene fires spurious +/- events every frame (the "event
        // view flickers with no motion" bug). A physical DVS never sees TAA
        // jitter. Must match uploadRasterCameraUbo's identical gate.
        // FSR requires jitter to reconstruct — force it on whenever FSR is the
        // active upscaler (even under MSAA, which otherwise rasterizes unjittered),
        // so the dispatch jitterOffset matches what was rendered. Event camera
        // still wins (a real DVS never sees jitter). Must match uploadRasterCameraUbo.
        const bool rasterJitterOn = !eventCamEnabled_ &&
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
                // A parallel projection has no field of viewMat. FSR only feeds
                // fovY into its disocclusion/reactivity heuristics, so hand it
                // the angle a perspective camera would need to cover the same
                // frustum height at the far plane — the closest honest answer,
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
        // Event camera ON forces UNJITTERED (must match updateCameraUbo's
        // gate): event_shade reads this raw gbuf, and the per-frame Halton
        // coverage flip at silhouettes makes a STATIC scene emit spurious
        // events every frame — the DVS "flickers with no motion". A real
        // event camera sees no jitter; transform/camera motion still flows
        // through motionMat, so genuine motion events are unaffected.
        // useFsr()/useDlss() force jitter on: both need it to reconstruct, even
        // under MSAA (which otherwise rasterizes unjittered), so the dispatch
        // jitterOffset matches the render. Event camera still wins (a real DVS
        // sees no jitter).
        const bool rasterJitterOn =
                kRasterJitterEnabled && !eventCamEnabled_ &&
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
        // because prevJitter.zw is otherwise unread by any gbuffer shader —
        // see normalMapToksvig_. .w stays reserved/unused.
        ubo.prevJitter[2] = normalMapToksvig_ ? 1.f : 0.f;
        ubo.prevJitter[3] = 0.f;

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
            // (HemisphereLight/LightProbe pass this gate and fall through the
            // chain unhandled, exactly as before.)
            if (!dynamic_cast<Light*>(&o)) return;
            if (auto* a = dynamic_cast<AmbientLight*>(&o)) {
                ubo.ambient[0] += a->color.r * a->intensity;
                ubo.ambient[1] += a->color.g * a->intensity;
                ubo.ambient[2] += a->color.b * a->intensity;
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
        static const auto cloudEpoch = std::chrono::steady_clock::now();
        ubo.timeSec     = std::chrono::duration<float>(
                                  std::chrono::steady_clock::now() - cloudEpoch).count();
        // Heterogeneous near-field froxels run whenever an AIR medium exists this
        // frame — Phase 2: scene.fog alone is enough (the resolved medium below),
        // not only the explicit setHeightFog. The froxel medium is the height-fog
        // profile ONLY — the far cloud march already integrates the cloud over the
        // whole 0→far ray (including below 512 m), so folding cloudDensity into the
        // froxels too would double-count it (see mediumExtinction in cloud_density.glsl).
        // updateFogUbo ran first (VulkanCoreFrame.cpp) and resolved the medium.
        ubo.heteroActive  = mediumActiveThisFrame_ ? 1.0f : 0.0f;
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

}// namespace threepp
