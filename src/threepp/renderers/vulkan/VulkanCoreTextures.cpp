#include "VulkanCoreImpl.hpp"

#include <random>

namespace threepp {

void VulkanRenderer::Impl::ensureParticleWhiteTexture() {
            if (particleWhiteTex_.view != VK_NULL_HANDLE) return;
            const uint8_t white[4] = {255, 255, 255, 255};
            particleWhiteTex_ = createSampledImage2D(
                    1, 1, VK_FORMAT_R8G8B8A8_UNORM, white, sizeof(white),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "particleWhiteDefault");
        }

void VulkanRenderer::Impl::destroyParticleGeomRec(ParticleGeomRec& rec) {
            destroyBuffer(ctx->allocator(), rec.position);
            destroyBuffer(ctx->allocator(), rec.normal);
            destroyBuffer(ctx->allocator(), rec.uv);
            destroyBuffer(ctx->allocator(), rec.color);
            destroyBuffer(ctx->allocator(), rec.index);
        }

void VulkanRenderer::Impl::createDefaultEnvImage() {
            const float pixels[4] = {0.f, 0.f, 0.f, 1.f};
            envImage = createSampledImage2D(
                    1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                    pixels, sizeof(pixels),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "envImage(default 1x1 black)");
            envIsDefault = true;
            envIsBgColor = false;
            envTextureIdUploaded = 0xFFFFFFFFu;
        }

// Shared boilerplate for the material policy samplers: linear/trilinear,
// aniso as requested (1 = isotropic), REPEAT or CLAMP_TO_EDGE wrap.
static VkSampler createMaterialSamplerWithAniso(VkDevice device, float aniso,
                                                VkSamplerAddressMode addr = VK_SAMPLER_ADDRESS_MODE_REPEAT) {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = addr;
            sci.addressModeV = addr;
            sci.addressModeW = addr;
            sci.anisotropyEnable = aniso > 1.0f ? VK_TRUE : VK_FALSE;
            sci.maxAnisotropy = std::max(aniso, 1.0f);
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = VK_LOD_CLAMP_NONE;
            VkSampler out = VK_NULL_HANDLE;
            check(vkCreateSampler(device, &sci, nullptr, &out),
                  "vkCreateSampler(material)");
            return out;
        }

void VulkanRenderer::Impl::createTextureSampler() {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const float maxAniso = std::min(16.0f, props.limits.maxSamplerAnisotropy);
            // The material-sampler set (see the member comment): the 16×
            // pair AUTO always hands out, and the isotropic pair that only a
            // forced setTextureAnisotropy(1) reaches — each in a REPEAT and a
            // CLAMP_TO_EDGE flavour.
            textureSampler_    = createMaterialSamplerWithAniso(ctx->device(), maxAniso);
            textureSamplerIso_ = createMaterialSamplerWithAniso(ctx->device(), 1.0f);
            textureSamplerClamp_    = createMaterialSamplerWithAniso(ctx->device(), maxAniso,
                                                                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            textureSamplerIsoClamp_ = createMaterialSamplerWithAniso(ctx->device(), 1.0f,
                                                                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            // THREEPP_VK_ANISO=<n>: startup override (same semantics as
            // setTextureAnisotropy — 0 = auto policy, 1..16 forces a level).
            if (const char* a = std::getenv("THREEPP_VK_ANISO"); a && a[0] != '\0') {
                textureAnisoOverride_ = std::clamp(std::strtof(a, nullptr), 0.0f, maxAniso);
                std::cout << "textureAnisotropy override = " << textureAnisoOverride_ << "\n";
            }
        }

VkSampler VulkanRenderer::Impl::materialSampler(bool clampUV) {
            // AUTO policy: 16x anisotropic, jittered or not. An earlier policy
            // dropped to isotropic under a jittered raster (TAA/DLSS/FSR) on
            // the theory that aniso re-sharpened grazing-angle content into
            // temporal shimmer — later triage attributed that shimmer to other
            // sources, and the isotropic fallback just mip-blurred every
            // ground/facade texture at distance for no stability gain.
            // setTextureAnisotropy / THREEPP_VK_ANISO remain as explicit
            // overrides (1 forces the old isotropic behaviour).
            const float want = (textureAnisoOverride_ >= 1.0f)
                                       ? textureAnisoOverride_
                                       : 16.0f;
            if (want <= 1.0f) return clampUV ? textureSamplerIsoClamp_ : textureSamplerIso_;
            if (want >= 16.0f) return clampUV ? textureSamplerClamp_ : textureSampler_;
            if (textureSamplerCustom_ != VK_NULL_HANDLE && textureSamplerCustomAniso_ == want)
                return clampUV ? textureSamplerCustomClamp_ : textureSamplerCustom_;
            // Forced intermediate level: build the wrap pair lazily; PARK the
            // previous customs (in-flight frames / bound sets may still
            // reference them — samplers are tiny, teardown reclaims them).
            if (textureSamplerCustom_ != VK_NULL_HANDLE)
                parkedSamplers_.push_back(textureSamplerCustom_);
            if (textureSamplerCustomClamp_ != VK_NULL_HANDLE)
                parkedSamplers_.push_back(textureSamplerCustomClamp_);
            textureSamplerCustom_ = createMaterialSamplerWithAniso(ctx->device(), want);
            textureSamplerCustomClamp_ = createMaterialSamplerWithAniso(
                    ctx->device(), want, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            textureSamplerCustomAniso_ = want;
            return clampUV ? textureSamplerCustomClamp_ : textureSamplerCustom_;
        }

void VulkanRenderer::Impl::setTextureAnisotropy(float aniso) {
            const float clamped = aniso <= 0.f ? 0.f : std::clamp(aniso, 1.0f, 16.0f);
            if (clamped == textureAnisoOverride_) return;
            textureAnisoOverride_ = clamped;
            markMaterialSamplerDirty();
        }

void VulkanRenderer::Impl::createDefaultMaterialTexture() {
            const uint8_t white[4] = {255, 255, 255, 255};
            Image2D tex = createSampledImage2D(
                    1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                    white, sizeof(white),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "materialTexture[0] (default white)");
            // The Image2D's own sampler is unused (we share textureSampler_),
            // but kept alive so destroyImage2D's clean-up still works.
            materialTextures.push_back(tex);
        }

int32_t VulkanRenderer::Impl::ensureMaterialTexture(const std::shared_ptr<Texture>& texSp) {
            if (!texSp) return -1;
            const Texture* tex = texSp.get();
            if (auto it = textureCache.find(tex); it != textureCache.end()) {
                // ADDRESS-RECYCLING guard: the key is a raw Texture*, so a
                // brand-new Texture allocated at a dead one's address hits the
                // dead entry. The fingerprint's held shared_ptrs prevent this
                // for its map slots, but not for textures that died after the
                // last prune. Only a LIVE original returns its cached slot;
                // a stale hit retires the entry (deferred reclaim — see
                // retiredTextureSlots_) and falls through to a fresh upload.
                if (auto sp = it->second.ref.lock(); sp && sp.get() == tex) {
                    return static_cast<int32_t>(it->second.slot);
                }
                retiredTextureSlots_.push_back(it->second.slot);
                textureCache.erase(it);
            }
            if (freeTextureSlots.empty() && materialTextures.size() >= kMaxMaterialTextures) {
                std::cerr << "[VulkanRenderer] material texture slots exhausted ("
                          << kMaxMaterialTextures << "); using -1 fallback\n";
                return -1;
            }
            Image2D out = buildMaterialImage2D(tex);
            if (!out.view) return -1;
            // Clamp-to-edge on BOTH axes selects the clamp policy sampler for
            // this slot (edge-inclusive atlases: terrain splats, UI sheets).
            // Mixed / mirrored wraps keep the REPEAT policy sampler.
            const uint8_t clampUV = (tex->wrapS == TextureWrapping::ClampToEdge &&
                                     tex->wrapT == TextureWrapping::ClampToEdge)
                                            ? 1u
                                            : 0u;
            uint32_t slot;
            if (!freeTextureSlots.empty()) {
                slot = freeTextureSlots.back();
                freeTextureSlots.pop_back();
                materialTextures[slot] = out;
            } else {
                slot = static_cast<uint32_t>(materialTextures.size());
                materialTextures.push_back(out);
            }
            if (materialTexClampUV_.size() <= slot) materialTexClampUV_.resize(slot + 1, 0u);
            materialTexClampUV_[slot] = clampUV;
            textureCache.emplace(tex, CachedTexture{std::weak_ptr<Texture>(texSp), slot, tex->version()});
            return static_cast<int32_t>(slot);
        }

void VulkanRenderer::Impl::refreshDirtyMaterialTextures() {
            bool any = false;
            for (auto& kv : textureCache) {
                const auto sp = kv.second.ref.lock();
                if (sp && sp->version() != kv.second.version) { any = true; break; }
            }
            if (!any) return;
            // No device drain: prior in-flight frames may still sample the OLD
            // images, so we RETIRE them (they stay alive until their referencing
            // frames provably complete — VulkanRetireQueue.hpp) and swap the
            // freshly-built image into the same bindless slot. Was a
            // vkDeviceWaitIdle stall on every live DataTexture edit.
            for (auto& kv : textureCache) {
                const auto sp = kv.second.ref.lock();
                if (!sp || sp->version() == kv.second.version) continue;
                if (kv.second.slot >= materialTextures.size()) continue;
                Image2D rebuilt = buildMaterialImage2D(kv.first);
                if (!rebuilt.view) continue;// keep the old image on failure
                retire(std::move(materialTextures[kv.second.slot]));
                materialTextures[kv.second.slot] = rebuilt;// same slot index, new view
                if (materialTexClampUV_.size() <= kv.second.slot)
                    materialTexClampUV_.resize(kv.second.slot + 1, 0u);
                materialTexClampUV_[kv.second.slot] =
                        (sp->wrapS == TextureWrapping::ClampToEdge &&
                         sp->wrapT == TextureWrapping::ClampToEdge)
                                ? 1u
                                : 0u;
                kv.second.version = sp->version();
            }
            // The bindless material array is referenced by the deferred-compute
            // set (binding 11) and the gbuffer raster set (binding 3). Neither
            // can be rewritten in place while in flight (no update-after-bind on
            // this device), so DEFER both: mark all FIF deferred sets dirty (the
            // frame-begin path rewrites only the current, fence-idle slot), and
            // invalidate the per-frame raster-set validity so binding 3 re-writes
            // on each slot's next uploadRasterCameraUbo. Both are already
            // per-frame-lazy, so the new views land the frame each slot renders.
            deferredDescDirty_.fill(true);
            // Every view — the flag guards per-view raster sets, and a
            // secondary skipped here keeps sampling the retired images.
            for (auto& v : views_) v->rasterMatTexValid_.fill(0);
        }

std::vector<uint8_t> VulkanRenderer::Impl::generateBlueNoiseTile_() {
            constexpr int N = 64;
            constexpr int M = N * N;
            constexpr float kSigma = 1.5f;
            constexpr int kRadius = 4;
            constexpr int kKernelSide = 2 * kRadius + 1;

            auto wrap = [](int v, int n) { return ((v % n) + n) % n; };

            // Precomputed Gaussian kernel for void/cluster filter.
            std::array<float, kKernelSide * kKernelSide> kernel{};
            for (int dy = -kRadius; dy <= kRadius; ++dy) {
                for (int dx = -kRadius; dx <= kRadius; ++dx) {
                    const int ki = (dy + kRadius) * kKernelSide + (dx + kRadius);
                    kernel[ki] = std::exp(-(dx * dx + dy * dy) /
                                          (2.0f * kSigma * kSigma));
                }
            }

            std::vector<uint8_t> binary(M, 0u);// 0 / 1 instead of bool for vec speed
            std::vector<float>   density(M, 0.0f);

            // Toggle a cell and incrementally update the density field over
            // the Gaussian kernel footprint (toroidal wrap for tileable noise).
            auto toggle = [&](int idx, bool toOn) {
                binary[idx] = toOn ? 1u : 0u;
                const int x = idx % N;
                const int y = idx / N;
                const float sign = toOn ? +1.0f : -1.0f;
                for (int dy = -kRadius; dy <= kRadius; ++dy) {
                    const int yy = wrap(y + dy, N);
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        const int xx = wrap(x + dx, N);
                        const int ki = (dy + kRadius) * kKernelSide + (dx + kRadius);
                        density[yy * N + xx] += sign * kernel[ki];
                    }
                }
            };

            // Find the tightest cluster: the "1" pixel with max density.
            auto findTightest = [&]() {
                int best = -1;
                float bestD = -1.0f;
                for (int i = 0; i < M; ++i) {
                    if (binary[i] && density[i] > bestD) {
                        bestD = density[i];
                        best = i;
                    }
                }
                return best;
            };
            // Find the largest void: the "0" pixel with min density.
            auto findLargestVoid = [&]() {
                int best = -1;
                float bestD = std::numeric_limits<float>::infinity();
                for (int i = 0; i < M; ++i) {
                    if (!binary[i] && density[i] < bestD) {
                        bestD = density[i];
                        best = i;
                    }
                }
                return best;
            };

            // Initial random pattern (~10% set). Deterministic seed for
            // reproducible tiles.
            constexpr int kInitOnes = M / 10;
            {
                std::mt19937 rng(0x12345678u);
                int placed = 0;
                while (placed < kInitOnes) {
                    const int idx = static_cast<int>(rng() % static_cast<uint32_t>(M));
                    if (!binary[idx]) {
                        toggle(idx, true);
                        ++placed;
                    }
                }
            }

            // Phase 1: stabilize via swap (tightest → void). Up to 200 iters;
            // typically converges in <100 for 64×64.
            for (int iter = 0; iter < 200; ++iter) {
                const int tight = findTightest();
                const int voidIdx = findLargestVoid();
                if (tight < 0 || voidIdx < 0 || tight == voidIdx) break;
                toggle(tight, false);
                toggle(voidIdx, true);
            }

            // Snapshot the stabilized prototype for phases 2 + 3.
            const std::vector<uint8_t> proto    = binary;
            const std::vector<float>   protoDen = density;

            std::vector<int> rank(M, 0);

            // Phase 2: rank prototype "ones" by progressive removal of the
            // tightest cluster. Lowest rank = first removed = most isolated.
            for (int r = kInitOnes - 1; r >= 0; --r) {
                const int tight = findTightest();
                if (tight < 0) break;
                rank[tight] = r;
                toggle(tight, false);
            }

            // Restore the prototype for phase 3.
            binary = proto;
            density = protoDen;

            // Phase 3: rank prototype "zeros" by progressive addition into the
            // largest void. Higher rank = added later = more clustered.
            for (int r = kInitOnes; r < M; ++r) {
                const int voidIdx = findLargestVoid();
                if (voidIdx < 0) break;
                rank[voidIdx] = r;
                toggle(voidIdx, true);
            }

            // Map rank ∈ [0, M-1] → uint8 ∈ [0, 255].
            std::vector<uint8_t> tile(static_cast<size_t>(M));
            for (int i = 0; i < M; ++i) {
                tile[i] = static_cast<uint8_t>((rank[i] * 255) / (M - 1));
            }
            return tile;
        }

void VulkanRenderer::Impl::createBlueNoiseImage_() {
            const auto tile = generateBlueNoiseTile_();
            blueNoiseImage = createSampledImage2D(
                    /*w*/ 64u, /*h*/ 64u,
                    VK_FORMAT_R8_UNORM,
                    tile.data(), tile.size(),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "blueNoiseImage (64x64 void-and-cluster tile)");
        }

std::vector<unsigned char> VulkanRenderer::Impl::generateFoamDetailTile_(int res) {
            auto hashf = [](int x, int y) {
                uint32_t h = static_cast<uint32_t>(x) * 374761393u +
                             static_cast<uint32_t>(y) * 668265263u;
                h = (h ^ (h >> 13)) * 1274126177u;
                h ^= h >> 16;
                return static_cast<float>(h) * (1.0f / 4294967296.0f);
            };
            // Tileable value noise: `cells` lattice points per tile edge,
            // wrapped; `seed` decorrelates octaves sharing a frequency.
            auto vnoiseT = [&](float u, float v, int cells, int seed) {
                const float x = u * static_cast<float>(cells);
                const float y = v * static_cast<float>(cells);
                const int xi = static_cast<int>(std::floor(x));
                const int yi = static_cast<int>(std::floor(y));
                float tx = x - static_cast<float>(xi);
                float ty = y - static_cast<float>(yi);
                tx = tx * tx * (3.f - 2.f * tx);
                ty = ty * ty * (3.f - 2.f * ty);
                auto at = [&](int ix, int iy) {
                    return hashf(((ix % cells) + cells) % cells + seed * 7919,
                                 ((iy % cells) + cells) % cells);
                };
                const float a = at(xi, yi), b = at(xi + 1, yi);
                const float c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
                return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
            };
            std::vector<unsigned char> px(static_cast<size_t>(res) * res * 2);
            for (int y = 0; y < res; ++y) {
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(res);
                for (int x = 0; x < res; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(res);
                    const float micro = 0.55f * vnoiseT(u, v, 26, 1) +
                                        0.30f * vnoiseT(u, v, 60, 2) +
                                        0.15f * vnoiseT(u, v, 132, 3);
                    const float nf = 0.62f * vnoiseT(u, v, 10, 4) +
                                     0.38f * vnoiseT(u, v, 23, 5);
                    const float lace = 1.f - std::fabs(nf - 0.5f) * 2.f;
                    const size_t i = (static_cast<size_t>(y) * res + x) * 2;
                    px[i + 0] = static_cast<unsigned char>(std::lround(std::clamp(micro, 0.f, 1.f) * 255.f));
                    px[i + 1] = static_cast<unsigned char>(std::lround(std::clamp(lace, 0.f, 1.f) * 255.f));
                }
            }
            return px;
        }

void VulkanRenderer::Impl::createFoamDetailImage_() {
            constexpr int kRes = 512;
            const auto tile = generateFoamDetailTile_(kRes);
            // LINEAR + size>1 → createSampledImage2D builds the full mip
            // chain (blit cascade) and a trilinear/aniso REPEAT sampler.
            foamDetailImage = createSampledImage2D(
                    /*w*/ kRes, /*h*/ kRes,
                    VK_FORMAT_R8G8_UNORM,
                    tile.data(), tile.size(),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "foamDetail (512x512 RG8 bubbles+lace, mipped)");
        }

void VulkanRenderer::Impl::createOceanFineDummy_() {
            const float zero = 0.0f;
            oceanFineHeightDummy = createSampledImage2D(
                    /*w*/ 1u, /*h*/ 1u,
                    VK_FORMAT_R32_SFLOAT,
                    &zero, sizeof(zero),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "oceanFineHeightDummy (1x1, binding 32 placeholder)");
            // Transition to GENERAL so the descriptor layout matches the
            // cascade-2 storage image's layout (also GENERAL after IFFT). One
            // declared layout simplifies the descriptor rewrite.
            {
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.image = oceanFineHeightDummy.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &imb);
                endAndSubmitOneShot(cb);
            }
            oceanFineHeightView    = oceanFineHeightDummy.view;
            oceanFineHeightSampler = oceanFineHeightDummy.sampler;
            oceanFineTileSize      = 0.0f;
        }

void VulkanRenderer::Impl::createOceanFoamDummy_() {
            const float zero = 0.0f;
            oceanFoamDummy = createSampledImage2D(
                    /*w*/ 1u, /*h*/ 1u,
                    VK_FORMAT_R32_SFLOAT,
                    &zero, sizeof(zero),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "oceanFoamDummy (1x1, binding 44 placeholder)");
            // Transition to GENERAL so the binding matches the foam image's
            // layout (compute leaves it in GENERAL; chit reads from there).
            {
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.image = oceanFoamDummy.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &imb);
                endAndSubmitOneShot(cb);
            }
            oceanFoamView     = oceanFoamDummy.view;
            oceanFoamSampler  = oceanFoamDummy.sampler;
            oceanFoamTileSize = 0.0f;
        }

bool VulkanRenderer::Impl::refreshEnvTextureFromScene(Object3D& scene) {
            auto* sc = dynamic_cast<Scene*>(&scene);
            std::shared_ptr<Texture> tex;
            if (sc) {
                tex = sc->environment;
                if (!tex && sc->background.isTexture()) {
                    tex = sc->background.texture();
                }
            }
            if (!tex) {
                // No environment and no background texture: a background COLOR
                // takes over, and with no background at all the renderer's
                // clearColor is the background — GL parity (glClearColor shows
                // wherever nothing is drawn). Default clearColor is black, so
                // scenes that set neither look exactly as before.
                if (sc) {
                    const Color& c = sc->background.isColor() ? sc->background.color()
                                                              : clearColor;
                    if (envIsBgColor && envBgColor.r == c.r && envBgColor.g == c.g && envBgColor.b == c.b)
                        return false;
                    // Retire the old env image (in-flight frames' descriptor
                    // sets may still sample it); the sole caller
                    // (beginDeferredFrame) drains + flushes + rewrites right after.
                    retire(std::move(envImage));
                    const float px[4] = {c.r, c.g, c.b, 1.f};
                    envImage = createSampledImage2D(
                            1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                            px, sizeof(px),
                            VK_FILTER_NEAREST,
                            VK_SAMPLER_ADDRESS_MODE_REPEAT,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            "envImage(scene.background color)");
                    envIsDefault  = false;
                    envIsBgColor  = true;
                    envBgColor    = c;
                    envTextureIdUploaded = 0xFFFFFFFFu;
                    envSun_ = {};
                    return true;
                }
                if (envIsDefault) return false;
                retire(std::move(envImage));// caller drains + flushes + rewrites
                createDefaultEnvImage();
                envIsBgColor = false;
                envSun_ = {};
                return true;
            }
            envIsBgColor = false;
            if (!envIsDefault && tex->id == envTextureIdUploaded) return false;

            // Only HDR float equirects (RGBELoader output) are supported.
            // LDR backgrounds are not handled — fall through to default.
            const auto& image = tex->image();
            if (tex->type != Type::Float) {
                std::cerr << "[VulkanRenderer] env texture is not float HDR; "
                             "ignoring (HDR equirect only)" << std::endl;
                return false;
            }
            const auto& src = image.data<float>();
            const uint32_t w = image.width();
            const uint32_t h = image.height();
            if (w == 0 || h == 0 || src.size() < 4u * w * h) {
                std::cerr << "[VulkanRenderer] env texture has unexpected size; ignoring" << std::endl;
                return false;
            }

            retire(std::move(envImage));// caller drains + flushes + rewrites

            // GGX-prefilter the env into a mip chain so the closest_hit
            // shader can sample at a roughness-derived LOD instead of the cheap
            // (1-r)² fade. Mip 0 is the source mirror, mip k is convolved with
            // GGX(α=(k/(N-1))²). closest_hit fades from mirror to fully diffuse
            // by walking the chain via textureLod.
            // Deferred leaf: extract the HDRI sun (mips 1+ sun-free; the disc's
            // energy returns via updateLightsUbo as an analytic dir light).
            envImage = envPrefilter_->buildPmrem(
                    w, h, src.data(), 4u * w * h * sizeof(float),
                    envSunExtractionWanted(), &envSun_);
            envIsDefault = false;
            envTextureIdUploaded = tex->id;
            return true;
        }


const Image2D* VulkanRenderer::Impl::ensureParticleTexture(const Texture* tex) {
            if (!tex) return nullptr;
            Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width();
            const uint32_t h = img.height();
            if (w == 0 || h == 0) return nullptr;

            const unsigned int curVersion = tex->version();
            auto it = particleTexCache_.find(tex);
            if (it != particleTexCache_.end()) {
                ParticleTexRec& rec = it->second;
                const bool stale = rec.version != curVersion ||
                                   rec.width != w || rec.height != h;
                if (!stale) return &rec.image;
                // Retire the old image instead of a full device drain: in-flight
                // frames may still sample it (particle descriptors are allocated
                // per-frame from reset pools, so there's no descriptor-set hazard
                // — image lifetime is the only concern). VulkanRetireQueue.hpp.
                retire(std::move(rec.image));
                particleTexCache_.erase(it);
            }

            std::vector<unsigned char> rgba;
            const size_t pixels = static_cast<size_t>(w) * h;
            try {
                auto& src = img.data<unsigned char>();
                if (src.size() == pixels * 4) {
                    rgba.assign(src.begin(), src.end());
                } else if (src.size() == pixels * 3) {
                    rgba.resize(pixels * 4);
                    for (size_t i = 0; i < pixels; ++i) {
                        rgba[i * 4 + 0] = src[i * 3 + 0];
                        rgba[i * 4 + 1] = src[i * 3 + 1];
                        rgba[i * 4 + 2] = src[i * 3 + 2];
                        rgba[i * 4 + 3] = 255u;
                    }
                } else {
                    return nullptr;
                }
            } catch (const std::bad_variant_access&) {
                return nullptr;
            }

            // Same colorSpace→format rule as the sprite/bindless paths: only an
            // explicitly sRGB-tagged texture gets hardware sRGB decode on sample;
            // particle.frag re-encodes the linear product for the UNORM swapchain.
            const VkFormat fmt = (tex->colorSpace == ColorSpace::sRGB)
                                         ? VK_FORMAT_R8G8B8A8_SRGB
                                         : VK_FORMAT_R8G8B8A8_UNORM;
            char name[64];
            std::snprintf(name, sizeof(name), "particleTex[%p]",
                          static_cast<const void*>(tex));
            Image2D up = createSampledImage2D(
                    w, h, fmt, rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    name);
            ParticleTexRec rec{};
            rec.image   = up;
            rec.version = curVersion;
            rec.width   = w;
            rec.height  = h;
            auto [ins, _] = particleTexCache_.emplace(tex, std::move(rec));
            return &ins->second.image;
        }

Image2D VulkanRenderer::Impl::buildMaterialImage2D(const Texture* tex) {
            Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width();
            const uint32_t h = img.height();
            if (w == 0 || h == 0) return {};

            // Half-float CPU pixel data (ImageData's uint16 alternative) has no
            // path here yet: everything below normalises to RGBA8 and the two
            // decoders it knows are u8 and f32. Say so and return the null
            // image, which the callers already treat as "no texture" -- the
            // alternative is std::bad_variant_access out of img.data<float>()
            // several branches down, which is a worse way to learn the same
            // thing. SplatCloud, the only half-float DataTexture in the tree,
            // is GL-only.
            if (img.isHalfFloat()) {
                std::cerr << "[VulkanRenderer] half-float DataTexture is not supported yet ("
                          << w << "x" << h << "); ignoring\n";
                return {};
            }

            // Normalise everything to tightly-packed RGBA8. The pipeline
            // treats the bindless array as a uniform u8x4 sampler set, so
            // BCn blocks decompress, mono/dual-channel maps replicate or
            // pad, and float defaults clamp+quantize.
            const size_t pixels = static_cast<size_t>(w) * h;
            std::vector<unsigned char> rgba;
            std::vector<std::uint8_t> bcnRgba;
            const std::uint8_t* srcPtr = nullptr;
            int channels = 0;

            if (img.compressedFormat.has_value()) {
                // Pass-through: if the file already ships BC blocks (DDS) AND a
                // complete mip chain, upload them VERBATIM — no decode, no
                // re-encode, no generation loss, and BC1 sources stay at their
                // native 0.5 B/px (half of what a BC7 transcode would use).
                // Anything short of that (partial chain, unmapped format,
                // THREEPP_NO_BC) falls through to the decode path below, which
                // re-encodes to BC7 with CPU-built mips.
                static const bool noBcPass = std::getenv("THREEPP_NO_BC") != nullptr;
                if (!noBcPass && bc7SampledSupported()) {
                    const VkFormat vkFmt = glCompressedToVk(
                            *img.compressedFormat, tex->colorSpace == ColorSpace::sRGB);
                    const uint32_t fullLevels =
                            1u + static_cast<uint32_t>(std::floor(std::log2(
                                         static_cast<float>(std::max(w, h)))));
                    if (vkFmt != VK_FORMAT_UNDEFINED &&
                        tex->mipmaps().size() + 1 == fullLevels) {
                        std::vector<std::vector<std::uint8_t>> levels;
                        levels.reserve(fullLevels);
                        bool ok = true;
                        try {
                            levels.push_back(img.data<unsigned char>());
                            uint32_t lw = w, lh = h;
                            for (auto& mip : const_cast<Texture*>(tex)->mipmaps()) {
                                lw = std::max(1u, lw >> 1);
                                lh = std::max(1u, lh >> 1);
                                if (mip.width() != lw || mip.height() != lh ||
                                    !mip.compressedFormat.has_value() ||
                                    *mip.compressedFormat != *img.compressedFormat) {
                                    ok = false;
                                    break;
                                }
                                levels.push_back(mip.data<unsigned char>());
                            }
                        } catch (const std::bad_variant_access&) {
                            ok = false;
                        }
                        if (ok) {
                            char bcName[80];
                            std::snprintf(bcName, sizeof(bcName),
                                          "materialTexture BC-passthrough (%ux%u, tex=%p)",
                                          w, h, static_cast<const void*>(tex));
                            return createSampledImageBC(
                                    w, h, vkFmt, levels,
                                    VK_FILTER_LINEAR,
                                    wrapToVk(tex->wrapS),
                                    wrapToVk(tex->wrapT),
                                    bcName);
                        }
                    }
                }

                const auto& blocks = img.data<unsigned char>();
                bcnRgba = bcn::bcnDecompress(
                        blocks.data(),
                        static_cast<int>(w),
                        static_cast<int>(h),
                        *img.compressedFormat);
                if (bcnRgba.empty()) {
                    std::cerr << "[VulkanRenderer] unsupported compressed format 0x"
                              << std::hex << *img.compressedFormat << std::dec
                              << " for material tex (" << w << "x" << h << ")\n";
                    return {};
                }
                srcPtr = bcnRgba.data();
                channels = 4;
            } else {
                bool isU8 = true;
                try {
                    auto& src = img.data<unsigned char>();
                    if (src.size() % pixels != 0) {
                        std::cerr << "[VulkanRenderer] unsupported pixel layout for material tex ("
                                  << src.size() << " bytes for " << w << "x" << h << ")\n";
                        return {};
                    }
                    channels = static_cast<int>(src.size() / pixels);
                    if (channels < 1 || channels > 4) {
                        std::cerr << "[VulkanRenderer] unsupported channel count " << channels
                                  << " for material tex (" << w << "x" << h << ")\n";
                        return {};
                    }
                    srcPtr = src.data();
                } catch (const std::bad_variant_access&) {
                    isU8 = false;
                }
                if (!isU8) {
                    // Float-pixel default (e.g. Bistro's 1×1 RGBA32F constants).
                    // Quantise to u8 with sRGB-agnostic clamp; tiny default
                    // textures only need the linear value, and HDR ranges are
                    // expressed via material scalars instead.
                    auto& srcF = img.data<float>();
                    if (srcF.size() % pixels != 0) {
                        std::cerr << "[VulkanRenderer] unsupported float-pixel layout for material tex ("
                                  << srcF.size() * sizeof(float) << " bytes for "
                                  << w << "x" << h << ")\n";
                        return {};
                    }
                    const int fch = static_cast<int>(srcF.size() / pixels);
                    if (fch < 1 || fch > 4) {
                        std::cerr << "[VulkanRenderer] unsupported float channel count " << fch
                                  << " for material tex\n";
                        return {};
                    }
                    rgba.resize(pixels * 4);
                    for (size_t i = 0; i < pixels; ++i) {
                        float r = srcF[i * fch + 0];
                        float g = (fch >= 2) ? srcF[i * fch + 1] : r;
                        float b = (fch >= 3) ? srcF[i * fch + 2] : ((fch == 1) ? r : 0.f);
                        float a = (fch >= 4) ? srcF[i * fch + 3] : 1.f;
                        auto q = [](float v) {
                            if (!(v == v)) v = 0.f;// NaN→0
                            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                            return static_cast<unsigned char>(v * 255.f + 0.5f);
                        };
                        rgba[i * 4 + 0] = q(r);
                        rgba[i * 4 + 1] = q(g);
                        rgba[i * 4 + 2] = q(b);
                        rgba[i * 4 + 3] = q(a);
                    }
                }
            }

            // Expand srcPtr (BCn or u8) into rgba; float branch already filled it.
            if (rgba.empty()) {
                rgba.resize(pixels * 4);
                if (channels == 4) {
                    std::memcpy(rgba.data(), srcPtr, pixels * 4);
                } else {
                    for (size_t i = 0; i < pixels; ++i) {
                        const unsigned char r = srcPtr[i * channels + 0];
                        const unsigned char g = (channels >= 2) ? srcPtr[i * channels + 1] : r;
                        const unsigned char b = (channels >= 3) ? srcPtr[i * channels + 2]
                                                                : ((channels == 1) ? r : 0);
                        const unsigned char a = (channels >= 4) ? srcPtr[i * channels + 3] : 255u;
                        rgba[i * 4 + 0] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                }
            }

            // sRGB tag → hardware decode at sample time. Loaders should mark
            // albedo maps as SRGBColorSpace; legacy (untagged) textures fall
            // through as UNORM so the shader sees raw channel values.
            const bool srgb = tex->colorSpace == ColorSpace::sRGB;
            char texName[80];
            std::snprintf(texName, sizeof(texName),
                          "materialTexture (%ux%u, tex=%p)",
                          w, h, static_cast<const void*>(tex));

            // BC7 transcode: 4x less VRAM and 4x less bandwidth per sample.
            // Everything above was normalised to RGBA8 — including BCn (DDS)
            // sources, which the old path decompressed and uploaded FAT at 4x
            // their on-disk size. Mips are built + encoded on the CPU (BC
            // images cannot blit-downsample); tiny images (LUT-like defaults,
            // 1x1 constants) stay uncompressed, as does everything when the
            // device lacks BC7 or THREEPP_NO_BC=1 (same-binary A/B hatch).
            static const bool noBc = std::getenv("THREEPP_NO_BC") != nullptr;
            if (!noBc && w >= 8u && h >= 8u && bc7SampledSupported()) {
                auto mips = bcn::buildMipChainRGBA8(rgba.data(), static_cast<int>(w),
                                                    static_cast<int>(h), srgb);
                std::vector<std::vector<std::uint8_t>> blocks;
                blocks.reserve(mips.size() + 1);
                blocks.push_back(bcn::bc7EncodeMode6(rgba.data(), static_cast<int>(w),
                                                     static_cast<int>(h)));
                int mw = static_cast<int>(w), mh = static_cast<int>(h);
                for (const auto& lvl : mips) {
                    mw = std::max(1, mw >> 1);
                    mh = std::max(1, mh >> 1);
                    blocks.push_back(bcn::bc7EncodeMode6(lvl.data(), mw, mh));
                }
                return createSampledImageBC(
                        w, h,
                        srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK,
                        blocks,
                        VK_FILTER_LINEAR,
                        wrapToVk(tex->wrapS),
                        wrapToVk(tex->wrapT),
                        texName);
            }

            const VkFormat fmt = srgb ? VK_FORMAT_R8G8B8A8_SRGB
                                      : VK_FORMAT_R8G8B8A8_UNORM;
            return createSampledImage2D(
                    w, h, fmt,
                    rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    wrapToVk(tex->wrapS),
                    wrapToVk(tex->wrapT),
                    texName);
        }

VkFormat VulkanRenderer::Impl::glCompressedToVk(unsigned int glFmt, bool srgb) {
            switch (glFmt) {
                case 0x83F0u:// DXT1 RGB
                    return srgb ? VK_FORMAT_BC1_RGB_SRGB_BLOCK : VK_FORMAT_BC1_RGB_UNORM_BLOCK;
                case 0x83F1u:// DXT1 RGBA
                case 0x8C4Cu:// DXT1 sRGB
                case 0x8C4Du:// DXT1 sRGB+A
                    return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                case 0x83F2u:// DXT3
                    return srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
                case 0x83F3u:// DXT5
                case 0x8C4Fu:// DXT5 sRGB
                    return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
                case 0x8DBBu:// RGTC1 / BC4
                    return VK_FORMAT_BC4_UNORM_BLOCK;
                case 0x8DBDu:// RGTC2 / BC5
                    return VK_FORMAT_BC5_UNORM_BLOCK;
                case 0x8E8Cu:// BPTC / BC7
                case 0x8E8Du:// BPTC sRGB
                    return srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
                default:
                    return VK_FORMAT_UNDEFINED;
            }
        }
}// namespace threepp
