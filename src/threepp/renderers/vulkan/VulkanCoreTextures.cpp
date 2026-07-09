#include "VulkanCoreImpl.hpp"

#include <random>

namespace threepp {

void VulkanRendererCore::CoreImpl::ensureParticleWhiteTexture() {
            if (particleWhiteTex_.view != VK_NULL_HANDLE) return;
            const uint8_t white[4] = {255, 255, 255, 255};
            particleWhiteTex_ = createSampledImage2D(
                    1, 1, VK_FORMAT_R8G8B8A8_UNORM, white, sizeof(white),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "particleWhiteDefault");
        }

void VulkanRendererCore::CoreImpl::destroyParticleGeomRec(ParticleGeomRec& rec) {
            destroyBuffer(ctx->allocator(), rec.position);
            destroyBuffer(ctx->allocator(), rec.normal);
            destroyBuffer(ctx->allocator(), rec.uv);
            destroyBuffer(ctx->allocator(), rec.color);
            destroyBuffer(ctx->allocator(), rec.index);
        }

void VulkanRendererCore::CoreImpl::createDefaultEnvImage() {
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

void VulkanRendererCore::CoreImpl::createTextureSampler() {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const float maxAniso = std::min(16.0f, props.limits.maxSamplerAnisotropy);

            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.anisotropyEnable = VK_TRUE;
            sci.maxAnisotropy = maxAniso;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = VK_LOD_CLAMP_NONE;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &textureSampler_),
                  "vkCreateSampler(material)");
        }

void VulkanRendererCore::CoreImpl::createDefaultMaterialTexture() {
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

int32_t VulkanRendererCore::CoreImpl::ensureMaterialTexture(const std::shared_ptr<Texture>& texSp) {
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
            uint32_t slot;
            if (!freeTextureSlots.empty()) {
                slot = freeTextureSlots.back();
                freeTextureSlots.pop_back();
                materialTextures[slot] = out;
            } else {
                slot = static_cast<uint32_t>(materialTextures.size());
                materialTextures.push_back(out);
            }
            textureCache.emplace(tex, CachedTexture{std::weak_ptr<Texture>(texSp), slot, tex->version()});
            return static_cast<int32_t>(slot);
        }

void VulkanRendererCore::CoreImpl::refreshDirtyMaterialTextures() {
            bool any = false;
            for (auto& kv : textureCache) {
                const auto sp = kv.second.ref.lock();
                if (sp && sp->version() != kv.second.version) { any = true; break; }
            }
            if (!any) return;
            // Prior in-flight frames may still sample these images; drain first.
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (material texture refresh)");
            for (auto& kv : textureCache) {
                const auto sp = kv.second.ref.lock();
                if (!sp || sp->version() == kv.second.version) continue;
                if (kv.second.slot >= materialTextures.size()) continue;
                Image2D rebuilt = buildMaterialImage2D(kv.first);
                if (!rebuilt.view) continue;// keep the old image on failure
                destroyImage2D(ctx->allocator(), ctx->device(), materialTextures[kv.second.slot]);
                materialTextures[kv.second.slot] = rebuilt;// same slot index, new view
                kv.second.version = sp->version();
            }
            // The bindless material array is referenced by the deferred-compute
            // set (binding 11), rewritten here; the gbuffer raster set (binding
            // 3) is refreshed lazily by invalidating its per-frame validity
            // flag. Both must see the new image views or the (deferred) gbuffer
            // keeps sampling the freed view.
            rewriteDeferredDescriptors();
            rasterMatTexValid_.fill(0);
        }

std::vector<uint8_t> VulkanRendererCore::CoreImpl::generateBlueNoiseTile_() {
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

void VulkanRendererCore::CoreImpl::createBlueNoiseImage_() {
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

std::vector<unsigned char> VulkanRendererCore::CoreImpl::generateFoamDetailTile_(int res) {
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

void VulkanRendererCore::CoreImpl::createFoamDetailImage_() {
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

void VulkanRendererCore::CoreImpl::createOceanFineDummy_() {
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

void VulkanRendererCore::CoreImpl::createOceanFoamDummy_() {
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

bool VulkanRendererCore::CoreImpl::refreshEnvTextureFromScene(Object3D& scene) {
            auto* sc = dynamic_cast<Scene*>(&scene);
            std::shared_ptr<Texture> tex;
            if (sc) {
                tex = sc->environment;
                if (!tex && sc->background.isTexture()) {
                    tex = sc->background.texture();
                }
            }
            if (!tex) {
                if (sc && sc->background.isColor()) {
                    const Color& c = sc->background.color();
                    if (envIsBgColor && envBgColor.r == c.r && envBgColor.g == c.g && envBgColor.b == c.b)
                        return false;
                    vkDeviceWaitIdle(ctx->device());
                    destroyImage2D(ctx->allocator(), ctx->device(), envImage);
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
                vkDeviceWaitIdle(ctx->device());
                destroyImage2D(ctx->allocator(), ctx->device(), envImage);
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

            vkDeviceWaitIdle(ctx->device());
            destroyImage2D(ctx->allocator(), ctx->device(), envImage);

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

}// namespace threepp
