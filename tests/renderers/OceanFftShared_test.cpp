// OceanFftShared_test: the shared-memory ocean IFFT (ifft_shared.comp,
// water::IFFT::Path::Shared) computes the same transform as the butterfly path
// it replaces (ifft_twiddle/horizontal/vertical/permute.comp), to float
// rounding, at every resolution the ocean and the wake ripples use.
//
// For each size N, three kinds of input go through both paths:
//   delta   one nonzero wavenumber; the output is a plane wave
//   wave    a plane wave; the output is a single spike of N*N
//   random  complex Gaussian, like a spectrum
// and the two outputs must agree to 1e-5 relative RMS. Both must also match a
// double-precision reference of the transform the butterfly path has always
// computed, which pins the sign the waves' travel direction depends on:
//   out(x,y) = (-1)^(x+y) * sum over (k,l) of in(k,l) * exp(+2*pi*i*(k*x + l*y)/N)
// (unnormalised; the analytic answer for delta and wave, a direct DFT for
// random up to 256). Last, one recordApplyBatch over three images with a mask
// of two must transform those two exactly as single calls do and leave the
// third untouched.
//
// Plain exit-code program like the other Vulkan tests here: 0 pass, 1 fail,
// 42 (CTest "Skipped") without a Vulkan device.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;
using cd = std::complex<double>;

namespace {

    constexpr int kSkipCode = 42;
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kSharedVsButterfly = 1e-5;// relative RMS
    constexpr double kVsReference = 1e-4;     // fp32 FFT vs double DFT

    void vkCheck(VkResult r, const char* what) {
        if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (" + std::to_string(int(r)) + ")");
    }

    struct Gpu {
        vulkan::VulkanContext& ctx;
        VkCommandPool pool = VK_NULL_HANDLE;

        explicit Gpu(vulkan::VulkanContext& c): ctx(c) {
            VkCommandPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pci.queueFamilyIndex = ctx.queueFamilies().graphics;
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            vkCheck(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool), "vkCreateCommandPool");
        }
        ~Gpu() { vkDestroyCommandPool(ctx.device(), pool, nullptr); }

        template<class F>
        void submit(F&& record) {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            vkCheck(vkAllocateCommandBuffers(ctx.device(), &ai, &cb), "vkAllocateCommandBuffers");
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkCheck(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            record(cb);
            vkCheck(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            vkCheck(vkQueueSubmit(ctx.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
            vkCheck(vkQueueWaitIdle(ctx.graphicsQueue()), "vkQueueWaitIdle");
            vkFreeCommandBuffers(ctx.device(), pool, 1, &cb);
        }
    };

    void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcA,
                 VkPipelineStageFlags dst, VkAccessFlags dstA) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = srcA;
        mb.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // An RG32F N x N storage image, left in GENERAL, and a host buffer of the
    // same size for its uploads and readbacks.
    struct Plane {
        Gpu& gpu;
        water::OceanImage img{};
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation bufAlloc = VK_NULL_HANDLE;
        void* mapped = nullptr;
        uint32_t n;

        Plane(Gpu& g, uint32_t size): gpu(g), n(size) {
            auto& ctx = gpu.ctx;
            img.format = VK_FORMAT_R32G32_SFLOAT;
            img.width = img.height = n;
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = img.format;
            ici.extent = {n, n, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            vkCheck(vmaCreateImage(ctx.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr), "vmaCreateImage");
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = img.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCheck(vkCreateImageView(ctx.device(), &vci, nullptr, &img.view), "vkCreateImageView");

            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = VkDeviceSize(n) * n * 8u;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo bai{};
            bai.usage = VMA_MEMORY_USAGE_AUTO;
            bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            vkCheck(vmaCreateBuffer(ctx.allocator(), &bci, &bai, &buf, &bufAlloc, &info), "vmaCreateBuffer");
            mapped = info.pMappedData;

            gpu.submit([&](VkCommandBuffer cb) {
                VkImageMemoryBarrier br{};
                br.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                br.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                br.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                br.image = img.image;
                br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                br.srcQueueFamilyIndex = br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &br);
            });
            img.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        ~Plane() {
            auto& ctx = gpu.ctx;
            vkDestroyImageView(ctx.device(), img.view, nullptr);
            vmaDestroyImage(ctx.allocator(), img.image, img.alloc);
            vmaDestroyBuffer(ctx.allocator(), buf, bufAlloc);
        }
        Plane(const Plane&) = delete;
        Plane& operator=(const Plane&) = delete;

        VkBufferImageCopy region() const {
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {n, n, 1};
            return r;
        }
        // Into the image, in this command buffer (the host data is staged now).
        void recordUpload(VkCommandBuffer cb, const std::vector<float>& data) {
            std::memcpy(mapped, data.data(), data.size() * sizeof(float));
            vmaFlushAllocation(gpu.ctx.allocator(), bufAlloc, 0, VK_WHOLE_SIZE);
            barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const auto r = region();
            vkCmdCopyBufferToImage(cb, buf, img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        void recordReadback(VkCommandBuffer cb) {
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const auto r = region();
            vkCmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &r);
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        }
        std::vector<float> host() const {
            vmaInvalidateAllocation(gpu.ctx.allocator(), bufAlloc, 0, VK_WHOLE_SIZE);
            std::vector<float> out(size_t(n) * n * 2);
            std::memcpy(out.data(), mapped, out.size() * sizeof(float));
            return out;
        }
    };

    // The transform of `in` through one path. The planes are reused call to
    // call: an IFFT writes a descriptor set once per image pair and caps them.
    std::vector<float> transform(Gpu& gpu, water::IFFT& fft, water::IFFT::Path path, Plane& a, Plane& scratch,
                                 const std::vector<float>& in) {
        fft.setPath(path);
        gpu.submit([&](VkCommandBuffer cb) {
            a.recordUpload(cb, in);
            fft.recordApply(cb, a.img, scratch.img);
            a.recordReadback(cb);
        });
        return a.host();
    }

    double relRms(const std::vector<float>& got, const std::vector<double>& want) {
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = double(got[i]) - want[i];
            num += d * d;
            den += want[i] * want[i];
        }
        return std::sqrt(num / std::max(den, 1e-300));
    }
    double relRms(const std::vector<float>& got, const std::vector<float>& want) {
        return relRms(got, std::vector<double>(want.begin(), want.end()));
    }

    std::vector<double> toDouble(const std::vector<cd>& v) {
        std::vector<double> out(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            out[2 * i] = v[i].real();
            out[2 * i + 1] = v[i].imag();
        }
        return out;
    }

    // The reference transform in double, separably: rows then columns.
    std::vector<cd> referenceIfft(const std::vector<float>& in, int n) {
        std::vector<cd> a(size_t(n) * n), b(size_t(n) * n);
        for (size_t i = 0; i < a.size(); ++i) a[i] = cd(in[2 * i], in[2 * i + 1]);
        std::vector<cd> w(n);
        for (int k = 0; k < n; ++k) w[k] = std::polar(1.0, 2.0 * kPi * k / n);
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                cd s = 0;
                for (int k = 0; k < n; ++k) s += a[size_t(y) * n + k] * w[(size_t(k) * x) % n];
                b[size_t(y) * n + x] = s;
            }
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y) {
                cd s = 0;
                for (int l = 0; l < n; ++l) s += b[size_t(l) * n + x] * w[(size_t(l) * y) % n];
                a[size_t(y) * n + x] = s * double(((x + y) & 1) ? -1 : 1);
            }
        return a;
    }

}// namespace

int main() {
    std::unique_ptr<Canvas> canvas;
    GLFWwindow* window = nullptr;
    std::unique_ptr<vulkan::VulkanContext> ctx;
    try {
        // The canvas brings GLFW up the way the renderer's does (the Null
        // platform where there is no display); the context gets its own hidden
        // no-API window and no ray tracing, which the FFT does not need.
        canvas = std::make_unique<Canvas>(Canvas::Parameters().title("OceanFftShared_test").size(64, 64).headless(true));
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(64, 64, "OceanFftShared_test", nullptr, nullptr);
        if (!window) throw std::runtime_error("glfwCreateWindow failed");
        ctx = std::make_unique<vulkan::VulkanContext>(window, /*enableRayTracing*/ false, /*vsync*/ false,
                                                      /*preferHeadlessSurface*/ true);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan device unavailable: %s\n", e.what());
        return kSkipCode;
    }

    int failures = 0;
    auto expect = [&](bool ok, const char* what, uint32_t n, double value, double limit) {
        std::printf("  %-34s N=%-5u %.3e  (limit %.0e)%s\n", what, n, value, limit, ok ? "" : "  FAIL");
        if (!ok) ++failures;
    };

    {
        Gpu gpu(*ctx);
        std::mt19937 rng(0x0CEA4u);
        std::normal_distribution<float> gauss(0.f, 1.f);

        for (const uint32_t n : {8u, 64u, 128u, 256u, 512u, 1024u}) {
            water::IFFT fft(*ctx, n, 8);
            if (!fft.sharedSupported()) {
                std::printf("  N=%u: the shared path is not supported on this device; skipped\n", n);
                continue;
            }
            const size_t count = size_t(n) * n;
            Plane a(gpu, n), scratch(gpu, n);

            // delta at (kx, ky): out = (-1)^(x+y) exp(+2 pi i (kx x + ky y) / n)
            const int kx = 3 % int(n), ky = 5 % int(n);
            std::vector<float> delta(count * 2, 0.f);
            delta[2 * (size_t(ky) * n + kx)] = 1.f;
            std::vector<cd> deltaWant(count);
            for (uint32_t y = 0; y < n; ++y)
                for (uint32_t x = 0; x < n; ++x)
                    deltaWant[size_t(y) * n + x] =
                            std::polar(((x + y) & 1) ? -1.0 : 1.0, 2.0 * kPi * double(kx * x + ky * y) / n);

            // plane wave exp(-2 pi i (px x + py y) / n): out = n*n at (px, py), 0 elsewhere
            // ((-1)^(px+py) times; px + py is even here).
            const int px = 2 % int(n), py = 6 % int(n);
            std::vector<float> wave(count * 2);
            std::vector<cd> waveWant(count, 0.0);
            for (uint32_t y = 0; y < n; ++y)
                for (uint32_t x = 0; x < n; ++x) {
                    const cd v = std::polar(1.0, -2.0 * kPi * double((px * x + py * y) % n) / n);
                    wave[2 * (size_t(y) * n + x)] = float(v.real());
                    wave[2 * (size_t(y) * n + x) + 1] = float(v.imag());
                }
            waveWant[size_t(py) * n + px] = double(n) * n * (((px + py) & 1) ? -1.0 : 1.0);

            std::vector<float> random(count * 2);
            for (auto& v : random) v = gauss(rng);

            struct Case {
                const char* name;
                const std::vector<float>* in;
                const std::vector<cd>* want;// null: a direct DFT where affordable
            };
            const Case cases[] = {{"delta", &delta, &deltaWant}, {"wave", &wave, &waveWant}, {"random", &random, nullptr}};
            for (const auto& c : cases) {
                const auto bfly = transform(gpu, fft, water::IFFT::Path::Butterfly, a, scratch, *c.in);
                const auto shared = transform(gpu, fft, water::IFFT::Path::Shared, a, scratch, *c.in);
                const double rr = relRms(shared, bfly);
                std::string label = std::string(c.name) + ": shared vs butterfly";
                expect(rr <= kSharedVsButterfly, label.c_str(), n, rr, kSharedVsButterfly);

                std::vector<cd> ref;
                const std::vector<cd>* want = c.want;
                if (!want && n <= 256) {
                    ref = referenceIfft(*c.in, int(n));
                    want = &ref;
                }
                if (want) {
                    const auto wd = toDouble(*want);
                    const double eb = relRms(bfly, wd), es = relRms(shared, wd);
                    label = std::string(c.name) + ": butterfly vs reference";
                    expect(eb <= kVsReference, label.c_str(), n, eb, kVsReference);
                    label = std::string(c.name) + ": shared vs reference";
                    expect(es <= kVsReference, label.c_str(), n, es, kVsReference);
                }
            }

            // One batch of three, mask 0b101: images 0 and 2 transformed as
            // single calls do, image 1 untouched.
            {
                std::vector<std::vector<float>> in(3, std::vector<float>(count * 2));
                for (auto& v : in)
                    for (auto& f : v) f = gauss(rng);
                std::vector<std::vector<float>> want(3);
                for (int i = 0; i < 3; ++i) want[i] = transform(gpu, fft, water::IFFT::Path::Butterfly, a, scratch, in[i]);

                fft.setPath(water::IFFT::Path::Shared);
                Plane p0(gpu, n), p1(gpu, n), p2(gpu, n);
                water::OceanImage* imgs[3] = {&p0.img, &p1.img, &p2.img};
                std::vector<float> got[3];
                Plane* planes[3] = {&p0, &p1, &p2};
                gpu.submit([&](VkCommandBuffer cb) {
                    for (int i = 0; i < 3; ++i) planes[i]->recordUpload(cb, in[i]);
                });
                gpu.submit([&](VkCommandBuffer cb) {
                    fft.recordApplyBatch(cb, imgs, 3, scratch.img, 0b101u);
                    for (int i = 0; i < 3; ++i) planes[i]->recordReadback(cb);
                });
                for (int i = 0; i < 3; ++i) got[i] = planes[i]->host();
                const double r0 = relRms(got[0], want[0]);
                const double r2 = relRms(got[2], want[2]);
                const bool untouched = std::memcmp(got[1].data(), in[1].data(), in[1].size() * sizeof(float)) == 0;
                expect(r0 <= kSharedVsButterfly, "batch slot 0 vs butterfly", n, r0, kSharedVsButterfly);
                expect(r2 <= kSharedVsButterfly, "batch slot 2 vs butterfly", n, r2, kSharedVsButterfly);
                expect(untouched, "batch slot 1 (masked out) unchanged", n, untouched ? 0.0 : 1.0, 0.0);
            }
        }
    }

    // recordApplyJobs: two resolutions (the cascades' 128 and 256) through
    // their own IFFTs in one rows dispatch set and one columns set.
    {
        Gpu gpu(*ctx);
        std::mt19937 rng(0x0CEA5u);
        std::normal_distribution<float> gauss(0.f, 1.f);
        water::IFFT small(*ctx, 128, 8), large(*ctx, 256, 8);
        if (small.sharedSupported() && large.sharedSupported()) {
            Plane sa(gpu, 128), sb(gpu, 128), ss(gpu, 128), la(gpu, 256), lb(gpu, 256), ls(gpu, 256);
            std::vector<float> in[4];
            Plane* planes[4] = {&sa, &sb, &la, &lb};
            for (int i = 0; i < 4; ++i) {
                in[i].resize(size_t(planes[i]->n) * planes[i]->n * 2);
                for (auto& f : in[i]) f = gauss(rng);
            }
            std::vector<float> want[4];
            for (int i = 0; i < 4; ++i) {
                water::IFFT& f = i < 2 ? small : large;
                Plane& s = i < 2 ? ss : ls;
                want[i] = transform(gpu, f, water::IFFT::Path::Butterfly, *planes[i], s, in[i]);
            }
            small.setPath(water::IFFT::Path::Shared);
            large.setPath(water::IFFT::Path::Shared);
            water::OceanImage* smallImgs[2] = {&sa.img, &sb.img};
            water::OceanImage* largeImgs[2] = {&la.img, &lb.img};
            const water::IFFT::Job jobs[2] = {{&small, smallImgs, 2, &ss.img}, {&large, largeImgs, 2, &ls.img}};
            gpu.submit([&](VkCommandBuffer cb) {
                for (int i = 0; i < 4; ++i) planes[i]->recordUpload(cb, in[i]);
                water::IFFT::recordApplyJobs(cb, jobs, 2);
                for (int i = 0; i < 4; ++i) planes[i]->recordReadback(cb);
            });
            for (int i = 0; i < 4; ++i) {
                const double r = relRms(planes[i]->host(), want[i]);
                expect(r <= kSharedVsButterfly, "jobs (128 + 256) vs butterfly", planes[i]->n, r, kSharedVsButterfly);
            }
        }
    }

    ctx.reset();
    if (window) glfwDestroyWindow(window);
    std::printf(failures ? "OceanFftShared_test: %d FAILED\n" : "OceanFftShared_test: all passed\n", failures);
    return failures ? 1 : 0;
}
