#include "VulkanContext.hpp"

#include "threepp/renderers/vulkan/ValidationReport.hpp"

// kFramesInFlight: the suppressed-present swapchain must be deep enough for the
// frame loop to keep one acquired image per in-flight slot (see createSwapchain).
#include "VulkanImplCommon.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// For the process id that makes each pipeline-cache temp file unique. <process.h>
// and <unistd.h> are the two smallest headers that declare it; windows.h is not
// dragged into this TU for one integer.
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace threepp::vulkan {

#if defined(THREEPP_WITH_DLSS)
    // Defined in DlssUpscaler.cpp (free function so this TU doesn't pull in
    // DlssUpscaler.hpp → VulkanResources.hpp, whose check() helper collides
    // with this file's local one).
    std::vector<const char*> dlssRequiredDeviceExtensions();
#endif

    namespace {

        constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

        constexpr std::array<const char*, 1> kBaseDeviceExtensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        constexpr std::array<const char*, 4> kRayTracingExtensions{
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        };

        bool hasInstanceLayer(const char* name) {
            uint32_t n = 0;
            vkEnumerateInstanceLayerProperties(&n, nullptr);
            std::vector<VkLayerProperties> layers(n);
            vkEnumerateInstanceLayerProperties(&n, layers.data());
            for (const auto& l : layers) {
                if (std::strcmp(l.layerName, name) == 0) return true;
            }
            return false;
        }

        bool hasInstanceExtension(const char* name) {
            uint32_t n = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> exts(n);
            vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
            for (const auto& e : exts) {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        }

        std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice dev) {
            uint32_t n = 0;
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> exts(n);
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data());
            return exts;
        }

        bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
            for (const auto& e : exts) {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        }

        // Validation tally behind threepp/renderers/vulkan/ValidationReport.hpp.
        //
        // Process-wide and atomic rather than VulkanContext members, for two
        // reasons: the layer reports instance-level violations before any context
        // exists (and object-lifetime ones after one is destroyed), and the
        // messenger callback is a C function pointer the loader may invoke from
        // whichever thread made the offending call.
        std::atomic<std::uint32_t> gValidationErrors{0};
        std::atomic<std::uint32_t> gValidationWarnings{0};
        std::atomic<bool>          gValidationActive{false};

        // THREEPP_VULKAN_STRICT_VALIDATION — turn a counted error into a nonzero
        // process exit, so every Vulkan-backed executable becomes a validation
        // gate with no edit of its own.
        //
        // Checked at normal termination rather than inside the callback, which
        // runs in the loader with a Vulkan call on the stack: throwing from there
        // would unwind through C frames, and aborting there would lose every
        // message after the first. Deferring costs nothing — the frames that
        // follow a violation are worth seeing, and often name the resource that
        // the first message only hinted at.
        void strictValidationExitCheck() {
            const auto errs  = gValidationErrors.load();
            const auto warns = gValidationWarnings.load();
            if (errs == 0) {
                std::cerr << "[VulkanContext] strict validation: clean ("
                          << warns << " warning(s))\n";
                return;
            }
            std::cerr << "[VulkanContext] strict validation FAILED: " << errs
                      << " validation error(s), " << warns << " warning(s). "
                      << "Exiting " << kStrictValidationExitCode
                      << " (THREEPP_VULKAN_STRICT_VALIDATION is set).\n";
            std::cerr.flush();
            std::cout.flush();
            // _Exit, not exit: this IS an atexit handler, so exit() here would
            // re-enter termination. Streams are flushed by hand just above,
            // since _Exit does not.
            std::_Exit(kStrictValidationExitCode);
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                VkDebugUtilsMessageTypeFlagsEXT type,
                const VkDebugUtilsMessengerCallbackDataEXT* data,
                void*) {
            if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                // Known third-party noise: FidelityFX FSR 3.1's own SPIR-V
                // declares its luma-history storage image Rgba8 while its own
                // runtime allocates it R16G16B16A16_SFLOAT
                // (Undefined-Value-StorageImage-FormatMismatch-ImageView, once
                // per FSR dispatch). Both sides live inside
                // amd_fidelityfx_vk.dll, so it is not fixable here and no
                // first-party code is involved — the resource name appears
                // nowhere outside that DLL, which is what makes this filter
                // safe. Drop it so validation output stays readable on FSR
                // runs. Note the layer's duplicate_message_limit still counts
                // the suppressed reports; a first-party instance of the same
                // VUID would surface among the first few reports regardless.
                if (data->pMessage && std::strstr(data->pMessage, "rw_luma_history")) {
                    return VK_FALSE;
                }
                // Counted AFTER the third-party filter above, so suppressed noise
                // cannot fail a gate, and BEFORE the print, so a message that is
                // counted is always also visible.
                //
                // An ERROR counts only when it carries the VALIDATION message
                // type — i.e. it is a spec violation. The messenger also hears
                // GENERAL-type errors, which describe the ENVIRONMENT rather
                // than this code: the instance-creation messenger surfaced the
                // loader failing to open a stale third-party layer manifest
                // (Rockstar's SocialClubVulkanLayer.json, registered but
                // deleted) the first time it ran. Gating on those would make
                // the verdict depend on which machine ran it. They are tallied
                // as warnings, so they stay visible without being fatal.
                const bool isSpecViolation =
                        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 &&
                        (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0;
                if (isSpecViolation) {
                    gValidationErrors.fetch_add(1, std::memory_order_relaxed);
                } else {
                    gValidationWarnings.fetch_add(1, std::memory_order_relaxed);
                }
                std::cerr << "[Vulkan] " << (isSpecViolation ? "ERROR: " : "WARNING: ")
                          << data->pMessage << "\n";
            }
            return VK_FALSE;
        }

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("[VulkanContext] ") + what + " failed: " + std::to_string(r));
            }
        }

        // One definition of what the messenger listens to, used twice: chained
        // into VkInstanceCreateInfo::pNext (covering vkCreateInstance /
        // vkDestroyInstance, which run before/after the persistent messenger
        // exists) and passed to vkCreateDebugUtilsMessengerEXT for everything in
        // between. Keeping them the same struct matters — a severity added
        // in one place but not the other would make instance-time coverage
        // silently diverge from runtime coverage.
        VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo() {
            VkDebugUtilsMessengerCreateInfoEXT ci{};
            ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            ci.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            ci.messageType =
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            ci.pfnUserCallback = debugCallback;
            return ci;
        }

    }// namespace

    VulkanContext::VulkanContext(GLFWwindow* window, bool enableRayTracing, bool vsync,
                                 bool preferHeadlessSurface)
        : window_(window), vsync_(vsync) {

        // Surface mode. A headless canvas prefers VK_EXT_headless_surface: the
        // swapchain and frame loop run unmodified, but the surface has no
        // window behind it, so no display server is needed — the requirement
        // that actually matters on cloud GPU instances, where a window surface
        // either can't be created at all (no X server) or lands on an X server
        // the GPU cannot present to (Xvfb). Supported by NVIDIA, Mesa and the
        // Khronos loader; when an ICD lacks it, fall back to the hidden-window
        // surface where a window system exists (today's headless behaviour),
        // and fail with a direct message where none does (GLFW Null platform).
        if (preferHeadlessSurface) {
#ifdef GLFW_PLATFORM// 3.4+
            const bool haveWindowSystem = glfwGetPlatform() != GLFW_PLATFORM_NULL;
#else
            // GLFW before 3.4 has no Null platform, so glfwInit having succeeded
            // means a window system is there (see initGLfw in Canvas.cpp).
            const bool haveWindowSystem = true;
#endif
            if (hasInstanceExtension(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) {
                headlessSurface_ = true;
            } else if (haveWindowSystem) {
                std::cerr << "[VulkanContext] VK_EXT_headless_surface not available; "
                             "headless canvas falls back to a hidden-window surface.\n";
            } else {
                throw std::runtime_error(
                        "[VulkanContext] headless mode needs VK_EXT_headless_surface, "
                        "which this Vulkan driver does not expose, and no window system "
                        "is available to create a window surface instead");
            }
        }

        // ── THREEPP_VULKAN_SUPPRESS_PRESENT: opt-in, and it stays opt-in ─────
        // A headless canvas displays nothing, so it has nothing to present: its
        // swapchain is only an image ring for the frame loop and the capture
        // paths. Presenting anyway is NOT free. On Windows/NVIDIA,
        // vkQueuePresentKHR to a hidden (never-composited) window performs a
        // HOST-side wait for the presented image's rendering to complete, so the
        // CPU cannot get a frame ahead of the GPU. Measured on an RTX 4070 with
        // the granular-conveyor bench: frame.K3_present tracks gpuTotalMs 1:1
        // (5.5 ms at 2k grains, 10.6 ms at 78.4k), is unaffected by swapchain
        // image count (3 vs 8), moves wholesale into an explicit pre-present
        // fence wait when one is added, and vanishes (0.22 ms) the moment the
        // same window is shown. MAILBOX and IMMEDIATE behave identically; FIFO
        // does not block but pins the loop to the refresh rate, which is exactly
        // what vsync=off exists to escape. So no present mode is a fix — the
        // only fix is not presenting, which is what this flag does.
        //
        // It is OFF by default because removing the stall MEASURED SLOWER on
        // every population of that bench: 25k 68.0→62.9 fps, 50k 49.8→46.1, 78.4k
        // 34.2→30.1 (N=5 interleaved). The stall was doing unintended work — it
        // kept the app's PhysX GPU (CUDA) step from overlapping the Vulkan
        // command buffer, and when the CPU is freed to run ahead the two
        // time-slice on the same device instead: the whole-command-buffer GPU
        // span inflates 10.8→17.6 ms at 78.4k for identical graphics work (the
        // same effect as feedback_vulkan_async_same_family_timeslice).
        //
        // A frame loop with NO co-resident GPU compute has nothing to time-slice
        // against, and there it is the win the stall's size predicts:
        // VulkanMultiView_test --bench (headless, vsync off, pure renderer, N=5
        // interleaved) goes 1.419→1.144 ms/frame with no secondary views (-19%)
        // and 4.458→4.000 with three (-10%), with no overlap between the two
        // groups' samples. So this is a real capability with a real cost, and
        // which it is depends on the caller's GPU tenancy — hence a flag rather
        // than a default either way.
        presentSuppressed_ = false;
        if (const char* sp = std::getenv("THREEPP_VULKAN_SUPPRESS_PRESENT");
            sp && *sp == '1' && preferHeadlessSurface) {
            presentSuppressed_ = true;
            std::cout << "[VulkanContext] presents suppressed (headless canvas): the frame "
                         "loop pins one swapchain image per in-flight slot\n";
        }

        // Validation defaults on in debug builds and off elsewhere, but
        // THREEPP_VULKAN_VALIDATION overrides either way ("0" forces off, any
        // other value forces on). Without the override, RelWithDebInfo — the
        // build everyone actually runs — could never report a VUID, so spec
        // violations went unnoticed there. Forcing the layer on at the
        // loader level (VK_LOADER_LAYERS_ENABLE=*validation*) is not a
        // substitute: the app then still skips VK_EXT_DEBUG_UTILS, which costs
        // the messenger callback and every setObjectName() label, leaving bare
        // handles in the layer's output.
        bool wantValidation =
#ifndef NDEBUG
                true;
#else
                false;
#endif
        if (const char* env = std::getenv("THREEPP_VULKAN_VALIDATION"); env && *env) {
            wantValidation = *env != '0';
        }
        const bool enableValidation = wantValidation && hasInstanceLayer(kValidationLayer);
        if (wantValidation && !enableValidation) {
            std::cerr << "[VulkanContext] " << kValidationLayer
                      << " not found; running without validation.\n";
        }
        rayTracingEnabled_ = enableRayTracing;

        createInstance(enableValidation);
        if (enableValidation) {
            createDebugMessenger();
            gValidationActive.store(true);

            // Strict mode is armed here rather than at the env read above so it
            // can never be armed without a messenger to feed it: a process that
            // asked for strict validation but got no layer must not exit 0 as if
            // it had been checked. validationActive() is what tells a caller the
            // difference, and the arming message says which of the two happened.
            if (const char* env = std::getenv("THREEPP_VULKAN_STRICT_VALIDATION"); env && *env && *env != '0') {
                static std::once_flag once;
                std::call_once(once, [] { std::atexit(strictValidationExitCheck); });
                std::cerr << "[VulkanContext] strict validation armed: any validation error "
                             "exits "
                          << kStrictValidationExitCode << " at termination.\n";
            }
        } else if (const char* env = std::getenv("THREEPP_VULKAN_STRICT_VALIDATION"); env && *env && *env != '0') {
            std::cerr << "[VulkanContext] THREEPP_VULKAN_STRICT_VALIDATION is set but the "
                         "validation layer is not active — nothing is being checked.\n";
        }
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createPipelineCache();
        createAllocator();
        createSwapchain();
        createSwapchainImageViews();
    }

    VulkanContext::~VulkanContext() {
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);

        destroySwapchainResources();

        if (allocator_ != VK_NULL_HANDLE) vmaDestroyAllocator(allocator_);
        if (pipelineCache_ != VK_NULL_HANDLE) {
            savePipelineCache();
            vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        }
        if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
        if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_, surface_, nullptr);

        if (debugMessenger_ != VK_NULL_HANDLE) {
            auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
                    instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (fn) fn(instance_, debugMessenger_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

    namespace {

        std::filesystem::path pipelineCacheDir() {
            std::error_code ec;
            auto dir = std::filesystem::temp_directory_path(ec);
            if (ec) return {};
            return dir;
        }

        // The blob is keyed to the device that produced it. The driver already
        // refuses a foreign one, but a single shared file name means the two
        // devices of a dual-GPU box overwrite each other on every launch and
        // both then report COLD forever. pipelineCacheUUID is the driver's own
        // identity for "these binaries are still valid" — a driver update
        // changes it — so it is exactly the right key, and blobs for other
        // devices simply sit alongside instead of fighting over one name.
        std::string pipelineCacheKey(const uint8_t uuid[VK_UUID_SIZE]) {
            static constexpr char kHex[] = "0123456789abcdef";
            std::string s;
            s.reserve(VK_UUID_SIZE * 2);
            for (size_t i = 0; i < VK_UUID_SIZE; ++i) {
                s.push_back(kHex[(uuid[i] >> 4) & 0xF]);
                s.push_back(kHex[uuid[i] & 0xF]);
            }
            return s;
        }

        std::filesystem::path pipelineCachePath(const std::string& key) {
            const auto dir = pipelineCacheDir();
            if (dir.empty()) return {};
            return dir / ("threepp_pipeline_cache." + key + ".bin");
        }

        // The unkeyed name older builds wrote. Read as a fallback so an
        // existing warm blob survives the rename, and deleted only once this
        // device has a keyed file of its own — i.e. after the migration has
        // actually landed, never on the run that still depends on it.
        std::filesystem::path legacyPipelineCachePath() {
            const auto dir = pipelineCacheDir();
            if (dir.empty()) return {};
            return dir / "threepp_pipeline_cache.bin";
        }

        // Process id, decimal, for the temp file name. <process.h> and
        // <unistd.h> spell it differently and nothing else in this TU needs it.
        std::string currentProcessIdString() {
#ifdef _WIN32
            return std::to_string(_getpid());
#else
            return std::to_string(static_cast<long long>(getpid()));
#endif
        }

        std::vector<char> readPipelineCacheFile(const std::filesystem::path& path) {
            std::vector<char> bytes;
            if (path.empty()) return bytes;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return bytes;
            const std::streamsize sz = f.tellg();
            if (sz <= 0) return bytes;
            bytes.resize(static_cast<size_t>(sz));
            f.seekg(0);
            if (!f.read(bytes.data(), sz)) bytes.clear();
            return bytes;
        }

        // Collect temps left behind by a save that was killed between the write
        // and the rename; nothing else ever removes them and each is the full
        // blob size. Every filesystem call here takes the error_code overload
        // and every failure is ignored on purpose: on Windows a temp another
        // process still has open cannot be deleted, which is precisely the file
        // that must not be touched. On POSIX the unlink would succeed, but the
        // writer's descriptor stays valid and its rename then fails, so the
        // worst case is one lost save and never a damaged blob.
        void sweepPipelineCacheTemps() {
            const auto dir = pipelineCacheDir();
            if (dir.empty()) return;
            std::error_code ec;
            std::filesystem::directory_iterator it(dir, ec);
            const std::filesystem::directory_iterator end;
            while (!ec && it != end) {
                const auto p = it->path();
                const auto name = p.filename().string();
                if (name.rfind("threepp_pipeline_cache.", 0) == 0 && p.extension() == ".tmp") {
                    std::error_code rmEc;
                    std::filesystem::remove(p, rmEc);
                }
                it.increment(ec);
            }
        }

    }// namespace

    void VulkanContext::createPipelineCache() {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        const auto path = pipelineCachePath(pipelineCacheKey(props.pipelineCacheUUID));

        sweepPipelineCacheTemps();

        // Read the previous run's blob, if any, preferring this device's own.
        std::vector<char> initial = readPipelineCacheFile(path);
        auto source = path;
        const auto legacy = legacyPipelineCachePath();
        if (initial.empty()) {
            initial = readPipelineCacheFile(legacy);
            if (!initial.empty()) source = legacy;
        } else if (!legacy.empty()) {
            std::error_code ec;
            std::filesystem::remove(legacy, ec);
        }

        // Validate the cache header against THIS device — vendor/device/UUID
        // must match or the blob is from another GPU/driver and must be dropped
        // (some drivers reject foreign data outright). Header layout
        // (VkPipelineCacheHeaderVersionOne): u32 size, u32 version, u32 vendorID,
        // u32 deviceID, u8 uuid[VK_UUID_SIZE] — 32 bytes total. Note that the
        // leading u32 is the length of the HEADER, not of the blob, so no amount
        // of header checking can spot a truncated file; that is what the atomic
        // rename in savePipelineCache is for.
        bool usable = false;
        if (initial.size() >= 32) {
            uint32_t hdrVersion = 0, vendorID = 0, deviceID = 0;
            std::memcpy(&hdrVersion, initial.data() + 4, 4);
            std::memcpy(&vendorID, initial.data() + 8, 4);
            std::memcpy(&deviceID, initial.data() + 12, 4);
            usable = hdrVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                     vendorID == props.vendorID && deviceID == props.deviceID &&
                     std::memcmp(initial.data() + 16, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        }

        VkPipelineCacheCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (usable) {
            ci.initialDataSize = initial.size();
            ci.pInitialData = initial.data();
        }
        if (vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_) != VK_SUCCESS) {
            pipelineCache_ = VK_NULL_HANDLE;// non-fatal: pipelines just compile cold
        }
        std::cout << "[VulkanContext] pipeline cache: "
                  << (usable ? "WARM - loaded " : "COLD - ignoring ")
                  << initial.size() << " bytes from " << source.string()
                  << (usable ? " (pipelines reused)" : " (recompiling all pipelines)") << std::endl;
    }

    void VulkanContext::savePipelineCache() {
        if (pipelineCache_ == VK_NULL_HANDLE) return;
        size_t sz = 0;
        if (vkGetPipelineCacheData(device_, pipelineCache_, &sz, nullptr) != VK_SUCCESS || sz == 0) return;
        std::vector<char> data(sz);
        if (vkGetPipelineCacheData(device_, pipelineCache_, &sz, data.data()) != VK_SUCCESS) return;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        const auto path = pipelineCachePath(pipelineCacheKey(props.pipelineCacheUUID));
        if (path.empty()) return;

        // Write a temp beside the blob and rename it into place instead of
        // truncating the blob itself. The blob runs to tens of megabytes, so the
        // write is wide enough that a kill lands inside it regularly, and the
        // loader cannot tell a fragment from a whole file (see the header note
        // above) — it would hand the fragment to the driver as initialData. The
        // rename is atomic, so a reader sees either the whole old blob or the
        // whole new one and never a splice of the two. The temp carries this
        // process's pid so two threepp processes saving at once write to
        // separate files rather than interleaving into one.
        const auto tmp = path.parent_path() /
                         (path.stem().string() + "." + currentProcessIdString() + ".tmp");
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return;
            f.write(data.data(), static_cast<std::streamsize>(sz));
            // Closed explicitly rather than by the destructor, because the
            // status of the final flush has to be read before deciding whether
            // to rename, and because Windows refuses both the rename and the
            // remove below while this process still holds the file open.
            f.close();
            if (!f) {
                // Out of space, most likely. Leave the previous blob alone and
                // take the half-written temp with us.
                std::error_code rmEc;
                std::filesystem::remove(tmp, rmEc);
                return;
            }
        }

        // error_code overload, never the throwing one — this also runs from
        // ~VulkanContext, where an escaping exception terminates the process. A
        // rename can legitimately fail (on Windows, over a blob another process
        // has open), and the right answer then is to keep the old blob and drop
        // the temp: a stale-but-whole cache costs a partial recompile, a
        // half-written one costs correctness.
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::error_code rmEc;
            std::filesystem::remove(tmp, rmEc);
            return;
        }
        std::cout << "[VulkanContext] pipeline cache: saved " << sz
                  << " bytes to " << path.string() << std::endl;
    }

    void VulkanContext::createInstance(bool enableValidation) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "threepp";
        app.pEngineName = "threepp";
        app.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char*> extensions;
        if (headlessSurface_) {
            // No window system involved: name the surface machinery directly
            // instead of asking GLFW, whose Null platform cannot create
            // surfaces and reports no extensions at all.
            extensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                          VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
        } else {
            uint32_t glfwExtCount = 0;
            const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
            extensions.assign(glfwExts, glfwExts + glfwExtCount);
        }
        if (enableValidation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // A portability driver — MoltenVK, or any other non-conformant
        // implementation — is hidden by the Khronos loader unless the
        // application opts in. Without the opt-in vkCreateInstance fails with
        // VK_ERROR_INCOMPATIBLE_DRIVER outright, or (loader version depending)
        // succeeds and then enumerates zero physical devices. Queried rather
        // than keyed on __APPLE__ so it holds for any portability
        // implementation on any platform.
        //
        // This is spec compliance, not macOS support: the renderer still needs
        // KHR ray tracing, which MoltenVK does not provide (see
        // pickPhysicalDevice). It only buys a clearer failure further in.
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        const bool portability =
                hasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        if (portability) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
#endif

        std::vector<const char*> layers;
        if (enableValidation) layers.push_back(kValidationLayer);

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (portability) ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ci.ppEnabledLayerNames = layers.data();

        // The persistent messenger (createDebugMessenger below) only exists once
        // the instance does, so vkCreateInstance / vkDestroyInstance themselves
        // would be the one stretch validation couldn't report on. Chaining the
        // same create-info through pNext is the spec's remedy: the layer runs a
        // messenger scoped to exactly those two calls.
        const auto dbgCi = debugMessengerCreateInfo();
        if (enableValidation) ci.pNext = &dbgCi;

        check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
    }

    void VulkanContext::createDebugMessenger() {
        const auto ci = debugMessengerCreateInfo();
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
                instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) check(fn(instance_, &ci, nullptr, &debugMessenger_), "vkCreateDebugUtilsMessengerEXT");
    }

    void VulkanContext::createSurface() {
        if (headlessSurface_) {
            // Loaded via vkGetInstanceProcAddr — extension entry points are
            // not exported by every loader.
            auto fn = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
                    vkGetInstanceProcAddr(instance_, "vkCreateHeadlessSurfaceEXT"));
            if (!fn) {
                throw std::runtime_error(
                        "[VulkanContext] vkCreateHeadlessSurfaceEXT not found despite "
                        "VK_EXT_headless_surface being advertised");
            }
            VkHeadlessSurfaceCreateInfoEXT ci{};
            ci.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
            check(fn(instance_, &ci, nullptr, &surface_), "vkCreateHeadlessSurfaceEXT");
            std::cerr << "[VulkanContext] surface: headless (VK_EXT_headless_surface)\n";
            return;
        }
        check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
              "glfwCreateWindowSurface");
    }

    void VulkanContext::pickPhysicalDevice() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (n == 0) throw std::runtime_error("[VulkanContext] no Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance_, &n, devs.data());

        // Prefer discrete GPU with required extensions.
        auto deviceScore = [&](VkPhysicalDevice d) -> int {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);

            const auto exts = deviceExtensions(d);
            for (const auto* base : kBaseDeviceExtensions) {
                if (!hasExtension(exts, base)) return -1;
            }
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
            if (rayTracingEnabled_) {
                bool allRT = true;
                for (const auto* rt : kRayTracingExtensions) {
                    if (!hasExtension(exts, rt)) { allRT = false; break; }
                }
                if (!allRT) return -1;
                score += 100;
            }
            return score;
        };

        int bestScore = -1;
        for (auto d : devs) {
            int s = deviceScore(d);
            if (s > bestScore) {
                bestScore = s;
                physicalDevice_ = d;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            if (!rayTracingEnabled_) {
                throw std::runtime_error("[VulkanContext] no GPU with swapchain support found");
            }
            // Name the requirement rather than just the symptom. This renderer
            // shades through ray query — deferred_shade.comp traces for
            // shadows, reflections and GI, and probe/froxel/particle passes do
            // the same — so a device without KHR ray tracing cannot run it at
            // all. There is no raster fallback to degrade to.
            std::string msg =
                    "[VulkanContext] no GPU with ray-tracing extensions found. The Vulkan "
                    "backend requires VK_KHR_ray_tracing_pipeline, VK_KHR_acceleration_structure, "
                    "VK_KHR_deferred_host_operations and VK_KHR_buffer_device_address; its shading "
                    "path traces rays for shadows, reflections and GI and has no raster fallback.";
            msg += "\n[VulkanContext] devices seen:";
            for (auto d : devs) {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(d, &props);
                msg += "\n  - ";
                msg += props.deviceName;
            }
#ifdef __APPLE__
            msg += "\n[VulkanContext] MoltenVK implements none of the KHR ray-tracing extensions, "
                   "so the Vulkan backend cannot run on macOS. Use GLRenderer there.";
#endif
            throw std::runtime_error(msg);
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        std::cerr << "[VulkanContext] picked GPU: " << props.deviceName << "\n";

        if (rayTracingEnabled_) {
            rtPipelineProperties_.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &rtPipelineProperties_;
            vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

            // Probe for VK_KHR_ray_query — lets compute shaders trace inline
            // rays (rayQueryEXT). Used by the raster-first deferred shading pass
            // for hard shadow rays. Optional: ReferencePT doesn't need it, so a
            // device that has the RT pipeline but not ray query still runs (the
            // renderer falls RasterFirst back to ReferencePT). All current RT
            // hardware exposes both.
            const auto pickedExtsRq = deviceExtensions(physicalDevice_);
            if (hasExtension(pickedExtsRq, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{};
                rqFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
                VkPhysicalDeviceFeatures2 feat2{};
                feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                feat2.pNext = &rqFeat;
                vkGetPhysicalDeviceFeatures2(physicalDevice_, &feat2);
                rayQuerySupported_ = rqFeat.rayQuery == VK_TRUE;
            }
            std::cerr << "[VulkanContext] ray query (VK_KHR_ray_query): "
                      << (rayQuerySupported_ ? "enabled" : "unavailable") << "\n";
        }

        // Probe for exportable external memory (the platform handle extension;
        // VK_KHR_external_memory itself is core since 1.1). Lets device-local
        // buffers be exported as OS handles and imported by CUDA — the zero-copy
        // path for PhysX soft-body tet positions. Independent of ray tracing.
        {
            // Name spelled as a literal: the macro lives in the platform header
            // (vulkan_win32.h), which would drag windows.h into this TU.
            const auto exts = deviceExtensions(physicalDevice_);
#ifdef _WIN32
            externalMemorySupported_ = hasExtension(exts, "VK_KHR_external_memory_win32");
#else
            externalMemorySupported_ = hasExtension(exts, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
            std::cerr << "[VulkanContext] external memory export: "
                      << (externalMemorySupported_ ? "enabled" : "unavailable") << "\n";

            // Diagnostic only, and opt-in: capturing statistics changes how
            // pipelines are compiled, so it must never be on by default.
            if (const char* env = std::getenv("THREEPP_VULKAN_PIPELINE_STATS"); env && *env && *env != '0') {
                pipelineStatsEnabled_ = hasExtension(exts, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                std::cerr << "[VulkanContext] pipeline statistics: "
                          << (pipelineStatsEnabled_ ? "ENABLED (diagnostic)"
                                                    : "requested but VK_KHR_pipeline_executable_properties unavailable")
                          << "\n";
            }
        }

        // Find queue families.
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qn, qprops.data());
        for (uint32_t i = 0; i < qn; ++i) {
            const auto& q = qprops[i];
            if ((q.queueFlags & VK_QUEUE_GRAPHICS_BIT) && queueFamilies_.graphics == UINT32_MAX) {
                queueFamilies_.graphics = i;
            }
            if ((q.queueFlags & VK_QUEUE_COMPUTE_BIT) && queueFamilies_.compute == UINT32_MAX) {
                queueFamilies_.compute = i;
            }
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, surface_, &presentSupport);
            if (presentSupport && queueFamilies_.present == UINT32_MAX) {
                queueFamilies_.present = i;
            }
        }
        // A headless surface has no real presentation engine; ICDs generally
        // report present support for it on the graphics family, but the
        // extension leaves that implementation-defined. Presenting there is a
        // no-op regardless, so route it through the graphics queue when the
        // driver declined to name a family.
        if (headlessSurface_ && queueFamilies_.present == UINT32_MAX) {
            queueFamilies_.present = queueFamilies_.graphics;
        }
        if (queueFamilies_.graphics == UINT32_MAX || queueFamilies_.present == UINT32_MAX) {
            throw std::runtime_error("[VulkanContext] required queue families not present on GPU");
        }
    }

    void VulkanContext::createLogicalDevice() {
        const float prio = 1.0f;
        std::set<uint32_t> uniqueFams{queueFamilies_.graphics,
                                     queueFamilies_.present,
                                     queueFamilies_.compute};
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t fam : uniqueFams) {
            if (fam == UINT32_MAX) continue;
            VkDeviceQueueCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            ci.queueFamilyIndex = fam;
            ci.queueCount = 1;
            ci.pQueuePriorities = &prio;
            qcis.push_back(ci);
        }

        std::vector<const char*> extensions(kBaseDeviceExtensions.begin(), kBaseDeviceExtensions.end());
        if (rayTracingEnabled_) {
            extensions.insert(extensions.end(), kRayTracingExtensions.begin(), kRayTracingExtensions.end());
            if (rayQuerySupported_) {
                extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            }
        }
        // Required by the spec (VUID-VkDeviceCreateInfo-pProperties-04451)
        // whenever the physical device advertises it: a portability
        // implementation must refuse vkCreateDevice without it. Spelled out as
        // a literal because VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME lives in
        // vulkan_beta.h, behind VK_ENABLE_BETA_EXTENSIONS.
        if (hasExtension(deviceExtensions(physicalDevice_), "VK_KHR_portability_subset")) {
            extensions.push_back("VK_KHR_portability_subset");
        }
        if (pipelineStatsEnabled_) {
            extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        }
        if (externalMemorySupported_) {
#ifdef _WIN32
            extensions.push_back("VK_KHR_external_memory_win32");// macro lives in vulkan_win32.h
#else
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }

#if defined(THREEPP_WITH_FSR)
        // AMD FidelityFX FSR 3.1's VK backend loads vkGetBufferMemoryRequirements2KHR
        // (and the image variant) by their KHR-SUFFIXED names via vkGetDeviceProcAddr.
        // On a core-1.1+ device those return NULL unless VK_KHR_get_memory_requirements2
        // is explicitly enabled, and the backend then calls the null pointer during
        // resource creation → an access-violation crash (execution at 0x0) deep inside
        // ffxCreateContext (FidelityFX-SDK issue #73). Enable it (+ dedicated
        // allocation, which the backend also probes) when the device advertises it —
        // Cauldron, the FFX sample framework, enables the same set.
        {
            const auto avail = deviceExtensions(physicalDevice_);
            auto has = [&](const char* n) {
                for (const auto& e : avail)
                    if (std::strcmp(e.extensionName, n) == 0) return true;
                return false;
            };
            if (has(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME))
                extensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
            if (has(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME))
                extensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
        }
#endif

#if defined(THREEPP_WITH_DLSS)
        // NVIDIA NGX/DLSS device extensions (VK_NVX_binary_import,
        // VK_NVX_image_view_handle, ...), queried from the NGX static lib —
        // no NGX init needed. Enable only the subset the device actually
        // supports and isn't already enabling: on non-NVIDIA hardware the NVX
        // extensions are absent, DLSS feature creation later fails cleanly, and
        // the FSR/TAA fallback runs. The instance extensions NGX asks for are
        // core in Vulkan 1.3 (this renderer's instance version).
        {
            const auto avail = deviceExtensions(physicalDevice_);
            auto supported = [&](const char* n) {
                for (const auto& e : avail)
                    if (std::strcmp(e.extensionName, n) == 0) return true;
                return false;
            };
            auto alreadyEnabled = [&](const char* n) {
                for (const char* e : extensions)
                    if (std::strcmp(e, n) == 0) return true;
                return false;
            };
            for (const char* n : dlssRequiredDeviceExtensions()) {
                // NGX still asks for VK_EXT_buffer_device_address, the
                // pre-promotion alias of what this device gets from
                // VkPhysicalDeviceVulkan12Features::bufferDeviceAddress below
                // (plus VK_KHR_buffer_device_address on the ray-tracing path).
                // Enabling the EXT alias alongside either is illegal —
                // VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-03328 and
                // VUID-VkDeviceCreateInfo-pNext-04748 — and buys nothing: the
                // core 1.2 feature is what NGX actually consumes.
                if (std::strcmp(n, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) continue;
                if (supported(n) && !alreadyEnabled(n)) extensions.push_back(n);
            }
        }
#endif

        // Required core 1.2 / 1.3 features (BDA, dynamic rendering, sync2).
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        // Shaders are compiled with glslangValidator --target-env vulkan1.3,
        // which lowers `discard` (e.g. overlay_point.frag's round-point cutout)
        // to OpDemoteToHelperInvocation rather than OpKill. That op needs the
        // DemoteToHelperInvocation SPIR-V capability enabled at device creation,
        // else vkCreateShaderModule warns (VUID-VkShaderModuleCreateInfo-pCode-08740).
        // The feature is core in Vulkan 1.3 (apiVersion is VK_API_VERSION_1_3).
        f13.shaderDemoteToHelperInvocation = VK_TRUE;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.descriptorIndexing = VK_TRUE;
        f12.runtimeDescriptorArray = VK_TRUE;
        // nonuniformEXT() on the bindless albedoMaps[] indices (gbuffer.frag,
        // deferred_shade.comp, closest_hit*.rahit/rchit). Per-hit/per-instance
        // material indices are wave-DIVERGENT; without this feature + qualifier
        // the descriptor load is spec-UB (may be hoisted wave-uniform → wrong
        // texture per wave, TAA-jitter-dependent → flicker). NOT implied by
        // descriptorIndexing above — it is a separate fine-grained feature bit.
        f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        f12.scalarBlockLayout = VK_TRUE;// closest-hit reads GeometryDesc[] / normals via scalar layout
        f12.pNext = &f13;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR fAS{};
        fAS.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        fAS.accelerationStructure = VK_TRUE;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR fRT{};
        fRT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        fRT.rayTracingPipeline = VK_TRUE;

        // Inline ray query (rayQueryEXT in compute) for the raster-first
        // deferred shadow pass. Chained at the tail (after f13) when supported.
        VkPhysicalDeviceRayQueryFeaturesKHR fRQ{};
        fRQ.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        fRQ.rayQuery = VK_TRUE;

        if (rayTracingEnabled_) {
            fRT.pNext = &f12;
            fAS.pNext = &fRT;
            if (rayQuerySupported_) {
                fRQ.pNext = f13.pNext;// preserve any existing tail (currently null)
                f13.pNext = &fRQ;
            }
        }

        // Diagnostic (THREEPP_VULKAN_PIPELINE_STATS): chained at the head of
        // f13's tail so it coexists with the ray-query feature above.
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fPES{};
        fPES.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        fPES.pipelineExecutableInfo = VK_TRUE;
        if (pipelineStatsEnabled_) {
            fPES.pNext = f13.pNext;
            f13.pNext  = &fPES;
        }

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.features.shaderInt64 = VK_TRUE;// uint64_t buffer-reference addresses in rchit
        features2.features.samplerAnisotropy = VK_TRUE;// material-tex sampler enables aniso filtering
        // Hybrid gbuf indirect-drawing path encodes the per-draw DrawInfo
        // index into VkDrawIndirectCommand::firstInstance; the VS reads
        // it back as gl_InstanceIndex. Without this feature, firstInstance
        // must be 0 and that encoding breaks.
        features2.features.drawIndirectFirstInstance = VK_TRUE;
        // gl_DrawIDARB / gl_BaseInstanceARB / gl_BaseVertexARB — not strictly
        // required by the current gbuf shader (it uses gl_InstanceIndex), but
        // pulling the feature in keeps the door open for future per-draw
        // pulls and is universally supported on RT-capable hardware.
        features2.features.multiDrawIndirect = VK_TRUE;
        // VK_POLYGON_MODE_LINE for the wireframe overlay pipeline.
        features2.features.fillModeNonSolid = VK_TRUE;
        // Storage-image writes without a format qualifier — lets shaders write
        // to BGRA8 swap targets without declaring rgba8 (which would mismatch
        // the underlying VkImageView format and produce a validation warning).
        features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        // ...and READS without a format qualifier, for the one pass that loads
        // the BGRA8 swapchain back in (event_shade.comp's Final source). Not
        // mandatory like the write side — the renderer runs without it and
        // the event camera falls back to its G-buffer proxy — so query first
        // and enable only when the device has it (every desktop GPU and
        // lavapipe do).
        {
            VkPhysicalDeviceFeatures sup{};
            vkGetPhysicalDeviceFeatures(physicalDevice_, &sup);
            storageImageReadWithoutFormat_ = sup.shaderStorageImageReadWithoutFormat == VK_TRUE;
            features2.features.shaderStorageImageReadWithoutFormat =
                    storageImageReadWithoutFormat_ ? VK_TRUE : VK_FALSE;
        }
        // Per-attachment color-blend on the gbuffer MRT: the decal pipeline
        // blends only the albedo attachment (SRC_ALPHA) while the other targets
        // stay non-blended. Without independentBlend, all
        // VkPipelineColorBlendAttachmentState entries must be identical
        // (VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605). Core
        // 1.0, universally supported on RT-capable hardware.
        features2.features.independentBlend = VK_TRUE;
        if (rayTracingEnabled_) {
            features2.pNext = &fAS;
        } else {
            features2.pNext = &f12;
        }

#if defined(THREEPP_WITH_FSR)
        // ── AMD FidelityFX FSR 3.1 device features ──────────────────────────
        // The FSR VK backend selects FP16 shader permutations when the device
        // advertises 16-bit float support, and its shaders use subgroup-size
        // control. Those SPIR-V capabilities must be ENABLED on the logical
        // device, or the backend creates pipelines using capabilities the device
        // never enabled and crashes (a null-path deref) inside ffxCreateContext.
        // All of these are core 1.1/1.2/1.3 feature bits — no extra device
        // extensions — and enabling them is inert for threepp's own shaders
        // (which don't use them). Query support first so vkCreateDevice never
        // fails on a GPU that lacks a bit; enable only the supported subset.
        VkPhysicalDeviceVulkan13Features supF13{};
        supF13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceVulkan12Features supF12{};
        supF12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        supF12.pNext = &supF13;
        VkPhysicalDeviceVulkan11Features supF11{};
        supF11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        supF11.pNext = &supF12;
        VkPhysicalDeviceFeatures2 sup2{};
        sup2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        sup2.pNext = &supF11;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &sup2);

        // 16-bit storage (Vulkan 1.1) — threepp has no VkPhysicalDeviceVulkan11Features
        // yet, so add one and prepend it to the chain (doesn't disturb the tail).
        VkPhysicalDeviceVulkan11Features fsrF11{};
        fsrF11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        fsrF11.storageBuffer16BitAccess           = supF11.storageBuffer16BitAccess;
        fsrF11.uniformAndStorageBuffer16BitAccess = supF11.uniformAndStorageBuffer16BitAccess;
        f12.shaderFloat16              = supF12.shaderFloat16;              // FP16 shader path
        f13.subgroupSizeControl        = supF13.subgroupSizeControl;        // required subgroup size
        f13.computeFullSubgroups       = supF13.computeFullSubgroups;
        features2.features.shaderInt16 = sup2.features.shaderInt16;
        fsrF11.pNext    = features2.pNext;
        features2.pNext = &fsrF11;
#endif

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &features2;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();

        check(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_), "vkCreateDevice");

        vkGetDeviceQueue(device_, queueFamilies_.graphics, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueFamilies_.present, 0, &presentQueue_);
        if (queueFamilies_.compute != UINT32_MAX) {
            vkGetDeviceQueue(device_, queueFamilies_.compute, 0, &computeQueue_);
        } else {
            computeQueue_ = graphicsQueue_;
        }

        // EXT_debug_utils object-name function — loaded once for the whole
        // app, used by setObjectName helpers. Null when the extension isn't
        // enabled (validation off); the helpers detect that and no-op.
        setObjectNameFn_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));

        if (rayTracingEnabled_) {
            auto load = [this](const char* name, void** dst) {
                *dst = reinterpret_cast<void*>(vkGetDeviceProcAddr(device_, name));
                if (!*dst) {
                    throw std::runtime_error(std::string("[VulkanContext] vkGetDeviceProcAddr(") +
                                             name + ") returned null");
                }
            };
            load("vkCreateAccelerationStructureKHR",        reinterpret_cast<void**>(&rt_.createAccelerationStructure));
            load("vkDestroyAccelerationStructureKHR",       reinterpret_cast<void**>(&rt_.destroyAccelerationStructure));
            load("vkGetAccelerationStructureBuildSizesKHR", reinterpret_cast<void**>(&rt_.getAccelerationStructureBuildSizes));
            load("vkCmdBuildAccelerationStructuresKHR",     reinterpret_cast<void**>(&rt_.cmdBuildAccelerationStructures));
            load("vkGetAccelerationStructureDeviceAddressKHR",
                 reinterpret_cast<void**>(&rt_.getAccelerationStructureDeviceAddress));
            load("vkCreateRayTracingPipelinesKHR",       reinterpret_cast<void**>(&rt_.createRayTracingPipelines));
            load("vkGetRayTracingShaderGroupHandlesKHR", reinterpret_cast<void**>(&rt_.getRayTracingShaderGroupHandles));
            load("vkCmdTraceRaysKHR",                    reinterpret_cast<void**>(&rt_.cmdTraceRays));
        }
    }

    namespace {
        // Shared body of the three setObjectName overloads — same call shape,
        // only objectType + handle differ. Validation-off path early-outs at
        // the null function-pointer check.
        void setObjectNameImpl(PFN_vkSetDebugUtilsObjectNameEXT fn,
                               VkDevice device, VkObjectType type,
                               uint64_t handle, const char* name) {
            if (!fn || handle == 0 || !name) return;
            VkDebugUtilsObjectNameInfoEXT info{};
            info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            info.objectType   = type;
            info.objectHandle = handle;
            info.pObjectName  = name;
            fn(device, &info);
        }
    }

    void VulkanContext::dumpPipelineStats(VkPipeline pipe, const char* label) const {
        if (!pipelineStatsEnabled_ || pipe == VK_NULL_HANDLE) return;

        auto getProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
                vkGetDeviceProcAddr(device_, "vkGetPipelineExecutablePropertiesKHR"));
        auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
                vkGetDeviceProcAddr(device_, "vkGetPipelineExecutableStatisticsKHR"));
        if (!getProps || !getStats) return;

        VkPipelineInfoKHR pi{};
        pi.sType    = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
        pi.pipeline = pipe;

        uint32_t nExec = 0;
        if (getProps(device_, &pi, &nExec, nullptr) != VK_SUCCESS || nExec == 0) return;
        std::vector<VkPipelineExecutablePropertiesKHR> execs(nExec);
        for (auto& e : execs) e.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
        getProps(device_, &pi, &nExec, execs.data());

        for (uint32_t i = 0; i < nExec; ++i) {
            VkPipelineExecutableInfoKHR ei{};
            ei.sType           = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
            ei.pipeline        = pipe;
            ei.executableIndex = i;

            uint32_t nStats = 0;
            if (getStats(device_, &ei, &nStats, nullptr) != VK_SUCCESS || nStats == 0) continue;
            std::vector<VkPipelineExecutableStatisticKHR> stats(nStats);
            for (auto& s : stats) s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            getStats(device_, &ei, &nStats, stats.data());

            std::cerr << "[pipeline-stats] " << label << " / " << execs[i].name << ":\n";
            for (const auto& s : stats) {
                std::cerr << "    " << s.name << " = ";
                switch (s.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                        std::cerr << (s.value.b32 ? "true" : "false"); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                        std::cerr << s.value.i64; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                        std::cerr << s.value.u64; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                        std::cerr << s.value.f64; break;
                    default: std::cerr << "?"; break;
                }
                std::cerr << "\n";
            }
        }
    }

    void VulkanContext::setObjectName(VkImage image, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_IMAGE,
                          reinterpret_cast<uint64_t>(image), name);
    }

    void VulkanContext::setObjectName(VkImageView view, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_IMAGE_VIEW,
                          reinterpret_cast<uint64_t>(view), name);
    }

    void VulkanContext::setObjectName(VkBuffer buffer, const char* name) const {
        setObjectNameImpl(setObjectNameFn_, device_, VK_OBJECT_TYPE_BUFFER,
                          reinterpret_cast<uint64_t>(buffer), name);
    }

    void VulkanContext::createAllocator() {
        VmaAllocatorCreateInfo ci{};
        ci.physicalDevice = physicalDevice_;
        ci.device = device_;
        ci.instance = instance_;
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        if (rayTracingEnabled_) {
            ci.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }
        check(vmaCreateAllocator(&ci, &allocator_), "vmaCreateAllocator");
    }

    void VulkanContext::createSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &caps);

        uint32_t fmtN = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &fmtN, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(fmtN);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &fmtN, formats.data());

        VkSurfaceFormatKHR chosenFmt = formats[0];
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosenFmt = f;
                break;
            }
        }
        if (chosenFmt.format != VK_FORMAT_B8G8R8A8_UNORM) {
            // Every CPU readback (readRGBPixels, scene capture, view readback)
            // and the event-camera shaders assume BGRA byte order in the
            // swapchain image. No known desktop ICD omits BGRA8_UNORM, but if
            // one ever does, swapped channels should be diagnosable from the
            // log rather than from staring at pink screenshots.
            std::cerr << "[VulkanContext] surface does not offer B8G8R8A8_UNORM; using format "
                      << chosenFmt.format << " - CPU readbacks assume BGRA byte order\n";
        }

        uint32_t pmN = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmN, nullptr);
        std::vector<VkPresentModeKHR> presentModes(pmN);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmN, presentModes.data());
        // Honor the canvas vsync flag. vsync (default) -> FIFO: present blocks on
        // the display refresh, capping the render loop to the monitor rate instead
        // of spinning the GPU at 100% (and avoids starving
        // co-resident compute such as on-device inference). vsync off -> prefer
        // MAILBOX (uncapped, no tearing) then IMMEDIATE, for lowest latency / fastest
        // progressive temporal (TAA/ReSTIR) convergence.
        VkPresentModeKHR chosenMode = VK_PRESENT_MODE_FIFO_KHR;// guaranteed; vsync-capped
        if (!vsync_) {
            bool hasMailbox = false, hasImmediate = false;
            for (auto m : presentModes) {
                if (m == VK_PRESENT_MODE_MAILBOX_KHR) hasMailbox = true;
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) hasImmediate = true;
            }
            if (hasMailbox) chosenMode = VK_PRESENT_MODE_MAILBOX_KHR;
            else if (hasImmediate) chosenMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        std::cout << "[VulkanContext] present mode: "
                  << (chosenMode == VK_PRESENT_MODE_FIFO_KHR      ? "FIFO (vsync)"
                      : chosenMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX"
                                                                  : "IMMEDIATE")
                  << "\n";

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX || headlessSurface_) {
            // No window defines the extent, so the canvas window size does.
            // The special "application chooses" value covers most of this, but
            // a headless surface needs the override unconditionally: nothing
            // pins its currentExtent to reality, and not every ICD returns the
            // special value there (SwiftShader reports a fixed 1280x720).
            int w, h;
            glfwGetFramebufferSize(window_, &w, &h);
            extent.width = std::clamp<uint32_t>(static_cast<uint32_t>(w),
                                                 caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp<uint32_t>(static_cast<uint32_t>(h),
                                                  caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t imageCount = caps.minImageCount + 1;
        if (presentSuppressed_) {
            // Suppressed presents mean the frame loop holds one acquired image
            // per frame-in-flight slot for the swapchain's whole lifetime, and
            // vkAcquireNextImageKHR may only be relied on to return without
            // blocking while at most (imageCount - minImageCount + 1) images are
            // app-owned. minImageCount + kFramesInFlight keeps that headroom at
            // one spare image no matter what the surface reports as its minimum.
            imageCount = caps.minImageCount + impl::kFramesInFlight;
        }
        if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
            imageCount = caps.maxImageCount;
        }

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imageCount;
        ci.imageFormat = chosenFmt.format;
        ci.imageColorSpace = chosenFmt.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        // readRGBPixels()/recordSceneCapture() copy straight from the swapchain
        // image, which is only legal when the image was created with
        // TRANSFER_SRC usage (VUID-vkCmdCopyImageToBuffer-srcImage-00186).
        // Every desktop WSI supports it in practice, but the spec only
        // guarantees COLOR_ATTACHMENT — request it conditionally and remember
        // the answer so the readback entry points can fail loudly instead of
        // issuing an invalid copy.
        swapchainTransferSrc_ = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        if (swapchainTransferSrc_) ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = chosenMode;
        ci.clipped = VK_TRUE;

        const uint32_t fams[] = {queueFamilies_.graphics, queueFamilies_.present};
        if (queueFamilies_.graphics != queueFamilies_.present) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = fams;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        swapchainFormat_ = chosenFmt.format;
        swapchainExtent_ = extent;

        uint32_t imN = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &imN, nullptr);
        swapchainImages_.resize(imN);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imN, swapchainImages_.data());
    }

    void VulkanContext::createSwapchainImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ci.subresourceRange.baseMipLevel = 0;
            ci.subresourceRange.levelCount = 1;
            ci.subresourceRange.baseArrayLayer = 0;
            ci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]), "vkCreateImageView");
        }
    }

    void VulkanContext::destroySwapchainResources() {
        for (auto v : swapchainImageViews_) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device_, v, nullptr);
        }
        swapchainImageViews_.clear();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchainImages_.clear();
    }

    void VulkanContext::recreateSwapchain() {
        // Spin while minimised.
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        destroySwapchainResources();
        createSwapchain();
        createSwapchainImageViews();
    }

    // ── ValidationReport.hpp ────────────────────────────────────────────────
    // Defined here, next to the messenger callback that feeds them, so the
    // counters stay internal to this TU. The public declarations carry no Vulkan
    // types precisely so that a test can read them without the PRIVATE Vulkan /
    // VMA include paths.

    std::uint32_t validationErrorCount() {
        return gValidationErrors.load(std::memory_order_relaxed);
    }

    std::uint32_t validationWarningCount() {
        return gValidationWarnings.load(std::memory_order_relaxed);
    }

    bool validationActive() {
        return gValidationActive.load();
    }

}// namespace threepp::vulkan
