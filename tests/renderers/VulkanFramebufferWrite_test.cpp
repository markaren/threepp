// VulkanFramebufferWrite_test — the framebuffer readback/write path is sized
// from the swapchain, not from the size the canvas asked for.
//
// The bug this pins: VulkanRenderer::writeFramebuffer took its width/height
// from size() (the canvas's requested window size) while readRGBPixels copies
// the presented SWAPCHAIN image and returns a buffer shaped by the swapchain
// extent. Those two numbers are usually equal, which is why it survived — but
// the platform is free to disagree with a size request. A 1920x1200 window on
// a 1920x1200 Windows desktop gets its client area clamped to the work area
// (1920x1181 with a taskbar on screen), and GLFW fires no resize callback for
// a size it only ever set once, at creation. stb_image_write was then handed
// h=1200 over an 1181-row buffer and read 19 rows — 109,440 bytes — off the
// end of the allocation: a hard 0xC0000005, deterministic, and only on the
// screenshot path.
//
// Note what is NOT the trigger, because the first report framed it that way:
// an odd framebuffer height. Odd heights are fine (gate 1 holds one); what
// broke was requested != actual. Both are gated below.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/canvas/Monitor.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kSkipCode = 42;// CTest SKIP_RETURN_CODE (no Vulkan GPU)

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %-64s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // PNG IHDR: 8-byte signature, then a chunk whose 8-byte header is followed
    // by width and height as big-endian u32. Reading the file back is the point
    // — asserting on what writeFramebuffer was *asked* for would re-assert the
    // bug rather than catch it.
    bool pngSize(const std::filesystem::path& p, int& w, int& h) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        unsigned char hdr[24]{};
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        if (f.gcount() != static_cast<std::streamsize>(sizeof(hdr))) return false;
        static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        for (int i = 0; i < 8; ++i) {
            if (hdr[i] != sig[i]) return false;
        }
        if (hdr[12] != 'I' || hdr[13] != 'H' || hdr[14] != 'D' || hdr[15] != 'R') return false;
        auto be32 = [](const unsigned char* b) {
            return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
                   (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
        };
        w = static_cast<int>(be32(hdr + 16));
        h = static_cast<int>(be32(hdr + 20));
        return w > 0 && h > 0;
    }

    void buildScene(Scene& scene) {
        scene.background = Color(0.05f, 0.06f, 0.09f);
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(0.8f, 0.3f, 0.2f);
        mat->roughness = 0.6f;
        scene.add(Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f), mat));
        scene.add(AmbientLight::create(0xffffff, 1.f));
    }

    // One capture at a requested canvas size. `label` names the gate; the
    // requested size is a REQUEST — every assertion below is against the
    // framebuffer the renderer actually got.
    void gate(const std::string& label, int reqW, int reqH,
              const std::filesystem::path& out) {

        Canvas canvas(Canvas::Parameters()
                              .title("fbwrite")
                              .size(reqW, reqH)
                              .vsync(false)
                              .resizable(false));
        VulkanRenderer renderer(canvas);

        Scene scene;
        buildScene(scene);
        PerspectiveCamera camera(60.f, canvas.aspect(), 0.1f, 100.f);
        camera.position.z = 4.f;

        const auto fb = renderer.framebufferSize();
        std::printf("%s: requested %dx%d -> framebuffer %dx%d%s\n",
                    label.c_str(), reqW, reqH, fb.width(), fb.height(),
                    (fb.width() != reqW || fb.height() != reqH) ? "  (platform clamped)" : "");

        // The invariant the crash violated. size() is what the rest of the
        // renderer sizes viewports and aspect from; the readback path is
        // swapchain-shaped. If these two can drift, every consumer that pairs
        // one with a buffer from the other is an out-of-bounds access waiting
        // for a window manager to disagree with a size request.
        const auto sz = renderer.size();
        check(sz.width() == fb.width() && sz.height() == fb.height(),
              label + ": size() agrees with framebufferSize()");

        canvas.animateOnce([&] { renderer.render(scene, camera); });

        const auto after = renderer.size();
        check(after.width() == fb.width() && after.height() == fb.height(),
              label + ": size() still agrees after render()");

        // readRGBPixels is the buffer writeFramebuffer over-read. Tightly
        // packed RGB8 at framebuffer resolution, exactly — no stride slack to
        // absorb a wrong height.
        const auto pixels = renderer.readRGBPixels();
        const auto expected = static_cast<std::size_t>(fb.width()) *
                              static_cast<std::size_t>(fb.height()) * 3;
        check(pixels.size() == expected,
              label + ": readRGBPixels() is exactly w*h*3 (" + std::to_string(pixels.size()) +
                      " vs " + std::to_string(expected) + ")");

        std::error_code ec;
        std::filesystem::remove(out, ec);

        // The crash site. Under the bug this call did not throw or return
        // false — it read off the end of the heap allocation and took the
        // process with it, so reaching the next line at all is half the gate.
        canvas.animateOnce([&] {
            renderer.render(scene, camera);
            renderer.writeFramebuffer(out);
        });

        check(std::filesystem::exists(out) && std::filesystem::file_size(out) > 0,
              label + ": writeFramebuffer produced a non-empty file");

        int pw = 0, ph = 0;
        const bool parsed = pngSize(out, pw, ph);
        check(parsed, label + ": output is a readable PNG");
        if (parsed) {
            check(pw == fb.width() && ph == fb.height(),
                  label + ": PNG is " + std::to_string(fb.width()) + "x" +
                          std::to_string(fb.height()) + " (got " + std::to_string(pw) + "x" +
                          std::to_string(ph) + ")");
        }

        std::filesystem::remove(out, ec);
    }

}// namespace

int main() {

    const auto tmp = std::filesystem::temp_directory_path();

    try {
        // Gate 1 — an ODD framebuffer height, the shape the bug was first
        // blamed on. A small window is granted verbatim on every platform
        // tried, so this gate is about the readback path handling h % 2 == 1
        // (no (h+1)/2 half-resolution assumption leaking into the copy or the
        // stride), NOT about the clamp.
        gate("odd-height", 641, 481, tmp / "threepp_fbwrite_odd.png");

        // Gate 2 — requested != actual, the real trigger. A window taller than
        // the desktop is the portable way to provoke a clamp: Windows pins the
        // client area to the work area, and the canvas's cached size is left
        // describing a window that never existed. Where the platform grants
        // the request instead, this degrades to another honest capture — the
        // assertions are all relative to framebufferSize(), so it passes
        // either way and never gives a false failure on a machine that does
        // not clamp.
        const auto mon = monitor::monitorSize();
        const int tallW = mon.width() > 640 ? 640 : mon.width();
        const int tallH = mon.height() > 0 ? mon.height() + 200 : 1400;
        gate("clamped-window", tallW, tallH, tmp / "threepp_fbwrite_clamped.png");

    } catch (const std::exception& e) {
        // No Vulkan device / no display: skip rather than fail. Matches the
        // other Vulkan tests in this directory.
        std::printf("skipping: %s\n", e.what());
        return kSkipCode;
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "OK" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
