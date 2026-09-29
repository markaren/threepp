// capture.hpp: only for the film. app.cpp includes it when LESSON_CAPTURE is defined,
// after every threepp header; the Snake programs get it force-included by their capture
// targets (so their sources stay as shown). It swaps three names for stand-ins:
//
//   Canvas      a headless canvas whose animate() plays a mouse script and saves frames
//   GLRenderer  the same renderer, which tells the canvas where to read its pixels
//   Clock       a clock that advances exactly 1/60 s per frame
//
// Everything else is the program as written. Two environment variables drive it:
//
//   LESSON_SCRIPT   a text file: "frames N", then "<frame> move x y", "<frame> down b x y",
//                   "<frame> up b x y" and "<frame> wheel dy", in window pixels, and
//                   "<frame> key NAME" (a press and release; NAME as in KeyFromName.hpp: UP, A, SPACE)
//   LESSON_OUT      where the frames go: N x height x width x 3 bytes, top row first
//                   ("-" streams them to stdout, for a reader that takes them as they come)
//   LESSON_STATS    optional: a text file for the renderer's draw calls and triangles per
//                   frame (default <LESSON_OUT>.txt)
//   LESSON_MARK_FRAMES  optional: print "LESSON_FRAME k" before frame k, so whatever the
//                   program prints can be matched to the frame it was printed in
//
// The mouse reaches the program through the canvas's own event path (so OrbitControls,
// listeners and IOCapture see it as they would a real one), and Dear ImGui through its
// input queue, as its GLFW backend would feed it. Keys go through the same event path.

#ifndef THREEPP_LESSON_CAPTURE_HPP
#define THREEPP_LESSON_CAPTURE_HPP

// all of threepp first, so no threepp header is read after the names below are swapped
#include "threepp/input/KeyFromName.hpp"
#include "threepp/threepp.hpp"

#if __has_include(<imgui.h>)
#include <imgui.h>
#define LESSON_CAPTURE_IMGUI
#endif

#include <cstdio>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace lesson_capture {

    struct Event {
        std::string kind;
        int button{0};
        float x{0}, y{0};
        std::string key;
    };

    class CaptureRenderer;

    class CaptureCanvas: public threepp::Canvas {
    public:
        CaptureCanvas(const std::string& name, std::unordered_map<std::string, ParameterValue> values)
            : Canvas(name, (values["headless"] = true, values)) {}

        void animate(const std::function<void()>& f) {
            const char* scriptPath = std::getenv("LESSON_SCRIPT");
            const char* outPath = std::getenv("LESSON_OUT");
            if (!scriptPath || !outPath || !renderer) {
                std::cerr << "capture: set LESSON_SCRIPT and LESSON_OUT\n";
                std::exit(2);
            }
            int frames = 0;
            std::map<int, std::vector<Event>> events;
            std::ifstream script(scriptPath);
            for (std::string line; std::getline(script, line);) {
                std::istringstream in(line);
                std::string first;
                if (!(in >> first) || first[0] == '#') continue;
                if (first == "frames") {
                    in >> frames;
                    continue;
                }
                Event e;
                in >> e.kind;
                if (e.kind == "down" || e.kind == "up") in >> e.button;
                if (e.kind == "wheel") in >> e.y;
                else if (e.kind == "key") in >> e.key;
                else in >> e.x >> e.y;
                events[std::stoi(first)].push_back(e);
            }

            const auto size = this->size();
            const int w = size.width(), h = size.height();
            std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3), row(static_cast<size_t>(w) * 3);
            const bool toStdout = std::string(outPath) == "-";
            std::FILE* out;
            if (toStdout) {
                // frames get the real stdout to themselves; anything the program prints from
                // here on goes to stderr, and a marker line tells the reader where frames begin
                std::fflush(stdout);
#ifdef _WIN32
                out = _fdopen(_dup(_fileno(stdout)), "wb");
                _setmode(_fileno(out), _O_BINARY);
                _dup2(_fileno(stderr), _fileno(stdout));
#else
                out = fdopen(dup(fileno(stdout)), "wb");
                dup2(fileno(stderr), fileno(stdout));
#endif
                std::fprintf(out, "LESSON_FRAMES %d %d %d\n", frames, w, h);
            } else {
                out = std::fopen(outPath, "wb");
            }
            const char* statsPath = std::getenv("LESSON_STATS");
            std::ofstream stats(statsPath ? std::string(statsPath) : std::string(outPath) + ".txt");
            threepp::Vector2 cursor{-1, -1};
            const bool markFrames = std::getenv("LESSON_MARK_FRAMES") != nullptr;
            for (int k = 0; k < frames; k++) {
                if (markFrames) std::cout << "LESSON_FRAME " << k << std::endl;
                for (const auto& e : events[k]) {
                    if (e.kind == "key") {
                        const threepp::KeyEvent key(threepp::keyFromName(e.key), 0, 0);
                        onKeyEvent(key, KeyAction::PRESS);
                        onKeyEvent(key, KeyAction::RELEASE);
                        continue;
                    }
                    if (e.kind != "wheel") cursor.set(e.x, e.y);
                    feedImgui(e, cursor);
                    if (e.kind == "move") onMouseMoveEvent(cursor);
                    else if (e.kind == "down") onMousePressedEvent(e.button, cursor, MouseAction::PRESS);
                    else if (e.kind == "up") onMousePressedEvent(e.button, cursor, MouseAction::RELEASE);
                    else if (e.kind == "wheel") onMouseWheelEvent(threepp::Vector2(0.f, e.y));
                }
                animateOnce([&] {
                    f();
                    const auto& info = rendererInfo();
                    stats << info.render.calls << " " << info.render.triangles << "\n";
                    readPixels(rgb.data(), w, h);
                    for (int y = 0; y < h / 2; y++) {// GL's rows run bottom-up
                        auto* a = rgb.data() + static_cast<size_t>(y) * w * 3;
                        auto* b = rgb.data() + static_cast<size_t>(h - 1 - y) * w * 3;
                        std::copy(a, a + w * 3, row.data());
                        std::copy(b, b + w * 3, a);
                        std::copy(row.data(), row.data() + w * 3, b);
                    }
                    std::fwrite(rgb.data(), 1, rgb.size(), out);
                });
            }
            if (toStdout) std::fflush(out);
            else std::fclose(out);
            std::cerr << "capture: " << frames << " frames of " << w << " x " << h << "\n";
        }

        CaptureRenderer* renderer = nullptr;

    private:
        static void feedImgui([[maybe_unused]] const Event& e, [[maybe_unused]] const threepp::Vector2& cursor) {
#ifdef LESSON_CAPTURE_IMGUI
            if (!ImGui::GetCurrentContext()) return;
            auto& io = ImGui::GetIO();
            if (e.kind == "wheel") io.AddMouseWheelEvent(0, e.y);
            else io.AddMousePosEvent(cursor.x, cursor.y);
            if (e.kind == "down" || e.kind == "up") io.AddMouseButtonEvent(e.button, e.kind == "down");
#endif
        }

        const threepp::gl::GLInfo& rendererInfo() const;
        void readPixels(unsigned char* data, int w, int h) const;
    };

    class CaptureRenderer: public threepp::GLRenderer {
    public:
        explicit CaptureRenderer(CaptureCanvas& canvas): GLRenderer(canvas) {
            canvas.renderer = this;
        }
    };

    inline const threepp::gl::GLInfo& CaptureCanvas::rendererInfo() const {
        return renderer->info();
    }

    inline void CaptureCanvas::readPixels(unsigned char* data, int w, int h) const {
        renderer->readPixels({0, 0}, {w, h}, threepp::Format::RGB, data);
    }

    // one frame at 60 Hz, whatever the wall clock says
    struct FixedClock {
        float elapsed = 0;
        float getDelta() {
            elapsed += 1.f / 60;
            return 1.f / 60;
        }
        [[nodiscard]] float getElapsedTime() const { return elapsed; }
    };

}// namespace lesson_capture

#define Canvas lesson_capture::CaptureCanvas
#define GLRenderer lesson_capture::CaptureRenderer
#define Clock lesson_capture::FixedClock

#endif//THREEPP_LESSON_CAPTURE_HPP
