
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include "threepp/canvas/WindowSize.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace threepp {

    enum class GraphicsAPI {
        OpenGL,
        Vulkan
    };

    class GLRenderer;
    class VulkanRenderer;

    class Canvas: public PeripheralsEventSource {

    public:
        struct Parameters;
        typedef std::variant<bool, int, std::string, WindowSize> ParameterValue;

        explicit Canvas(const Parameters& params = Parameters());

        explicit Canvas(const std::string& name);

        Canvas(const std::string& name, const std::unordered_map<std::string, ParameterValue>& values);

        //the current size of the Canvas window
        [[nodiscard]] WindowSize size() const override;

        [[nodiscard]] float aspect() const;

        void exitOnKeyEscape(bool value);

        void setSize(std::pair<int, int> size);

        /// Move the window so its OUTER top-left corner — decorations
        /// included — lands at (x, y) in screen coordinates: {0, 0} puts the
        /// whole window, title bar and all, in the corner (raw GLFW positions
        /// the content area, which would tuck the title bar off-screen).
        /// Records the position when the window is not yet created. No-op on
        /// Wayland (the protocol has no client-side positioning; GLFW reports
        /// the attempt through the error callback) and in the browser.
        void setPosition(std::pair<int, int> position);

        /// The outer top-left corner of the window in screen coordinates,
        /// decorations included (matching setPosition). Before the window
        /// exists this is the requested position, or {0, 0} when none was
        /// requested.
        [[nodiscard]] std::pair<int, int> position() const;

        void onWindowResize(std::function<void(WindowSize)> f);

        void onMonitorChange(std::function<void(int)> f) const;

        void animate(const std::function<void()>& f);

        // returns false if application should quit, true otherwise
        bool animateOnce(const std::function<void()>& f);

        [[nodiscard]] bool isOpen() const;

        void close();

        [[nodiscard]] void* windowPtr() const;

        [[nodiscard]] GraphicsAPI graphicsApi() const;

        [[nodiscard]] bool vsync() const;

        [[nodiscard]] int samples() const;

        /// True when the canvas was created with headless=true: the window (if
        /// any) is hidden, and a Vulkan renderer prefers a
        /// VK_EXT_headless_surface over a window surface, which needs no
        /// display server at all.
        [[nodiscard]] bool headless() const;

        /// Register a callback invoked at the end of each frame (after the user
        /// animate callback, before glfwPollEvents). Used by swapchain-based
        /// backends to present, analogous to glfwSwapBuffers for GL.
        void setFrameEndCallback(std::function<void()> callback);

        /// Reveal a window that was created hidden, once there is something to
        /// show in it. A Vulkan canvas starts hidden (see initWindow): device
        /// and pipeline setup takes long enough that a visible-but-blank window
        /// invites the user to click somewhere else, and the first real frame
        /// then surfaces behind whatever they clicked. The renderer calls this
        /// after presenting its first frame; focus follows the reveal
        /// (GLFW_FOCUS_ON_SHOW), at a moment when the window has pixels and the
        /// app can actually hold the foreground. Idempotent; never un-hides a
        /// headless canvas.
        void showWindow();

        /// True while inside animateOnce() user callback (between f() call
        /// and frame-end callback). Lets a swapchain-based backend decide
        /// whether to auto-present after render() or defer to the frame-end
        /// callback.
        [[nodiscard]] bool isInsideAnimateLoop() const;

        ~Canvas() override;

    private:
        void initWindow(GraphicsAPI api);

        friend class GLRenderer;
        friend class VulkanRenderer;

        struct Impl;
        std::unique_ptr<Impl> pimpl_;

    public:
        struct Parameters {

            Parameters();

            explicit Parameters(const std::unordered_map<std::string, ParameterValue>& values);

            Parameters& title(std::string value);

            Parameters& size(WindowSize size);

            Parameters& size(int width, int height);

            /// Place the window's outer top-left corner — decorations
            /// included, see Canvas::setPosition — at (x, y) in screen
            /// coordinates instead of letting the OS choose. Ignored when
            /// fullscreen(true) — the window covers the primary monitor from
            /// its own origin — and on Wayland, which has no client-side
            /// window positioning.
            Parameters& position(int x, int y);

            Parameters& antialiasing(int antialiasing);

            Parameters& vsync(bool flag);

            Parameters& resizable(bool flag);

            Parameters& favicon(const std::filesystem::path& path);

            Parameters& exitOnKeyEscape(bool flag);

            Parameters& headless(bool flag);

            /// Borderless windowed fullscreen: an undecorated, non-resizable
            /// window the size of the primary monitor's current video mode,
            /// placed at that monitor's origin. Any requested size() or
            /// position() is ignored. This is deliberately NOT exclusive fullscreen — the
            /// display mode is never changed, alt-tab behaves like any other
            /// window, and a Vulkan canvas can still stay hidden until its
            /// first present. Ignored when headless(true): there is no window
            /// to make fullscreen, and the size stays whatever was requested.
            Parameters& fullscreen(bool flag);

        private:
            std::optional<WindowSize> size_;
            std::optional<std::pair<int, int>> position_;
            int antialiasing_{4};
            std::string title_{"threepp"};
            bool vsync_{true};
            bool resizable_{true};
            bool exitOnKeyEscape_{true};
            bool headless_{false};
            bool fullscreen_{false};
            GraphicsAPI graphicsApi_{GraphicsAPI::OpenGL};
            std::optional<std::filesystem::path> favicon_;

            friend struct Impl;
            friend class Canvas;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
