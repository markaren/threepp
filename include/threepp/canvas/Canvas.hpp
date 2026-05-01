
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
        WebGPU,
        // Hybrid renderer: WebGPU does scene rendering, OpenGL does display.
        // Selected via createRenderer; the canvas itself is initialised as
        // OpenGL by the wrapping CrossRenderer.
        Cross,
        Vulkan
    };

    class WgpuRenderer;
    class GLRenderer;
    class CrossRenderer;
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

        /// Register a callback invoked at the end of each frame (after the user
        /// animate callback, before glfwPollEvents). Used by WgpuRenderer to
        /// present the surface texture, analogous to glfwSwapBuffers for GL.
        void setFrameEndCallback(std::function<void()> callback);

        /// True while inside animateOnce() user callback (between f() call
        /// and frame-end callback). Used by WgpuRenderer to decide whether
        /// to auto-present after render() or defer to the frame-end callback.
        [[nodiscard]] bool isInsideAnimateLoop() const;

        ~Canvas() override;

    private:
        void initWindow(GraphicsAPI api);

        friend class WgpuRenderer;
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

            Parameters& antialiasing(int antialiasing);

            Parameters& vsync(bool flag);

            Parameters& resizable(bool flag);

            Parameters& favicon(const std::filesystem::path& path);

            Parameters& exitOnKeyEscape(bool flag);

            Parameters& headless(bool flag);

        private:
            std::optional<WindowSize> size_;
            int antialiasing_{4};
            std::string title_{"threepp"};
            bool vsync_{true};
            bool resizable_{true};
            bool exitOnKeyEscape_{true};
            bool headless_{false};
            GraphicsAPI graphicsApi_{GraphicsAPI::OpenGL};
            std::optional<std::filesystem::path> favicon_;

            friend struct Impl;
            friend class Canvas;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
