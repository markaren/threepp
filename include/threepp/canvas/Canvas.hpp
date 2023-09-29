
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include "threepp/canvas/CanvasOptions.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/textures/Image.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <variant>

namespace threepp {

    namespace detail {
        typedef std::pair<std::function<void()>, float> Task;

        struct CustomComparator {
            bool operator()(const Task& l, const Task& r) const { return l.second > r.second; }
        };
    }// namespace detail

    class CanvasBase: public PeripheralsEventSource {

    public:
        explicit CanvasBase(const WindowSize& size);

        [[nodiscard]] const WindowSize& size() const;

        [[nodiscard]] float aspect() const;

        void onWindowResize(std::function<void(WindowSize)> f);

        void animate(const std::function<void()>& f);

        void invokeLater(const std::function<void()>& f, float t = 0);

        virtual void setSize(WindowSize size) = 0;

        // returns false if application should quit, true otherwise
        virtual bool animateOnce(const std::function<void()>& f) = 0;

        ~CanvasBase() override;

    protected:
        WindowSize size_;
        std::optional<std::function<void(WindowSize)>> resizeListener_;

        void handleTasks();

        static std::optional<Image> loadFavicon();

        virtual float getTime() = 0;

#ifdef HAS_IMGUI
        virtual void initImguiContext() = 0;
        virtual void newImguiFrame() = 0;
        virtual void destroyImguiContext() = 0;

        friend class ImguiContext;
#endif

    private:
        std::priority_queue<detail::Task, std::vector<detail::Task>, detail::CustomComparator> tasks_;

    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
