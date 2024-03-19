
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/canvas/WindowSize.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <functional>
#include <memory>

namespace threepp {

    class Object3D;
    class GLRenderer;

    class HUD {

    public:
        enum class VerticalAlignment {
            TOP,
            BOTTOM,
            CENTER
        };

        enum class HorizontalAlignment {
            LEFT,
            RIGHT,
            CENTER
        };

        struct Options {

            Options(): margin_(5, 5),
                       verticalAlignment_(VerticalAlignment::BOTTOM),
                       horizontalAlignment_(HorizontalAlignment::LEFT) {}

            Options& setVerticalAlignment(VerticalAlignment va) {
                this->verticalAlignment_ = va;

                return *this;
            }

            Options& setHorizontalAlignment(HorizontalAlignment ha) {
                this->horizontalAlignment_ = ha;

                return *this;
            }

            Options& setNormalizedPosition(const Vector2& pos) {
                this->pos.copy(pos);

                return *this;
            }

            Options& setMargin(const Vector2& margin) {
                this->margin_.copy(margin);

                return *this;
            }

            Options& onMouseUp(const std::function<void(int)>& callback) {
                this->onMouseUp_ = callback;

                return *this;
            }

            Options& onMouseDown(const std::function<void(int)>& callback) {
                this->onMouseDown_ = callback;

                return *this;
            }

            void updateElement(Object3D& o, WindowSize size);

        private:
            friend class HUD;

            Vector2 pos;
            Vector2 margin_{5, 5};
            std::function<void(int)> onMouseDown_ = [](int) {};
            std::function<void(int)> onMouseUp_ = [](int) {};

            VerticalAlignment verticalAlignment_;
            HorizontalAlignment horizontalAlignment_;
        };

        explicit HUD(WindowSize size);
        explicit HUD(PeripheralsEventSource* eventSource);

        void apply(GLRenderer& renderer);

        void add(Object3D& object, Options opts = {});

        void add(const std::shared_ptr<Object3D>& object, Options opts);

        void remove(Object3D& object);

        void setSize(WindowSize size);

        void needsUpdate(Object3D& o);

        ~HUD();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
