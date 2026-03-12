
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/input/PeripheralsEventSource.hpp"

#include <functional>
#include <memory>
#include <utility>

namespace threepp {

    class Object3D;
    class GLRenderer;

    class HUD {

    public:
        enum class VerticalAlignment {
            BELOW,
            ABOVE,
            CENTER
        };

        enum class HorizontalAlignment {
            LEFT,
            RIGHT,
            CENTER
        };

        struct Options {

            Options(): margin_(5, 5),
                       verticalAlignment_(VerticalAlignment::ABOVE),
                       horizontalAlignment_(HorizontalAlignment::LEFT) {}

            Options& setVerticalAlignment(VerticalAlignment va) {
                this->verticalAlignment_ = va;

                needsUpdate_ = true;

                return *this;
            }

            Options& setHorizontalAlignment(HorizontalAlignment ha) {
                this->horizontalAlignment_ = ha;

                needsUpdate_ = true;

                return *this;
            }

            Options& setNormalizedPosition(float x, float y) {
                this->pos.set(x, y);

                needsUpdate_ = true;

                return *this;
            }

            Options& setNormalizedPosition(const Vector2& pos) {
                return setNormalizedPosition(pos.x, pos.y);
            }

            Options& setMargin(const Vector2& margin) {
                this->margin_.copy(margin);

                needsUpdate_ = true;

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

        private:
            friend class HUD;

            bool needsUpdate_{true};

            Vector2 pos;
            Vector2 margin_;
            std::function<void(int)> onMouseDown_ = [](int) {};
            std::function<void(int)> onMouseUp_ = [](int) {};

            VerticalAlignment verticalAlignment_;
            HorizontalAlignment horizontalAlignment_;

            void updateElement(std::pair<int, int> size);

            Object3D* object_{nullptr};
        };

        explicit HUD(GLRenderer& renderer, PeripheralsEventSource* eventSource = nullptr);

        void render();

        Options& add(Object3D& object);

        Options& add(const std::shared_ptr<Object3D>& object);

        Options* getStoredOptions(Object3D& object);

        void remove(Object3D& object);

        ~HUD();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
