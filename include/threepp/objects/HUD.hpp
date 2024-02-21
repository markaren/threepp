
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/extras/core/Font.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <optional>
#include <string>

namespace threepp {

    class GLRenderer;

    class HUD: public Scene, public MouseListener {

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

            Options& onClick(const std::function<void(int)>& onClick) {
                this->onClick_ = onClick;

                return *this;
            }

            void updateElement(Object3D& o, WindowSize size);

        private:
            friend class HUD;

            Vector2 pos;
            Vector2 margin_{5, 5};
            std::function<void(int)> onClick_ = [](int){};

            VerticalAlignment verticalAlignment_;
            HorizontalAlignment horizontalAlignment_;
        };

        explicit HUD(PeripheralsEventSource& eventSource);

        void apply(GLRenderer& renderer);

        void add(Object3D& object, Options opts = {});

        void add(const std::shared_ptr<Object3D>& object, Options opts);

        void remove(Object3D& object) override;

        void setSize(WindowSize size);

        void needsUpdate(Object3D& o);

        void onMouseUp(int button, const Vector2& pos) override;
        void onMouseMove(const Vector2& pos) override;

    private:
        WindowSize size_;
        OrthographicCamera camera_;

        Raycaster raycaster_;
        Vector2 mouse_{-Infinity<float>, -Infinity<float>};

        std::unordered_map<Object3D*, Options> map_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
