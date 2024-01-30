
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/extras/core/Font.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <optional>
#include <string>

namespace threepp {

    class GLRenderer;

    class HUD: public Scene {

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

            Options(): margin(5, 5),
                       verticalAlignment(VerticalAlignment::BOTTOM),
                       horizontalAlignment(HorizontalAlignment::LEFT) {}

            Options& setVerticalAlignment(VerticalAlignment va) {
                this->verticalAlignment = va;

                return *this;
            }

            Options& setHorizontalAlignment(HorizontalAlignment ha) {
                this->horizontalAlignment = ha;

                return *this;
            }

            Options& setNormalizedPosition(const Vector2& pos) {
                this->pos.copy(pos);

                return *this;
            }

            Options& setMargin(const Vector2& margin) {
                this->margin.copy(margin);

                return *this;
            }

            void updateElement(Object3D& o, WindowSize size);

        private:
            Vector2 pos;
            Vector2 margin{5, 5};

            VerticalAlignment verticalAlignment;
            HorizontalAlignment horizontalAlignment;
        };

        explicit HUD(WindowSize size): size_(size), camera_(0, size.width, size.height, 0, 0.1, 10) {

            camera_.position.z = 1;
        }

        void apply(GLRenderer& renderer);

        void add(Object3D& object, Options opts = {});

        void add(const std::shared_ptr<Object3D>& object, Options opts);

        void remove(Object3D& object) override;

        void setSize(WindowSize size);

        void needsUpdate(Object3D& o);

    private:
        WindowSize size_;
        OrthographicCamera camera_;

        std::unordered_map<Object3D*, Options> map_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
