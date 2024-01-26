
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/extras/core/Font.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/extras/core/Font.hpp"

#include <string>
#include <optional>

namespace threepp {

    class GLRenderer;

    struct HudText {

        enum class VerticalAlignment {
            TOP,
            BOTTOM,
            CENTER
        };

        enum class HorizontallAlignment {
            LEFT,
            RIGHT,
            CENTER
        };

        explicit HudText(Font font, std::optional<unsigned int> size = std::nullopt);

        void setText(const std::string& str);

        void setColor(const Color& color);

        void scale(float scale);

        void setPosition(float x, float y);

        void setVerticalAlignment(VerticalAlignment verticalAlignment);
        void setHorizontalAlignment(HorizontallAlignment horizontalAlignment);

        void setMargin(const Vector2& margin);

        static std::shared_ptr<HudText> create(const Font& font, std::optional<unsigned int> size = std::nullopt) {

            return std::make_shared<HudText>(font, size);
        }

    private:
        Font font_;
        unsigned int size_ = 1;
        std::shared_ptr<Mesh> mesh_;

        VerticalAlignment verticalAlignment_ = VerticalAlignment::BOTTOM;
        HorizontallAlignment horizontalAlignment_ = HorizontallAlignment::LEFT;

        Vector2 margin_{2, 2};
        Vector2 offset_;
        Vector2 pos_;

        float scale_{0.01};

        void updateSettings();

        friend class HUD;
    };

    class HUD: public Scene {

    public:
        HUD(): camera_(0, 1, 1, 0, 0.1, 10) {

            camera_.position.z = 1;
        }

        void addText(HudText& text) {

            add(text.mesh_);
        }

        void addText(std::shared_ptr<HudText>& text) {

            add(text->mesh_);
        }

        void removeText(HudText& text) {

            remove(*text.mesh_);
        }

        void apply(GLRenderer& renderer);

    private:
        OrthographicCamera camera_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
