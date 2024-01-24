
#ifndef THREEPP_HUD_HPP
#define THREEPP_HUD_HPP

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/WindowSize.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/geometries/SphereGeometry.hpp"

#include <string>
#include <filesystem>

namespace threepp {

    struct TextRef {

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

        explicit TextRef(const std::filesystem::path& fontPath, unsigned int size = 5);

        void setText(const std::string& str);

        void setColor(const Color& color);

        void setSize(unsigned int size);

        void setPosition(float x, float y);

        void setVerticalAlignment(VerticalAlignment verticalAlignment);
        void setHorizontalAlignment(HorizontallAlignment horizontalAlignment);

    private:
        Font font_;
        unsigned int size_ = 1;
        std::shared_ptr<Mesh> mesh_;

        VerticalAlignment verticalAlignment_ = VerticalAlignment::BOTTOM;
        HorizontallAlignment horizontalAlignment_ = HorizontallAlignment::LEFT;

        Vector2 offset_;
        Vector2 pos_;

        friend class HUD;
    };

    class HUD {

    public:
        HUD(): camera_(0, 100, 100, 0) {}

        TextRef* addText(const std::filesystem::path& fontPath) {

            auto& text = refs_.emplace_back(fontPath);
            scene_.add(text.mesh_);

            return &text;
        }

        Scene& scene() {

            return scene_;
        }

        Camera& camera() {

            return camera_;
        }

    private:
        Scene scene_;
        OrthographicCamera camera_;

        std::vector<TextRef> refs_;
    };

}// namespace threepp

#endif//THREEPP_HUD_HPP
