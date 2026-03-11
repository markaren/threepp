
#ifndef THREEPP_TEXTNODE_HPP
#define THREEPP_TEXTNODE_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Sprite.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace threepp {

    // A class for creating 2D text sprites in a 3D scene.
    class TextSprite: public Sprite {

    public:
        explicit TextSprite(const Font& font);

        void setColor(const Color& color);

        void setText(const std::string& text, float worldScale = 1);

        [[nodiscard]] const Color& getColor() const;

        [[nodiscard]] std::string getText() const;

        static std::shared_ptr<TextSprite> create(const Font& font);

        ~TextSprite() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTNODE_HPP
