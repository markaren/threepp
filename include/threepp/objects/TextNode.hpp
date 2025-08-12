
#ifndef THREEPP_TEXTNODE_HPP
#define THREEPP_TEXTNODE_HPP

#include "threepp/math/Color.hpp"
#include "threepp/objects/Sprite.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace threepp {

    // A class for creating 2D text nodes in a 3D scene.
    // Only works with TrueType fonts.
    class TextNode: public Sprite {

    public:
        explicit TextNode(const std::filesystem::path& fontPath);

        void setColor(const Color& color);

        void setText(const std::string& text, float worldScale);

        static std::shared_ptr<TextNode> create(const std::filesystem::path& fontPath);

        ~TextNode() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTNODE_HPP
