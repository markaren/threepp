
#ifndef THREEPP_FONTTEXTURE_HPP
#define THREEPP_FONTTEXTURE_HPP

#include "threepp/math/Color.hpp"
#include "threepp/textures/Texture.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace threepp {

    class FontTexture: public Texture {

    public:
        [[nodiscard]] std::string text() const {
            return text_;
        }

        void setText(const std::string& text);

        void setColor(const Color& color);

        ~FontTexture();

        static std::shared_ptr<FontTexture> create(const std::filesystem::path& fontFile, float fontSize = 32) {
            return std::shared_ptr<FontTexture>(new FontTexture(fontFile, fontSize));
        }

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        std::string text_{"pppp"};

        FontTexture(const std::filesystem::path& fontFile, float fontSize, unsigned int padding = 2);
    };

}// namespace threepp

#endif//THREEPP_FONTTEXTURE_HPP
