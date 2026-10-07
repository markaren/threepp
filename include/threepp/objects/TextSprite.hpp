
#ifndef THREEPP_TEXTNODE_HPP
#define THREEPP_TEXTNODE_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Sprite.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace threepp {

    // A class for creating 2D text sprites in a 3D scene.
    //
    // `worldScale` is the sprite's height, and `box` says what of the text that
    // height spans:
    //   TextBox::Ink (default)  the string's own ink. Simple for a lone label,
    //       but the glyph size then depends on the string: at one worldScale
    //       "OTTER" is drawn about 40 % larger than "Kayak", whose y hangs below
    //       the baseline, and their baselines differ.
    //   TextBox::Line  one line of the font, ascender to descender. Every string
    //       gets the same glyph size, and its baseline lies
    //       -descender / (ascender - descender) of worldScale above the sprite's
    //       bottom edge (Font::ascender, Font::descender), so text can be laid
    //       out without measuring it. Each line break in the text adds a line
    //       to the sprite's height.
    // The sprite's width is the string's ink in both.
    class TextSprite: public Sprite {

    public:
        enum class HorizontalAlignment { Left, Center, Right };
        enum class VerticalAlignment { Above, Center, Below };

        explicit TextSprite(const Font& font, std::optional<float> worldScale = {}, TextBox box = TextBox::Ink);

        void setColor(const Color& color);

        void setWorldScale(float worldScale);

        // What worldScale spans: the string's ink, or a line of the font.
        void setTextBox(TextBox box);

        [[nodiscard]] TextBox getTextBox() const;

        void setText(const std::string& text);

        // Alignment controls the sprite pivot (center property).
        // Vertical: Above = sprite extends upward from position, Center = centered, Below = sprite extends downward.
        void setVerticalAlignment(VerticalAlignment v);

        // Alignment controls the sprite pivot (center property).
        // Horizontal: Left = left edge at position, Center = centered, Right = right edge at position.
        void setHorizontalAlignment(HorizontalAlignment h);

        VerticalAlignment getVerticalAlignment() const;

        HorizontalAlignment getHorizontalAlignment() const;

        [[nodiscard]] const Color& getColor() const;

        [[nodiscard]] std::string getText() const;

        static std::shared_ptr<TextSprite> create(const Font& font, std::optional<float> worldScale = {}, TextBox box = TextBox::Ink);

        ~TextSprite() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTNODE_HPP
