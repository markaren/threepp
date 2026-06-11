// Shared overlay primitives for the Vulkan inference demos (yolov8 / rtdetr /
// rfdetr): detection box outlines and class-label tags, drawn in the demos'
// 640x640 ortho space through the renderer's HUD overlay (Sprite +
// LineSegments + TextSprite — see examples' viewer blocks).

#ifndef THREEPP_EXAMPLES_DETECTION_OVERLAY_HPP
#define THREEPP_EXAMPLES_DETECTION_OVERLAY_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/extras/core/Font.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace detviz {

    inline const threepp::Color kPalette[6] = {
            threepp::Color(0xff3333), threepp::Color(0x33ff33), threepp::Color(0x3333ff),
            threepp::Color(0xffff33), threepp::Color(0xff33ff), threepp::Color(0x33ffff)};

    inline std::shared_ptr<threepp::LineSegments> makeBoxLines(
            float x1, float y1, float x2, float y2, const threepp::Color& col) {
        std::vector<float> positions = {
                x1, y1, 0, x2, y1, 0,
                x2, y1, 0, x2, y2, 0,
                x2, y2, 0, x1, y2, 0,
                x1, y2, 0, x1, y1, 0};
        auto geo = threepp::BufferGeometry::create();
        geo->setAttribute("position", threepp::FloatBufferAttribute::create(positions, 3));
        auto mat = threepp::LineBasicMaterial::create();
        mat->color = col;
        mat->depthTest = false;
        return threepp::LineSegments::create(geo, mat);
    }

    // 1x1 solid-color texture, cached per color. The renderer's sprite overlay
    // requires a texture map (map-less sprites are skipped), so the label
    // plate carries one of these.
    inline std::shared_ptr<threepp::Texture> solidTexture(const threepp::Color& c) {
        static std::unordered_map<unsigned int, std::shared_ptr<threepp::Texture>> cache;
        const auto hex = static_cast<unsigned int>(c.getHex());
        if (auto it = cache.find(hex); it != cache.end()) return it->second;
        std::vector<unsigned char> px{
                static_cast<unsigned char>(c.r * 255.f),
                static_cast<unsigned char>(c.g * 255.f),
                static_cast<unsigned char>(c.b * 255.f), 255};
        auto tex = threepp::Texture::create(threepp::Image(std::move(px), 1, 1));
        tex->needsUpdate();
        cache[hex] = tex;
        return tex;
    }

    // Class-label tag: colored plate + contrast text, anchored to the box's
    // top-left corner. Sits ABOVE the box (the classic detector look); flips
    // to inside the box when that would leave the 640-unit view. `yTop` is
    // the box's top edge in the demos' y-up ortho coordinates.
    inline std::shared_ptr<threepp::Object3D> makeLabel(
            const threepp::Font& font, const std::string& text,
            const threepp::Color& col, float x1, float yTop, float viewH = 640.f) {
        constexpr float kTextH = 16.f;// world units ≈ px at a 640-unit viewport
        constexpr float kPadX = 4.f, kPadY = 3.f;

        auto group = threepp::Group::create();

        auto label = threepp::TextSprite::create(font, kTextH);
        // black text on bright plate colors, white on dark ones
        const float lum = 0.299f * col.r + 0.587f * col.g + 0.114f * col.b;
        label->setColor(lum > 0.5f ? threepp::Color(0x000000) : threepp::Color(0xffffff));
        label->setText(text);
        label->setHorizontalAlignment(threepp::TextSprite::HorizontalAlignment::Left);
        label->setVerticalAlignment(threepp::TextSprite::VerticalAlignment::Center);

        const float tagH = kTextH + 2.f * kPadY;
        const float tagW = label->scale.x + 2.f * kPadX;// setText sized the sprite
        float yBase = yTop;                              // plate bottom on the box top edge
        if (yBase + tagH > viewH) yBase = std::max(0.f, yTop - tagH);

        // Explicit z layering (image/boxes sit at 0): same-z sprites are an
        // order-dependent depth tie — Vulkan happened to draw the text on
        // top, WGPU drew the plate over it.
        auto plateMat = threepp::SpriteMaterial::create();
        plateMat->map = solidTexture(col);
        auto plate = threepp::Sprite::create(plateMat);
        plate->center.set(0.f, 0.f);
        plate->scale.set(tagW, tagH, 1.f);
        plate->position.set(x1, yBase, 0.4f);
        group->add(plate);

        label->position.set(x1 + kPadX, yBase + tagH * 0.5f, 0.5f);
        group->add(label);
        return group;
    }

    // "name 0.87" — the conventional detector label text.
    inline std::string labelText(const char* name, float confidence) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s %.2f", name, confidence);
        return buf;
    }

}// namespace detviz

#endif//THREEPP_EXAMPLES_DETECTION_OVERLAY_HPP
