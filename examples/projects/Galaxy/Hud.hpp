// A small 2D layer drawn over the finished 3D picture: rounded cards and text.
//
// It is an ordinary threepp scene, looked at through an orthographic camera whose
// units are pixels (x from the left edge, y from the BOTTOM edge), and drawn last,
// without clearing what is already there. Nothing in it is a picture of text: letters
// are triangles made from a font's outlines (Text2D), cards are rounded rectangles made
// from a Shape. So everything stays sharp at any size.
//
// The fonts are the system's (Segoe UI and Bahnschrift on Windows, DejaVu on Linux).
// Where none is found, threepp's built-in font is used.

#ifndef THREEPP_GALAXY_HUD_HPP
#define THREEPP_GALAXY_HUD_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/Text.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class Hud {

public:
    enum class Face { Regular,
                      Semibold,
                      Numeric };// digits that all have the same width, so numbers do not jitter

    // One line of text.
    class Label {

    public:
        // Turning a string into triangles costs time, so this does nothing when the
        // text is the one already shown.
        void set(const std::string& text) {
            if (text != text_) {
                text_ = text;
                mesh_->setText(text);
            }
        }

        // where the left end of the baseline goes
        void place(float x, float y) {
            mesh_->position.set(x, y, 0);
        }

    private:
        friend class Hud;

        Label(const threepp::Font& font, float size, const threepp::Color& color, const std::string& text)
            : text_(text),
              mesh_(threepp::Text2D::create(threepp::TextGeometry::Options(font, size), text, flat(color, 1.f))) {}

        std::string text_;
        std::shared_ptr<threepp::Text2D> mesh_;
    };

    Hud()
        : camera_(std::make_shared<threepp::OrthographicCamera>(0.f, 1.f, 1.f, 0.f, -10.f, 10.f)) {

        const threepp::Font fallback = threepp::FontLoader().defaultFont();
        fonts_[static_cast<size_t>(Face::Regular)] = loadSystemFont({"segoeui.ttf", "DejaVuSans.ttf"}).value_or(fallback);
        fonts_[static_cast<size_t>(Face::Semibold)] = loadSystemFont({"seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"}).value_or(fallback);
        fonts_[static_cast<size_t>(Face::Numeric)] = loadSystemFont({"bahnschrift.ttf", "DejaVuSansMono.ttf"}).value_or(fallback);
    }

    // Each of these adds one thing to the layer and returns it, for the caller to
    // place (position = its bottom left corner). What is added later is drawn on
    // top of what was added before.
    Label& text(const std::string& text, float size, const threepp::Color& color, Face face = Face::Regular) {
        labels_.push_back(std::unique_ptr<Label>(new Label(fonts_[static_cast<size_t>(face)], size, color, text)));
        Label& label = *labels_.back();
        label.mesh_->renderOrder = drawOrder_++;
        scene_.add(label.mesh_);
        return label;
    }

    threepp::Mesh& panel(float width, float height, float radius, const threepp::Color& fill, float alpha) {
        return add(panelGeometry(width, height, radius), fill, alpha);
    }

    threepp::Mesh& frame(float width, float height, float radius, float thickness, const threepp::Color& color, float alpha) {
        return add(frameGeometry(width, height, radius, thickness), color, alpha);
    }

    // How wide a text would be, without creating it (to size the card behind it).
    [[nodiscard]] float measure(const std::string& text, float size, Face face = Face::Regular) const {
        return fonts_[static_cast<size_t>(face)].advance(text, size);
    }

    static void reshapePanel(threepp::Mesh& panel, float width, float height, float radius) {
        panel.setGeometry(panelGeometry(width, height, radius));
    }

    static void reshapeFrame(threepp::Mesh& frame, float width, float height, float radius, float thickness) {
        frame.setGeometry(frameGeometry(width, height, radius, thickness));
    }

    void resize(int width, int height) {
        camera_->right = static_cast<float>(width);
        camera_->top = static_cast<float>(height);
        camera_->updateProjectionMatrix();
    }

    void draw(threepp::GLRenderer& renderer) {
        // autoClear off: draw over the picture instead of wiping it first.
        const bool autoClear = renderer.autoClear;
        renderer.autoClear = false;
        renderer.render(scene_, *camera_);
        renderer.autoClear = autoClear;
    }

private:
    threepp::Scene scene_;
    std::shared_ptr<threepp::OrthographicCamera> camera_;
    std::array<threepp::Font, 3> fonts_;
    std::vector<std::unique_ptr<Label>> labels_;
    int drawOrder_ = 0;

    // The first of `files` that exists in one of the system's font folders.
    static std::optional<threepp::Font> loadSystemFont(std::initializer_list<const char*> files) {
        std::vector<std::filesystem::path> folders;
#ifdef _MSC_VER
#pragma warning(suppress : 4996)// getenv is fine here: read once, at start, on one thread
#endif
        if (const char* windows = std::getenv("WINDIR")) {
            folders.emplace_back(std::filesystem::path(windows) / "Fonts");
        }
        folders.emplace_back("/usr/share/fonts/truetype/dejavu");

        threepp::FontLoader loader;
        for (const char* file : files) {
            for (const auto& folder : folders) {
                const auto path = folder / file;
                if (std::filesystem::exists(path)) {
                    if (auto font = loader.load(path)) {
                        return font;
                    }
                }
            }
        }
        return std::nullopt;
    }

    // The outline of a rectangle with rounded corners, its bottom left corner at
    // (x, y). Each corner is a quadratic curve whose control point is the corner
    // the rectangle would have had.
    static void traceRoundedRect(threepp::Path& path, float x, float y, float width, float height, float radius) {
        radius = std::min(radius, std::min(width, height) / 2);
        path.moveTo(x + radius, y);
        path.lineTo(x + width - radius, y);
        path.quadraticCurveTo(x + width, y, x + width, y + radius);
        path.lineTo(x + width, y + height - radius);
        path.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        path.lineTo(x + radius, y + height);
        path.quadraticCurveTo(x, y + height, x, y + height - radius);
        path.lineTo(x, y + radius);
        path.quadraticCurveTo(x, y, x + radius, y);
    }

    static std::shared_ptr<threepp::BufferGeometry> panelGeometry(float width, float height, float radius) {
        threepp::Shape shape;
        traceRoundedRect(shape, 0, 0, width, height, radius);
        return threepp::ShapeGeometry::create(shape, 8);
    }

    // A frame is a rounded rectangle with a slightly smaller one cut out of it.
    static std::shared_ptr<threepp::BufferGeometry> frameGeometry(float width, float height, float radius, float thickness) {
        threepp::Shape shape;
        traceRoundedRect(shape, 0, 0, width, height, radius);
        auto hole = std::make_shared<threepp::Path>();
        traceRoundedRect(*hole, thickness, thickness, width - 2 * thickness, height - 2 * thickness, std::max(radius - thickness, 0.f));
        shape.holes.push_back(hole);
        return threepp::ShapeGeometry::create(shape, 8);
    }

    // A flat colour that ignores the 3D picture's depth buffer and its tone
    // mapping (the exposure keys must not brighten the text), and may be see-through.
    static std::shared_ptr<threepp::MeshBasicMaterial> flat(const threepp::Color& color, float alpha) {
        auto material = threepp::MeshBasicMaterial::create();
        material->color = color;
        material->opacity = alpha;
        material->transparent = true;
        material->depthTest = false;
        material->depthWrite = false;
        material->toneMapped = false;
        return material;
    }

    threepp::Mesh& add(const std::shared_ptr<threepp::BufferGeometry>& geometry, const threepp::Color& color, float alpha) {
        auto mesh = threepp::Mesh::create(geometry, flat(color, alpha));
        // See-through things are drawn in renderOrder. Counting up gives "what was
        // added later is on top", which is how one thinks about a 2D layout.
        mesh->renderOrder = drawOrder_++;
        scene_.add(mesh);
        return *mesh;
    }
};

#endif//THREEPP_GALAXY_HUD_HPP
