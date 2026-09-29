#include "MinimapView.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace snake;
using namespace threepp;

namespace {

    constexpr int pixelsPerCell = 8;
    constexpr int margin = 16;

    void paint(std::vector<unsigned char>& pixels, int width, Vec2i cell, unsigned int rgb) {
        auto* p = &pixels[(cell.y * width + cell.x) * 4];
        p[0] = rgb >> 16 & 0xff;
        p[1] = rgb >> 8 & 0xff;
        p[2] = rgb & 0xff;
        p[3] = 255;
    }

}// namespace

MinimapView::MinimapView(const Game& game)
    : camera_(-0.5f, 0.5f, 0.5f, -0.5f, 0.1f, 10),
      texture_(DataTexture::create(4, game.width(), game.height())),
      width_(game.width()), height_(game.height()) {

    camera_.position.z = 1;
    texture_->colorSpace = ColorSpace::sRGB;// the colours below are the ones you see

    auto material = MeshBasicMaterial::create();
    material->map = texture_;
    scene_.add(Mesh::create(PlaneGeometry::create(1, 1), material));
}

void MinimapView::update(const Game& game) {
    auto& pixels = texture_->image().data();
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) paint(pixels, width_, {x, y}, 0x1e2820);
    }
    if (!game.won()) paint(pixels, width_, game.apple(), 0xe63946);
    for (const Vec2i cell : game.snake().body()) paint(pixels, width_, cell, game.alive() ? 0x5fb35a : 0x6c757d);
    paint(pixels, width_, game.snake().head(), game.alive() ? 0x9be564 : 0x8d99ae);
    texture_->needsUpdate();
}

void MinimapView::render(Renderer& renderer, WindowSize window) {
    const int w = width_ * pixelsPerCell;
    const int h = height_ * pixelsPerCell;
    const int x = window.width() - w - margin;
    const int y = window.height() - h - margin;// viewports count from the bottom

    renderer.setScissorTest(true);
    renderer.setScissor(x, y, w, h);
    renderer.setViewport(x, y, w, h);
    renderer.clearDepth();
    renderer.render(scene_, camera_);

    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, window.width(), window.height());
}
