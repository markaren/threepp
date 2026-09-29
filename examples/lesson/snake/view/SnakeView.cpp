#include "SnakeView.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/RingGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"

#include <algorithm>
#include <cmath>

using namespace snake;
using namespace threepp;

namespace {

    constexpr float ringLife = 0.8f;// seconds

}// namespace

SnakeView::SnakeView(const Game& game)
    : root_(Group::create()),
      segmentGeometry_(BoxGeometry::create(0.9f, 0.9f, 0.9f)),
      headMaterial_(MeshStandardMaterial::create()),
      bodyMaterial_(MeshStandardMaterial::create()),
      width_(game.width()), height_(game.height()) {

    headMaterial_->color = 0x9be564;
    bodyMaterial_->color = 0x5fb35a;

    auto boardMaterial = MeshStandardMaterial::create();
    boardMaterial->color = 0x2d3a2e;
    auto board = Mesh::create(PlaneGeometry::create(static_cast<float>(width_), static_cast<float>(height_)), boardMaterial);
    board->rotation.x = -math::PI / 2;
    board->receiveShadow = true;
    root_->add(board);

    auto grid = GridHelper::create(std::max(width_, height_), std::max(width_, height_), 0x3b4a3c, 0x3b4a3c);
    grid->position.y = 0.005f;
    grid->scale.set(width_ / static_cast<float>(std::max(width_, height_)), 1, height_ / static_cast<float>(std::max(width_, height_)));
    root_->add(grid);

    auto appleMaterial = MeshStandardMaterial::create();
    appleMaterial->color = 0xe63946;
    apple_ = Mesh::create(SphereGeometry::create(0.4f, 24, 16), appleMaterial);
    apple_->castShadow = true;
    root_->add(apple_);
}

void SnakeView::update(const Game& game, float alpha, float dt) {
    time_ += dt;

    // a new tick: what was current becomes where the snake came from
    if (game.ticks() != seenTick_) {
        seenTick_ = game.ticks();
        previous_ = current_;
        const auto& body = game.snake().body();
        current_.assign(body.begin(), body.end());
        if (previous_.empty()) previous_ = current_;
    }

    while (segments_.size() < current_.size()) {
        auto segment = Mesh::create(segmentGeometry_, segments_.empty() ? headMaterial_ : bodyMaterial_);
        segment->castShadow = true;
        root_->add(segment);
        segments_.push_back(segment);
    }

    // each segment slides from where it was to where it is; a new tail starts at the old one
    for (size_t i = 0; i < current_.size(); i++) {
        const Vec2i from = previous_[std::min(i, previous_.size() - 1)];
        segments_[i]->position.copy(toWorld(from, current_[i], alpha, 0.45f));
    }

    if (!game.alive()) {
        headMaterial_->color = 0x8d99ae;
        bodyMaterial_->color = 0x6c757d;
    }

    apple_->position.copy(toWorld(game.apple(), 0.45f + 0.08f * std::sin(time_ * 4)));
    apple_->visible = !game.won();

    for (auto& ring : rings_) {
        ring.age += dt;
        const float t = std::min(ring.age / ringLife, 1.f);
        ring.mesh->scale.setScalar(0.8f + 2.7f * t);
        ring.mesh->material()->opacity = 1 - t;
    }
    std::erase_if(rings_, [this](const Ring& ring) {
        if (ring.age < ringLife) return false;
        root_->remove(*ring.mesh);
        return true;
    });
}

void SnakeView::pop(Vec2i cell) {
    auto material = MeshBasicMaterial::create();
    material->color = 0xffe066;
    material->transparent = true;
    auto mesh = Mesh::create(RingGeometry::create(0.4f, 0.55f, 32), material);
    mesh->rotation.x = -math::PI / 2;
    mesh->position.copy(toWorld(cell, 0.05f));
    root_->add(mesh);
    rings_.push_back({mesh});
}

Vector3 SnakeView::toWorld(Vec2i cell, float y) const {
    // the board is centred on the origin, and "up" on the board is away from the camera
    return {cell.x - (width_ - 1) / 2.f, y, (height_ - 1) / 2.f - cell.y};
}

Vector3 SnakeView::toWorld(Vec2i from, Vec2i to, float alpha, float y) const {
    return toWorld(from, y).lerp(toWorld(to, y), alpha);
}
