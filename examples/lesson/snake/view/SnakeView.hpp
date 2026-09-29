// SnakeView.hpp: how a Game looks in 3D. It reads the game and never changes it.

#ifndef SNAKE_SNAKEVIEW_HPP
#define SNAKE_SNAKEVIEW_HPP

#include "model/Game.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <memory>
#include <vector>

namespace snake {

    class SnakeView {
    public:
        explicit SnakeView(const Game& game);

        // what to add to a scene; a view HAS an object, it isn't one
        std::shared_ptr<threepp::Object3D> object() const { return root_; }

        // alpha: how far we are from the last tick to the next one, 0 to 1
        void update(const Game& game, float alpha, float dt);

        // a ring that grows and fades, where something just happened
        void pop(Vec2i cell);

    private:
        std::shared_ptr<threepp::Object3D> root_;
        std::shared_ptr<threepp::BufferGeometry> segmentGeometry_;
        std::shared_ptr<threepp::MeshStandardMaterial> headMaterial_;
        std::shared_ptr<threepp::MeshStandardMaterial> bodyMaterial_;
        std::vector<std::shared_ptr<threepp::Mesh>> segments_;
        std::shared_ptr<threepp::Mesh> apple_;

        struct Ring {
            std::shared_ptr<threepp::Mesh> mesh;
            float age = 0;
        };
        std::vector<Ring> rings_;

        int width_;
        int height_;
        int seenTick_ = -1;
        std::vector<Vec2i> previous_;
        std::vector<Vec2i> current_;
        float time_ = 0;

        threepp::Vector3 toWorld(Vec2i cell, float y) const;
        threepp::Vector3 toWorld(Vec2i from, Vec2i to, float alpha, float y) const;
    };

}// namespace snake

#endif//SNAKE_SNAKEVIEW_HPP
