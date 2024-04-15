
#ifndef THREEPP_SNAKESCENE_HPP
#define THREEPP_SNAKESCENE_HPP

#include "threepp/threepp.hpp"

#include "SnakeGame.hpp"

using namespace threepp;


class SnakeScene: public Scene, public KeyListener {

public:
    explicit SnakeScene(SnakeGame& game): game_(game) {

        int size = game.gridSize();
        auto grid = GridHelper::create(size, size, 0x444444, 0x444444);
        grid->rotation.x = math::PI / 2;
        grid->position.set(static_cast<float>(size) / 2, static_cast<float>(size) / 2, 0);
        add(grid);

        boxGeometry_ = BoxGeometry::create(0.95, 0.95, 0.1);
        boxGeometry_->translate(0.5, 0.5, 0);

        auto foodMaterial = MeshBasicMaterial::create();
        foodMaterial->color = Color::green;

        food_ = Mesh::create(boxGeometry_, foodMaterial);
        add(food_);

        snakeMaterial_ = MeshBasicMaterial::create();
        snake_.emplace_back(Mesh::create(boxGeometry_, snakeMaterial_));
        add(snake_.back());
    }

    void onKeyPressed(KeyEvent evt) override {

        if (game_.isRunning()) {

            switch (evt.key) {
                case Key::UP:
                case Key::W: {
                    if (game_.direction != Direction::DOWN) {
                        game_.nextDirection = Direction::UP;
                    }
                } break;
                case Key::DOWN:
                case Key::S: {
                    if (game_.direction != Direction::UP) {
                        game_.nextDirection = Direction::DOWN;
                    }
                } break;
                case Key::LEFT:
                case Key::A: {
                    if (game_.direction != Direction::RIGHT) {
                        game_.nextDirection = Direction::LEFT;
                    }
                } break;
                case Key::RIGHT:
                case Key::D: {
                    if (game_.direction != Direction::LEFT) {
                        game_.nextDirection = Direction::RIGHT;
                    }
                }
            }
        }

        if (evt.key == Key::R) {

            game_.reset();
            reset();
        }
    }

    void update() {
        auto foodPos = game_.foodPos();
        food_->position.set(foodPos.x, foodPos.y, 0);

        auto& positions = game_.snake().positions();
        for (unsigned i = 0; i < positions.size(); ++i) {
            auto& pos = positions.at(i);
            if (positions.size() != snake_.size()) {
                snake_.emplace_back(Mesh::create(boxGeometry_, snakeMaterial_));
                add(snake_.back());
            }
            snake_.at(i)->position.set(pos.x, pos.y, 0);
        }

        if (!game_.isRunning()) {
            snakeMaterial_->color = Color::red;
        }
    }

    void reset() {
        // keep initial box
        auto head = snake_.front();
        for (unsigned i = 1; i < snake_.size(); ++i) {
            remove(*snake_.at(i));
        }
        snake_.clear();
        snake_.emplace_back(head);

        snakeMaterial_->color = Color::white;
    }

private:
    SnakeGame& game_;
    std::shared_ptr<Mesh> food_;

    std::shared_ptr<BoxGeometry> boxGeometry_;
    std::shared_ptr<MeshBasicMaterial> snakeMaterial_;
    std::vector<std::shared_ptr<Mesh>> snake_;
};


#endif//THREEPP_SNAKESCENE_HPP
