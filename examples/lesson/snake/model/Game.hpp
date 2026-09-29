// Game.hpp: the rules of Snake. No window, no meshes, no threepp:
// this library doesn't even link threepp, so it can't include it by accident.

#ifndef SNAKE_GAME_HPP
#define SNAKE_GAME_HPP

#include "Snake.hpp"

#include <functional>
#include <random>
#include <vector>

namespace snake {

    class Game {
    public:
        using Listener = std::function<void(const Game&)>;

        // a new game: a short snake in the middle, heading right, and an apple
        Game(int width, int height, unsigned seed);

        // a game set up by hand, as tests do
        Game(int width, int height, Snake snake, Vec2i apple, unsigned seed = 1);

        // what the player wants; it takes effect on the next tick
        void steer(Direction d);

        // one step of the rules
        void tick();

        int width() const { return width_; }
        int height() const { return height_; }
        const Snake& snake() const { return snake_; }
        Vec2i apple() const { return apple_; }
        int score() const { return score_; }
        int ticks() const { return ticks_; }
        bool alive() const { return alive_; }
        bool won() const { return won_; }

        bool inside(Vec2i cell) const;
        bool isFree(Vec2i cell) const;

        // anyone may listen; the game never knows who does
        void onEat(Listener f);
        void onDie(Listener f);

    private:
        int width_;
        int height_;
        Snake snake_;
        Vec2i apple_;
        Direction wanted_;
        int score_ = 0;
        int ticks_ = 0;
        bool alive_ = true;
        bool won_ = false;
        std::mt19937 rng_;

        std::vector<Listener> eatListeners_;
        std::vector<Listener> dieListeners_;

        void placeApple();
    };

}// namespace snake

#endif//SNAKE_GAME_HPP
