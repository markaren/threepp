
#ifndef THREEPP_SNAKE_HPP
#define THREEPP_SNAKE_HPP

#include "threepp/math/Vector2.hpp"

#include <algorithm>
#include <vector>

using namespace threepp;

class Snake {

public:

    Snake(int xStart, int yStart) : xStart_(xStart), yStart_(yStart) {
        reset();
    }

    [[nodiscard]] size_t size() const {
        return positions_.size();
    }

    [[nodiscard]] const Vector2 &headPosition() const {
        return positions_.front();
    }

    bool move(const Vector2 &nextPos, const Vector2 food) {

        float foodDist = nextPos.distanceTo(food);
        if (foodDist < 0.1) {
            grow(nextPos);
            return true;
        }

        auto posCopy = positions_;

        positions_.front().copy(nextPos);
        for (unsigned i = 1; i < positions_.size(); ++i) {
            positions_[i].copy(posCopy[i-1]);
        }

        return false;
    }

    void grow(const Vector2& pos) {
        positions_.insert(positions_.begin(), pos);
    }

    void reset() {
        positions_.clear();
        positions_.emplace_back(xStart_, yStart_);
    }

    [[nodiscard]] const std::vector<Vector2> &positions() const {
        return positions_;
    }

    [[nodiscard]] bool checkSelfCollision(const Vector2& pos) const {
        return std::any_of(positions_.begin(), positions_.end(), [&](auto &p) {
            return pos.distanceTo(p) < 0.1;
        });
    }


private:
    int xStart_, yStart_;
    std::vector<Vector2> positions_;
};



#endif//THREEPP_SNAKE_HPP
