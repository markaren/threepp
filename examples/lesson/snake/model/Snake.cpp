#include "Snake.hpp"

#include <algorithm>
#include <stdexcept>

using namespace snake;

Snake::Snake(std::vector<Vec2i> body, Direction heading)
    : body_(body.begin(), body.end()), heading_(heading) {

    if (body_.empty()) throw std::invalid_argument("a snake needs at least one cell");
}

bool Snake::occupies(Vec2i cell) const {
    return std::find(body_.begin(), body_.end(), cell) != body_.end();
}

void Snake::turn(Direction d) {
    if (d != opposite(heading_)) heading_ = d;
}

void Snake::advance(bool grow) {
    body_.push_front(next());
    if (!grow) body_.pop_back();
}
