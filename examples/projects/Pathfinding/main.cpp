
#include "threepp/threepp.hpp"

#include "TileBasedMap.hpp"
#include "astar/AStar.hpp"
#include "heuristics/ClosestHeuristic.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

using namespace threepp;


class GameMap: public TileBasedMap {

public:
    explicit GameMap(std::vector<std::string> data): data_(std::move(data)) {

        height_ = data_.size();

        // check that width is consistent
        unsigned width = data_[0].size();
        for (int i = 1; i < data_.size(); i++) {
            if (data_[i].size() != width) {
                throw std::runtime_error("Input breadth mismatch!");
            }
        }
        width_ = width;
    }

    [[nodiscard]] unsigned int width() const override {

        return width_;
    }

    [[nodiscard]] unsigned int height() const override {

        return height_;
    }

    [[nodiscard]] char get(int x, int y) const {

        return data_[y][x];
    }

    [[nodiscard]] bool blocked(const Coordinate& v) const override {

        char c = get(v.x, v.y);
        bool blocked = (c == '1');
        return blocked;
    }

    float getCost(const Coordinate& start, const Coordinate& target) override {

        return 1;
    }

    [[nodiscard]] std::vector<std::string> data() const {
        return data_;
    }

private:
    unsigned int width_, height_;
    std::vector<std::string> data_;
};


int main() {

    std::vector<std::string> data{"1001111100",
                                  "1100011110",
                                  "0111001101",
                                  "1110010001",
                                  "1000111001",
                                  "1011011001",
                                  "1010011001",
                                  "1001111001",
                                  "1011001001",
                                  "1000000001"};

    auto map = std::make_unique<GameMap>(data);
    auto heuristic = std::make_unique<ClosestHeuristic>();

    AStar aStar(std::move(map), std::move(heuristic));
    aStar.setAllowDiagMovement(false);

    Coordinate start{2, 0};
    Coordinate target{4, 9};

    auto path = aStar.findPath(start, target);
    data[start.y][start.x] = 's';
    data[target.y][target.x] = 'g';

    if (path) {
        for (unsigned i = 0; i < path->length(); i++) {
            auto c = (*path)[i];
            auto& value = data[c.y][c.x];
            if (value != 's' || value != 'g') {
                value = 'x';
            }
        }
    } else {
        std::cout << "no path" << std::endl;
    }

    Canvas canvas("AStar pathfinding");
    GLRenderer renderer(canvas.size());

    Scene scene;

    int size = 10;
    OrthographicCamera camera(-size * canvas.aspect() / 2, size * canvas.aspect() / 2, -size / 2, size / 2, 0.1, 100);
    camera.position.z = 1;

    auto grid = GridHelper::create(size, size, Color::white, Color::white);
    grid->rotateX(math::PI / 2);
    scene.add(grid);

    auto boxGeometry = BoxGeometry::create(0.95, 0.95, 0.1);
    boxGeometry->translate(0.5, 0.5, 0);

    std::unordered_map<char, Color> colors = {{'0', Color::black},
                                              {'1', Color::white},
                                              {'s', Color::blue},
                                              {'g', Color::green},
                                              {'x', Color::yellow}};

    for (unsigned i = 0; i < 10; i++) {
        for (unsigned j = 0; j < 10; j++) {
            auto value = data[i][j];

            auto material = MeshBasicMaterial::create({{"color", colors[value]}});
            auto box = Mesh::create(boxGeometry, material);
            box->position.set(static_cast<float>(j) - 5, static_cast<float>(i) - 5, 0);
            scene.add(box);
        }
    }

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
