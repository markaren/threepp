
#include "threepp/threepp.hpp"

#include "GameMap.hpp"
#include "astar/AStar.hpp"
#include "heuristics/ClosestHeuristic.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

using namespace threepp;


int main() {

    // 10x10 map
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
            if (!(value == 's' || value == 'g')) {
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

    std::shared_ptr<BufferGeometry> boxGeometry = BoxGeometry::create(0.95, 0.95, 0.1);
    boxGeometry->translate(0.5, 0.5, 0);

    std::shared_ptr<BufferGeometry> sphereGeometry = SphereGeometry::create(0.45);
    sphereGeometry->translate(0.5, 0.5, 0);

    std::unordered_map<char, Color> colors = {{'0', Color::black},
                                              {'1', Color::gray},
                                              {'s', Color::blue},
                                              {'g', Color::green},
                                              {'x', Color::yellow}};

    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = 0; j < size; j++) {
            auto value = data[i][j];

            auto material = MeshBasicMaterial::create({{"color", colors[value]}});
            auto geometry = (value == 's' || value == 'g') ? sphereGeometry : boxGeometry;
            auto box = Mesh::create(geometry, material);
            box->position.set(static_cast<float>(j) - size/2, static_cast<float>(i) - size/2, 0);
            scene.add(box);
        }
    }

    canvas.onWindowResize([&](WindowSize s){
        camera.left = -size * s.aspect() / 2;
        camera.right = size * s.aspect() / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(s);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
