
#include "threepp/threepp.hpp"

#include "GameMap.hpp"
#include "pathfinding/algorithm/AStar.hpp"
#include "pathfinding/heuristics/ClosestHeuristic.hpp"
#include "pathfinding/heuristics/ClosestSquaredHeuristic.hpp"
#include "pathfinding/heuristics/ManhattanHeuristic.hpp"

#include <stdexcept>
#include <vector>

using namespace threepp;

namespace {

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

    std::unordered_map<char, int> colors =
            {{'0', Color::black},
             {'1', Color::gray},
             {'s', Color::blue},
             {'g', Color::green},
             {'x', Color::yellow}};

    std::string getMeshName(unsigned i, unsigned j) {
        return std::to_string(j) + "x" + std::to_string(i);
    }

    std::string getMeshName(Coordinate c) {
        return std::to_string(c.x) + "x" + std::to_string(c.y);
    }

}// namespace

int main() {

    AStar algorithm(std::make_unique<GameMap>(data), std::make_unique<ClosestHeuristic>());
    algorithm.setAllowDiagMovement(true);

    std::optional<Coordinate> start;
    std::optional<Coordinate> target;

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


    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = 0; j < size; j++) {
            auto value = data[i][j];

            auto material = MeshBasicMaterial::create({{"color", colors[value]}});
            auto mesh = Mesh::create(boxGeometry, material);
            mesh->name = getMeshName(i, j);
            mesh->layers.enable(1);
            mesh->position.set(static_cast<float>(j) - size / 2, static_cast<float>(i) - size / 2, 0);
            scene.add(mesh);
        }
    }

    auto resetBlockColors = [&] {
        for (unsigned i = 0; i < size; i++) {
            for (unsigned j = 0; j < size; j++) {
                auto value = data[i][j];
                auto mesh = scene.getObjectByName(getMeshName(i, j));
                mesh->material()->as<MaterialWithColor>()->color.setHex(colors[value]);
            }
        }
    };

    Raycaster raycaster;
    raycaster.layers.set(1); // ignore grid
    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseUpListener mouseListener([&](int button, Vector2 pos) {
        if (start && target) return;

        const auto s = canvas.size();
        mouse.x = (pos.x / static_cast<float>(s.width)) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(s.height)) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        auto intersects = raycaster.intersectObjects(scene.children);
        if (!intersects.empty()) {
            auto point = intersects.front().point;
            point.floor() += size / 2;
            if (!start) {
                start = Coordinate(point.x, point.y);
                scene.getObjectByName(getMeshName(*start))->material()->as<MaterialWithColor>()->color = Color::green;
            } else if (!target) {
                target = Coordinate(point.x, point.y);
                scene.getObjectByName(getMeshName(*target))->material()->as<MaterialWithColor>()->color = Color::green;
            }

            if (start && target) {

                auto path = algorithm.findPath(*start, *target);

                if (path) {
                    std::cout << "Found path between " << *start << " and " << *target << ", length=" << path->length() << std::endl;
                    for (unsigned i = 1; i < path->length() - 1; i++) {
                        auto c = (*path)[i];

                        auto obj = scene.getObjectByName(getMeshName(c));
                        obj->material()->as<MaterialWithColor>()->color.setHex(Color::green);
                    }
                    renderer.invokeLater([&] {
                        start = std::nullopt;
                        target = std::nullopt;
                        resetBlockColors(); }, 2);
                } else {
                    std::cerr << "Unable to find path between " << *start << " and " << *target << std::endl;
                    renderer.invokeLater([&] {
                        start = std::nullopt;
                        target = std::nullopt;
                        resetBlockColors(); }, 1);
                }
            }
        }
    });
    canvas.addMouseListener(mouseListener);

    canvas.onWindowResize([&](WindowSize s) {
        camera.left = -size * s.aspect() / 2;
        camera.right = size * s.aspect() / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(s);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
