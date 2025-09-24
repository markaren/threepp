#include "threepp/math/Lut.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>

#include "functions.hpp"
#include "optimization/DifferentialEvolution.hpp"
#include "optimization/ParticleSwarmOptimization.hpp"

using namespace threepp;

namespace {

    const float maxHeight = 2;
    const float gridSize = 4;

    std::pair<float, float> normalizeAndApplyLut(BufferGeometry& geometry) {

        const auto pos = geometry.getAttribute<float>("position");

        std::vector<float> yValues(pos->count());
        for (auto i = 0; i < pos->count(); i++) {
            yValues[i] = pos->getY(i);
        }

        const auto [min, max] = std::minmax_element(yValues.begin(), yValues.end());
        for (auto i = 0; i < pos->count(); i++) {
            pos->setY(i, math::mapLinear(pos->getY(i), *min, *max, 0, maxHeight));
        }

        Lut lut("rainbow", 256 * 256);
        auto colors = std::vector<float>(pos->count() * 3);

        for (auto i = 0, j = 0; i < pos->count(); i++, j += 3) {

            const auto y = pos->getY(i);

            Color c = lut.getColor(math::mapLinear(y, 0, maxHeight, 0, 1));
            c.toArray(colors, j);
        }
        if (!geometry.hasAttribute("color")) {
            geometry.setAttribute("color", FloatBufferAttribute::create(colors, 3));
        } else {
            geometry.getAttribute<float>("color")->array() = colors;
            geometry.getAttribute<float>("color")->needsUpdate();
        }

        return {*min, *max};
    }

    void applyFunc(BufferGeometry& geometry, const Problem& func) {

        const auto pos = geometry.getAttribute<float>("position");

        for (auto i = 0; i < pos->count(); i++) {
            const auto x = math::mapLinear(pos->getX(i), -gridSize / 2, gridSize / 2, 0, 1);
            const auto z = math::mapLinear(pos->getZ(i), -gridSize / 2, gridSize / 2, 0, 1);
            pos->setY(i, func.eval({x, z}));
        }

        pos->needsUpdate();
    }

}// namespace

int main() {

    std::unordered_map<std::string, std::unique_ptr<Optimizer>> algorithms;
    algorithms.emplace("DifferentialEvolution", std::make_unique<DifferentialEvolution>(50, 0.2f, 0.9f));
    algorithms.emplace("ParticleSwarmOptimization", std::make_unique<ParticleSwarmOptimization>(30, 0.9f, 1.41f, 1.41f));
    std::string selectedAlgorithm = algorithms.begin()->first;

    std::string selectedFunction;
    std::unordered_map<std::string, std::unique_ptr<Problem>> functions;
    functions.emplace("Rosenbrock", std::make_unique<Rosenbruck>());
    functions.emplace("Ackleys", std::make_unique<Ackleys>());
    functions.emplace("HolderTable", std::make_unique<HolderTable>());
    functions.emplace("Rastrigin", std::make_unique<Rastrigin>());
    functions.emplace("Himmelblaus", std::make_unique<Himmelblaus>());
    functions.emplace("Beale", std::make_unique<Beale>());
    functions.emplace("ThreeHumpCamel", std::make_unique<ThreeHumpCamel>());
    functions.emplace("EggHolder", std::make_unique<EggHolder>());
    functions.emplace("GoldsteinPrice", std::make_unique<GoldsteinPrice>());
    functions.emplace("Booth", std::make_unique<Booth>());
    functions.emplace("Sphere", std::make_unique<SphereFunction>());

    Lut::addColorMap("rainbow", {{0.f, 0x0000ff}, {0.001f, 0x00ffff}, {0.02f, 0xffff00}, {0.2f, 0xff0000}, {1.f, Color::darkred}});

    Canvas canvas("Optimization", {{"aa", 6}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect(), 0.01f, 1000);
    camera.position.set(5, 4, -5);

    OrbitControls controls(camera, canvas);

    auto planeGeometry = PlaneGeometry::create(gridSize, gridSize, 200, 200);
    auto planeGeometry2 = PlaneGeometry::create(gridSize, gridSize, 50, 50);
    planeGeometry->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));
    planeGeometry2->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));

    auto plane = Mesh::create(planeGeometry, MeshBasicMaterial::create({{"vertexColors", true}}));
    auto wireframe = Mesh::create(planeGeometry2, MeshBasicMaterial::create({{"wireframe", true}}));
    wireframe->material()->depthTest = false;
    wireframe->material()->opacity = 0.25;
    wireframe->material()->transparent = true;

    plane->add(wireframe);
    scene.add(plane);

    auto solutionMesh = InstancedMesh::create(
            CylinderGeometry::create(0.01, 0.01, maxHeight, 32),
            MeshBasicMaterial::create({{"color", Color::greenyellow}}), 10);
    solutionMesh->setCount(0);
    scene.add(solutionMesh);

    auto searchSpace = InstancedMesh::create(
            SphereGeometry::create(0.05),
            MeshBasicMaterial::create({{"color", Color::black}}), 500);
    searchSpace->setCount(algorithms[selectedAlgorithm]->size());
    scene.add(searchSpace);


    std::pair<float, float> minmax;
    auto changeFunction = [&](const std::string& name) {
        selectedFunction = name;

        Problem& func = *functions[selectedFunction];

        applyFunc(*planeGeometry, func);
        applyFunc(*planeGeometry2, func);

        minmax = normalizeAndApplyLut(*planeGeometry);
        normalizeAndApplyLut(*planeGeometry2);

        algorithms[selectedAlgorithm]->init(func);
        const auto solutions = func.solutions();
        solutionMesh->setCount(solutions.size());
        Matrix4 tmp;
        for (int i = 0; i < solutions.size(); i++) {
            auto solution = func.normalize(solutions[i]);
            float solutionX = math::mapLinear(solution[0], 0, 1, -gridSize / 2, gridSize / 2);
            float solutionZ = math::mapLinear(solution[1], 0, 1, -gridSize / 2, gridSize / 2);

            solutionMesh->setMatrixAt(i, tmp.setPosition(solutionX, maxHeight / 2, solutionZ));
        }
        solutionMesh->instanceMatrix()->needsUpdate();
        solutionMesh->computeBoundingSphere();
    };
    changeFunction(functions.begin()->first);


    auto changeAlgorithm = [&](const std::string& name) {
        selectedAlgorithm = name;

        auto& algorithm = algorithms[name];
        searchSpace->setCount(algorithm->size());
        algorithm->init(*functions[selectedFunction]);
    };


    float searchSpeed = 0.7;
    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);

        ImGui::Begin("Optimization");
        ImGui::SliderFloat("Search speed", &searchSpeed, 0.1, 1);
        if (ImGui::BeginCombo("Algorithm", selectedAlgorithm.c_str())) {
            for (const auto& [name, algorithm] : algorithms) {
                if (ImGui::Selectable(name.c_str())) {

                    changeAlgorithm(name);
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Functions", selectedFunction.c_str())) {
            for (const auto& [name, functor] : functions) {
                if (ImGui::Selectable(name.c_str())) {

                    changeFunction(name);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    capture.preventScrollEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        std::optional<int> key;
        if (evt.key == Key::NUM_1) {
            key = 0;
        } else if (evt.key == Key::NUM_2) {
            key = 1;
        } else if (evt.key == Key::NUM_3) {
            key = 2;
        } else if (evt.key == Key::NUM_4) {
            key = 3;
        } else if (evt.key == Key::NUM_5) {
            key = 4;
        } else if (evt.key == Key::NUM_6) {
            key = 5;
        } else if (evt.key == Key::NUM_7) {
            key = 6;
        } else if (evt.key == Key::NUM_8) {
            key = 7;
        } else if (evt.key == Key::NUM_9) {
            key = 8;
        } else if (evt.key == Key::NUM_0) {
            key = 9;
        }

        if (key) {
            if (*key >= functions.size()) return;

            auto it = functions.begin();
            std::advance(it, *key);

            changeFunction(it->first);
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    Matrix4 tmp;
    Clock clock;
    canvas.animate([&] {
        renderer.render(scene, camera);
        ui.render();

        if (clock.getElapsedTime() > (1 - searchSpeed)) {
            auto& algorithm = algorithms[selectedAlgorithm];
            algorithm->step(*functions[selectedFunction]);

            for (auto i = 0; i < algorithm->size(); i++) {
                const auto& candidate = algorithm->getCandiateAt(i).data();
                float x = math::mapLinear(candidate[0], 0, 1, -gridSize / 2, gridSize / 2);
                float z = math::mapLinear(candidate[1], 0, 1, -gridSize / 2, gridSize / 2);
                float y = math::mapLinear(functions[selectedFunction]->eval({candidate[0], candidate[1]}), minmax.first, minmax.second, 0, maxHeight);

                searchSpace->setMatrixAt(i, tmp.setPosition(x, y, z));
            }

            searchSpace->instanceMatrix()->needsUpdate();
            searchSpace->computeBoundingSphere();
            clock.start();
        }
    });
}
