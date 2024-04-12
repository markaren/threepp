#include "threepp/math/Lut.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>
#include <iostream>

#include "optimization/DifferentialEvolution.hpp"
#include "optimization/ParticleSwarmOptimization.hpp"

using namespace threepp;

namespace {

    const float maxHeight = 2;
    const float gridSize = 4;

    struct Ackleys: Problem {

        Ackleys(): Problem(2, {-5, 5}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return -20 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + z * z))) - std::exp(0.5 * (std::cos(2 * math::PI * x) + std::cos(2 * math::PI * z))) + 20 + std::exp(1);
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{0, 0}};
        }
    };

    struct Rosenbruck: Problem {

        Rosenbruck(): Problem(2, {{-2, 2}, {-1, 3}}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            float a = 1, b = 100;
            return ((a - x) * (a - x)) + b * ((z - (x * x)) * (z - (x * x)));
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{1, 1}};
        }
    };

    struct HolderTable: Problem {

        HolderTable(): Problem(2, {-10, 10}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return -std::abs(std::sin(x) * std::cos(z) * std::exp(std::abs(1 - std::sqrt(x * x + z * z) / math::PI)));
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{8.055, 9.66}, {-8.055, 9.66}, {8.055, -9.66}, {-8.055, -9.66}};
        }
    };

    struct Rastrigin: Problem {

        Rastrigin(): Problem(2, {-5.12, 5.12}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return 20 + x * x - 10 * std::cos(2 * math::PI * x) + z * z - 10 * std::cos(2 * math::PI * z);
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{0, 0}};
        }
    };

    struct Himmelblaus: Problem {

        Himmelblaus(): Problem(2, {-5, 5}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return std::pow(x * x + z - 11, 2) + std::pow(x + z * z - 7, 2);
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{3, 2}, {-2.805118, 3.131312}, {-3.779310, -3.283186}, {3.584428, -1.848126}};
        }
    };

    struct Beale: Problem {

        Beale(): Problem(2, {-4.5, 4.5}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return std::pow(1.5 - x + x * z, 2) + std::pow(2.25 - x + x * z * z, 2) + std::pow(2.625 - x + x * z * z * z, 2);
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{3, 0.5}};
        }
    };

    struct ThreeHumpCamel: Problem {

        ThreeHumpCamel(): Problem(2, {-5, 5}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return 2 * x * x - 1.05 * x * x * x * x + x * x * x * x * x * x / 6 + x * z + z * z;
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{0, 0}};
        }
    };

    struct Booth: Problem {

        Booth(): Problem(2, {-10, 10}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return (x + 2 * z - 7) * (x + 2 * z - 7) + (2 * x + z - 5) * (2 * x + z - 5);
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{1, 3}};
        }
    };

    struct EggHolder: Problem {

        EggHolder(): Problem(2, {-512, 512}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return -(z + 47) * std::sin(std::sqrt(std::abs(z + x / 2 + 47))) - x * std::sin(std::sqrt(std::abs(x - (z + 47))));
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{512, 404.2319}};
        }
    };

    struct GoldsteinPrice: Problem {

        GoldsteinPrice(): Problem(2, {-2, 2}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return (1 + (x + z + 1) * (x + z + 1) * (19 - 14 * x + 3 * x * x - 14 * z + 6 * x * z + 3 * z * z)) *
                   (30 + (2 * x - 3 * z) * (2 * x - 3 * z) * (18 - 32 * x + 12 * x * x + 48 * z - 36 * x * z + 27 * z * z));
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{0, -1}};
        }
    };

    struct SphereFunction: Problem {

        SphereFunction(): Problem(2, {-5.12, 5.12}) {}

        [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

            float x = candidate[0];
            float z = candidate[1];

            return x * x + z * z;
        }

        [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
            return {{0, 0}};
        }
    };

    std::pair<float, float> normalizeAndApplyLut(BufferGeometry& geometry) {

        auto pos = geometry.getAttribute<float>("position");

        std::vector<float> yValues;
        yValues.reserve(pos->count());
        for (auto i = 0; i < pos->count(); i++) {
            yValues.emplace_back(pos->getY(i));
        }

        const auto minmax = std::minmax_element(yValues.begin(), yValues.end());
        for (auto i = 0; i < pos->count(); i++) {
            pos->setY(i, math::mapLinear(pos->getY(i), *minmax.first, *minmax.second, 0, maxHeight));
        }

        Lut lut("rainbow", 256 * 256);
        auto colors = std::vector<float>(pos->count() * 3);

        for (auto i = 0, j = 0; i < pos->count(); i++, j += 3) {

            float y = pos->getY(i);

            Color c = lut.getColor(math::mapLinear(y, 0, maxHeight, 0, 1));
            c.toArray(colors, j);
        }
        if (!geometry.hasAttribute("color")) {
            geometry.setAttribute("color", FloatBufferAttribute::create(colors, 3));
        } else {
            geometry.getAttribute<float>("color")->array() = colors;
            geometry.getAttribute<float>("color")->needsUpdate();
        }

        return {*minmax.first, *minmax.second};
    }

    void applyFunc(BufferGeometry& geometry, const Problem& func) {

        auto pos = geometry.getAttribute<float>("position");

        for (auto i = 0; i < pos->count(); i++) {
            auto x = math::mapLinear(pos->getX(i), -gridSize / 2, gridSize / 2, 0, 1);
            auto z = math::mapLinear(pos->getZ(i), -gridSize / 2, gridSize / 2, 0, 1);
            pos->setY(i, func.eval({x, z}));
        }

        pos->needsUpdate();
    }

}// namespace

int main() {

//    auto algorithm = std::make_unique<DifferentialEvolution>(50, 0.2f, 0.9f);
    auto algorithm = std::make_unique<ParticleSwarmOptimization>(30, 0.9f, 1.41f, 1.41f);

    Lut::addColorMap("rainbow", {{0.f, 0x0000ff}, {0.001f, 0x00ffff}, {0.02f, 0xffff00}, {0.2f, 0xff0000}, {1.f, Color::darkred}});

    Canvas canvas("Lut", {{"aa", 6}});
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

    std::pair<float, float> minmax;
    auto changeFunction = [&](const std::string& name) {
        selectedFunction = name;
        Problem& func = *functions[name];

        applyFunc(*planeGeometry, func);
        applyFunc(*planeGeometry2, func);

        minmax = normalizeAndApplyLut(*planeGeometry);
        normalizeAndApplyLut(*planeGeometry2);

        algorithm->init(func);
        auto solutions = func.solutions();
        solutionMesh->setCount(solutions.size());
        for (int i = 0; i < solutions.size(); i++) {
            auto solution = func.normalize(solutions[i]);
            float solutionX = math::mapLinear(solution[0], 0, 1, -gridSize / 2, gridSize / 2);
            float solutionZ = math::mapLinear(solution[1], 0, 1, -gridSize / 2, gridSize / 2);

            solutionMesh->setMatrixAt(i, Matrix4().setPosition(solutionX, maxHeight / 2, solutionZ));
        }
        solutionMesh->instanceMatrix()->needsUpdate();
        solutionMesh->computeBoundingSphere();
    };
    changeFunction(functions.begin()->first);

    auto instancedMesh = InstancedMesh::create(
            SphereGeometry::create(0.05),
            MeshBasicMaterial::create({{"color", Color::black}}), algorithm->size());
    scene.add(instancedMesh);

    float searchSpeed = 0.7;
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({260, 0}, 0);

        ImGui::Begin("Optimization");
        ImGui::SliderFloat("Search speed", &searchSpeed, 0.1, 1);
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


    Clock clock;
    Matrix4 m;
    canvas.animate([&] {
        renderer.render(scene, camera);
        ui.render();

        if (clock.getElapsedTime() > (1-searchSpeed)) {
            algorithm->step(*functions[selectedFunction]);

            for (auto i = 0; i < algorithm->size(); i++) {
                const auto& candidate = algorithm->getCandiateAt(i).data();
                float x = math::mapLinear(candidate[0], 0, 1, -gridSize / 2, gridSize / 2);
                float z = math::mapLinear(candidate[1], 0, 1, -gridSize / 2, gridSize / 2);
                float y = math::mapLinear(functions[selectedFunction]->eval({candidate[0], candidate[1]}), minmax.first, minmax.second, 0, maxHeight);

                instancedMesh->setMatrixAt(i, m.setPosition(x, y, z));
            }

            instancedMesh->instanceMatrix()->needsUpdate();
            instancedMesh->computeBoundingSphere();
            clock.start();
        }
    });
}
