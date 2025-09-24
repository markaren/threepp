#include "threepp/math/Lut.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>

using namespace threepp;

namespace {

    constexpr float maxHeight = 2;
    constexpr float gridSize = 4;

    void normalizeAndApplyLut(BufferGeometry& geometry) {

        const auto pos = geometry.getAttribute<float>("position");

        std::vector<float> yValues;
        yValues.reserve(pos->count());
        for (auto i = 0; i < pos->count(); i++) {
            yValues.emplace_back(pos->getY(i));
        }

        const auto minmax = std::minmax_element(yValues.begin(), yValues.end());
        for (auto i = 0; i < pos->count(); i++) {
            pos->setY(i, math::mapLinear(pos->getY(i), *minmax.first, *minmax.second, 0, maxHeight));
        }

        Lut::addColorMap("rainbow", {{0.f, 0x0000ff}, {0.001f, 0x00ffff}, {0.02f, 0xffff00}, {0.2f, 0xff0000}, {1.f, Color::darkred}});
        const Lut lut("rainbow", 256 * 256);
        auto colors = std::vector<float>(pos->count() * 3);

        for (auto i = 0, j = 0; i < pos->count(); i++, j += 3) {

            const float y = pos->getY(i);

            Color c = lut.getColor(math::mapLinear(y, 0, maxHeight, 0, 1));
            c.toArray(colors, j);
        }
        if (!geometry.hasAttribute("color")) {
            geometry.setAttribute("color", FloatBufferAttribute::create(colors, 3));
        } else {
            geometry.getAttribute<float>("color")->array() = colors;
            geometry.getAttribute<float>("color")->needsUpdate();
        }
    }

    float rosenbrock(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -2, 2);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -1, 3);

        constexpr float a = 1, b = 100;
        return ((a - x) * (a - x)) + b * ((z - (x * x)) * (z - (x * x)));
    }

    float ackleys(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -5, 5);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -5, 5);

        return -20 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + z * z))) - std::exp(0.5 * (std::cos(2 * math::PI * x) + std::cos(2 * math::PI * z))) + 20 + std::exp(1);
    }

    float holderTable(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -10, 10);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -10, 10);

        return -std::abs(std::sin(x) * std::cos(z) * std::exp(std::abs(1 - std::sqrt(x * x + z * z) / math::PI)));
    }

    float rastrigin(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -5.12, 5.12);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -5.12, 5.12);

        return 20 + x * x - 10 * std::cos(2 * math::PI * x) + z * z - 10 * std::cos(2 * math::PI * z);
    }

    float beale(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -4.5, 4.5);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -4.5, 4.5);

        return std::pow(1.5 - x + x * z, 2) + std::pow(2.25 - x + x * z * z, 2) + std::pow(2.625 - x + x * z * z * z, 2);
    }

    float goldsteinPrice(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -2, 2);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -2, 2);

        return (1 + std::pow(x + z + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * z + 6 * x * z + 3 * z * z)) *
               (30 + std::pow(2 * x - 3 * z, 2) * (18 - 32 * x + 12 * x * x + 48 * z - 36 * x * z + 27 * z * z));
    }

    float booth(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -10, 10);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -10, 10);

        return std::pow(x + 2 * z - 7, 2) + std::pow(2 * x + z - 5, 2);
    }

    float levino13(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -10, 10);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -10, 10);

        return std::pow(x, 2) + std::pow(z, 2) + 2 * std::pow(std::sin(2 * math::PI * x), 2) + 2 * std::pow(std::sin(2 * math::PI * z), 2);
    }

    float himmelblaus(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -5, 5);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -5, 5);

        return std::pow(x * x + z - 11, 2) + std::pow(x + z * z - 7, 2);
    }

    float eggholder(float x, float z) {

        x = math::mapLinear(x, -gridSize / 2, gridSize / 2, -512, 512);
        z = math::mapLinear(z, -gridSize / 2, gridSize / 2, -512, 512);

        return -(z + 47) * std::sin(std::sqrt(std::abs(z + x / 2 + 47))) - x * std::sin(std::sqrt(std::abs(x - (z + 47))));
    }

    void applyFunc(BufferGeometry& geometry, const std::function<float(float, float)>& func) {

        const auto pos = geometry.getAttribute<float>("position");

        for (auto i = 0; i < pos->count(); i++) {
            const auto x = pos->getX(i);
            const auto z = pos->getZ(i);
            pos->setY(i, func(x, z));
        }

        pos->needsUpdate();
    }

}// namespace

int main() {

    Canvas canvas("Lut", {{"aa", 6}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect());
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

    std::string selectedFunction;
    std::unordered_map<std::string, std::function<float(float, float)>> functions{
            {"Rosenbrock", rosenbrock},
            {"Ackleys", ackleys},
            {"Holder Table", holderTable},
            {"Rastrigin", rastrigin},
            {"Beale", beale},
            {"Goldstein Price", goldsteinPrice},
            {"Booth", booth},
            {"Levino 13", levino13},
            {"Himmelblaus", himmelblaus},
            {"Eggholder", eggholder}};

    auto changeFunction = [&](const std::string& name) {
        selectedFunction = name;
        applyFunc(*planeGeometry, functions[name]);
        applyFunc(*planeGeometry2, functions[name]);

        normalizeAndApplyLut(*planeGeometry);
        normalizeAndApplyLut(*planeGeometry2);
    };

    changeFunction(functions.begin()->first);

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);

        ImGui::Begin("Lut");
        if (ImGui::BeginCombo("Functions", selectedFunction.c_str())) {
            for (const auto& name : functions | std::views::keys) {
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
        std::optional<size_t> key;
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

    canvas.animate([&] {
        renderer.render(scene, camera);
        ui.render();
    });
}
