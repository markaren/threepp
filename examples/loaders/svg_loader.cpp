#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/threepp.hpp"

#ifdef HAS_IMGUI
#include "threepp/extras/imgui/imgui_context.hpp"
#endif

using namespace threepp;

namespace {

#ifdef HAS_IMGUI

    struct MyUI: public imgui_context {

    public:
        explicit MyUI(void* ptr): imgui_context(ptr) {}

        [[nodiscard]] bool newSelection() const {
            return lastSelectedIndex != selectedIndex;
        }

        [[nodiscard]] std::string selected() const {
            return names[selectedIndex];
        }

    protected:
        void onRender() override {

            lastSelectedIndex = selectedIndex;

            ImGui::SetNextWindowPos({}, 0, {});
            ImGui::SetNextWindowSize({230, 0}, 0);

            ImGui::Begin("SVGLoader");

            if (ImGui::BeginCombo("SVG file", names[selectedIndex].c_str())) {
                for (int i = 0; i < names.size(); ++i) {
                    const bool isSelected = (selectedIndex == i);
                    if (ImGui::Selectable(names[i].c_str(), isSelected)) {
                        selectedIndex = i;
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::End();
        }

    private:
        int lastSelectedIndex = -1;
        int selectedIndex = 0;
        std::vector<std::string> names{"tiger.svg", "threejs.svg", "hexagon.svg"};
    };

#endif

    auto loadSvg(const std::string& name = "tiger.svg") {

        SVGLoader loader;
        auto shapes = loader.load("data/models/svg/" + name);

        auto svg = Group::create();

        for (const auto& shape : shapes) {

            auto& path = shape.path;

            auto material = MeshBasicMaterial::create(
                    {{"color", path.color},
                     {"opacity", shape.style.fillOpacity},
                     {"transparent", shape.style.fillOpacity != 1},
                     {"side", DoubleSide},
                     {"depthWrite", false}});

            auto geometry = ShapeGeometry::create(path.toShapes(true, false));
            auto mesh = Mesh::create(geometry, material);
            mesh->name = shape.id;

            svg->add(mesh);

//            if (shape.style.stroke) {
//                auto strokeMaterial = MeshBasicMaterial::create(
//                        {{"color", *shape.style.stroke},
//                         {"opacity", shape.style.strokeOpacity},
//                         {"transparent", shape.style.strokeOpacity != 1},
//                         {"side", DoubleSide},
////                         {"wireframe", true},
//                         {"depthWrite", false}});
//
//                const auto& subPaths = path.getSubPaths();
//                for (const auto& subPath : subPaths) {
//
//                    auto strokeGeometry = SVGLoader::pointsToStroke(subPath->getPoints(), shape.style);
//
//                    if (geometry) {
//
//                        auto strokeMesh = Mesh::create(strokeGeometry, strokeMaterial);
//                        svg->add(strokeMesh);
//                    }
//
//                }
//            }
        }

        svg->scale.multiplyScalar(0.25f);
        svg->position.x = -70;
        svg->position.y = 70;
        svg->scale.y *= -1;

        return svg;
    }

}// namespace

int main() {

    Canvas canvas("SVGLoader", {{"antialiasing", 4}});

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 100;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto gridHelper = GridHelper::create(160, 10);
    gridHelper->rotation.x = math::PI / 2;
    scene->add(gridHelper);

    std::shared_ptr<Object3D> svg;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    OrbitControls controls{camera, canvas};

#ifdef HAS_IMGUI
    MyUI ui(canvas.window_ptr());

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);
#else
    svg = loadSvg();
    scene->add(svg);
#endif

    canvas.animate([&]() {
        renderer.render(scene, camera);

#ifdef HAS_IMGUI

        if (ui.newSelection()) {
            if (svg) {
                svg->removeFromParent();
            }

            svg = loadSvg(ui.selected());
            scene->add(svg);
        }

        ui.render();
#endif
    });
}
