#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/threepp.hpp"

#ifdef HAS_IMGUI
#include "threepp/extras/imgui/ImguiContext.hpp"
#endif

using namespace threepp;

namespace {

#ifdef HAS_IMGUI

    struct MyUI: public ImguiContext {

    public:
        explicit MyUI(void* ptr): ImguiContext(ptr) {}

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
            ImGui::SetNextWindowSize({250, 0}, 0);

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
        std::vector<std::string> names{
                "tiger.svg", "threejs.svg", "OpenBridge_Heading.svg",
                "hexagon.svg", "lineJoinsAndCaps.svg", "energy.svg",
                "style-css-inside-defs.svg", "ordering.svg",
                "styles.svg", "units.svg", "ellipseTransform.svg"};
    };

#endif

    auto loadSvg(const std::string& name = "tiger.svg") {

        SVGLoader loader;
        auto svgData = loader.load("data/models/svg/" + name);

        auto svg = Group::create();
        svg->name = std::filesystem::path(name).stem().string();
        svg->scale.multiplyScalar(0.25f);
        svg->position.x = -70;
        svg->position.y = 70;
        svg->scale.y *= -1;

        for (const auto& data : svgData) {

            auto fillColor = data.style.fill;
            if (fillColor && *fillColor != "none") {

                auto material = MeshBasicMaterial::create(
                        {{"color", data.path.color},
                         {"opacity", data.style.fillOpacity},
                         {"transparent", true},
                         {"side", Side::Double},
                         {"depthWrite", false}});

                const auto shapes = SVGLoader::createShapes(data);

                auto geometry = ShapeGeometry::create(shapes);
                auto mesh = Mesh::create(geometry, material);
                mesh->name = data.style.id;
                mesh->visible = data.style.visibility;
                svg->add(mesh);
            }

            auto strokeColor = data.style.stroke;
            if (strokeColor && *strokeColor != "none") {
                auto strokeMaterial = MeshBasicMaterial::create(
                        {{"color", Color().setStyle(*data.style.stroke)},
                         {"opacity", data.style.strokeOpacity},
                         {"transparent", true},
                         {"side", Side::Double},
                         {"depthWrite", false}});

                for (const auto& subPath : data.path.subPaths) {

                    auto strokeGeometry = SVGLoader::pointsToStroke(subPath->getPoints(), data.style);

                    if (strokeGeometry) {

                        auto strokeMesh = Mesh::create(strokeGeometry, strokeMaterial);
                        svg->add(strokeMesh);
                    }
                }
            }
        }

        return svg;
    }

    void rotateAround(Object3D& o, const Vector3& axis, Vector3 point, float angle) {

        Matrix4 rotation;
        Matrix4 origin;
        Matrix4 temp;

        origin.setPosition(point);

        rotation.makeRotationAxis(axis, angle);
        rotation.premultiply(origin).multiply(temp.copy(origin).invert());

        o.applyMatrix4(rotation);
    }

}// namespace

int main() {

    Canvas canvas("SVGLoader", {{"antialiasing", 4}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 100;

    auto gridHelper = GridHelper::create(160, 10);
    gridHelper->rotation.x = math::PI / 2;
    scene->add(gridHelper);

    std::shared_ptr<Object3D> svg;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    OrbitControls controls{*camera, canvas};

#ifdef HAS_IMGUI
    MyUI ui(canvas.windowPtr());

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    capture.preventScrollEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);
#else
    svg = loadSvg();
    scene->add(svg);
#endif

    Box3 bb;
    auto box3Helper = Box3Helper::create(bb, Color::grey);
    box3Helper->material()->opacity = 0.2f;
    box3Helper->material()->transparent = true;
    scene->add(box3Helper);

    Vector3 center;

    Clock clock;
    canvas.animate([&]() {
        renderer.render(*scene, *camera);

#ifdef HAS_IMGUI

        if (ui.newSelection()) {
            if (svg) {
                svg->removeFromParent();
            }

            svg = loadSvg(ui.selected());
            bb.setFromObject(*svg);


            if (svg->name == "OpenBridge_Heading") {
                bb.getCenter(center);
                svg->worldToLocal(center);
                center.sub({-1.25, -4.375, 0});
            }

            scene->add(svg);
        }

        float dt = clock.getDelta();

        if (svg->name == "OpenBridge_Heading") {

            auto cog = svg->getObjectByName("COG_2");
            rotateAround(*cog, {0, 0, 1}, center, -0.3f * dt);

            auto hdg = svg->getObjectByName("HDG_2");
            rotateAround(*hdg, {0, 0, 1}, center, 0.5f * dt);

            auto sp = svg->getObjectByName("Shape");
            rotateAround(*sp, {0, 0, 1}, center, 0.1f * dt);
        }

        ui.render();
#endif
    });
}
