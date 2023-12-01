#include "threepp/geometries/EdgesGeometry.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/loaders/FontLoader.hpp"
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

            ImGui::Begin("Font");

            if (ImGui::BeginCombo("Select Font", names[selectedIndex].c_str())) {
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
        int selectedIndex = 4;
        std::vector<std::string> names{
                "gentilis_bold", "gentilis_regular", "helvetiker_bold",
                "helvetiker_regular", "optimer_bold", "optimer_regular"};
    };

#endif

    auto createTextGeometry(const Font& font, const std::string& text, int size = 10) {
        TextGeometry::Options opts(font, size);
        opts.height = 1;
        auto geometry = TextGeometry::create(text, opts);
        geometry->center();

        return geometry;
    }

    auto createPlane() {

        auto planeMaterial = MeshPhongMaterial::create();
        planeMaterial->color = Color::gray;
        auto plane = Mesh::create(PlaneGeometry::create(1000, 1000), planeMaterial);
        plane->position.y = -8;
        plane->rotateX(math::degToRad(-90));
        plane->receiveShadow = true;

        return plane;
    }

    auto createAndAddLights(Scene& scene) {

        auto light = DirectionalLight::create();
        light->position.set(10, 5, 10);
        light->lookAt(Vector3::ZEROS());
        light->castShadow = true;
        auto shadowCamera = light->shadow->camera->as<OrthographicCamera>();
        shadowCamera->left = shadowCamera->bottom = -20;
        shadowCamera->right = shadowCamera->top = 20;
        scene.add(light);

        auto pointLight = PointLight::create();
        pointLight->intensity = 0.2f;
        pointLight->position.set(0, 2, 10);
        scene.add(pointLight);
    }

}// namespace

int main() {

    std::string displayText = "threepp!";
    std::filesystem::path fontPath{"data/fonts"};

    Canvas canvas("Fonts", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFCSoft;

    auto scene = Scene::create();
    scene->background = Color::black;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(0, 5, 40);

    createAndAddLights(*scene);

    OrbitControls controls{*camera, canvas};

    FontLoader loader;
    auto font = loader.load(fontPath / "optimer_bold.typeface.json");

    std::shared_ptr<Mesh> textMesh;
    if (font) {
        auto material = MeshPhongMaterial::create();
        material->color = Color::orange;

        auto geometry = createTextGeometry(*font, displayText, 10);

        textMesh = Mesh::create(geometry, material);
        textMesh->castShadow = true;
        scene->add(textMesh);
    }

    scene->add(createPlane());

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

#ifdef HAS_IMGUI
    MyUI ui(canvas.windowPtr());
#endif

    canvas.animate([&]() {
        renderer.render(*scene, *camera);

#ifdef HAS_IMGUI
        ui.render();

        if (ui.newSelection()) {
            font = loader.load(fontPath / std::string(ui.selected() + ".typeface.json"));
            if (font) {
                auto geometry = createTextGeometry(*font, displayText);
                textMesh->setGeometry(geometry);
            }
        }
#endif
    });
}
