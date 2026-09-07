
#include "renderer_factory.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/Text.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    std::vector<std::string> fonts{
        "gentilis_bold.typeface.json", "gentilis_regular.typeface.json", "helvetiker_bold.typeface.json",
        "helvetiker_regular.typeface.json", "optimer_bold.typeface.json", "optimer_regular.typeface.json",
        "Roboto-Regular.ttf", "Roboto-Bold.ttf"
    };

    std::filesystem::path getFontPath(const std::string& fontName) {
        std::filesystem::path fontPath{std::string(DATA_FOLDER) + "/fonts"};
        if (fontName.ends_with(".typeface.json")) {
            return fontPath / "typeface" / fontName;
        } else {
            return fontPath / "truetype"  / fontName;
        }
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
        light->intensity = 1.5f;
        light->position.set(15, 5, 15);
        light->lookAt(Vector3::ZEROS());
        light->castShadow = true;
        auto shadowCamera = light->shadow->camera->as<OrthographicCamera>();
        shadowCamera->left = shadowCamera->bottom = -20;
        shadowCamera->right = shadowCamera->top = 20;
        scene.add(light);

        auto pointLight = PointLight::create();
        pointLight->intensity = 0.3f;
        pointLight->position.set(0, 2, 10);
        scene.add(pointLight);
    }

}// namespace

int main() {

    std::string displayText = "threepp!";


    Canvas canvas("Fonts", {{"aa", 8}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    auto scene = Scene::create();
    scene->background = Color::black;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(0, 5, 40);

    createAndAddLights(*scene);

    OrbitControls controls{*camera, canvas};

    FontLoader loader;
    auto font = loader.load(getFontPath(fonts.front()));

    constexpr float textSize = 10;
    std::shared_ptr<Text3D> textMesh3d;
    std::shared_ptr<Text2D> textMesh2d;

    if (font) {
        const auto material = MeshPhongMaterial::create();
        material->side = Side::Double;
        material->color = Color::orange;

        textMesh3d = Text3D::create(ExtrudeTextGeometry::Options(*font, textSize, 1), displayText, material);
        textMesh3d->castShadow = true;

        textMesh2d = Text2D::create(TextGeometry::Options(*font, textSize), displayText, material);
        textMesh2d->position.z = 2;

        textMesh3d->geometry()->center();
        textMesh2d->geometry()->center();

        scene->add(textMesh3d);
        scene->add(textMesh2d);
    }

    scene->add(createPlane());

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });


    // Font picker rides in the shared renderer-settings window (which also
    // exposes tone map + shadow controls for the GL renderer).
    int selectedIndex = 4;
    int lastSelectedIndex = selectedIndex;
    RendererSettingsUi ui(canvas, *renderer, [&] {
        if (ImGui::BeginCombo("Select Font", fonts[selectedIndex].c_str())) {
            for (unsigned i = 0; i < fonts.size(); ++i) {
                const auto isSelected = (selectedIndex == i);
                if (ImGui::Selectable(fonts[i].c_str(), isSelected)) {
                    selectedIndex = i;
                }
            }
            ImGui::EndCombo();
        }
    }, "Font");

    canvas.animate([&] {
        renderer->render(*scene, *camera);

        ui.render();

        if (lastSelectedIndex != selectedIndex) {
            lastSelectedIndex = selectedIndex;
            font = loader.load(getFontPath(fonts[selectedIndex]));
            if (font) {
                textMesh3d->setText(displayText, ExtrudeTextGeometry::Options(*font, textSize, 1));
                textMesh2d->setText(displayText, TextGeometry::Options(*font, textSize));

                textMesh3d->geometry()->center();
                textMesh2d->geometry()->center();
            }
        }
    });
}
