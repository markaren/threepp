// https://threejs.org/examples/?q=clipping#webgl_clipping

#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <cmath>

using namespace threepp;

int main() {

    Canvas canvas("Clipping", {{"aa", 6}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(36, canvas.size().aspect(), 0.25, 16);
    camera->position.set(0, 1.3, 3);

    scene->add(AmbientLight::create(0x505050));

    auto spotLight = SpotLight::create(0xffffff);
    spotLight->angle = math::PI / 5;
    spotLight->penumbra = 0.2;
    spotLight->position.set(2, 3, 3);
    spotLight->castShadow = true;
    spotLight->shadow->camera->near = 3;
    spotLight->shadow->camera->far = 10;
    spotLight->shadow->mapSize.x = 1024;
    spotLight->shadow->mapSize.y = 1024;
    scene->add(spotLight);

    auto dirLight = DirectionalLight::create(0x55505a, 1.f);
    dirLight->position.set(0, 3, 0);
    dirLight->castShadow = true;
    dirLight->shadow->camera->near = 1;
    dirLight->shadow->camera->far = 10;

    dirLight->shadow->camera->as<OrthographicCamera>()->right = 1;
    dirLight->shadow->camera->as<OrthographicCamera>()->left = -1;
    dirLight->shadow->camera->as<OrthographicCamera>()->top = 1;
    dirLight->shadow->camera->as<OrthographicCamera>()->bottom = -1;

    dirLight->shadow->mapSize.x = 1024;
    dirLight->shadow->mapSize.y = 1024;
    scene->add(dirLight);

    // Geometry

    auto material = MeshPhongMaterial::create({{"color", 0x80ee10},
                                               {"shininess", 100.f},
                                               {"side", Side::Double}});

    // ***** Clipping setup (material): *****
    auto& localPlane = material->clippingPlanes.emplace_back(Vector3(0, -1, 0), 0.8f);
    material->clipShadows = true;

    auto geometry = TorusKnotGeometry::create(0.4, 0.08, 95, 20);

    auto object = Mesh::create(geometry, material);
    object->castShadow = true;
    scene->add(object);

    auto ground = Mesh::create(
            PlaneGeometry::create(9, 9, 1, 1),
            MeshPhongMaterial::create({{"color", 0xa0adaf}, {"shininess", 150.f}}));

    ground->rotation.x = -math::PI / 2;// rotates X/Y to X/Z
    ground->receiveShadow = true;
    scene->add(ground);

    // ***** Clipping setup (renderer): *****
    Plane globalPlane(Vector3(-1, 0, 0), 0.1);
    renderer.localClippingEnabled = true;

    bool globalClipping = !renderer.clippingPlanes.empty();
    auto ui = ImguiFunctionalContext(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);

        ImGui::Begin("Local clipping");
        ImGui::Checkbox("Enabled", &renderer.localClippingEnabled);
        ImGui::Checkbox("Shadows", &material->clipShadows);
        ImGui::SliderFloat("Plane", &localPlane.constant, 0.3f, 1.25f);
        ImGui::End();

        ImGui::SetNextWindowPos({230, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);

        ImGui::Begin("Global clipping");
        ImGui::Checkbox("Enabled", &globalClipping);
        if (ImGui::IsItemEdited()) {
            if (globalClipping) {
                renderer.clippingPlanes.emplace_back(globalPlane);
            } else {
                renderer.clippingPlanes.clear();
            }
        }
        ImGui::SliderFloat("Plane", &globalPlane.constant, -0.4f, 3.f);
        if (!renderer.clippingPlanes.empty() && ImGui::IsItemEdited()) {
            renderer.clippingPlanes[0].copy(globalPlane);
        }
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    // Controls

    OrbitControls controls(*camera, canvas);
    controls.target.set(0, 1, 0);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        auto time = clock.getElapsedTime();

        object->position.y = 0.8f;
        object->rotation.x = time * 0.5f;
        object->rotation.y = time * 0.2f;
        object->scale.setScalar(std::cos(time) * 0.125f + 0.875f);

        renderer.render(*scene, *camera);

        ui.render();
    });
}
