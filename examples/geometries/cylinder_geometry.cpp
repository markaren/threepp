#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<LineSegments> createWireframe(const BufferGeometry& geometry) {

        auto line = LineSegments::create(WireframeGeometry::create(geometry));
        line->material()->as<LineBasicMaterial>()->color = Color::black;
        return line;
    }

    void updateGroupGeometry(Mesh& mesh, const CylinderGeometry::Params& params) {

        auto g = CylinderGeometry::create(params);
        mesh.setGeometry(g);

        mesh.children[0]->removeFromParent();
        mesh.add(createWireframe(*g));
    }

    std::shared_ptr<Mesh> createMesh(const CylinderGeometry::Params& params) {

        auto geometry = CylinderGeometry::create(params);
        auto material = MeshBasicMaterial::create({{"side", Side::Double}});

        auto mesh = Mesh::create(geometry, material);
        mesh->add(createWireframe(*geometry));

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("CylinderGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.z = 5;

    CylinderGeometry::Params params{};

    auto mesh = createMesh(params);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    bool paramsChanged = false;
    auto ui = ImguiFunctionalContext(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("CylinderGeometry");
        ImGui::SliderFloat("radiusTop", &params.radiusTop, 0.1f, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("radiusBottom", &params.radiusBottom, 0.1f, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("height", &params.height, 0.1f, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("radialSegments", reinterpret_cast<int*>(&params.radialSegments), 1, 32);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&params.heightSegments), 1, 32);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::Checkbox("openEnded", &params.openEnded);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("thetaStart", &params.thetaStart, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("thetaLength", &params.thetaLength, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::End();
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        mesh->rotation.y += 0.8f * dt;
        mesh->rotation.x += 0.5f * dt;

        renderer.render(*scene, *camera);

        ui.render();

        if (paramsChanged) {
            paramsChanged = false;
            updateGroupGeometry(*mesh, params);
        }
    });
}
