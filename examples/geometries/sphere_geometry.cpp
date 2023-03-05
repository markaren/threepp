#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    struct Params {
        float radius = 1;
        int widthSegments = 16;
        int heightSegments = 12;
        float phiStart = 0;
        float phiLength = math::TWO_PI;
        float thetaStart = 0;
        float thetaLength = math::PI;
    };

    std::shared_ptr<BufferGeometry> createGeometry(const Params& params) {

        return SphereGeometry::create(params.radius, params.widthSegments, params.heightSegments, params.phiStart, params.phiLength, params.thetaStart, params.thetaLength);
    }

    std::shared_ptr<LineSegments> createWireframe(const BufferGeometry& geometry) {

        auto line = LineSegments::create(WireframeGeometry::create(geometry));
        line->material()->as<LineBasicMaterial>()->color = Color::black;
        return line;
    }

    void updateGroupGeometry(Mesh& mesh, const Params& params) {

        auto g = createGeometry(params);
        mesh.setGeometry(g);

        mesh.children[0]->removeFromParent();
        mesh.add(createWireframe(*g));
    }

    std::shared_ptr<Mesh> createMesh(const Params& params) {

        auto geometry = createGeometry(params);
        auto material = MeshBasicMaterial::create();
        material->side = DoubleSide;

        auto mesh = Mesh::create(geometry, material);
        mesh->add(createWireframe(*geometry));

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    Params params{};

    auto mesh = createMesh(params);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    bool paramsChanged = false;
    auto ui = imgui_functional_context(canvas.window_ptr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("SphereGeometry");
        ImGui::SliderFloat("radius", &params.radius, 0.1f, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("widthSegments", &params.widthSegments, 2, 32);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("heightSegments", &params.heightSegments, 2, 32);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("phiStart", &params.phiStart, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("phiLength", &params.phiLength, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("thetaStart", &params.thetaStart, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("thetaLength", &params.thetaLength, 0, math::TWO_PI);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::End();
    });

    canvas.animate([&](float dt) {
        mesh->rotation.y += 0.8f * dt;
        mesh->rotation.x += 0.5f * dt;

        renderer.render(scene, camera);

        ui.render();

        if (paramsChanged) {
            paramsChanged = false;
            updateGroupGeometry(*mesh, params);
        }
    });
}
