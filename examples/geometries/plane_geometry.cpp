#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createWireframe(const BufferGeometry& geometry) {

        auto material = LineBasicMaterial::create({{"color", Color::black}});
        return LineSegments::create(WireframeGeometry::create(geometry), material);
    }

    void updateGroupGeometry(Mesh& mesh, const PlaneGeometry::Params& params) {

        auto g = PlaneGeometry::create(params);
        mesh.setGeometry(g);

        mesh.children[0]->removeFromParent();
        mesh.add(createWireframe(*g));
    }

    auto createMesh(const PlaneGeometry::Params& params) {

        auto geometry = PlaneGeometry::create(params);
        auto material = MeshBasicMaterial::create();
        material->side = Side::Double;

        auto mesh = Mesh::create(geometry, material);
        mesh->add(createWireframe(*geometry));

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("PlaneGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.z = 5;

    PlaneGeometry::Params params{};

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
        ImGui::Begin("PlaneGeometry");
        ImGui::SliderFloat("width", &params.width, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("height", &params.height, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("widthSegments", reinterpret_cast<int*>(&params.widthSegments), 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&params.heightSegments), 1, 10);
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
