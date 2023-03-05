#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    struct Params {
        float width = 1;
        float height = 1;
        float depth = 1;

        int widthSegements = 1;
        int heightSegements = 1;
        int depthSegements = 1;
    };

    std::shared_ptr<BufferGeometry> createGeometry(const Params& params) {

        return BoxGeometry::create(params.width, params.height, params.depth, params.widthSegements, params.heightSegements, params.depthSegements);
    }

    std::shared_ptr<LineSegments> createWireframe(const BufferGeometry& geometry) {

        auto line = LineSegments::create(WireframeGeometry::create(geometry));
        line->material()->as<LineBasicMaterial>()->alphaTest = false;
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
        ImGui::Begin("BoxGeometry");
        ImGui::SliderFloat("width", &params.width, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("height", &params.height, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("depth", &params.depth, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("widthSegements", &params.widthSegements, 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("heightSegments", &params.heightSegements, 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("depthSegments", &params.depthSegements, 1, 10);
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
