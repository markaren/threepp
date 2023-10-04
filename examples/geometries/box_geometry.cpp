#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createWireframe(const BufferGeometry& geometry) {

        auto material = LineBasicMaterial::create({{"color", Color::black}});
        return LineSegments::create(WireframeGeometry::create(geometry), material);
    }

    auto createMesh(const BoxGeometry::Params& params) {

        auto geometry = BoxGeometry::create(params);
        auto material = MeshBasicMaterial::create();

        Mesh mesh(geometry, material);
        mesh.add(createWireframe(*geometry));

        return mesh;
    }

    void updateGroupGeometry(Mesh& mesh, const BoxGeometry::Params& params) {

        auto g = BoxGeometry::create(params);
        mesh.setGeometry(g);

        mesh.children[0]->removeFromParent();
        mesh.add(createWireframe(*g));
    }


}// namespace

int main() {

    Canvas canvas("BoxGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::blue;
    PerspectiveCamera camera(60, canvas.aspect(), 0.1f, 100);
    camera.position.z = 5;

    BoxGeometry::Params params{};

    auto mesh = createMesh(params);
    scene.add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    bool paramsChanged = false;
    auto ui = ImguiFunctionalContext(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("BoxGeometry");
        ImGui::SliderFloat("width", &params.width, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("height", &params.height, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderFloat("depth", &params.depth, 0.1, 2);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("widthSegments", reinterpret_cast<int*>(&params.widthSegments), 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&params.heightSegments), 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::SliderInt("depthSegments", reinterpret_cast<int*>(&params.depthSegments), 1, 10);
        paramsChanged = paramsChanged || ImGui::IsItemEdited();
        ImGui::End();
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        mesh.rotation.y += 0.8f * dt;
        mesh.rotation.x += 0.5f * dt;

        renderer.render(scene, camera);

        ui.render();

        if (paramsChanged) {
            paramsChanged = false;
            updateGroupGeometry(mesh, params);
        }
    });
}
