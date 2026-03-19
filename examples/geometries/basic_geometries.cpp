#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createWireframe(const BufferGeometry& geometry) {
        auto line = LineSegments::create(WireframeGeometry::create(geometry));
        line->material()->as<LineBasicMaterial>()->color = Color::black;
        return line;
    }

    auto createMesh() {
        const auto geometry = BoxGeometry::create();
        const auto material = MeshBasicMaterial::create({{"side", Side::Double}});
        auto mesh = Mesh::create(geometry, material);
        mesh->add(createWireframe(*geometry));
        return mesh;
    }

    void setGeometry(Mesh& mesh, const std::shared_ptr<BufferGeometry>& geometry) {
        mesh.setGeometry(geometry);
        mesh.children[0]->removeFromParent();
        mesh.add(createWireframe(*geometry));
    }

}// namespace

int main() {

    Canvas canvas("Geometries", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    const auto scene = Scene::create();
    scene->background = Color::blue;
    const auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    const auto mesh = createMesh();
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    enum class GeomType { Box, Sphere, Cylinder, Plane };
    const char* geomNames[] = {"Box", "Sphere", "Cylinder", "Plane"};

    GeomType currentType = GeomType::Box;

    BoxGeometry::Params boxParams{};
    SphereGeometry::Params sphereParams{};
    CylinderGeometry::Params cylParams{};
    PlaneGeometry::Params planeParams{};

    bool paramsChanged = false;

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({340*ui.dpiScale(), 0}, ImGuiCond_Always);
        ImGui::Begin("Geometry");

        int typeIdx = static_cast<int>(currentType);
        if (ImGui::Combo("Type", &typeIdx, geomNames, 4)) {
            currentType = static_cast<GeomType>(typeIdx);
            paramsChanged = true;
        }

        ImGui::Separator();

        if (currentType == GeomType::Box) {
            ImGui::SliderFloat("width", &boxParams.width, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("height", &boxParams.height, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("depth", &boxParams.depth, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("widthSegments", reinterpret_cast<int*>(&boxParams.widthSegments), 1, 10);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&boxParams.heightSegments), 1, 10);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("depthSegments", reinterpret_cast<int*>(&boxParams.depthSegments), 1, 10);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
        } else if (currentType == GeomType::Sphere) {
            ImGui::SliderFloat("radius", &sphereParams.radius, 0.1f, 2);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("widthSegments", reinterpret_cast<int*>(&sphereParams.widthSegments), 2, 32);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&sphereParams.heightSegments), 2, 32);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("phiStart", &sphereParams.phiStart, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("phiLength", &sphereParams.phiLength, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("thetaStart", &sphereParams.thetaStart, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("thetaLength", &sphereParams.thetaLength, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
        } else if (currentType == GeomType::Cylinder) {
            ImGui::SliderFloat("radiusTop", &cylParams.radiusTop, 0.0f, 2);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("radiusBottom", &cylParams.radiusBottom, 0.0f, 2);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("height", &cylParams.height, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("radialSegments", reinterpret_cast<int*>(&cylParams.radialSegments), 3, 32);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&cylParams.heightSegments), 1, 16);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::Checkbox("openEnded", &cylParams.openEnded);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("thetaStart", &cylParams.thetaStart, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("thetaLength", &cylParams.thetaLength, 0, math::TWO_PI);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
        } else if (currentType == GeomType::Plane) {
            ImGui::SliderFloat("width", &planeParams.width, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderFloat("height", &planeParams.height, 0.1f, 4);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("widthSegments", reinterpret_cast<int*>(&planeParams.widthSegments), 1, 10);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
            ImGui::SliderInt("heightSegments", reinterpret_cast<int*>(&planeParams.heightSegments), 1, 10);
            paramsChanged = paramsChanged || ImGui::IsItemEdited();
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        mesh->rotation.y += 0.8f * dt;
        mesh->rotation.x += 0.5f * dt;

        renderer.render(*scene, *camera);
        ui.render();

        if (paramsChanged) {
            paramsChanged = false;
            if (currentType == GeomType::Box) {
                setGeometry(*mesh, BoxGeometry::create(boxParams));
            } else if (currentType == GeomType::Sphere) {
                setGeometry(*mesh, SphereGeometry::create(sphereParams));
            } else if (currentType == GeomType::Cylinder) {
                setGeometry(*mesh, CylinderGeometry::create(cylParams));
            } else if (currentType == GeomType::Plane) {
                setGeometry(*mesh, PlaneGeometry::create(planeParams));
            }
        }
    });
}