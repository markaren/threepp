
#include "threepp/extras/imgui/ImguiContext.hpp"
#include <threepp/core/Raycaster.hpp>
#include <threepp/geometries/DecalGeometry.hpp>
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/threepp.hpp>


using namespace threepp;

namespace {

    auto decalMaterial() {

        TextureLoader tl;
        auto decalMaterial = MeshPhongMaterial::create();
        decalMaterial->specular = 0x444444;
        decalMaterial->map = tl.load(std::string(DATA_FOLDER) + "/textures/decal/decal-diffuse.png");
        decalMaterial->normalMap = tl.load(std::string(DATA_FOLDER) + "/textures/decal/decal-normal.jpg");
        decalMaterial->normalScale.set(1, 1);
        decalMaterial->shininess = 30;
        decalMaterial->depthTest = true;
        decalMaterial->depthWrite = false;
        decalMaterial->transparent = true;
        decalMaterial->polygonOffset = true;
        decalMaterial->polygonOffsetFactor = -4;

        return decalMaterial;
    }

    class MyMouseListener: public MouseListener {

    public:
        Vector2 mouse{-Infinity<float>, -Infinity<float>};

        explicit MyMouseListener(Canvas& canvas): canvas(canvas) {}

        bool mouseClick() {
            if (mouseDown) {
                mouseDown = false;
                return true;
            }

            return false;
        }

        void onMouseDown(int button, const Vector2& pos) override {
            if (button == 0) {// left mousebutton
                mouseDown = true;
            }
        }

        void onMouseMove(const Vector2& pos) override {
            updateMousePos(pos);
        }

    private:
        Canvas& canvas;
        bool mouseDown = false;

        void updateMousePos(Vector2 pos) {
            const auto size = canvas.size();
            mouse.x = (pos.x / static_cast<float>(size.width())) * 2 - 1;
            mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
        }
    };

    struct MyGui: ImguiContext {

        bool clear = false;

        explicit MyGui(const Canvas& canvas): ImguiContext(canvas) {}

        void onRender() override {

            ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
            ImGui::SetNextWindowSize({100, 0}, 0);

            ImGui::Begin("Options");
            ImGui::Checkbox("Clear", &clear);

            ImGui::End();
        }
    };

    void addLights(Scene& scene) {

        const auto light = AmbientLight::create(0x443333, 0.8f);
        scene.add(light);

        const auto light2 = DirectionalLight::create(0xffddcc, 1.f);
        light2->position.set(1, 0.75, 0.5);
        scene.add(light2);

        const auto light3 = DirectionalLight::create(0xccccff, 1.f);
        light3->position.set(-1, 0.75, -0.5);
        scene.add(light3);
    }

}// namespace

int main() {

    Canvas canvas{"Decals", {{"aa", 8}}};
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 1, 10);

    addLights(*scene);

    OrbitControls controls{*camera, canvas};

    TextureLoader tl;
    AssimpLoader loader;
    std::filesystem::path folder = std::string(DATA_FOLDER) + "/models/gltf/LeePerrySmith";
    auto model = loader.load(folder / "LeePerrySmith.glb");
    Mesh* mesh = nullptr;
    model->traverseType<Mesh>([&](Mesh& _) {
        mesh = &_;
        const auto mat = MeshPhongMaterial::create({{
                {"map", tl.load(folder / "Map-COL.jpg", false)},
                {"specularMap", tl.load(folder / "Map-SPEC.jpg", false)},
                {"normalMap", tl.load(folder / "Infinite-Level_02_Tangent_SmoothUV.jpg", false)},
                {"shininess", 25.f},
        }});
        mesh->setMaterial(mat);
    });
    scene->add(model);

    auto lineGeometry = BufferGeometry::create();
    lineGeometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 1}, 3));
    auto line = Line::create(lineGeometry);
    scene->add(line);

    MyMouseListener mouseListener(canvas);
    canvas.addMouseListener(mouseListener);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    MyGui ui(canvas);
    std::vector<Mesh*> decals;

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Matrix4 mouseHelper;
    Vector3 position;
    Euler orientation;

    auto decalMat = decalMaterial();

    Raycaster raycaster;
    canvas.animate([&]() {
        raycaster.setFromCamera(mouseListener.mouse, *camera);
        const auto intersects = raycaster.intersectObject(*mesh, false);

        bool click = mouseListener.mouseClick();

        if (!intersects.empty()) {

            auto& i = intersects.front();
            Vector3 n = i.face->normal;

            mouseHelper.setPosition(i.point);
            n.transformDirection(*mesh->matrixWorld);
            n.multiplyScalar(10);
            n.add(i.point);
            mouseHelper.lookAt(position.setFromMatrixPosition(mouseHelper), n, Vector3::Z());
            orientation.setFromRotationMatrix(mouseHelper);

            line->position.copy(position);
            line->lookAt(n);

            if (click) {

                const auto scale = Vector3::ONES() * math::randFloat(0.6f, 1.2f);

                const auto mat = decalMat->clone<MeshPhongMaterial>();
                mat->color.randomize();
                orientation.z = math::PI * math::randFloat();
                auto m = Mesh::create(DecalGeometry::create(*mesh, position, orientation, scale), mat);
                decals.emplace_back(m.get());
                scene->add(m);
            }
        }

        renderer.render(*scene, *camera);

        if (ui.clear) {
            for (auto decal : decals) {
                decal->removeFromParent();
            }
            decals.clear();
            ui.clear = false;
        }
        ui.render();
    });
}
