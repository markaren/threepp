
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
        decalMaterial->map = tl.loadTexture("data/textures/decal/decal-diffuse.png");
        decalMaterial->normalMap = tl.loadTexture("data/textures/decal/decal-normal.jpg");
        decalMaterial->normalScale = Vector2(1, 1);
        decalMaterial->shininess = 30;
        decalMaterial->depthTest = true;
        decalMaterial->depthWrite = false;
        decalMaterial->transparent = true;
        decalMaterial->polygonOffset = true;
        decalMaterial->polygonOffsetFactor = -4;

        return decalMaterial;
    }

    class MyMouseListener : public MouseListener {

    public:
        Vector2 mouse{-1, -1};

        explicit MyMouseListener(Canvas &canvas) : canvas(canvas) {}

        bool mouseClick() {
            if (mouseDown) {
                mouseDown = false;
                return true;
            } else {
                return false;
            }
        }

        void onMouseDown(int button, const Vector2 &pos) override {
            mouseDown = true;
            updateMousePos(pos);
        }

        void onMouseMove(const Vector2 &pos) override {
            updateMousePos(pos);
        }

    private:
        Canvas &canvas;
        bool mouseDown = false;

        void updateMousePos(Vector2 pos) {
            auto &size = canvas.getSize();
            mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
            mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
        }
    };

}// namespace

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(0, 1, 10);

    OrbitControls controls{camera, canvas};

    TextureLoader texLoader;
    AssimpLoader loader;
    std::filesystem::path folder = "data/models/gltf/LeePerrySmith";
    auto model = loader.load(folder / "LeePerrySmith.glb");
    auto mesh = model->children[0]->children[0]->as<Mesh>();
    auto mat = MeshPhongMaterial::create();
    mat->map = texLoader.loadTexture(folder / "Map-COL.jpg", false);
    mat->specularMap = texLoader.loadTexture(folder / "Map-SPEC.jpg", false);
    mat->normalMap = texLoader.loadTexture(folder / "Infinite-Level_02_Tangent_SmoothUV.jpg", false);
    mat->shininess = 25;
    mesh->materials_.front() = mat;

    scene->add(model);

    auto light = AmbientLight::create(0x443333, 1.f);
    scene->add(light);

    auto light2 = DirectionalLight::create(0xffddcc, 1.f);
    light2->position.set(1, 0.75, 0.5);
    scene->add(light2);

    auto light3 = DirectionalLight::create(0xccccff, 1.f);
    light3->position.set(-1, 0.75, -0.5);
    scene->add(light3);

    auto lineGeometry = BufferGeometry::create();
    lineGeometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 1}, 3));
    auto line = Line::create(lineGeometry);
    scene->add(line);

    MyMouseListener mouseListener(canvas);
    canvas.addMouseListener(&mouseListener);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Matrix4 mouseHelper;
    Vector3 position;
    Euler orientation;

    auto decalMat = decalMaterial();

    Raycaster raycaster;
    canvas.animate([&](float dt) {
        raycaster.setFromCamera(mouseListener.mouse, camera);
        auto intersects = raycaster.intersectObject(mesh, false);

        bool click = mouseListener.mouseClick();

        if (!intersects.empty()) {

            auto &i = intersects.front();
            Vector3 n = i.face->normal;

            mouseHelper.setPosition(i.point);
            n.transformDirection(*mesh->matrixWorld);
            n.multiplyScalar(10);
            n.add(i.point);
            mouseHelper.lookAt(position.setFromMatrixPosition(mouseHelper), n, Vector3::Z);
            orientation.setFromRotationMatrix(mouseHelper);

            line->position.copy(position);
            line->lookAt(n);

            if (click) {

                Vector3 scale = Vector3::ONES * math::randomInRange(0.4f, 1.f);

                auto mat = decalMat->clone();
                mat->color.randomize();
                orientation.z = math::PI * math::randomInRange(0.f, 1.f);
                auto m = Mesh::create(DecalGeometry::create(*mesh, position, orientation, scale), mat);
                scene->add(m);
            }
        }

        renderer.render(scene, camera);
    });
}
