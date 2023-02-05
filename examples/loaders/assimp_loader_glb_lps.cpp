
#include <threepp/core/Raycaster.hpp>
#include <threepp/geometries/DecalGeometry.hpp>
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

namespace {

    TextureLoader tl;

    std::shared_ptr<MeshPhongMaterial> decalMaterial() {

        auto decalMaterial = MeshPhongMaterial::create();
        decalMaterial->specular = 0x444444;
        decalMaterial->map = tl.loadTexture("data/textures/decal/decal-diffuse.png");
        decalMaterial->specularMap = tl.loadTexture("data/textures/decal/decal-normal.jpg");
        decalMaterial->normalScale = Vector2(1, 1);
        decalMaterial->shininess = 30;
        decalMaterial->transparent = true;
        decalMaterial->depthTest = true;
        decalMaterial->depthWrite = true;
        decalMaterial->polygonOffset = true;
        decalMaterial->polygonOffsetFactor = -4;
        decalMaterial->wireframe = false;

        return decalMaterial;
    }

    class MyMouseListener : public MouseListener {

    public:


        explicit MyMouseListener(Canvas &canvas) : canvas(canvas) {}

        std::optional<Vector2> mouseClick() {
            if (mouseDown) {
                mouseDown = false;
                return mouse;
            } else {
                return std::nullopt;
            }
        }

        void onMouseDown(int button, Vector2 pos) override {
            auto& size = canvas.getSize();
            mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
            mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
            mouseDown = true;
        }

    private:
        Canvas &canvas;
        bool mouseDown = false;
        Vector2 mouse{-1, -1};
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
    mat->as<MaterialWithMap>()->map = texLoader.loadTexture(folder / "Map-COL.jpg", false);
    mat->as<MaterialWithSpecularMap>()->specularMap = texLoader.loadTexture(folder / "Map-SPEC.jpg", false);
    mat->as<MaterialWithNormalMap>()->normalMap = texLoader.loadTexture(folder / "Infinite-Level_02_Tangent_SmoothUV.jpg", false);
    mat->as<MaterialWithSpecular>()->shininess = 25;
    mesh->materials_.front() = mat;

    scene->add(model);

    auto light = AmbientLight::create(0x443333, 1);
    scene->add(light);

    auto light2 = DirectionalLight::create(0xffddcc, 1);
    light2->position.set(1, 0.75, 0.5);
    scene->add(light2);

    auto light3 = DirectionalLight::create(0xccccff, 1);
    light3->position.set(-1, 0.75, -0.5);
    scene->add(light3);

    Raycaster raycaster;

    Vector3 position;
    Euler orientation;

    auto mouseListener = std::make_shared<MyMouseListener>(canvas);
    canvas.addMouseListener(mouseListener);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto mouseHelper = Mesh::create(BoxGeometry::create(), MeshBasicMaterial::create());
    mouseHelper->visible = false;
    scene->add(mouseHelper);

    std::vector<std::shared_ptr<Mesh>> decals;
    auto decalMat = decalMaterial();
    canvas.animate([&](float dt) {

        auto mouse = mouseListener->mouseClick();
        if (mouse) {

            raycaster.setFromCamera(*mouse, camera);
            auto intersects = raycaster.intersectObject(mesh, false);

            if (!intersects.empty()) {

                auto& i = intersects.front();
                Vector3 n = i.face->normal;

                mouseHelper->position.copy(i.point);
                n.transformDirection( *mesh->matrixWorld );
                n.multiplyScalar( 10 );
                n.add( i.point );
                mouseHelper->lookAt(n);

                Vector3 scale = Vector3::ONES * math::randomInRange(0.1f, 1.f);


                decalMat->color.randomize();

                auto m = Mesh::create(DecalGeometry::create(*mesh, i.point, mouseHelper->rotation, scale), decalMat);
                decals.emplace_back(m);
                scene->add(m);

                if (decals.size() > 10) {
                    scene->remove(decals.front());
                    decals.erase(decals.begin());
                }
            }

        }


        renderer.render(scene, camera);
    });
}
