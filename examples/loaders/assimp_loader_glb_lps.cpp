
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

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
    light3->position.set(- 1, 0.75, - 0.5);
    scene->add(light3);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        renderer.render(scene, camera);
    });
}
