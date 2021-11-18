
#include "threepp/objects/Water.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(55, canvas.getAspect(), 1, 2000);
    camera->position.set(30, 30, 100);

    OrbitControls controls{camera, canvas};
    controls.maxPolarAngle = math::PI * 0.495f;
    controls.target.set(0, 10, 0);
    controls.minDistance = 40;
    controls.maxDistance = 200;
    controls.update();

    auto light = DirectionalLight::create(0xffffff);
    light->position.set(-1,-1,-1);
    scene->add(light);

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.setClearColor(Color::aliceblue);
    renderer.setSize(canvas.getSize());

    const auto sphereGeometry = SphereGeometry::create(30);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x0000ff);
    sphereMaterial->wireframe = true;
    sphereMaterial->wireframeLinewidth = 10;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    scene->add(sphere);

    TextureLoader textureLoader{};
    auto texture = textureLoader.loadTexture("textures/waternormals.jpg");
    texture->wrapS = RepeatWrapping;
    texture->wrapT = RepeatWrapping;

    Water::Options opt;
    opt.alpha = 0.9f;
    opt.waterNormals = texture;
    opt.distortionScale = 3.7f;
    opt.sunDirection = Vector3{};
    opt.sunColor = 0xffffff;
    opt.waterColor = 0x001e0f;
    opt.fog = scene->fog.has_value();

    auto waterGeometry = PlaneGeometry::create(10000, 10000);

    auto water = Water::create(waterGeometry, opt);
    water->rotateX(math::degToRad(-90));
    water->position.setY(-1);
    scene->add(water);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    float t = 0;
    sphere->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        sphere->position.y = std::sin(t) * 20 + 5;
        sphere->rotation.x = t * 0.05f;
        sphere->rotation.z = t * 0.051f;

        water->material()->uniforms->operator[]("time").setValue(t);

        renderer.render(scene, camera);

        t += dt;
    });
}
