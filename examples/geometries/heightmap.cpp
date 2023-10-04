
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/Water.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include <fstream>
#include <future>

using namespace threepp;

std::vector<float> loadHeights() {

    std::ifstream file("data/models/terrain/aalesund.bin", std::ios::binary);

    std::vector<float> data;
    // Read the data
    while (true) {
        uint16_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(uint16_t));

        if (file.fail()) {
            break;
        }

        data.emplace_back(static_cast<float>(value) / 65535.f * 255);
    }

    return data;
}

int main() {

    Canvas canvas("Heightmap");
    GLRenderer renderer{canvas.size()};
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 5000);
    camera->position.set(500, 200, 500);

    OrbitControls controls{*camera, canvas};

    auto light = DirectionalLight::create();
    light->position.set(-1, 1, 0);
    light->intensity = 0.5;
    scene->add(light);

    auto sky = Sky::create();
    sky->scale.setScalar(10000);
    auto shaderUniforms = sky->material()->as<ShaderMaterial>()->uniforms;
    shaderUniforms->at("turbidity").value<float>() = 10;
    shaderUniforms->at("rayleigh").value<float>() = 1;
    shaderUniforms->at("mieCoefficient").value<float>() = 0.005;
    shaderUniforms->at("mieDirectionalG").value<float>() = 0.8;
    shaderUniforms->at("sunPosition").value<Vector3>().copy(light->position);
    scene->add(sky);

    TextureLoader textureLoader{};
    auto texture = textureLoader.load("data/textures/waternormals.jpg");
    texture->wrapS = TextureWrapping::Repeat;
    texture->wrapT = TextureWrapping::Repeat;

    Water::Options opt;
    opt.textureHeight = 512;
    opt.textureWidth = 512;
    opt.alpha = 0.8f;
    opt.waterNormals = texture;
    opt.distortionScale = 3.7f;
    opt.sunDirection = light->position.clone().normalize();
    opt.sunColor = light->color;
    opt.waterColor = 0x001e0f;
    opt.fog = scene->fog.has_value();

    auto waterGeometry = PlaneGeometry::create(5041, 5041);

    auto water = Water::create(waterGeometry, opt);
    water->rotateX(math::degToRad(-90));
    water->position.y = 5;
    scene->add(water);

    std::promise<std::shared_ptr<Material>> promise;
    auto future = promise.get_future();

    utils::ThreadPool pool(2);
    pool.submit([&] {
        TextureLoader tl;
        auto texture = tl.load("data/textures/terrain/aalesund_terrain.png");
        auto material = MeshPhongMaterial::create({{"map", texture}});
        promise.set_value(material);
    });
    pool.submit([&] {
        auto data = loadHeights();

        auto geometry = PlaneGeometry::create(5041, 5041, 1023, 1023);
        geometry->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));

        auto pos = geometry->getAttribute<float>("position");
        for (unsigned i = 0, j = 0, l = data.size(); i < l; ++i, j += 3) {
            pos->setY(i, data[i]);
        }

        auto material = future.get();
        auto mesh = Mesh::create(geometry, material);

        canvas.invokeLater([&, mesh] {
            scene->add(mesh);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float t = clock.getElapsedTime();
        water->material()->as<ShaderMaterial>()->uniforms->at("time").setValue(t);

        renderer.render(*scene, *camera);
    });
}
