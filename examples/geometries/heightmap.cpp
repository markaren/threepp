
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/Water.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include <fstream>
#include <future>
#include <iostream>

using namespace threepp;

namespace {

    std::vector<float> loadHeights() {

        std::ifstream file("data/models/terrain/aalesund.bin", std::ios::binary);

        file.seekg(0, std::ios_base::end);
        auto size = file.tellg();
        file.seekg(0, std::ios_base::beg);
        std::vector<uint16_t> raw(size / sizeof(uint16_t));
        file.read((char*) &raw[0], static_cast<std::streamsize>(size));
        file.close();

        std::vector<float> data(raw.size());
        for (unsigned i = 0; i < raw.size(); ++i) {
            data[i] = static_cast<float>(raw[i]) / 65535.f * 255;
        }

        return data;
    }

    auto createSky(const Vector3& lightPos) {
        auto sky = Sky::create();
        sky->scale.setScalar(10000);
        auto shaderUniforms = sky->material()->as<ShaderMaterial>()->uniforms;
        shaderUniforms->at("turbidity").value<float>() = 10;
        shaderUniforms->at("rayleigh").value<float>() = 1;
        shaderUniforms->at("mieCoefficient").value<float>() = 0.005;
        shaderUniforms->at("mieDirectionalG").value<float>() = 0.8;
        shaderUniforms->at("sunPosition").value<Vector3>().copy(lightPos);

        return sky;
    }

    auto createWater(const DirectionalLight& light, bool fog) {
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
        opt.sunDirection = light.position.clone().normalize();
        opt.sunColor = light.color;
        opt.waterColor = 0x001e0f;
        opt.fog = fog;

        auto waterGeometry = PlaneGeometry::create(5041, 5041);

        auto water = Water::create(waterGeometry, opt);
        water->rotateX(math::degToRad(-90));
        water->position.y = 5;

        return water;
    }

}// namespace

int main() {

    Canvas canvas("Heightmap");
    GLRenderer renderer{canvas.size()};
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    scene->fog = Fog(0xcccccc, 500, 3500);
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 5000);
    camera->position.set(500, 200, 500);

    OrbitControls controls{*camera, canvas};

    auto light = DirectionalLight::create();
    light->position.set(-1, 1, 0);
    light->intensity = 0.7;
    scene->add(light);

    auto sky = createSky(light->position);
    scene->add(sky);

    auto water = createWater(*light, scene->fog.has_value());
    scene->add(water);

    std::promise<std::shared_ptr<Material>> promise;
    auto future = promise.get_future();

    std::cout << "Loading terrain.." << std::endl;

    std::mutex m;
    utils::ThreadPool pool(2);
    pool.submit([&] {
        TextureLoader tl;
        auto texture = tl.load("data/textures/terrain/aalesund_terrain.png");
        auto material = MeshPhongMaterial::create({{"map", texture}});
        promise.set_value(material);

        std::lock_guard<std::mutex> lck(m);
        std::cout << "Material loaded" << std::endl;
    });

    pool.submit([&] {
        auto data = loadHeights();

        auto geometry = PlaneGeometry::create(5041, 5041, 1023, 1023);
        geometry->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));

        auto pos = geometry->getAttribute<float>("position");
        for (unsigned i = 0, j = 0, l = data.size(); i < l; ++i, j += 3) {
            pos->setY(i, data[i]);
        }

        {
            std::lock_guard<std::mutex> lck(m);
            std::cout << "Geometry loaded" << std::endl;
        }

        auto material = future.get();
        auto mesh = Mesh::create(geometry, material);

        std::cout << "Terrain loaded" << std::endl;

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
