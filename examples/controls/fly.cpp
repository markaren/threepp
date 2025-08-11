
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createEarth(float radius) {
        TextureLoader loader;
        auto materialNormalMap = MeshPhongMaterial::create({{"specular", 0x333333},
                                                            {"shininess", 10.f},
                                                            {"map", loader.load(std::string(DATA_FOLDER) + "/textures/planets/earth_atmos_2048.jpg")},
                                                            {"specularMap", loader.load(std::string(DATA_FOLDER) + "/textures/planets/earth_specular_2048.jpg")},
                                                            {"normalMap", loader.load(std::string(DATA_FOLDER) + "/textures/planets/earth_normal_2048.jpg")},
                                                            {"normalScale", Vector2(0.85f, -0.85f)}});

        auto geometry = SphereGeometry::create(radius, 100, 50);

        auto earth = Mesh::create(geometry, materialNormalMap);
        earth->rotation.y = 0;
        earth->rotation.z = 0.41;

        return earth;
    }

    auto createMoon(float radius, float moonScale) {
        TextureLoader loader;
        auto materialMoon = MeshPhongMaterial::create({{
                {"map", loader.load(std::string(DATA_FOLDER) + "/textures/planets/moon_1024.jpg")},
        }});

        auto geometry = SphereGeometry::create(radius, 100, 50);

        auto moon = Mesh::create(geometry, materialMoon);
        moon->position.set(radius * 5, 0, 0);
        moon->scale *= moonScale;

        return moon;
    }


    auto createClouds(float radius) {
        TextureLoader loader;
        auto materialMoon = MeshLambertMaterial::create({{{"map", loader.load(std::string(DATA_FOLDER) + "/textures/planets/earth_clouds_1024.png")},
                                                          {"transparent", true}}});

        auto geometry = SphereGeometry::create(radius, 100, 50);

        auto clouds = Mesh::create(geometry, materialMoon);
        clouds->scale *= 1.005;
        clouds->rotation.z = 0.41;

        return clouds;
    }

    auto createStars(float radius) {

        auto starGroup = Group::create();

        std::vector<std::shared_ptr<BufferGeometry>> starsGeometry{
                BufferGeometry::create(),
                BufferGeometry::create()};

        std::vector<float> vertices1;
        std::vector<float> vertices2;

        Vector3 vertex;

        for (unsigned i = 0; i < 250; i++) {

            vertex.x = math::randFloat() * 2 - 1;
            vertex.y = math::randFloat() * 2 - 1;
            vertex.z = math::randFloat() * 2 - 1;
            vertex.multiplyScalar(radius);

            vertices1.insert(vertices1.end(), {vertex.x, vertex.y, vertex.z});
        }

        for (unsigned i = 0; i < 1500; i++) {

            vertex.x = math::randFloat() * 2 - 1;
            vertex.y = math::randFloat() * 2 - 1;
            vertex.z = math::randFloat() * 2 - 1;
            vertex.multiplyScalar(radius);

            vertices2.insert(vertices2.end(), {vertex.x, vertex.y, vertex.z});
        }

        starsGeometry[0]->setAttribute("position", FloatBufferAttribute::create(vertices1, 3));
        starsGeometry[1]->setAttribute("position", FloatBufferAttribute::create(vertices2, 3));

        std::vector<std::shared_ptr<PointsMaterial>> starsMaterials{
                PointsMaterial::create({{"color", 0x9c9c9c}, {"size", 2.f}, {"sizeAttenuation", false}}),
                PointsMaterial::create({{"color", 0x9c9c9c}, {"size", 1.f}, {"sizeAttenuation", false}}),
                PointsMaterial::create({{"color", 0x7c7c7c}, {"size", 2.f}, {"sizeAttenuation", false}}),
                PointsMaterial::create({{"color", 0x838383}, {"size", 1.f}, {"sizeAttenuation", false}}),
                PointsMaterial::create({{"color", 0x5a5a5a}, {"size", 2.f}, {"sizeAttenuation", false}}),
                PointsMaterial::create({{"color", 0x5a5a5a}, {"size", 1.f}, {"sizeAttenuation", false}}),
        };

        for (unsigned i = 10; i < 30; i++) {

            auto stars = Points::create(starsGeometry[i % 2], starsMaterials[i % 6]);

            stars->rotation.x = math::randFloat() * 6;
            stars->rotation.y = math::randFloat() * 6;
            stars->rotation.z = math::randFloat() * 6;
            stars->scale.setScalar(static_cast<float>(i) * 10);

            stars->matrixAutoUpdate = false;
            stars->updateMatrix();

            starGroup->add(stars);
        }

        return starGroup;
    }

}// namespace

int main() {

    float radius = 6731;
    float moonScale = 0.23;

    Canvas canvas{"FlyControls", {{"aa", 6}}};
    GLRenderer renderer{canvas.size()};

    Scene scene;
    scene.fog = FogExp2(0x000000, 0.00000025);

    PerspectiveCamera camera(25, canvas.aspect(), 50, 1e7);
    camera.position.z = radius * 5;

    auto dirLight = DirectionalLight::create(0xffffff);
    dirLight->position.set(-1, 0, 1).normalize();
    scene.add(dirLight);

    auto earth = createEarth(radius);
    scene.add(earth);

    auto moon = createMoon(radius, moonScale);
    scene.add(moon);

    auto stars = createStars(radius);
    scene.add(stars);

    auto clouds = createClouds(radius);
    scene.add(clouds);

    FlyControls controls{camera, canvas};
    controls.movementSpeed = 1000;
    controls.rollSpeed = math::PI / 24;
    controls.autoForward = false;
    controls.dragToLook = false;

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    Vector3 dMoonVec;
    float rotationSpeed = 0.02;
    canvas.animate([&] {
        float delta = clock.getDelta();

        earth->rotation.y += rotationSpeed * delta;
        clouds->rotation.y += 1.25f * rotationSpeed * delta;

        float dPlanet = camera.position.length();

        dMoonVec.subVectors(camera.position, moon->position);
        float dMoon = dMoonVec.length();

        float d;
        if (dMoon < dPlanet) {

            d = (dMoon - radius * moonScale * 1.01f);

        } else {

            d = (dPlanet - radius * 1.01f);
        }

        renderer.render(scene, camera);
        controls.movementSpeed = 0.33f * d;
        controls.update(delta);
    });
}
