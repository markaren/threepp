
#include "threepp/extras/curves/CubicBezierCurve.hpp"
#include "threepp/extras/curves/CubicBezierCurve3.hpp"
#include "threepp/threepp.hpp"


using namespace threepp;

namespace {

    float getZ(Vector3 a) {
        return a.z;
    }

    float getZ(...) {
        return 0;
    }

    template<typename T>
    auto createSpheres(const std::vector<T>& points) {

        Matrix4 tmp;
        auto spheresMaterial = MeshBasicMaterial::create();
        spheresMaterial->color = Color::black;
        auto spheres = InstancedMesh::create(SphereGeometry::create(0.075f), spheresMaterial, points.size());
        int index = 0;
        for (const auto& p : points) {
            tmp.setPosition(p.x, p.y, getZ(p));
            spheres->setMatrixAt(index++, tmp);
        }

        return spheres;
    }

    template<typename Curve>
    auto createCurveObject(Curve& curve) {

        auto curvePoints = curve.getPoints(50);
        const auto geometry = BufferGeometry::create();
        geometry->setFromPoints(curvePoints);
        auto curveObject = Line::create(geometry, LineBasicMaterial::create({{"color", Color::red}}));

        curveObject->add(createSpheres(curvePoints));
        curveObject->castShadow = true;

        return curveObject;
    }

    auto createLight() {
        auto light = SpotLight::create(0xffffff, 0.5f);
        light->position.set(0, 25, 10);
        light->angle = math::PI * 0.2f;
        light->decay = 0;
        light->castShadow = true;

        return light;
    }

    auto createPlane() {

        auto geometry = PlaneGeometry::create(50, 50);
        auto material = ShadowMaterial::create();
        material->color = 0x000000;
        material->opacity = 0.7f;

        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.x = -math::PI / 2;
        mesh->receiveShadow = true;

        auto grid = GridHelper::create(50, 50);
        grid->rotation.x = math::PI / 2;
        grid->material()->opacity = 0.8f;
        grid->material()->transparent = true;
        mesh->add(grid);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("CubicBezierCurve", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 15, 15);

    OrbitControls controls{*camera, canvas};

    auto light = createLight();
    scene->add(light);

    auto plane = createPlane();
    scene->add(plane);

    auto curve = CubicBezierCurve(
            Vector2(-10, 0),
            Vector2(-5, 15),
            Vector2(20, 15),
            Vector2(10, 0));

    auto curve3 = CubicBezierCurve3(
            Vector3(-10, 0, 0),
            Vector3(-5, 5, -15),
            Vector3(20, 5, -15),
            Vector3(10, 0, 0));

    auto curveObject2 = createCurveObject(curve);
    auto curveObject3 = createCurveObject(curve3);

    scene->add(curveObject2);
    scene->add(curveObject3);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(*scene, *camera);
    });
}
