
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createCurve() {
        CatmullRomCurve3 curve;
        curve.points.insert(curve.points.end(),
                            {Vector3(-10, 0, 10),
                             Vector3(-5, 5, 5),
                             Vector3(0, 0, 0),
                             Vector3(5, -5, 5),
                             Vector3(10, 0, 10)});


        const auto points = curve.getPoints(50);
        const auto geometry = BufferGeometry::create();
        geometry->setFromPoints(points);
        const auto material = LineBasicMaterial::create();
        material->color = 0xff0000;
        const auto curveObject = Line::create(geometry, material);

        Matrix4 tmp;
        auto spheresMaterial = MeshBasicMaterial::create();
        spheresMaterial->color = Color::black;
        auto spheres = InstancedMesh::create(SphereGeometry::create(0.075f), spheresMaterial, points.size());
        for (unsigned i = 0; i < points.size(); i++) {
            tmp.setPosition(points[i]);
            spheres->setMatrixAt(i, tmp);
        }
        spheres->castShadow = true;
        curveObject->add(spheres);

        curveObject->castShadow = true;

        return curveObject;
    }

    auto createLight() {
        auto light = SpotLight::create(0xffffff, 0.5f);
        light->position.set(0, 20, 10);
        light->angle = math::PI * 0.2f;
        light->decay = 0;
        light->castShadow = true;

        return light;
    }

    auto createPlane() {

        auto geometry = PlaneGeometry::create(50, 50);
        auto material = ShadowMaterial::create();
        material->color = 0x000000;
        material->opacity = 0.2f;

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

    Canvas canvas("CatmullRoomCurve", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = 0xf0f0f0;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.set(5, 10, 20);

    OrbitControls controls{*camera, canvas};

    auto curve = createCurve();
    curve->position.y = 5;
    scene->add(curve);

    auto plane = createPlane();
    scene->add(plane);

    auto light = createLight();
    scene->add(light);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(*scene, *camera);
    });
}
