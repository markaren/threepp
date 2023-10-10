
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("CatmullRoomCurve", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.set(5, 10, 20);

    OrbitControls controls{*camera, canvas};

    auto curve = CatmullRomCurve3();
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
    scene->add(curveObject);

    Matrix4 tmp;
    auto spheresMaterial = MeshBasicMaterial::create();
    spheresMaterial->color = Color::black;
    auto spheres = InstancedMesh::create(SphereGeometry::create(0.075f), spheresMaterial, points.size());
    for (unsigned i = 0; i < points.size(); i++) {
        tmp.setPosition(points[i]);
        spheres->setMatrixAt(i, tmp);
    }
    scene->add(spheres);


    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(*scene, *camera);
    });
}
