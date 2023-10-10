
#include "threepp/extras/curves/CubicBezierCurve.hpp"
#include "threepp/extras/curves/CubicBezierCurve3.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("CubicBezierCurve", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 5, 20);

    OrbitControls controls{*camera, canvas};

    const auto material = LineBasicMaterial::create();
    material->color = 0xff0000;

    std::vector<Vector2> curve2Points;
    std::vector<Vector3> curve3Points;

    //2D
    {
        auto curve = CubicBezierCurve(
                Vector2(-10, 0),
                Vector2(-5, 15),
                Vector2(20, 15),
                Vector2(10, 0));

        curve2Points = curve.getPoints(50);
        const auto geometry = BufferGeometry::create();
        geometry->setFromPoints(curve2Points);
        const auto curveObject = Line::create(geometry, material);
        scene->add(curveObject);
    }

    //3D
    {
        auto curve = CubicBezierCurve3(
                Vector3(-10, 0, 0),
                Vector3(-5, 5, -15),
                Vector3(20, 5, -15),
                Vector3(10, 0, 0));

        curve3Points = curve.getPoints(50);
        const auto geometry = BufferGeometry::create();
        geometry->setFromPoints(curve3Points);

        const auto curveObject = Line::create(geometry, material);
        scene->add(curveObject);
    }

    Matrix4 tmp;
    auto spheresMaterial = MeshBasicMaterial::create();
    spheresMaterial->color = Color::black;
    auto spheres = InstancedMesh::create(SphereGeometry::create(0.075f), spheresMaterial, curve2Points.size() + curve3Points.size());
    int index = 0;
    for (const auto& p : curve2Points) {
        tmp.setPosition(p.x, p.y, 0);
        spheres->setMatrixAt(index++, tmp);
    }
    for (const auto& p : curve3Points) {
        tmp.setPosition(p.x, p.y, p.z);
        spheres->setMatrixAt(index++, tmp);
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
