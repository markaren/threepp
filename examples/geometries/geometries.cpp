
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/threepp.hpp"

#include <cmath>


using namespace threepp;

namespace {

    struct CustomSineCurve: Curve3 {

        explicit CustomSineCurve(float scale): scale(scale) {}

        void getPoint(float t, Vector3& target) const override {
            float tx = t * 3 - 1.5f;
            float ty = std::sin(math::TWO_PI * t);
            float tz = 0;

            target.set(tx, ty, tz).multiplyScalar(scale);
        }

    private:
        float scale;

    };

    auto createBox(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = BoxGeometry::create();
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createSphere(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = SphereGeometry::create(0.5f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createPlane(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = PlaneGeometry::create();
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createCylinder(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = CylinderGeometry::create(0.5f, 0.5f, 1.f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createCone(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = ConeGeometry::create(0.5f, 1.f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createRing(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = RingGeometry::create(0.25f, 0.5f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createCircle(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = CircleGeometry::create(0.5f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createTorus(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = TorusGeometry::create(0.5, 0.1);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createTorusKnot(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        const auto geometry = TorusKnotGeometry::create(0.5, 0.1);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

    auto createTube(const std::shared_ptr<Material>& m1, const std::shared_ptr<LineBasicMaterial>& m2) {
        auto curve = std::make_shared<CustomSineCurve>(0.5f);
        const auto geometry = TubeGeometry::create(curve, 32, 0.1f);
        auto mesh = Mesh::create(geometry, m1);
        auto wire = LineSegments::create(WireframeGeometry::create(*geometry), m2);
        mesh->add(wire);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Geometries", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.z = 8;

    TextureLoader tl;
    auto material = MeshBasicMaterial::create();
    material->map = tl.load("data/textures/uv_grid_opengl.jpg");
    material->side = Side::Double;

    auto lineMaterial = LineBasicMaterial::create();
    lineMaterial->color = Color::black;
    lineMaterial->opacity = 0.8f;
    lineMaterial->transparent = true;
    lineMaterial->depthWrite = false;

    std::vector<std::shared_ptr<Mesh>> meshes{
            createBox(material, lineMaterial),
            createCircle(material, lineMaterial),
            createSphere(material, lineMaterial),
            createCone(material, lineMaterial),
            createCylinder(material, lineMaterial),
            createPlane(material, lineMaterial),
            createRing(material, lineMaterial),
            createTorus(material, lineMaterial),
            createTorusKnot(material, lineMaterial),
            createTube(material, lineMaterial)};

    Box3 box;
    float x = 0;
    for (unsigned i = 0; i < meshes.size(); i++) {
        auto& m = meshes[i];
        m->position.x = x;
        if (i % 2 == 0) {
            m->position.y = 2;
        } else {
            x += 2;
        }

        scene->add(m);
        box.expandByObject(*m);
    }

    Vector3 center;
    box.getCenter(center);
    camera->position.x = center.x;
    camera->position.y = center.y;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        for (auto& m : meshes) {
            m->rotation.y += 1 * dt;
        }

        renderer.render(*scene, *camera);
    });
}
