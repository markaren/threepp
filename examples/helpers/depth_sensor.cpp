
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <random>

using namespace threepp;

namespace {

    // Build a simple scene: ground plane + scattered boxes
    void setupScene(Scene& scene) {
        // Ground
        auto ground = Mesh::create(
                BoxGeometry::create(30, 0.2f, 30),
                MeshStandardMaterial::create({{"color", Color(0x888888)}}));
        scene.add(ground);

        // Random boxes
        std::srand(42);
        auto boxMat = MeshStandardMaterial::create({{"color", Color(0x4488cc)}});
        for (int i = 0; i < 20; ++i) {
            float w = 0.5f + (std::rand() % 100) / 50.f;
            float h = 0.5f + (std::rand() % 100) / 25.f;
            float d = 0.5f + (std::rand() % 100) / 50.f;
            auto box = Mesh::create(BoxGeometry::create(w, h, d), boxMat);
            box->position.set(
                    (std::rand() % 240 - 120) / 10.f,
                    h / 2.f + 0.1f,
                    (std::rand() % 240 - 120) / 10.f);
            scene.add(box);
        }

        // Lights
        scene.add(AmbientLight::create(0xffffff, 0.4f));
        auto dirLight = DirectionalLight::create(0xffffff, 1.f);
        dirLight->position.set(5, 10, 5);
        scene.add(dirLight);
    }

    // Update a Points object's position and color attributes from a point cloud.
    // Colors are mapped by distance: near=green, far=red.
    void updatePointCloud(InstancedMesh& points, const std::vector<Vector3>& cloud,
                          const Vector3& sensorPos, float maxDist) {


        Color c;
        Matrix4 m;
        int i = 0;
        for (const auto& p : cloud) {
            m.identity();
            points.setMatrixAt(i, m.setPosition(p.x, p.y, p.z));
            points.setColorAt(i, c.setHSL(0.33f * (1.f - std::min(p.distanceTo(sensorPos) / maxDist, 1.f)), 1.f, 0.5f));
            i++;
        }

        points.setCount(i);
        points.instanceMatrix()->needsUpdate();
        points.instanceColor()->needsUpdate();

        points.computeBoundingSphere();
    }

}// namespace

int main() {

    Canvas canvas("Depth sensor", {{"antialiasing", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color(0x111122);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 200.f);
    camera->position.set(0, 12, 18);

    setupScene(*scene);

    // --- Lidar sensor ---
    DepthSensor lidar(90.f, 512, 256, 0.5f, 20.f);
    lidar.position.set(0, 1, 0);
    scene->add(lidar);

    OrbitControls controls{*camera, canvas};

    // Small axes gizmo so the sensor orientation is visible
    auto sensorAxes = AxesHelper::create(0.5f);
    sensorAxes->rotateY(math::PI);
    lidar.add(sensorAxes);


    // --- Point cloud visualisation ---
    auto pcMaterial = MeshBasicMaterial::create();
    auto geom = SphereGeometry::create(0.025f);
    auto points = InstancedMesh::create(geom, pcMaterial, lidar.width() * lidar.height());
    points->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    scene->add(points);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    std::vector<Vector3> cloud;
    canvas.animate([&] {
        const float t = clock.getElapsedTime();

        // Slowly sweep the sensor in yaw and pitch
        lidar.rotation.y = t * 0.4f;
        lidar.rotation.x = -0.4f + 0.25f * std::sin(t * 0.3f);

        // Scan the scene and update the visualised point cloud
        points->visible = false;
        lidar.scan(renderer, *scene, cloud);
        points->visible = true;

        Vector3 sensorWorld;
        lidar.getWorldPosition(sensorWorld);
        updatePointCloud(*points, cloud, sensorWorld, lidar.far());

        renderer.render(*scene, *camera);
    });
}
