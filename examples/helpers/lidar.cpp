
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/Lidar.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/threepp.hpp"

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
    void updatePointCloud(Points& points, const std::vector<Vector3>& cloud,
                          const Vector3& sensorPos, float maxDist) {
        std::vector<float> positions;
        std::vector<float> colors;
        positions.reserve(cloud.size() * 3);
        colors.reserve(cloud.size() * 3);

        for (const auto& p : cloud) {
            positions.insert(positions.end(), {p.x, p.y, p.z});

            float t = std::min(p.distanceTo(sensorPos) / maxDist, 1.f);
            // green (near) → red (far)
            colors.insert(colors.end(), {t, 1.f - t, 0.f});
        }

        auto buf = BufferGeometry::create();

        buf->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        buf->setAttribute("color", FloatBufferAttribute::create(colors, 3));

        points.setGeometry(buf);
        // if (positions.empty()) {
        //     geo->deleteAttribute("position");
        //     geo->deleteAttribute("color");
        // } else {
        //     geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        //     geo->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        // }
    }

}// namespace

int main() {

    Canvas canvas("Lidar", {{"antialiasing", 4}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color(0x111122));

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 200.f);
    camera->position.set(0, 12, 18);

    setupScene(*scene);

    // --- Lidar sensor ---
    // 90° vertical FOV, 128×32 resolution, 20m range
    auto lidar = std::make_shared<Lidar>(90.f, 128, 32, 0.5f, 20.f);
    lidar->position.set(0, 6, 0);
    scene->add(lidar);

    OrbitControls controls{lidar->postCamera_, canvas};

    // Small axes gizmo so the sensor orientation is visible
    auto sensorAxes = AxesHelper::create(0.5f);
    lidar->add(sensorAxes);

    // --- Point cloud visualisation ---
    auto pcMaterial = PointsMaterial::create({{"size", 0.12f}, {"vertexColors", true}});
    auto pointCloud = Points::create(BufferGeometry::create(), pcMaterial);
    pointCloud->frustumCulled = false;// geometry is rebuilt every frame
    scene->add(pointCloud);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const float t = clock.getElapsedTime();

        // Slowly sweep the sensor in yaw and pitch
        lidar->rotation.y = t * 0.4f;
        lidar->rotation.x = -0.4f + 0.25f * std::sin(t * 0.3f);

        // Scan the scene and update the visualised point cloud
        auto cloud = lidar->scan(renderer, *scene);

        Vector3 sensorWorld;
        lidar->getWorldPosition(sensorWorld);
        updatePointCloud(*pointCloud, cloud, sensorWorld, lidar->far());

        renderer.render(*scene, *camera);
        // renderer.render(lidar->postScene_, lidar->postCamera_);

        controls.update();
    });
}
