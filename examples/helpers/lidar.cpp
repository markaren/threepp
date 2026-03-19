
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/helpers/LidarSensor.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdlib>

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
    void updatePointCloud(const Points& points, const std::vector<Vector3>& cloud,
                          const Vector3& sensorPos, float maxDist) {

        auto& geom = *points.geometry();
        auto* posAttr = geom.getAttribute<float>("position");
        auto* colAttr = geom.getAttribute<float>("color");

        Color c;
        int i = 0;
        for (const auto& p : cloud) {
            posAttr->setXYZ(i, p.x, p.y, p.z);

            c.setHSL(0.33f * (1.f - std::min(p.distanceTo(sensorPos) / maxDist, 1.f)), 1.f, 0.5f);
            colAttr->setXYZ(i, c.r, c.g, c.b);

            ++i;
        }

        geom.setDrawRange(0, i);
        posAttr->needsUpdate();
        colAttr->needsUpdate();
    }

}// namespace

int main() {

    Canvas canvas("Lidar", {{"antialiasing", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color(0x111122);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 200.f);
    camera->position.set(0, 12, 18);

    setupScene(*scene);

    // --- Lidar sensor ---
    auto lidar = std::make_unique<LidarSensor>(LidarModel::OS0_128(),  512, 0.5f, 20.f);
    lidar->position.set(0, 2, 0);
    scene->add(*lidar);

    OrbitControls controls{*camera, canvas};

    // --- Point cloud visualisation ---
    const size_t maxPoints = 6 * std::pow(lidar->faceSize(), 2);
    auto pcGeom = BufferGeometry::create();
    pcGeom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    pcGeom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    pcGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    pcGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);

    auto pcMaterial = PointsMaterial::create({{"size", 0.1f}, {"vertexColors", true}});
    auto points = Points::create(pcGeom, pcMaterial);
    points->layers.set(1);
    points->frustumCulled = false;
    scene->add(points);


    const char* modeNames[] = {"Dense Grid", "VLP-16", "HDL-32E", "OS1-64", "OS0-128"};
    int currentMode = 4;// start on OS0-128

    auto changeLidar = [&](std::unique_ptr<LidarSensor> newLidar) {
        scene->remove(*lidar);
        newLidar->rangeNoise = lidar->rangeNoise;
        newLidar->position.copy(lidar->position);
        newLidar->rotation.copy(lidar->rotation);
        lidar = std::move(newLidar);
        scene->add(*lidar);
    };

    bool senorDataOnly = false;
    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin("Settings");
        ImGui::Checkbox("Show senor data only", &senorDataOnly);
        ImGui::SliderFloat("Range noise", &lidar->rangeNoise, 0.f, 0.1f);

        int prevMode = currentMode;
        ImGui::Combo("Mode", &currentMode, modeNames, 5);
        if (currentMode != prevMode) {
            switch (currentMode) {
                case 0: changeLidar(std::make_unique<LidarSensor>(64, 0.5f, 20.f)); break;
                case 1: changeLidar(std::make_unique<LidarSensor>(LidarModel::VLP16(), 512, 0.5f, 20.f)); break;
                case 2: changeLidar(std::make_unique<LidarSensor>(LidarModel::HDL32E(), 512, 0.5f, 20.f)); break;
                case 3: changeLidar(std::make_unique<LidarSensor>(LidarModel::OS1_64(), 512, 0.5f, 20.f)); break;
                case 4: changeLidar(std::make_unique<LidarSensor>(LidarModel::OS0_128(), 512, 0.5f, 20.f)); break;
            }
        }

        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    std::vector<Vector3> cloud;
    std::vector<Color> colors;
    canvas.animate([&] {
        const float t = clock.getElapsedTime();

        // Slowly sweep the sensor in yaw and pitch
        lidar->rotation.y = t * 0.4f;
        lidar->rotation.x = -0.4f + 0.25f * std::sin(t * 0.3f);

        // Scan the scene and update the visualised point cloud
        points->visible = false;
        colors.clear();
        lidar->scan(renderer, *scene, cloud);
        points->visible = true;

        Vector3 sensorWorld;
        lidar->getWorldPosition(sensorWorld);
        updatePointCloud(*points, cloud, sensorWorld, lidar->far());

        if (senorDataOnly) {
            camera->layers.set(1);
        } else {
            camera->layers.enableAll();
        }

        renderer.render(*scene, *camera);
        ui.render();
    });
}
