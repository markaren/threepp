
#include "renderer_factory.hpp"

#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/USDLoader.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_set>

using namespace threepp;

int main(int argc, char** argv) {
    // usd_loader [model.usd] [--shot out.png] [--frames N] [--pose mid|v1,v2,...]
    // --shot: headless capture — frame the loaded scene from its bounding box,
    // render N warm-up frames (TAA/denoiser converge), write one PNG, exit.
    // --pose: drive a robot's joints, to each range's midpoint or to the
    // listed values (degrees for revolute joints, metres for prismatic).
    std::string modelPath;
    std::string shotPath;
    std::string pose;
    int   shotFrames = 240;
    float shotZoom   = 1.f;// <1 = closer (distance scales with it)
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) {
            shotPath = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            shotFrames = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
            shotZoom = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (std::strcmp(argv[i], "--pose") == 0 && i + 1 < argc) {
            pose = argv[++i];
            continue;
        }
        std::filesystem::path arg = argv[i];
        if (std::filesystem::exists(arg) && std::filesystem::is_regular_file(arg)) {
            auto ext = arg.extension().string();
            std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz") {
                modelPath = arg.string();
            } else {
                std::cerr << "Ignoring argument (not a .usd/.usda/.usdc/.usdz file): " << arg << "\n";
            }
        } else {
            std::cerr << "Ignoring argument (file not found): " << arg << "\n";
        }
    }
    const bool capturing = !shotPath.empty();
    int shotFrame = 0;

    Canvas canvas("USD loader example", {{"size", WindowSize{1280, 720}}, {"headless", capturing}});
    auto renderer= createRenderer(canvas);

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) +
                         "/textures/env/autumn_field_puresky_2k.hdr");

    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01f, 100000.f);
    camera->position.set(0, 1, 5);

    OrbitControls controls{*camera, canvas};

    auto dirLight = DirectionalLight::create(0xffffff, 1.5f);
    dirLight->position.set(1, 1, 1);
    scene.add(dirLight);

    USDLoader loader;

    if (!modelPath.empty()) {
        std::cout << "Loading: " << modelPath << "\n";
        auto result = loader.loadFull(modelPath);
        if (!result.scene) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        std::cout << "Loaded: " << result.scene->name << "\n";

        // What came through: meshes, triangles, distinct materials and how
        // many of those carry a colour or normal texture.
        int meshes = 0;
        size_t triangles = 0;
        std::unordered_set<Material*> materials;
        int textured = 0, normalMapped = 0;
        result.scene->traverseType<Mesh>([&](Mesh& m) {
            ++meshes;
            auto g = m.geometry();
            if (g->hasIndex()) triangles += g->getIndex()->count() / 3;
            else if (auto* pos = g->getAttribute<float>("position")) triangles += pos->count() / 3;
            for (auto& mat : m.materials()) {
                if (!materials.insert(mat.get()).second) continue;
                if (auto* mm = dynamic_cast<MaterialWithMap*>(mat.get()); mm && mm->map) ++textured;
                if (auto* nm = dynamic_cast<MaterialWithNormalMap*>(mat.get()); nm && nm->normalMap) ++normalMapped;
            }
        });
        Box3 bounds;
        bounds.setFromObject(*result.scene);
        Vector3 extent;
        bounds.getSize(extent);
        std::cout << "Stats: meshes=" << meshes << " triangles=" << triangles
                  << " materials=" << materials.size() << " textured=" << textured
                  << " normalMapped=" << normalMapped
                  << " size=" << extent.x << "x" << extent.y << "x" << extent.z << "\n";
        if (result.environment) {
            scene.background = result.environment;
            scene.environment = result.environment;
        }
        if (auto robot = result.robot) {
            const auto infos = robot->getArticulatedJointInfo();
            std::cout << "Robot: dofs=" << robot->numDOF() << " joints=" << robot->jointInfos().size()
                      << " endEffector=" << robot->endEffectorLink() << "\n";
            for (size_t i = 0; i < infos.size(); ++i) {
                const auto range = robot->getJointRange(i, true);
                std::cout << "  " << infos[i].name << " " << (infos[i].type == Robot::JointType::Revolute ? "revolute" : "prismatic")
                          << " axis=(" << infos[i].axis.x << "," << infos[i].axis.y << "," << infos[i].axis.z << ")"
                          << " range=[" << range.min << "," << range.max << "]"
                          << " loaded=" << robot->getJointValue(i, true) << "\n";
            }
            if (pose == "mid") {
                for (size_t i = 0; i < robot->numDOF(); ++i) {
                    const auto range = robot->getJointRange(i);
                    if (std::isfinite(range.min) && std::isfinite(range.max)) robot->setJointValue(i, range.mid());
                }
            } else if (!pose.empty()) {
                std::stringstream ss(pose);
                std::string value;
                for (size_t i = 0; std::getline(ss, value, ',') && i < robot->numDOF(); ++i) {
                    robot->setJointValue(i, std::stof(value), infos[i].type == Robot::JointType::Revolute);
                }
            }
            if (!pose.empty()) {
                robot->updateMatrixWorld(true);
                Vector3 ee;
                ee.setFromMatrixPosition(robot->getEndEffectorTransform());
                std::cout << "End effector (world, Y-up): " << ee.x << " " << ee.y << " " << ee.z << "\n";
            }
            scene.add(robot);
        } else {
            scene.add(result.scene);
        }
    } else {
        auto catPlane = loader.load(std::string(DATA_FOLDER) + "/models/usd/texture-cat-plane.usdz");

        if (!catPlane) {
            std::cerr << "Failed to load model\n";
            return 1;
        }

        catPlane->position.x = -2;
        scene.add(catPlane);

        auto suzanne = loader.load(std::string(DATA_FOLDER) + "/models/usd/suzanne-pbr.usda");

        if (!suzanne) {
            std::cerr << "Failed to load model\n";
            return 1;
        }

        suzanne->position.x = 2;
        scene.add(suzanne);
    }

    if (capturing) {
        // Frame the scene from its bounds: low oblique view across the model
        // (close enough that glass/reflective details fill the frame).
        Box3 bbox;
        bbox.setFromObject(scene);
        Vector3 center, size;
        bbox.getCenter(center);
        bbox.getSize(size);
        const float maxDim = std::max({size.x, size.y, size.z, 0.01f});
        const float d = maxDim * shotZoom;
        camera->position.set(center.x + 0.55f * d,
                             center.y + 0.30f * d,
                             center.z + 0.55f * d);
        camera->lookAt(Vector3(center.x, center.y + 0.05f * d, center.z));
        camera->updateMatrixWorld();
    }

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer->setSize(newSize);
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer->render(scene, *camera);
        if (capturing && ++shotFrame >= shotFrames) {
            renderer->writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << "\n";
            std::exit(0);
        }
    });

}
