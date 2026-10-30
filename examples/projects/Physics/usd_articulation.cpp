// A USD robot simulated by PhysX: bodies, joints, drives, mass and colliders all
// come from the file through loadUsdArticulation. The arm tracks a sine on every
// joint and spheres fall onto it, so the colliders show.
//
//   usd_articulation model.usd [--shot out.png] [--frames N]
//   usd_articulation model.usd --report [--dt S] [--settle S]
//
// --report opens no window. It prints what was imported, then one JSON line per
// test pose: the commanded and settled joint values, the torque the drive holds,
// and every link's world pose.

#include "renderer_factory.hpp"

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UsdArticulation.hpp"
#include "threepp/threepp.hpp"

#include <PxPhysicsAPI.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    void printImport(const USDArticulationResult& robot) {
        const auto* raw = robot.articulation->rawArt();
        std::printf("articulation: links=%u dofs=%zu fixedBase=%d selfCollision=%d iterations=%u\n",
                    raw->getNbLinks(), robot.numDof(),
                    static_cast<bool>(raw->getArticulationFlags() & ::physx::PxArticulationFlag::eFIX_BASE),
                    !(raw->getArticulationFlags() & ::physx::PxArticulationFlag::eDISABLE_SELF_COLLISION),
                    [&] { ::physx::PxU32 p, v; raw->getSolverIterationCounts(p, v); return p; }());
        for (size_t i = 0; i + 1 < robot.linkForName.size(); i += 2) {
            const auto& [name, link] = robot.linkForName[i + 1];
            const auto* px = link.raw();
            const auto com = px->getCMassLocalPose().p;
            const auto inertia = px->getMassSpaceInertiaTensor();
            std::printf("  %-14s mass=%7.3f com=(%7.4f %7.4f %7.4f) inertia=(%.5f %.5f %.5f) shapes=%u\n",
                        name.c_str(), px->getMass(), com.x, com.y, com.z,
                        inertia.x, inertia.y, inertia.z, px->getNbShapes());
            std::vector<::physx::PxShape*> shapes(px->getNbShapes());
            px->getShapes(shapes.data(), static_cast<::physx::PxU32>(shapes.size()));
            for (const auto* shape : shapes) {
                if (shape->getGeometry().getType() != ::physx::PxGeometryType::eCONVEXMESH) continue;
                const auto* hull = static_cast<const ::physx::PxConvexMeshGeometry&>(shape->getGeometry()).convexMesh;
                const auto bounds = hull->getLocalBounds();
                std::printf("    {\"hull\":\"%s\",\"vertices\":%u,\"min\":[%.5f,%.5f,%.5f],\"max\":[%.5f,%.5f,%.5f]}\n",
                            name.c_str(), hull->getNbVertices(), bounds.minimum.x, bounds.minimum.y, bounds.minimum.z,
                            bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
            }
        }
        for (size_t i = 0; i < robot.numDof(); ++i) {
            ::physx::PxArticulationJointReducedCoordinate* joint = robot.links[i].raw()->getInboundJoint();
            const auto axis = robot.links[i].axis();
            const auto drive = joint->getDriveParams(axis);
            const auto limit = joint->getLimitParams(axis);
            std::printf("  %-6s kp=%.1f kd=%.1f maxForce=%.1f limit=[%.4f %.4f] maxVelocity=%.4f\n",
                        robot.jointNames[i].c_str(), drive.stiffness, drive.damping, drive.maxForce,
                        limit.low, limit.high, joint->getMaxJointVelocity());
        }
        for (const auto& m : robot.mimics) {
            std::printf("  mimic: %s %+.4f * %s %+.4f = 0%s\n", m.joint.c_str(), m.gearing, m.referenceJoint.c_str(),
                        m.offset, m.naturalFrequency > 0.f && m.dampingRatio > 0.f ? " (compliant)" : "");
        }
        for (const auto& message : robot.diagnostics) std::printf("  diagnostic: %s\n", message.c_str());
    }

    void printPose(const USDArticulationResult& robot, const std::vector<float>& target) {
        const auto q = robot.articulation->jointPositions();
        std::printf("{\"target\":[");
        for (size_t i = 0; i < target.size(); ++i) std::printf("%s%.7f", i ? "," : "", target[i]);
        std::printf("],\"q\":[");
        for (size_t i = 0; i < q.size(); ++i) std::printf("%s%.7f", i ? "," : "", q[i]);
        std::printf("],\"torque\":[");
        for (size_t i = 0; i < q.size(); ++i) {
            const float kp = robot.links[i].raw()->getInboundJoint()->getDriveParams(robot.links[i].axis()).stiffness;
            std::printf("%s%.4f", i ? "," : "", kp * (target[i] - q[i]));
        }
        std::printf("],\"links\":{");
        bool first = true;
        for (size_t i = 0; i + 1 < robot.linkForName.size(); i += 2) {
            const auto& [name, link] = robot.linkForName[i + 1];
            const auto p = link.position();
            const auto r = link.quaternion();
            std::printf("%s\"%s\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]", first ? "" : ",", name.c_str(),
                        p.x, p.y, p.z, static_cast<float>(r.x), static_cast<float>(r.y),
                        static_cast<float>(r.z), static_cast<float>(r.w));
            first = false;
        }
        std::printf("}}\n");
    }

    int report(PhysxWorld& world, USDArticulationResult& robot, float settle) {
        printImport(robot);

        // A sleeping articulation stops short of its equilibrium, and the hold
        // torques below are read off how far short.
        robot.articulation->rawArt()->setSleepThreshold(0.f);

        const auto settings = world.settings();
        const int steps = static_cast<int>(settle / settings.fixedTimestep);
        // Inside any reasonable arm's limits, away from the home pose.
        const std::vector<std::vector<float>> poses{
                {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                {0.f, 0.8f, 0.f, 0.f, 0.f, 0.f},
                {0.f, -0.6f, 0.9f, 0.f, 0.f, 0.f},
                {0.5f, -0.7f, 0.9f, 1.f, -0.8f, 0.6f},
                {-1.2f, 0.4f, -0.5f, -0.7f, 1.1f, -1.4f},
        };
        for (auto target : poses) {
            target.resize(robot.numDof());
            robot.articulation->setDriveTargets(target.data(), target.size());
            for (int i = 0; i < steps; ++i) world.step(settings.fixedTimestep);
            printPose(robot, target);
        }

        // Step response of the first joint from the last pose: the effort limit
        // sets the initial acceleration, the speed limit the plateau.
        auto target = robot.articulation->jointPositions();
        const float start = target[0];
        target[0] = start + 1.5f;
        robot.articulation->setDriveTargets(target.data(), target.size());
        const int stepSamples = static_cast<int>(1.f / settings.fixedTimestep);
        std::printf("{\"step\":[");
        for (int i = 0; i < stepSamples; ++i) {
            world.step(settings.fixedTimestep);
            std::printf("%s[%.5f,%.6f,%.6f]", i ? "," : "", static_cast<float>(i + 1) * settings.fixedTimestep,
                        robot.links[0].jointPosition() - start, robot.links[0].jointVelocity());
        }
        std::printf("]}\n");
        return 0;
    }

}// namespace

int main(int argc, char** argv) {
    std::string modelPath, shotPath;
    bool reportOnly = false;
    int shotFrames = 240;
    float dt = 1.f / 240.f;
    float settle = 3.f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--report") == 0) reportOnly = true;
        else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc) dt = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--settle") == 0 && i + 1 < argc) settle = static_cast<float>(std::atof(argv[++i]));
        else modelPath = argv[i];
    }
    if (modelPath.empty()) {
        std::cerr << "usage: usd_articulation model.usd [--report] [--shot out.png] [--frames N]\n";
        return 1;
    }

    PhysxWorld::Settings settings;
    settings.fixedTimestep = dt;
    settings.maxSubSteps = 16;
    PhysxWorld world(settings);

    auto robot = loadUsdArticulation(world, modelPath);
    if (!robot.articulation) {
        std::cerr << robot.error << "\n";
        return 1;
    }
    if (reportOnly) return report(world, robot, settle);
    printImport(robot);

    const bool capturing = !shotPath.empty();
    Canvas canvas("USD articulation", {{"size", WindowSize{1280, 720}}, {"headless", capturing}, {"antialiasing", 4}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    Scene scene;
    scene.background = Color(0xb9c7d1);
    auto hemi = HemisphereLight::create(0xffffff, 0x445566, 1.f);
    scene.add(hemi);
    auto sun = DirectionalLight::create(0xffffff, 2.f);
    sun->position.set(1.5f, 3.f, 1.f);
    sun->castShadow = true;
    scene.add(sun);

    for (auto& mesh : robot.meshes) {
        scene.add(mesh);
        mesh->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
    }
    scene.add(robot.scene);

    auto floor = Mesh::create(BoxGeometry::create(6.f, 0.2f, 6.f), MeshStandardMaterial::create({{"color", 0x8a939b}}));
    floor->position.y = -0.1f;
    floor->receiveShadow = true;
    scene.add(floor);
    world.addStatic(*floor);

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.02f, 100.f);
    camera->position.set(1.5f, 1.2f, 1.7f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.2f, 0.55f, 0.f);
    controls.update();
    camera->lookAt(controls.target);

    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
    });

    auto ballGeometry = SphereGeometry::create(0.05f);
    auto ballMaterial = MeshStandardMaterial::create({{"color", 0xd9622b}});
    std::vector<std::shared_ptr<Mesh>> balls;

    // Sweep every joint about the pose the file authors, inside its range.
    std::vector<float> home = robot.articulation->jointPositions();
    std::vector<float> target = home;

    Clock clock;
    float time = 0.f;
    float nextBall = 0.5f;
    int frame = 0;
    canvas.animate([&] {
        const float step = capturing ? 1.f / 60.f : std::min(clock.getDelta(), 1.f / 20.f);
        time += step;
        for (size_t i = 0; i < target.size(); ++i) {
            target[i] = home[i] + 0.6f * std::sin(0.7f * time + static_cast<float>(i));
        }
        robot.articulation->setDriveTargets(target.data(), target.size());

        if (time > nextBall && balls.size() < 40) {
            nextBall = time + 0.5f;
            auto ball = Mesh::create(ballGeometry, ballMaterial);
            const auto index = static_cast<float>(balls.size());
            ball->position.set(0.1f + 0.3f * std::sin(index * 2.4f), 1.6f, 0.3f * std::cos(index * 2.4f));
            ball->castShadow = true;
            scene.add(ball);
            world.add(*ball, 800.f);
            balls.push_back(ball);
        }

        world.step(step);
        renderer->render(scene, *camera);
        if (capturing && ++frame >= shotFrames) {
            renderer->writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << "\n";
            std::exit(0);
        }
    });
}
