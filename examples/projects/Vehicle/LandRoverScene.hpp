
#ifndef THREEPP_LANDROVER_SCENE_HPP
#define THREEPP_LANDROVER_SCENE_HPP

#include "threepp/threepp.hpp"

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"

#include <PxPhysicsAPI.h>

namespace threepp::landrover {

    // Builds the world for the PhysX vehicle demo: HDR sky + environment,
    // sun + ambient, fallback ground (visual + static collider), and the drift
    // track (visual + per-sub-mesh trimesh static colliders, plus dynamic
    // cones / barriers / spring-mounted barrier cylinders).
    inline void buildScene(Scene& scene, PhysxWorld& world) {

        using namespace ::physx;

        RGBELoader hdrLoader;
        if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr")) {
            scene.background = hdrTexture;
            scene.environment = hdrTexture;
        }

        auto sun = DirectionalLight::create(0xffffff, 1.2f);
        sun->position.set(20, 30, 20);
        scene.add(sun);
        scene.add(AmbientLight::create(0xffffff, 0.1f));

        // Fallback ground far below the track to catch the car if it leaves the
        // drivable surface — keeps the demo from spawning the vehicle into the void.
        auto groundMat = MeshLambertMaterial::create();
        groundMat->color = Color::darkolivegreen;
        auto ground = Mesh::create(BoxGeometry::create(500, 1, 500), groundMat);
        ground->position.y = -10.f;
        scene.add(ground);
        world.addStatic(*ground);

        // Race track: visual + trimesh collider per sub-mesh. Native AABB is ~480m
        // across, roughly 2× a real drift course — scale down so the car (real-world
        // meters) reads correctly against it.
        ModelLoader modelLoader;
        auto track = modelLoader.load(std::string(DATA_FOLDER) + "/models/gltf/drift_track/drift_race_track_free.glb");
        track->scale.set(0.5f, 0.5f, 0.5f);
        scene.add(track);

        world.addStaticTrimeshTree(*track, [](const Mesh& m) {
            return m.name.rfind("Rails", 0) == 0
                || m.name.rfind("Road", 0) == 0
                || m.name.rfind("Object", 0) == 0
                || m.name.rfind("Terrain", 0) == 0;
        });

        // Cones become dynamic convex obstacles the car can knock around. Barrier
        // cylinders are tethered to a ground anchor with a D6 joint: linear DOFs
        // locked, angular DOFs free with a spring drive that returns them to
        // upright after the car bumps them. Order of name checks matters:
        // "BarrierCylinder" also starts with "Barrier", so cylinder must win first.
        track->traverseType<Mesh>([&](Mesh& m) {

            const bool isCone = m.name.rfind("Cone", 0) == 0;
            const bool isBarrierCylinder = m.name.rfind("BarrierCylinder", 0) == 0;
            const bool isBarrier = m.name.rfind("Barrier", 0) == 0;

            if (isCone) {
                world.addDynamicConvex(m, 5.f);
            } else if (isBarrierCylinder) {
                auto* dyn = world.addDynamicConvex(m, 100.f);
                if (!dyn) return;
                const PxTransform pose = dyn->getGlobalPose();
                auto* anchor = world.physics().createRigidStatic(pose);
                world.scene().addActor(*anchor);

                auto* joint = PxD6JointCreate(
                        world.physics(),
                        anchor, PxTransform(PxIdentity),
                        dyn, PxTransform(PxIdentity));
                joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
                joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
                joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
                joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
                joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
                joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
                // Acceleration-mode spring (last arg true) feels mass-independent.
                const PxD6JointDrive drive(2000.f, 300.f, PX_MAX_F32, true);
                joint->setDrive(PxD6Drive::eSWING, drive);
                joint->setDrive(PxD6Drive::eTWIST, drive);
                joint->setDrivePosition(PxTransform(PxIdentity));
            } else if (isBarrier) {
                world.addDynamicConvex(m, 500.f);
            }
        });
    }

}// namespace threepp::landrover

#endif//THREEPP_LANDROVER_SCENE_HPP
