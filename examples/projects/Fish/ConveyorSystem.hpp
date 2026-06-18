// Sim-side conveyor system: rebuilds the layout authored in the conveyor designer
// (data/models/conveyor/layout.json) into a PhysX world + threepp scene, and animates it.
//
// This is the PhysX-dependent counterpart to ConveyorAssets.hpp (which is the PhysX-free
// schema + geometry shared with the designer). Keeping it separate lets the fish simulation
// (fish_softbody.cpp) stay focused on fish: it just constructs a ConveyorSystem, reads its
// inlets() for spawn points, and calls update() each frame.
#ifndef THREEPP_FISH_CONVEYOR_SYSTEM_HPP
#define THREEPP_FISH_CONVEYOR_SYSTEM_HPP

#include "ConveyorAssets.hpp"

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fishsim {

    // Builds and animates the conveyor layout. Construction loads the layout and registers
    // the per-substep belt/cleat motion with the world; update() drives the per-frame visuals.
    class ConveyorSystem {

    public:
        // Build the layout at <convDir>/layout.json (or one default belt if absent) into the
        // world + scene, and register the belt-drag / cleat-march substep hooks.
        ConveyorSystem(threepp::PhysxWorld& world, threepp::Scene& scene, std::filesystem::path convDir);

        // Per-frame visual update (belt texture scroll, roller spin, cleat meshes follow their
        // colliders). dt is real elapsed seconds; pass the same value as world.step(dt).
        void update(float dt);

        // Fish drop points, one per belt path (or a single fallback). The fish sim streams
        // fish onto these.
        [[nodiscard]] const std::vector<threepp::Vector3>& inlets() const { return inlets_; }

        // Number of belt colliders (for UI display).
        [[nodiscard]] int beltCount() const { return static_cast<int>(belts_.size()); }

        // Global speed multiplier applied to belt drag, roller spin and cleat travel (UI).
        float beltSpeedScale = 1.f;

    private:
        // A kinematic belt collider driven by the fake-velocity trick (translated or rotated
        // each substep, then teleported back so the surface never actually moves).
        struct Belt {
            ::physx::PxRigidDynamic* actor;
            bool rotational = false;
            ::physx::PxVec3 velocity{0, 0, 0};// straight: world m/s
            float omega = 0.f;                // curve: rad/s about the actor's own +Y (arc centre)
            ::physx::PxTransform saved{::physx::PxIdentity};
        };

        // A belt ribbon whose texture is scrolled each frame to fake surface motion.
        struct AnimatedBelt {
            std::shared_ptr<threepp::Texture> tex;
            std::shared_ptr<threepp::MeshStandardMaterial> mat;// bumped each frame so the scroll re-uploads
            float scrollRate;                                  // d(offset.y)/dt at beltSpeedScale = 1
        };

        // A bank of roller cylinders spun in place to read as a powered-roller bed.
        struct RollerBank {
            std::vector<std::shared_ptr<threepp::Mesh>> rollers;
            std::vector<threepp::Quaternion> base;// per-roller axis orientation (parallel to rollers)
            float omega = 0.f;                    // rad/s about each roller's own +Y at beltSpeedScale = 1
            float angle = 0.f;                    // accumulated spin
        };

        // One cleat bar of a travelling band.
        struct MovingCleat {
            ::physx::PxRigidDynamic* actor;
            std::shared_ptr<threepp::Mesh> mesh;
            float offset;       // base arc-length along the track
            float prevS = -1.f; // last position; a backward jump = a wrap (teleport, no push)
        };

        // A cleated belt: bars that genuinely travel along the centerline, wrapping at the ends.
        struct CleatTrack {
            std::vector<threepp::Vector3> poly;// travel-ordered centerline the cleats ride
            float length = 0.f;                // total arc length
            float speed = 0.f;                 // m/s at beltSpeedScale = 1
            float height = 0.f;
            float rampLen = 0.f;// distance over which a bar folds flat at each end
            float phase = 0.f;  // advances with time, wrapped to [0,length)
            std::vector<MovingCleat> cleats;
        };

        // Lazily load + re-origin a conveyor model (cached), applying any type override.
        conveyor::ModelTemplate& convTemplate(const std::string& name);

        void addConveyorVisual(const conveyor::Piece& piece);
        void addStraightBelt(const conveyor::Piece& piece);
        void addStraightSeg(const threepp::Vector3& a, const threepp::Vector3& b, float width, float speed);
        void addArcBelt(const threepp::Vector3& A, const threepp::Vector3& C, const threepp::Vector3& B,
                        float width, float speed, const threepp::Vector3& incoming);
        void buildPathBelts(const std::vector<conveyor::Waypoint>& ctrl, float width, float speed,
                            bool reverse, bool smooth, float rollerRadius, float cleatHeight, float cleatSpacing);
        void buildPathWall(const std::vector<conveyor::Waypoint>& ctrl, float height, bool smooth);

        void load();           // build everything from the layout (or a default)
        void preSubstep(float dt);// belt drag + cleat march
        void postSubstep();       // teleport belts back

        threepp::PhysxWorld& world_;
        threepp::Scene& scene_;
        std::filesystem::path convDir_;
        threepp::USDLoader usdLoader_;

        ::physx::PxMaterial* beltMat_ = nullptr;
        std::shared_ptr<threepp::MeshStandardMaterial> beltVisualMat_, rollerVisualMat_, cleatVisualMat_, wallVisualMat_;
        std::shared_ptr<threepp::DataTexture> beltTexture_;

        std::unordered_map<std::string, std::string> typeOverrides_;
        std::unordered_map<std::string, conveyor::ModelTemplate> convTemplates_;

        std::vector<Belt> belts_;
        std::vector<threepp::Vector3> inlets_;
        std::vector<AnimatedBelt> beltVisuals_;
        std::vector<RollerBank> rollerBanks_;
        std::vector<CleatTrack> cleatTracks_;
    };

}// namespace fishsim

#endif// THREEPP_FISH_CONVEYOR_SYSTEM_HPP
