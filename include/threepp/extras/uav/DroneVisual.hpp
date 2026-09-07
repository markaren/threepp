// Primitive-built X-quad visual (hull box, crossed booms, rotor discs, amber
// nose marker). The ROOT mesh carries Box geometry sized to the airframe so
// PhysxWorld::add() can infer the collider from it directly; decorations are
// children and never touch physics.
//
// Authoring frame (must match FrameConv): forward = -Z, right = +X, up = +Y.

#ifndef THREEPP_EXTRAS_UAV_DRONEVISUAL_HPP
#define THREEPP_EXTRAS_UAV_DRONEVISUAL_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

namespace threepp::uav {

    class DroneVisual {
    public:
        static constexpr float hullX = 0.32f, hullY = 0.10f, hullZ = 0.32f;

        explicit DroneVisual(float armLength = 0.25f) {
            using namespace threepp;

            auto bodyMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0.16f, 0.17f, 0.19f)).roughness(0.6f));
            auto boomMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0.35f, 0.36f, 0.38f)).roughness(0.5f));
            auto rotorMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0.10f, 0.10f, 0.10f)).roughness(0.35f));
            auto noseMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(1.f, 0.65f, 0.05f)).roughness(0.4f));

            root_ = Mesh::create(BoxGeometry::create(hullX, hullY, hullZ), bodyMat);
            root_->castShadow = true;

            const float a = armLength / std::sqrt(2.f);
            const float boomLen = 2.f * armLength + 0.10f;

            // Two crossed booms at +-45 deg in the XZ plane.
            for (const float angle : {math::PI / 4.f, -math::PI / 4.f}) {
                auto boom = Mesh::create(BoxGeometry::create(0.03f, 0.02f, boomLen), boomMat);
                boom->rotation.y = angle;
                boom->position.y = 0.05f;
                boom->castShadow = true;
                root_->add(boom);
            }

            // Nose marker on -Z (forward): the one asymmetry that makes yaw
            // readable from the chase camera.
            auto nose = Mesh::create(BoxGeometry::create(0.06f, 0.02f, 0.10f), noseMat);
            nose->position.set(0.f, 0.065f, -hullZ / 2.f - 0.02f);
            root_->add(nose);

            // MeshBASIC, not Standard: the Vulkan deferred renderer shades
            // Standard materials through the opaque G-buffer where opacity is
            // meaningless (the disc rendered solid); unlit transparent meshes
            // go through its blended overlay pass, matching GL.
            auto blurMat = MeshBasicMaterial::create();
            blurMat->color = Color(0.22f, 0.22f, 0.24f);
            blurMat->transparent = true;
            blurMat->opacity = 0.f;// faded in with throttle (shared by all four)
            blurMat->depthWrite = false;
            blurMat_ = blurMat.get();

            // Propellers at the motor positions; layout and spin direction must
            // agree with QuadSim's motor table (M1 FR, M2 BL, M3 FL, M4 BR).
            // Each is a pivot group carrying two pitched blades; a translucent
            // disc stands in for motion blur once the motors spin up.
            const std::array<std::array<float, 2>, 4> xz{{{a, -a}, {-a, a}, {-a, -a}, {a, a}}};
            spinDir_ = {1.f, 1.f, -1.f, -1.f};
            for (int i = 0; i < 4; ++i) {
                auto hub = Mesh::create(CylinderGeometry::create(0.014f, 0.02f, 0.05f), rotorMat);
                hub->position.set(xz[i][0], 0.075f, xz[i][1]);
                root_->add(hub);

                auto prop = Group::create();
                prop->position.set(xz[i][0], 0.105f, xz[i][1]);
                for (int b = 0; b < 2; ++b) {
                    // Blade: long thin plate, offset so it sweeps a 0.13 m
                    // radius, pitched about its long axis like a real prop
                    // (opposite pitch for opposite spin so the airfoil leads).
                    auto blade = Mesh::create(BoxGeometry::create(0.115f, 0.0035f, 0.026f), rotorMat);
                    blade->position.x = (b == 0 ? 1.f : -1.f) * 0.0675f;
                    blade->rotation.x = (b == 0 ? 1.f : -1.f) * spinDir_[i] * 0.30f;
                    blade->castShadow = true;
                    prop->add(blade);
                }
                auto blur = Mesh::create(CylinderGeometry::create(0.13f, 0.13f, 0.002f), blurMat);
                blur->visible = false;// only while spinning fast (see setMotors)
                prop->add(blur);
                blurs_[i] = blur.get();
                root_->add(prop);
                props_[i] = prop.get();
            }
        }

        /// The physics/pose root; add this to the scene and hand it to QuadSim.
        [[nodiscard]] std::shared_ptr<threepp::Mesh> root() const { return root_; }

        /// Spin the propellers from the filtered throttles (visual only).
        void setMotors(const float throttle[4], float dtRender) {
            float maxT = 0.f;
            for (int i = 0; i < 4; ++i) {
                props_[i]->rotation.y +=
                        spinDir_[i] * (2.f + 110.f * throttle[i]) * dtRender;
                blurs_[i]->visible = throttle[i] > 0.25f;
                maxT = std::max(maxT, throttle[i]);
            }
            // Blades smear into a translucent disc as the motors spin up.
            blurMat_->opacity = std::min(maxT * 0.9f, 0.45f);
        }

    private:
        std::shared_ptr<threepp::Mesh> root_;
        std::array<threepp::Group*, 4> props_{};
        std::array<threepp::Mesh*, 4> blurs_{};
        std::array<float, 4> spinDir_{};
        threepp::MeshBasicMaterial* blurMat_ = nullptr;
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_DRONEVISUAL_HPP
