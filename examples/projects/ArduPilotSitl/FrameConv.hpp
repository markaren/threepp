// The ONE place where ArduPilot's frames meet threepp's. Everything crossing
// the bridge goes through these functions; nothing else is allowed to flip a
// sign. Frame bugs are the classic failure mode of SITL backends — if the
// copter flies mirrored or flips on takeoff, audit THIS file first.
//
// ArduPilot world frame: NED — x North, y East, z Down.
// ArduPilot body frame:  FRD — x Forward, y Right, z Down.
// threepp world frame:   three.js — right-handed, +Y up.
// Drone model authoring: forward = -Z, right = +X, up = +Y (three.js "camera
// forward" convention; DroneVisual puts the nose marker on -Z).
//
// Chosen mapping (a proper rotation, det +1):
//     North -> -Z      East -> +X      Down -> -Y
//
// As a matrix T (columns are N, E, D expressed in threepp axes):
//         [  0  1  0 ]                 tp.x =  ned.y (E)
//     T = [  0  0 -1 ]    i.e.         tp.y = -ned.z (-D)
//         [ -1  0  0 ]                 tp.z = -ned.x (-N)
//
// The SAME matrix maps body FRD -> node-local (forward -Z, right +X, belly
// -Y), so one pair of vector functions serves world vectors and body vectors,
// and attitudes convert by conjugation: q_tp = qT * q_ned * qT^-1.

#ifndef THREEPP_EXAMPLE_SITL_FRAMECONV_HPP
#define THREEPP_EXAMPLE_SITL_FRAMECONV_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <cmath>

namespace sitl::frame {

    /// NED vector -> threepp vector (positions, velocities; also FRD body
    /// vector -> drone-node-local vector).
    inline threepp::Vector3 nedToTp(double n, double e, double d) {
        return {static_cast<float>(e), static_cast<float>(-d), static_cast<float>(-n)};
    }

    /// threepp vector -> NED (out params keep double precision; also
    /// node-local -> FRD for IMU samples).
    inline void tpToNed(const threepp::Vector3& v, double& n, double& e, double& d) {
        n = -v.z;
        e = v.x;
        d = -v.y;
    }

    /// Ground-query helper: threepp XZ ground-plane coordinates -> N,E.
    inline void tpXZtoNE(float x, float z, double& n, double& e) {
        n = -z;
        e = x;
    }

    /// The basis-change quaternion qT for matrix T above (built once).
    inline const threepp::Quaternion& qT() {
        static const threepp::Quaternion q = [] {
            threepp::Matrix4 m;
            // three.js Matrix4::set takes ROW-major arguments.
            m.set(0.f, 1.f, 0.f, 0.f,
                  0.f, 0.f, -1.f, 0.f,
                  -1.f, 0.f, 0.f, 0.f,
                  0.f, 0.f, 0.f, 1.f);
            threepp::Quaternion r;
            r.setFromRotationMatrix(m);
            return r;
        }();
        return q;
    }

    /// Body-FRD->NED attitude quaternion -> threepp node world quaternion.
    inline threepp::Quaternion nedAttToTp(double qw, double qx, double qy, double qz) {
        threepp::Quaternion q(static_cast<float>(qx), static_cast<float>(qy),
                              static_cast<float>(qz), static_cast<float>(qw));
        threepp::Quaternion inv = qT();
        inv.conjugate();
        // q_tp = qT * q_ned * qT^-1
        q.premultiply(qT()).multiply(inv);
        return q;
    }

    /// threepp node world quaternion -> body-FRD->NED quaternion components.
    inline void tpAttToNed(const threepp::Quaternion& qTp,
                           double& qw, double& qx, double& qy, double& qz) {
        threepp::Quaternion inv = qT();
        inv.conjugate();
        // q_ned = qT^-1 * q_tp * qT
        threepp::Quaternion q = qTp;
        q.premultiply(inv).multiply(qT());
        qw = q.w;
        qx = q.x;
        qy = q.y;
        qz = q.z;
    }

    /// Aerospace ZYX Euler (roll, pitch, yaw) from a body-FRD->NED quaternion.
    inline void nedQuatToRpy(double qw, double qx, double qy, double qz,
                             double& roll, double& pitch, double& yaw) {
        constexpr double halfPi = 1.57079632679489661923;
        const double sinp = 2.0 * (qw * qy - qz * qx);
        roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
        pitch = std::abs(sinp) >= 1.0 ? std::copysign(halfPi, sinp) : std::asin(sinp);
        yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    }

}// namespace sitl::frame

#endif// THREEPP_EXAMPLE_SITL_FRAMECONV_HPP
