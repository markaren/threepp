
#include "threepp/extras/editor/VehicleConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    VehicleConfig::Drive driveFrom(std::string_view text, VehicleConfig::Drive fallback) {

        if (text == "direct") return VehicleConfig::Drive::Direct;
        if (text == "engine") return VehicleConfig::Drive::Engine;
        return fallback;
    }

    const char* driveToken(VehicleConfig::Drive drive) {

        switch (drive) {
            case VehicleConfig::Drive::Direct: return "direct";
            case VehicleConfig::Drive::Engine: return "engine";
        }
        return "direct";
    }

    VehicleConfig::Driven drivenFrom(std::string_view text, VehicleConfig::Driven fallback) {

        if (text == "awd") return VehicleConfig::Driven::All;
        if (text == "rwd") return VehicleConfig::Driven::Rear;
        if (text == "fwd") return VehicleConfig::Driven::Front;
        return fallback;
    }

    const char* drivenToken(VehicleConfig::Driven driven) {

        switch (driven) {
            case VehicleConfig::Driven::All: return "awd";
            case VehicleConfig::Driven::Rear: return "rwd";
            case VehicleConfig::Driven::Front: return "fwd";
        }
        return "awd";
    }

    // Union of every mesh's geometry bounds under `node`, with each mesh's
    // world matrix mapped through `intoFrame` — pass identity for world-space
    // bounds, a frame inverse for frame-space ones. Empty when nothing under
    // the node has measurable geometry.
    Box3 boundsInFrame(Object3D& node, const Matrix4& intoFrame) {

        Box3 bounds;
        bounds.makeEmpty();
        node.traverseType<Mesh>([&](Mesh& mesh) {
            const auto geometry = mesh.geometry();
            if (!geometry) return;
            if (!geometry->boundingBox) geometry->computeBoundingBox();
            if (!geometry->boundingBox) return;
            Matrix4 rel(intoFrame);
            rel.multiply(*mesh.matrixWorld);
            Box3 b = *geometry->boundingBox;
            b.applyMatrix4(rel);
            bounds.union_(b);
        });
        return bounds;
    }

}// namespace


const char* VehicleConfig::label(Drive drive) {

    switch (drive) {
        case Drive::Direct: return "Direct";
        case Drive::Engine: return "Engine";
    }
    return "Direct";
}

const char* VehicleConfig::label(Driven driven) {

    switch (driven) {
        case Driven::All: return "All Wheels";
        case Driven::Rear: return "Rear Wheels";
        case Driven::Front: return "Front Wheels";
    }
    return "All Wheels";
}

std::string VehicleConfig::encode() const {

    std::string out;
    out += "drive=";
    out += driveToken(drive);
    out += ";driven=";
    out += drivenToken(driven);
    // Every field rides along whatever the flags are, so flipping auto (or
    // drive type) and back does not quietly reset the ones the other mode
    // hides — the same rule PhysicsConfig and JointConfig follow.
    out += ";auto=";
    out += (autoGeometry ? "1" : "0");
    out += ";chassiswidth=";
    out += number(chassisWidth);
    out += ";chassisheight=";
    out += number(chassisHeight);
    out += ";chassislength=";
    out += number(chassisLength);
    out += ";wheelradius=";
    out += number(wheelRadius);
    out += ";wheelwidth=";
    out += number(wheelWidth);
    out += ";track=";
    out += number(trackWidth);
    out += ";wheelbase=";
    out += number(wheelbase);
    out += ";suspensiony=";
    out += number(suspensionY);
    out += ";mass=";
    out += number(mass);
    out += ";travel=";
    out += number(suspensionTravel);
    out += ";stiffness=";
    out += number(suspensionStiffness);
    out += ";damping=";
    out += number(suspensionDamping);
    out += ";friction=";
    out += number(tireFriction);
    out += ";brake=";
    out += number(maxBrakeTorque);
    out += ";steer=";
    out += number(maxSteerAngle);
    out += ";throttle=";
    out += number(throttleTorque);
    return out;
}

std::optional<VehicleConfig> VehicleConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    VehicleConfig config;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "drive") {
            config.drive = driveFrom(value, config.drive);
        } else if (key == "driven") {
            config.driven = drivenFrom(value, config.driven);
        } else if (key == "auto") {
            config.autoGeometry = toInt(value, config.autoGeometry ? 1 : 0) != 0;
        } else if (key == "chassiswidth") {
            config.chassisWidth = toFloat(value, config.chassisWidth);
        } else if (key == "chassisheight") {
            config.chassisHeight = toFloat(value, config.chassisHeight);
        } else if (key == "chassislength") {
            config.chassisLength = toFloat(value, config.chassisLength);
        } else if (key == "wheelradius") {
            config.wheelRadius = toFloat(value, config.wheelRadius);
        } else if (key == "wheelwidth") {
            config.wheelWidth = toFloat(value, config.wheelWidth);
        } else if (key == "track") {
            config.trackWidth = toFloat(value, config.trackWidth);
        } else if (key == "wheelbase") {
            config.wheelbase = toFloat(value, config.wheelbase);
        } else if (key == "suspensiony") {
            config.suspensionY = toFloat(value, config.suspensionY);
        } else if (key == "mass") {
            config.mass = toFloat(value, config.mass);
        } else if (key == "travel") {
            config.suspensionTravel = toFloat(value, config.suspensionTravel);
        } else if (key == "stiffness") {
            config.suspensionStiffness = toFloat(value, config.suspensionStiffness);
        } else if (key == "damping") {
            config.suspensionDamping = toFloat(value, config.suspensionDamping);
        } else if (key == "friction") {
            config.tireFriction = toFloat(value, config.tireFriction);
        } else if (key == "brake") {
            config.maxBrakeTorque = toFloat(value, config.maxBrakeTorque);
        } else if (key == "steer") {
            config.maxSteerAngle = toFloat(value, config.maxSteerAngle);
        } else if (key == "throttle") {
            config.throttleTorque = toFloat(value, config.throttleTorque);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<VehicleConfig> VehicleConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    auto config = decode(std::any_cast<const std::string&>(it->second));
    if (config) {
        for (int i = 0; i < 4; ++i) {
            config->wheels[i] = readString(object, wheelKeys[i]);
        }
    }
    return config;
}

bool VehicleConfig::isVehicle(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

void VehicleConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
    for (int i = 0; i < 4; ++i) {
        if (wheels[i].empty()) {
            object.userData.erase(wheelKeys[i]);
        } else {
            object.userData[wheelKeys[i]] = wheels[i];
        }
    }
}

void VehicleConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
    for (const auto* key : wheelKeys) {
        object.userData.erase(key);
    }
}

VehicleGeometry VehicleConfig::derived(Object3D& root) const {

    VehicleGeometry geo;

    // --- resolve the four picks -------------------------------------------
    for (int i = 0; i < 4; ++i) {
        if (wheels[i].empty()) {
            geo.problem = std::string(wheelLabels[i]) + " wheel is not picked";
            return geo;
        }
        auto* node = root.getObjectByName(wheels[i]);
        if (!node || node == &root) {
            geo.problem = std::string(wheelLabels[i]) + " wheel \"" + wheels[i] +
                          "\" is not a node under the model";
            return geo;
        }
        for (int j = 0; j < i; ++j) {
            if (geo.wheels[j] == node) {
                geo.problem = std::string(wheelLabels[j]) + " and " + wheelLabels[i] +
                              " both resolve to \"" + wheels[i] + "\"";
                return geo;
            }
        }
        geo.wheels[i] = node;
    }

    root.updateWorldMatrix(true, true);

    // --- pass 1: world hub estimates, to orient the chassis frame ----------
    // The frame is built FROM the picks: up is the root's own +Y, forward is
    // rear axle towards front axle. That is what makes a model facing -Z (or
    // +X) drive the way it looks, with no facing convention imposed.
    const Matrix4 identity;
    std::array<Vector3, 4> hubWorld;
    for (int i = 0; i < 4; ++i) {
        const Box3 bounds = boundsInFrame(*geo.wheels[i], identity);
        if (bounds.isEmpty()) {
            geo.problem = std::string(wheelLabels[i]) + " wheel \"" + wheels[i] +
                          "\" has no geometry to measure";
            return geo;
        }
        hubWorld[i] = bounds.getCenter();
    }

    Vector3 rootPos, rootScl;
    Quaternion rootRot;
    root.matrixWorld->decompose(rootPos, rootRot, rootScl);
    Vector3 up(0.f, 1.f, 0.f);
    up.applyQuaternion(rootRot);
    up.normalize();

    Vector3 frontMid;
    frontMid.addVectors(hubWorld[0], hubWorld[1]).multiplyScalar(0.5f);
    Vector3 rearMid;
    rearMid.addVectors(hubWorld[2], hubWorld[3]).multiplyScalar(0.5f);
    Vector3 forward;
    forward.subVectors(frontMid, rearMid);
    Vector3 lift(up);
    lift.multiplyScalar(forward.dot(up));
    forward.sub(lift);// flatten: the wheelbase is a ground-plane distance
    if (forward.length() < 0.05f) {
        geo.problem = "the front and rear wheels sit on top of each other - check the picks";
        return geo;
    }
    forward.normalize();
    Vector3 right;
    right.crossVectors(up, forward).normalize();
    Vector3 upOrtho;
    upOrtho.crossVectors(forward, right).normalize();

    Matrix4 basis;
    basis.makeBasis(right, upOrtho, forward);
    Quaternion frameRot;
    frameRot.setFromRotationMatrix(basis);
    Matrix4 frame;
    frame.compose(rootPos, frameRot, Vector3(1.f, 1.f, 1.f));
    Matrix4 frameInv(frame);
    frameInv.invert();

    // --- pass 2: measure in the frame --------------------------------------
    // Frame-space bounds, so a car parked at any angle in the scene measures
    // the same as one on the world axes.
    float radius = 0.f;
    float width = 0.f;
    std::array<Vector3, 4> hubs;
    for (int i = 0; i < 4; ++i) {
        const Box3 bounds = boundsInFrame(*geo.wheels[i], frameInv);
        const Vector3 size = bounds.getSize();
        hubs[i] = bounds.getCenter();
        // The vertical extent is a diameter whichever way the axle points;
        // the flattest horizontal extent is the width.
        radius += 0.5f * size.y;
        width += std::min(size.x, size.z);
    }
    radius /= 4.f;
    width /= 4.f;
    if (!(radius > 0.005f)) {
        geo.problem = "the picked wheels measure no radius";
        return geo;
    }

    const Box3 model = boundsInFrame(root, frameInv);
    const Vector3 modelSize = model.getSize();
    const Vector3 modelCentre = model.getCenter();

    const float frontZ = 0.5f * (hubs[0].z + hubs[1].z);
    const float rearZ = 0.5f * (hubs[2].z + hubs[3].z);
    geo.wheelbase = std::abs(frontZ - rearZ);
    geo.trackWidth = 0.5f * (std::abs(hubs[0].x - hubs[1].x) +
                             std::abs(hubs[2].x - hubs[3].x));
    if (geo.wheelbase < 0.05f) {
        geo.problem = "the picked wheels give no wheelbase - front and rear share an axle";
        return geo;
    }
    if (geo.trackWidth < 0.05f) {
        geo.problem = "the picked wheels give no track width - left and right coincide";
        return geo;
    }

    // The chassis actor's origin: X and Z centred on the AXLES (PhysX places
    // the wheels symmetrically about it, so this keeps the contact patches
    // under the visual wheels even on a model with body overhang), Y at the
    // model's vertical centre (the chassis box is centred on the origin).
    const float cx = 0.25f * (hubs[0].x + hubs[1].x + hubs[2].x + hubs[3].x);
    const float cz = 0.5f * (frontZ + rearZ);
    const Vector3 centre(cx, modelCentre.y, cz);

    Vector3 origin(centre);
    origin.applyMatrix4(frame);
    geo.position = origin;
    geo.rotation = frameRot;

    float hubY = 0.f;
    for (int i = 0; i < 4; ++i) {
        hubs[i].sub(centre);
        geo.hubs[i] = hubs[i];
        hubY += hubs[i].y;
    }
    hubY /= 4.f;

    geo.chassisWidth = std::max(modelSize.x, 0.1f);
    geo.chassisHeight = std::max(modelSize.y, 0.1f);
    geo.chassisLength = std::max(modelSize.z, 0.1f);
    geo.wheelRadius = radius;
    geo.wheelWidth = std::clamp(width, 0.02f, 4.f * radius);

    // The attachment is the wheel pose at MAX COMPRESSION; at rest the wheel
    // hangs (travel - static jounce) below it. Compensate, so the car RESTS
    // at its authored ride height instead of sagging on the first Play.
    const float travel = std::max(suspensionTravel, 0.f);
    const float jounceStatic = std::clamp(
            0.25f * std::max(mass, 1.f) * 9.81f / std::max(suspensionStiffness, 1.f),
            0.f, travel);
    geo.suspensionY = hubY + (travel - jounceStatic);

    geo.valid = true;
    return geo;
}
