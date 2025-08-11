
#include "Sphero.hpp"

#include "threepp/extras/core/Shape.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <iostream>

using namespace threepp;

namespace {

    float shortest_signed_angle_path(float angle1, float angle2) {
        float diff = angle2 - angle1;
        diff = std::fmod(diff + math::PI, 2 * math::PI) - math::PI;

        return diff;
    }

    auto createTrack(float width, float height, float length) {

        CylinderGeometry::Params params;
        params.radiusBottom = params.radiusTop = height / 2;
        params.height = width;
        params.openEnded = true;
        params.thetaLength = math::PI;
        params.thetaStart = math::PI / 2;

       const  auto startCap = CylinderGeometry::create(params);
        startCap->applyMatrix4(Matrix4().makeTranslation(0, 0, -length / 2 + (height / 2)).premultiply(Matrix4().makeRotationZ(math::PI / 2)));

        params.thetaStart = -math::PI / 2;
        const auto endCap = CylinderGeometry::create(params);
        endCap->applyMatrix4(Matrix4().makeTranslation(0, 0, length / 2 - (height / 2)).premultiply(Matrix4().makeRotationZ(math::PI / 2)));

        const auto top = PlaneGeometry::create(length - (height), width);
        top->applyMatrix4(Matrix4().makeRotationX(math::PI / 2).premultiply(Matrix4().makeRotationY(math::PI / 2)));
        top->translate(0, height / 2, 0);

        const auto bottom = PlaneGeometry::create(length - height, width);
        bottom->applyMatrix4(Matrix4().makeRotationX(math::PI / 2).premultiply(Matrix4().makeRotationY(-math::PI / 2)));
        bottom->translate(0, -height / 2, 0);

        std::vector<std::shared_ptr<BufferGeometry>> geometries{
                startCap,
                endCap,
                top,
                bottom};

        return mergeBufferGeometries(geometries);
    }

    auto innerShape(float width, float height, float length) {
        Shape shape;
        shape.moveTo(-length / 2, height / 2)
                .lineTo(length / 2, height / 2)
                .arc(length / 2, -height / 2, height / 2, math::PI / 2, -math::PI / 2, true)
                .lineTo(-length / 2, -height / 2)
                .arc(-length / 2, height / 2, height / 2, -math::PI / 2, math::PI / 2, true);

        ExtrudeGeometry::Options opts;
        opts.depth = width;
        auto geometry = ExtrudeGeometry::create({shape}, opts);
        geometry->center();
        geometry->applyMatrix4(Matrix4().makeRotationY(-math::PI / 2));

        return geometry;
    }
}// namespace

Sphero::Sphero()
    : tofReadings_{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()} {

    TextureLoader tl;
    const auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/rubber_texture.jpg");
    tex->wrapS = tex->wrapT = threepp::TextureWrapping::Repeat;

    auto body = BoxGeometry::create(1, 0.5, 2);
    auto bodyMaterial = MeshStandardMaterial::create();
    bodyMaterial->color = Color::red;

    Vector3 trackSize(0.5, 0.5, 2);

    auto leftTrackMaterial = MeshStandardMaterial::create();
    leftTrackMaterial->side = Side::Double;
    leftTrackMaterial->map = tex;
    auto leftTrack = Mesh::create(createTrack(trackSize.x, trackSize.y, trackSize.z), leftTrackMaterial);
    leftTrack->name = "leftTrack";

    auto rightTrackMaterial = MeshStandardMaterial::create();
    rightTrackMaterial->side = Side::Double;
    rightTrackMaterial->map = tex->clone();
    rightTrackMaterial->map->needsUpdate();
    auto rightTrack = Mesh::create(createTrack(trackSize.x, trackSize.y, trackSize.z), rightTrackMaterial);
    rightTrack->name = "rightTrack";

    auto innerTrackRight = Mesh::create(innerShape(0.1, trackSize.y * 0.5, trackSize.z * 0.35));
    innerTrackRight->position.x = 0.76;
    add(innerTrackRight);

    auto innerTrackLeft = Mesh::create(innerShape(0.1, trackSize.y * 0.5, trackSize.z * 0.35));
    innerTrackLeft->position.x = -0.76;
    add(innerTrackLeft);

    float sep = 0.76f;
    leftTrack->position.x = -sep;
    rightTrack->position.x = sep;

    auto sphero = Mesh::create(body, bodyMaterial);
    sphero->add(leftTrack);
    sphero->add(rightTrack);

    camera_ = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
    camera_->position.set(0, 1, 0.5f);
    camera_->rotation.y = math::PI;

    auto tofMaterial = MeshBasicMaterial::create({{"color", Color::green}});

    auto tof1Geometry = BufferGeometry::create();
    tof1Geometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 0}, 3));
    auto tofSensor1 = Line::create(tof1Geometry, tofMaterial);
    tofSensor1->name = "tof1";
    tofSensor1->position.z = 1.1;
    tofSensor1->visible = false;
    tofSensor1->layers.set(1);

    auto tof2Geometry = BufferGeometry::create();
    tof2Geometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 0}, 3));
    auto tofSensor2 = Line::create(tof2Geometry, tofMaterial);
    tofSensor2->position.z = -1.1;
    tofSensor2->rotation.y = math::PI;
    tofSensor2->name = "tof2";
    tofSensor2->visible = false;
    tofSensor2->layers.set(1);

    add(tofSensor1);
    add(tofSensor2);

    add(sphero);

    Object3D::add(*camera_);

    raycaster_.params.lineThreshold = 0.01f;
    raycaster_.layers.set(0);
}

PerspectiveCamera &Sphero::camera() {

    return *camera_;
}

void Sphero::update(float dt) {

    updateTofMeasurements();

    if (tofReadings_.first < 0.5 && translationDelta_ > 0) {
        translationDelta_ = 0;
        rotationDelta_ = 0;
    } else if (tofReadings_.second < 0.5 && translationDelta_ < 0) {
        translationDelta_ = 0;
        rotationDelta_ = 0;
    }

    translateZ(translationDelta_ * dt);
    rotation.y += rotationDelta_ * dt;

    translationDelta_ = 0;
    rotationDelta_ = 0;
}

void Sphero::driveRaw(uint8_t leftMode, uint8_t leftSpeed, uint8_t rightMode, uint8_t rightSpeed) {

    float leftSpeedModifier, rightSpeedModifier;

    switch (leftMode) {
        case 0x0: {
            leftSpeedModifier = 0;
        } break;
        case 0x1: {
            leftSpeedModifier = 1;
        } break;
        case 0x2: {
            leftSpeedModifier = -1;
        } break;
    }

    switch (rightMode) {
        case 0x0: {
            rightSpeedModifier = 0;
        } break;
        case 0x1: {
            rightSpeedModifier = 1;
        } break;
        case 0x2: {
            rightSpeedModifier = -1;
        } break;
    }

    translationDelta_ = (((float(leftSpeed) * leftSpeedModifier + float(rightSpeed) * rightSpeedModifier) / 2) / 255) * maxSpeed_;
    rotationDelta_ = ((float(rightSpeed) * rightSpeedModifier) - (float(leftSpeed) * leftSpeedModifier)) / 255;
}

void Sphero::driveWithHeading(uint8_t speed, uint16_t heading, Flags flags) {

    auto currentRotationY = rotation.y;
    float targetRotationY = float(heading) * math::DEG2RAD;

    rotationDelta_ = shortest_signed_angle_path(currentRotationY, targetRotationY) * math::RAD2DEG;
    if (flags.isFlagSet(Flags::FastTurnMode)) {
        rotationDelta_ *= 3;
    }

    translationDelta_ = float(speed / 255) * maxSpeed_ * (flags.isFlagSet(Flags::DriveReverse) ? -1.f : 1.f);
    if (flags.isFlagSet(Flags::Boost)) {
        translationDelta_ *= 3;
    }
}

std::pair<float, float> Sphero::getTofMeasurements() {

    return tofReadings_;
}

void Sphero::updateTofMeasurements() {

    tofReadings_.first = getTofMeasurement("tof1");
    tofReadings_.second = getTofMeasurement("tof2");
}

float Sphero::getTofMeasurement(const std::string &tofId) {

    float reading = std::numeric_limits<float>::infinity();

    static Vector3 worldPos, worldDir;
    auto tof = getObjectByName<Line>(tofId);
    tof->visible = false;
    tof->getWorldPosition(worldPos);
    tof->getWorldDirection(worldDir);

    raycaster_.set(worldPos, worldDir);
    auto intersects = raycaster_.intersectObject(*parent, true);
    if (!intersects.empty()) {
        const auto &intersect = intersects.front();

        if (intersect.distance <= 10) {

            reading = intersect.distance;
            tof->visible = true;

            auto pos = tof->geometry()->getAttribute<float>("position");
            pos->setZ(1, intersect.distance);
            pos->needsUpdate();
        }
    }

    return reading;
}
