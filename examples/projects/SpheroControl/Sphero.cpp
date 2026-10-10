
#include "Sphero.hpp"

#include "threepp/extras/core/Shape.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace threepp;

namespace {

    // The drive train, in metres. The tread is a belt round two wheels a side.
    constexpr float wheelRadius = 0.038f;
    constexpr float wheelWidth = 0.033f;
    constexpr float trackRadius = 0.043f;// wheel plus tread
    constexpr float treadWidth = 0.036f;
    constexpr float treadThickness = 0.005f;
    constexpr float axleZ = 0.065f;
    constexpr float trackX = 0.0745f;
    constexpr float trackWidth = 2 * trackX;
    constexpr int treadPads = 15;// pad rows round the belt
    constexpr float treadPerimeter = 4 * axleZ + 2 * math::PI * trackRadius;

    constexpr float motorTimeConstant = 0.12f;// [s]
    constexpr float headingGain = 8.f;        // turn rate per radian of heading error [1/s]
    constexpr float turnRate = 3.5f;          // [rad/s]
    constexpr float boostGain = 1.5f;

    constexpr float deckY = 0.068f;

    Shape roundedRect(float width, float height, float radius) {
        const float x = -width / 2, y = -height / 2;

        Shape shape;
        shape.moveTo(x + radius, y)
                .lineTo(x + width - radius, y)
                .absarc(x + width - radius, y + radius, radius, -math::PI / 2, 0)
                .lineTo(x + width, y + height - radius)
                .absarc(x + width - radius, y + height - radius, radius, 0, math::PI / 2)
                .lineTo(x + radius, y + height)
                .absarc(x + radius, y + height - radius, radius, math::PI / 2, math::PI)
                .lineTo(x, y + radius)
                .absarc(x + radius, y + radius, radius, math::PI, math::PI * 1.5f);

        return shape;
    }

    // Extrudes a profile drawn in (z, y) along x, centred on x.
    std::shared_ptr<BufferGeometry> extrudeAlongX(const Shape& profile, float width, float bevel) {
        ExtrudeGeometry::Options opts;
        opts.depth = width - 2 * bevel;
        opts.curveSegments = 10;
        opts.bevelEnabled = bevel > 0;
        opts.bevelThickness = bevel;
        opts.bevelSize = bevel;
        opts.bevelSegments = 2;

        auto geometry = ExtrudeGeometry::create({profile}, opts);
        geometry->translate(0, 0, -(width - 2 * bevel) / 2);
        geometry->rotateY(-math::PI / 2);

        return geometry;
    }

    // A width (x) by height (y) by length (z) block, its side profile rounded
    // by `radius` and the two side faces eased over by `bevel`.
    std::shared_ptr<BufferGeometry> roundedBlock(float width, float height, float length, float radius, float bevel = 0.002f) {

        return extrudeAlongX(roundedRect(length - 2 * bevel, height - 2 * bevel, radius - bevel), width, bevel);
    }

    // One side of the roll cage, in (z, y) from the deck up: a cab outline with two windows.
    Shape cageProfile() {
        Shape shape;
        shape.moveTo(-0.088f, 0)
                .bezierCurveTo(-0.088f, 0.03f, -0.07f, 0.046f, -0.045f, 0.046f)
                .lineTo(0.005f, 0.046f)
                .bezierCurveTo(0.03f, 0.046f, 0.055f, 0.02f, 0.072f, 0)
                .lineTo(-0.088f, 0);

        auto rear = std::make_shared<Path>();
        rear->moveTo(-0.076f, 0.006f)
                .bezierCurveTo(-0.076f, 0.026f, -0.064f, 0.038f, -0.045f, 0.038f)
                .lineTo(-0.026f, 0.038f)
                .lineTo(-0.026f, 0.006f)
                .lineTo(-0.076f, 0.006f);
        shape.holes.emplace_back(rear);

        auto front = std::make_shared<Path>();
        front->moveTo(-0.018f, 0.006f)
                .lineTo(-0.018f, 0.038f)
                .lineTo(0.003f, 0.038f)
                .bezierCurveTo(0.022f, 0.038f, 0.042f, 0.02f, 0.055f, 0.006f)
                .lineTo(-0.018f, 0.006f);
        shape.holes.emplace_back(front);

        return shape;
    }

    // The belt round the two wheels of one side, centred on the axle line. The
    // outer face carries the tread texture: u across the belt, v once per pad
    // row along it, so the texture can be scrolled to make the belt run. v
    // grows forwards along the top.
    std::shared_ptr<BufferGeometry> createTread() {
        struct Point {
            float z, y, nz, ny, s;
        };

        constexpr int arcSegments = 24;
        std::vector<Point> path;
        float s = 2 * axleZ;// the top run lies between the first two points
        for (int i = 0; i <= arcSegments; ++i) {
            const float phi = math::PI * static_cast<float>(i) / arcSegments;
            path.push_back({axleZ + trackRadius * std::sin(phi), trackRadius * std::cos(phi), std::sin(phi), std::cos(phi), s + trackRadius * phi});
        }
        s += math::PI * trackRadius + 2 * axleZ;
        for (int i = 0; i <= arcSegments; ++i) {
            const float phi = math::PI * static_cast<float>(i) / arcSegments;
            path.push_back({-axleZ - trackRadius * std::sin(phi), -trackRadius * std::cos(phi), -std::sin(phi), -std::cos(phi), s + trackRadius * phi});
        }
        path.insert(path.begin(), Point{-axleZ, trackRadius, 0, 1, 0});

        std::vector<float> positions, normals, uvs;
        std::vector<unsigned int> indices;
        const auto vertex = [&](float x, float y, float z, float nx, float ny, float nz, float u, float v) {
            positions.insert(positions.end(), {x, y, z});
            normals.insert(normals.end(), {nx, ny, nz});
            uvs.insert(uvs.end(), {u, v});
        };
        const auto strip = [&](unsigned int first, unsigned int count, bool flip) {
            for (unsigned int i = 0; i + 1 < count; ++i) {
                const unsigned int a = first + 2 * i;
                if (flip) {
                    indices.insert(indices.end(), {a, a + 2, a + 1, a + 1, a + 2, a + 3});
                } else {
                    indices.insert(indices.end(), {a, a + 1, a + 2, a + 1, a + 3, a + 2});
                }
            }
        };

        const auto count = static_cast<unsigned int>(path.size());
        const float half = treadWidth / 2;

        for (const auto& p : path) {
            const float v = p.s / treadPerimeter * treadPads;
            vertex(half, p.y, p.z, 0, p.ny, p.nz, 0, v);
            vertex(-half, p.y, p.z, 0, p.ny, p.nz, 1, v);
        }
        strip(0, count, false);

        // The belt's two edges. Their uv stays on the u = 0 line of the
        // texture, which runs between the pads.
        for (const float side : {1.f, -1.f}) {
            const auto first = static_cast<unsigned int>(positions.size() / 3);
            for (const auto& p : path) {
                vertex(side * half, p.y, p.z, side, 0, 0, 0, 0);
                vertex(side * half, p.y - p.ny * treadThickness, p.z - p.nz * treadThickness, side, 0, 0, 0, 0);
            }
            strip(first, count, side > 0);
        }

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        geometry->setIndex(std::move(indices));

        return geometry;
    }

    // One pad row of the tread, two round pads, staggered, as the three maps
    // that make them stand proud of the belt: colour (with the shade a pad
    // casts round its foot), a tangent-space normal map and roughness.
    struct TreadMaps {
        std::shared_ptr<Texture> color, normal, roughness;
    };

    TreadMaps createTreadMaps() {
        constexpr int size = 128;
        constexpr float padTop = 0.14f, padFoot = 0.215f;// radii, in tile widths
        constexpr float relief = 0.1f;                   // pad height, in tile widths

        const auto distance = [](float u, float v) {
            return std::min(std::hypot(u - 0.25f, v - 0.25f), std::hypot(u - 0.75f, v - 0.75f));
        };
        // 1 on a pad (a little domed), 0 on the belt
        const auto height = [&](float u, float v) {
            const float d = distance(u, v);
            const float dome = 1 - 0.2f * std::min(1.f, (d / padFoot) * (d / padFoot));
            return (1 - math::smoothstep(padTop, padFoot, d)) * dome;
        };

        const Color belt(0x8a9198), pad(0xa9b0b7);

        std::vector<unsigned char> colors(size * size * 4), normals(size * size * 4), roughnesses(size * size * 4);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / size;
                const float v = (static_cast<float>(y) + 0.5f) / size;
                const float d = distance(u, v);
                const float onPad = 1 - math::smoothstep(padTop, padFoot, d);
                const auto i = static_cast<size_t>(y * size + x) * 4;

                // The shade stops short of the tile's edge (0.25 from a pad's
                // centre), where the belt's edges take their colour from.
                Color c = belt;
                c.multiplyScalar(0.72f + 0.28f * math::smoothstep(padFoot - 0.01f, 0.25f, d));
                c.lerp(pad, onPad);
                colors[i + 0] = static_cast<unsigned char>(c.r * 255);
                colors[i + 1] = static_cast<unsigned char>(c.g * 255);
                colors[i + 2] = static_cast<unsigned char>(c.b * 255);
                colors[i + 3] = 255;

                const float e = 0.5f / size;
                Vector3 n(-relief * (height(u + e, v) - height(u - e, v)) / (2 * e),
                          -relief * (height(u, v + e) - height(u, v - e)) / (2 * e), 1);
                n.normalize();
                normals[i + 0] = static_cast<unsigned char>((n.x * 0.5f + 0.5f) * 255);
                normals[i + 1] = static_cast<unsigned char>((n.y * 0.5f + 0.5f) * 255);
                normals[i + 2] = static_cast<unsigned char>((n.z * 0.5f + 0.5f) * 255);
                normals[i + 3] = 255;

                // the pads are worn smoother than the belt between them
                const auto roughness = static_cast<unsigned char>((0.95f - 0.4f * onPad) * 255);
                roughnesses[i + 0] = roughnesses[i + 1] = roughnesses[i + 2] = roughness;
                roughnesses[i + 3] = 255;
            }
        }

        const auto texture = [](std::vector<unsigned char>& pixels, ColorSpace colorSpace) {
            auto texture = DataTexture::create(ImageData{std::move(pixels)}, size, size);
            texture->wrapS = texture->wrapT = TextureWrapping::Repeat;
            texture->magFilter = Filter::Linear;
            texture->minFilter = Filter::LinearMipmapLinear;
            texture->generateMipmaps = true;
            texture->anisotropy = 8;
            texture->colorSpace = colorSpace;
            texture->needsUpdate();

            return texture;
        };

        return {texture(colors, ColorSpace::sRGB), texture(normals, ColorSpace::NoColorSpace), texture(roughnesses, ColorSpace::NoColorSpace)};
    }

    std::shared_ptr<MeshStandardMaterial> plastic(const Color& color, float roughness = 0.4f) {
        auto material = MeshStandardMaterial::create();
        material->color = color;
        material->roughness = roughness;
        material->metalness = 0;

        return material;
    }

}// namespace

Sphero::Sphero()
    : tofFront_{nullptr, nullptr, nullptr, std::numeric_limits<float>::infinity()},
      tofRear_{nullptr, nullptr, nullptr, std::numeric_limits<float>::infinity()} {

    buildModel();

    // A Raspberry Pi camera: 4:3, about 62 degrees across. It looks along the
    // robot's +z, pitched a little down.
    camera_ = std::make_unique<PerspectiveCamera>(48.8f, 4.f / 3.f, 0.01f, 30.f);
    camera_->position.set(0, deckY + 0.059f, 0.012f);
    camera_->rotation.set(0.12f, math::PI, 0);
    Object3D::addRef(*camera_);

    raycaster_.layers.set(0);
    raycaster_.farPlane = tofRange;
}

Sphero::~Sphero() = default;

void Sphero::buildModel() {

    const auto white = plastic(0xdcdfe2);
    const auto grey = plastic(0xaeb4ba, 0.5f);
    const auto wheelWhite = plastic(0xe8eaec, 0.3f);
    const auto dish = plastic(0xdde0e3, 0.45f);
    const auto slot = plastic(0xaab0b6, 0.6f);
    const auto dark = plastic(0x17191c, 0.5f);
    const auto blue = plastic(0x2f8fe0, 0.35f);

    auto clear = plastic(0xf2f8ff, 0.08f);
    clear->transparent = true;
    clear->opacity = 0.38f;
    clear->depthWrite = false;

    ledMaterial_ = MeshBasicMaterial::create();
    ledMaterial_->color = Color(0x35ff6b);

    const auto part = [this](const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material, float x, float y, float z) {
        auto mesh = Mesh::create(geometry, material);
        mesh->position.set(x, y, z);
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        add(mesh);

        return mesh;
    };

    // Body, and the pod that fills each tread loop between its two wheels.
    part(roundedBlock(0.108f, 0.048f, 0.196f, 0.012f), white, 0, 0.044f, 0);
    part(roundedBlock(0.1f, 0.016f, 0.17f, 0.006f), grey, 0, 0.02f, 0);// underbody
    const auto pod = roundedBlock(0.031f, 0.04f, 0.046f, 0.008f);
    const auto podLed = BoxGeometry::create(0.002f, 0.003f, 0.014f);
    for (const float side : {1.f, -1.f}) {
        part(pod, white, side * 0.0725f, 0.044f, 0);
        part(podLed, ledMaterial_, side * 0.088f, 0.034f, 0)->castShadow = false;
    }

    // Power button, on the left pod.
    auto buttonRing = CylinderGeometry::create(0.0055f, 0.0055f, 0.0015f, 24);
    buttonRing->rotateZ(math::PI / 2);
    part(buttonRing, blue, 0.088f, 0.052f, 0);
    auto button = CylinderGeometry::create(0.0042f, 0.0042f, 0.002f, 24);
    button->rotateZ(math::PI / 2);
    part(button, wheelWhite, 0.0885f, 0.052f, 0);

    // Lights: a chevron either side of the nose, a bar either side of the tail.
    const float chevronAngle = std::atan2(0.009f, 0.016f);
    const auto chevronArm = BoxGeometry::create(0.0215f, 0.004f, 0.003f);
    const auto tailLight = BoxGeometry::create(0.022f, 0.004f, 0.003f);
    for (const float side : {1.f, -1.f}) {
        for (const float arm : {1.f, -1.f}) {
            auto mesh = part(chevronArm, ledMaterial_, side * 0.034f, 0.047f + arm * 0.0045f, 0.098f);
            mesh->rotation.z = arm * side * chevronAngle;
            mesh->castShadow = false;
        }
        part(tailLight, ledMaterial_, side * 0.034f, 0.047f, -0.098f)->castShadow = false;
    }
    part(BoxGeometry::create(0.024f, 0.006f, 0.002f), dark, 0, 0.036f, 0.0975f);// expansion port

    // Clear roll cage over the deck, and the badge under its windscreen.
    const auto cageSide = extrudeAlongX(cageProfile(), 0.005f, 0);
    for (const float side : {1.f, -1.f}) {
        part(cageSide, clear, side * 0.046f, deckY, 0)->castShadow = false;
    }
    const auto roofBar = BoxGeometry::create(0.087f, 0.006f, 0.012f);
    for (const float z : {-0.04f, -0.022f, 0.f}) {
        part(roofBar, clear, 0, deckY + 0.043f, z)->castShadow = false;
    }
    const auto sillBar = BoxGeometry::create(0.087f, 0.005f, 0.008f);
    for (const float z : {-0.082f, 0.064f}) {
        part(sillBar, clear, 0, deckY + 0.0025f, z)->castShadow = false;
    }
    part(roundedBlock(0.034f, 0.004f, 0.022f, 0.0015f, 0.001f), blue, 0, deckY + 0.001f, 0.052f);

    // Camera module on the roof.
    auto cameraModule = Group::create();
    cameraModule->position.set(0, deckY + 0.059f, 0);
    cameraModule->rotation.x = 0.12f;
    add(cameraModule);
    auto board = Mesh::create(BoxGeometry::create(0.026f, 0.024f, 0.008f), dark);
    board->castShadow = true;
    cameraModule->add(board);
    auto lensGeometry = CylinderGeometry::create(0.006f, 0.006f, 0.005f, 24);
    lensGeometry->rotateX(math::PI / 2);
    auto lens = Mesh::create(lensGeometry, plastic(0x0a0a0c, 0.15f));
    lens->position.z = 0.0065f;
    cameraModule->add(lens);

    // Tracks.
    const auto treadGeometry = createTread();
    auto wheelGeometry = CylinderGeometry::create(wheelRadius, wheelRadius, wheelWidth, 48);
    wheelGeometry->rotateZ(math::PI / 2);
    auto dishGeometry = CylinderGeometry::create(wheelRadius * 0.8f, wheelRadius * 0.8f, 0.002f, 48);
    dishGeometry->rotateZ(math::PI / 2);
    auto hubGeometry = CylinderGeometry::create(0.008f, 0.008f, 0.008f, 24);
    hubGeometry->rotateZ(math::PI / 2);
    const auto slotGeometry = BoxGeometry::create(0.002f, 0.016f, 0.0045f);

    const auto buildTrack = [&](Track& track, float side) {
        // Each track scrolls its own set of maps.
        const auto maps = createTreadMaps();
        track.treadMaps = {maps.color, maps.normal, maps.roughness};

        auto treadMaterial = plastic(Color::white, 1);
        treadMaterial->map = maps.color;
        treadMaterial->normalMap = maps.normal;
        treadMaterial->roughnessMap = maps.roughness;
        treadMaterial->side = Side::Double;
        part(treadGeometry, treadMaterial, side * trackX, trackRadius, 0);

        const float face = side * wheelWidth / 2;
        for (int i = 0; i < 2; ++i) {
            auto wheel = Group::create();
            wheel->position.set(side * trackX, trackRadius, i == 0 ? axleZ : -axleZ);
            add(wheel);

            const auto wheelPart = [&](const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material, float x) {
                auto mesh = Mesh::create(geometry, material);
                mesh->position.x = x;
                mesh->castShadow = true;
                mesh->receiveShadow = true;
                wheel->add(mesh);

                return mesh;
            };

            wheelPart(wheelGeometry, wheelWhite, 0);
            wheelPart(dishGeometry, dish, face + side * 0.0005f);
            wheelPart(hubGeometry, wheelWhite, face + side * 0.003f);
            for (int k = 0; k < 4; ++k) {
                const float angle = math::PI / 2 * static_cast<float>(k);
                auto mesh = wheelPart(slotGeometry, slot, face + side * 0.0012f);
                mesh->position.y = std::cos(angle) * 0.021f;
                mesh->position.z = std::sin(angle) * 0.021f;
                mesh->rotation.x = angle;
            }

            track.wheels[i] = wheel;
        }
    };
    buildTrack(left_, 1);
    buildTrack(right_, -1);

    // Time-of-flight sensors. Each mount looks along its own +z.
    auto beamGeometry = BufferGeometry::create();
    beamGeometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 1}, 3));
    auto beamMaterial = LineBasicMaterial::create();
    beamMaterial->color = Color(0xff3b30);
    auto dotMaterial = MeshBasicMaterial::create();
    dotMaterial->color = Color(0xff3b30);
    const auto dotGeometry = SphereGeometry::create(0.006f, 16, 8);
    const auto sensorGeometry = BoxGeometry::create(0.016f, 0.008f, 0.003f);

    const auto buildTof = [&](Tof& tof, float direction) {
        part(sensorGeometry, dark, 0, 0.047f, direction * 0.0985f);

        tof.mount = Object3D::create();
        tof.mount->position.set(0, 0.047f, direction * 0.101f);
        tof.mount->rotation.y = direction > 0 ? 0 : math::PI;
        add(tof.mount);

        tof.beam = Line::create(beamGeometry, beamMaterial);
        tof.beam->frustumCulled = false;// it is stretched by its scale
        tof.beam->layers.set(1);
        tof.beam->visible = false;
        tof.mount->add(tof.beam);

        tof.dot = Mesh::create(dotGeometry, dotMaterial);
        tof.dot->layers.set(1);
        tof.dot->visible = false;
        tof.mount->add(tof.dot);
    };
    buildTof(tofFront_, 1);
    buildTof(tofRear_, -1);
}

PerspectiveCamera& Sphero::camera() {

    return *camera_;
}

void Sphero::update(float dt) {

    commandAge_ += dt;
    if (commandAge_ > commandTimeout) command_ = Command::None;

    float targetLeft = 0, targetRight = 0;
    if (command_ == Command::Raw) {
        targetLeft = rawLeft_;
        targetRight = rawRight_;
    } else if (command_ == Command::Heading) {
        const float error = std::remainder(headingYaw_ - rotation.y, math::TWO_PI);
        const float turn = std::clamp(headingGain * error, -headingTurnRate_, headingTurnRate_);
        targetLeft = headingSpeed_ - turn * trackWidth / 2;
        targetRight = headingSpeed_ + turn * trackWidth / 2;

        // A motor at its limit costs speed, not the turn.
        const float limit = maxSpeed * boostGain;
        const float peak = std::max(std::abs(targetLeft), std::abs(targetRight));
        if (peak > limit) {
            targetLeft *= limit / peak;
            targetRight *= limit / peak;
        }
    }

    const float blend = 1 - std::exp(-dt / motorTimeConstant);
    left_.speed += (targetLeft - left_.speed) * blend;
    right_.speed += (targetRight - right_.speed) * blend;

    rotation.y += (right_.speed - left_.speed) / trackWidth * dt;
    translateZ(speed() * dt);

    for (auto* track : {&left_, &right_}) {
        const float travel = track->speed * dt;
        const float offset = std::fmod(track->treadMaps.front()->offset.y - travel / (treadPerimeter / treadPads), 1.f);
        for (const auto& map : track->treadMaps) {
            map->offset.y = offset;
        }
        for (const auto& wheel : track->wheels) {
            wheel->rotation.x += travel / trackRadius;
        }
    }

    updateTof(tofFront_);
    updateTof(tofRear_);
}

void Sphero::driveRaw(uint8_t leftMode, uint8_t leftSpeed, uint8_t rightMode, uint8_t rightSpeed) {

    const auto trackSpeed = [](uint8_t mode, uint8_t speed) {
        const float direction = mode == 0x1 ? 1.f : mode == 0x2 ? -1.f
                                                                : 0.f;
        return direction * static_cast<float>(speed) / 255 * maxSpeed;
    };

    rawLeft_ = trackSpeed(leftMode, leftSpeed);
    rawRight_ = trackSpeed(rightMode, rightSpeed);
    command_ = Command::Raw;
    commandAge_ = 0;
}

void Sphero::driveWithHeading(uint8_t speed, uint16_t heading, Flags flags) {

    // Headings run clockwise, yaw about +y runs anticlockwise.
    headingYaw_ = -static_cast<float>(heading) * math::DEG2RAD;
    headingTurnRate_ = flags.isFlagSet(Flags::FastTurnMode) ? 2 * turnRate : turnRate;

    headingSpeed_ = static_cast<float>(speed) / 255 * maxSpeed;
    if (flags.isFlagSet(Flags::Boost)) headingSpeed_ *= boostGain;
    if (flags.isFlagSet(Flags::DriveReverse)) headingSpeed_ = -headingSpeed_;

    command_ = Command::Heading;
    commandAge_ = 0;
}

void Sphero::setAllLeds(const Color& color) {

    ledMaterial_->color = color;
}

std::pair<float, float> Sphero::getTofMeasurements() const {

    return {tofFront_.reading, tofRear_.reading};
}

float Sphero::heading() const {

    const float degrees = std::fmod(-rotation.y * math::RAD2DEG, 360.f);

    return degrees < 0 ? degrees + 360 : degrees;
}

float Sphero::speed() const {

    return (left_.speed + right_.speed) / 2;
}

void Sphero::showTofBeams(bool show) {

    showTofBeams_ = show;
}

void Sphero::updateTof(Tof& tof) {

    tof.reading = std::numeric_limits<float>::infinity();

    if (parent) {
        Vector3 origin, direction;
        tof.mount->getWorldPosition(origin);
        tof.mount->getWorldDirection(direction);
        raycaster_.set(origin, direction);

        // Everything in the world but the robot itself.
        std::vector<Object3D*> others;
        for (auto* child : parent->children) {
            if (child != this) others.emplace_back(child);
        }

        const auto intersects = raycaster_.intersectObjects(others, true);
        if (!intersects.empty()) tof.reading = intersects.front().distance;
    }

    const bool hit = std::isfinite(tof.reading);
    tof.beam->visible = tof.dot->visible = hit && showTofBeams_;
    if (hit) {
        tof.beam->scale.z = tof.reading;
        tof.dot->position.z = tof.reading;
    }
}
