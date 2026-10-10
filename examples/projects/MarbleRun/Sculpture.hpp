// The sculpture: a screw lift in the middle, two tracks winding down around
// it, and a turned bowl at the foot that swirls the marbles back to the lift.
//
// A marble's centre follows a line that is worked out once (Path). On a track
// it speeds up with the slope and is held back by drag, in the bowl it does the
// same on a much gentler slope, and a marble never passes the one ahead of it.
// The screw carries two columns of marbles, one on either side, and each track
// ends at the foot of the other column: a marble takes the two tracks in turn.

#ifndef THREEPP_MARBLERUN_SCULPTURE_HPP
#define THREEPP_MARBLERUN_SCULPTURE_HPP

#include "threepp/threepp.hpp"

#include "threepp/geometries/LatheGeometry.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace marblerun {

    using namespace threepp;

    // All of it in metres: the sculpture stands 0.9 m tall on its plinth.
    constexpr float marbleRadius = 0.0125f;
    constexpr float railRadius = 0.002f;
    constexpr float gauge = 0.016f;// between the centres of the two rails

    constexpr float plinthRadius = 0.44f;
    constexpr float plinthTop = 0.06f;

    constexpr float bowlFloor = 0.075f;// its surface at the inner edge
    constexpr float bowlDepth = 0.07f;
    constexpr float bowlInner = 0.03f;
    constexpr float bowlOuter = 0.29f;
    constexpr float rimRadius = 0.27f;// where a track sets its marbles down

    constexpr float columnRadius = 0.045f;// a marble on the lift, from the axis
    constexpr float liftTop = 0.76f;
    constexpr float liftPitch = 0.05f;
    constexpr float liftRate = 1.2f;// turns per second

    constexpr float postRadius = 0.37f;
    constexpr int postCount = 5;
    constexpr float rodRadius = 0.058f;// the lift's guide rods, from the axis

    constexpr float gravityAlong = 7.f;// a rolling ball gets 5/7 of g down a slope [m/s^2]
    constexpr float drag = 2.f;        // [1/s]
    constexpr float minSpeed = 0.1f;   // [m/s]

    // How far a marble's centre rides above the centres of the rails it sits on.
    inline float rideHeight() {
        const float reach = marbleRadius + railRadius;
        return std::sqrt(reach * reach - gauge * gauge / 4);
    }

    inline float bowlHeight(float r) {
        const float x = std::clamp((r - bowlInner) / (bowlOuter - bowlInner), 0.f, 1.f);
        return bowlFloor + bowlDepth * std::pow(x, 0.6f);
    }

    inline float bowlSlope(float r) {
        const float x = std::clamp((r - bowlInner) / (bowlOuter - bowlInner), 0.01f, 1.f);
        return bowlDepth * 0.6f / (bowlOuter - bowlInner) * std::pow(x, -0.4f);
    }

    // The height of a marble's centre when it rests on the bowl at radius r.
    inline float bowlMarbleHeight(float r) {
        const float slope = bowlSlope(r);
        return bowlHeight(r) + marbleRadius * std::sqrt(1 + slope * slope);
    }

    inline float columnAngle(int column) {
        return column == 0 ? 0.f : math::PI;
    }

    // The line a marble's centre follows from the top of one lift column to
    // the foot of the other, in even steps.
    class Path {

    public:
        static constexpr float step = 0.003f;

        std::vector<Vector3> points;
        float railEnd = 0;// how far along the rails stop and the bowl takes over

        [[nodiscard]] float length() const {
            return step * static_cast<float>(points.size() - 1);
        }

        [[nodiscard]] Vector3 at(float s) const {
            const float f = std::clamp(s / step, 0.f, static_cast<float>(points.size() - 1));
            const auto i = std::min(static_cast<size_t>(f), points.size() - 2);
            return Vector3().lerpVectors(points[i], points[i + 1], f - static_cast<float>(i));
        }

        [[nodiscard]] Vector3 tangent(float s) const {
            const auto i = std::min(static_cast<size_t>(std::max(s, 0.f) / step), points.size() - 2);
            return Vector3().subVectors(points[i + 1], points[i]).normalize();
        }

        // which = 0: the wide track, three turns. which = 1: the tight one
        // inside it, six turns. Both wind the same way, so that the two
        // spirals they end in lie side by side in the bowl and never cross.
        static Path create(int which) {
            const float start = columnAngle(which);
            const float helixAngle = math::TWO_PI * (which == 0 ? 3.f : 6.f);
            const float bowlAngle = math::TWO_PI * 1.5f;
            const float launchAngle = math::TWO_PI * (which == 0 ? 0.8f : 0.5f);
            const float landAngle = math::TWO_PI * (which == 0 ? 0.25f : 0.6f);

            const auto radiusAt = [&](float angle) {
                if (angle >= helixAngle) return math::lerp(rimRadius, columnRadius, (angle - helixAngle) / bowlAngle);

                // drawn in at the waist, like an hourglass
                const float f = angle / helixAngle;
                const float waist = (2 * f - 1) * (2 * f - 1);
                const float body = which == 0 ? 0.25f + 0.08f * waist : 0.115f + 0.02f * waist;

                const float r = math::lerp(columnRadius, body, math::smoothstep(0.f, launchAngle, angle));
                return math::lerp(r, rimRadius, math::smoothstep(helixAngle - landAngle, helixAngle, angle));
            };

            const int helixSteps = static_cast<int>(helixAngle / 0.004f), bowlSteps = static_cast<int>(bowlAngle / 0.004f);

            std::vector<Vector3> raw;
            for (int i = 0; i <= helixSteps + bowlSteps; ++i) {
                const float angle = i <= helixSteps
                                            ? helixAngle * static_cast<float>(i) / static_cast<float>(helixSteps)
                                            : helixAngle + bowlAngle * static_cast<float>(i - helixSteps) / static_cast<float>(bowlSteps);
                const float r = radiusAt(angle);
                raw.emplace_back(r * std::cos(start + angle), 0.f, r * std::sin(start + angle));
            }

            // Heights. On the rails: a gentle start, steepest in the middle,
            // and a chute's slope where the marble is set down in the bowl.
            std::vector<float> ground{0};
            for (int i = 1; i <= helixSteps; ++i) ground.push_back(ground.back() + raw[i].distanceTo(raw[i - 1]));
            const float drop = liftTop - bowlMarbleHeight(rimRadius);
            for (int i = 0; i <= helixSteps; ++i) {
                constexpr float k0 = 0.5f, k1 = 0.9f;
                const float u = ground[i] / ground.back(), u2 = u * u, u3 = u2 * u;
                raw[i].y = liftTop - drop * (k0 * (u3 - 2 * u2 + u) + (3 * u2 - 2 * u3) + k1 * (u3 - u2));
            }
            for (int i = helixSteps + 1; i <= helixSteps + bowlSteps; ++i) {
                raw[i].y = bowlMarbleHeight(std::hypot(raw[i].x, raw[i].z));
            }

            // Into even steps.
            std::vector<float> along{0};
            for (size_t i = 1; i < raw.size(); ++i) along.push_back(along.back() + raw[i].distanceTo(raw[i - 1]));

            Path path;
            path.railEnd = along[helixSteps];
            const int count = static_cast<int>(along.back() / step);
            size_t k = 0;
            for (int i = 0; i <= count; ++i) {
                const float s = step * static_cast<float>(i);
                while (k + 2 < raw.size() && along[k + 1] < s) ++k;
                path.points.push_back(Vector3().lerpVectors(raw[k], raw[k + 1], (s - along[k]) / (along[k + 1] - along[k])));
            }

            return path;
        }
    };

    // A line through points that are already evenly spaced.
    class Polyline: public Curve3 {

    public:
        explicit Polyline(std::vector<Vector3> points): points_(std::move(points)) {
            arcLengthDivisions = static_cast<int>(points_.size());
        }

        void getPoint(float t, Vector3& target) const override {
            const float f = std::clamp(t, 0.f, 1.f) * static_cast<float>(points_.size() - 1);
            const auto i = std::min(static_cast<size_t>(f), points_.size() - 2);
            target.lerpVectors(points_[i], points_[i + 1], f - static_cast<float>(i));
        }

        void getPointAt(float u, Vector3& target) const override {
            getPoint(u, target);
        }

    private:
        std::vector<Vector3> points_;
    };

    class Helix: public Curve3 {

    public:
        Helix(float radius, float pitch, float turns): radius_(radius), pitch_(pitch), turns_(turns) {}

        void getPoint(float t, Vector3& target) const override {
            const float angle = t * turns_ * math::TWO_PI;
            target.set(radius_ * std::cos(angle), pitch_ * t * turns_, radius_ * std::sin(angle));
        }

    private:
        float radius_, pitch_, turns_;
    };

    class Sculpture: public Object3D {

    public:
        struct Marble {
            Color color;
            Mesh* mesh = nullptr;          // a glowing marble is a mesh of its own, with a light in it
            InstancedMesh* batch = nullptr;// the others are drawn by the dozen
            size_t slot = 0;

            bool onLift = false;
            int line = 0;       // the path it is on, or the lift column
            float s = 0, v = 0; // along the path [m], [m/s]
            float boardedAt = 0;// the screw's turn count when it stepped on

            Vector3 position, heading;
        };

        // A hoop around a track that lights up as a marble goes through it.
        struct Gate {
            int path = 0;
            float s = 0;
            Vector3 position;
            float glow = 0;
            Color color;
            std::shared_ptr<MeshStandardMaterial> material;
        };

        // Called as a marble goes through a gate: the gate's number and the marble's speed.
        std::function<void(int, float)> onGate;

        explicit Sculpture(int marbleCount = 96) {
            paths_[0] = Path::create(0);
            paths_[1] = Path::create(1);

            createMaterials();
            createBase();
            createLift();
            createCage();
            for (int p = 0; p < 2; ++p) createTrack(p);
            createMarbles(marbleCount);

            // Start with the sculpture already running.
            for (int i = 0; i < 60 * 90; ++i) step(1.f / 60);
            for (auto& gate : gates_) gate.glow = 0;
            laps_ = 0;
            writeTransforms();
        }

        void update(float dt) {
            step(dt);

            for (auto& gate : gates_) {
                gate.glow *= std::exp(-3.f * dt);
                gate.material->emissive = gate.color;
                gate.material->emissiveIntensity = 5.f * gate.glow;
            }

            writeTransforms();
        }

        [[nodiscard]] const std::vector<Marble>& marbles() const {
            return marbles_;
        }

        [[nodiscard]] int laps() const {
            return laps_;
        }

        // Where the hoops are, in the sculpture's own frame.
        [[nodiscard]] std::vector<Vector3> gatePositions() const {
            std::vector<Vector3> positions;
            for (const auto& gate : gates_) positions.push_back(gate.position);

            return positions;
        }

        // The marbles that glow, as indices into marbles().
        [[nodiscard]] const std::vector<size_t>& glowing() const {
            return glowing_;
        }

    private:
        std::array<Path, 2> paths_;
        std::array<std::deque<size_t>, 2> onPath_;// the marbles on each path, the one furthest along first
        std::vector<Marble> marbles_;
        std::vector<size_t> glowing_;
        std::vector<Gate> gates_;

        std::shared_ptr<Group> screw_;
        float screwAngle_ = 0;
        std::array<float, 2> lastTurn_{0, 0};
        int laps_ = 0;

        std::shared_ptr<MeshStandardMaterial> steel_, brass_, dark_, chrome_;
        std::vector<std::shared_ptr<InstancedMesh>> batches_;

        struct Frame {
            Vector3 position, tangent, up, side;
        };

        static std::shared_ptr<MeshStandardMaterial> metal(const Color& color, float roughness) {
            auto material = MeshStandardMaterial::create();
            material->color = color;
            material->metalness = 1;
            material->roughness = roughness;

            return material;
        }

        template<class T>
        static std::shared_ptr<T> shadowed(std::shared_ptr<T> object) {
            object->castShadow = true;
            object->receiveShadow = true;

            return object;
        }

        // The matrix that lays a cylinder of radius 1 and height 1 between two points.
        static Matrix4 rod(const Vector3& from, const Vector3& to, float radius) {
            Vector3 along = Vector3().subVectors(to, from);
            const float length = along.length();
            Quaternion q;
            q.setFromUnitVectors(Vector3(0, 1, 0), along.divideScalar(length));

            return Matrix4().compose(Vector3().addVectors(from, to).multiplyScalar(0.5f), q, Vector3(radius, length, radius));
        }

        static Vector3 onCircle(float radius, float angle, float y) {
            return {radius * std::cos(angle), y, radius * std::sin(angle)};
        }

        // The screw's turn count as seen from a column: a whole number each
        // time its wire comes past the column's foot.
        [[nodiscard]] float turnsAt(int column) const {
            return (columnAngle(column) + screwAngle_) / math::TWO_PI;
        }

        void createMaterials() {
            steel_ = metal(0xb4bac2, 0.3f);
            brass_ = metal(0xc9953a, 0.32f);
            dark_ = metal(0x2a2e34, 0.45f);
            chrome_ = metal(0xf2f4f7, 0.06f);
        }

        void createBase() {
            auto lacquer = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                                .color(Color(0x0c0d10))
                                                                .roughness(0.55f)
                                                                .clearcoat(1.f)
                                                                .clearcoatRoughness(0.08f));
            auto plinth = shadowed(Mesh::create(CylinderGeometry::create(plinthRadius, plinthRadius + 0.015f, plinthTop, 96), lacquer));
            plinth->position.y = plinthTop / 2;
            add(plinth);

            auto trim = shadowed(Mesh::create(TorusGeometry::create(plinthRadius - 0.004f, 0.004f, 12, 128), brass_));
            trim->rotation.x = math::PI / 2;
            trim->position.y = plinthTop;
            add(trim);

            // The bowl, turned from one piece: a funnel that steepens towards the lift.
            std::vector<Vector2> profile;
            profile.emplace_back(bowlInner - 0.006f, bowlFloor - 0.012f);
            profile.emplace_back(bowlInner - 0.005f, bowlFloor + 0.006f);
            profile.emplace_back(bowlInner - 0.002f, bowlFloor + 0.006f);
            for (int i = 0; i <= 48; ++i) {
                const float x = static_cast<float>(i) / 48;
                const float r = bowlInner + (bowlOuter - bowlInner) * x * x;
                profile.emplace_back(r, bowlHeight(r));
            }
            const float lip = bowlHeight(bowlOuter);
            profile.emplace_back(bowlOuter + 0.006f, lip + 0.001f);
            profile.emplace_back(bowlOuter + 0.01f, lip - 0.004f);
            profile.emplace_back(bowlOuter + 0.008f, lip - 0.014f);
            profile.emplace_back(bowlOuter - 0.03f, lip - 0.045f);
            profile.emplace_back(bowlOuter - 0.1f, plinthTop + 0.012f);
            profile.emplace_back(bowlOuter - 0.16f, plinthTop);

            auto wood = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                             .map(createWoodTexture(profile))
                                                             .roughness(0.6f)
                                                             .clearcoat(0.25f)
                                                             .clearcoatRoughness(0.2f));
            wood->side = Side::Double;
            add(shadowed(Mesh::create(LatheGeometry::create(profile, 160), wood)));
        }

        // Walnut, as a solid block that the bowl is cut from: the growth rings
        // are worked out at each point of the bowl's surface, so that they
        // run across it the way they do on a turned piece.
        static std::shared_ptr<Texture> createWoodTexture(const std::vector<Vector2>& profile) {
            constexpr int width = 2048, height = 512;

            const auto hash = [](int x, int y, int z) {
                unsigned int h = static_cast<unsigned int>(x) * 374761393u + static_cast<unsigned int>(y) * 668265263u + static_cast<unsigned int>(z) * 2147483647u;
                h = (h ^ (h >> 13)) * 1274126177u;
                return static_cast<float>((h ^ (h >> 16)) & 0xffff) / 65535.f;
            };
            const auto noise = [&](const Vector3& p) {
                const float fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
                const int x = static_cast<int>(fx), y = static_cast<int>(fy), z = static_cast<int>(fz);
                const auto fade = [](float t) { return t * t * (3 - 2 * t); };
                const float u = fade(p.x - fx), v = fade(p.y - fy), w = fade(p.z - fz);
                const auto along = [&](int dy, int dz) { return math::lerp(hash(x, y + dy, z + dz), hash(x + 1, y + dy, z + dz), u); };
                return math::lerp(math::lerp(along(0, 0), along(1, 0), v), math::lerp(along(0, 1), along(1, 1), v), w);
            };

            // the heart of the tree, running under the bowl and a little askew
            const Vector3 pith(0.05f, -0.16f, 0.02f);
            const Vector3 grain = Vector3(1.f, 0.12f, 0.35f).normalize();
            const Color light(0x8a5631), dark(0x33190c);

            std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
            for (int j = 0; j < height; ++j) {
                // the row's place along the profile
                const float along = (static_cast<float>(j) + 0.5f) / height * static_cast<float>(profile.size() - 1);
                const auto k = std::min(static_cast<size_t>(along), profile.size() - 2);
                const Vector2 at = Vector2().lerpVectors(profile[k], profile[k + 1], along - static_cast<float>(k));

                for (int i = 0; i < width; ++i) {
                    const float phi = (static_cast<float>(i) + 0.5f) / width * math::TWO_PI;
                    const Vector3 p(at.x * std::sin(phi), at.y, at.x * std::cos(phi));

                    const Vector3 fromPith = Vector3().subVectors(p, pith);
                    const float alongGrain = fromPith.dot(grain);
                    const float distance = Vector3(fromPith).addScaledVector(grain, -alongGrain).length();

                    const float wobble = 0.012f * noise(Vector3(p).multiplyScalar(14.f)) + 0.003f * noise(Vector3(p).multiplyScalar(60.f));
                    const float ring = (distance + wobble) / 0.009f;
                    const float late = std::pow(ring - std::floor(ring), 3.f);// dark towards the end of each year

                    // pores, drawn out along the grain
                    const Vector3 stretched = Vector3(p).addScaledVector(grain, -0.93f * p.dot(grain)).multiplyScalar(900.f);
                    const float pores = noise(stretched);
                    const float figure = noise(Vector3(p).multiplyScalar(6.f));

                    Color c = light;
                    c.lerp(dark, std::clamp(0.25f + 0.6f * late + 0.3f * (figure - 0.5f), 0.f, 1.f));
                    const float shade = 0.85f + 0.3f * pores;

                    auto* pixel = &pixels[(static_cast<size_t>(j) * width + i) * 4];
                    pixel[0] = static_cast<unsigned char>(std::clamp(c.r * shade, 0.f, 1.f) * 255);
                    pixel[1] = static_cast<unsigned char>(std::clamp(c.g * shade, 0.f, 1.f) * 255);
                    pixel[2] = static_cast<unsigned char>(std::clamp(c.b * shade, 0.f, 1.f) * 255);
                    pixel[3] = 255;
                }
            }

            auto texture = DataTexture::create(ImageData{std::move(pixels)}, width, height);
            texture->magFilter = Filter::Linear;
            texture->minFilter = Filter::LinearMipmapLinear;
            texture->generateMipmaps = true;
            texture->anisotropy = 16;
            texture->wrapS = TextureWrapping::Repeat;
            texture->colorSpace = ColorSpace::sRGB;
            texture->needsUpdate();

            return texture;
        }

        void createLift() {
            // The wire sits this far under the centre of the marble it carries.
            constexpr float wireRadius = 0.0035f, wireHelix = 0.04f, wireDrop = 0.0135f;

            const float foot = bowlMarbleHeight(columnRadius);
            const float bottom = foot - wireDrop - liftPitch;
            const float turns = (liftTop - wireDrop - bottom) / liftPitch + 0.3f;

            screw_ = Group::create();
            screw_->position.y = bottom;
            add(screw_);

            auto wire = shadowed(Mesh::create(TubeGeometry::create(std::make_shared<Helix>(wireHelix, liftPitch, turns),
                                                                   static_cast<unsigned int>(turns * 48), wireRadius, 10),
                                              brass_));
            screw_->add(wire);

            constexpr float shaftRadius = 0.011f;
            const float shaftTop = liftTop + 0.1f - bottom;
            auto shaft = shadowed(Mesh::create(CylinderGeometry::create(shaftRadius, shaftRadius, shaftTop, 24), steel_));
            shaft->position.y = shaftTop / 2;
            screw_->add(shaft);

            // spokes, from the shaft out to the wire
            const int spokeCount = static_cast<int>(turns * 2);
            auto spokes = shadowed(InstancedMesh::create(CylinderGeometry::create(1, 1, 1, 8), brass_, spokeCount));
            for (int i = 0; i < spokeCount; ++i) {
                const float angle = math::PI * static_cast<float>(i), y = liftPitch * static_cast<float>(i) / 2;
                spokes->setMatrixAt(i, rod(onCircle(shaftRadius, angle, y), onCircle(wireHelix, angle, y), 0.0015f));
            }
            spokes->frustumCulled = false;
            screw_->add(spokes);
        }

        // What holds it all up: five posts that bend in over the top to a
        // crown, and the rods the marbles on the lift lean against.
        void createCage() {
            constexpr float arcStart = 0.7f, crownHeight = 0.89f, crownRadius = 0.07f;

            std::vector<Vector3> line;
            for (int i = 0; i <= 60; ++i) {
                line.emplace_back(postRadius, math::lerp(plinthTop, arcStart, static_cast<float>(i) / 60), 0.f);
            }
            for (int i = 1; i <= 60; ++i) {
                const float t = math::PI / 2 * static_cast<float>(i) / 60;
                line.emplace_back(crownRadius + (postRadius - crownRadius) * std::cos(t), arcStart + (crownHeight - arcStart) * std::sin(t), 0.f);
            }
            const auto postGeometry = TubeGeometry::create(std::make_shared<Polyline>(line), 120, 0.0045f, 10);
            const auto footGeometry = CylinderGeometry::create(0.011f, 0.014f, 0.012f, 24);
            for (int i = 0; i < postCount; ++i) {
                auto post = shadowed(Mesh::create(postGeometry, dark_));
                post->rotation.y = -postAngle(i);
                add(post);

                auto foot = shadowed(Mesh::create(footGeometry, brass_));
                foot->position.copy(onCircle(postRadius, postAngle(i), plinthTop + 0.006f));
                add(foot);
            }

            auto crown = shadowed(Mesh::create(TorusGeometry::create(crownRadius, 0.006f, 12, 64), brass_));
            crown->rotation.x = math::PI / 2;
            crown->position.y = crownHeight;
            add(crown);

            auto finial = shadowed(Mesh::create(SphereGeometry::create(0.02f, 32, 16), brass_));
            finial->position.y = liftTop + 0.11f;
            add(finial);

            // The guide rods, two to a column, and the rings that tie them.
            const float rodBottom = bowlMarbleHeight(columnRadius) + 0.035f, rodTop = liftTop - 0.022f;
            constexpr float ringRadius = 0.0625f;

            std::vector<Matrix4> rods;
            for (int k = 0; k < 4; ++k) {
                rods.push_back(rod(onCircle(rodRadius, rodAngle(k), rodBottom), onCircle(rodRadius, rodAngle(k), rodTop), 0.002f));
            }
            // and what the lift hangs from
            for (const float angle : {-0.6f, 1.9f, math::PI - 0.6f, math::PI + 1.9f}) {
                rods.push_back(rod(onCircle(ringRadius, angle, rodTop), onCircle(crownRadius, angle, crownHeight), 0.002f));
            }
            addRods(rods, steel_);

            const auto ringGeometry = TorusGeometry::create(ringRadius, 0.002f, 8, 64);
            for (int i = 0; i < 5; ++i) {
                auto ring = shadowed(Mesh::create(ringGeometry, steel_));
                ring->rotation.x = math::PI / 2;
                ring->position.y = math::lerp(rodBottom, rodTop, static_cast<float>(i) / 4);
                add(ring);
            }
        }

        static float postAngle(int i) {
            return math::TWO_PI * (static_cast<float>(i) + 0.35f) / postCount;
        }

        static float rodAngle(int k) {
            return columnAngle(k / 2) + (k % 2 == 0 ? -0.2f : 0.2f);
        }

        void addRods(const std::vector<Matrix4>& matrices, const std::shared_ptr<Material>& material) {
            if (matrices.empty()) return;

            auto rods = shadowed(InstancedMesh::create(CylinderGeometry::create(1, 1, 1, 10), material, matrices.size()));
            for (size_t i = 0; i < matrices.size(); ++i) rods->setMatrixAt(i, matrices[i]);
            rods->frustumCulled = false;
            add(rods);
        }

        // Frames along the railed part of a path, leaning into its bends.
        static std::vector<Frame> createFrames(const Path& path) {
            const auto count = static_cast<size_t>(path.railEnd / Path::step) + 1;
            const Vector3 worldUp(0, 1, 0);

            std::vector<Frame> frames(count);
            for (size_t i = 0; i < count; ++i) {
                auto& f = frames[i];
                f.position = path.points[i];
                f.tangent.subVectors(path.points[i + 1], path.points[i > 0 ? i - 1 : 0]).normalize();
                f.up.copy(worldUp).addScaledVector(f.tangent, -worldUp.dot(f.tangent)).normalize();
                f.side.crossVectors(f.tangent, f.up);
            }

            // the harder the path turns to one side, the more it leans that way
            constexpr size_t reach = 12;
            std::vector<float> bank(count);
            for (size_t i = 0; i < count; ++i) {
                const size_t a = i >= reach ? i - reach : 0, b = std::min(i + reach, count - 1);
                const float bend = Vector3().subVectors(frames[b].tangent, frames[a].tangent).dot(frames[i].side) / (Path::step * static_cast<float>(b - a));
                bank[i] = std::clamp(0.07f * bend, -0.55f, 0.55f);
            }
            constexpr size_t window = 24;
            for (size_t i = 0; i < count; ++i) {
                const size_t a = i >= window ? i - window : 0, b = std::min(i + window, count - 1);
                float angle = 0;
                for (size_t k = a; k <= b; ++k) angle += bank[k];
                angle /= static_cast<float>(b - a + 1);

                auto& f = frames[i];
                const Vector3 up = f.up, side = f.side;
                f.up.copy(up).multiplyScalar(std::cos(angle)).addScaledVector(side, std::sin(angle));
                f.side.copy(side).multiplyScalar(std::cos(angle)).addScaledVector(up, -std::sin(angle));
            }

            return frames;
        }

        void createTrack(int which) {
            const auto& path = paths_[which];
            const auto frames = createFrames(path);
            const float ride = rideHeight();

            // The ties and the spine under them stop short of the bowl, and
            // the two rails alone set the marble down on it.
            constexpr float spineDepth = gauge / 2 + 0.0025f, tieRadius = 0.0013f, spineRadius = 0.0025f;
            const size_t carried = frames.size() - static_cast<size_t>(0.12f / Path::step);

            std::vector<Vector3> left, right, spine;
            for (size_t i = 0; i < frames.size(); ++i) {
                const auto& f = frames[i];
                const Vector3 bed = Vector3(f.position).addScaledVector(f.up, -ride);
                left.push_back(Vector3(bed).addScaledVector(f.side, gauge / 2));
                right.push_back(Vector3(bed).addScaledVector(f.side, -gauge / 2));
                if (i < carried) spine.push_back(Vector3(bed).addScaledVector(f.up, -spineDepth));
            }
            const auto tube = [&](const std::vector<Vector3>& line, float radius) {
                add(shadowed(Mesh::create(TubeGeometry::create(std::make_shared<Polyline>(line), static_cast<unsigned int>(line.size() / 2), radius, 8), steel_)));
            };
            tube(left, railRadius);
            tube(right, railRadius);
            tube(spine, spineRadius);

            // A tie every three centimetres: half a ring, hung under the rails.
            constexpr size_t tieEvery = 10;
            const size_t tieCount = carried / tieEvery;
            auto ties = shadowed(InstancedMesh::create(TorusGeometry::create(gauge / 2, tieRadius, 6, 12, math::PI), steel_, tieCount));
            for (size_t i = 0; i < tieCount; ++i) {
                const auto& f = frames[i * tieEvery + tieEvery / 2];
                Matrix4 m;
                m.makeBasis(f.side, Vector3(f.up).negate(), f.tangent);
                m.setPosition(Vector3(f.position).addScaledVector(f.up, -ride));
                ties->setMatrixAt(i, m);
            }
            ties->frustumCulled = false;
            add(ties);

            // An arm wherever the track comes past a post or a guide rod.
            std::vector<Matrix4> arms;
            const auto reachFor = [&](float angle, float anchorRadius, float nearest, float furthest) {
                for (size_t i = 1; i < carried; ++i) {
                    const auto& a = frames[i - 1].position;
                    const auto& b = frames[i].position;
                    const float before = std::remainder(std::atan2(a.z, a.x) - angle, math::TWO_PI);
                    const float after = std::remainder(std::atan2(b.z, b.x) - angle, math::TWO_PI);
                    if (!(before < 0 && after >= 0 && after - before < 0.5f)) continue;

                    const float r = std::hypot(b.x, b.z);
                    if (r < nearest || r > furthest) continue;

                    const Vector3 hold = Vector3(b).addScaledVector(frames[i].up, -ride - spineDepth);
                    arms.push_back(rod(onCircle(anchorRadius, angle, hold.y), hold, 0.002f));
                }
            };
            for (int i = 0; i < postCount; ++i) reachFor(postAngle(i), postRadius, 0.2f, 1.f);
            for (int k = 0; k < 4; ++k) reachFor(rodAngle(k), rodRadius, 0.085f, 0.2f);
            addRods(arms, dark_);

            // The gates.
            const auto hoop = TorusGeometry::create(0.028f, 0.0022f, 10, 48);
            const float spacing = which == 0 ? 0.82f : 0.7f;
            for (float s = 0.75f; s < path.railEnd - 0.4f; s += spacing) {
                const auto& f = frames[static_cast<size_t>(s / Path::step)];

                Gate gate;
                gate.path = which;
                gate.s = s;
                gate.position = f.position;
                gate.material = metal(0xd9a441, 0.3f);
                gates_.push_back(gate);

                auto mesh = shadowed(Mesh::create(hoop, gate.material));
                Matrix4 m;
                m.makeBasis(f.side, f.up, f.tangent);
                mesh->quaternion.setFromRotationMatrix(m);
                mesh->position.copy(f.position);
                add(mesh);
            }
        }

        void createMarbles(int count) {
            static const std::array<Color, 6> glassColors{Color(0x7fd0ff), Color(0xff8fa3), Color(0x9dffb0), Color(0xffd07a), Color(0xcdb0ff), Color(0xf2faff)};
            static const std::array<Color, 6> glowColors{Color(0x2bd9ff), Color(0xff2fc4), Color(0xffa51f), Color(0x35ff6b), Color(0x9d7bff), Color(0xff5a36)};

            enum Kind { Steel,
                        Glass,
                        Glow };
            // one in sixteen glows: each of those is a light, and every surface pays for every light
            const auto kindOf = [](int i) { return i % 16 == 5 ? Glow : i % 3 == 1 ? Glass
                                                                                  : Steel; };
            size_t steelCount = 0, glassCount = 0;
            for (int i = 0; i < count; ++i) {
                steelCount += kindOf(i) == Steel;
                glassCount += kindOf(i) == Glass;
            }

            const auto sphere = SphereGeometry::create(marbleRadius, 40, 20);

            auto glass = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                              .roughness(0.04f)
                                                              .transmission(1.f)
                                                              .thickness(1.6f * marbleRadius)
                                                              .ior(1.5f));

            auto steelBatch = shadowed(InstancedMesh::create(sphere, chrome_, std::max<size_t>(steelCount, 1)));
            auto glassBatch = shadowed(InstancedMesh::create(sphere, glass, std::max<size_t>(glassCount, 1)));
            for (const auto& batch : {steelBatch, glassBatch}) {
                batch->frustumCulled = false;
                batches_.push_back(batch);
                add(batch);
            }

            size_t steelSlot = 0, glassSlot = 0;
            for (int i = 0; i < count; ++i) {
                Marble m;
                switch (kindOf(i)) {
                    case Steel:
                        m.color = Color(0xcfe0ff);
                        m.batch = steelBatch.get();
                        m.slot = steelSlot++;
                        break;
                    case Glass:
                        m.color = glassColors[glassSlot % glassColors.size()];
                        m.batch = glassBatch.get();
                        m.slot = glassSlot++;
                        glassBatch->setColorAt(m.slot, m.color);
                        break;
                    case Glow: {
                        m.color = glowColors[glowing_.size() % glowColors.size()];

                        auto material = MeshStandardMaterial::create();
                        material->color = Color(0x101010);
                        material->emissive = m.color;
                        material->emissiveIntensity = 3.5f;
                        material->roughness = 0.3f;

                        auto mesh = Mesh::create(sphere, material);
                        mesh->add(PointLight::create(m.color, 0.1f, 0.3f, 2.f));
                        add(mesh);
                        m.mesh = mesh.get();
                        glowing_.push_back(marbles_.size());
                        break;
                    }
                }

                // queued up in the bowl, at the foot of the lift
                m.line = i % 2;
                m.s = paths_[m.line].length() - 2.04f * marbleRadius * static_cast<float>(onPath_[m.line].size());
                onPath_[m.line].push_back(marbles_.size());
                marbles_.push_back(m);
            }
        }

        void step(float dt) {
            screwAngle_ += liftRate * math::TWO_PI * dt;
            screw_->rotation.y = screwAngle_;

            // A turn of the wire comes past the foot of a column: the marble
            // waiting there steps on.
            for (int column = 0; column < 2; ++column) {
                const float turn = std::floor(turnsAt(column));
                if (turn == lastTurn_[column]) continue;
                lastTurn_[column] = turn;

                auto& queue = onPath_[1 - column];
                if (queue.empty()) continue;
                auto& m = marbles_[queue.front()];
                if (m.s < paths_[1 - column].length() - 1e-4f) continue;

                queue.pop_front();
                m.onLift = true;
                m.line = column;
                m.boardedAt = turn;
            }

            const float foot = bowlMarbleHeight(columnRadius);
            for (size_t i = 0; i < marbles_.size(); ++i) {
                auto& m = marbles_[i];
                if (!m.onLift) continue;

                const float angle = columnAngle(m.line);
                const float y = foot + liftPitch * (turnsAt(m.line) - m.boardedAt);
                if (y < liftTop) {
                    m.position = onCircle(columnRadius, angle, y);
                    m.heading.set(-std::sin(angle), 0, std::cos(angle));
                    continue;
                }

                // off the top and onto the track
                m.onLift = false;
                m.s = 0;
                m.v = 0.12f;
                onPath_[m.line].push_back(i);
                ++laps_;
            }

            for (int p = 0; p < 2; ++p) {
                const auto& path = paths_[p];
                const Marble* ahead = nullptr;
                for (const auto index : onPath_[p]) {
                    auto& m = marbles_[index];

                    const Vector3 tangent = path.tangent(m.s);
                    m.v = std::clamp(m.v + (gravityAlong * -tangent.y - drag * m.v) * dt, minSpeed, 1.5f);

                    float s = m.s + m.v * dt;
                    const float limit = ahead ? ahead->s - 2.04f * marbleRadius : path.length();
                    if (s > limit) {
                        s = limit;
                        m.v = ahead ? std::min(m.v, ahead->v) : 0.f;
                    }

                    for (size_t g = 0; g < gates_.size(); ++g) {
                        auto& gate = gates_[g];
                        if (gate.path != p || m.s >= gate.s || s < gate.s) continue;
                        gate.glow = 1;
                        gate.color = m.color;
                        if (onGate) onGate(static_cast<int>(g), m.v);
                    }

                    m.s = s;
                    m.position = path.at(s);
                    m.heading = tangent;
                    ahead = &m;
                }
            }
        }

        void writeTransforms() {
            Matrix4 matrix;
            for (const auto& m : marbles_) {
                if (m.mesh) {
                    m.mesh->position.copy(m.position);
                } else {
                    matrix.setPosition(m.position);
                    m.batch->setMatrixAt(m.slot, matrix);
                }
            }
            for (const auto& batch : batches_) batch->instanceMatrix()->needsUpdate();
        }
    };

}// namespace marblerun

#endif//THREEPP_MARBLERUN_SCULPTURE_HPP
