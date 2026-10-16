// FJORD YARD — what makes the ground round the hytte look lived in.
//
// A cabin set down in an unbroken meadow reads as a model on a lawn. The
// things that say somebody uses the place, each built here from the cabin's
// own materials and a few primitives, merged so a prop is one or two draws:
//
//   woodpile      spruce rounds stacked against the gable, under the rake,
//                 dark bark and pale sawn ends, held by a stake at each side
//   choppingBlock a fat round with an axe left in it
//   skigard       the slanted split-rail fence of rural Norway: pairs of
//                 upright poles a stride apart, rails laid aslant between them
//   fireRing      a ring of stones on a dark ash bed, two logs to sit on
//   pennant       the red "vimpel" a Norwegian cabin flies from its pole, a
//                 tapered streamer that the wind waves (update() per frame)

#ifndef THREEPP_FJORD_YARD_HPP
#define THREEPP_FJORD_YARD_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/terrain/RockGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CircleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace fjord {

    // A set of pieces baked into one geometry (all pieces carry position,
    // normal and uv, so they merge).
    class PieceBag {
    public:
        void add(const std::shared_ptr<threepp::BufferGeometry>& g, const threepp::Matrix4& m) {
            auto c = g->clone();
            c->applyMatrix4(m);
            parts_.push_back(c);
        }
        [[nodiscard]] std::shared_ptr<threepp::BufferGeometry> build() const {
            if (parts_.empty()) return nullptr;
            auto g = threepp::mergeBufferGeometries(parts_);
            if (g) {
                g->computeBoundingBox();
                g->computeBoundingSphere();
            }
            return g;
        }
        [[nodiscard]] bool empty() const { return parts_.empty(); }

    private:
        std::vector<std::shared_ptr<threepp::BufferGeometry>> parts_;
    };

    inline threepp::Matrix4 trs(const threepp::Vector3& p, const threepp::Quaternion& q,
                                const threepp::Vector3& s = {1, 1, 1}) {
        threepp::Matrix4 m;
        m.compose(p, q, s);
        return m;
    }

    inline threepp::Quaternion euler(float x, float y, float z) {
        threepp::Quaternion qx, qy, qz;
        qx.setFromAxisAngle({1, 0, 0}, x);
        qy.setFromAxisAngle({0, 1, 0}, y);
        qz.setFromAxisAngle({0, 0, 1}, z);
        return qy.multiply(qx).multiply(qz);
    }

    // Spruce only. A share of birch rounds was tried (2026-10-16) and taken out
    // again: the pile is loosely stacked, and the pale birch sides showing
    // through the gaps between the rounds read as a rendering defect, not as
    // firewood.
    struct BarkAndEnds {
        std::shared_ptr<threepp::BufferGeometry> bark;// sides: the bark material
        std::shared_ptr<threepp::BufferGeometry> ends;// sawn faces: the log-end material
    };

    // Local frame: the pile's back against a wall in the plane z = 0, the
    // rounds lying along +z (their ends face out), bottom at y = 0, centred on x.
    inline BarkAndEnds makeWoodpile(float width, float height, float logLength, unsigned int seed) {
        using namespace threepp;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const auto side = CylinderGeometry::create(1.f, 1.f, 1.f, 9, 1, true);
        const auto cap = CircleGeometry::create(1.f, 9);
        PieceBag bark, ends;
        float y = 0.f;
        int row = 0;
        while (y < height) {
            float x = -width * 0.5f + (row % 2 ? 0.06f : 0.f);
            float rowTop = y;
            while (x < width * 0.5f) {
                const float r = 0.065f + 0.05f * u01(rng);
                const float len = logLength * (0.92f + 0.12f * u01(rng));
                const float zc = 0.5f * len + 0.04f * (u01(rng) - 0.5f);
                const Vector3 c(x + r, y + r, zc);
                // Roll about the round's own axis (the cylinder's +y), lay it
                // down along +z, then a hair of yaw: no two rounds sit alike.
                Quaternion roll, lay, yaw;
                roll.setFromAxisAngle({0, 1, 0}, u01(rng) * 6.283f);
                lay.setFromAxisAngle({1, 0, 0}, 1.5708f + 0.05f * (u01(rng) - 0.5f));
                yaw.setFromAxisAngle({0, 1, 0}, 0.04f * (u01(rng) - 0.5f));
                Quaternion q = yaw;
                q.multiply(lay).multiply(roll);
                bark.add(side, trs(c, q, {r, len, r}));
                // The two sawn faces, each a disc at an end of the round.
                for (int e = 0; e < 2; ++e) {
                    const float sz = e == 0 ? 1.f : -1.f;
                    Quaternion qe = q;
                    Quaternion flip;
                    flip.setFromAxisAngle({1, 0, 0}, sz > 0 ? -1.5708f : 1.5708f);
                    qe.multiply(flip);
                    Vector3 off(0.f, sz * len * 0.5f, 0.f);
                    off.applyQuaternion(q);
                    ends.add(cap, trs(Vector3(c).add(off), qe, {r * 0.97f, r * 0.97f, 1.f}));
                }
                x += 2.f * r + 0.01f;
                rowTop = std::max(rowTop, y + 2.f * r);
            }
            y = y + (rowTop - y) * 0.86f;// rounds nest into the gaps below
            ++row;
        }
        // A stake at each end holds the stack.
        const auto stake = BoxGeometry::create(0.07f, 1.f, 0.07f);
        for (float sx : {-1.f, 1.f})
            bark.add(stake, trs({sx * (width * 0.5f + 0.06f), height * 0.55f, logLength * 0.8f},
                                euler(0.f, 0.f, sx * 0.06f), {1.f, height * 1.1f, 1.f}));
        return {bark.build(), ends.build()};
    }

    // A round set on end with an axe left in it. Local origin at its foot.
    struct ChoppingBlock {
        std::shared_ptr<threepp::BufferGeometry> bark, top, handle, head;
    };
    inline ChoppingBlock makeChoppingBlock() {
        using namespace threepp;
        ChoppingBlock b;
        PieceBag bark, top, handle, head;
        bark.add(CylinderGeometry::create(1.f, 1.f, 1.f, 12, 1, true), trs({0, 0.22f, 0}, Quaternion(), {0.26f, 0.44f, 0.26f}));
        top.add(CircleGeometry::create(1.f, 12), trs({0, 0.44f, 0}, euler(-1.5708f, 0, 0), {0.25f, 0.25f, 1}));
        // Axe: the bit sunk in the top, the haft leaning out over the rim.
        handle.add(CylinderGeometry::create(1.f, 1.f, 1.f, 6), trs({0.18f, 0.66f, 0.05f}, euler(0.f, 0.3f, -0.95f), {0.018f, 0.72f, 0.018f}));
        head.add(BoxGeometry::create(1.f, 1.f, 1.f), trs({-0.04f, 0.47f, 0.f}, euler(0.f, 0.3f, -0.95f + 1.5708f), {0.05f, 0.17f, 0.025f}));
        b.bark = bark.build();
        b.top = top.build();
        b.handle = handle.build();
        b.head = head.build();
        return b;
    }

    // Skigard along a polyline of ground points (world space). Pairs of
    // uprights a stride apart, a hand apart across the line; between them the
    // split rails lie steeply aslant, four to a stride, each resting up on the
    // next pairs. Nothing in a hand-split fence is regular: every rail has its
    // own length, lean, thickness and a little twist, every upright its own
    // height and lean. `open(x, z)` true = a gap (a gate): no upright there, and
    // no rail that would cross it.
    inline std::shared_ptr<threepp::BufferGeometry> makeSkigard(
            const std::vector<threepp::Vector3>& line, const std::function<float(float, float)>& groundY, unsigned int seed,
            const std::function<bool(float, float)>& open = {}) {
        using namespace threepp;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        PieceBag bag;
        const auto pole = CylinderGeometry::create(0.8f, 1.f, 1.f, 6);
        const auto rail = BoxGeometry::create(1.f, 1.f, 1.f);
        constexpr float stride = 1.2f;
        const auto isOpen = [&](float x, float z) { return open && open(x, z); };
        for (size_t i = 1; i < line.size(); ++i) {
            const Vector3 a = line[i - 1], b = line[i];
            Vector3 dir(b.x - a.x, 0.f, b.z - a.z);
            const float segLen = dir.length();
            if (segLen < 0.1f) continue;
            dir.multiplyScalar(1.f / segLen);
            const Vector3 across(-dir.z, 0.f, dir.x);
            const float yaw = std::atan2(dir.x, dir.z);
            for (float s = 0.f; s < segLen - 0.2f; s += stride) {
                const float x = a.x + dir.x * s, z = a.z + dir.z * s;
                if (isOpen(x, z)) continue;
                const float gy = groundY(x, z);
                for (float k : {-1.f, 1.f}) {
                    const float h = 1.40f + 0.18f * u01(rng);
                    const float r = 0.035f + 0.012f * u01(rng);
                    bag.add(pole, trs({x + across.x * k * 0.075f, gy + h * 0.5f - 0.2f, z + across.z * k * 0.075f},
                                      euler(0.07f * (u01(rng) - 0.5f), 0.f, 0.07f * (u01(rng) - 0.5f)), {r, h, r}));
                }
                for (int rr = 0; rr < 4; ++rr) {
                    const float tilt = 0.52f + 0.10f * (u01(rng) - 0.5f);// ~30°: a skigard is chest-high
                    const float len = 2.3f + 0.5f * u01(rng);
                    const float along = s + 0.30f * static_cast<float>(rr) + 0.08f * u01(rng);
                    const float run = len * std::cos(tilt);
                    if (along + run > segLen + 0.3f) continue;
                    const float mx = a.x + dir.x * (along + run * 0.5f), mz = a.z + dir.z * (along + run * 0.5f);
                    bool blocked = false;
                    for (float q = 0.f; q <= run + 0.01f && !blocked; q += 0.4f)
                        blocked = isOpen(a.x + dir.x * (along + q), a.z + dir.z * (along + q));
                    if (blocked) continue;
                    const float gy0 = groundY(a.x + dir.x * along, a.z + dir.z * along);
                    const float side = (u01(rng) - 0.5f) * 0.05f;
                    bag.add(rail, trs({mx + across.x * side, gy0 + 0.04f + len * 0.5f * std::sin(tilt), mz + across.z * side},
                                      euler(-tilt, yaw + 0.03f * (u01(rng) - 0.5f), 0.5f * (u01(rng) - 0.5f)),
                                      {0.055f + 0.035f * u01(rng), 0.028f + 0.014f * u01(rng), len}));
                }
            }
        }
        return bag.build();
    }

    // Grey field stone with lichen: a tiling sRGB albedo for the yard's rocks.
    // `base` is the linear stone colour; lichen patches lift it a little and
    // pull it toward grey-green, so the stone keeps the tone it was given.
    inline std::shared_ptr<threepp::DataTexture> makeLichenRockTexture(int size, unsigned int seed,
                                                                      const threepp::Color& base) {
        using namespace threepp;
        std::vector<unsigned char> px(static_cast<size_t>(size) * size * 4);
        auto hash = [seed](int x, int y) {
            uint32_t n = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u + seed * 2246822519u;
            n = (n ^ (n >> 13)) * 1274126177u;
            return static_cast<float>((n ^ (n >> 16)) & 0xffffffu) / static_cast<float>(0xffffff);
        };
        auto vn = [&](float x, float y, int period) {
            const int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
            const float fx = x - std::floor(x), fy = y - std::floor(y);
            const float sx = fx * fx * (3.f - 2.f * fx), sy = fy * fy * (3.f - 2.f * fy);
            auto h = [&](int a, int b) { return hash(((a % period) + period) % period, ((b % period) + period) % period); };
            const float a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
            return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
        };
        const auto toSrgb = [](float c) {
            c = std::clamp(c, 0.f, 1.f);
            return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
        };
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / size, v = static_cast<float>(y) / size;
                const float n = 0.5f * vn(u * 6.f, v * 6.f, 6) + 0.3f * vn(u * 17.f, v * 17.f, 17) + 0.2f * vn(u * 41.f, v * 41.f, 41);
                const float lichen = std::clamp((vn(u * 9.f + 3.f, v * 9.f, 9) - 0.62f) * 6.f, 0.f, 1.f) *
                                     (0.6f + 0.4f * vn(u * 53.f, v * 53.f, 53));
                float r = base.r * (0.70f + 0.6f * n), g = base.g * (0.70f + 0.6f * n), b = base.b * (0.70f + 0.6f * n);
                r += (base.r * 1.55f - r) * 0.6f * lichen;// paler grey-green crust
                g += (base.g * 1.75f - g) * 0.6f * lichen;
                b += (base.b * 1.30f - b) * 0.6f * lichen;
                const size_t o = (static_cast<size_t>(y) * size + x) * 4;
                px[o] = static_cast<unsigned char>(toSrgb(r) * 255.f + 0.5f);
                px[o + 1] = static_cast<unsigned char>(toSrgb(g) * 255.f + 0.5f);
                px[o + 2] = static_cast<unsigned char>(toSrgb(b) * 255.f + 0.5f);
                px[o + 3] = 255;
            }
        auto t = DataTexture::create(ImageData{std::move(px)}, static_cast<unsigned int>(size), static_cast<unsigned int>(size));
        t->colorSpace = ColorSpace::sRGB;
        t->wrapS = t->wrapT = TextureWrapping::Repeat;
        t->magFilter = Filter::Linear;
        t->minFilter = Filter::LinearMipmapLinear;
        t->generateMipmaps = true;
        t->needsUpdate();
        return t;
    }

    // A ring of stones on an ash bed with two logs to sit on. Local origin at
    // the centre of the hearth, on the ground.
    struct FireRing {
        std::shared_ptr<threepp::BufferGeometry> stones, ash, seats;
    };
    inline FireRing makeFireRing(unsigned int seed) {
        using namespace threepp;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        FireRing f;
        PieceBag stones, ash, seats;
        const auto rock = terrain::makeRockGeometry(seed);
        constexpr int n = 12;
        for (int i = 0; i < n; ++i) {
            const float a = 6.2832f * (static_cast<float>(i) + 0.3f * u01(rng)) / n;
            const float s = 0.16f + 0.06f * u01(rng);
            stones.add(rock, trs({0.78f * std::cos(a), s * 0.35f, 0.78f * std::sin(a)}, euler(0.f, u01(rng) * 6.28f, 0.f),
                                 {s * 1.2f, s * 0.8f, s}));
        }
        ash.add(CircleGeometry::create(1.f, 16), trs({0, 0.02f, 0}, euler(-1.5708f, 0, 0), {0.72f, 0.72f, 1}));
        const auto log = CylinderGeometry::create(1.f, 1.f, 1.f, 9);
        seats.add(log, trs({0.f, 0.17f, 1.85f}, euler(0.f, 0.f, 1.5708f), {0.17f, 1.7f, 0.17f}));
        seats.add(log, trs({1.75f, 0.17f, -0.55f}, euler(0.f, 1.25f, 1.5708f), {0.16f, 1.5f, 0.16f}));
        f.stones = stones.build();
        f.ash = ash.build();
        f.seats = seats.build();
        return f;
    }

    // The red vimpel: a tapered streamer of `segs` panels, `length` metres,
    // hung from the masthead. update() lays it out down the wind with a
    // travelling wave whose amplitude grows toward the free end.
    class Pennant {
    public:
        explicit Pennant(float length = 4.2f, float width = 0.30f, int segs = 28)
            : length_(length), width_(width), segs_(segs) {
            using namespace threepp;
            geometry = BufferGeometry::create();
            std::vector<float> pos(static_cast<size_t>(segs + 1) * 2 * 3, 0.f), nrm(pos.size(), 0.f), uv;
            std::vector<unsigned int> idx;
            for (int i = 0; i <= segs; ++i) {
                for (int s = 0; s < 2; ++s) uv.insert(uv.end(), {static_cast<float>(i) / segs, static_cast<float>(s)});
                if (i < segs) {
                    const auto a = static_cast<unsigned int>(i * 2);
                    idx.insert(idx.end(), {a, a + 2, a + 1, a + 1, a + 2, a + 3});
                }
            }
            geometry->setIndex(idx);
            geometry->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geometry->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geometry->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            update(0.f, {1.f, 0.f}, 4.f);
        }

        // windXZ: unit heading the wind blows toward; speed in m/s at the masthead.
        void update(float t, const threepp::Vector2& windXZ, float speed) {
            auto* pa = geometry->getAttribute<float>("position");
            auto* na = geometry->getAttribute<float>("normal");
            auto& p = pa->array();
            auto& n = na->array();
            const float droop = 0.55f / (0.6f + 0.25f * speed);// a slack streamer hangs
            const float wx = windXZ.x, wz = windXZ.y;
            const float sx = -wz, sz = wx;// sideways
            for (int i = 0; i <= segs_; ++i) {
                const float f = static_cast<float>(i) / segs_;
                const float d = f * length_;
                const float amp = 0.10f + 0.32f * f * f;
                const float wave = amp * std::sin(6.0f * f * 3.1416f - t * (3.0f + 0.6f * speed));
                const float lift = 0.12f * f * std::sin(4.f * f * 3.1416f - t * 2.3f);
                const float y = -droop * d * f + lift;
                const float half = 0.5f * width_ * (1.f - 0.92f * f);
                for (int s = 0; s < 2; ++s) {
                    const float k = s == 0 ? half : -half;
                    const size_t o = (static_cast<size_t>(i) * 2 + s) * 3;
                    p[o] = wx * d + sx * wave;
                    p[o + 1] = y + k;
                    p[o + 2] = wz * d + sz * wave;
                    n[o] = sx;
                    n[o + 1] = 0.f;
                    n[o + 2] = sz;
                }
            }
            pa->needsUpdate();
            na->needsUpdate();
            geometry->computeBoundingSphere();
        }

        std::shared_ptr<threepp::BufferGeometry> geometry;

    private:
        float length_, width_;
        int segs_;
    };

}// namespace fjord

#endif//THREEPP_FJORD_YARD_HPP
