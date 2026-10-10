// The table's physics: sand as a height in each cell, water as a depth on top.
//
// Water moves by the "virtual pipes" scheme (O'Brien and Hodgins 1995; Mei,
// Decaudin and Hu 2007). Every cell is a column of water joined to its four
// neighbours by pipes. The difference in surface level between two columns
// speeds up the flow in the pipe between them, and a column never gives away
// more water than it holds. It is not the full shallow-water equations (there
// is no momentum carried along with the flow), but it fills, spills, pools and
// sloshes, it cannot lose or make water, and it stays stable whatever is done
// to the sand under it.
//
// Sand does two things. It slumps: a slope steeper than its angle of repose
// slides until it is not. And it is carried: fast water picks sand up and slow water drops it, so a
// trickle over a dam cuts a notch, the notch takes more water, and the dam goes.

#ifndef THREEPP_SANDTABLE_HPP
#define THREEPP_SANDTABLE_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sandtable {

    using namespace threepp;

    constexpr int columns = 240;
    constexpr int rows = 160;
    constexpr float cellSize = 0.005f;// [m]: the sand is 1.2 m by 0.8 m
    constexpr float wallHeight = 0.15f;// how high the box holds sand and water [m]

    // Less than the real thing, which slows the water to the pace of a
    // landscape seen from far off: a table-top lake sloshes several times a second.
    constexpr float gravity = 5.f;// [m/s^2]

    constexpr int substeps = 4;

    inline int cellIndex(int i, int j) {
        return i + j * columns;
    }

    // The table's own frame: its middle at the origin, y up from the floor of the box.
    inline float xOf(int i) {
        return (static_cast<float>(i) - (columns - 1) / 2.f) * cellSize;
    }

    inline float zOf(int j) {
        return (static_cast<float>(j) - (rows - 1) / 2.f) * cellSize;
    }

    constexpr float halfWidth = (columns - 1) / 2.f * cellSize;
    constexpr float halfDepth = (rows - 1) / 2.f * cellSize;

    // where the crater lake is, to begin with
    constexpr float lakeX = -0.32f, lakeZ = -0.03f, lakeReach = 0.125f;

    // The tap pours into the crater lake.
    constexpr float tapX = lakeX, tapZ = lakeZ;
    constexpr float tapFlow = 0.00006f;// 60 ml a second [m^3/s]

    // The drain is an overflow in the wall at the sea's end: a weir this wide,
    // which lets out whatever stands above its sill.
    constexpr float drainZ = 0.f, drainHalfWidth = 0.03f;
    // The sill is under the sea's level by the head the weir needs to pass
    // what the tap gives: with both open the sea stays about where it starts.
    constexpr float drainSill = 0.011f;// above the floor of the box [m]

    class Simulation {

    public:
        std::vector<float> ground, water;       // [m]
        std::vector<float> velocityX, velocityZ;// of the water [m/s]
        std::vector<float> wetness, foam;       // 0..1, for the look of it
        std::vector<float> sediment;            // sand in the water, as the height it would settle to [m]

        bool tap = false;
        bool drain = false;
        bool rain = false;
        bool erosion = true;

        Simulation()
            : ground(count), water(count), velocityX(count), velocityZ(count), wetness(count), foam(count), sediment(count),
              left_(count), right_(count), up_(count), down_(count), scratch_(count) {

            reset();
        }

        // A mountain that holds a crater lake at one end, a plain with a dry
        // river bed falling away from it, hills to either side, and the sea at
        // the other end.
        void reset() {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const float x = xOf(i), z = zOf(j);
                    const auto hill = [&](float cx, float cz, float stretch, float inner, float outer) {
                        return 1 - math::smoothstep(inner, outer, std::hypot((x - cx) / stretch, z - cz));
                    };

                    float h = math::lerp(0.046f, 0.024f, math::smoothstep(-0.35f, 0.3f, x));
                    h -= 0.02f * math::smoothstep(0.3f, 0.5f, x);// down to the sea

                    h += 0.08f * hill(lakeX, lakeZ, 1.f, 0.1f, 0.3f);    // the mountain
                    h -= 0.045f * hill(lakeX, lakeZ, 1.f, 0.f, lakeReach);// and its crater
                    h += 0.05f * hill(0.1f, -0.33f, 1.7f, 0.f, 0.16f);
                    h += 0.032f * hill(0.2f, 0.33f, 1.4f, 0.f, 0.15f);

                    const float bed = 0.045f * std::sin(8.f * x + 1.f);
                    h -= 0.009f * std::exp(-(z - bed) * (z - bed) / (0.03f * 0.03f)) * math::smoothstep(-0.1f, 0.f, x);

                    // A notch in the crater's rim and a gully from it down to the
                    // river bed: where the lake spills when it is full. Without
                    // one it spills all round the rim, as a sheet that never
                    // gathers into a stream.
                    const float gully = math::lerp(lakeZ, 0.025f, math::smoothstep(lakeX + 0.07f, -0.05f, x));
                    h -= 0.007f * std::exp(-(z - gully) * (z - gully) / (0.016f * 0.016f)) * math::smoothstep(lakeX + 0.06f, lakeX + 0.1f, x) * (1 - math::smoothstep(-0.1f, -0.02f, x));

                    h += 0.005f * (fbm(x * 9.f, z * 9.f) - 0.5f) + 0.009f * (fbm(x * 3.f + 7.f, z * 3.f) - 0.5f);

                    ground[cellIndex(i, j)] = std::clamp(h, 0.006f, wallHeight - 0.02f);
                }
            }

            // the lake: up to the lowest point of the crater's rim, less a little
            float rim = wallHeight;
            for (int k = 0; k < 90; ++k) {
                const float a = math::TWO_PI * static_cast<float>(k) / 90;
                rim = std::min(rim, groundAt(lakeX + lakeReach * std::cos(a), lakeZ + lakeReach * std::sin(a)));
            }
            const float lake = rim - 0.004f;
            constexpr float sea = 0.019f;
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    const bool inside = std::hypot(xOf(i) - lakeX, zOf(j) - lakeZ) < lakeReach;
                    water[c] = inside ? std::max(0.f, lake - ground[c]) : xOf(i) > 0.25f ? std::max(0.f, sea - ground[c])
                                                                                        : 0.f;
                    wetness[c] = water[c] > 0 ? 1.f : 0.f;
                    velocityX[c] = velocityZ[c] = foam[c] = sediment[c] = 0;
                    left_[c] = right_[c] = up_[c] = down_[c] = 0;
                }
            }
        }

        // One frame.
        void step(float dt) {
            const float h = dt / substeps;
            for (int s = 0; s < substeps; ++s) {
                if (rain) {
                    for (auto& depth : water) depth += 0.00025f * h;
                }
                if (tap) addVolume(tapX, tapZ, 0.012f, tapFlow * h);
                if (drain) overflow(h);
                flow(h);
            }

            slump();
            if (erosion) erode(dt);

            const float fade = std::exp(-1.5f * dt);
            for (int c = 0; c < count; ++c) {
                const float speed = std::hypot(velocityX[c], velocityZ[c]);

                // wet under water and in the rain, and half a minute in drying
                wetness[c] = rain || water[c] > 0.0003f ? 1.f : std::max(0.f, wetness[c] - dt / 30);

                // white water: where it runs fast and shallow
                const float churn = water[c] > 0.0008f ? std::max(0.f, speed - 0.16f) * (0.02f / (water[c] + 0.02f)) : 0.f;
                foam[c] = std::min(1.f, foam[c] * fade + 7.f * churn * dt);
            }
        }

        // Adds a volume of water [m^3], spread under a brush.
        void addVolume(float x, float z, float radius, float volume) {
            float weights = 0;
            brush(x, z, radius, [&](int, float weight) { weights += weight; });
            if (weights <= 0) return;

            const float depth = volume / (weights * cellSize * cellSize);
            brush(x, z, radius, [&](int c, float weight) {
                water[c] += depth * weight;
                foam[c] = std::max(foam[c], 0.8f * weight);// it lands white
            });
        }

        // Adds sand under a brush (or takes it, when amount is negative). amount
        // is the height added at the middle of the brush [m].
        void sculpt(float x, float z, float radius, float amount) {
            brush(x, z, radius, [&](int c, float weight) {
                ground[c] = std::clamp(ground[c] + amount * weight, 0.f, wallHeight - 0.01f);
            });
        }

        // Adds water under a brush (or soaks it up).
        void pour(float x, float z, float radius, float amount) {
            brush(x, z, radius, [&](int c, float weight) {
                water[c] = std::clamp(water[c] + amount * weight, 0.f, wallHeight);
            });
        }

        [[nodiscard]] float groundAt(float x, float z) const {
            return sample(ground, x, z);
        }

        [[nodiscard]] float waterAt(float x, float z) const {
            return sample(water, x, z);
        }

        [[nodiscard]] Vector2 velocityAt(float x, float z) const {
            return {sample(velocityX, x, z), sample(velocityZ, x, z)};
        }

        // Which way is uphill on the sand, and how steeply.
        [[nodiscard]] Vector2 slopeAt(float x, float z) const {
            return {(groundAt(x + cellSize, z) - groundAt(x - cellSize, z)) / (2 * cellSize),
                    (groundAt(x, z + cellSize) - groundAt(x, z - cellSize)) / (2 * cellSize)};
        }

        // Where a ray, given in the table's frame, first meets the sand or the water.
        [[nodiscard]] bool pick(const Ray& ray, Vector3& hit) const {
            const auto above = [&](float t) {
                const Vector3 p = Vector3(ray.origin).addScaledVector(ray.direction, t);
                return p.y - (groundAt(p.x, p.z) + waterAt(p.x, p.z));
            };
            const auto inside = [&](float t) {
                const Vector3 p = Vector3(ray.origin).addScaledVector(ray.direction, t);
                return std::abs(p.x) <= halfWidth && std::abs(p.z) <= halfDepth;
            };

            if (ray.direction.y >= 0) return false;

            // from where the ray comes down through the top of the box
            float t = std::max(0.f, (wallHeight - ray.origin.y) / ray.direction.y);
            const float end = -ray.origin.y / ray.direction.y;
            const float stride = cellSize * 0.5f;
            for (; t <= end + stride; t += stride) {
                if (!inside(t) || above(t) > 0) continue;

                float before = t - stride, after = t;
                for (int k = 0; k < 8; ++k) {
                    const float middle = 0.5f * (before + after);
                    (above(middle) > 0 ? before : after) = middle;
                }
                hit.copy(ray.origin).addScaledVector(ray.direction, after);

                return true;
            }

            return false;
        }

        // Water on the table [litres].
        [[nodiscard]] float volume() const {
            double sum = 0;
            for (const float depth : water) sum += depth;

            return static_cast<float>(sum * cellSize * cellSize * 1000);
        }

    private:
        static constexpr int count = columns * rows;

        // what each cell sends to its neighbour on that side [m^3/s]: left is -x, up is -z
        std::vector<float> left_, right_, up_, down_;
        std::vector<float> scratch_;

        static float hash(int x, int y) {
            unsigned int h = static_cast<unsigned int>(x) * 374761393u + static_cast<unsigned int>(y) * 668265263u;
            h = (h ^ (h >> 13)) * 1274126177u;
            return static_cast<float>((h ^ (h >> 16)) & 0xffff) / 65535.f;
        }

        static float noise(float x, float y) {
            const float fx = std::floor(x), fy = std::floor(y);
            const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
            const float u = (x - fx) * (x - fx) * (3 - 2 * (x - fx)), v = (y - fy) * (y - fy) * (3 - 2 * (y - fy));
            return math::lerp(math::lerp(hash(ix, iy), hash(ix + 1, iy), u), math::lerp(hash(ix, iy + 1), hash(ix + 1, iy + 1), u), v);
        }

        static float fbm(float x, float y) {
            float sum = 0, weight = 0.5f;
            for (int octave = 0; octave < 4; ++octave) {
                sum += weight * noise(x, y);
                x *= 2.03f;
                y *= 2.03f;
                weight *= 0.5f;
            }
            return sum / 0.9375f;
        }

        [[nodiscard]] float sample(const std::vector<float>& field, float x, float z) const {
            const float fi = std::clamp(x / cellSize + (columns - 1) / 2.f, 0.f, columns - 1.001f);
            const float fj = std::clamp(z / cellSize + (rows - 1) / 2.f, 0.f, rows - 1.001f);
            const int i = static_cast<int>(fi), j = static_cast<int>(fj);
            const float u = fi - static_cast<float>(i), v = fj - static_cast<float>(j);
            const int c = cellIndex(i, j);
            return math::lerp(math::lerp(field[c], field[c + 1], u), math::lerp(field[c + columns], field[c + columns + 1], u), v);
        }

        // Calls apply(cell, weight) for the cells under a round brush, the
        // weight falling smoothly from 1 in the middle to 0 at the edge.
        template<class Apply>
        void brush(float x, float z, float radius, Apply&& apply) {
            const int reach = static_cast<int>(radius / cellSize) + 1;
            const int ci = static_cast<int>(std::lround(x / cellSize + (columns - 1) / 2.f));
            const int cj = static_cast<int>(std::lround(z / cellSize + (rows - 1) / 2.f));
            for (int j = std::max(0, cj - reach); j <= std::min(rows - 1, cj + reach); ++j) {
                for (int i = std::max(0, ci - reach); i <= std::min(columns - 1, ci + reach); ++i) {
                    const float distance = std::hypot(xOf(i) - x, zOf(j) - z) / radius;
                    if (distance >= 1) continue;
                    const float falloff = 1 - distance * distance;
                    apply(cellIndex(i, j), falloff * falloff);
                }
            }
        }

        // Over the sill and out of the box. Sand piled in front of it holds
        // the water back like anywhere else.
        void overflow(float dt) {
            const auto rowAt = [](float z) { return std::clamp(static_cast<int>(std::lround(z / cellSize + (rows - 1) / 2.f)), 0, rows - 1); };
            const float weir = 0.4f * std::sqrt(2 * gravity);// through a metre of width, per head^1.5

            for (int j = rowAt(drainZ - drainHalfWidth); j <= rowAt(drainZ + drainHalfWidth); ++j) {
                const int c = cellIndex(columns - 1, j);
                const float head = ground[c] + water[c] - drainSill;
                if (head <= 0) continue;

                water[c] -= std::min({water[c], head, weir * head * std::sqrt(head) * dt / cellSize});
            }
        }

        void flow(float dt) {
            constexpr float area = cellSize * cellSize;
            const float push = dt * gravity * cellSize;// what a metre of head adds to a pipe's flow [m^3/s]

            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    const float depth = water[c];
                    const float level = ground[c] + depth;

                    // a thin sheet of water drags on the sand, deep water hardly at all
                    const float keep = 1 / (1 + dt * (0.4f + 0.02f / (depth + 0.001f)));

                    const auto towards = [&](float flowNow, int neighbour) {
                        return std::max(0.f, flowNow * keep + push * (level - ground[neighbour] - water[neighbour]));
                    };
                    float l = i > 0 ? towards(left_[c], c - 1) : 0.f;
                    float r = i < columns - 1 ? towards(right_[c], c + 1) : 0.f;
                    float u = j > 0 ? towards(up_[c], c - columns) : 0.f;
                    float d = j < rows - 1 ? towards(down_[c], c + columns) : 0.f;

                    // never more than is there
                    const float leaving = (l + r + u + d) * dt;
                    if (leaving > depth * area) {
                        const float scale = leaving > 0 ? depth * area / leaving : 0.f;
                        l *= scale;
                        r *= scale;
                        u *= scale;
                        d *= scale;
                    }

                    left_[c] = l;
                    right_[c] = r;
                    up_[c] = u;
                    down_[c] = d;
                }
            }

            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);

                    const float fromLeft = i > 0 ? right_[c - 1] : 0.f;
                    const float fromRight = i < columns - 1 ? left_[c + 1] : 0.f;
                    const float fromUp = j > 0 ? down_[c - columns] : 0.f;
                    const float fromDown = j < rows - 1 ? up_[c + columns] : 0.f;

                    const float before = water[c];
                    float depth = before + dt * (fromLeft + fromRight + fromUp + fromDown - left_[c] - right_[c] - up_[c] - down_[c]) / area;
                    // the last film soaks into the sand
                    if (depth < 0.0002f) depth -= 0.00008f * dt;
                    depth = std::max(0.f, depth);
                    water[c] = depth;

                    const float mean = 0.5f * (before + depth);
                    if (mean > 0.0003f) {
                        const float vx = (fromLeft - left_[c] + right_[c] - fromRight) / (2 * cellSize * mean);
                        const float vz = (fromUp - up_[c] + down_[c] - fromDown) / (2 * cellSize * mean);
                        const float speed = std::hypot(vx, vz);
                        const float limit = speed > 1.5f ? 1.5f / speed : 1.f;
                        velocityX[c] = vx * limit;
                        velocityZ[c] = vz * limit;
                    } else {
                        velocityX[c] = velocityZ[c] = 0;
                    }
                }
            }
        }

        // Sand steeper than its angle of repose slides to the cell below.
        void slump() {
            constexpr float repose = 0.67f;// tan of 34 degrees
            constexpr float diagonal = 1.41421356f;

            // Between each cell and the eight around it (four, counting each
            // pair once): with the four along the grid alone, a bank that runs
            // across the grid comes out as a flight of steps.
            std::fill(scratch_.begin(), scratch_.end(), 0.f);
            const auto settle = [&](int a, int b, float apart) {
                const float steepest = repose * cellSize * apart;
                const float difference = ground[a] - ground[b];
                const float excess = std::abs(difference) - steepest;
                if (excess <= 0) return;

                const float moved = std::copysign(0.12f * excess, difference);
                scratch_[a] -= moved;
                scratch_[b] += moved;
            };
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    if (i < columns - 1) settle(c, c + 1, 1);
                    if (j < rows - 1) settle(c, c + columns, 1);
                    if (i < columns - 1 && j < rows - 1) {
                        settle(c, c + columns + 1, diagonal);
                        settle(c + 1, c + columns, diagonal);
                    }
                }
            }
            for (int c = 0; c < count; ++c) ground[c] = std::max(0.f, ground[c] + scratch_[c]);
        }

        // Water takes up sand where it runs fast and drops it where it slows.
        void erode(float dt) {
            constexpr float carrying = 0.006f;// how much a metre per second can carry, as a height [m]
            constexpr float pickUp = 0.6f, settle = 1.2f;// [1/s]

            for (int j = 1; j < rows - 1; ++j) {
                for (int i = 1; i < columns - 1; ++i) {
                    const int c = cellIndex(i, j);
                    if (water[c] < 0.0005f) {
                        // left high and dry: what was in the water stays where it is
                        ground[c] += sediment[c];
                        sediment[c] = 0;
                        continue;
                    }

                    const float sx = (ground[c + 1] - ground[c - 1]) / (2 * cellSize), sz = (ground[c + columns] - ground[c - columns]) / (2 * cellSize);
                    const float tilt = std::hypot(sx, sz);
                    const float steepness = tilt / std::sqrt(1 + tilt * tilt);
                    // What sets the sand moving is fast water on a slope. Below a
                    // threshold it carries nothing, so that a sheet of water
                    // spreading over flat ground leaves it flat (it used to
                    // ripple it into rows of little dams), and neither does a
                    // film of run-off: rain on a hillside cut it into terraces.
                    const float body = math::smoothstep(0.0005f, 0.002f, water[c]);
                    const float capacity = carrying * std::max(0.f, std::hypot(velocityX[c], velocityZ[c]) * steepness - 0.008f) * body;

                    if (capacity > sediment[c]) {
                        const float taken = std::min({pickUp * (capacity - sediment[c]) * dt, ground[c], 0.0004f});
                        ground[c] -= taken;
                        sediment[c] += taken;
                    } else {
                        const float dropped = settle * (sediment[c] - capacity) * dt;
                        ground[c] += dropped;
                        sediment[c] -= dropped;
                    }
                }
            }

            // and the water carries it along
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    scratch_[c] = sample(sediment, xOf(i) - velocityX[c] * dt, zOf(j) - velocityZ[c] * dt);
                }
            }
            sediment.swap(scratch_);
        }
    };

    // The balls that float on the water and roll on the sand, to show which
    // way things are going.
    class Floats {

    public:
        static constexpr float radius = 0.009f;

        struct Ball {
            Vector3 position;
            float vx = 0, vz = 0;
        };

        std::vector<Ball> balls;

        explicit Floats(int count, const Simulation& simulation) {
            balls.resize(count);
            scatter(simulation);
        }

        // Onto the lake.
        void scatter(const Simulation& simulation) {
            for (size_t k = 0; k < balls.size(); ++k) {
                // a sunflower's spiral, which spreads them evenly
                const float r = 0.75f * lakeReach * std::sqrt((static_cast<float>(k) + 0.5f) / static_cast<float>(balls.size()));
                const float a = 2.39996f * static_cast<float>(k);
                auto& ball = balls[k];
                ball.position.x = lakeX + r * std::cos(a);
                ball.position.z = lakeZ + r * std::sin(a);
                ball.position.y = rest(simulation, ball);
                ball.vx = ball.vz = 0;
            }
        }

        void update(float dt, const Simulation& simulation) {
            for (auto& ball : balls) {
                auto& p = ball.position;
                const float depth = simulation.waterAt(p.x, p.z);

                if (depth > 0.5f * radius) {
                    // afloat: taken along by the water
                    const Vector2 current = simulation.velocityAt(p.x, p.z);
                    const float follow = std::min(1.f, 5.f * dt);
                    ball.vx += (current.x - ball.vx) * follow;
                    ball.vz += (current.y - ball.vz) * follow;
                } else {
                    // aground: rolls downhill, and the sand soon stops it
                    const Vector2 slope = simulation.slopeAt(p.x, p.z);
                    ball.vx -= 3.5f * slope.x * dt;
                    ball.vz -= 3.5f * slope.y * dt;
                    const float keep = std::exp(-3.f * dt);
                    ball.vx *= keep;
                    ball.vz *= keep;
                }

                p.x += ball.vx * dt;
                p.z += ball.vz * dt;

                // the walls of the box
                const float edgeX = halfWidth - radius, edgeZ = halfDepth - radius;
                if (std::abs(p.x) > edgeX) {
                    p.x = std::copysign(edgeX, p.x);
                    ball.vx *= -0.3f;
                }
                if (std::abs(p.z) > edgeZ) {
                    p.z = std::copysign(edgeZ, p.z);
                    ball.vz *= -0.3f;
                }
            }

            // they do not pass through each other
            for (size_t a = 0; a < balls.size(); ++a) {
                for (size_t b = a + 1; b < balls.size(); ++b) {
                    auto& pa = balls[a].position;
                    auto& pb = balls[b].position;
                    const float dx = pb.x - pa.x, dz = pb.z - pa.z;
                    const float distance = std::hypot(dx, dz);
                    if (distance >= 2 * radius || distance < 1e-6f) continue;

                    const float half = 0.5f * (2 * radius - distance) / distance;
                    pa.x -= dx * half;
                    pa.z -= dz * half;
                    pb.x += dx * half;
                    pb.z += dz * half;
                }
            }

            for (auto& ball : balls) {
                const float target = rest(simulation, ball);
                ball.position.y += (target - ball.position.y) * std::min(1.f, 18.f * dt);
            }
        }

    private:
        // The height at which a ball sits: on the sand, or a third sunk in the water.
        static float rest(const Simulation& simulation, const Ball& ball) {
            const float sand = simulation.groundAt(ball.position.x, ball.position.z);
            const float depth = simulation.waterAt(ball.position.x, ball.position.z);
            return sand + std::max(radius, depth + 0.35f * radius);
        }
    };

}// namespace sandtable

#endif//THREEPP_SANDTABLE_HPP
