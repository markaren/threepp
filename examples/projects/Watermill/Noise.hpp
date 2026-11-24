// Noise for the watermill's textures, ground and shapes: hashes, gradient (Perlin) noise in 2D and 3D, optionally
// periodic so a texture tiles, fractal sums, and cellular (Worley) noise that also reports which cell won.

#ifndef THREEPP_EXAMPLE_WATERMILL_NOISE_HPP
#define THREEPP_EXAMPLE_WATERMILL_NOISE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mill {

    inline uint32_t hash32(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    inline uint32_t hash2i(int x, int y, uint32_t seed) {
        return hash32(static_cast<uint32_t>(x) * 0x8da6b343U ^ hash32(static_cast<uint32_t>(y) * 0xd8163841U ^ seed * 0xcb1ab31fU));
    }

    inline uint32_t hash3i(int x, int y, int z, uint32_t seed) {
        return hash32(static_cast<uint32_t>(z) * 0x9e3779b1U ^ hash2i(x, y, seed));
    }

    // [0, 1)
    inline float unit(uint32_t h) { return static_cast<float>(h >> 8) * (1.f / 16777216.f); }

    inline float hash2f(int x, int y, uint32_t seed) { return unit(hash2i(x, y, seed)); }
    inline float hash3f(int x, int y, int z, uint32_t seed) { return unit(hash3i(x, y, z, seed)); }

    inline float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
    inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
    inline float clamp01(float v) { return std::clamp(v, 0.f, 1.f); }
    inline float smooth(float e0, float e1, float x) {
        const float t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3.f - 2.f * t);
    }

    inline int wrapi(int i, int period) {
        if (period <= 0) return i;
        i %= period;
        return i < 0 ? i + period : i;
    }

    inline float grad2(uint32_t h, float x, float y) {
        // 16 directions round the circle
        const float a = static_cast<float>(h & 15u) * (6.2831853f / 16.f);
        return std::cos(a) * x + std::sin(a) * y;
    }

    // gradient noise, about [-0.7, 0.7]; px/py > 0 make it periodic in lattice units
    inline float perlin2(float x, float y, uint32_t seed = 0, int px = 0, int py = 0) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        const int x0 = wrapi(ix, px), x1 = wrapi(ix + 1, px), y0 = wrapi(iy, py), y1 = wrapi(iy + 1, py);
        const float n00 = grad2(hash2i(x0, y0, seed), tx, ty);
        const float n10 = grad2(hash2i(x1, y0, seed), tx - 1, ty);
        const float n01 = grad2(hash2i(x0, y1, seed), tx, ty - 1);
        const float n11 = grad2(hash2i(x1, y1, seed), tx - 1, ty - 1);
        const float u = fade(tx), v = fade(ty);
        return lerpf(lerpf(n00, n10, u), lerpf(n01, n11, u), v);
    }

    inline float grad3(uint32_t h, float x, float y, float z) {
        switch (h % 12u) {
            case 0: return x + y;
            case 1: return -x + y;
            case 2: return x - y;
            case 3: return -x - y;
            case 4: return x + z;
            case 5: return -x + z;
            case 6: return x - z;
            case 7: return -x - z;
            case 8: return y + z;
            case 9: return -y + z;
            case 10: return y - z;
            default: return -y - z;
        }
    }

    inline float perlin3(float x, float y, float z, uint32_t seed = 0) {
        const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
        const float tx = x - fx, ty = y - fy, tz = z - fz;
        const float u = fade(tx), v = fade(ty), w = fade(tz);
        float n[8];
        for (int k = 0; k < 8; ++k) {
            const int dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
            n[k] = grad3(hash3i(ix + dx, iy + dy, iz + dz, seed), tx - static_cast<float>(dx), ty - static_cast<float>(dy), tz - static_cast<float>(dz));
        }
        const float x00 = lerpf(n[0], n[1], u), x10 = lerpf(n[2], n[3], u), x01 = lerpf(n[4], n[5], u), x11 = lerpf(n[6], n[7], u);
        return 0.5f * lerpf(lerpf(x00, x10, v), lerpf(x01, x11, v), w);
    }

    // fractal sums, roughly [-1, 1]
    inline float fbm2(float x, float y, int octaves, uint32_t seed = 0, int px = 0, int py = 0, float gain = 0.5f) {
        float sum = 0, amp = 1, norm = 0;
        octaves = std::clamp(octaves, 1, 12);
        for (int o = 0; o < octaves; ++o) {
            sum += amp * perlin2(x, y, seed + 101u * static_cast<uint32_t>(o), px, py);
            norm += amp;
            amp *= gain;
            x *= 2;
            y *= 2;
            px *= 2;
            py *= 2;
        }
        return 1.4f * sum / norm;
    }

    inline float fbm3(float x, float y, float z, int octaves, uint32_t seed = 0, float gain = 0.5f) {
        float sum = 0, amp = 1, norm = 0;
        octaves = std::clamp(octaves, 1, 12);
        for (int o = 0; o < octaves; ++o) {
            sum += amp * perlin3(x, y, z, seed + 131u * static_cast<uint32_t>(o));
            norm += amp;
            amp *= gain;
            x *= 2.03f;
            y *= 2.03f;
            z *= 2.03f;
        }
        return 2.2f * sum / norm;
    }

    // ridged: sharp crests where the noise crosses zero, [0, 1]
    inline float ridged2(float x, float y, int octaves, uint32_t seed = 0, int px = 0, int py = 0) {
        float sum = 0, amp = 1, norm = 0;
        octaves = std::clamp(octaves, 1, 12);
        for (int o = 0; o < octaves; ++o) {
            const float n = 1.f - std::abs(1.4f * perlin2(x, y, seed + 71u * static_cast<uint32_t>(o), px, py));
            sum += amp * n * n;
            norm += amp;
            amp *= 0.5f;
            x *= 2;
            y *= 2;
            px *= 2;
            py *= 2;
        }
        return sum / norm;
    }

    // cellular noise: the distances to the nearest and second nearest feature point (one per cell, jittered by
    // `jitter`), and the nearest one's cell hash; periodic when px/py > 0
    struct Cell {
        float f1 = 1e9f, f2 = 1e9f;
        uint32_t id = 0;
        float cx = 0, cy = 0;// the nearest feature point
    };

    inline Cell worley2(float x, float y, uint32_t seed = 0, int px = 0, int py = 0, float jitter = 1.f) {
        const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
        Cell c;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int cx = ix + dx, cy = iy + dy;
                const uint32_t h = hash2i(wrapi(cx, px), wrapi(cy, py), seed);
                const float fx = static_cast<float>(cx) + 0.5f + jitter * (unit(h) - 0.5f);
                const float fy = static_cast<float>(cy) + 0.5f + jitter * (unit(hash32(h)) - 0.5f);
                const float d = std::hypot(fx - x, fy - y);
                if (d < c.f1) {
                    c.f2 = c.f1;
                    c.f1 = d;
                    c.id = h;
                    c.cx = fx;
                    c.cy = fy;
                } else if (d < c.f2) {
                    c.f2 = d;
                }
            }
        }
        return c;
    }

    // a small deterministic generator for placement
    struct Rng {
        uint32_t s;
        explicit Rng(uint32_t seed): s(hash32(seed + 0x51ed27u)) {}
        uint32_t next() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            return s;
        }
        float operator()() { return unit(next()); }
        float range(float a, float b) { return a + (b - a) * (*this)(); }
        float normal() {
            const float u1 = std::max((*this)(), 1e-7f), u2 = (*this)();
            return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
        }
    };

}// namespace mill

#endif
