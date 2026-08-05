
#include "threepp/splats/SplatData.hpp"

#include <algorithm>
#include <cstdint>

using namespace threepp;

namespace {

    // Fixed xorshift32. Not a good generator, but a *reproducible* one: the
    // state transition and the fixed-point conversion below are fully
    // specified, so a seed maps to the same cloud on every platform and every
    // standard library. See the note on SplatGenerator.
    struct Rng {

        explicit Rng(unsigned int seed)
            : state_(seed ? seed : 0x9e3779b9u) {}

        std::uint32_t next() {

            state_ ^= state_ << 13;
            state_ ^= state_ >> 17;
            state_ ^= state_ << 5;
            return state_;
        }

        // [0, 1) with 24 bits of mantissa — exactly representable as float.
        float unit() {

            return static_cast<float>(next() >> 8) * (1.f / 16777216.f);
        }

        float range(float lo, float hi) {

            return lo + (hi - lo) * unit();
        }

    private:
        std::uint32_t state_;
    };

}// namespace


void SplatData::resize(size_t n, int degree) {

    shDegree = std::clamp(degree, 0, splats::MAX_SH_DEGREE);

    means.assign(n, Vector3{});
    scales.assign(n, Vector3{});
    rotations.assign(n, Quaternion{});
    opacities.assign(n, 0.f);
    sh.assign(n * static_cast<size_t>(coeffCount()) * 3, 0.f);
}

void SplatData::normalizeRotations() {

    for (auto& q : rotations) {

        // A zero quaternion has no rotation to recover; identity is the only
        // answer that keeps the covariance finite.
        if (q.lengthSq() <= 0.f) {

            q.set(0.f, 0.f, 0.f, 1.f);

        } else {

            q.normalize();
        }
    }
}

void SplatData::setDcColor(size_t splat, const Vector3& rgb) {

    const auto dc = splats::dcFromColor(rgb);
    auto* c = shAt(splat);
    c[0] = dc.x;
    c[1] = dc.y;
    c[2] = dc.z;
}

void SplatData::computeCovariance(size_t i, float* out) const {

    const auto& s = scales[i];
    const auto& q = rotations[i];

    // R from the (already normalised) quaternion, then M = R * S with
    // S = diag(scale). Sigma = M * M^T, which is symmetric by construction —
    // only the six distinct entries are written.
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x * x2, xy = x * y2, xz = x * z2;
    const float yy = y * y2, yz = y * z2, zz = z * z2;
    const float wx = w * x2, wy = w * y2, wz = w * z2;

    // Column-scaled rotation: column j of R multiplied by scale[j].
    const float m00 = (1.f - (yy + zz)) * s.x;
    const float m10 = (xy + wz) * s.x;
    const float m20 = (xz - wy) * s.x;

    const float m01 = (xy - wz) * s.y;
    const float m11 = (1.f - (xx + zz)) * s.y;
    const float m21 = (yz + wx) * s.y;

    const float m02 = (xz + wy) * s.z;
    const float m12 = (yz - wx) * s.z;
    const float m22 = (1.f - (xx + yy)) * s.z;

    out[0] = m00 * m00 + m01 * m01 + m02 * m02;// xx
    out[1] = m00 * m10 + m01 * m11 + m02 * m12;// xy
    out[2] = m00 * m20 + m01 * m21 + m02 * m22;// xz
    out[3] = m10 * m10 + m11 * m11 + m12 * m12;// yy
    out[4] = m10 * m20 + m11 * m21 + m12 * m22;// yz
    out[5] = m20 * m20 + m21 * m21 + m22 * m22;// zz
}

Box3 SplatData::computeBounds(float sigma) const {

    Box3 box;
    box.makeEmpty();

    for (size_t i = 0; i < count(); ++i) {

        const float r = sigma * std::max({scales[i].x, scales[i].y, scales[i].z});
        box.expandByPoint(Vector3{means[i].x - r, means[i].y - r, means[i].z - r});
        box.expandByPoint(Vector3{means[i].x + r, means[i].y + r, means[i].z + r});
    }

    return box;
}

namespace {

    // Exact order statistic, not an interpolated quantile: nth_element puts
    // THE element of that rank in place, which is the same element on every
    // standard library. `v` is taken by value because it gets reordered.
    float percentile(std::vector<float> v, float q) {

        if (v.empty()) return 0.f;

        const auto rank = static_cast<size_t>(
                std::clamp(q, 0.f, 1.f) * static_cast<float>(v.size() - 1));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(rank), v.end());
        return v[rank];
    }

    float medianOf(std::vector<float> v) {

        if (v.empty()) return 0.f;
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
        return v[mid];
    }

    // Keeps the elements of `values` whose `keep` flag is set, in order, in
    // place. `stride` elements per splat, so one call handles the SH block
    // (coeffCount * 3 floats per splat) as well as a plain per-splat array.
    template<class T>
    void compact(std::vector<T>& values, const std::vector<bool>& keep, size_t stride = 1) {

        size_t out = 0;
        for (size_t i = 0; i < keep.size(); ++i) {

            if (!keep[i]) continue;
            if (out != i) {

                std::copy(values.begin() + static_cast<std::ptrdiff_t>(i * stride),
                          values.begin() + static_cast<std::ptrdiff_t>((i + 1) * stride),
                          values.begin() + static_cast<std::ptrdiff_t>(out * stride));
            }
            ++out;
        }
        values.resize(out * stride);
    }

    // Applies `perm` (new index -> old index) to `values` in place, `stride`
    // elements per splat, by rotating each cycle of the permutation. In place
    // rather than gathering into a fresh vector on purpose: the degree-3 SH
    // block of a five-million-splat scan is ~960 MB and a gather would need a
    // second copy of it live at the same time. `moved` is caller-owned scratch
    // (one bit per splat, reused across every array) and is reset here.
    template<class T>
    void permute(std::vector<T>& values, const std::vector<std::uint32_t>& perm,
                 std::vector<bool>& moved, size_t stride = 1) {

        const size_t n = perm.size();
        if (values.size() != n * stride) return;// see the note on reorderMorton

        std::fill(moved.begin(), moved.end(), false);
        std::vector<T> head(stride);

        const auto at = [stride](std::vector<T>& v, size_t i) {
            return v.begin() + static_cast<std::ptrdiff_t>(i * stride);
        };

        for (size_t i = 0; i < n; ++i) {

            if (moved[i]) continue;

            // The head of the cycle is the only element whose original value is
            // overwritten before something else needs it, so it is the only one
            // that has to be held aside.
            std::copy(at(values, i), at(values, i + 1), head.begin());

            size_t dst = i;
            for (;;) {

                moved[dst] = true;
                const size_t src = perm[dst];

                if (src == i) {

                    std::copy(head.begin(), head.end(), at(values, dst));
                    break;
                }

                std::copy(at(values, src), at(values, src + 1), at(values, dst));
                dst = src;
            }
        }
    }

    // Cell index on one axis, always in [0, 1023]. Written so that the two
    // comparisons are false for a NaN: a non-finite coordinate takes the first
    // branch and lands in cell 0, and no NaN ever reaches the cast.
    std::uint32_t mortonCell(float v, float lo, float invSpan) {

        const float t = (v - lo) * invSpan;
        if (!(t > 0.f)) return 0u;
        if (!(t < 1.f)) return 1023u;

        // t < 1 strictly and multiplying by a power of two is exact, so the
        // product is below 1024 and the cast cannot reach it; the min is there
        // so that stays true if anyone edits the constants.
        return std::min(static_cast<std::uint32_t>(t * 1024.f), 1023u);
    }

    // Ten bits spread to every third bit: 0b--98'7654'3210 becomes
    // 0b9--8--7--6--5--4--3--2--1--0. The published shift-and-mask expansion;
    // three of these, shifted by 2 / 1 / 0, are the whole encoder.
    std::uint32_t spread3(std::uint32_t v) {

        v &= 0x000003ffu;
        v = (v | (v << 16)) & 0xff0000ffu;
        v = (v | (v << 8)) & 0x0300f00fu;
        v = (v | (v << 4)) & 0x030c30c3u;
        v = (v | (v << 2)) & 0x09249249u;
        return v;
    }

    // x most significant within each 3-bit group. See the convention note on
    // SplatData::reorderMorton.
    std::uint32_t mortonKey(std::uint32_t x, std::uint32_t y, std::uint32_t z) {

        return (spread3(x) << 2) | (spread3(y) << 1) | spread3(z);
    }

}// namespace

size_t SplatData::removeOutliers(const OutlierPolicy& policy) {

    const size_t n = count();
    if (n == 0) return 0;

    // --- the two distributions the rule is expressed against ----------------
    std::vector<float> sizes;
    sizes.reserve(n);
    for (const auto& s : scales) sizes.push_back(std::max({s.x, s.y, s.z}));

    std::vector<float> xs, ys, zs;
    xs.reserve(n);
    ys.reserve(n);
    zs.reserve(n);
    for (const auto& m : means) {

        xs.push_back(m.x);
        ys.push_back(m.y);
        zs.push_back(m.z);
    }
    // Component-wise median, not the bounding-box centre: on a scan the box
    // centre is halfway between two outliers and points at empty air.
    const Vector3 centre{medianOf(std::move(xs)), medianOf(std::move(ys)), medianOf(std::move(zs))};

    std::vector<float> radii;
    radii.reserve(n);
    for (const auto& m : means) radii.push_back(m.distanceTo(centre));

    // The cloud's robust radius, which is what "big" and "far" are measured
    // against. A cloud with no spread at all (one splat, or every splat on
    // top of the median) has no scale to reason with: r is 0, every limit
    // collapses to 0, and the strictly-greater tests below would remove
    // everything — so the rules are skipped outright.
    const float r = percentile(radii, policy.radiusPercentile);
    if (!(r > 0.f)) return 0;

    const float sizeLimit = std::max(policy.sizeVsRadius * r,
                                     policy.sizeVsPeers * percentile(sizes, policy.sizePercentile));
    const float distanceLimit = policy.distanceVsRadius * r;

    // --- the mask -----------------------------------------------------------
    // Strictly greater throughout. Non-finite inputs fail the comparison and
    // are kept; culling is not the place to launder NaNs, and the shader
    // already refuses to draw them.
    std::vector<bool> keep(n, true);
    size_t removed = 0;
    for (size_t i = 0; i < n; ++i) {

        if (sizes[i] > sizeLimit || radii[i] > distanceLimit) {

            keep[i] = false;
            ++removed;
        }
    }

    if (removed == 0) return 0;

    compact(means, keep);
    compact(scales, keep);
    compact(rotations, keep);
    compact(opacities, keep);
    compact(sh, keep, static_cast<size_t>(coeffCount()) * 3);

    for (auto& [name, values] : extras) compact(values, keep);

    return removed;
}

std::vector<std::uint32_t> SplatData::reorderMorton(float boundsPercentile) {

    const size_t n = count();

    std::vector<std::uint32_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = static_cast<std::uint32_t>(i);
    if (n < 2) return perm;

    // --- the robust grid ----------------------------------------------------
    // One axis at a time, so only one coordinate array (plus percentile()'s
    // working copy) is ever live: at five million splats that is the
    // difference between 40 MB and 100 MB of scratch.
    const float p = std::clamp(boundsPercentile, 0.5f, 1.f);

    float lo[3]{}, invSpan[3]{};
    for (int axis = 0; axis < 3; ++axis) {

        std::vector<float> v;
        v.reserve(n);
        for (const auto& m : means) v.push_back(axis == 0 ? m.x : (axis == 1 ? m.y : m.z));

        const float hi = percentile(v, p);
        lo[axis] = percentile(std::move(v), 1.f - p);

        // An axis with no spread — a planar cloud, a single column — has no
        // grid to build, so every splat lands in cell 0 and the other two axes
        // order the cloud between them. Written as `hi > lo` rather than
        // `hi <= lo` so that non-finite bounds take this branch as well, which
        // degrades the whole reorder to a stable no-op instead of to nonsense.
        invSpan[axis] = (hi > lo[axis]) ? 1.f / (hi - lo[axis]) : 0.f;
    }

    // --- the keys, and a stable sort of the indices by them ------------------
    std::vector<std::uint32_t> keys(n);
    for (size_t i = 0; i < n; ++i) {

        keys[i] = mortonKey(mortonCell(means[i].x, lo[0], invSpan[0]),
                            mortonCell(means[i].y, lo[1], invSpan[1]),
                            mortonCell(means[i].z, lo[2], invSpan[2]));
    }

    // Stable: splats sharing a cell keep file order, which is what makes a
    // second call an exact no-op rather than an arbitrary reshuffle of ties.
    std::stable_sort(perm.begin(), perm.end(),
                     [&keys](std::uint32_t a, std::uint32_t b) { return keys[a] < keys[b]; });

    bool identity = true;
    for (size_t i = 0; i < n && identity; ++i) identity = perm[i] == static_cast<std::uint32_t>(i);
    if (identity) return perm;

    // --- move every array together ------------------------------------------
    std::vector<bool> moved(n);

    permute(means, perm, moved);
    permute(scales, perm, moved);
    permute(rotations, perm, moved);
    permute(opacities, perm, moved);
    permute(sh, perm, moved, static_cast<size_t>(coeffCount()) * 3);

    for (auto& [name, values] : extras) permute(values, perm, moved);

    return perm;
}

bool SplatData::validate(std::string* why) const {

    auto fail = [why](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };

    if (shDegree < 0 || shDegree > splats::MAX_SH_DEGREE) {

        return fail("SH degree " + std::to_string(shDegree) + " out of range 0-" +
                    std::to_string(splats::MAX_SH_DEGREE));
    }

    const auto n = count();
    if (scales.size() != n) return fail("scales size mismatch");
    if (rotations.size() != n) return fail("rotations size mismatch");
    if (opacities.size() != n) return fail("opacities size mismatch");
    if (sh.size() != n * static_cast<size_t>(coeffCount()) * 3) return fail("sh size mismatch");

    for (const auto& [name, values] : extras) {

        if (values.size() != n) return fail("extra property '" + name + "' size mismatch");
    }

    return true;
}


SplatData SplatGenerator::generate(const Options& options) {

    SplatData data;
    data.resize(options.count, options.shDegree);

    Rng rng(options.seed);

    const int coeffs = data.coeffCount();

    for (size_t i = 0; i < options.count; ++i) {

        data.means[i].set(
                rng.range(-0.5f * options.extent.x, 0.5f * options.extent.x),
                rng.range(-0.5f * options.extent.y, 0.5f * options.extent.y),
                rng.range(-0.5f * options.extent.z, 0.5f * options.extent.z));

        const float base = rng.range(options.minScale, options.maxScale);
        data.scales[i].set(
                base * rng.range(1.f, options.anisotropy),
                base * rng.range(1.f, options.anisotropy),
                base * rng.range(1.f, options.anisotropy));

        // Uniform-ish random orientation, then optionally de-normalised so the
        // consumer has to deal with it.
        Quaternion q(rng.range(-1.f, 1.f), rng.range(-1.f, 1.f),
                     rng.range(-1.f, 1.f), rng.range(-1.f, 1.f));
        if (q.lengthSq() <= 1e-12f) q.set(0.f, 0.f, 0.f, 1.f);
        q.normalize();
        if (options.unnormalizedRotations) {

            const float k = rng.range(0.3f, 3.f);
            q.set(q.x * k, q.y * k, q.z * k, q.w * k);
        }
        data.rotations[i] = q;

        data.opacities[i] = rng.range(options.minOpacity, options.maxOpacity);

        // Saturated, well-separated colours: correctness is much easier to see
        // by eye when neighbouring splats are not all the same beige.
        const Vector3 rgb{
                0.15f + 0.85f * rng.unit(),
                0.15f + 0.85f * rng.unit(),
                0.15f + 0.85f * rng.unit()};
        data.setDcColor(i, rgb);

        auto* c = data.shAt(i);
        for (int k = 1; k < coeffs; ++k) {

            c[k * 3 + 0] = options.higherOrderAmplitude * rng.range(-1.f, 1.f);
            c[k * 3 + 1] = options.higherOrderAmplitude * rng.range(-1.f, 1.f);
            c[k * 3 + 2] = options.higherOrderAmplitude * rng.range(-1.f, 1.f);
        }

        if (options.includeDegenerates) {

            if (i % 37 == 0) data.scales[i].set(0.f, 0.f, 0.f);
            if (i % 53 == 0) data.opacities[i] = 1e-6f;
        }
    }

    return data;
}
