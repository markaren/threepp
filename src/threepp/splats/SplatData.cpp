
#include "threepp/splats/SplatData.hpp"

#include "threepp/math/Rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

using namespace threepp;


float splats::medianNeighbourSpacing(const std::vector<Vector3>& points, size_t sampleCount) {

    const size_t n = points.size();

    Vector3 lo{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()};
    Vector3 hi{-lo.x, -lo.y, -lo.z};
    size_t finite = 0;
    for (const auto& p : points) {

        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        lo.min(p);
        hi.max(p);
        ++finite;
    }
    if (finite < 2) return 0.f;

    // Cell edge: twice the spacing n points would have spread evenly through
    // the bounds. A flat or linear cloud has a zero extent along one or two
    // axes; those axes are left out and the root is taken over the axes that
    // remain, so a sheet is sized by area and a line by length.
    const Vector3 extent{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
    const float longest = std::max({extent.x, extent.y, extent.z});
    if (!(longest > 0.f)) return 0.f;// every point coincides
    float measure = 1.f;
    int axes = 0;
    for (const float e : {extent.x, extent.y, extent.z}) {
        if (e > longest * 1e-3f) {
            measure *= e;
            ++axes;
        }
    }
    float cell = 2.f * std::pow(measure / static_cast<float>(finite), 1.f / static_cast<float>(axes));
    if (!(cell > 0.f)) cell = longest;

    // 21 bits per axis. The grid is at most longest / cell cells wide, and
    // the cell size ties that to the point count, so the clamp below never
    // engages for a cloud that fits in memory.
    constexpr uint64_t kAxisBits = 21;
    constexpr uint64_t kAxisMask = (uint64_t{1} << kAxisBits) - 1;
    const auto cellOf = [&](const Vector3& p, int64_t& ix, int64_t& iy, int64_t& iz) {
        ix = static_cast<int64_t>(std::floor((p.x - lo.x) / cell));
        iy = static_cast<int64_t>(std::floor((p.y - lo.y) / cell));
        iz = static_cast<int64_t>(std::floor((p.z - lo.z) / cell));
    };
    const auto keyOf = [&](int64_t ix, int64_t iy, int64_t iz) {
        const uint64_t ux = static_cast<uint64_t>(std::clamp<int64_t>(ix, 0, static_cast<int64_t>(kAxisMask)));
        const uint64_t uy = static_cast<uint64_t>(std::clamp<int64_t>(iy, 0, static_cast<int64_t>(kAxisMask)));
        const uint64_t uz = static_cast<uint64_t>(std::clamp<int64_t>(iz, 0, static_cast<int64_t>(kAxisMask)));
        return (ux << (2 * kAxisBits)) | (uy << kAxisBits) | uz;
    };

    // Sorted (cell key, point index): a cell's members are one contiguous run,
    // found by lower_bound.
    std::vector<std::pair<uint64_t, uint32_t>> cells;
    cells.reserve(finite);
    for (size_t i = 0; i < n; ++i) {

        const auto& p = points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        int64_t ix, iy, iz;
        cellOf(p, ix, iy, iz);
        cells.emplace_back(keyOf(ix, iy, iz), static_cast<uint32_t>(i));
    }
    std::sort(cells.begin(), cells.end());

    const auto nearestInRing = [&](const Vector3& q, uint32_t self, int64_t ring, float& bestD2) {
        int64_t cx, cy, cz;
        cellOf(q, cx, cy, cz);
        for (int64_t dz = -ring; dz <= ring; ++dz)
            for (int64_t dy = -ring; dy <= ring; ++dy)
                for (int64_t dx = -ring; dx <= ring; ++dx) {

                    // Only the shell of this ring: the interior was searched
                    // by the previous, smaller ring.
                    if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != ring) continue;
                    const uint64_t key = keyOf(cx + dx, cy + dy, cz + dz);
                    auto it = std::lower_bound(cells.begin(), cells.end(),
                                               std::make_pair(key, uint32_t{0}));
                    for (; it != cells.end() && it->first == key; ++it) {

                        if (it->second == self) continue;
                        const float d2 = q.distanceToSquared(points[it->second]);
                        if (d2 < bestD2) bestD2 = d2;
                    }
                }
    };

    const size_t stride = std::max<size_t>(1, cells.size() / std::max<size_t>(1, sampleCount));
    std::vector<float> spacing;
    spacing.reserve(cells.size() / stride + 1);
    constexpr int64_t kMaxRing = 3;
    for (size_t s = 0; s < cells.size(); s += stride) {

        const uint32_t idx = cells[s].second;
        const Vector3& q = points[idx];
        float bestD2 = std::numeric_limits<float>::infinity();
        for (int64_t ring = 0; ring <= kMaxRing; ++ring) {

            nearestInRing(q, idx, ring, bestD2);
            // Exact once the best candidate is closer than the unsearched
            // shell can be.
            const float reach = static_cast<float>(ring) * cell;
            if (bestD2 <= reach * reach) break;
        }
        if (bestD2 == 0.f) continue;// a duplicate point says nothing about spacing
        spacing.push_back(std::isfinite(bestD2) ? std::sqrt(bestD2)
                                                : static_cast<float>(kMaxRing + 1) * cell);
    }
    if (spacing.empty()) return 0.f;

    const size_t mid = spacing.size() / 2;
    std::nth_element(spacing.begin(), spacing.begin() + static_cast<std::ptrdiff_t>(mid), spacing.end());
    return spacing[mid];
}

void SplatData::resize(size_t n, int degree) {

    shDegree = std::clamp(degree, 0, splats::MAX_SH_DEGREE);

    means.assign(n, Vector3{});
    scales.assign(n, Vector3{});
    rotations.assign(n, SplatQuat{});
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

size_t SplatData::removeOutliers() {

    return removeOutliers(OutlierPolicy{});
}

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


SplatData SplatGenerator::generate() {

    return generate(Options{});
}

SplatData SplatGenerator::generate(const Options& options) {

    SplatData data;
    data.resize(options.count, options.shDegree);

    math::Rng rng(options.seed);

    const int coeffs = data.coeffCount();

    for (size_t i = 0; i < options.count; ++i) {

        // Every multi-draw below is sequenced through named locals: draws in
        // constructor argument lists happen in unspecified order, which is
        // exactly how "same seed, same cloud on every platform" quietly
        // becomes false. The old xorshift had this bug; don't reintroduce it.
        const float mx = rng.nextFloatSpread(options.extent.x);
        const float my = rng.nextFloatSpread(options.extent.y);
        const float mz = rng.nextFloatSpread(options.extent.z);
        data.means[i].set(mx, my, mz);

        const float base = rng.nextFloat(options.minScale, options.maxScale);
        const float sx = rng.nextFloat(1.f, options.anisotropy);
        const float sy = rng.nextFloat(1.f, options.anisotropy);
        const float sz = rng.nextFloat(1.f, options.anisotropy);
        data.scales[i].set(base * sx, base * sy, base * sz);

        // Uniform-ish random orientation, then optionally de-normalised so the
        // consumer has to deal with it.
        const float qx = rng.nextFloat(-1.f, 1.f);
        const float qy = rng.nextFloat(-1.f, 1.f);
        const float qz = rng.nextFloat(-1.f, 1.f);
        const float qw = rng.nextFloat(-1.f, 1.f);
        SplatQuat q(qx, qy, qz, qw);
        if (q.lengthSq() <= 1e-12f) q.set(0.f, 0.f, 0.f, 1.f);
        q.normalize();
        if (options.unnormalizedRotations) {

            const float k = rng.nextFloat(0.3f, 3.f);
            q.set(q.x * k, q.y * k, q.z * k, q.w * k);
        }
        data.rotations[i] = q;

        data.opacities[i] = rng.nextFloat(options.minOpacity, options.maxOpacity);

        // Saturated, well-separated colours: correctness is much easier to see
        // by eye when neighbouring splats are not all the same beige.
        const float cr = rng.nextFloat();
        const float cg = rng.nextFloat();
        const float cb = rng.nextFloat();
        const Vector3 rgb{0.15f + 0.85f * cr, 0.15f + 0.85f * cg, 0.15f + 0.85f * cb};
        data.setDcColor(i, rgb);

        auto* c = data.shAt(i);
        for (int k = 1; k < coeffs; ++k) {

            c[k * 3 + 0] = options.higherOrderAmplitude * rng.nextFloat(-1.f, 1.f);
            c[k * 3 + 1] = options.higherOrderAmplitude * rng.nextFloat(-1.f, 1.f);
            c[k * 3 + 2] = options.higherOrderAmplitude * rng.nextFloat(-1.f, 1.f);
        }

        if (options.includeDegenerates) {

            if (i % 37 == 0) data.scales[i].set(0.f, 0.f, 0.f);
            if (i % 53 == 0) data.opacities[i] = 1e-6f;
        }
    }

    return data;
}
