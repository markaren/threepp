
#include "threepp/extras/editor/TerrainSculpt.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // One falloff shape for every brush (the plan pins this): full strength at
    // the centre, zero at the rim, with no corner at either end. Two brushes
    // with two different falloffs feel like two tools that disagree.
    float falloff(float normalisedDistance) {

        const float t = std::clamp(1.f - normalisedDistance, 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

}// namespace


const char* TerrainBrush::label(Kind kind) {

    switch (kind) {
        case Kind::Raise: return "Raise / Lower";
        case Kind::Smooth: return "Smooth";
        case Kind::Flatten: return "Flatten";
    }
    return "Raise / Lower";
}


float TerrainLattice::cellSize() const {

    return std::min(std::abs(stepX), std::abs(stepZ));
}

TerrainLattice TerrainLattice::of(const BufferGeometry& geometry, int dim) {

    TerrainLattice out;
    const auto* position = geometry.getAttribute<float>("position");
    if (!position || dim < 2) return out;
    if (position->count() != dim * dim) return out;

    const auto& a = position->array();
    const auto at = [&a](int index, int component) {
        return a[static_cast<size_t>(index) * 3 + component];
    };

    out.dim = dim;
    out.x0 = at(0, 0);
    out.z0 = at(0, 2);
    out.stepX = at(1, 0) - out.x0;                              // ix + 1
    out.stepZ = at(dim, 2) - out.z0;                            // iz + 1
    // A degenerate step would make every lattice lookup a division by zero.
    if (std::abs(out.stepX) < 1e-9f || std::abs(out.stepZ) < 1e-9f) out.dim = 0;
    return out;
}


void TerrainSculpt::Rect::add(int x, int z) {

    if (empty()) {
        x0 = x1 = x;
        z0 = z1 = z;
        return;
    }
    x0 = std::min(x0, x);
    x1 = std::max(x1, x);
    z0 = std::min(z0, z);
    z1 = std::max(z1, z);
}

void TerrainSculpt::Rect::grow(int ring, int dim) {

    if (empty()) return;
    x0 = std::max(x0 - ring, 0);
    z0 = std::max(z0 - ring, 0);
    x1 = std::min(x1 + ring, dim - 1);
    z1 = std::min(z1 + ring, dim - 1);
}


bool TerrainSculpt::sample(const std::vector<float>& heights, const TerrainLattice& lattice,
                           float wx, float wz, float& out) {

    if (!lattice.valid()) return false;
    const int dim = lattice.dim;
    if (heights.size() != static_cast<size_t>(dim) * dim) return false;

    const float fx = lattice.latticeX(wx);
    const float fz = lattice.latticeZ(wz);
    if (fx < 0.f || fz < 0.f) return false;
    if (fx > static_cast<float>(dim - 1) || fz > static_cast<float>(dim - 1)) return false;

    const int x0 = std::min(static_cast<int>(fx), dim - 2);
    const int z0 = std::min(static_cast<int>(fz), dim - 2);
    const float tx = fx - static_cast<float>(x0);
    const float tz = fz - static_cast<float>(z0);

    const auto at = [&](int x, int z) { return heights[static_cast<size_t>(z) * dim + x]; };
    const float a = at(x0, z0), b = at(x0 + 1, z0);
    const float c = at(x0, z0 + 1), d = at(x0 + 1, z0 + 1);
    out = (a + (b - a) * tx) * (1.f - tz) + (c + (d - c) * tx) * tz;
    return true;
}

bool TerrainSculpt::raycast(const std::vector<float>& heights, const TerrainLattice& lattice,
                            const Vector3& origin, const Vector3& direction,
                            float maxDistance, Vector3& hit) {

    if (!lattice.valid()) return false;

    Vector3 dir = direction;
    if (dir.length() < 1e-9f) return false;
    dir.normalize();

    const float step = std::max(lattice.cellSize() * 0.5f, 1e-4f);
    const int maxSteps = std::min(static_cast<int>(maxDistance / step) + 1, 20000);

    // Walk until the ray is under the surface, then bisect the last interval.
    // Starting "already under" is not a hit: that is a ray whose origin is
    // inside the ground, and reporting the entry point of the NEXT crossing is
    // what a brush wants, not the point behind the camera.
    float previousT = 0.f;
    float previousGap = 0.f;
    bool havePrevious = false;

    for (int i = 0; i <= maxSteps; ++i) {
        const float t = std::min(static_cast<float>(i) * step, maxDistance);
        const Vector3 p = origin + dir * t;
        float surface = 0.f;
        if (!sample(heights, lattice, p.x, p.z, surface)) {
            havePrevious = false;// off the patch: the interval is not bracketing
            if (t >= maxDistance) break;
            continue;
        }
        const float gap = p.y - surface;
        if (gap <= 0.f) {
            if (!havePrevious) {
                if (i == 0) return false;// origin is already underground
                havePrevious = true;
                previousT = std::max(t - step, 0.f);
                previousGap = 1.f;
            }
            float lo = previousT, hi = t;
            float loGap = previousGap;
            for (int b = 0; b < 24; ++b) {
                const float mid = 0.5f * (lo + hi);
                const Vector3 m = origin + dir * mid;
                float s = 0.f;
                if (!sample(heights, lattice, m.x, m.z, s)) break;
                const float g = m.y - s;
                if ((g <= 0.f) == (loGap <= 0.f)) {
                    lo = mid;
                    loGap = g;
                } else {
                    hi = mid;
                }
            }
            hit = origin + dir * (0.5f * (lo + hi));
            return true;
        }
        previousT = t;
        previousGap = gap;
        havePrevious = true;
        if (t >= maxDistance) break;
    }
    return false;
}

TerrainSculpt::Rect TerrainSculpt::apply(std::vector<float>& heights, const TerrainLattice& lattice,
                                         const TerrainBrush& brush, float cx, float cz,
                                         float dt, float flattenTarget) {

    Rect rect;
    if (!lattice.valid()) return rect;
    const int dim = lattice.dim;
    if (heights.size() != static_cast<size_t>(dim) * dim) return rect;

    const float radius = std::max(brush.radius, 1e-3f);
    // The brush is round in WORLD units; the lattice may not be square, so the
    // index-space half-extent is per axis.
    const float spanX = radius / std::abs(lattice.stepX);
    const float spanZ = radius / std::abs(lattice.stepZ);
    const float centreX = lattice.latticeX(cx);
    const float centreZ = lattice.latticeZ(cz);

    const int ix0 = std::max(static_cast<int>(std::floor(centreX - spanX)), 0);
    const int ix1 = std::min(static_cast<int>(std::ceil(centreX + spanX)), dim - 1);
    const int iz0 = std::max(static_cast<int>(std::floor(centreZ - spanZ)), 0);
    const int iz1 = std::min(static_cast<int>(std::ceil(centreZ + spanZ)), dim - 1);
    if (ix1 < ix0 || iz1 < iz0) return rect;

    const auto index = [dim](int x, int z) { return static_cast<size_t>(z) * dim + x; };
    // Smooth reads its neighbourhood, so it cannot write in place: a cell
    // already blended would feed the next one and the brush would smear along
    // the scan order instead of blurring.
    std::vector<float> source;
    if (brush.kind == TerrainBrush::Kind::Smooth) source = heights;

    const float sign = brush.invert ? -1.f : 1.f;
    const float amount = std::max(dt, 0.f);

    for (int z = iz0; z <= iz1; ++z) {
        for (int x = ix0; x <= ix1; ++x) {
            const float dx = (static_cast<float>(x) - centreX) * lattice.stepX;
            const float dz = (static_cast<float>(z) - centreZ) * lattice.stepZ;
            const float d = std::sqrt(dx * dx + dz * dz) / radius;
            if (d >= 1.f) continue;
            const float w = falloff(d);
            if (w <= 0.f) continue;

            float& h = heights[index(x, z)];
            switch (brush.kind) {
                case TerrainBrush::Kind::Raise:
                    h += sign * brush.strength * w * amount;
                    break;
                case TerrainBrush::Kind::Smooth: {
                    const int xm = std::max(x - 1, 0), xp = std::min(x + 1, dim - 1);
                    const int zm = std::max(z - 1, 0), zp = std::min(z + 1, dim - 1);
                    const float average = 0.25f * (source[index(xm, z)] + source[index(xp, z)] +
                                                   source[index(x, zm)] + source[index(x, zp)]);
                    const float k = std::clamp(brush.strength * w * amount, 0.f, 1.f);
                    h += (average - h) * k;
                    break;
                }
                case TerrainBrush::Kind::Flatten: {
                    const float k = std::clamp(brush.strength * w * amount, 0.f, 1.f);
                    h += (flattenTarget - h) * k;
                    break;
                }
            }
            rect.add(x, z);
        }
    }
    return rect;
}

void TerrainSculpt::refresh(BufferGeometry& geometry, const std::vector<float>& heights,
                            const TerrainLattice& lattice, const Rect& rect) {

    if (!lattice.valid() || rect.empty()) return;
    const int dim = lattice.dim;
    if (heights.size() != static_cast<size_t>(dim) * dim) return;

    auto* position = geometry.getAttribute<float>("position");
    auto* normal = geometry.getAttribute<float>("normal");
    if (!position || position->count() != dim * dim) return;

    // The normals of the ring OUTSIDE the written cells change too — a central
    // difference reaches one cell out — so the refresh covers rect+1 while the
    // heights only need rect.
    Rect touched = rect;
    touched.grow(1, dim);

    auto& pos = position->array();
    const auto index = [dim](int x, int z) { return static_cast<size_t>(z) * dim + x; };

    for (int z = rect.z0; z <= rect.z1; ++z) {
        for (int x = rect.x0; x <= rect.x1; ++x) {
            pos[index(x, z) * 3 + 1] = heights[index(x, z)];
        }
    }
    position->needsUpdate();

    if (!normal || normal->count() != dim * dim) return;
    auto& nrm = normal->array();
    // Analytic lattice normal: n = normalize(-dh/dx, 1, -dh/dz), with the axis
    // signs the geometry actually uses folded in through stepX/stepZ.
    for (int z = touched.z0; z <= touched.z1; ++z) {
        for (int x = touched.x0; x <= touched.x1; ++x) {
            const int xm = std::max(x - 1, 0), xp = std::min(x + 1, dim - 1);
            const int zm = std::max(z - 1, 0), zp = std::min(z + 1, dim - 1);
            const float runX = static_cast<float>(xp - xm) * lattice.stepX;
            const float runZ = static_cast<float>(zp - zm) * lattice.stepZ;
            const float dhdx = std::abs(runX) > 1e-9f
                                       ? (heights[index(xp, z)] - heights[index(xm, z)]) / runX
                                       : 0.f;
            const float dhdz = std::abs(runZ) > 1e-9f
                                       ? (heights[index(x, zp)] - heights[index(x, zm)]) / runZ
                                       : 0.f;
            Vector3 n(-dhdx, 1.f, -dhdz);
            n.normalize();
            const size_t o = index(x, z) * 3;
            nrm[o + 0] = n.x;
            nrm[o + 1] = n.y;
            nrm[o + 2] = n.z;
        }
    }
    normal->needsUpdate();
}

TerrainSculpt::Patch TerrainSculpt::diff(const std::vector<float>& before,
                                         const std::vector<float>& after, int dim) {

    Patch patch;
    if (dim < 1 || before.size() != after.size()) return patch;
    if (before.size() != static_cast<size_t>(dim) * dim) return patch;

    Rect rect;
    for (int z = 0; z < dim; ++z) {
        for (int x = 0; x < dim; ++x) {
            const size_t i = static_cast<size_t>(z) * dim + x;
            // Bit-exact, not epsilon: a height the stroke moved by a hair is
            // still a height undo has to put back exactly.
            if (before[i] != after[i]) rect.add(x, z);
        }
    }
    if (rect.empty()) return patch;

    patch.x0 = rect.x0;
    patch.z0 = rect.z0;
    patch.w = rect.x1 - rect.x0 + 1;
    patch.h = rect.z1 - rect.z0 + 1;
    patch.before.resize(static_cast<size_t>(patch.w) * patch.h);
    patch.after.resize(patch.before.size());
    for (int z = 0; z < patch.h; ++z) {
        for (int x = 0; x < patch.w; ++x) {
            const size_t src = static_cast<size_t>(patch.z0 + z) * dim + (patch.x0 + x);
            const size_t dst = static_cast<size_t>(z) * patch.w + x;
            patch.before[dst] = before[src];
            patch.after[dst] = after[src];
        }
    }
    return patch;
}

void TerrainSculpt::applyPatch(std::vector<float>& heights, int dim,
                               const Patch& patch, bool useBefore) {

    if (patch.empty() || dim < 1) return;
    if (heights.size() != static_cast<size_t>(dim) * dim) return;
    const auto& source = useBefore ? patch.before : patch.after;
    if (source.size() != static_cast<size_t>(patch.w) * patch.h) return;

    for (int z = 0; z < patch.h; ++z) {
        for (int x = 0; x < patch.w; ++x) {
            const size_t dst = static_cast<size_t>(patch.z0 + z) * dim + (patch.x0 + x);
            heights[dst] = source[static_cast<size_t>(z) * patch.w + x];
        }
    }
}
