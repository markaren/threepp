
#include "threepp/extras/curves/RoadAlignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace threepp;

namespace {

    constexpr float kPi = 3.14159265358979f;
    // Curvature under this is a straight: 1e-7 per metre is a radius of ten
    // thousand kilometres.
    constexpr float kStraight = 1e-7f;
    // A piece shorter than this carries no surface and no station.
    constexpr float kShortest = 1e-6f;

    float dot2(const Vector2& a, const Vector2& b) { return a.x * b.x + a.y * b.y; }
    float cross2(const Vector2& a, const Vector2& b) { return a.x * b.y - a.y * b.x; }
    Vector2 add2(const Vector2& a, const Vector2& b) { return {a.x + b.x, a.y + b.y}; }
    Vector2 sub2(const Vector2& a, const Vector2& b) { return {a.x - b.x, a.y - b.y}; }
    Vector2 mul2(const Vector2& a, float s) { return {a.x * s, a.y * s}; }
    float len2(const Vector2& a) { return std::sqrt(a.x * a.x + a.y * a.y); }

    Vector2 unit2(const Vector2& a) {

        const float length = len2(a);
        return length > 1e-12f ? Vector2(a.x / length, a.y / length) : Vector2(1.f, 0.f);
    }

    float angleBetween(const Vector2& a, const Vector2& b) {

        return std::atan2(std::abs(cross2(a, b)), dot2(a, b));
    }

    // The arc that leaves `from` along `tangent` and reaches `to`. A chord
    // subtends twice the angle its own direction makes with the tangent — that
    // identity is the whole construction, and it is exact.
    Arc2 arcThrough(const Vector2& from, const Vector2& tangent, const Vector2& to) {

        Arc2 arc;
        arc.start = from;
        arc.tangent = tangent;

        const Vector2 chord = sub2(to, from);
        const float square = dot2(chord, chord);
        if (square < 1e-16f) return arc;// no length, no piece

        const float sideways = cross2(tangent, chord);
        const float ahead = dot2(tangent, chord);
        const float curvature = 2.f * sideways / square;
        if (std::abs(curvature) < kStraight) {
            arc.length = std::sqrt(square);
            return arc;
        }

        arc.curvature = curvature;
        // Signed sweep. atan2 carries the sign of `sideways`, which is also the
        // sign of the curvature, so the two never disagree about which way the
        // arc turns.
        const float turn = 2.f * std::atan2(sideways, ahead);
        arc.length = std::abs(turn) / std::abs(curvature);
        return arc;
    }

    // The classical biarc: two circular arcs from (a.point, a.tangent) to
    // (b.point, b.tangent) meeting tangentially at a joint the construction
    // places. Equal tangent lengths, so the joint follows from closing
    // V = d(T0 + T1 + 2Tj) with Tj a unit vector — one quadratic, one root.
    bool solveBiarc(const BiarcChain::Seed& a, const BiarcChain::Seed& b,
                    Arc2& first, Arc2& second) {

        const Vector2 span = sub2(b.point, a.point);
        const float square = dot2(span, span);
        if (square < 1e-14f) return false;

        // Closing |V/d - (T0 + T1)| = 2 gives A.d^2 + B.d - |V|^2 = 0 with
        // A = 2(1 - T0.T1) and B = 2 V.(T0 + T1); the tangent length is its
        // positive root.
        const Vector2 sum = add2(a.tangent, b.tangent);
        const float quadratic = 2.f * (1.f - dot2(a.tangent, b.tangent));
        const float linear = 2.f * dot2(span, sum);
        const float discriminant = std::sqrt(linear * linear + 4.f * quadratic * square);

        float d;
        if (linear > 0.f) {
            // The citardauq form. Adjacent seeds off a smooth curve have very
            // nearly PARALLEL tangents, so A is near zero and the textbook root
            // subtracts two numbers that agree to seven digits — it returned
            // noise for exactly the case that occurs everywhere, and a noisy
            // tangent length puts the joint somewhere that makes one arc sweep
            // most of a circle. This form has nothing to cancel, and it needs
            // no separate branch for A = 0.
            d = 2.f * square / (linear + discriminant);
        } else if (quadratic > 1e-9f) {
            d = (discriminant - linear) / (2.f * quadratic);
        } else {
            return false;// tangents pointing back at each other
        }
        if (!std::isfinite(d) || d <= 1e-9f) return false;

        const Vector2 joinTangent = unit2(mul2(sub2(mul2(span, 1.f / d), sum), 0.5f));
        const Vector2 join = add2(a.point, mul2(add2(a.tangent, joinTangent), d));

        first = arcThrough(a.point, a.tangent, join);
        second = arcThrough(join, joinTangent, b.point);
        // An arc that sweeps past half a circle between two adjacent seeds is
        // the road doubling back on itself, not a bend. Refuse it: a straight
        // costs one joint's continuity, where a loop costs the chain its whole
        // arc length.
        return std::abs(first.curvature) * first.length < kPi + 1e-3f &&
               std::abs(second.curvature) * second.length < kPi + 1e-3f;
    }

}// namespace


Vector2 Arc2::pointAt(float s) const {

    if (std::abs(curvature) < kStraight) {
        return {start.x + tangent.x * s, start.y + tangent.y * s};
    }
    const float signedRadius = 1.f / curvature;
    const Vector2 centre{start.x - tangent.y * signedRadius, start.y + tangent.x * signedRadius};
    const float sweep = s * curvature;
    const float c = std::cos(sweep), n = std::sin(sweep);
    const float ox = start.x - centre.x, oy = start.y - centre.y;
    return {centre.x + ox * c - oy * n, centre.y + ox * n + oy * c};
}

Vector2 Arc2::tangentAt(float s) const {

    const float sweep = s * curvature;
    const float c = std::cos(sweep), n = std::sin(sweep);
    return {tangent.x * c - tangent.y * n, tangent.x * n + tangent.y * c};
}

float Arc2::radius() const {

    return std::abs(curvature) < kStraight ? std::numeric_limits<float>::infinity()
                                           : 1.f / std::abs(curvature);
}

float Arc2::distanceTo(const Vector2& query) const {

    if (std::abs(curvature) < kStraight) {
        const Vector2 delta = sub2(query, start);
        const float along = std::clamp(dot2(delta, tangent), 0.f, length);
        return len2(sub2(query, {start.x + tangent.x * along, start.y + tangent.y * along}));
    }

    const float signedRadius = 1.f / curvature;
    const Vector2 centre{start.x - tangent.y * signedRadius, start.y + tangent.x * signedRadius};
    const Vector2 out = sub2(query, centre);
    const Vector2 begin = sub2(start, centre);

    // How far round the arc `query` lies, measured the way the arc sweeps.
    float angle = std::atan2(cross2(begin, out), dot2(begin, out));
    if (curvature < 0.f) angle = -angle;
    if (angle < 0.f) angle += 2.f * kPi;

    if (angle <= length * std::abs(curvature)) {
        return std::abs(len2(out) - std::abs(signedRadius));
    }
    // Past either end: the nearest point is the end itself.
    return std::min(len2(sub2(query, start)), len2(sub2(query, end())));
}


BiarcChain BiarcChain::fit(std::vector<Seed> seeds, bool closed,
                           const Limits& limits, Relaxation* relaxation) {

    BiarcChain chain;
    const std::size_t n = seeds.size();
    if (n < 2) return chain;
    const std::size_t intervals = closed ? n : n - 1;

    const std::vector<Seed> authored = seeds;
    std::vector<char> everMoved(n, 0);
    Relaxation report;

    // Which seed interval each piece came out of — what a violating radius is
    // traced back through to the seeds that have to give.
    std::vector<std::size_t> owner;

    const auto solve = [&] {
        chain.pieces_.clear();
        chain.starts_.clear();
        chain.seedStations_.assign(intervals + 1, 0.f);
        chain.seedPieces_.assign(intervals + 1, 0);
        owner.clear();

        float station = 0.f;
        for (std::size_t i = 0; i < intervals; ++i) {
            chain.seedStations_[i] = station;
            chain.seedPieces_[i] = chain.pieces_.size();

            const Seed& from = seeds[i];
            const Seed& to = seeds[(i + 1) % n];
            Arc2 first, second;
            if (solveBiarc(from, to, first, second)) {
                // A biarc through collinear seeds is two straights end to end,
                // which is one straight. Emitting both would put a station
                // boundary in the middle of a stretch that is not doing
                // anything.
                if (first.curvature == 0.f && second.curvature == 0.f &&
                    angleBetween(first.tangent, second.tangent) < 1e-6f) {
                    first.length += second.length;
                    second.length = 0.f;
                }
                for (const Arc2& piece : {first, second}) {
                    if (piece.length <= kShortest) continue;
                    chain.starts_.push_back(station);
                    chain.pieces_.push_back(piece);
                    owner.push_back(i);
                    station += piece.length;
                }
            } else {
                // Seeds no biarc joins: coincident, or with tangents that point
                // back at each other. A straight keeps the chain connected;
                // refinement never leaves a pair of curve samples like this.
                const Vector2 delta = sub2(to.point, from.point);
                const float span = len2(delta);
                if (span > kShortest) {
                    Arc2 line;
                    line.start = from.point;
                    line.tangent = unit2(delta);
                    line.length = span;
                    chain.starts_.push_back(station);
                    chain.pieces_.push_back(line);
                    owner.push_back(i);
                    station += span;
                }
            }
        }
        chain.seedStations_[intervals] = station;
        chain.seedPieces_[intervals] = chain.pieces_.size();
        chain.length_ = station;
    };

    solve();

    if (limits.minRadius > 0.f && !chain.pieces_.empty()) {

        const float alpha = std::clamp(limits.relaxation, 0.01f, 0.9f);
        // Smoothing window, in ARC LENGTH rather than in seeds. A bend has to
        // be spread over roughly the radius it is being opened out to, and a
        // stencil counted in seeds would take passes proportional to the SQUARE
        // of how finely the curve happened to be sampled to move that far.
        //
        // It also has to be WIDE ENOUGH FOR ITS OWN TAPER: displacing a seed by
        // h over a window w bends the road there by about h(pi/w)^2, so too
        // narrow a window creates a fresh violation at the edge of the stretch
        // it just fixed, and the clamp chases its own boundary outward forever.
        // That is what the stagnation test below widens it for.
        float window = std::max(limits.minRadius, 1e-3f);
        const float widest = std::max(limits.minRadius, 0.5f * chain.length_);

        // Where the chain would run if it were low-passed over `window` of arc
        // length: a raised-cosine average of the CHAIN, sampled at even
        // stations rather than at seeds, so refinement having left the seeds
        // unevenly spaced does not bias where the average lands.
        //
        // A low-pass is the operator that suits this. Pulling a seed toward the
        // midpoint of two others is not one — at a corner the two neighbours
        // straddle it and the midpoint is deep inside the turn, so the seed
        // overshoots and the polyline develops the kink the pass was called to
        // remove.
        const auto smoothedAt = [&](float centre) {
            constexpr int taps = 8;
            Vector2 sum{0.f, 0.f};
            float weight = 0.f;
            for (int m = -taps; m <= taps; ++m) {
                const float ratio = static_cast<float>(m) / static_cast<float>(taps);
                float at = centre + window * ratio;
                if (closed && chain.length_ > 1e-6f) {
                    at = std::fmod(at, chain.length_);
                    if (at < 0.f) at += chain.length_;
                }
                const float w = 0.5f * (1.f + std::cos(kPi * ratio));
                if (w <= 0.f) continue;
                sum = add2(sum, mul2(chain.pointAt(at), w));
                weight += w;
            }
            return weight > 1e-6f ? mul2(sum, 1.f / weight) : Vector2{};
        };

        // Smoothing opens a bend monotonically, so a chain whose tightest
        // radius has stopped improving is a chain that has converged as far as
        // it will — carrying on only bends the road further off the drawing for
        // nothing.
        float bestRadius = 0.f;
        int stale = 0;

        for (int pass = 0; pass <= std::max(limits.maxPasses, 0); ++pass) {

            // How hard each seed is being asked to give: how far the arc is
            // under the floor, tapering to nothing a window away. Continuous in
            // BOTH — a flag would flicker on and off between passes as
            // different arcs crossed the floor, and a displacement field
            // flickering at the seed spacing is exactly the high-frequency
            // wobble the pass exists to remove.
            std::vector<float> urgency(n, 0.f);
            int violating = 0;
            for (std::size_t p = 0; p < chain.pieces_.size(); ++p) {
                const float radius = chain.pieces_[p].radius();
                if (radius >= limits.minRadius) continue;
                ++violating;
                // Never quite zero as the radius reaches the floor: a weight
                // that vanishes exactly there takes hundreds of passes to
                // creep the last per cent.
                const float deficit = std::clamp(1.05f - radius / limits.minRadius, 0.05f, 1.f);
                const float from = chain.starts_[p];
                const float to = from + chain.pieces_[p].length;
                for (std::size_t j = 0; j < n; ++j) {
                    const float at = chain.seedStations_[j];
                    float gap = (at < from) ? from - at : (at > to ? at - to : 0.f);
                    if (closed && chain.length_ > 1e-6f) {
                        gap = std::min(gap, chain.length_ - gap);
                    }
                    if (gap >= window) continue;
                    const float taper = 0.5f * (1.f + std::cos(kPi * gap / window));
                    urgency[j] = std::max(urgency[j], deficit * taper);
                }
            }
            if (pass == 0) report.bends = violating;
            if (violating == 0 || pass == limits.maxPasses) break;

            const float tightest = chain.minRadius();
            if (tightest > bestRadius * 1.001f) {
                bestRadius = tightest;
                stale = 0;
            } else if (++stale > 12) {
                stale = 0;
                if (window >= widest) break;// as wide as the road: nothing left to spread over
                window = std::min(window * 1.6f, widest);
            }

            // Smooth, then SOLVE AGAIN. A radius is never patched in place: what
            // comes back is a biarc chain, so it is still tangent-continuous.
            std::vector<Vector2> next(n);
            for (std::size_t j = 0; j < n; ++j) next[j] = seeds[j].point;
            // No seed moves more than a fraction of the window in one pass, nor
            // more than a fraction of the SEED SPACING: the taper is what keeps
            // the deformation smooth, and a seed that outruns its own
            // neighbours by more than the gap between them puts a kink exactly
            // where the pass was called to put a curve.
            const float spacing = chain.length_ / static_cast<float>(intervals);
            const float furthest = std::min(0.15f * window, 0.35f * spacing);
            for (std::size_t j = 0; j < n; ++j) {
                if (urgency[j] <= 0.f) continue;
                if (!closed && (j == 0 || j + 1 == n)) continue;// the road starts where it was drawn
                const Vector2 target = smoothedAt(chain.seedStations_[j]);

                if (limits.freezeX) {
                    // The profile's abscissa IS plan station; moving it would
                    // put a seed at a station it is not at, so only the height
                    // gives.
                    const float rise = target.y - seeds[j].point.y;
                    next[j].y += std::clamp(alpha * urgency[j] * rise, -furthest, furthest);
                } else {
                    const Vector2 pull = sub2(target, seeds[j].point);
                    const float reach = len2(pull);
                    if (reach < 1e-9f) continue;
                    const float step = std::min(alpha * urgency[j] * reach, furthest);
                    next[j] = add2(seeds[j].point, mul2(pull, step / reach));
                }
                everMoved[j] = 1;
            }
            for (std::size_t j = 0; j < n; ++j) seeds[j].point = next[j];

            // Tangents follow the seeds, EVERYWHERE, from here on. A tangent
            // that disagrees with the chords either side of it is what makes a
            // biarc put out one arc a millimetre long turning sixty degrees —
            // a cusp the clamp then chases for the rest of its passes. The
            // bisector of the two chords cannot do that: it lies between them,
            // so the arc it produces turns no further than the polyline does.
            //
            // Where nothing moved this is the same tangent to within the seed
            // spacing squared, so the untouched stretches are not disturbed by
            // being included.
            //
            // The END tangents follow too. Pinning them pins the TOTAL TURN the
            // road has to make, and smoothing shortens it — same turn over less
            // road is a TIGHTER bend, so a clamp that held the end headings
            // fixed drove the radius the wrong way and never converged. The
            // ends keep the positions they were drawn at; where the road points
            // when it gets there is part of the bending.
            for (std::size_t j = 0; j < n; ++j) {
                Vector2 wanted;
                if (!closed && j == 0) wanted = unit2(sub2(seeds[1].point, seeds[0].point));
                else if (!closed && j + 1 == n) wanted = unit2(sub2(seeds[n - 1].point, seeds[n - 2].point));
                else {
                    const Vector2 back = unit2(sub2(seeds[j].point, seeds[(j + n - 1) % n].point));
                    const Vector2 ahead = unit2(sub2(seeds[(j + 1) % n].point, seeds[j].point));
                    const Vector2 bisector = add2(back, ahead);
                    wanted = len2(bisector) > 1e-6f ? unit2(bisector) : ahead;
                }
                seeds[j].tangent = wanted;
            }
            solve();
        }
    }

    for (std::size_t j = 0; j < n; ++j) {
        if (!everMoved[j]) continue;
        ++report.seeds;
        report.moved = std::max(report.moved, len2(sub2(seeds[j].point, authored[j].point)));
    }
    if (relaxation) *relaxation = report;
    return chain;
}

std::size_t BiarcChain::pieceAt(float station, float& local) const {

    if (pieces_.empty()) {
        local = 0.f;
        return 0;
    }
    const float clamped = std::clamp(station, 0.f, length_);
    std::size_t lo = 0, hi = pieces_.size();
    while (lo + 1 < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (starts_[mid] <= clamped) lo = mid;
        else hi = mid;
    }
    local = std::clamp(clamped - starts_[lo], 0.f, pieces_[lo].length);
    return lo;
}

Vector2 BiarcChain::pointAt(float station) const {

    if (pieces_.empty()) return {};
    float local = 0.f;
    return pieces_[pieceAt(station, local)].pointAt(local);
}

Vector2 BiarcChain::tangentAt(float station) const {

    if (pieces_.empty()) return {1.f, 0.f};
    float local = 0.f;
    return pieces_[pieceAt(station, local)].tangentAt(local);
}

float BiarcChain::curvatureAt(float station) const {

    if (pieces_.empty()) return 0.f;
    float local = 0.f;
    return pieces_[pieceAt(station, local)].curvature;
}

float BiarcChain::distanceTo(const Vector2& query, std::size_t first, std::size_t last) const {

    float best = std::numeric_limits<float>::infinity();
    for (std::size_t i = first; i < last && i < pieces_.size(); ++i) {
        best = std::min(best, pieces_[i].distanceTo(query));
    }
    return best;
}

float BiarcChain::minRadius() const {

    float best = std::numeric_limits<float>::infinity();
    for (const auto& piece : pieces_) best = std::min(best, piece.radius());
    return best;
}

float BiarcChain::maxAngleBreak(bool closed) const {

    float worst = 0.f;
    const std::size_t count = pieces_.size();
    for (std::size_t i = 0; i + 1 < count; ++i) {
        worst = std::max(worst, angleBetween(pieces_[i].endTangent(), pieces_[i + 1].tangent));
    }
    if (closed && count > 1) {
        worst = std::max(worst, angleBetween(pieces_[count - 1].endTangent(), pieces_[0].tangent));
    }
    return worst;
}


void RoadAlignment::sampleProfile(float station, float& height, float& grade) const {

    const auto& pieces = profile_.pieces();
    if (pieces.empty()) {
        height = 0.f;
        grade = 0.f;
        return;
    }

    // Pieces run left to right along the abscissa, which IS plan station.
    std::size_t lo = 0, hi = pieces.size();
    while (lo + 1 < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (pieces[mid].start.x <= station) lo = mid;
        else hi = mid;
    }
    const Arc2& piece = pieces[lo];

    const auto slopeOf = [](const Vector2& tangent) {
        return std::abs(tangent.x) > 1e-6f ? tangent.y / tangent.x : 0.f;
    };

    // Off either end — a station a rounding error past the last piece — runs
    // out along the tangent rather than stopping dead.
    const float startX = piece.start.x;
    const float endX = piece.end().x;
    if (station <= startX) {
        const float slope = slopeOf(piece.tangent);
        height = piece.start.y + (station - startX) * slope;
        grade = slope;
        return;
    }
    if (station >= endX) {
        const float slope = slopeOf(piece.endTangent());
        height = piece.end().y + (station - endX) * slope;
        grade = slope;
        return;
    }

    float local;
    if (std::abs(piece.curvature) < 1e-7f) {
        local = (station - startX) / piece.tangent.x;
    } else {
        // The abscissa advances monotonically while the grade stays clear of
        // vertical, which a road's does, so a bisection lands on it.
        float low = 0.f, high = piece.length;
        for (int i = 0; i < 40; ++i) {
            const float mid = 0.5f * (low + high);
            if (piece.pointAt(mid).x < station) low = mid;
            else high = mid;
        }
        local = 0.5f * (low + high);
    }
    const Vector2 at = piece.pointAt(local);
    height = at.y;
    grade = slopeOf(piece.tangentAt(local));
}

RoadAlignment RoadAlignment::build(const Curve3& path, const Params& params) {

    RoadAlignment out;

    const float half = std::max(params.width, 1e-4f) * 0.5f;
    const float planFloor = half * std::max(params.minRadiusFactor, 1.f);
    const float tolerance = std::max(params.tolerance, 1e-4f);
    const float stationAngle = std::clamp(params.stationAngle, 1e-4f, 1.f);
    const unsigned int maxSeeds = std::max(params.maxSeeds, 8u);

    // Seed parameters. A closed curve's last seed IS its first, so it is not
    // listed twice.
    const unsigned int initial = std::clamp(params.seeds, 4u, maxSeeds);
    std::vector<float> ts;
    ts.reserve(initial + 1);
    for (unsigned int i = 0; i < (params.closed ? initial : initial + 1); ++i) {
        ts.push_back(static_cast<float>(i) / static_cast<float>(initial));
    }

    // A seed is a point and the direction the road leaves it in, both taken
    // from the authored curve, both in PLAN — elevation is a chain of its own
    // further down.
    const auto seedsFrom = [&path](const std::vector<float>& parameters) {
        std::vector<BiarcChain::Seed> seeds;
        seeds.reserve(parameters.size());
        Vector3 point, tangent;
        Vector2 carried{1.f, 0.f};
        for (const float t : parameters) {
            path.getPoint(t, point);
            path.getTangent(t, tangent);
            const Vector2 plan{tangent.x, tangent.z};
            // A stretch running straight up or down says nothing about where
            // the road is heading; it keeps the heading it had going in.
            if (len2(plan) > 1e-5f) carried = unit2(plan);
            seeds.push_back({Vector2(point.x, point.z), carried});
        }
        return seeds;
    };

    // Refine against the authored curve BEFORE any radius floor applies: what
    // is measured here is how well the chain follows what was drawn, not how
    // well it follows what it was allowed to become.
    float fitted = 0.f;
    for (int pass = 0; pass < 16; ++pass) {

        const BiarcChain trial = BiarcChain::fit(seedsFrom(ts), params.closed, BiarcChain::Limits{});
        if (trial.empty()) return out;

        const auto& ranges = trial.seedPieces();
        std::vector<float> extra;
        fitted = 0.f;
        Vector3 sample;
        for (std::size_t k = 0; k + 1 < ranges.size(); ++k) {
            const float from = ts[k];
            const float to = (k + 1 < ts.size()) ? ts[k + 1] : 1.f;
            float worst = 0.f;
            for (int q = 1; q < 8; ++q) {
                const float t = from + (to - from) * static_cast<float>(q) / 8.f;
                path.getPoint(t, sample);
                worst = std::max(worst, trial.distanceTo(Vector2(sample.x, sample.z),
                                                         ranges[k], ranges[k + 1]));
            }
            fitted = std::max(fitted, worst);
            if (worst > tolerance) extra.push_back(0.5f * (from + to));
        }

        if (extra.empty() || ts.size() + extra.size() > maxSeeds) break;
        ts.insert(ts.end(), extra.begin(), extra.end());
        std::sort(ts.begin(), ts.end());
    }

    BiarcChain::Limits planLimits;
    planLimits.minRadius = planFloor;
    BiarcChain::Relaxation planRelaxation;
    out.plan_ = BiarcChain::fit(seedsFrom(ts), params.closed, planLimits, &planRelaxation);
    if (out.plan_.empty() || out.plan_.length() <= 1e-5f) return out;

    // The profile: authored elevation against the station the relaxed plan puts
    // it at, joined by the same solver in the (station, height) plane.
    {
        const auto& marks = out.plan_.seedStations();
        std::vector<BiarcChain::Seed> seeds;
        seeds.reserve(marks.size());
        Vector3 point, tangent;
        for (std::size_t i = 0; i < marks.size(); ++i) {
            const float t = (i < ts.size()) ? ts[i] : 1.f;
            path.getPoint(t, point);
            path.getTangent(t, tangent);
            const float planSpeed = std::sqrt(tangent.x * tangent.x + tangent.z * tangent.z);
            const float grade = planSpeed > 1e-5f ? tangent.y / planSpeed : 0.f;
            const float scale = 1.f / std::sqrt(1.f + grade * grade);
            seeds.push_back({Vector2(marks[i], point.y), Vector2(scale, grade * scale)});
        }
        BiarcChain::Limits profileLimits;
        profileLimits.minRadius = std::max(params.profileMinRadius, 0.f);
        profileLimits.freezeX = true;
        out.profile_ = BiarcChain::fit(std::move(seeds), false, profileLimits);
    }

    // Stations: both chains' piece boundaries, then subdivision until neither
    // heading nor grade turns by more than `stationAngle` across one.
    const float total = out.plan_.length();
    std::vector<float> marks = out.plan_.pieceStations();
    marks.push_back(total);
    for (const auto& piece : out.profile_.pieces()) {
        if (piece.start.x > 1e-4f && piece.start.x < total - 1e-4f) marks.push_back(piece.start.x);
    }
    std::sort(marks.begin(), marks.end());
    // A millimetre, not a rounding error. Two cross-sections that close
    // together differ mostly in where the float landed, so the direction from
    // one to the other is noise — and a road that reads as facetted BETWEEN
    // TWO STATIONS 0.2 MM APART is what a tighter merge left behind.
    marks.erase(std::unique(marks.begin(), marks.end(),
                            [](float a, float b) { return std::abs(a - b) < 1e-3f; }),
                marks.end());
    if (marks.size() < 2) return out;
    marks.back() = total;// the road ends where it ends, merge or no merge

    std::vector<int> splits(marks.size() - 1, 1);
    long long counted = 0;
    for (std::size_t i = 0; i + 1 < marks.size(); ++i) {
        const float span = marks[i + 1] - marks[i];
        const float heading = std::abs(out.plan_.curvatureAt(0.5f * (marks[i] + marks[i + 1]))) * span;
        float lowHeight, lowGrade, highHeight, highGrade;
        out.sampleProfile(marks[i], lowHeight, lowGrade);
        out.sampleProfile(marks[i + 1], highHeight, highGrade);
        const float pitch = std::abs(std::atan(highGrade) - std::atan(lowGrade));
        splits[i] = std::clamp(static_cast<int>(std::ceil(std::max(heading, pitch) / stationAngle)), 1, 256);
        // Never closer than a millimetre. Two cross-sections a float epsilon
        // apart differ only in their rounding, so the direction between them is
        // noise — and a normal read off it is a facet that is not there.
        splits[i] = std::min(splits[i], std::max(1, static_cast<int>(span * 1000.f)));
        counted += splits[i];
    }
    if (counted + 1 > static_cast<long long>(params.maxStations)) {
        const float scale = static_cast<float>(params.maxStations - 1) / static_cast<float>(counted);
        for (auto& split : splits) split = std::max(1, static_cast<int>(std::floor(static_cast<float>(split) * scale)));
    }

    const auto stationAt = [&out](float distance) {
        Station station;
        station.distance = distance;
        const Vector2 point = out.plan_.pointAt(distance);
        const Vector2 heading = out.plan_.tangentAt(distance);
        float height = 0.f, grade = 0.f;
        out.sampleProfile(distance, height, grade);
        station.point.set(point.x, height, point.y);
        const float scale = 1.f / std::sqrt(1.f + grade * grade);
        station.tangent.set(heading.x * scale, grade * scale, heading.y * scale);
        // Level side to side, always: a road banks nowhere, so the cross
        // direction is the HORIZONTAL perpendicular and nothing else.
        station.side.set(heading.y, 0.f, -heading.x);
        station.normal.crossVectors(station.tangent, station.side);
        return station;
    };

    out.stations_.reserve(static_cast<std::size_t>(counted) + 1);
    for (std::size_t i = 0; i + 1 < marks.size(); ++i) {
        const float span = marks[i + 1] - marks[i];
        for (int j = 0; j < splits[i]; ++j) {
            out.stations_.push_back(
                    stationAt(marks[i] + span * static_cast<float>(j) / static_cast<float>(splits[i])));
        }
    }
    out.stations_.push_back(stationAt(total));

    out.report_.fit = fitted;
    out.report_.planMinRadius = out.plan_.minRadius();
    out.report_.profileMinRadius = out.profile_.minRadius();
    out.report_.bendsRelaxed = planRelaxation.bends;
    out.report_.seedsRelaxed = planRelaxation.seeds;
    out.report_.seeds = ts.size();
    out.report_.planPieces = out.plan_.pieces().size();
    out.report_.profilePieces = out.profile_.pieces().size();

    // What the road cost the drawing: how far the surface's own centreline
    // ended up from the curve the user authored, radius floors and all.
    {
        Vector3 sample;
        float worst = 0.f;
        for (int i = 0; i <= 128; ++i) {
            path.getPoint(static_cast<float>(i) / 128.f, sample);
            float best = std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k + 1 < out.stations_.size(); ++k) {
                const Vector3& a = out.stations_[k].point;
                const Vector3& b = out.stations_[k + 1].point;
                Vector3 edge, delta;
                edge.copy(b).sub(a);
                delta.copy(sample).sub(a);
                const float square = edge.lengthSq();
                const float along = square > 1e-12f ? std::clamp(delta.dot(edge) / square, 0.f, 1.f) : 0.f;
                delta.copy(a).addScaledVector(edge, along).sub(sample);
                best = std::min(best, delta.length());
            }
            worst = std::max(worst, best);
        }
        out.report_.deviation = worst;
    }

    return out;
}
