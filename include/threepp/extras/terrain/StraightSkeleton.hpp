// Straight skeleton of a simple polygon — constant-pitch roof faces.
//
// The wavefront of a polygon shrinking at unit speed along every edge normal
// sweeps out a "roof": one PLANAR face per input edge, rising at a constant
// pitch, meeting its neighbours in hips (convex corners) and valleys (reflex
// corners). That is exactly the roof shape a hipped building carries, and it
// works over L / T / U / courtyard-free footprints where a bounding-rectangle
// gable fit gives up.
//
// Implementation: Felkel-style event simulation. Every wavefront vertex moves
// at velocity m = (nL + nR) / (1 + nL·nR) — the unique velocity that keeps it
// at distance t from BOTH adjacent edge lines, valid for convex and reflex
// corners alike. Two event kinds retire vertices:
//   • EDGE  — two adjacent vertices meet, the edge between them vanishes;
//   • SPLIT — a reflex vertex reaches a non-adjacent wavefront edge and cuts
//             the loop in two.
// Loops of ≤ 3 vertices are resolved analytically (intersect the three offset
// planes), which also sidesteps the parallel-edge degeneracy a plain event
// queue trips over on rectangles.
//
// Faces are recovered WITHOUT tracking chain order: every retired vertex
// contributes one directed segment (birth → death) to the face on its left
// and the reverse to the face on its right; the base edge closes the loop, and
// the segments are stitched by endpoint matching. Split-pinched faces come out
// as ordinary simple polygons this way.
//
// Robustness: the caller VALIDATES (faces present, no NaN, plan areas summing
// to the footprint) and falls back to a simpler roof on failure — the numbers
// here are single precision over metre-scale coordinates, and pathological OSM
// footprints (slivers, near-duplicate points) exist. Pre-simplify with
// simplifyRing() before calling.
//
// Header-only, extras, no dependencies beyond threepp math.

#ifndef THREEPP_EXTRAS_TERRAIN_STRAIGHTSKELETON_HPP
#define THREEPP_EXTRAS_TERRAIN_STRAIGHTSKELETON_HPP

#include "threepp/math/Vector2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace threepp::terrain {

    struct SkeletonFace {
        int edge = -1;            // index of the input ring edge (p[i] → p[i+1])
        std::vector<Vector2> poly;// simple polygon, input winding
    };

    // Why a skeleton was refused. Counted per footprint by the caller: the
    // 39.8% fallback rate Phase C shipped had NO per-cause breakdown, so there
    // was nothing to fix.
    enum SkeletonFail {
        SK_OK = 0,
        SK_DEGENERATE,// fewer than 3 vertices, or a zero-length edge
        SK_ITERCAP,   // events still queued when maxIters ran out
        SK_OPEN,      // the queue drained with the wavefront still open
        SK_UNSTITCHED,// a face's birth-to-death segments would not close
        SK_NONFINITE, // NaN / inf in a face
        SK_NOFACES,
        SK_AREA,  // face plan areas do not sum to the footprint
        SK_OFFSET,// peak wavefront distance non-positive or non-finite
        SK_COUNT
    };

    inline const char* skeletonFailName(int r) {
        switch (r) {
            case SK_OK: return "ok";
            case SK_DEGENERATE: return "degenerate";
            case SK_ITERCAP: return "iter-cap";
            case SK_OPEN: return "open-wavefront";
            case SK_UNSTITCHED: return "unstitched-face";
            case SK_NONFINITE: return "non-finite";
            case SK_NOFACES: return "no-faces";
            case SK_AREA: return "area-bound";
            case SK_OFFSET: return "bad-offset";
            default: return "?";
        }
    }

    struct StraightSkeletonResult {
        bool ok = false;
        int reason = SK_DEGENERATE;
        std::vector<SkeletonFace> faces;
        std::vector<Vector2> ring;// the ring the faces were built over (may be jittered)
        float maxOffset = 0.f;// peak wavefront distance = ridge height / pitch
        int iterations = 0;
    };

    namespace detail {

        inline float ssDot(const Vector2& a, const Vector2& b) { return a.x * b.x + a.y * b.y; }
        inline float ssCross(const Vector2& a, const Vector2& b) { return a.x * b.y - a.y * b.x; }
        inline float ssLen(const Vector2& a) { return std::sqrt(a.x * a.x + a.y * a.y); }

        // Signed shoelace area of an open ring (Vector2 = (x, z)).
        inline float ssRingArea(const std::vector<Vector2>& r) {
            float a = 0.f;
            for (size_t i = 0, n = r.size(); i < n; ++i) {
                const Vector2& p = r[i];
                const Vector2& q = r[(i + 1) % n];
                a += p.x * q.y - q.x * p.y;
            }
            return 0.5f * a;
        }

        struct SkVert {
            Vector2 o;  // position extrapolated back to t = 0
            Vector2 m;  // velocity along the bisector
            Vector2 pb; // birth position
            float tb = 0.f;
            int el = -1, er = -1;// incoming / outgoing edge index
            int prev = -1, next = -1;
            bool active = true;
            bool moves = true;// false when the two edges are anti-parallel
        };

        struct SkEvent {
            float t = 0.f;
            int type = 0;// 0 = edge, 1 = split
            int va = -1, vb = -1;
            int oe = -1;
            Vector2 p;
            bool operator<(const SkEvent& other) const { return t > other.t; }// min-heap
        };

    }// namespace detail

    // Drop vertices that sit within `collinearTol` of the line through their
    // neighbours and merge vertices closer than `minEdge`. OSM footprints carry
    // both in quantity and both wreck a straight-skeleton event queue.
    inline std::vector<Vector2> simplifyRing(const std::vector<Vector2>& in,
                                             float collinearTol = 0.25f, float minEdge = 0.5f) {
        std::vector<Vector2> r = in;
        for (int pass = 0; pass < 4 && r.size() > 3; ++pass) {
            bool changed = false;
            // short edges
            std::vector<Vector2> out;
            out.reserve(r.size());
            for (size_t i = 0, n = r.size(); i < n; ++i) {
                if (out.size() >= 3 && i + 1 == n) {
                    const Vector2 d(r[i].x - out.front().x, r[i].y - out.front().y);
                    if (detail::ssLen(d) < minEdge) {
                        changed = true;
                        continue;
                    }
                }
                if (!out.empty()) {
                    const Vector2 d(r[i].x - out.back().x, r[i].y - out.back().y);
                    if (detail::ssLen(d) < minEdge) {
                        changed = true;
                        continue;
                    }
                }
                out.push_back(r[i]);
            }
            if (out.size() >= 3) r = out;
            // collinear vertices
            if (r.size() > 3) {
                std::vector<Vector2> out2;
                out2.reserve(r.size());
                const size_t n = r.size();
                for (size_t i = 0; i < n; ++i) {
                    const Vector2& a = r[(i + n - 1) % n];
                    const Vector2& b = r[i];
                    const Vector2& c = r[(i + 1) % n];
                    const Vector2 ac(c.x - a.x, c.y - a.y);
                    const float l = detail::ssLen(ac);
                    if (l > 1e-4f && out2.size() + (n - i - 1) >= 3) {
                        const Vector2 ab(b.x - a.x, b.y - a.y);
                        if (std::abs(detail::ssCross(ac, ab)) / l < collinearTol) {
                            changed = true;
                            continue;
                        }
                    }
                    out2.push_back(b);
                }
                if (out2.size() >= 3) r = out2;
            }
            if (!changed) break;
        }
        if (r.size() < 3) return {};
        return r;
    }

    // Straight skeleton of a simple, hole-free ring wound so the interior lies
    // to the LEFT of every directed edge (positive shoelace in (x, z)).
    inline StraightSkeletonResult computeStraightSkeleton(const std::vector<Vector2>& ring,
                                                          int maxIters = 4000) {
        using namespace detail;
        StraightSkeletonResult res;
        const int n = static_cast<int>(ring.size());
        if (n < 3) {
            res.reason = SK_DEGENERATE;
            return res;
        }

        // Original edge frames: direction and INWARD unit normal.
        std::vector<Vector2> eDir(n), eNrm(n);
        for (int i = 0; i < n; ++i) {
            const Vector2& p = ring[i];
            const Vector2& q = ring[(i + 1) % n];
            Vector2 d(q.x - p.x, q.y - p.y);
            const float l = ssLen(d);
            if (l < 1e-5f) {
                res.reason = SK_DEGENERATE;
                return res;
            }
            d.x /= l;
            d.y /= l;
            eDir[i] = d;
            eNrm[i] = Vector2(-d.y, d.x);// left of the direction = interior
        }

        std::vector<SkVert> V;
        V.reserve(static_cast<size_t>(n) * 4);
        // birth → death segment per face, directed for the face on the left.
        struct Seg {
            Vector2 a, b;
            bool used = false;
        };
        std::vector<std::vector<Seg>> faceSegs(n);

        const auto velocity = [&](int el, int er, Vector2& m) {
            const Vector2& nL = eNrm[el];
            const Vector2& nR = eNrm[er];
            const float d = 1.f + ssDot(nL, nR);
            if (std::abs(d) < 1e-4f) return false;
            m = Vector2((nL.x + nR.x) / d, (nL.y + nR.y) / d);
            return std::isfinite(m.x) && std::isfinite(m.y) && ssLen(m) < 5e3f;
        };

        for (int i = 0; i < n; ++i) {
            SkVert v;
            v.el = (i + n - 1) % n;
            v.er = i;
            v.pb = ring[i];
            v.tb = 0.f;
            v.o = ring[i];
            v.moves = velocity(v.el, v.er, v.m);
            v.prev = (i + n - 1) % n;
            v.next = (i + 1) % n;
            V.push_back(v);
        }

        const auto posAt = [&](int i, float t) {
            return Vector2(V[i].o.x + t * V[i].m.x, V[i].o.y + t * V[i].m.y);
        };
        const auto consume = [&](int i, const Vector2& p) {
            if (!V[i].active) return;
            V[i].active = false;
            if (V[i].el >= 0) faceSegs[V[i].el].push_back({V[i].pb, p, false});
            if (V[i].er >= 0) faceSegs[V[i].er].push_back({p, V[i].pb, false});
            res.maxOffset = std::max(res.maxOffset,
                                     ssDot(Vector2(p.x - ring[V[i].er].x, p.y - ring[V[i].er].y),
                                           eNrm[V[i].er]));
        };

        std::priority_queue<SkEvent> queue;

        // Collapse time of the edge shared by v and v.next.
        const auto edgeEvent = [&](int a, float after, SkEvent& out) {
            const int b = V[a].next;
            if (b < 0 || b == a || !V[a].moves || !V[b].moves) return false;
            const Vector2 d(V[a].m.x - V[b].m.x, V[a].m.y - V[b].m.y);
            const float dd = ssDot(d, d);
            if (dd < 1e-10f) return false;
            const Vector2 r(V[b].o.x - V[a].o.x, V[b].o.y - V[a].o.y);
            const float t = ssDot(r, d) / dd;
            if (!std::isfinite(t)) return false;
            const Vector2 resid(r.x - t * d.x, r.y - t * d.y);
            if (ssLen(resid) > 1e-2f) return false;
            if (t < after + 1e-5f || t < std::max(V[a].tb, V[b].tb) - 1e-5f) return false;
            out.t = t;
            out.type = 0;
            out.va = a;
            out.vb = b;
            out.p = posAt(a, t);
            return true;
        };

        // Cheapest split of a reflex vertex against a non-adjacent edge.
        const auto splitEvent = [&](int a, float after, SkEvent& out) {
            if (!V[a].moves) return false;
            const int el = V[a].el, er = V[a].er;
            // reflex test in the ORIGINAL frames (the edges never rotate)
            if (ssCross(eDir[el], eDir[er]) >= -1e-6f) return false;
            bool found = false;
            float best = 0.f;
            Vector2 bestP;
            int bestE = -1;
            for (int e = 0; e < n; ++e) {
                if (e == el || e == er) continue;
                const float den = ssDot(V[a].m, eNrm[e]) - 1.f;
                if (std::abs(den) < 1e-5f) continue;
                const Vector2 rel(V[a].o.x - ring[e].x, V[a].o.y - ring[e].y);
                const float t = -ssDot(rel, eNrm[e]) / den;
                if (!std::isfinite(t) || t < after + 1e-5f || t < V[a].tb - 1e-5f) continue;
                if (found && t >= best) continue;
                const Vector2 B(V[a].o.x + t * V[a].m.x, V[a].o.y + t * V[a].m.y);
                // must land inside the moving edge's span (checked against the
                // ORIGINAL endpoint trajectories — a pre-filter; the authority
                // is the active-wavefront test at processing time)
                const Vector2 s0 = posAt(e, t);
                const Vector2 s1 = posAt((e + 1) % n, t);
                const Vector2 sd(s1.x - s0.x, s1.y - s0.y);
                const float sl = ssLen(sd);
                if (sl < 1e-4f) continue;
                const float u = ((B.x - s0.x) * sd.x + (B.y - s0.y) * sd.y) / (sl * sl);
                if (u < -0.05f || u > 1.05f) continue;
                found = true;
                best = t;
                bestP = B;
                bestE = e;
            }
            if (!found) return false;
            out.t = best;
            out.type = 1;
            out.va = a;
            out.vb = -1;
            out.oe = bestE;
            out.p = bestP;
            return true;
        };

        const auto pushNext = [&](int a, float after) {
            if (a < 0 || !V[a].active) return;
            SkEvent e1, e2;
            const bool h1 = edgeEvent(a, after, e1);
            const bool h2 = splitEvent(a, after, e2);
            if (h1 && (!h2 || e1.t <= e2.t)) queue.push(e1);
            else if (h2)
                queue.push(e2);
        };

        const auto loopSize = [&](int a) {
            int k = 0, i = a;
            while (k < 5) {
                ++k;
                i = V[i].next;
                if (i < 0 || i == a) break;
            }
            return k;
        };

        // Loops of ≤ 3 vertices: intersect the offset planes analytically.
        const auto resolveSmall = [&](int a) {
            if (!V[a].active) return false;
            const int sz = loopSize(a);
            if (sz > 3) return false;
            int ids[3] = {a, V[a].next, -1};
            if (sz == 3) ids[2] = V[V[a].next].next;
            // gather up to three distinct edge planes
            int es[3] = {-1, -1, -1};
            int ne = 0;
            for (int k = 0; k < sz; ++k) {
                for (const int e : {V[ids[k]].el, V[ids[k]].er}) {
                    bool dup = false;
                    for (int j = 0; j < ne; ++j)
                        if (es[j] == e) dup = true;
                    if (!dup && ne < 3) es[ne++] = e;
                }
            }
            Vector2 P(0.f, 0.f);
            bool solved = false;
            for (int i = 0; i < ne && !solved; ++i)
                for (int j = i + 1; j < ne && !solved; ++j)
                    for (int k = j + 1; k < ne && !solved; ++k) {
                        // dot(P - ring[e], n_e) = t for the three planes
                        const Vector2 &n0 = eNrm[es[i]], &n1 = eNrm[es[j]], &n2 = eNrm[es[k]];
                        const float c0 = ssDot(n0, ring[es[i]]), c1 = ssDot(n1, ring[es[j]]),
                                    c2 = ssDot(n2, ring[es[k]]);
                        const Vector2 A(n0.x - n1.x, n0.y - n1.y);
                        const Vector2 B(n0.x - n2.x, n0.y - n2.y);
                        const float det = ssCross(A, B);
                        if (std::abs(det) < 1e-6f) continue;
                        const float ra = c0 - c1, rb = c0 - c2;
                        P = Vector2((ra * B.y - rb * A.y) / det, (A.x * rb - B.x * ra) / det);
                        solved = std::isfinite(P.x) && std::isfinite(P.y);
                    }
            if (!solved) {
                // Fewer than three INDEPENDENT planes: a 2-loop, or the flat
                // residue of a collapsed spine (every remaining edge parallel).
                // Retiring the whole loop at one common point keeps each face's
                // chain consistent — any point does, so take the mean.
                float sx = 0.f, sy = 0.f;
                for (int k = 0; k < sz; ++k) {
                    sx += V[ids[k]].pb.x;
                    sy += V[ids[k]].pb.y;
                }
                P = Vector2(sx / static_cast<float>(sz), sy / static_cast<float>(sz));
                solved = std::isfinite(P.x) && std::isfinite(P.y);
            }
            if (!solved) return false;
            for (int k = 0; k < sz; ++k) consume(ids[k], P);
            return true;
        };

        const auto newVert = [&](int el, int er, const Vector2& p, float t) {
            SkVert v;
            v.el = el;
            v.er = er;
            v.pb = p;
            v.tb = t;
            v.moves = velocity(el, er, v.m);
            if (!v.moves) v.m = Vector2(0.f, 0.f);
            v.o = Vector2(p.x - t * v.m.x, p.y - t * v.m.y);
            const int idx = static_cast<int>(V.size());
            V.push_back(v);
            return idx;
        };

        // Simultaneous events leave DEGENERATE wavefronts behind: an L whose
        // arms are the same width retires its whole spine at one instant, so
        // the loop ends up with zero-length edges (adjacent vertices coincide)
        // or pinched at a point (two NON-adjacent vertices coincide). Neither
        // generates a further event, and the queue would simply stall with a
        // live loop of zero area. Repair both in place: merge the first,
        // split the loop in two at the second. Returns the loops still alive.
        const auto repairLoops = [&](std::vector<int> work, float t) {
            std::vector<int> alive;
            int guard = 0;
            while (!work.empty() && guard++ < 512) {
                const int start = work.back();
                work.pop_back();
                if (start < 0 || !V[start].active) continue;
                if (resolveSmall(start)) continue;
                std::vector<int> mem;
                mem.reserve(16);
                for (int i = start, k = 0; k < 256; ++k) {
                    mem.push_back(i);
                    i = V[i].next;
                    if (i < 0 || i == start) break;
                }
                int u = -1, w = -1;
                for (size_t i = 0; i < mem.size() && u < 0; ++i)
                    for (size_t j = i + 1; j < mem.size(); ++j) {
                        const Vector2 pa = posAt(mem[i], t), pb = posAt(mem[j], t);
                        const float dx = pa.x - pb.x, dy = pa.y - pb.y;
                        if (dx * dx + dy * dy > 1e-6f) continue;
                        u = mem[i];
                        w = mem[j];
                        break;
                    }
                if (u < 0) {
                    // A loop that already encloses no area is a collapsed spine
                    // with nothing left to sweep; retire it at one common point
                    // so every face chain stays connected.
                    float a2 = 0.f, sx = 0.f, sy = 0.f;
                    for (size_t i = 0; i < mem.size(); ++i) {
                        const Vector2 p = posAt(mem[i], t);
                        const Vector2 q = posAt(mem[(i + 1) % mem.size()], t);
                        a2 += p.x * q.y - q.x * p.y;
                        sx += p.x;
                        sy += p.y;
                    }
                    if (std::abs(0.5f * a2) < 1e-3f && !mem.empty()) {
                        const float inv = 1.f / static_cast<float>(mem.size());
                        const Vector2 P(sx * inv, sy * inv);
                        for (const int mv : mem) consume(mv, P);
                        continue;
                    }
                    alive.push_back(start);
                    continue;
                }
                const Vector2 P = posAt(u, t);
                if (V[w].next == u) std::swap(u, w);
                if (V[u].next == w) {// zero-length edge: merge the pair
                    const int pu = V[u].prev, nw = V[w].next;
                    if (nw == u || pu == w) {// the whole loop is the pair
                        resolveSmall(u);
                        continue;
                    }
                    consume(u, P);
                    consume(w, P);
                    const int idx = newVert(V[u].el, V[w].er, P, t);
                    V[idx].prev = pu;
                    V[idx].next = nw;
                    V[pu].next = idx;
                    V[nw].prev = idx;
                    work.push_back(idx);
                } else {// pinch: cut the loop in two at P
                    const int pu = V[u].prev, nu = V[u].next;
                    const int pw = V[w].prev, nw = V[w].next;
                    consume(u, P);
                    consume(w, P);
                    const int x1 = newVert(V[u].el, V[w].er, P, t);
                    const int x2 = newVert(V[w].el, V[u].er, P, t);
                    V[x1].prev = pu;
                    V[x1].next = nw;
                    V[pu].next = x1;
                    V[nw].prev = x1;
                    V[x2].prev = pw;
                    V[x2].next = nu;
                    V[pw].next = x2;
                    V[nu].prev = x2;
                    work.push_back(x1);
                    work.push_back(x2);
                }
            }
            return alive;
        };

        for (int i = 0; i < n; ++i) pushNext(i, -1.f);

        // STALE pops must not spend the iteration budget. repairLoops re-pushes
        // a whole wavefront loop after every event, so on a 40-vertex ring the
        // queue carries ~n dead events per live one; counting those against
        // maxIters is what made 1005 of 3610 aalesund footprints report
        // "iteration cap" and fall back to a rectangle gable.
        // A retry that retires nothing must push the wavefront on by a real
        // millimetre. With the old `after = ev.t` the queue handed the SAME
        // rejected event back at t + 1e-5 and ground through the whole
        // iteration budget without retiring a vertex — that livelock, not the
        // budget's size, is what "iter-cap" was counting.
        constexpr float kSkRetryStep = 1e-3f;
        int iter = 0;
        long long pops = 0;
        const long long popCap = 400LL * n + 50000LL;
        while (!queue.empty() && iter < maxIters && pops < popCap) {
            ++pops;
            const SkEvent ev = queue.top();
            queue.pop();
            if (ev.va < 0 || !V[ev.va].active) continue;
            ++iter;

            if (ev.type == 0) {
                const int a = ev.va;
                if (V[a].next != ev.vb || !V[ev.vb].active) {
                    pushNext(a, ev.t + kSkRetryStep);
                    continue;
                }
                const int b = ev.vb;
                if (resolveSmall(a)) continue;
                const int pa = V[a].prev, nb = V[b].next;
                consume(a, ev.p);
                consume(b, ev.p);
                const int idx = newVert(V[a].el, V[b].er, ev.p, ev.t);
                V[idx].prev = pa;
                V[idx].next = nb;
                V[pa].next = idx;
                V[nb].prev = idx;
                for (const int s : repairLoops({idx}, ev.t))
                    for (int i = s, k = 0; k < 256; ++k) {
                        pushNext(i, ev.t - 1e-4f);
                        i = V[i].next;
                        if (i < 0 || i == s) break;
                    }
            } else {
                const int a = ev.va;
                const int e = ev.oe;
                // locate the ACTIVE wavefront edge e and check the hit lands on it
                int x = -1, y = -1;
                for (int i = 0; i < static_cast<int>(V.size()); ++i) {
                    if (!V[i].active || V[i].el != e || i == a) continue;
                    const int yi = V[i].prev;
                    if (yi < 0 || !V[yi].active || yi == a) continue;
                    const Vector2 px = posAt(i, ev.t), py = posAt(yi, ev.t);
                    const Vector2 sd(px.x - py.x, px.y - py.y);
                    const float sl = ssLen(sd);
                    if (sl < 1e-4f) continue;
                    const float u = ((ev.p.x - py.x) * sd.x + (ev.p.y - py.y) * sd.y) / (sl * sl);
                    if (u < -0.02f || u > 1.02f) continue;
                    x = i;
                    y = yi;
                    break;
                }
                if (x < 0) {
                    pushNext(a, ev.t + kSkRetryStep);
                    continue;
                }
                const int pv = V[a].prev, nv = V[a].next;
                if (pv < 0 || nv < 0 || pv == x || nv == y) {
                    pushNext(a, ev.t + kSkRetryStep);
                    continue;
                }
                consume(a, ev.p);
                const int v1 = newVert(V[a].el, e, ev.p, ev.t);
                const int v2 = newVert(e, V[a].er, ev.p, ev.t);
                V[v1].prev = pv;
                V[v1].next = x;
                V[v2].prev = y;
                V[v2].next = nv;
                V[pv].next = v1;
                V[x].prev = v1;
                V[y].next = v2;
                V[nv].prev = v2;
                for (const int s : repairLoops({v1, v2}, ev.t))
                    for (int i = s, k = 0; k < 256; ++k) {
                        pushNext(i, ev.t - 1e-4f);
                        i = V[i].next;
                        if (i < 0 || i == s) break;
                    }
            }
        }
        res.iterations = iter;
        const bool capped = (iter >= maxIters || pops >= popCap) && !queue.empty();
        for (const auto& v : V)
            if (v.active) {// wavefront never closed
                res.reason = capped ? SK_ITERCAP : SK_OPEN;
                return res;
            }

        // ── stitch each face from its directed segments ─────────────────────
        res.faces.reserve(n);
        for (int e = 0; e < n; ++e) {
            const Vector2 p1 = ring[e], p2 = ring[(e + 1) % n];
            std::vector<Vector2> poly{p1, p2};
            Vector2 cur = p2;
            auto& segs = faceSegs[e];
            const size_t guard = segs.size() + 2;
            bool closed = false;
            for (size_t step = 0; step < guard; ++step) {
                const float dx = cur.x - p1.x, dy = cur.y - p1.y;
                if (dx * dx + dy * dy < 1e-8f && poly.size() >= 3) {
                    closed = true;
                    break;
                }
                int pick = -1;
                float bestD = 1e-4f;
                for (size_t s = 0; s < segs.size(); ++s) {
                    if (segs[s].used) continue;
                    const float ex = segs[s].a.x - cur.x, ey = segs[s].a.y - cur.y;
                    const float d = ex * ex + ey * ey;
                    if (d < bestD) {
                        bestD = d;
                        pick = static_cast<int>(s);
                    }
                }
                if (pick < 0) break;
                segs[pick].used = true;
                cur = segs[pick].b;
                poly.push_back(cur);
            }
            if (!closed) {
                res.reason = SK_UNSTITCHED;
                return res;
            }
            // drop the duplicate closing point and any near-duplicates
            std::vector<Vector2> clean;
            clean.reserve(poly.size());
            for (const auto& p : poly) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                    res.reason = SK_NONFINITE;
                    return res;
                }
                if (!clean.empty()) {
                    const float dx = p.x - clean.back().x, dy = p.y - clean.back().y;
                    if (dx * dx + dy * dy < 1e-6f) continue;
                }
                clean.push_back(p);
            }
            if (clean.size() >= 2) {
                const float dx = clean.front().x - clean.back().x;
                const float dy = clean.front().y - clean.back().y;
                if (dx * dx + dy * dy < 1e-6f) clean.pop_back();
            }
            if (clean.size() < 3) continue;// degenerate sliver face — skipped
            res.faces.push_back({e, std::move(clean)});
        }
        if (res.faces.empty()) {
            res.reason = SK_NOFACES;
            return res;
        }

        // area check: the faces must tile the footprint
        float sum = 0.f;
        for (const auto& f : res.faces) sum += std::abs(ssRingArea(f.poly));
        const float target = std::abs(ssRingArea(ring));
        // Relative AND absolute slack: a 12 m2 sliver garage carries the same
        // half-square-metre of single-precision stitching error as a 400 m2
        // block, and a pure 6% ratio test threw the garage away.
        const float slack = 0.06f * target + 0.75f;
        if (target < 1e-3f || sum < target - slack || sum > target + slack) {
            res.reason = SK_AREA;
            return res;
        }
        if (!(res.maxOffset > 0.f) || !std::isfinite(res.maxOffset)) {
            res.reason = SK_OFFSET;
            return res;
        }

        res.reason = SK_OK;
        res.ok = true;
        res.ring = ring;
        return res;
    }

    // Exact-symmetry footprints (a T whose bar and stem are the same width)
    // retire three or more wavefront branches at ONE point; the repair above
    // only knows how to unpick a pinch two branches at a time. A sub-millimetre
    // deterministic jitter breaks the tie without moving a roof visibly, so
    // retry with one before giving up.
    inline StraightSkeletonResult computeRoofSkeleton(const std::vector<Vector2>& ring,
                                                      int maxIters = 4000) {
        StraightSkeletonResult r = computeStraightSkeleton(ring, maxIters);
        if (r.ok) return r;
        std::vector<Vector2> jittered = ring;
        for (int attempt = 1; attempt <= 2 && !r.ok; ++attempt) {
            std::uint32_t h = 0x9e3779b9u * static_cast<std::uint32_t>(attempt);
            for (auto& p : jittered) {
                h = h * 1664525u + 1013904223u;
                const float jx = (static_cast<float>((h >> 8) & 0xffff) / 65535.f - 0.5f) * 3e-3f;
                h = h * 1664525u + 1013904223u;
                const float jy = (static_cast<float>((h >> 8) & 0xffff) / 65535.f - 0.5f) * 3e-3f;
                p = Vector2(p.x + jx, p.y + jy);
            }
            r = computeStraightSkeleton(jittered, maxIters);
        }
        return r;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_STRAIGHTSKELETON_HPP
