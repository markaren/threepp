#include "threepp/renderers/vulkan/EdgeRefine.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace threepp::vulkan {

    namespace {

        constexpr uint32_t kNone = 0xFFFFFFFFu;

        // Edge → midpoint, open addressing (an unordered_map allocated a node
        // per edge: most of the refinement's time on a few hundred thousand).
        class EdgeTable {
        public:
            void reset(std::size_t edges) {
                std::size_t cap = 16;
                while (cap < edges * 2) cap <<= 1;
                keys_.assign(cap, kEmpty);
                vals_.resize(cap);
                mask_ = cap - 1;
            }
            // The slot for `key`, and whether it was just made.
            std::pair<uint32_t*, bool> find(uint64_t key) {
                std::size_t i = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ull) >> 20) & mask_;
                for (;; i = (i + 1) & mask_) {
                    if (keys_[i] == key) return {&vals_[i], false};
                    if (keys_[i] == kEmpty) {
                        keys_[i] = key;
                        return {&vals_[i], true};
                    }
                }
            }

        private:
            static constexpr uint64_t kEmpty = ~0ull;// no edge has two equal ends of 2^32 - 1
            std::vector<uint64_t> keys_;
            std::vector<uint32_t> vals_;
            std::size_t mask_ = 0;
        };

        float dist2(const std::vector<float>& p, uint32_t a, uint32_t b) {
            const float dx = p[3 * a] - p[3 * b], dy = p[3 * a + 1] - p[3 * b + 1], dz = p[3 * a + 2] - p[3 * b + 2];
            return dx * dx + dy * dy + dz * dz;
        }

    }// namespace

    RefineStats refineByEdges(std::vector<uint32_t>& indices, std::vector<RefineAttribute>& attrs,
                              const std::function<bool(const float* pa, const float* pb)>& split,
                              const RefineLimits& limits) {
        RefineStats st;
        st.triangles = indices.size() / 3;
        if (attrs.empty() || attrs[0].itemSize != 3) return st;
        std::vector<float>& pos = attrs[0].data;

        EdgeTable mids;// edge (lower index high) → its midpoint, or kNone
        std::vector<std::pair<uint32_t, uint32_t>> ends;// each new vertex's edge, in vertex order
        std::vector<uint32_t> triMids;// per triangle, the midpoints of edges ab, bc, ca
        std::vector<uint32_t> next;
        // A triangle none of whose edges split is final: an edge's verdict is a
        // function of its ends, so it never changes. Only the triangles the
        // last pass made are looked at again; the final ones collect in `done`.
        std::vector<uint32_t> done;
        std::vector<uint32_t>& active = indices;
        for (int pass = 0; pass < limits.maxPasses; ++pass) {
            const auto vertexCount = static_cast<uint32_t>(pos.size() / 3);
            mids.reset(active.size());// at most 3 edges a triangle, each shared or not
            ends.clear();
            triMids.resize(active.size());
            const auto midOf = [&](uint32_t a, uint32_t b) {
                const uint64_t key = a < b ? (uint64_t(a) << 32 | b) : (uint64_t(b) << 32 | a);
                auto [mid, fresh] = mids.find(key);
                if (fresh) {
                    *mid = kNone;
                    if (split(&pos[3 * a], &pos[3 * b])) {
                        *mid = vertexCount + static_cast<uint32_t>(ends.size());
                        ends.emplace_back(a, b);
                    }
                }
                return *mid;
            };
            // Every edge's verdict first, so the pass can be refused whole when
            // it would pass the budget.
            std::size_t triOut = 0;
            for (std::size_t t = 0; t < active.size(); t += 3) {
                const uint32_t a = active[t], b = active[t + 1], c = active[t + 2];
                triMids[t]     = midOf(a, b);
                triMids[t + 1] = midOf(b, c);
                triMids[t + 2] = midOf(c, a);
                const int n = (triMids[t] != kNone) + (triMids[t + 1] != kNone) + (triMids[t + 2] != kNone);
                triOut += static_cast<std::size_t>(n == 0 ? 1 : n + 1);
            }
            if (ends.empty()) break;
            if (done.size() / 3 + triOut > limits.maxTriangles) {
                st.hitBudget = true;
                break;
            }

            for (auto& at : attrs) {
                const std::size_t k = static_cast<std::size_t>(at.itemSize);
                at.data.resize((static_cast<std::size_t>(vertexCount) + ends.size()) * k);
                for (std::size_t v = 0; v < ends.size(); ++v) {
                    const float* pa = &at.data[ends[v].first * k];
                    const float* pb = &at.data[ends[v].second * k];
                    float* out = &at.data[(vertexCount + v) * k];
                    float len2 = 0.f;
                    for (std::size_t i = 0; i < k; ++i) {
                        out[i] = 0.5f * (pa[i] + pb[i]);
                        len2 += out[i] * out[i];
                    }
                    if (at.unit && len2 > 0.f) {
                        const float inv = 1.f / std::sqrt(len2);
                        for (std::size_t i = 0; i < k; ++i) out[i] *= inv;
                    }
                }
            }

            // Re-triangulate, each triangle rotated so its split edges lead
            // (the rotation keeps the winding): m[i] sits on edge v[i]-v[i+1].
            next.clear();
            next.reserve(triOut * 3);
            const auto emit = [&](uint32_t a, uint32_t b, uint32_t c) {
                next.push_back(a);
                next.push_back(b);
                next.push_back(c);
            };
            for (std::size_t t = 0; t < active.size(); t += 3) {
                const uint32_t v0[3] = {active[t], active[t + 1], active[t + 2]};
                const uint32_t m0[3] = {triMids[t], triMids[t + 1], triMids[t + 2]};
                const int n = (m0[0] != kNone) + (m0[1] != kNone) + (m0[2] != kNone);
                if (n == 0) {
                    done.insert(done.end(), v0, v0 + 3);
                    continue;
                }
                if (n == 3) {
                    emit(v0[0], m0[0], m0[2]);
                    emit(m0[0], v0[1], m0[1]);
                    emit(m0[2], m0[1], v0[2]);
                    emit(m0[0], m0[1], m0[2]);
                    continue;
                }
                // One split: rotate it to edge 0. Two: rotate the unsplit one to edge 2.
                int r = 0;
                if (n == 1) r = m0[0] != kNone ? 0 : (m0[1] != kNone ? 1 : 2);
                else r = m0[2] == kNone ? 0 : (m0[0] == kNone ? 1 : 2);
                const uint32_t A = v0[r], B = v0[(r + 1) % 3], C = v0[(r + 2) % 3];
                const uint32_t M0 = m0[r], M1 = m0[(r + 1) % 3];
                if (n == 1) {
                    emit(A, M0, C);
                    emit(M0, B, C);
                } else {
                    emit(M0, B, M1);
                    // The quad A, M0, M1, C, cut along its shorter diagonal.
                    if (dist2(pos, A, M1) <= dist2(pos, M0, C)) {
                        emit(A, M0, M1);
                        emit(A, M1, C);
                    } else {
                        emit(A, M0, C);
                        emit(M0, M1, C);
                    }
                }
            }
            active.swap(next);
            ++st.passes;
        }
        // The final triangles first, then whatever was still active (a stop at
        // maxPasses or the budget).
        done.insert(done.end(), active.begin(), active.end());
        indices.swap(done);
        st.triangles = indices.size() / 3;
        return st;
    }

    bool ScreenEdgeTest::inView(const float* c, float r) const {
        const auto& m = view;// the camera looks down -z
        float v[3];
        for (int i = 0; i < 3; ++i) v[i] = m[i] * c[0] + m[4 + i] * c[1] + m[8 + i] * c[2] + m[12 + i];
        const float kx = halfX * (1.f + frustumMargin), ky = halfY * (1.f + frustumMargin);
        if (orthoPixel > 0.f) return std::abs(v[0]) <= kx + r && std::abs(v[1]) <= ky + r;
        if (v[2] > r) return false;// behind the eye
        // The side planes through the eye, x = kx * depth (depth = -z): the
        // signed distance of the centre outside each, against the radius.
        const float sx = 1.f / std::sqrt(1.f + kx * kx), sy = 1.f / std::sqrt(1.f + ky * ky);
        return (v[0] + kx * v[2]) * sx <= r && (-v[0] + kx * v[2]) * sx <= r &&
               (v[1] + ky * v[2]) * sy <= r && (-v[1] + ky * v[2]) * sy <= r;
    }

    bool ScreenEdgeTest::operator()(const float* pa, const float* pb) const {
        const auto& m = world;
        float wa[3], wb[3];
        for (int r = 0; r < 3; ++r) {
            wa[r] = m[r] * pa[0] + m[4 + r] * pa[1] + m[8 + r] * pa[2] + m[12 + r];
            wb[r] = m[r] * pb[0] + m[4 + r] * pb[1] + m[8 + r] * pb[2] + m[12 + r];
        }
        float len2 = 0.f, dist2 = 0.f, mid[3];
        for (int r = 0; r < 3; ++r) {
            const float e = wa[r] - wb[r];
            mid[r] = 0.5f * (wa[r] + wb[r]);
            const float c = mid[r] - eye[r];
            len2 += e * e;
            dist2 += c * c;
        }
        const float pixel = orthoPixel > 0.f ? orthoPixel : pixelAngle * std::sqrt(dist2);
        if (reach < 0.5f * pixel) return false;
        float target = std::max(minEdge, pixelEdge * pixel);
        if (len2 <= target * target) return false;// short enough in view already: the limit only lengthens it
        if (frustum && !inView(mid, 0.5f * std::sqrt(len2) + reach)) target *= outsideFactor;
        return len2 > target * target;
    }

}// namespace threepp::vulkan
