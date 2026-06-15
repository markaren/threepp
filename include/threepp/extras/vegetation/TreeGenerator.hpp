// Procedural tree mesh generator (CPU) — space colonisation algorithm.
//
// Pipeline:
//   1. buildSkeleton()   — grow a branching skeleton inside a crown envelope
//                          by iteratively steering toward scattered attraction
//                          points (Runions et al. 2007).  Deterministic for a
//                          fixed seed.
//   2. makeTrunkGeometry() — skin the skeleton with tapered tube segments and
//                          return a single indexed BufferGeometry (position,
//                          normal, uv).
//   3. makeLeafGeometry()  — emit oriented quads at terminal branch tips,
//                          suitable for alpha-tested leaf materials.
//
// Header-only, dependency-free beyond threepp core.
//
// Coordinate convention: +Y is up.  The trunk grows from the origin upward;
// the crown envelope is centred above the trunk at (0, trunkHeight + crownHeight/2, 0).

#ifndef THREEPP_EXTRAS_VEGETATION_TREEGENERATOR_HPP
#define THREEPP_EXTRAS_VEGETATION_TREEGENERATOR_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace threepp::vegetation {

    // ── Crown envelope shapes ────────────────────────────────────────────
    enum class CrownShape {
        Sphere = 0,
        Ellipsoid = 1,
        Cone = 2,
        Hemisphere = 3,
        Cylinder = 4,
    };

    // ── Leaf rendering style ─────────────────────────────────────────────
    enum class LeafStyle {
        Quad = 0,     // single flat quad per leaf (cheap, needs a cutout texture to look good)
        Cluster = 1,  // several flat quads jittered around the tip
        CrossQuad = 2,// crossed quad pair (volumetric from any angle)
        Blob = 3,     // low-poly foliage puffs (spheres) — reads as a canopy untextured
    };

    // ── Configurator-facing knobs ────────────────────────────────────────
    struct TreeParams {
        unsigned int seed = 1337;

        // ── Trunk ────────────────────────────────────────────────────────
        float trunkHeight = 4.0f;
        float trunkRadius = 0.12f;

        // ── Crown envelope ───────────────────────────────────────────────
        CrownShape crownShape = CrownShape::Ellipsoid;
        float crownRadiusX = 3.0f;
        float crownRadiusZ = 3.0f;
        float crownHeight = 5.0f;

        // ── Space colonisation ───────────────────────────────────────────
        int attractorCount = 800;
        float influenceDistance = 4.0f;
        float killDistance = 0.8f;
        float segmentLength = 0.4f;
        int maxIterations = 200;
        float randomness = 0.08f;
        float tropism = -0.02f;

        // ── Branch geometry ──────────────────────────────────────────────
        float radiusExponent = 2.2f;
        float minBranchRadius = 0.006f;
        int radialSegments = 6;

        // ── Leaves ───────────────────────────────────────────────────────
        LeafStyle leafStyle = LeafStyle::CrossQuad;
        float leafSize = 0.7f;// quad half-size (card), or puff radius for Blob
        float leafDensity = 0.9f;
        int leavesPerCluster = 5;
        float leafSpread = 0.5f;

        // ── Albedo hints (sRGB) ──────────────────────────────────────────
        std::array<float, 3> barkColor = {0.35f, 0.25f, 0.18f};
        std::array<float, 3> leafColor = {0.25f, 0.45f, 0.15f};
    };

    // ── Internal skeleton node ───────────────────────────────────────────
    namespace detail {
        struct TreeNode {
            Vector3 position;
            int parent = -1;
            std::vector<int> children;
            float radius = 0.f;
            int depth = 0;
            bool terminal = false;
        };
    }// namespace detail

    // ── Generator ────────────────────────────────────────────────────────
    class TreeGenerator {

    public:
        explicit TreeGenerator(unsigned int seed = 1337) { reseed(seed); }

        void reseed(unsigned int seed) {
            seed_ = seed ? seed : 1u;
        }

        [[nodiscard]] unsigned int seed() const { return seed_; }
        [[nodiscard]] int nodeCount() const { return static_cast<int>(nodes_.size()); }

        // ── Step 1: build the branching skeleton ─────────────────────────
        //
        // Standard space colonisation (Runions et al. 2007): every node in
        // the tree competes for nearby attractors each iteration, not just
        // the active tips.  This is what produces branching — when several
        // nodes claim disjoint attractor subsets they each grow a child,
        // splitting the tree.
        void buildSkeleton(const TreeParams& tp) {
            std::mt19937 rng(tp.seed ? tp.seed : 1u);
            nodes_.clear();

            // Grow trunk as a straight chain of nodes from origin to crown base.
            const int trunkSegs = std::max(1, static_cast<int>(std::round(tp.trunkHeight / tp.segmentLength)));
            const float trunkStep = tp.trunkHeight / static_cast<float>(trunkSegs);
            for (int i = 0; i <= trunkSegs; ++i) {
                detail::TreeNode n;
                n.position = {0.f, static_cast<float>(i) * trunkStep, 0.f};
                n.parent = i > 0 ? i - 1 : -1;
                n.depth = i;
                if (i > 0) nodes_[static_cast<size_t>(i - 1)].children.push_back(i);
                nodes_.push_back(n);
            }

            // Scatter attraction points inside the crown envelope.
            std::vector<Vector3> attractors;
            attractors.reserve(static_cast<size_t>(tp.attractorCount));
            const Vector3 crownCentre{0.f, tp.trunkHeight + tp.crownHeight * 0.5f, 0.f};
            scatterAttractors(rng, tp, crownCentre, attractors);

            // Colonisation loop — ALL nodes participate every iteration.
            const float killDist2 = tp.killDistance * tp.killDistance;
            const float infDist2 = tp.influenceDistance * tp.influenceDistance;
            std::uniform_real_distribution<float> jitter(-1.f, 1.f);

            for (int iter = 0; iter < tp.maxIterations && !attractors.empty(); ++iter) {
                const int nodeCount = static_cast<int>(nodes_.size());

                struct GrowInfo {
                    Vector3 dir{0.f, 0.f, 0.f};
                    int count = 0;
                };
                std::vector<GrowInfo> grow(static_cast<size_t>(nodeCount));
                std::vector<bool> killMask(attractors.size(), false);

                // For each attractor find the single closest node (any node).
                for (size_t ai = 0; ai < attractors.size(); ++ai) {
                    const auto& ap = attractors[ai];
                    float bestDist2 = infDist2;
                    int bestNode = -1;
                    for (int ni = 0; ni < nodeCount; ++ni) {
                        const auto& np = nodes_[static_cast<size_t>(ni)].position;
                        const float dx = ap.x - np.x, dy = ap.y - np.y, dz = ap.z - np.z;
                        const float d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < bestDist2) {
                            bestDist2 = d2;
                            bestNode = ni;
                        }
                    }
                    if (bestNode >= 0) {
                        Vector3 toward;
                        toward.subVectors(ap, nodes_[static_cast<size_t>(bestNode)].position).normalize();
                        grow[static_cast<size_t>(bestNode)].dir.add(toward);
                        grow[static_cast<size_t>(bestNode)].count++;
                        if (bestDist2 < killDist2) killMask[ai] = true;
                    }
                }

                // Every node that attracted at least one point grows a child.
                for (int ni = 0; ni < nodeCount; ++ni) {
                    auto& gi = grow[static_cast<size_t>(ni)];
                    if (gi.count == 0) continue;

                    Vector3 dir;
                    dir.copy(gi.dir).divideScalar(static_cast<float>(gi.count)).normalize();
                    dir.x += jitter(rng) * tp.randomness;
                    dir.y += jitter(rng) * tp.randomness;
                    dir.z += jitter(rng) * tp.randomness;
                    dir.y += tp.tropism;
                    dir.normalize();

                    detail::TreeNode child;
                    child.position.copy(nodes_[static_cast<size_t>(ni)].position)
                            .addScaledVector(dir, tp.segmentLength);
                    child.parent = ni;
                    child.depth = nodes_[static_cast<size_t>(ni)].depth + 1;

                    const int childIdx = static_cast<int>(nodes_.size());
                    nodes_[static_cast<size_t>(ni)].children.push_back(childIdx);
                    nodes_.push_back(child);
                }

                // Kill attractors that are now within kill distance of any
                // node (check newly added nodes as well).
                for (size_t ai = 0; ai < attractors.size(); ++ai) {
                    if (killMask[ai]) continue;
                    const auto& ap = attractors[ai];
                    for (size_t ni = static_cast<size_t>(nodeCount); ni < nodes_.size(); ++ni) {
                        const auto& np = nodes_[ni].position;
                        const float dx = ap.x - np.x, dy = ap.y - np.y, dz = ap.z - np.z;
                        if (dx * dx + dy * dy + dz * dz < killDist2) {
                            killMask[ai] = true;
                            break;
                        }
                    }
                }

                // Remove killed attractors.
                size_t write = 0;
                for (size_t i = 0; i < attractors.size(); ++i) {
                    if (!killMask[i]) attractors[write++] = attractors[i];
                }
                attractors.resize(write);
            }

            // Mark terminals and compute radii.
            for (auto& n : nodes_) {
                n.terminal = n.children.empty();
                if (n.terminal) n.radius = tp.minBranchRadius;
            }
            computeRadii(tp);
        }

        // ── Step 2: skin the skeleton into trunk/branch geometry ─────────
        //
        // Traces branch chains (maximal sequences without splits) and
        // generates a continuous tube per chain with shared ring vertices
        // and parallel-transport frames — no seam artefacts.
        [[nodiscard]] std::shared_ptr<BufferGeometry> makeTrunkGeometry(const TreeParams& tp) const {
            if (nodes_.size() < 2) return std::make_shared<BufferGeometry>();

            const int R = std::max(3, tp.radialSegments);

            // Trace branch chains.  A chain starts at the root or at a node
            // whose parent has >1 child (i.e. a fork).  It continues as
            // long as each successive node has exactly one child.
            std::vector<std::vector<int>> chains;
            {
                std::vector<bool> visited(nodes_.size(), false);
                for (size_t ni = 0; ni < nodes_.size(); ++ni) {
                    const auto& nd = nodes_[ni];
                    const bool isChainStart =
                            (nd.parent < 0) ||
                            (nd.parent >= 0 && nodes_[static_cast<size_t>(nd.parent)].children.size() > 1);
                    if (!isChainStart || visited[ni]) continue;

                    // Include the branch point as the first ring so the
                    // tube connects visually to the parent branch.
                    std::vector<int> chain;
                    if (nd.parent >= 0) chain.push_back(nd.parent);

                    int cur = static_cast<int>(ni);
                    while (cur >= 0 && !visited[static_cast<size_t>(cur)]) {
                        visited[static_cast<size_t>(cur)] = true;
                        chain.push_back(cur);
                        if (nodes_[static_cast<size_t>(cur)].children.size() == 1)
                            cur = nodes_[static_cast<size_t>(cur)].children[0];
                        else
                            break;
                    }
                    if (chain.size() >= 2) chains.push_back(std::move(chain));
                }
            }

            std::vector<float> positions, normals, uvs;
            std::vector<unsigned int> indices;
            unsigned int baseVert = 0;

            for (const auto& chain : chains) {
                const int len = static_cast<int>(chain.size());

                // Tangent at each chain node.
                std::vector<Vector3> tangent(static_cast<size_t>(len));
                for (int i = 0; i < len - 1; ++i) {
                    tangent[static_cast<size_t>(i)].subVectors(
                            nodes_[static_cast<size_t>(chain[static_cast<size_t>(i + 1)])].position,
                            nodes_[static_cast<size_t>(chain[static_cast<size_t>(i)])].position);
                    if (tangent[static_cast<size_t>(i)].lengthSq() > 1e-12f)
                        tangent[static_cast<size_t>(i)].normalize();
                    else
                        tangent[static_cast<size_t>(i)].set(0.f, 1.f, 0.f);
                }
                tangent[static_cast<size_t>(len - 1)] = tangent[static_cast<size_t>(len - 2)];

                // Cumulative arc length for V coordinate.
                std::vector<float> arcLen(static_cast<size_t>(len), 0.f);
                for (int i = 1; i < len; ++i) {
                    arcLen[static_cast<size_t>(i)] = arcLen[static_cast<size_t>(i - 1)] +
                            nodes_[static_cast<size_t>(chain[static_cast<size_t>(i)])].position.distanceTo(
                                    nodes_[static_cast<size_t>(chain[static_cast<size_t>(i - 1)])].position);
                }
                // Parallel-transport frame along the chain.
                Vector3 P, Q;
                buildFrame(tangent[0], P, Q);

                for (int i = 0; i < len; ++i) {
                    // Rotate frame to follow curvature.
                    if (i > 0) {
                        Vector3 axis;
                        axis.crossVectors(tangent[static_cast<size_t>(i - 1)], tangent[static_cast<size_t>(i)]);
                        const float sinA = axis.length();
                        if (sinA > 1e-6f) {
                            axis.divideScalar(sinA);
                            const float cosA = tangent[static_cast<size_t>(i - 1)].dot(tangent[static_cast<size_t>(i)]);
                            const float angle = std::atan2(sinA, cosA);
                            P.applyAxisAngle(axis, angle);
                            Q.applyAxisAngle(axis, angle);
                        }
                    }

                    const auto& nd = nodes_[static_cast<size_t>(chain[static_cast<size_t>(i)])];
                    const float r = nd.radius;
                    // Absolute arc length (world units) so bark tiles at a
                    // consistent scale on trunk and twigs alike; material
                    // `repeat` controls the final tile density.
                    const float v = arcLen[static_cast<size_t>(i)];

                    for (int j = 0; j <= R; ++j) {
                        const float a = static_cast<float>(j) / static_cast<float>(R) * 6.28318530718f;
                        const float ca = std::cos(a), sa = std::sin(a);
                        const float nx = P.x * ca + Q.x * sa;
                        const float ny = P.y * ca + Q.y * sa;
                        const float nz = P.z * ca + Q.z * sa;
                        positions.push_back(nd.position.x + nx * r);
                        positions.push_back(nd.position.y + ny * r);
                        positions.push_back(nd.position.z + nz * r);
                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                        uvs.push_back(static_cast<float>(j) / static_cast<float>(R));
                        uvs.push_back(v);
                    }
                }

                // Connect adjacent rings (CCW winding from outside).
                const auto ringStride = static_cast<unsigned int>(R + 1);
                for (int i = 0; i < len - 1; ++i) {
                    const unsigned int row0 = baseVert + static_cast<unsigned int>(i) * ringStride;
                    const unsigned int row1 = row0 + ringStride;
                    for (int j = 0; j < R; ++j) {
                        const unsigned int a = row0 + static_cast<unsigned int>(j);
                        const unsigned int b = row0 + static_cast<unsigned int>(j + 1);
                        const unsigned int c = row1 + static_cast<unsigned int>(j);
                        const unsigned int d = row1 + static_cast<unsigned int>(j + 1);
                        indices.push_back(a);
                        indices.push_back(b);
                        indices.push_back(c);
                        indices.push_back(b);
                        indices.push_back(d);
                        indices.push_back(c);
                    }
                }
                baseVert += static_cast<unsigned int>(len) * ringStride;
            }

            auto geo = std::make_shared<BufferGeometry>();
            geo->setIndex(indices);
            geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            return geo;
        }

        // ── Step 3: leaf quads at terminal branch tips ───────────────────
        [[nodiscard]] std::shared_ptr<BufferGeometry> makeLeafGeometry(const TreeParams& tp) const {
            if (nodes_.empty()) return std::make_shared<BufferGeometry>();

            std::mt19937 rng(tp.seed ^ 0xBEEF);
            std::uniform_real_distribution<float> unit(0.f, 1.f);
            std::uniform_real_distribution<float> angle(0.f, 6.28318530718f);
            std::uniform_real_distribution<float> sizeVar(0.85f, 1.15f);

            std::vector<float> positions, normals, uvs;
            std::vector<unsigned int> indices;
            unsigned int baseVert = 0;

            // Compute max depth to determine which nodes are "near tips".
            int maxDepth = 0;
            for (auto& n : nodes_) maxDepth = std::max(maxDepth, n.depth);

            // Leaf-eligible: terminal nodes always, plus thin branches in the
            // upper canopy.  Foliage should never grow on the thick trunk, so
            // gate on radius relative to the trunk radius (not the tiny twig min).
            const int depthThresh = static_cast<int>(static_cast<float>(maxDepth) * 0.4f);
            const float radiusThresh = tp.trunkRadius * 0.4f;

            // ── Low-poly foliage puff (deformed UV sphere, radial normals) ──
            auto emitBlob = [&](const Vector3& c, float radius) {
                constexpr int latSegs = 4;
                constexpr int lonSegs = 6;
                constexpr float PI = 3.14159265358979f;
                const unsigned int start = baseVert;
                for (int lat = 0; lat <= latSegs; ++lat) {
                    const float v = static_cast<float>(lat) / static_cast<float>(latSegs);
                    const float theta = v * PI;
                    const float sinT = std::sin(theta), cosT = std::cos(theta);
                    for (int lon = 0; lon <= lonSegs; ++lon) {
                        const float u = static_cast<float>(lon) / static_cast<float>(lonSegs);
                        const float phi = u * 2.f * PI;
                        const float nx = sinT * std::cos(phi);
                        const float ny = cosT;
                        const float nz = sinT * std::sin(phi);
                        positions.push_back(c.x + nx * radius);
                        positions.push_back(c.y + ny * radius);
                        positions.push_back(c.z + nz * radius);
                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                        uvs.push_back(u);
                        uvs.push_back(v);
                    }
                }
                const int rowVerts = lonSegs + 1;
                for (int lat = 0; lat < latSegs; ++lat) {
                    for (int lon = 0; lon < lonSegs; ++lon) {
                        const unsigned int a = start + static_cast<unsigned int>(lat * rowVerts + lon);
                        const unsigned int b = a + static_cast<unsigned int>(rowVerts);
                        indices.push_back(a);
                        indices.push_back(b);
                        indices.push_back(a + 1);
                        indices.push_back(a + 1);
                        indices.push_back(b);
                        indices.push_back(b + 1);
                    }
                }
                baseVert += static_cast<unsigned int>((latSegs + 1) * rowVerts);
            };

            auto emitQuad = [&](const Vector3& pos, const Vector3& r2,
                                const Vector3& u2, const Vector3& qn, float hs) {
                Vector3 corners[4];
                corners[0].copy(pos).addScaledVector(r2, -hs).addScaledVector(u2, -hs);
                corners[1].copy(pos).addScaledVector(r2,  hs).addScaledVector(u2, -hs);
                corners[2].copy(pos).addScaledVector(r2,  hs).addScaledVector(u2,  hs);
                corners[3].copy(pos).addScaledVector(r2, -hs).addScaledVector(u2,  hs);

                for (int c = 0; c < 4; ++c) {
                    positions.push_back(corners[c].x);
                    positions.push_back(corners[c].y);
                    positions.push_back(corners[c].z);
                    normals.push_back(qn.x);
                    normals.push_back(qn.y);
                    normals.push_back(qn.z);
                }
                uvs.push_back(0.f); uvs.push_back(0.f);
                uvs.push_back(1.f); uvs.push_back(0.f);
                uvs.push_back(1.f); uvs.push_back(1.f);
                uvs.push_back(0.f); uvs.push_back(1.f);

                indices.push_back(baseVert);
                indices.push_back(baseVert + 1);
                indices.push_back(baseVert + 2);
                indices.push_back(baseVert);
                indices.push_back(baseVert + 2);
                indices.push_back(baseVert + 3);
                baseVert += 4;
            };

            for (size_t ni = 0; ni < nodes_.size(); ++ni) {
                const auto& node = nodes_[ni];
                const bool eligible = node.terminal ||
                        (node.depth >= depthThresh && node.radius <= radiusThresh);
                if (!eligible) continue;

                float prob = tp.leafDensity;
                if (!node.terminal) prob *= 0.6f;
                if (unit(rng) > prob) continue;

                // ── Foliage puffs ─────────────────────────────────────────
                if (tp.leafStyle == LeafStyle::Blob) {
                    const int puffs = std::max(1, tp.leavesPerCluster);
                    for (int p = 0; p < puffs; ++p) {
                        Vector3 pos;
                        pos.copy(node.position);
                        if (puffs > 1) {
                            pos.x += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                            pos.y += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                            pos.z += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        }
                        emitBlob(pos, tp.leafSize * sizeVar(rng));
                    }
                    continue;
                }

                const int count = (tp.leafStyle == LeafStyle::Cluster ||
                                   tp.leafStyle == LeafStyle::CrossQuad)
                                          ? tp.leavesPerCluster
                                          : 1;
                for (int li = 0; li < count; ++li) {
                    Vector3 pos;
                    pos.copy(node.position);
                    if (count > 1) {
                        pos.x += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        pos.y += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        pos.z += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                    }

                    // Growth axis: branch direction blended toward up, so the
                    // card stands roughly upright like a spray of leaves.
                    Vector3 branchDir{0.f, 1.f, 0.f};
                    if (node.parent >= 0) {
                        branchDir.subVectors(node.position,
                                nodes_[static_cast<size_t>(node.parent)].position);
                        if (branchDir.lengthSq() > 1e-8f)
                            branchDir.normalize();
                        else
                            branchDir.set(0.f, 1.f, 0.f);
                    }
                    Vector3 axis;
                    axis.set(branchDir.x * 0.5f,
                             branchDir.y * 0.5f + 0.5f,
                             branchDir.z * 0.5f);
                    if (axis.lengthSq() < 1e-8f) axis.set(0.f, 1.f, 0.f);
                    axis.normalize();

                    // Two perpendicular vectors around the axis (random roll).
                    Vector3 perpA, perpB;
                    {
                        Vector3 ref{0.f, 1.f, 0.f};
                        if (std::abs(axis.dot(ref)) > 0.95f) ref.set(1.f, 0.f, 0.f);
                        perpA.crossVectors(axis, ref).normalize();
                        perpB.crossVectors(axis, perpA).normalize();
                        const float roll = angle(rng);
                        const float cr = std::cos(roll), sr = std::sin(roll);
                        Vector3 a2, b2;
                        a2.set(perpA.x * cr + perpB.x * sr,
                               perpA.y * cr + perpB.y * sr,
                               perpA.z * cr + perpB.z * sr);
                        b2.set(-perpA.x * sr + perpB.x * cr,
                               -perpA.y * sr + perpB.y * cr,
                               -perpA.z * sr + perpB.z * cr);
                        perpA = a2;
                        perpB = b2;
                    }

                    const float hs = tp.leafSize * 0.5f * sizeVar(rng);

                    // Up-biased normals so foliage reads as lit from the sky
                    // (rather than going dark when a card is edge-on to the sun).
                    Vector3 nA, nB;
                    nA.set(perpB.x * 0.4f, perpB.y * 0.4f + 1.f, perpB.z * 0.4f).normalize();
                    nB.set(perpA.x * 0.4f, perpA.y * 0.4f + 1.f, perpA.z * 0.4f).normalize();

                    if (tp.leafStyle == LeafStyle::CrossQuad) {
                        // Proper 3D cross: two upright quads sharing the growth
                        // axis, with perpendicular normals — one always faces
                        // the viewer, no edge-on slivers.
                        emitQuad(pos, perpA, axis, nA, hs);// spans perpA × axis
                        emitQuad(pos, perpB, axis, nB, hs);// spans perpB × axis
                    } else {
                        emitQuad(pos, perpA, axis, nA, hs);
                    }
                }
            }

            if (positions.empty()) return std::make_shared<BufferGeometry>();

            auto geo = std::make_shared<BufferGeometry>();
            geo->setIndex(indices);
            geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            return geo;
        }

        // ── Convenience: build + bake in one call ────────────────────────
        [[nodiscard]] std::shared_ptr<BufferGeometry> createTrunkGeometry(const TreeParams& tp) {
            buildSkeleton(tp);
            return makeTrunkGeometry(tp);
        }

        [[nodiscard]] std::shared_ptr<BufferGeometry> createLeafGeometry(const TreeParams& tp) {
            return makeLeafGeometry(tp);
        }

        // ── Node access (for placement queries, debug viz) ───────────────
        [[nodiscard]] const std::vector<detail::TreeNode>& nodes() const { return nodes_; }

        // ── Leaf positions for external instancing ───────────────────────
        [[nodiscard]] std::vector<Vector3> getLeafPositions(const TreeParams& tp) const {
            std::vector<Vector3> out;
            std::mt19937 rng(tp.seed ^ 0xBEEF);
            std::uniform_real_distribution<float> unit(0.f, 1.f);
            for (auto& n : nodes_) {
                if (!n.terminal) continue;
                if (unit(rng) > tp.leafDensity) continue;
                if (tp.leafStyle == LeafStyle::Cluster) {
                    for (int li = 0; li < tp.leavesPerCluster; ++li) {
                        Vector3 p;
                        p.copy(n.position);
                        p.x += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        p.y += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        p.z += (unit(rng) - 0.5f) * tp.leafSpread * 2.f;
                        out.push_back(p);
                    }
                } else {
                    out.push_back(n.position);
                }
            }
            return out;
        }

    private:
        unsigned int seed_ = 1337;
        std::vector<detail::TreeNode> nodes_;

        // ── Scatter attraction points in the crown envelope ──────────────
        void scatterAttractors(std::mt19937& rng, const TreeParams& tp,
                               const Vector3& centre,
                               std::vector<Vector3>& out) const {
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            std::uniform_real_distribution<float> angle(0.f, 6.28318530718f);
            const float rx = tp.crownRadiusX;
            const float rz = tp.crownRadiusZ;
            const float hy = tp.crownHeight * 0.5f;

            int placed = 0;
            while (placed < tp.attractorCount) {
                Vector3 p;
                switch (tp.crownShape) {
                    case CrownShape::Sphere:
                    case CrownShape::Ellipsoid: {
                        // Uniform random in unit sphere, then scale.
                        float x, y, z;
                        do {
                            x = u01(rng) * 2.f - 1.f;
                            y = u01(rng) * 2.f - 1.f;
                            z = u01(rng) * 2.f - 1.f;
                        } while (x * x + y * y + z * z > 1.f);
                        p.set(centre.x + x * rx, centre.y + y * hy, centre.z + z * rz);
                        break;
                    }
                    case CrownShape::Cone: {
                        // Cone: apex at top, base radius at bottom.
                        const float t = u01(rng);// 0=apex, 1=base
                        const float coneR = t;
                        const float a = angle(rng);
                        const float r = std::sqrt(u01(rng)) * coneR;
                        p.set(centre.x + r * std::cos(a) * rx,
                              centre.y + hy - t * tp.crownHeight,
                              centre.z + r * std::sin(a) * rz);
                        break;
                    }
                    case CrownShape::Hemisphere: {
                        float x, y, z;
                        do {
                            x = u01(rng) * 2.f - 1.f;
                            y = u01(rng);
                            z = u01(rng) * 2.f - 1.f;
                        } while (x * x + y * y + z * z > 1.f);
                        p.set(centre.x + x * rx,
                              centre.y - hy + y * tp.crownHeight,
                              centre.z + z * rz);
                        break;
                    }
                    case CrownShape::Cylinder: {
                        const float a = angle(rng);
                        const float r = std::sqrt(u01(rng));
                        p.set(centre.x + r * std::cos(a) * rx,
                              centre.y + (u01(rng) - 0.5f) * tp.crownHeight,
                              centre.z + r * std::sin(a) * rz);
                        break;
                    }
                }
                out.push_back(p);
                ++placed;
            }
        }

        // ── Compute branch radii bottom-up via the pipe model ────────────
        void computeRadii(const TreeParams& tp) {
            // Process from leaves toward root: iterate in reverse topological
            // order.  Since children always have higher indices than parents
            // (we only append), a reverse sweep is correct.
            const float n = tp.radiusExponent;
            for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
                auto& node = nodes_[static_cast<size_t>(i)];
                if (node.children.empty()) {
                    node.radius = tp.minBranchRadius;
                } else {
                    float sum = 0.f;
                    for (int ci : node.children) {
                        sum += std::pow(nodes_[static_cast<size_t>(ci)].radius, n);
                    }
                    node.radius = std::pow(sum, 1.f / n);
                }
            }
            // Scale so root matches the requested trunk radius.
            if (!nodes_.empty() && nodes_[0].radius > 1e-8f) {
                const float scale = tp.trunkRadius / nodes_[0].radius;
                for (auto& nd : nodes_) nd.radius *= scale;
            }
        }

        // ── Build a local orthonormal frame from a direction vector ──────
        static void buildFrame(const Vector3& dir, Vector3& perp1, Vector3& perp2) {
            // Choose a non-parallel reference vector.
            Vector3 ref{0.f, 1.f, 0.f};
            if (std::abs(dir.dot(ref)) > 0.99f) ref.set(1.f, 0.f, 0.f);
            perp1.crossVectors(dir, ref).normalize();
            perp2.crossVectors(dir, perp1).normalize();
        }

        // ── Helpers ──────────────────────────────────────────────────────
    };

    // ── Species presets ──────────────────────────────────────────────────
    inline void applyPreset(int preset, TreeParams& p) {
        switch (preset) {
            case 0: {// Oak — wide spreading canopy
                p.trunkHeight = 3.5f;
                p.trunkRadius = 0.18f;
                p.crownShape = CrownShape::Sphere;
                p.crownRadiusX = 4.0f;
                p.crownRadiusZ = 4.0f;
                p.crownHeight = 5.0f;
                p.attractorCount = 1000;
                p.influenceDistance = 5.0f;
                p.killDistance = 0.9f;
                p.segmentLength = 0.35f;
                p.maxIterations = 250;
                p.randomness = 0.1f;
                p.tropism = -0.01f;
                p.radiusExponent = 2.2f;
                p.minBranchRadius = 0.006f;
                p.radialSegments = 8;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafSize = 0.8f;
                p.leafDensity = 0.9f;
                p.leavesPerCluster = 5;
                p.leafSpread = 0.6f;
                p.barkColor = {0.30f, 0.22f, 0.15f};
                p.leafColor = {0.24f, 0.44f, 0.13f};
                break;
            }
            case 1: {// Pine — tall conical conifer
                p.trunkHeight = 5.0f;
                p.trunkRadius = 0.10f;
                p.crownShape = CrownShape::Cone;
                p.crownRadiusX = 2.0f;
                p.crownRadiusZ = 2.0f;
                p.crownHeight = 7.0f;
                p.attractorCount = 600;
                p.influenceDistance = 3.5f;
                p.killDistance = 0.7f;
                p.segmentLength = 0.45f;
                p.maxIterations = 200;
                p.randomness = 0.05f;
                p.tropism = -0.04f;
                p.radiusExponent = 2.5f;
                p.minBranchRadius = 0.004f;
                p.radialSegments = 6;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafSize = 0.55f;
                p.leafDensity = 0.95f;
                p.leavesPerCluster = 6;
                p.leafSpread = 0.4f;
                p.barkColor = {0.32f, 0.20f, 0.12f};
                p.leafColor = {0.14f, 0.34f, 0.11f};
                break;
            }
            case 2: {// Birch — slender, upward branches
                p.trunkHeight = 4.5f;
                p.trunkRadius = 0.08f;
                p.crownShape = CrownShape::Ellipsoid;
                p.crownRadiusX = 2.0f;
                p.crownRadiusZ = 2.0f;
                p.crownHeight = 5.5f;
                p.attractorCount = 700;
                p.influenceDistance = 3.5f;
                p.killDistance = 0.6f;
                p.segmentLength = 0.35f;
                p.maxIterations = 220;
                p.randomness = 0.07f;
                p.tropism = 0.01f;
                p.radiusExponent = 2.0f;
                p.minBranchRadius = 0.004f;
                p.radialSegments = 6;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafSize = 0.65f;
                p.leafDensity = 0.9f;
                p.leavesPerCluster = 4;
                p.leafSpread = 0.5f;
                p.barkColor = {0.85f, 0.82f, 0.78f};
                p.leafColor = {0.38f, 0.52f, 0.18f};
                break;
            }
            case 3: {// Willow — drooping branches
                p.trunkHeight = 3.0f;
                p.trunkRadius = 0.14f;
                p.crownShape = CrownShape::Hemisphere;
                p.crownRadiusX = 4.5f;
                p.crownRadiusZ = 4.5f;
                p.crownHeight = 4.5f;
                p.attractorCount = 900;
                p.influenceDistance = 4.5f;
                p.killDistance = 0.7f;
                p.segmentLength = 0.3f;
                p.maxIterations = 280;
                p.randomness = 0.12f;
                p.tropism = -0.08f;
                p.radiusExponent = 2.0f;
                p.minBranchRadius = 0.003f;
                p.radialSegments = 6;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafSize = 0.6f;
                p.leafDensity = 0.95f;
                p.leavesPerCluster = 5;
                p.leafSpread = 0.45f;
                p.barkColor = {0.30f, 0.25f, 0.18f};
                p.leafColor = {0.32f, 0.50f, 0.20f};
                break;
            }
            default: break;// Custom — leave params unchanged
        }
    }

}// namespace threepp::vegetation

#endif// THREEPP_EXTRAS_VEGETATION_TREEGENERATOR_HPP
