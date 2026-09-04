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
#include "threepp/extras/vegetation/TreeTextures.hpp"// LeafShape (species blade outline)
#include "threepp/math/Rng.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
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
        Frond = 4,    // cards strung ALONG thin branch chains, laid in the branch
                      // plane and following its droop — a layered "shelf" of
                      // needles per branch (conifers). Use with a frond cutout
                      // texture (makeNeedleFrondTexture) + BranchingMode::Whorl.
    };

    // ── Skeleton construction mode ───────────────────────────────────────
    enum class BranchingMode {
        Colonise = 0,// space colonisation (Runions 2007) — broadleaf crowns
        Whorl = 1,   // monopodial trunk + whorled, drooping branches — conifers
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

        // ── Pendulous shoots (Colonise mode) ─────────────────────────────
        //
        // The long whip shoots that hang off the limbs of a birch — the
        // "pendula" in Betula pendula — and the trait that most identifies the
        // species at a distance.
        //
        // They cannot come out of the colonisation loop itself: growth is
        // steered by attractors, the crown envelope holds none BELOW the
        // branches, and a downward bias there just fights the envelope and
        // flattens the crown. So they are grown afterwards, hanging out of the
        // tips the crown already reached.
        float pendantLength = 0.f;  // world units; 0 disables
        float pendantDensity = 0.7f;// 0..1 fraction of tips that carry one

        // ── Skeleton mode ────────────────────────────────────────────────
        BranchingMode branchingMode = BranchingMode::Colonise;

        // ── Whorl mode (conifers) — ignored unless branchingMode == Whorl ─
        // A straight monopodial trunk carries rings ("whorls") of branches at
        // regular height intervals; branch length follows the crown profile
        // (long at the base → short at the apex) so the silhouette is conical.
        float whorlSpacing = 0.65f;      // vertical gap between whorls (world units)
        int branchesPerWhorl = 5;        // branches radiating from each whorl ring
        float whorlJitter = 0.35f;       // 0..1 azimuth + spacing randomisation
        float branchDroop = 0.38f;       // downward sag accumulated along a branch
        float branchTipUpturn = 0.35f;   // spruce: the last segments turn back up
        float crownProfileExponent = 1.3f;// >1 = slimmer spire, <1 = fuller cone
        float sideTwigDensity = 0.6f;    // 0..1 second-order twigs per branch node
        float branchLength = 0.f;        // base branch length; 0 → derive from crownRadiusX

        // ── Trunk shape variation (both modes) ───────────────────────────
        // Per-seed whole-tree lean + a multi-octave noise bend replace the old
        // single sine so no two trunks share a curve; a slow frame twist spirals
        // the bark ridges (per-seed handedness).
        float trunkLean = 0.05f;         // max whole-tree lean (radians, random azimuth)
        float trunkBend = 1.0f;          // multiplier on the bend-polyline amplitude
        float trunkTwist = 0.5f;         // bark-ridge spiral rate (radians / world unit)

        // ── Bark cross-section (both modes) ──────────────────────────────
        // Non-circular tube profile (periodic in the angular coordinate so the
        // j=0/j=R seam matches). Distinct per species: birch nearly smooth, oak
        // gnarly, spruce lightly plated.
        float barkBumpAmp = 0.10f;       // primary lobe amplitude (fraction of radius)
        int barkBumpLobes = 3;           // primary lobe count
        float barkBumpAmp2 = 0.05f;      // secondary (finer) lobe amplitude
        int barkBumpLobes2 = 7;          // secondary lobe count
        float rootFlareAsym = 0.5f;      // 0..1 buttress-lobe strength at the base
        // Surface character of the bark texture — see makeBarkTextures.
        BarkStyle barkStyle = BarkStyle::Furrowed;
        // How dark the thinnest twigs go relative to the bole, written into the
        // trunk mesh's vertex colours (1 = no change; needs vertexColors on the
        // bark material).
        //
        // One bark texture over the whole skeleton paints twigs in bole colour,
        // and on a birch that is wrong in the most visible way there is: the
        // hanging shoots come out as bright white wires through the canopy,
        // where the real thing is dark red-brown and only the trunk is white.
        // The darkening is warm-biased, so twigs go brown rather than grey.
        float twigShade = 1.0f;

        // ── Leaves ───────────────────────────────────────────────────────
        LeafStyle leafStyle = LeafStyle::CrossQuad;
        // Blade outline the leaf-card atlas is drawn with. Purely a texture
        // concern, but it belongs with the species description — see
        // makeLeafClusterTexture.
        LeafShape leafShape = LeafShape::Ovate;
        float leafSize = 0.7f;// quad half-size (card), or puff radius for Blob
        float leafDensity = 0.9f;
        int leavesPerCluster = 5;
        // Tessellation of a LeafStyle::Blob puff (UV sphere). 5×8 = 80 triangles
        // is the fjord-demo silhouette at ~1000 far trees; a forest planted from
        // a canopy model instances the same prototype tens of thousands of times,
        // where 80 tris × 3 puffs × ~700 tips is the whole frame budget. Drop to
        // 3×5 (30 tris) for a distance tier — the puff is a handful of pixels and
        // the lumpiness noise, not the segment count, carries its outline.
        // Default stays 5×8 so every existing caller is byte-identical.
        int blobLatSegs = 5;
        int blobLonSegs = 8;
        // The leaf map is an N×N grid of INDEPENDENT sprig variants; every card
        // picks one cell and a random horizontal mirror (2·N² distinct cards).
        //
        // With one tile, every card in the forest is the same image at the same
        // orientation, and a canopy stops reading as leaves the moment the
        // viewer resolves a card: it becomes one square stamp repeated a
        // thousand times. No amount of per-card tint or size jitter hides a
        // repeated SILHOUETTE ([[procedural-avoid-perfect-repetition]]).
        //
        // Must match the `variants` argument the leaf texture was generated
        // with — makeLeafClusterTexture / makeNeedleFrondTexture default to the
        // same 2, so the pair agrees unless a caller supplies its own atlas.
        // Set to 1 when using a hand-authored single-sprig texture.
        int leafAtlasCells = 2;
        // Radius of the per-leaf scatter about the twig axis (the cluster is
        // strung ALONG the parent segment, so this is a cross-section, not a
        // box half-extent — see makeLeafGeometry).
        float leafSpread = 0.5f;
        float leafClumping = 0.5f;// 0 = solid shell, →1 = clumped with sky-gaps
        // How far the card leans from "upright" toward "follows the shoot".
        //
        // 0 stands every card up regardless of the twig it grows on — right for
        // a canopy of ascending shoots. 1 aligns the card with the shoot, which
        // on a pendulous species points DOWN, so the sprig hangs from its stem
        // instead of standing on it. The atlas is drawn stem-at-low-v and the
        // card spans v along this axis, so pointing the axis down is what puts
        // the stem at the top and drapes the leaflets below it — no separate
        // "upside down" texture needed.
        float leafDroop = 0.f;
        // Strength of the baked canopy occlusion written into the leaf vertex
        // colours (see makeLeafGeometry). 0 disables it and leaves foliage lit
        // only by its own normals — flat, with no interior depth.
        float foliageOcclusion = 1.0f;
        // Baked canopy occlusion on LeafStyle::Blob puffs, PER VERTEX, as a
        // multiple of foliageOcclusion. 0 (the default, and byte-identical to
        // the old behaviour) shades a puff with ONE tint sampled at its CENTRE.
        //
        // A card carries the occlusion of the point it is drawn at, so a card
        // canopy's outer shell is exposed and its core is dark. A puff sampled
        // at its centre carries the CORE value over its whole surface — the
        // visible shell included — which at leaf-size radii is a voxel or two
        // away from where the pixels are. The tint is not just brightness (it
        // also swings hue from open-canopy yellow-green to buried blue-green),
        // so the far tier ends up hue-shifted against the near one even after
        // the albedos are matched. Sampling per vertex puts each part of the
        // puff on its own occlusion, and >1 pushes the contrast further (a
        // caller that solves its blob albedo against a measured target — see
        // CanopyForest's makeForestTreeVariant — renormalises the mean, so the
        // contrast costs it nothing).
        //
        // MEASURED, and it is why the default is 0. On the fjord demo's far
        // tier it is the wrong lever for the bright-speckle end: the visible
        // SHELL of a puff really is more exposed than its centre, so shading it
        // per vertex makes the sunward cap brighter, not darker. All-L0 against
        // all-L1 over the same frame (geiranger, a lit bench at ~250 m,
        // green-dominant tree mask): tree pixels over G=90 went 0.75% -> 1.00%
        // against the card tier's 0.59% under the demo sun, with the means and
        // G/R unchanged inside 2%. The far tier's residual is COVERAGE (an
        // opaque sphere against a cutout canopy), not baked occlusion.
        float blobOcclusion = 0.f;

        // ── Albedo hints (sRGB) ──────────────────────────────────────────
        std::array<float, 3> barkColor = {0.35f, 0.25f, 0.18f};
        std::array<float, 3> leafColor = {0.25f, 0.45f, 0.15f};

        // Field-wise, which is exactly what a parameter block means by equal:
        // two TreeParams that compare equal describe the same tree, so a
        // caller can skip regenerating one. The editor's TreeConfig defaults
        // its own comparison to this.
        bool operator==(const TreeParams&) const = default;
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
            // Multiplier on the foliage card size carried by this node.
            //
            // A conifer's apex sprays are SHORT — the crown profile takes branch
            // length to nearly zero at the leader — but a card size fixed by
            // leafSize alone stays full width there, so the top whorls end up
            // wider than the branches holding them and the tree finishes in a
            // club instead of a spire. The whorl builder tapers this with the
            // crown profile; space colonisation leaves it at 1.
            float leafScale = 1.f;
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
            // Conifers use the explicit whorl builder; broadleaf crowns use space
            // colonisation. Both fill the same nodes_ structure.
            if (tp.branchingMode == BranchingMode::Whorl) {
                buildWhorlSkeleton(tp);
                return;
            }

            math::Rng rng(tp.seed ? tp.seed : 1u);
            nodes_.clear();

            // Grow the trunk as a chain of nodes from origin to crown base.
            // The per-seed trunk curve (lean + multi-octave bend) keeps it from
            // reading as a ramrod-straight cylinder; the base stays planted.
            const int trunkSegs = std::max(1, static_cast<int>(std::round(tp.trunkHeight / tp.segmentLength)));
            const float trunkStep = tp.trunkHeight / static_cast<float>(trunkSegs);
            makeTrunkShape(rng, tp, 0.f, tp.trunkHeight);
            for (int i = 0; i <= trunkSegs; ++i) {
                detail::TreeNode n;
                const float y = static_cast<float>(i) * trunkStep;
                const Vector3 off = trunkShape_.offsetAt(y);
                n.position = {off.x, y, off.z};
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
            auto jitter = [](math::Rng& r) { return r.nextFloat(-1.f, 1.f); };

            for (int iter = 0; iter < tp.maxIterations && !attractors.empty(); ++iter) {
                const int nodeCount = static_cast<int>(nodes_.size());

                struct GrowInfo {
                    Vector3 dir{0.f, 0.f, 0.f};
                    int count = 0;
                };
                std::vector<GrowInfo> grow(static_cast<size_t>(nodeCount));
                std::vector<bool> killMask(attractors.size(), false);

                // For each attractor find the single closest node that can still
                // fork. A node grows one child per iteration for as long as it
                // keeps winning attractors, and NOTHING stopped it winning: a
                // node sitting in the middle of a pocket of attractors stays the
                // closest one to the points on every side it has not grown
                // toward yet, so it spawns a child, and another, and another —
                // measured at 248 children on one node. That is a hedgehog of
                // stubs at a single point, which the pipe model then makes into
                // an unusually thick branch, and which the leaf pass covers in a
                // cluster per stub: the packed clump of foliage hanging off a
                // fat limb. A real fork is 2- or 3-way, so cap it — and cap it
                // by EXCLUDING saturated nodes from this search, not by skipping
                // them when growing. Skipping at growth time leaves the
                // attractor assigned to a node that cannot use it, and that
                // region of the crown simply stops growing.
                constexpr size_t kMaxChildren = 3;
                for (size_t ai = 0; ai < attractors.size(); ++ai) {
                    const auto& ap = attractors[ai];
                    float bestDist2 = infDist2;
                    int bestNode = -1;
                    for (int ni = 0; ni < nodeCount; ++ni) {
                        if (nodes_[static_cast<size_t>(ni)].children.size() >= kMaxChildren) continue;
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

            // Hang the whip shoots off the crown the loop just grew.
            growPendants(tp, rng);

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

            std::vector<float> positions, normals, uvs, colors;
            std::vector<unsigned int> indices;
            unsigned int baseVert = 0;

            // Twig darkening ramp: full bole colour at a third of the trunk
            // radius, fading to twigShade at twig scale. Always written, so the
            // attribute is there whether or not a material reads it.
            const float shadeHi = std::max(tp.trunkRadius * 0.33f, 1e-4f);
            const float shadeLo = std::max(tp.minBranchRadius * 2.f, 1e-5f);
            auto twigTint = [&](float r) {
                float t = (r - shadeLo) / std::max(shadeHi - shadeLo, 1e-5f);
                t = std::clamp(t, 0.f, 1.f);
                t = t * t * (3.f - 2.f * t);// smoothstep — no visible band
                const float s = tp.twigShade + (1.f - tp.twigShade) * t;
                // Warm bias so a darkened twig reads as bark, not as shadow.
                return Vector3{std::min(1.f, s * 1.06f), s * 0.88f, s * 0.78f};
            };

            // Tree vertical extent — drives the non-axisymmetric root flare below.
            float treeMinY = std::numeric_limits<float>::max();
            float treeMaxY = -std::numeric_limits<float>::max();
            for (const auto& nd : nodes_) {
                treeMinY = std::min(treeMinY, nd.position.y);
                treeMaxY = std::max(treeMaxY, nd.position.y);
            }
            const float flareH = std::max(treeMaxY - treeMinY, 1e-3f) * 0.12f;

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
                        // Slow per-seed twist of the frame about the tangent, so
                        // the bark ridges (and the cross-section lobes below)
                        // spiral gently up the trunk instead of running dead
                        // straight. Handedness/rate come from twistRate_ (seed).
                        const float dArc = arcLen[static_cast<size_t>(i)] - arcLen[static_cast<size_t>(i - 1)];
                        if (twistRate_ != 0.f && dArc > 0.f) {
                            P.applyAxisAngle(tangent[static_cast<size_t>(i)], twistRate_ * dArc);
                            Q.applyAxisAngle(tangent[static_cast<size_t>(i)], twistRate_ * dArc);
                        }
                    }

                    const auto& nd = nodes_[static_cast<size_t>(chain[static_cast<size_t>(i)])];
                    const float r = nd.radius;
                    // Absolute arc length (world units) so bark tiles at a
                    // consistent scale on trunk and twigs alike; material
                    // `repeat` controls the final tile density.
                    const float v = arcLen[static_cast<size_t>(i)];
                    // Non-axisymmetric root flare: near the very base, low-frequency
                    // angular lobes swell into buttress roots that fade out with
                    // height (flareF → 0 above flareH). Only ADDS radius (the
                    // 0.5+0.5·sin term is ≥ 0), so it never pinches the trunk.
                    const float aboveBase = nd.position.y - treeMinY;
                    float flareF = std::clamp((flareH - aboveBase) / flareH, 0.f, 1.f);
                    flareF *= flareF;
                    const Vector3 tint = twigTint(r);

                    for (int j = 0; j <= R; ++j) {
                        const float a = static_cast<float>(j) / static_cast<float>(R) * 6.28318530718f;
                        const float ca = std::cos(a), sa = std::sin(a);
                        const float nx = P.x * ca + Q.x * sa;
                        const float ny = P.y * ca + Q.y * sa;
                        const float nz = P.z * ca + Q.z * sa;
                        // Species cross-section: two periodic-in-`a` lobe bands
                        // (integer lobe counts keep the j=0/j=R seam matched) plus
                        // the buttress flare. Amplitudes/lobe counts are per-species
                        // (barkBumpAmp/Lobes…): birch nearly smooth, oak gnarly.
                        const float bump =
                                1.f
                                + tp.barkBumpAmp  * std::sin(static_cast<float>(tp.barkBumpLobes)  * a + bumpPhase_)
                                + tp.barkBumpAmp2 * std::sin(static_cast<float>(tp.barkBumpLobes2) * a - bumpPhase_ * 0.7f)
                                + tp.rootFlareAsym * flareF * 0.6f
                                        * (0.5f + 0.5f * std::sin(3.f * a + bumpPhase_ * 1.3f));
                        const float rr = r * bump;
                        positions.push_back(nd.position.x + nx * rr);
                        positions.push_back(nd.position.y + ny * rr);
                        positions.push_back(nd.position.z + nz * rr);
                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                        uvs.push_back(static_cast<float>(j) / static_cast<float>(R));
                        uvs.push_back(v);
                        colors.push_back(tint.x);
                        colors.push_back(tint.y);
                        colors.push_back(tint.z);
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
            geo->setAttribute("color", FloatBufferAttribute::create(colors, 3));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            return geo;
        }

        // ── Step 3: leaf quads at terminal branch tips ───────────────────
        [[nodiscard]] std::shared_ptr<BufferGeometry> makeLeafGeometry(const TreeParams& tp) const {
            if (nodes_.empty()) return std::make_shared<BufferGeometry>();

            math::Rng rng(tp.seed ^ 0xBEEF);
            auto unit = [](math::Rng& r) { return r.nextFloat(); };
            auto angle = [](math::Rng& r) { return r.nextFloat(0.f, 6.28318530718f); };
            auto sizeVar = [](math::Rng& r) { return r.nextFloat(0.85f, 1.15f); };

            std::vector<float> positions, normals, uvs, colors;
            std::vector<unsigned int> indices;
            unsigned int baseVert = 0;

            // Compute max depth (for "near tip" test) and the canopy vertical
            // extent (for the top-lit tonal gradient).
            int maxDepth = 0;
            float canopyMinY = std::numeric_limits<float>::max();
            float canopyMaxY = -std::numeric_limits<float>::max();
            for (auto& n : nodes_) {
                maxDepth = std::max(maxDepth, n.depth);
                if (n.children.empty() || n.radius <= tp.trunkRadius * 0.4f) {
                    canopyMinY = std::min(canopyMinY, n.position.y);
                    canopyMaxY = std::max(canopyMaxY, n.position.y);
                }
            }
            const float canopySpan = std::max(0.5f, canopyMaxY - canopyMinY);

            // Leaf-eligible: terminal nodes always, plus thin branches in the
            // upper canopy.  Foliage should never grow on the thick trunk, so
            // gate on radius relative to the trunk radius (not the tiny twig min).
            const int depthThresh = static_cast<int>(static_cast<float>(maxDepth) * 0.4f);
            const float radiusThresh = tp.trunkRadius * 0.4f;

            // ── Canopy occupancy grid (drives per-card AO) ────────────────
            //
            // Without this, every card is lit purely by its own normal, so a
            // canopy renders as full sun on whatever faces the light and pure
            // shadow-map black behind it — no mid-tones, which is exactly what
            // makes procedural foliage read as plastic. Real canopies are a
            // participating volume: a leaf is dimmed by how much foliage sits
            // between it and the sky, and by how buried it is in the crown.
            //
            // Both terms are cheap to bake here: voxelise the leaf-bearing nodes
            // once, then per card (a) march the column above it for sky
            // occlusion and (b) read the local neighbourhood for burial depth.
            // The result goes into the vertex colour, so it costs nothing at
            // draw time and works identically on every backend.
            const CanopyField field = buildCanopyField(tp, depthThresh, radiusThresh);

            // Foliage tint at a POINT, with the per-card/per-puff brightness
            // jitter passed in rather than drawn here — so the same shading can
            // be evaluated many times (once per puff vertex) without touching
            // the rng stream, which the geometry itself depends on. See
            // `tintFor` below for what the terms mean.
            auto shadeFor = [&](const Vector3& p, float jitter, float aoScale) {
                const float h = std::clamp((p.y - canopyMinY) / canopySpan, 0.f, 1.f);
                const float ao = std::clamp(tp.foliageOcclusion * aoScale, 0.f, 2.f);
                const float sky = field.skyOcclusion(p) * ao;
                const float deep = field.burial(p) * ao;
                // Never reach zero: a canopy interior is dim, not black — it is
                // still fed by sky bounce through the leaves around it.
                const float occ = std::clamp(1.f - 0.70f * sky - 0.38f * deep, 0.18f, 1.f);
                // Every factor is capped at 1, so the vertex colour only ever
                // DARKENS the texture. Letting it climb above 1 turns it into a
                // brightness gain on top of the albedo, and the best-lit foliage
                // blows past the leaf colour into a pale mint that no amount of
                // tone-mapping brings back.
                const float bright = occ * jitter * (0.88f + 0.12f * h);
                // exposure: 1 in full view of the sky, 0 buried.
                const float expo = std::clamp(1.f - std::max(sky, deep), 0.f, 1.f);
                Vector3 c;
                c.set(bright * (0.86f + 0.14f * expo),
                      bright * (0.94f + 0.06f * expo),
                      bright * (1.00f - 0.18f * expo));
                return c;
            };

            // ── Low-poly foliage puff (deformed UV sphere, radial normals) ──
            //
            // `col` is the puff's tint sampled at its centre; `jitter` is the
            // brightness draw that produced it, replayed per vertex when
            // TreeParams::blobOcclusion turns on per-vertex occlusion (so a
            // puff keeps ONE random brightness and only its baked occlusion
            // varies across the sphere).
            const float blobOcc = std::max(0.f, tp.blobOcclusion);
            auto emitBlob = [&](const Vector3& c, float radius, const Vector3& col,
                                float jitter) {
                const int latSegs = std::max(2, tp.blobLatSegs);
                const int lonSegs = std::max(3, tp.blobLonSegs);
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
                        // Lumpy, not spherical. A smooth UV sphere reads as a
                        // green balloon on a stick — the single thing that gives
                        // away a low-poly foliage puff at any distance. Two
                        // octaves of noise in DIRECTION space (offset per puff by
                        // its centre, so neighbours differ) break the outline
                        // into an irregular clump.
                        //
                        // Sampling by direction rather than by world position is
                        // what keeps the poles safe: at lat 0 and latSegs every
                        // lon shares the same direction, so they all displace
                        // identically and stay coincident. Displacing by a
                        // phi-dependent amount instead would fan the pole vertex
                        // into long sliver triangles whose sub-pixel coverage
                        // flickers with the TAA jitter.
                        const float n1 = noise3(nx * 2.3f + c.x * 0.7f,
                                                ny * 2.3f + c.y * 0.7f,
                                                nz * 2.3f + c.z * 0.7f);
                        const float n2 = noise3(nx * 5.7f + c.z * 1.3f,
                                                ny * 5.7f + c.x * 1.3f,
                                                nz * 5.7f + c.y * 1.3f);
                        const float rr = radius * (0.74f + 0.36f * n1 + 0.16f * (n2 - 0.5f));
                        const Vector3 vp{c.x + nx * rr, c.y + ny * rr, c.z + nz * rr};
                        positions.push_back(vp.x);
                        positions.push_back(vp.y);
                        positions.push_back(vp.z);
                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                        uvs.push_back(u);
                        uvs.push_back(v);
                        // Off (default): the centre tint, flat over the sphere.
                        // On: this vertex's own burial / sky occlusion, so the
                        // sunward cap of a buried puff is not lit as if it were
                        // the crown's top and the shell is not tinted with the
                        // core's hue. Costs one field lookup per vertex at BUILD
                        // time and nothing at draw time.
                        const Vector3 vc = blobOcc > 0.f ? shadeFor(vp, jitter, blobOcc) : col;
                        colors.push_back(vc.x);
                        colors.push_back(vc.y);
                        colors.push_back(vc.z);
                    }
                }
                const int rowVerts = lonSegs + 1;
                for (int lat = 0; lat < latSegs; ++lat) {
                    for (int lon = 0; lon < lonSegs; ++lon) {
                        const unsigned int a = start + static_cast<unsigned int>(lat * rowVerts + lon);
                        const unsigned int b = a + static_cast<unsigned int>(rowVerts);
                        // CCW from outside — matches the radial normals. Wound the
                        // other way the puff renders inside-out: culling shows the
                        // far INNER wall whose normal points away from the viewer,
                        // so the canopy shades inverted ("lit" on its dark side).
                        indices.push_back(a);
                        indices.push_back(a + 1);
                        indices.push_back(b);
                        indices.push_back(a + 1);
                        indices.push_back(b + 1);
                        indices.push_back(b);
                    }
                }
                baseVert += static_cast<unsigned int>((latSegs + 1) * rowVerts);
            };

            // Which cell of the leaf atlas a card samples, and whether it is
            // mirrored. Drawn per card — see TreeParams::leafAtlasCells.
            struct CardUv {
                float u0 = 0.f, v0 = 0.f, du = 1.f, dv = 1.f;
                bool mirror = false;
            };
            const int atlasN = std::max(1, tp.leafAtlasCells);
            auto pickCard = [&](math::Rng& r) {
                CardUv c;
                if (atlasN > 1) {
                    const int cells = atlasN * atlasN;
                    const int cell = std::min(cells - 1,
                            static_cast<int>(unit(r) * static_cast<float>(cells)));
                    const float s = 1.f / static_cast<float>(atlasN);
                    // Inset the sampled rect: coarse mips average across the
                    // cell boundary, and without a margin a card picks up its
                    // neighbour's leaflets as a fringe at distance.
                    const float inset = s * 0.02f;
                    c.u0 = static_cast<float>(cell % atlasN) * s + inset;
                    c.v0 = static_cast<float>(cell / atlasN) * s + inset;
                    c.du = s - 2.f * inset;
                    c.dv = s - 2.f * inset;
                }
                c.mirror = unit(r) < 0.5f;
                return c;
            };

            auto emitQuad = [&](const Vector3& pos, const Vector3& r2,
                                const Vector3& u2, const Vector3& qn, float hs,
                                const Vector3& col, const CardUv& uvc) {
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
                    colors.push_back(col.x);
                    colors.push_back(col.y);
                    colors.push_back(col.z);
                }
                const float uLo = uvc.mirror ? uvc.u0 + uvc.du : uvc.u0;
                const float uHi = uvc.mirror ? uvc.u0 : uvc.u0 + uvc.du;
                const float vLo = uvc.v0, vHi = uvc.v0 + uvc.dv;
                uvs.push_back(uLo); uvs.push_back(vLo);
                uvs.push_back(uHi); uvs.push_back(vLo);
                uvs.push_back(uHi); uvs.push_back(vHi);
                uvs.push_back(uLo); uvs.push_back(vHi);

                // WIND THE CARD SO ITS FRONT FACE IS THE SIDE ITS NORMAL POINTS.
                //
                // A leaf card's shading normal is deliberately NOT its plane
                // normal — it is the crown-hull/up-biased normal, chosen to make
                // the canopy shade as a volume. But the material is
                // Side::Double, and the double-sided path multiplies the normal
                // by gl_FrontFacing (`normal = normal * faceDirection`), which
                // is decided by WINDING. With winding unrelated to the supplied
                // normal, that flip is arbitrary: for any given viewpoint about
                // half the cards get their normal negated into pointing away
                // from the viewer and downward, so they sample lower-hemisphere
                // (ground) irradiance and render as flat desaturated grey
                // patches that swim around the canopy as the camera moves.
                //
                // Ordering the triangles so cross(r2, u2) agrees with the normal
                // makes the flip meaningful again: a viewer on the far side sees
                // the back face and gets -normal, which then points back toward
                // them, instead of the pathological case of a front face whose
                // normal already faced away.
                Vector3 geo;
                geo.crossVectors(r2, u2);
                const bool flip = geo.dot(qn) < 0.f;
                if (flip) {
                    indices.push_back(baseVert);
                    indices.push_back(baseVert + 2);
                    indices.push_back(baseVert + 1);
                    indices.push_back(baseVert);
                    indices.push_back(baseVert + 3);
                    indices.push_back(baseVert + 2);
                } else {
                    indices.push_back(baseVert);
                    indices.push_back(baseVert + 1);
                    indices.push_back(baseVert + 2);
                    indices.push_back(baseVert);
                    indices.push_back(baseVert + 2);
                    indices.push_back(baseVert + 3);
                }
                baseVert += 4;
            };

            // Per-card tint (vertex colour, multiplies the leaf texture).
            //
            // Baked canopy occlusion: `sky` is how much foliage stands between
            // this card and the sky, `deep` how buried it is in the crown. Both
            // darken, but they are kept separate because they do different jobs
            // — sky occlusion produces the top-lit falloff that gives a canopy
            // its form, burial produces the dark core that makes it read as a
            // solid volume rather than a hollow shell of cards.
            //
            // Shaded foliage also shifts HUE, not just brightness: leaves in the
            // open are yellow-green (direct sun, thinner shade leaves), leaves
            // in the core are a deep blue-green. Darkening alone looks like a
            // dirty texture; the hue shift is what sells the depth.
            auto tintFor = [&](const Vector3& p) {
                return shadeFor(p, 0.86f + unit(rng) * 0.14f, 1.f);
            };

            // Crown-hull normal. A leaf card is a flat quad, but the canopy it
            // belongs to is a volume, and it should shade like one — so the
            // card's normal is pushed toward "outward from the crown centre"
            // rather than its own geometric facing. This is what turns a cloud
            // of independently-lit quads into a lit mass with a bright top, a
            // soft terminator down the sides and a dark underside.
            //
            // Keeping a slice of +Y in the blend preserves the original intent
            // of the up-biased normals (foliage catches sky light instead of
            // going black when a card is edge-on to the sun); dropping to a pure
            // hull normal makes the crown underside unreadably dark.
            const Vector3 crownCentre = field.valid()
                    ? field.centroid
                    : Vector3{0.f, tp.trunkHeight + tp.crownHeight * 0.5f, 0.f};
            // Broadleaf crowns are round masses, so they take a strong hull
            // blend. Conifer sprays are near-horizontal SHELVES whose real
            // normal points up; blending those hard toward "outward from the
            // crown axis" turns them side-facing and starves them of overhead
            // sun, so the whole tree goes black. They keep mostly their own
            // up-biased normal and take only a touch of hull for the terminator.
            const float hullBlend = (tp.leafStyle == LeafStyle::Frond) ? 0.34f : 0.78f;
            auto hullNormal = [&](const Vector3& p, const Vector3& fallback) {
                Vector3 outward;
                outward.subVectors(p, crownCentre);
                if (outward.lengthSq() < 1e-8f) outward.copy(fallback);
                outward.normalize();
                const float k = hullBlend, m = 1.f - hullBlend;
                Vector3 n;
                n.set(outward.x * k + fallback.x * m,
                      outward.y * k + fallback.y * m + 0.30f,
                      outward.z * k + fallback.z * m);
                if (n.lengthSq() < 1e-8f) n.set(0.f, 1.f, 0.f);
                return n.normalize();
            };

            for (size_t ni = 0; ni < nodes_.size(); ++ni) {
                const auto& node = nodes_[ni];
                // Frond conifers carry foliage ALONG every thin branch node (not
                // just tips / upper canopy), so the whole drooping branch reads as
                // a needle shelf. Broadleaf styles keep the tip+upper-canopy gate.
                // Fronds hang on any thin node, INCLUDING the trunk leader near
                // the apex (depth 0 but already down to twig radius). Requiring
                // depth >= 1 excluded the leader entirely, which is what left a
                // bare spike above the topmost whorl.
                const bool eligible = (tp.leafStyle == LeafStyle::Frond)
                        ? (node.parent >= 0 && node.radius <= radiusThresh)
                        : (node.terminal || (node.depth >= depthThresh && node.radius <= radiusThresh));
                if (!eligible) continue;

                // Spatial clumping: low-frequency noise carves whole regions of
                // foliage away, so the canopy outline looks grown (irregular,
                // with sky-gaps) rather than a solid trimmed shell.
                if (tp.leafClumping > 0.f) {
                    constexpr float f = 0.6f;
                    const float n = noise3(node.position.x * f + 13.1f,
                                           node.position.y * f + 7.7f,
                                           node.position.z * f + 41.3f);
                    if (n < tp.leafClumping * 0.55f) continue;
                }

                float prob = tp.leafDensity;
                if (tp.leafStyle != LeafStyle::Frond && !node.terminal) prob *= 0.6f;
                if (unit(rng) > prob) continue;

                // ── Needle frond shelves (conifers) ───────────────────────
                // A pair of cards lying in the BRANCH PLANE at this node — flat
                // shelf + one rolled about the branch axis — so the branch reads
                // as a layered spray of needles that FOLLOWS the droop. Same
                // flicker discipline as CrossQuad: a shared up-biased normal and a
                // per-card depth de-tie offset (the shelf cards are near-coplanar).
                if (tp.leafStyle == LeafStyle::Frond) {
                    Vector3 along{0.f, 1.f, 0.f};
                    Vector3 parentPos = node.position;
                    if (node.parent >= 0) {
                        parentPos = nodes_[static_cast<size_t>(node.parent)].position;
                        along.subVectors(node.position, parentPos);
                        if (along.lengthSq() > 1e-8f) along.normalize();
                        else along.set(0.f, 1.f, 0.f);
                    }
                    // Horizontal spread direction of the shelf (⟂ branch, ~level).
                    Vector3 side;
                    side.crossVectors(along, Vector3(0.f, 1.f, 0.f));
                    if (side.lengthSq() < 1e-6f) side.set(1.f, 0.f, 0.f);
                    side.normalize();
                    // Shelf normal, biased up so the frond catches sky light rather
                    // than going dark edge-on (deliberate look; see CrossQuad note).
                    Vector3 nrm;
                    nrm.crossVectors(side, along);
                    if (nrm.lengthSq() < 1e-8f) nrm.set(0.f, 1.f, 0.f);
                    nrm.normalize();
                    Vector3 nShared;
                    nShared.set(nrm.x * 0.4f, std::abs(nrm.y) * 0.4f + 1.f, nrm.z * 0.4f).normalize();
                    // Conifer sprays get the same crown-hull treatment, so a
                    // whorl shades bright on top and dark underneath instead of
                    // every shelf in the tree being lit identically.
                    nShared = hullNormal(node.position, nShared);

                    // Clothe the WHOLE segment from the parent to this node with a
                    // short row of overlapping frond blades, so a spruce branch
                    // reads as a continuous dense needle shelf rather than a stick
                    // with a tuft at each joint. Two blades per station (flat +
                    // slightly rolled about the branch axis), shared up-biased
                    // normal + a per-card depth de-tie (near-coplanar cards flicker
                    // otherwise). The roll is kept SHALLOW (~30°): with a steep
                    // roll the blades fuzz vertically, whorl shelves merge into a
                    // solid fur column and the serrated silhouette is lost.
                    const float rc = std::cos(0.55f), rs = std::sin(0.55f);
                    Vector3 sideWide, sideRoll;
                    sideWide.copy(side).multiplyScalar(1.7f);
                    sideRoll.set(side.x * rc + nrm.x * rs,
                                 side.y * rc + nrm.y * rs,
                                 side.z * rc + nrm.z * rs).multiplyScalar(1.7f);
                    constexpr int stations = 2;// along-segment blade rows
                    for (int st = 0; st < stations; ++st) {
                        const float f = (static_cast<float>(st) + 0.5f) / static_cast<float>(stations);
                        Vector3 base;
                        base.copy(parentPos).lerp(node.position, f);
                        // node.leafScale tapers the spray with the crown profile:
                        // full width on the long base whorls, roughly half that on
                        // the apex shoots and the leader (see TreeNode::leafScale).
                        const float hs = tp.leafSize * 0.5f * sizeVar(rng) * node.leafScale;
                        const Vector3 col = tintFor(base);
                        const float depthSep = tp.leafSize * 0.08f * (unit(rng) - 0.5f);
                        Vector3 posSep;
                        posSep.copy(base).addScaledVector(nShared, depthSep);
                        // The two blades of a station share a cell: they are the
                        // same spray seen rolled, not two different sprays.
                        const CardUv uvc = pickCard(rng);
                        emitQuad(posSep, sideWide, along, nShared, hs, col, uvc);
                        emitQuad(posSep, sideRoll, along, nShared, hs, col, uvc);
                    }
                    continue;
                }

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
                        // Hoisted, not inlined as call arguments: both draw from
                        // `rng`, and C++ leaves the evaluation order of call
                        // arguments unspecified — inline, the Blob path's output
                        // depends on the compiler, so a "deterministic for a
                        // given seed" tree differs between toolchains.
                        const float puffR = tp.leafSize * sizeVar(rng) * node.leafScale;
                        const float puffJitter = 0.86f + unit(rng) * 0.14f;
                        const Vector3 puffCol = shadeFor(pos, puffJitter, 1.f);
                        emitBlob(pos, puffR, puffCol, puffJitter);
                    }
                    continue;
                }

                const int count = (tp.leafStyle == LeafStyle::Cluster ||
                                   tp.leafStyle == LeafStyle::CrossQuad)
                                          ? tp.leavesPerCluster
                                          : 1;
                // Growth axis: branch direction blended toward up, so the card
                // stands roughly upright like a spray of leaves. Per NODE, not
                // per card — nothing about it depends on which leaf of the
                // cluster this is.
                Vector3 branchDir{0.f, 1.f, 0.f};
                Vector3 parentPos = node.position;
                if (node.parent >= 0) {
                    parentPos = nodes_[static_cast<size_t>(node.parent)].position;
                    branchDir.subVectors(node.position, parentPos);
                    if (branchDir.lengthSq() > 1e-8f)
                        branchDir.normalize();
                    else
                        branchDir.set(0.f, 1.f, 0.f);
                }
                // Frame for scattering the cluster ABOUT the twig axis.
                Vector3 twigU, twigV;
                buildFrame(branchDir, twigU, twigV);

                for (int li = 0; li < count; ++li) {
                    // LEAVES GROW ALONG A TWIG, NOT IN A BALL AT ITS END.
                    //
                    // The old scatter was an independent uniform offset per
                    // axis, i.e. a CUBE centred on the node — so a cluster of
                    // cards filled an axis-aligned box, and once the cards are
                    // dense enough to close it the canopy shows the box: a
                    // packed square of leaves, one per node, all the same size
                    // and all aligned with the world axes. Walking the segment
                    // from the parent node and scattering RADIALLY (a disc
                    // about the twig, sqrt for uniform area) puts the foliage
                    // where the wood is, covers the twig instead of drifting
                    // off it, and has no flat sides to give away.
                    Vector3 pos;
                    if (count > 1) {
                        const float f = std::clamp(
                                (static_cast<float>(li) + 0.5f + (unit(rng) - 0.5f) * 0.8f) /
                                        static_cast<float>(count),
                                0.f, 1.f);
                        pos.copy(parentPos).lerp(node.position, f);
                        const float a = angle(rng);
                        const float r = tp.leafSpread * std::sqrt(unit(rng));
                        pos.addScaledVector(twigU, std::cos(a) * r)
                                .addScaledVector(twigV, std::sin(a) * r);
                    } else {
                        pos.copy(node.position);
                    }

                    // leafDroop slides the card from "upright" (+0.5 of world up
                    // mixed in) to "follows the shoot" (−0.5, i.e. hanging on a
                    // pendulous twig). See TreeParams::leafDroop.
                    const float upMix = 0.5f - std::clamp(tp.leafDroop, 0.f, 1.f);
                    Vector3 axis;
                    axis.set(branchDir.x * 0.5f,
                             branchDir.y * 0.5f + upMix,
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

                    const float hs = tp.leafSize * 0.5f * sizeVar(rng) * node.leafScale;
                    const Vector3 col = tintFor(pos);

                    // Up-biased normals so foliage reads as lit from the sky
                    // (rather than going dark when a card is edge-on to the sun).
                    Vector3 nA, nB;
                    nA.set(perpB.x * 0.4f, perpB.y * 0.4f + 1.f, perpB.z * 0.4f).normalize();
                    nB.set(perpA.x * 0.4f, perpA.y * 0.4f + 1.f, perpA.z * 0.4f).normalize();

                    // DEPTH DE-TIE. Nearly-coplanar cards (cluster mates, or
                    // single quads from two adjacent tips) raster at near-equal
                    // depth; under a jittered projection (Vulkan TAA) the two
                    // planes shift by DIFFERENT sub-pixel amounts each frame, so
                    // the depth-test winner ALTERNATES — visible as 1-2
                    // "z-fighting" leaves flickering per tree wherever the pair
                    // also differs in normal/texel. A per-card random offset
                    // along the card normal keeps overlapping cards decisively
                    // separated; ±4% of leafSize is invisible inside a canopy.
                    const float depthSep = tp.leafSize * 0.08f * (unit(rng) - 0.5f);

                    // Bend both toward the crown hull (see hullNormal above).
                    nA = hullNormal(pos, nA);
                    nB = hullNormal(pos, nB);

                    if (tp.leafStyle == LeafStyle::CrossQuad) {
                        // Proper 3D cross: two upright quads sharing the growth
                        // axis — one always faces the viewer, no edge-on slivers.
                        // ONE SHARED normal for both halves: the planes intersect
                        // along the axis by design, and in the grazing band where
                        // they raster at near-equal depth the jittered winner
                        // alternates per frame. At that seam both quads sample
                        // the same texel column (u=0.5) and share the tint, so
                        // with a shared normal the G-buffer output is IDENTICAL
                        // whichever quad wins — the alternation becomes
                        // invisible. (Per-quad normals nA/nB made it flicker as
                        // a brightness flip; both were heavily up-biased anyway,
                        // so their average barely changes the lighting.)
                        Vector3 nShared;
                        nShared.addVectors(nA, nB).normalize();
                        Vector3 posSep;
                        posSep.copy(pos).addScaledVector(nShared, depthSep);
                        // ONE atlas cell for BOTH halves, for the same reason
                        // they share the normal and the tint: at the seam the
                        // two planes raster at near-equal depth and the jittered
                        // winner alternates per frame, and the alternation is
                        // only invisible while every G-buffer channel matches.
                        // Give the halves different cells and the seam samples
                        // two different sprigs — the flicker fixed in 9b598c5d,
                        // back. The variety has to come from card to card, not
                        // from the two halves of one card.
                        const CardUv uvc = pickCard(rng);
                        emitQuad(posSep, perpA, axis, nShared, hs, col, uvc);// spans perpA × axis
                        emitQuad(posSep, perpB, axis, nShared, hs, col, uvc);// spans perpB × axis
                    } else {
                        Vector3 posSep;
                        posSep.copy(pos).addScaledVector(nA, depthSep);
                        emitQuad(posSep, perpA, axis, nA, hs, col, pickCard(rng));
                    }
                }
            }

            if (positions.empty()) return std::make_shared<BufferGeometry>();

            auto geo = std::make_shared<BufferGeometry>();
            geo->setIndex(indices);
            geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
            geo->setAttribute("color", FloatBufferAttribute::create(colors, 3));
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
            math::Rng rng(tp.seed ^ 0xBEEF);
            auto unit = [](math::Rng& r) { return r.nextFloat(); };
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

        // ── Voxelised canopy, for baking foliage occlusion ───────────────
        struct CanopyField {
            Vector3 origin;      // grid corner (world)
            Vector3 centroid;    // centre of foliage mass — the "crown centre"
            float cell = 1.f;    // voxel edge (world units)
            int nx = 0, ny = 0, nz = 0;
            std::vector<float> density;// per-voxel leaf-node count
            // Normalisers: the 90th percentile of each raw measure taken over
            // the occupied voxels of THIS tree. Normalising by the busiest voxel
            // instead makes the whole effect hostage to a single outlier cell,
            // and its magnitude then drifts with crown size, leaf size and
            // attractor count — so the same AO settings that look right on an
            // oak wash out completely on a birch.
            float skyNorm = 1.f;
            float burialNorm = 1.f;

            [[nodiscard]] bool valid() const { return nx > 0 && ny > 0 && nz > 0; }

            [[nodiscard]] float at(int ix, int iy, int iz) const {
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return 0.f;
                return density[(static_cast<size_t>(iz) * static_cast<size_t>(ny) +
                                static_cast<size_t>(iy)) *
                                       static_cast<size_t>(nx) +
                               static_cast<size_t>(ix)];
            }

            void index(const Vector3& p, int& ix, int& iy, int& iz) const {
                ix = static_cast<int>(std::floor((p.x - origin.x) / cell));
                iy = static_cast<int>(std::floor((p.y - origin.y) / cell));
                iz = static_cast<int>(std::floor((p.z - origin.z) / cell));
            }

            // How much foliage sits between this point and the sky, 0..1.
            // Nearer voxels shade more (they subtend more of the hemisphere),
            // hence the 1/(1+d) falloff rather than a plain sum.
            [[nodiscard]] float rawSky(int ix, int iy, int iz) const {
                float occ = 0.f;
                for (int y = iy + 1; y < ny; ++y) {
                    const float d = static_cast<float>(y - iy);
                    // Sample a 3-wide column: a 1-voxel ray through a sparse
                    // grid aliases badly (a card either finds a hit directly
                    // above or nothing at all, giving a blotchy canopy).
                    float slab = 0.f;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dx = -1; dx <= 1; ++dx)
                            slab += at(ix + dx, y, iz + dz);
                    occ += (slab / 9.f) / (1.f + d);
                }
                return occ;
            }

            [[nodiscard]] float rawBurial(int ix, int iy, int iz) const {
                float sum = 0.f, wsum = 0.f;
                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            const float d2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
                            const float w = 1.f / (1.f + d2);
                            sum += at(ix + dx, iy + dy, iz + dz) * w;
                            wsum += w;
                        }
                    }
                }
                return sum / wsum;
            }

            // How much foliage stands between this point and the sky, 0..1.
            [[nodiscard]] float skyOcclusion(const Vector3& p) const {
                if (!valid()) return 0.f;
                int ix, iy, iz;
                index(p, ix, iy, iz);
                return std::clamp(rawSky(ix, iy, iz) / skyNorm, 0.f, 1.f);
            }

            // How buried the point is in surrounding foliage, 0 (exposed) .. 1.
            [[nodiscard]] float burial(const Vector3& p) const {
                if (!valid()) return 0.f;
                int ix, iy, iz;
                index(p, ix, iy, iz);
                return std::clamp(rawBurial(ix, iy, iz) / burialNorm, 0.f, 1.f);
            }
        };

        // Voxelise the leaf-bearing skeleton nodes. Uses the same eligibility
        // and clumping tests as the emitter, so the density field matches the
        // foliage that actually gets drawn (the per-card `leafDensity` dice roll
        // is a uniform thinning and does not change the spatial distribution).
        [[nodiscard]] CanopyField buildCanopyField(const TreeParams& tp, int depthThresh,
                                                   float radiusThresh) const {
            CanopyField f;
            std::vector<Vector3> pts;
            pts.reserve(nodes_.size());
            for (const auto& node : nodes_) {
                // Must match the emitter's gate in makeLeafGeometry exactly, or
                // the density field describes foliage that is not there.
                const bool eligible = (tp.leafStyle == LeafStyle::Frond)
                        ? (node.parent >= 0 && node.radius <= radiusThresh)
                        : (node.terminal || (node.depth >= depthThresh && node.radius <= radiusThresh));
                if (!eligible) continue;
                if (tp.leafClumping > 0.f) {
                    constexpr float cf = 0.6f;
                    const float n = noise3(node.position.x * cf + 13.1f,
                                           node.position.y * cf + 7.7f,
                                           node.position.z * cf + 41.3f);
                    if (n < tp.leafClumping * 0.55f) continue;
                }
                pts.push_back(node.position);
            }
            if (pts.size() < 8) return f;

            Vector3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max()};
            Vector3 hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                       -std::numeric_limits<float>::max()};
            Vector3 sum{0.f, 0.f, 0.f};
            for (const auto& p : pts) {
                lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
                hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
                sum.add(p);
            }
            f.centroid.copy(sum).divideScalar(static_cast<float>(pts.size()));

            // Voxel about the size of one leaf card: fine enough to resolve the
            // sky-gaps that clumping carves, coarse enough to stay cheap.
            f.cell = std::max(tp.leafSize * 1.2f, 0.15f);
            const float pad = f.cell * 2.f;
            f.origin.set(lo.x - pad, lo.y - pad, lo.z - pad);
            f.nx = std::clamp(static_cast<int>((hi.x - lo.x + 2.f * pad) / f.cell) + 1, 1, 96);
            f.ny = std::clamp(static_cast<int>((hi.y - lo.y + 2.f * pad) / f.cell) + 1, 1, 96);
            f.nz = std::clamp(static_cast<int>((hi.z - lo.z + 2.f * pad) / f.cell) + 1, 1, 96);
            f.density.assign(static_cast<size_t>(f.nx) * static_cast<size_t>(f.ny) *
                                     static_cast<size_t>(f.nz),
                             0.f);
            for (const auto& p : pts) {
                int ix, iy, iz;
                f.index(p, ix, iy, iz);
                if (ix < 0 || iy < 0 || iz < 0 || ix >= f.nx || iy >= f.ny || iz >= f.nz) continue;
                f.density[(static_cast<size_t>(iz) * static_cast<size_t>(f.ny) +
                           static_cast<size_t>(iy)) *
                                  static_cast<size_t>(f.nx) +
                          static_cast<size_t>(ix)] += 1.f;
            }
            // Calibrate against this tree: evaluate both raw measures over every
            // occupied voxel and take the 90th percentile as "fully occluded".
            std::vector<float> skySamples, burialSamples;
            skySamples.reserve(pts.size());
            burialSamples.reserve(pts.size());
            for (int iz = 0; iz < f.nz; ++iz) {
                for (int iy = 0; iy < f.ny; ++iy) {
                    for (int ix = 0; ix < f.nx; ++ix) {
                        if (f.at(ix, iy, iz) <= 0.f) continue;
                        skySamples.push_back(f.rawSky(ix, iy, iz));
                        burialSamples.push_back(f.rawBurial(ix, iy, iz));
                    }
                }
            }
            auto percentile90 = [](std::vector<float>& v, float fallback) {
                if (v.empty()) return fallback;
                const size_t k = (v.size() * 9) / 10;
                std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
                return std::max(v[k], 1e-3f);
            };
            f.skyNorm = percentile90(skySamples, 1.f);
            f.burialNorm = percentile90(burialSamples, 1.f);
            return f;
        }

        // ── Per-tree trunk curve (workstream 2) ──────────────────────────
        // A whole-tree lean (random azimuth) + a 3-octave sinusoidal "noise"
        // bend, so every seed traces a different centreline. offsetAt() returns
        // the XZ displacement of the trunk axis at a given height; the base stays
        // planted (displacement → 0 as t → 0). Shared by both skeleton modes.
        struct TrunkShape {
            float leanX = 0.f, leanZ = 0.f; // lean displacement at the crown top
            float amp = 0.f;                // bend amplitude (world units)
            float ph[3] = {0, 0, 0};        // per-octave phase
            float fr[3] = {2.1f, 4.7f, 9.3f};// per-octave frequency (in t)
            float baseY = 0.f, spanY = 1.f; // height range this curve maps over

            [[nodiscard]] Vector3 offsetAt(float y) const {
                const float t = std::clamp((y - baseY) / std::max(spanY, 1e-3f), 0.f, 1.f);
                float ox = leanX * t;
                float oz = leanZ * t;
                for (int o = 0; o < 3; ++o) {
                    const float a = amp * std::pow(0.5f, static_cast<float>(o));
                    ox += a * std::sin(t * fr[o] + ph[o]);
                    oz += a * std::cos(t * fr[o] * 1.13f + ph[o] * 1.7f + 2.0f);
                }
                // Plant the base: fade the whole displacement out near the ground.
                const float plant = std::sin(std::clamp(t, 0.f, 1.f) * 1.57079633f);
                return {ox * plant, 0.f, oz * plant};
            }
        };
        TrunkShape trunkShape_;
        float twistRate_ = 0.f;// bark-ridge frame spiral (radians / world unit)
        float bumpPhase_ = 0.f;// per-seed root-flare lobe phase

        // Derive the per-tree trunk curve + twist from the seed.
        void makeTrunkShape(math::Rng& rng, const TreeParams& tp,
                            float baseY, float spanY) {
            auto u01 = [](math::Rng& r) { return r.nextFloat(); };
            auto u11 = [](math::Rng& r) { return r.nextFloat(-1.f, 1.f); };
            const float leanAz = u01(rng) * 6.28318530718f;
            const float leanAmt = spanY * tp.trunkLean * (0.4f + 0.6f * u01(rng));
            trunkShape_.leanX = std::cos(leanAz) * leanAmt;
            trunkShape_.leanZ = std::sin(leanAz) * leanAmt;
            trunkShape_.amp = spanY * 0.05f * tp.trunkBend * (0.5f + u01(rng));
            for (int o = 0; o < 3; ++o) {
                trunkShape_.ph[o] = u01(rng) * 6.28318530718f;
                trunkShape_.fr[o] = (1.5f + 2.2f * static_cast<float>(o)) * (0.8f + 0.4f * u01(rng));
            }
            trunkShape_.baseY = baseY;
            trunkShape_.spanY = spanY;
            twistRate_ = tp.trunkTwist * u11(rng);// per-seed handedness + rate
            bumpPhase_ = u01(rng) * 6.28318530718f;
        }

        // ── Whorl (conifer) skeleton ─────────────────────────────────────
        // Emits into the SAME nodes_ structure as space colonisation, so
        // makeTrunkGeometry / computeRadii / makeLeafGeometry work unchanged.
        // Deterministic for a fixed seed.
        void buildWhorlSkeleton(const TreeParams& tp) {
            math::Rng rng(tp.seed ? tp.seed : 1u);
            nodes_.clear();

            const float H = tp.trunkHeight + tp.crownHeight;// total tree height
            makeTrunkShape(rng, tp, 0.f, H);

            auto u01 = [](math::Rng& r) { return r.nextFloat(); };
            auto u11 = [](math::Rng& r) { return r.nextFloat(-1.f, 1.f); };

            // Whorl band. Branches start above `trunkHeight` (the bare bole) and
            // stop BELOW a bare leader shoot at the very top — a spruce tapers to
            // a pointed apex, so the crown profile must reach ~zero at the top
            // whorl (no length floor) and the topmost stretch of trunk stays
            // branch-free except for tiny stubs.
            const float whorlStart = tp.trunkHeight;
            // Keep the bare leader SHORT. At 7% of tree height it left the top
            // ~1m of trunk with no branches at all, and since the crown profile
            // also drives branch length to ~0 near the top, the last whorls were
            // then dropped by the stub test below — so a spruce ended in a bare
            // wire poking out of the foliage.
            const float leaderH = std::clamp(H * 0.035f, 0.20f, 0.55f);
            const float whorlTop = H - leaderH;
            const float whorlSpan = std::max(whorlTop - whorlStart, 1e-3f);

            // 1) Monopodial trunk: one straight (curved) leader from the ground
            //    to the apex, following the per-seed trunk curve.
            const int trunkSegs = std::max(2, static_cast<int>(std::round(H / tp.segmentLength)));
            const float trunkStep = H / static_cast<float>(trunkSegs);
            std::vector<int> trunkIdx(static_cast<size_t>(trunkSegs + 1));
            for (int i = 0; i <= trunkSegs; ++i) {
                detail::TreeNode n;
                const float y = static_cast<float>(i) * trunkStep;
                const Vector3 off = trunkShape_.offsetAt(y);
                n.position = {off.x, y, off.z};
                n.parent = i > 0 ? trunkIdx[static_cast<size_t>(i - 1)] : -1;
                n.depth = 0;// trunk stays depth 0 → radii/leaf gates treat it as trunk
                // The leader is thin enough near the apex to carry foliage itself
                // (that is what fills the spire instead of leaving a bare wire),
                // but what grows there is a single young shoot — a narrow sleeve
                // of needles, not the full spray a base branch carries. Left at 1
                // it is the widest foliage on the tree sitting on its thinnest
                // wood, which is most of the "overloaded on top" read.
                n.leafScale = 0.45f + 0.35f * (1.f - std::clamp((y - whorlStart) / whorlSpan, 0.f, 1.f));
                const int idx = static_cast<int>(nodes_.size());
                if (i > 0) nodes_[static_cast<size_t>(trunkIdx[static_cast<size_t>(i - 1)])].children.push_back(idx);
                trunkIdx[static_cast<size_t>(i)] = idx;
                nodes_.push_back(n);
            }

            // 2) Whorls of branches.
            const float baseLen = (tp.branchLength > 0.f) ? tp.branchLength
                                                          : std::max(tp.crownRadiusX, tp.crownRadiusZ);
            const float spacing = std::max(tp.whorlSpacing, tp.segmentLength * 0.5f);
            const int perW = std::max(2, tp.branchesPerWhorl);

            int whorlIdx = 0;
            for (float wy = whorlStart; wy <= whorlTop + 1e-3f; wy += spacing, ++whorlIdx) {
                // Nearest trunk node to attach this whorl to.
                const int ti = std::min(trunkSegs,
                        std::max(0, static_cast<int>(std::round(wy / trunkStep))));
                const int parentTrunk = trunkIdx[static_cast<size_t>(ti)];
                const Vector3 trunkPos = nodes_[static_cast<size_t>(parentTrunk)].position;

                // Crown profile: full length at the base whorl → ~0 at the apex.
                // The floor is a tiny stub (fraction of baseLen, NOT clamped up to
                // a usable branch), so the last whorls serrate into the pointed
                // leader instead of capping the tree with a rounded tuft.
                const float tW = std::clamp((wy - whorlStart) / whorlSpan, 0.f, 1.f);
                const float lenScale = std::pow(1.f - tW, tp.crownProfileExponent);
                // The floor has to leave the top whorls LONG enough to survive
                // the stub test, or the apex loses its branches entirely.
                const float branchLen = baseLen * (0.10f + 0.90f * lenScale);
                if (branchLen < tp.segmentLength * 0.4f) continue;// too short even for a stub
                // Foliage follows branch LENGTH, or the apex whorls carry sprays
                // wider than the branches under them and the silhouette stops
                // tapering (see TreeNode::leafScale). Through sqrt and off a high
                // floor: the shelves have to keep OVERLAPPING as they shorten,
                // and tracking the length profile directly opened bare trunk down
                // the whole upper crown — a bottle brush for a club.
                const float whorlLeafScale = 0.60f + 0.40f * std::sqrt(lenScale);

                // Alternate the whorl's phase so successive rings interleave.
                const float baseAz = static_cast<float>(whorlIdx) * 0.61803399f * 6.28318530718f;
                for (int b = 0; b < perW; ++b) {
                    const float az = baseAz + (static_cast<float>(b) / static_cast<float>(perW)) * 6.28318530718f
                                   + u11(rng) * tp.whorlJitter * (6.28318530718f / static_cast<float>(perW));
                    // A slight per-branch initial pitch: base whorls angle down a
                    // touch, apex whorls angle up (young leader shoots).
                    const float pitch0 = (-0.15f + 0.5f * tW) + u11(rng) * 0.08f;
                    Vector3 dir{std::cos(az) * std::cos(pitch0), std::sin(pitch0), std::sin(az) * std::cos(pitch0)};
                    if (dir.lengthSq() < 1e-8f) dir.set(std::cos(az), 0.f, std::sin(az));
                    dir.normalize();
                    growBranch(tp, rng, parentTrunk, trunkPos, dir, branchLen, 1, tW, whorlLeafScale);
                }
            }

            // Terminals + radii (identical bookkeeping to the colonise path).
            for (auto& n : nodes_) {
                n.terminal = n.children.empty();
                if (n.terminal) n.radius = tp.minBranchRadius;
            }
            computeRadii(tp);
        }

        // Grow one drooping branch (a chain of nodes) outward from `startPos`,
        // attaching its first node to `parentNode`. Recurses once for second-
        // order side twigs. `order` = 1 primary branch, 2 side twig.
        void growBranch(const TreeParams& tp, math::Rng& rng, int parentNode,
                        const Vector3& startPos, Vector3 dir, float length,
                        int order, float crownT, float leafScale) {
            auto u01 = [](math::Rng& r) { return r.nextFloat(); };
            auto u11 = [](math::Rng& r) { return r.nextFloat(-1.f, 1.f); };
            const int segs = std::max(1, static_cast<int>(std::round(length / tp.segmentLength)));
            const float step = length / static_cast<float>(segs);
            int prev = parentNode;
            Vector3 pos = startPos;
            const int depthBase = order;// primary=1, twig=2
            for (int s = 0; s < segs; ++s) {
                const float u = static_cast<float>(s + 1) / static_cast<float>(segs);// 0..1 along branch
                // Gravity droop accumulates along the branch; the last ~30% turns
                // the tip back up (spruce) via branchTipUpturn.
                float droop = tp.branchDroop * step * (0.6f + 0.8f * u);
                if (u > 0.7f) droop -= tp.branchTipUpturn * step * ((u - 0.7f) / 0.3f);
                dir.y -= droop;
                // A little horizontal wander so branches aren't dead straight.
                dir.x += u11(rng) * tp.randomness * 0.5f;
                dir.z += u11(rng) * tp.randomness * 0.5f;
                if (dir.lengthSq() < 1e-8f) dir.set(0.f, -1.f, 0.f);
                dir.normalize();

                Vector3 next;
                next.copy(pos).addScaledVector(dir, step);
                detail::TreeNode n;
                n.position = next;
                n.parent = prev;
                n.depth = depthBase + s;
                n.leafScale = leafScale;
                const int idx = static_cast<int>(nodes_.size());
                nodes_[static_cast<size_t>(prev)].children.push_back(idx);
                nodes_.push_back(n);

                // Second-order side twigs branch off primary branches in the
                // branch plane, drooping — this is what fills a spruce branch into
                // a layered frond shelf rather than a bare stick.
                if (order == 1 && s > 0 && s < segs - 1 && tp.sideTwigDensity > 0.f &&
                    u01(rng) < tp.sideTwigDensity) {
                    // Perp to the branch, roughly horizontal, alternating sides.
                    Vector3 side;
                    side.crossVectors(dir, Vector3(0.f, 1.f, 0.f));
                    if (side.lengthSq() < 1e-6f) side.set(1.f, 0.f, 0.f);
                    side.normalize();
                    const float sgn = (s & 1) ? 1.f : -1.f;
                    Vector3 twigDir;
                    twigDir.copy(dir).multiplyScalar(0.55f)
                            .addScaledVector(side, sgn * (0.7f + 0.2f * u01(rng)));
                    twigDir.y -= 0.1f;
                    twigDir.normalize();
                    const float twigLen = length * (0.30f + 0.25f * u01(rng)) * (1.f - crownT * 0.4f);
                    if (twigLen > tp.segmentLength * 0.5f)
                        growBranch(tp, rng, idx, next, twigDir, twigLen, 2, crownT, leafScale);
                }
                prev = idx;
                pos = next;
            }
        }

        // ── Pendulous whip shoots ────────────────────────────────────────
        //
        // Grown from the tips the colonisation loop finished on, each starting
        // along the shoot that carries it and falling to vertical over its own
        // length — a birch whip leaves the limb at whatever angle the limb ends
        // at and is hanging straight down well before its tip. Gravity is
        // integrated as a weight that RAMPS (t²) rather than a constant
        // downward bias: a constant one bends the first segment as hard as the
        // last, which reads as a branch that was aimed at the ground rather
        // than one that fell to it.
        //
        // These carry foliage like any other thin twig (they are terminal, and
        // deep), so the crown's lower fringe becomes the curtain of leaves that
        // gives the species its silhouette.
        void growPendants(const TreeParams& tp, math::Rng& rng) {
            if (tp.pendantLength <= 0.f || tp.pendantDensity <= 0.f) return;

            auto u01 = [](math::Rng& r) { return r.nextFloat(); };
            auto u11 = [](math::Rng& r) { return r.nextFloat(-1.f, 1.f); };

            const int segs = std::max(2, static_cast<int>(std::round(tp.pendantLength / tp.segmentLength)));
            const float step = tp.pendantLength / static_cast<float>(segs);
            // Snapshot the count: the pendants appended below are themselves
            // tips, and letting them grow pendants of their own would run the
            // crown into the ground one whip at a time.
            const size_t tipCount = nodes_.size();

            for (size_t ni = 0; ni < tipCount; ++ni) {
                if (!nodes_[ni].children.empty()) continue;
                if (u01(rng) > tp.pendantDensity) continue;
                // Length varies per whip; a curtain of equal-length shoots
                // hangs to a dead-level hem.
                const float lengthScale = 0.55f + 0.75f * u01(rng);

                Vector3 dir{0.f, 1.f, 0.f};
                if (nodes_[ni].parent >= 0) {
                    dir.subVectors(nodes_[ni].position,
                                   nodes_[static_cast<size_t>(nodes_[ni].parent)].position);
                    if (dir.lengthSq() > 1e-8f) dir.normalize();
                    else dir.set(0.f, 1.f, 0.f);
                }

                Vector3 pos;
                pos.copy(nodes_[ni].position);
                int prev = static_cast<int>(ni);
                for (int s = 0; s < segs; ++s) {
                    const float t = (static_cast<float>(s) + 1.f) / static_cast<float>(segs);
                    const float w = t * t;// gravity accumulates along the shoot
                    Vector3 d;
                    d.set(dir.x * (1.f - w),
                          dir.y * (1.f - w) - w,
                          dir.z * (1.f - w));
                    d.x += u11(rng) * tp.randomness * 0.8f;
                    d.z += u11(rng) * tp.randomness * 0.8f;
                    if (d.lengthSq() < 1e-8f) d.set(0.f, -1.f, 0.f);
                    d.normalize();

                    detail::TreeNode n;
                    n.position.copy(pos).addScaledVector(d, step * lengthScale);
                    n.parent = prev;
                    n.depth = nodes_[static_cast<size_t>(prev)].depth + 1;
                    const int idx = static_cast<int>(nodes_.size());
                    nodes_[static_cast<size_t>(prev)].children.push_back(idx);
                    nodes_.push_back(n);
                    prev = idx;
                    pos.copy(n.position);
                }
            }
        }

        // ── Scatter attraction points in the crown envelope ──────────────
        void scatterAttractors(math::Rng& rng, const TreeParams& tp,
                               const Vector3& centre,
                               std::vector<Vector3>& out) const {
            auto u01 = [](math::Rng& r) { return r.nextFloat(); };
            auto angle = [](math::Rng& r) { return r.nextFloat(0.f, 6.28318530718f); };
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

            // Height-based taper + root flare. The pipe model leaves a
            // single-child trunk at constant radius (→ a plain cylinder); a
            // continuous taper narrows it toward the top, and a flare swells
            // the lowest stretch so the base spreads into the ground.
            float minY = std::numeric_limits<float>::max();
            float maxY = -std::numeric_limits<float>::max();
            for (const auto& nd : nodes_) {
                minY = std::min(minY, nd.position.y);
                maxY = std::max(maxY, nd.position.y);
            }
            const float span = std::max(maxY - minY, 1e-3f);
            const float flareH = span * 0.10f;
            for (auto& nd : nodes_) {
                const float t = (nd.position.y - minY) / span;       // 0 base .. 1 top
                const float taper = 1.f - 0.40f * t;                 // narrow upward
                const float fl = std::clamp((flareH - (nd.position.y - minY)) / flareH, 0.f, 1.f);
                const float flare = 1.f + 0.6f * fl * fl;            // swell at the very base
                nd.radius *= taper * flare;
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

        // ── 3D value noise (trilinear, hash lattice) for canopy clumping ──
        static float hashf(int x, int y, int z) {
            uint32_t n = static_cast<uint32_t>(x) * 374761393u +
                         static_cast<uint32_t>(y) * 668265263u +
                         static_cast<uint32_t>(z) * 1274126177u;
            n = (n ^ (n >> 13)) * 1274126177u;
            n = n ^ (n >> 16);
            return static_cast<float>(n & 0xffffffu) / static_cast<float>(0xffffff);
        }
        static float noise3(float x, float y, float z) {
            const int xi = static_cast<int>(std::floor(x));
            const int yi = static_cast<int>(std::floor(y));
            const int zi = static_cast<int>(std::floor(z));
            auto sm = [](float t) { return t * t * (3.f - 2.f * t); };
            const float fx = sm(x - static_cast<float>(xi));
            const float fy = sm(y - static_cast<float>(yi));
            const float fz = sm(z - static_cast<float>(zi));
            auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
            const float c000 = hashf(xi, yi, zi), c100 = hashf(xi + 1, yi, zi);
            const float c010 = hashf(xi, yi + 1, zi), c110 = hashf(xi + 1, yi + 1, zi);
            const float c001 = hashf(xi, yi, zi + 1), c101 = hashf(xi + 1, yi, zi + 1);
            const float c011 = hashf(xi, yi + 1, zi + 1), c111 = hashf(xi + 1, yi + 1, zi + 1);
            return lerp(lerp(lerp(c000, c100, fx), lerp(c010, c110, fx), fy),
                        lerp(lerp(c001, c101, fx), lerp(c011, c111, fx), fy), fz);
        }
    };

    // ── Species presets ──────────────────────────────────────────────────
    //
    // Applying a preset RESETS to defaults first. Each case only assigns the
    // fields it cares about, so without the reset the previous species leaks
    // through: `branchingMode` and `leafClumping` are set by the Pine case
    // alone, so Pine → Oak used to leave a whorled, unclumped "oak" that
    // matches no preset. The seed is carried across, since it identifies the
    // individual tree rather than the species.
    inline void applyPreset(int preset, TreeParams& p) {
        if (preset < 0 || preset > 3) return;// Custom — leave params untouched

        const unsigned int keepSeed = p.seed;
        p = TreeParams{};
        p.seed = keepSeed;

        switch (preset) {
            case 0: {// Oak — wide spreading canopy
                p.trunkHeight = 3.5f;
                p.trunkRadius = 0.18f;
                p.crownShape = CrownShape::Sphere;
                // Envelopes grew to hold the crown SIZE steady: the foliage used
                // to bulge a whole card-plus-spread past the attractor envelope,
                // and shrinking the cards to twig scale took that padding with it.
                p.crownRadiusX = 4.5f;
                p.crownRadiusZ = 4.5f;
                p.crownHeight = 5.0f;
                // Twig resolution — see the birch case for why kill distance
                // decides whether a canopy reads as leaves or as clumps.
                p.attractorCount = 1800;
                p.influenceDistance = 2.6f;
                p.killDistance = 0.45f;
                p.segmentLength = 0.24f;
                p.maxIterations = 300;
                p.randomness = 0.1f;
                p.tropism = -0.01f;
                p.radiusExponent = 2.2f;
                p.minBranchRadius = 0.006f;
                p.radialSegments = 8;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafShape = LeafShape::Lobed;// deeply scalloped oak blade
                p.leafSize = 0.45f;
                p.leafDensity = 0.9f;
                p.leavesPerCluster = 3;
                p.leafSpread = 0.20f;
                p.barkColor = {0.30f, 0.22f, 0.15f};
                p.leafColor = {0.24f, 0.44f, 0.13f};
                // Oak: gnarly, deeply-ridged bark + strong buttress roots.
                p.barkBumpAmp = 0.16f; p.barkBumpLobes = 5;
                p.barkBumpAmp2 = 0.08f; p.barkBumpLobes2 = 11;
                p.rootFlareAsym = 0.8f; p.trunkTwist = 0.35f;
                p.barkStyle = BarkStyle::Furrowed;
                break;
            }
            case 1: {// Pine / spruce — monopodial conifer (whorled branches)
                p.branchingMode = BranchingMode::Whorl;
                p.trunkHeight = 1.6f;   // short bare bole; whorls start low
                p.trunkRadius = 0.17f;
                p.crownShape = CrownShape::Cone;
                p.crownRadiusX = 2.4f;
                p.crownRadiusZ = 2.4f;
                p.crownHeight = 8.0f;
                p.segmentLength = 0.4f;
                p.randomness = 0.05f;
                p.radiusExponent = 2.4f;
                p.minBranchRadius = 0.004f;
                p.radialSegments = 6;
                // Whorl structure. The pitch has to be SMALLER than the vertical
                // reach of one whorl's foliage, or the rings read as a stack of
                // separate pancakes with bare trunk showing between them — the
                // single thing that most gives a procedural conifer away.
                p.whorlSpacing = 0.40f;
                p.branchesPerWhorl = 7;
                p.whorlJitter = 0.45f;
                p.branchDroop = 0.42f;
                p.branchTipUpturn = 0.4f;
                p.crownProfileExponent = 1.5f;
                p.sideTwigDensity = 0.7f;
                // Frond foliage strung along the drooping branches.
                p.leafStyle = LeafStyle::Frond;
                p.leafSize = 0.6f;
                p.leafDensity = 0.9f;
                p.leafClumping = 0.0f;
                p.barkColor = {0.32f, 0.20f, 0.12f};
                p.leafColor = {0.17f, 0.38f, 0.13f};
                // A conifer already occludes itself hard through sheer geometry
                // — overlapping whorls of dense sprays — so the full baked term
                // on top only crushes it to a flat silhouette. Measured on the
                // Vulkan deferred path (which adds ray-traced AO of its own),
                // full strength took the crown to a near-black mass with almost
                // no tonal range left.
                p.foliageOcclusion = 0.72f;
                // A conifer LEADER IS STRAIGHT. The default bend is written for
                // broadleaves, where a wandering trunk is the point; scaled by
                // tree height it puts well over a metre of lateral wander into a
                // 13 m spruce, and every whorl above the bend rides along with
                // it — the tree snakes. A monopodial conifer grows to the light
                // in one shot, so keep only enough to break the ramrod look.
                p.trunkBend = 0.12f;
                p.trunkLean = 0.02f;
                // Spruce bark: lightly plated, gentle spiral.
                p.barkBumpAmp = 0.08f; p.barkBumpLobes = 4;
                p.barkBumpAmp2 = 0.04f; p.barkBumpLobes2 = 9;
                p.rootFlareAsym = 0.45f; p.trunkTwist = 0.5f;
                p.barkStyle = BarkStyle::Plated;
                break;
            }
            case 2: {// Birch — conical crown of pendulous shoots
                // Slender is not a wire: at 0.08 over a 10 m tree the bole came
                // out ~16 cm through, and under a 6 m crown that reads as a pole
                // holding up a bush rather than a trunk.
                p.trunkRadius = 0.12f;
                // CONICAL, not a ball on a stick: a birch carries its widest
                // limbs low and tapers to the leader, so the envelope has to
                // taper too. An ellipsoid puts the widest ring at mid-height and
                // pinches the bottom, which is the opposite of the tree.
                p.crownShape = CrownShape::Cone;
                p.crownRadiusX = 2.6f;
                p.crownRadiusZ = 2.6f;
                p.trunkHeight = 4.0f;
                p.crownHeight = 7.0f;
                // KILL DISTANCE IS THE TWIG-RESOLUTION KNOB, and it decides
                // whether the canopy reads as leaves or as clumps.
                //
                // At 0.6 the crown grows ~45 tips, so a 4 m-wide crown is built
                // from about a hundred leaf-bearing nodes — each carrying a
                // whole cluster, each cluster ~1.5 m across. That is a bag of
                // balls, not foliage, and every ball is plainly visible as a
                // packed clump sitting on the end of a limb. Halving it roughly
                // eightfolds the tips, so the same leaf area is spread over
                // many small sprays instead of a few big ones.
                p.attractorCount = 1400;
                p.influenceDistance = 2.0f;
                p.killDistance = 0.30f;
                p.segmentLength = 0.22f;
                p.maxIterations = 260;
                p.randomness = 0.07f;
                p.tropism = 0.01f;
                p.radiusExponent = 2.0f;
                p.minBranchRadius = 0.004f;
                p.radialSegments = 6;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafShape = LeafShape::Serrate;// small toothed birch blade
                // A card is one SPRIG, so its world size is the size of a twig
                // end, not of a branch. At 0.65 — and these get instanced at up
                // to 2× in scenes — a single card spanned well over a metre and
                // the canopy resolved into its individual squares.
                // Card AREA goes as the square, so shrinking one has to be paid
                // for in count or the canopy thins out and the white bark on the
                // twigs shows straight through it — on a birch that is the most
                // conspicuous way to lose canopy cover there is.
                // Cards shrink with the twigs: total leaf AREA (and so overdraw)
                // is about what it was, spread over ~4× the sprays.
                p.leafSize = 0.36f;
                p.leafDensity = 0.9f;
                p.leavesPerCluster = 3;
                p.leafSpread = 0.16f;
                // Betula PENDULA: whip shoots hanging off ascending limbs, with
                // the foliage draping from them rather than standing on them.
                p.pendantLength = 1.5f;
                p.pendantDensity = 0.75f;
                p.leafDroop = 0.85f;
                p.barkColor = {0.85f, 0.82f, 0.78f};
                p.leafColor = {0.38f, 0.52f, 0.18f};
                // Birch: nearly smooth papery bark, faint spiral.
                p.barkBumpAmp = 0.035f; p.barkBumpLobes = 3;
                p.barkBumpAmp2 = 0.02f; p.barkBumpLobes2 = 7;
                p.rootFlareAsym = 0.25f; p.trunkTwist = 0.25f;
                p.barkStyle = BarkStyle::Papery;// lenticels — the birch tell
                // White bole, dark red-brown shoots. Without this the pendant
                // whips hang through the canopy as bright white wires.
                p.twigShade = 0.30f;
                break;
            }
            case 3: {// Willow — drooping branches
                p.trunkHeight = 3.0f;
                p.trunkRadius = 0.14f;
                p.crownShape = CrownShape::Hemisphere;
                p.crownRadiusX = 4.9f;
                p.crownRadiusZ = 4.9f;
                p.crownHeight = 4.5f;
                p.attractorCount = 1800;
                p.influenceDistance = 2.6f;
                p.killDistance = 0.38f;
                p.segmentLength = 0.22f;
                p.maxIterations = 320;
                p.randomness = 0.12f;
                p.tropism = -0.08f;
                p.radiusExponent = 2.0f;
                p.minBranchRadius = 0.003f;
                p.radialSegments = 6;
                p.leafStyle = LeafStyle::CrossQuad;
                p.leafShape = LeafShape::Lanceolate;// long narrow willow blade
                p.leafSize = 0.38f;
                p.leafDensity = 0.95f;
                p.leavesPerCluster = 3;
                p.leafSpread = 0.16f;
                p.barkColor = {0.30f, 0.25f, 0.18f};
                p.leafColor = {0.32f, 0.50f, 0.20f};
                break;
            }
            default: break;// Custom — leave params unchanged
        }
    }

}// namespace threepp::vegetation

#endif// THREEPP_EXTRAS_VEGETATION_TREEGENERATOR_HPP
