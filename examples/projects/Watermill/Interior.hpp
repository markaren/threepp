// The inside of the watermill: its floors, stairs and posts, the machinery the wheel drives, and what was left
// behind when the miller went. Long abandoned but whole: dust lies thick on everything that faces up, cobwebs hang in
// a few corners, leaves have blown in under a broken pane, but the gearing is sound and turns when the wheel turns.
// InteriorLayout.hpp says where everything stands and gives the gear train's numbers.

#ifndef THREEPP_EXAMPLE_WATERMILL_INTERIOR_HPP
#define THREEPP_EXAMPLE_WATERMILL_INTERIOR_HPP

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleSprites.hpp"

#include "GearParts.hpp"
#include "InteriorLayout.hpp"
#include "InteriorParts.hpp"
#include "InteriorTextures.hpp"
#include "Masonry.hpp"
#include "Materials.hpp"
#include "Mill.hpp"
#include "Site.hpp"

#include <chrono>
#include <iostream>
#include <map>

namespace mill {

    using threepp::Vector3;

    namespace inside {

        using threepp::Vector3;

        // dust on indoor timber: thick on what faces up, a film on the sides; multiplies the worn wood's brown toward a
        // pale grey
        inline V3 dusty(const Vector3& p, const Vector3& n, const V3& c) {
            const float up = smooth(0.25f, 0.9f, n.y);
            const float patch = clamp01(0.55f + 0.8f * fbm3(p.x * 1.3f, p.y * 1.3f, p.z * 1.3f, 3, 91u));
            V3 col = mix(c, c * V3(1.45f, 1.62f, 1.95f), 0.78f * up * patch);
            col = mix(col, c * V3(1.12f, 1.16f, 1.26f), 0.35f * (1.f - up) * patch);
            // grime: what stands low on a floor is darker at its foot
            const float floorY = p.y < site::stoneFloor ? site::basementFloor : p.y < site::binFloor ? site::stoneFloor : p.y < site::garnerFloor ? site::binFloor : site::garnerFloor;
            col = col * (0.82f + 0.18f * smooth(0.f, 0.35f, p.y - floorY));
            return col;
        }

        // the floors: dust everywhere, thickest along the walls and in the corners, a few darker patches where meal
        // was spilled and went damp
        inline V3 dustyFloor(const Vector3& p, const Vector3& n, const V3& c) {
            if (n.y < 0.5f) return dusty(p, n, c);
            const float wall = std::min({p.x - 6.2f, 12.3f - p.x, p.z + 4.3f, 4.3f - p.z});
            const float patch = clamp01(0.5f + 0.9f * fbm2(p.x * 0.9f, p.z * 0.9f, 4, 92u));
            const float d = clamp01(0.35f + 0.45f * smooth(1.6f, 0.1f, wall) + 0.35f * (patch - 0.5f));
            V3 col = mix(c, c * V3(1.5f, 1.68f, 2.05f), d);
            const float stain = smooth(0.35f, 0.6f, fbm2(p.x * 0.5f + 3.f, p.z * 0.5f, 3, 93u));
            col = col * (1.f - 0.25f * stain);
            return col;
        }

        inline WoodLook woodLook(const V3& tint, uint32_t seed) {
            WoodLook l;
            l.tint = tint;
            l.seed = seed;
            l.weather = dusty;
            return l;
        }

    }// namespace inside

    class Interior {
    public:
        explicit Interior(const Materials& materials): root_(threepp::Group::create()) {
            using namespace inside;
            const auto t0 = std::chrono::steady_clock::now();
            buildMaterials(materials);
            buildFloors(materials);
            buildPostsAndStairs(materials);
            buildDrive(materials);
            buildStones(materials);
            buildHead(materials);
            buildBins(materials);
            buildLeftBehind(materials);
            buildMotes();
            flush(root_, static_);
            size_t tris = 0;
            root_->traverseType<threepp::Mesh>([&](threepp::Mesh& mesh) {
                if (const auto idx = mesh.geometry()->getIndex()) tris += idx->count() / 3;
            });
            std::cout << "interior " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s, " << tris << " tris" << std::endl;
        }

        [[nodiscard]] std::shared_ptr<threepp::Group> root() const { return root_; }

        // pose every moving part from the waterwheel's angle (radians about +x) and speed (rad/s)
        void update(float wheelAngle, float wheelSpeed, float dt) {
            using namespace inside;
            (void) wheelSpeed;
            if (first_) {
                theta_ = wheelAngle;
                prevAngle_ = wheelAngle;
                first_ = false;
            }
            // the wheel's angle, unwrapped and kept in double so the fast shafts stay in mesh for hours
            double d = static_cast<double>(wheelAngle) - static_cast<double>(prevAngle_);
            const double twoPiD = 6.283185307179586;
            d -= twoPiD * std::floor((d + 0.5 * twoPiD) / twoPiD);
            theta_ += d;
            prevAngle_ = wheelAngle;
            const auto wrap = [&](double a) { return static_cast<float>(a - twoPiD * std::floor(a / twoPiD)); };
            const double piD = 3.141592653589793;

            // the pit wheel turns with the waterwheel; its cog 24 is at the bottom (in mesh) at angle 0
            shaft_->rotation.x = wheelAngle;
            // the wallower: a gap between staves faces the pit wheel (west, psi = pi) whenever a cog is at the bottom;
            // the bottom cogs move north (-z), so the upright turns clockwise seen from above
            const double phiU = piD - piD / wallowerStaves - theta_ * static_cast<double>(pitCogs) / wallowerStaves;
            upright_->rotation.y = wrap(phiU);
            // the stone nuts: a gap faces the spur wheel whenever one of its cogs points at the nut
            for (int i = 0; i < 2; ++i) {
                const double beta = stoneAngle[i];
                const double phiN = beta + piD - piD / nutTeeth - (phiU - beta) * static_cast<double>(spurCogs) / nutTeeth;
                nut_[i]->rotation.y = wrap(phiN);
                // the damsel knocks the shoe four times a turn
                const double knock = std::abs(std::sin(2.0 * phiN));
                shoe_[i]->rotation.z = static_cast<float>(0.012 * knock * knock * knock);
            }
            // the lay shaft's lantern: a gap at its bottom when a crown cog points east
            const double chi = piD - piD / layStaves + static_cast<double>(crownCogs) / layStaves * phiU;
            lay_->rotation.x = wrap(chi);
            updateMotes(dt);
        }

    private:
        using MatPtr = std::shared_ptr<threepp::Material>;
        using Builds = std::map<MatPtr, std::pair<MeshBuild, bool>>;// bool: vertex colours

        std::shared_ptr<threepp::Group> root_;
        std::shared_ptr<threepp::Group> shaft_ = threepp::Group::create();
        std::shared_ptr<threepp::Group> upright_ = threepp::Group::create();
        std::shared_ptr<threepp::Group> lay_ = threepp::Group::create();
        std::array<std::shared_ptr<threepp::Group>, 2> nut_{threepp::Group::create(), threepp::Group::create()};
        std::array<std::shared_ptr<threepp::Group>, 2> shoe_{threepp::Group::create(), threepp::Group::create()};
        Builds static_, shaftB_, uprightB_, layB_;
        std::array<Builds, 2> nutB_, shoeB_;

        std::shared_ptr<threepp::MeshStandardMaterial> burrFace_, burrPlain_, sacking_, cobweb_, leaves_, leather_, rope_;
        MatPtr interior_, boards_, iron_, black_, masonry_;

        double theta_ = 0;
        float prevAngle_ = 0;
        bool first_ = true;

        // dust in the light from the windows
        std::shared_ptr<threepp::ParticleSprites> motes_;
        struct Mote {
            Vector3 p, v;
            float size, alpha, phase;
        };
        std::vector<Mote> moteState_;
        float moteTime_ = 0;

        MeshBuild& b(Builds& set, const MatPtr& m, bool colours = true) {
            auto& e = set[m];
            e.second = colours;
            return e.first;
        }

        static void flush(const std::shared_ptr<threepp::Group>& g, Builds& set) {
            for (auto& [mat, e] : set) {
                if (e.first.index.empty()) continue;
                auto mesh = e.first.mesh(mat, e.second);
                if (mat->transparent) mesh->castShadow = false;
                g->add(mesh);
            }
            set.clear();
        }

        // ---- materials ------------------------------------------------------------------------------------------------
        void buildMaterials(const Materials& m) {
            using namespace threepp;
            interior_ = m.interior;
            boards_ = m.floorBoards;
            iron_ = m.iron;
            black_ = m.blackIron;
            masonry_ = m.masonry;
            const auto wear = [](const TexSet& set, float normalScale) {
                auto mat = set.material(normalScale);
                mat->vertexColors = true;
                return mat;
            };
            burrFace_ = wear(burrStoneTexture(201, inside::stoneRadius, true), 1.2f);
            burrPlain_ = wear(burrStoneTexture(202, inside::stoneRadius, false), 1.f);
            sacking_ = wear(sackingTexture(203), 1.f);
            cobweb_ = wear(cobwebTexture(204), 0.f);
            cobweb_->transparent = true;
            cobweb_->depthWrite = false;
            cobweb_->side = Side::Double;
            cobweb_->opacity = 0.42f;
            cobweb_->metalnessMap = nullptr;
            cobweb_->metalness = 0.f;
            leaves_ = wear(deadLeafAtlas(205), 1.f);
            leaves_->alphaTest = 0.5f;
            leaves_->side = Side::Double;
            leaves_->metalnessMap = nullptr;
            leaves_->metalness = 0.f;
            leather_ = MeshStandardMaterial::create();
            leather_->color.setRGB(0.075f, 0.05f, 0.035f);
            leather_->roughness = 0.75f;
            leather_->vertexColors = true;
            leather_->side = Side::Double;
            rope_ = MeshStandardMaterial::create();
            rope_->color.setRGB(0.16f, 0.125f, 0.085f);
            rope_->roughness = 0.95f;
            rope_->vertexColors = true;
        }

        // ---- floors ---------------------------------------------------------------------------------------------------
        struct FloorSpec {
            float y;
            float x0, x1, z0, z1;     // the boards
            float jx0, jx1, jz0, jz1; // the joists' spread and length
            float bx0, bx1;           // the summer beams' length
            bool ledgers;
            std::vector<inside::Rect> holes, joistOnly, boardOnly;
            uint32_t seed;
        };

        void buildFloor(const FloorSpec& F) {
            using namespace inside;
            Rng rng(F.seed);
            MeshBuild& boards = b(static_, boards_);
            MeshBuild& frame = b(static_, interior_);
            // the boards, along x, cut round the holes; butt joints here and there
            float z = F.z0;
            while (z < F.z1 - 0.02f) {
                float bw = rng.range(0.15f, 0.27f);
                if (F.z1 - (z + bw) < 0.1f) bw = F.z1 - z;
                const float zc = z + 0.5f * bw;
                std::vector<std::pair<float, float>> spans{{F.x0, F.x1}};
                std::vector<Rect> boardCuts = F.holes;
                boardCuts.insert(boardCuts.end(), F.boardOnly.begin(), F.boardOnly.end());
                for (const auto& h : boardCuts) {
                    // a board mostly over the hole is cut by it
                    const float over = std::min(z + bw, h.z1) - std::max(z, h.z0);
                    if (over < 0.3f * bw) continue;
                    std::vector<std::pair<float, float>> next;
                    for (auto [a, c] : spans) {
                        if (h.x1 <= a || h.x0 >= c) {
                            next.emplace_back(a, c);
                            continue;
                        }
                        if (h.x0 > a) next.emplace_back(a, h.x0);
                        if (h.x1 < c) next.emplace_back(h.x1, c);
                    }
                    spans = next;
                }
                V3 tint = V3(1, 1, 1) * rng.range(0.82f, 1.08f);
                if (rng() < 0.07f) tint = V3(1.12f, 1.0f, 0.86f);// a later replacement
                if (rng() < 0.06f) tint = tint * 0.72f;          // one that took the damp
                for (auto [a, c] : spans) {
                    // butt joints
                    float x = a;
                    while (x < c - 0.02f) {
                        float len = rng.range(2.4f, 4.4f);
                        if (c - (x + len) < 0.6f) len = c - x;
                        WoodLook look;
                        look.tint = tint * rng.range(0.95f, 1.04f);
                        look.seed = rng.next();
                        look.twist = rng.range(-0.01f, 0.01f);
                        look.weather = dustyFloor;
                        appendBeam(boards, Vector3(x + 0.002f, F.y - 0.5f * boardT, zc), Vector3(x + len - 0.002f, F.y - 0.5f * boardT, zc), Vector3(0, 1, 0),
                                   bw - rng.range(0.003f, 0.008f), boardT, look);
                        x += len;
                    }
                }
                z += bw;
            }
            // joists along z, cut round the holes, with headers across the cuts
            const float jy = F.y - boardT - 0.5f * joistD;
            const int nj = static_cast<int>(std::round((F.jx1 - F.jx0) / 0.48f));
            std::vector<float> jx;
            for (int k = 0; k <= nj; ++k) jx.push_back(F.jx0 + (F.jx1 - F.jx0) * static_cast<float>(k) / static_cast<float>(nj));
            std::vector<Rect> cuts = F.holes;
            cuts.insert(cuts.end(), F.joistOnly.begin(), F.joistOnly.end());
            for (float x : jx) {
                std::vector<std::pair<float, float>> spans{{F.jz0, F.jz1}};
                for (const auto& h : cuts) {
                    if (!h.overlapsX(x - 0.5f * joistW, x + 0.5f * joistW)) continue;
                    std::vector<std::pair<float, float>> next;
                    for (auto [a, c] : spans) {
                        const float h0 = h.z0 - 0.04f, h1 = h.z1 + 0.04f;
                        if (h1 <= a || h0 >= c) {
                            next.emplace_back(a, c);
                            continue;
                        }
                        if (h0 > a) next.emplace_back(a, h0);
                        if (h1 < c) next.emplace_back(h1, c);
                    }
                    spans = next;
                }
                for (auto [a, c] : spans) {
                    if (c - a < 0.1f) continue;
                    WoodLook look = woodLook(V3(0.7f, 0.63f, 0.56f) * rng.range(0.85f, 1.05f), rng.next());
                    look.sag = 0.006f * (c - a) / 4.f;
                    appendBeam(frame, Vector3(x, jy, a), Vector3(x, jy, c), Vector3(0, 1, 0), joistW, joistD, look);
                }
            }
            for (const auto& h : cuts) {
                // headers between the joists either side of the hole
                float xa = F.jx0, xb = F.jx1;
                for (float x : jx) {
                    if (x + 0.5f * joistW <= h.x0) xa = x;
                }
                for (auto it = jx.rbegin(); it != jx.rend(); ++it) {
                    if (*it - 0.5f * joistW >= h.x1) xb = *it;
                }
                bool any = false;
                for (float x : jx) any |= h.overlapsX(x - 0.5f * joistW, x + 0.5f * joistW);
                if (!any) continue;
                for (float hz : {h.z0 - 0.04f - 0.5f * joistW, h.z1 + 0.04f + 0.5f * joistW}) {
                    if (hz < F.jz0 + 0.05f || hz > F.jz1 - 0.05f) continue;
                    appendBeam(frame, Vector3(xa + 0.5f * joistW, jy, hz), Vector3(xb - 0.5f * joistW, jy, hz), Vector3(0, 1, 0), joistW, joistD,
                               woodLook(V3(0.68f, 0.61f, 0.54f), rng.next()));
                }
            }
            // the summer beams across the mill, and the ledgers on the gables
            const float by = F.y - boardT - joistD - 0.5f * beamD;
            for (float bz : {-beamZ, beamZ}) {
                WoodLook look = woodLook(V3(0.62f, 0.56f, 0.5f), rng.next());
                look.sag = 0.012f;
                look.twist = 0.01f;
                appendBeam(frame, Vector3(F.bx0, by, bz), Vector3(F.bx1, by, bz), Vector3(0, 1, 0), beamW, beamD, look);
            }
            if (F.ledgers) {
                for (float lz : {-4.2f, 4.2f}) {
                    std::vector<std::pair<float, float>> spans{{6.3f, 12.2f}};
                    for (const auto& h : F.holes) {
                        if (lz < h.z0 - 0.1f || lz > h.z1 + 0.1f) continue;
                        std::vector<std::pair<float, float>> next;
                        for (auto [a, c] : spans) {
                            if (h.x1 <= a || h.x0 >= c) {
                                next.emplace_back(a, c);
                                continue;
                            }
                            if (h.x0 > a) next.emplace_back(a, h.x0);
                            if (h.x1 < c) next.emplace_back(h.x1, c);
                        }
                        spans = next;
                    }
                    for (auto [a, c] : spans)
                        appendBeam(frame, Vector3(a, jy - 0.5f * joistD - 0.08f, lz), Vector3(c, jy - 0.5f * joistD - 0.08f, lz), Vector3(0, 1, 0), 0.14f, 0.16f,
                                   woodLook(V3(0.66f, 0.6f, 0.54f), rng.next()));
                }
            }
        }

        void buildFloors(const Materials&) {
            using namespace inside;
            using namespace site;
            const Rect upright = uprightHole(), hatch = sackHatch();
            // the stone floor: its joists bear on the foundation, its beams go into the walls' stone
            // the pit wheel's top rises between the joists, and the upright beside it: one trimmed opening for both
            FloorSpec stone{stoneFloor, 6.3f, 12.2f, -4.2f, 4.2f, 6.75f, 11.75f, -4.2f, 4.2f, 6.3f, 12.2f, false, {hatch, basementWell()},
                            {{pitX - 0.15f, upright.x1 + 0.03f, axleZ - 0.75f, axleZ + 0.75f}}, {upright}, 301};
            buildFloor(stone);
            // sleepers on the foundation's tops carry the boards' ends
            MeshBuild& frame = b(static_, interior_);
            for (float x : {6.45f, 12.05f})
                appendBeam(frame, Vector3(x, sillTop + 0.0325f, -4.2f), Vector3(x, sillTop + 0.0325f, 4.2f), Vector3(0, 1, 0), 0.3f, 0.065f, woodLook(V3(0.6f, 0.55f, 0.5f), 311));
            FloorSpec bin{binFloor, 6.1f, 12.4f, -4.38f, 4.38f, 6.45f, 12.1f, -4.27f, 4.27f, 6.3f, 12.2f, true, {upright, hatch, binWell()}, {}, {}, 302};
            buildFloor(bin);
            FloorSpec garner{garnerFloor, 6.1f, 12.4f, -4.38f, 4.38f, 6.45f, 12.1f, -4.27f, 4.27f, 6.3f, 12.2f, true, {upright, hatch, garnerWell()}, {}, {}, 303};
            buildFloor(garner);
            // the sack traps: a raised curb round each hatch and two flaps, closed
            for (float y : {stoneFloor, binFloor, garnerFloor}) {
                const Rect h = hatch;
                MeshBuild& boards = b(static_, boards_);
                for (int s = 0; s < 2; ++s) {
                    const float zc = s ? h.z1 + 0.03f : h.z0 - 0.03f;
                    appendBeam(frame, Vector3(h.x0 - 0.06f, y + 0.03f, zc), Vector3(h.x1 + 0.06f, y + 0.03f, zc), Vector3(0, 1, 0), 0.06f, 0.06f, woodLook(V3(0.7f, 0.64f, 0.56f), 320 + s));
                    const float xc = s ? h.x1 + 0.03f : h.x0 - 0.03f;
                    appendBeam(frame, Vector3(xc, y + 0.03f, h.z0), Vector3(xc, y + 0.03f, h.z1), Vector3(0, 1, 0), 0.06f, 0.06f, woodLook(V3(0.7f, 0.64f, 0.56f), 322 + s));
                    // a flap: three boards across, hinged at the curb
                    const float fx0 = s ? 0.5f * (h.x0 + h.x1) + 0.004f : h.x0 + 0.004f, fx1 = s ? h.x1 - 0.004f : 0.5f * (h.x0 + h.x1) - 0.004f;
                    for (int k = 0; k < 3; ++k) {
                        const float zz = h.z0 + (h.z1 - h.z0) * (static_cast<float>(k) + 0.5f) / 3.f;
                        WoodLook look = woodLook(V3(0.95f, 0.88f, 0.8f), 330 + static_cast<uint32_t>(k + 3 * s + 7 * static_cast<int>(y)));
                        look.weather = dustyFloor;
                        appendBeam(boards, Vector3(fx0, y - 0.018f, zz), Vector3(fx1, y - 0.018f, zz), Vector3(0, 1, 0), (h.z1 - h.z0) / 3.f - 0.006f, 0.034f, look);
                    }
                    // the battens under it
                    appendBeam(frame, Vector3(0.5f * (fx0 + fx1), y - 0.05f, h.z0 + 0.04f), Vector3(0.5f * (fx0 + fx1), y - 0.05f, h.z1 - 0.04f), Vector3(0, 1, 0), 0.08f, 0.03f,
                               woodLook(V3(0.7f, 0.64f, 0.56f), 340 + s));
                }
            }
        }

        // ---- posts, braces, stairs, rails ----------------------------------------------------------------------------
        void appendPost(MeshBuild& frame, float x, float z, float y0, float y1, uint32_t seed, bool braces) {
            using namespace inside;
            WoodLook look = woodLook(V3(0.62f, 0.56f, 0.5f), seed);
            appendBeam(frame, Vector3(x, y0, z), Vector3(x, y1, z), Vector3(1, 0, 0), postW, postW, look);
            if (!braces) return;
            // knee braces up to the beam, along it both ways
            for (float s : {-1.f, 1.f}) {
                appendBeam(frame, Vector3(x + s * 0.08f, y1 - 0.62f, z), Vector3(x + s * 0.62f, y1 - 0.06f, z), Vector3(0, 0, 1), 0.1f, 0.12f, woodLook(V3(0.6f, 0.55f, 0.5f), seed + 5 + (s > 0)));
            }
        }

        // a steep mill stair: two stringers and open treads from `bottom` to `top` (the middles of its foot and head,
        // at floor level), `width` across; a handrail on the side `railSide` (+1 / -1 across, 0 none)
        void appendStair(const Vector3& bottom, const Vector3& top, float width, float railSide, uint32_t seed) {
            using namespace inside;
            MeshBuild& frame = b(static_, interior_);
            MeshBuild& treads = b(static_, boards_);
            Vector3 run = top - bottom;
            run.y = 0;
            const float runLen = run.length();
            run.normalize();
            const Vector3 across = Vector3(0, 1, 0).cross(run).normalize();
            const float rise = top.y - bottom.y;
            const int steps = static_cast<int>(std::round(rise / 0.215f));
            Vector3 slope = top - bottom;
            slope.normalize();
            const Vector3 slopeUp = across.clone().cross(slope).normalize();
            Rng rng(seed);
            for (float s : {-1.f, 1.f}) {
                const Vector3 off = across * (s * (0.5f * width - 0.025f));
                const Vector3 a = bottom + off + slopeUp * 0.06f - slope * 0.05f;
                const Vector3 c = top + off + slopeUp * 0.0f - Vector3(0, 0.14f, 0);
                appendBeam(frame, a, c, slopeUp, 0.05f, 0.25f, woodLook(V3(0.68f, 0.61f, 0.54f), rng.next()));
            }
            for (int k = 1; k < steps; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(steps);
                const Vector3 p = bottom + run * (t * runLen) + Vector3(0, t * rise - 0.02f, 0);
                WoodLook look = woodLook(V3(0.72f, 0.66f, 0.58f) * rng.range(0.9f, 1.05f), rng.next());
                // the treads are worn hollow in the middle
                look.sag = 0.006f;
                appendBeam(treads, p - across * (0.5f * width - 0.05f), p + across * (0.5f * width - 0.05f), Vector3(0, 1, 0), 0.24f, 0.04f, look);
            }
            if (railSide != 0.f) {
                const Vector3 off = across * (railSide * (0.5f * width + 0.03f));
                const Vector3 a = bottom + off + run * 0.1f, c = top + off - run * 0.05f;
                appendBeam(frame, a, a + Vector3(0, 0.95f, 0), run, 0.06f, 0.06f, woodLook(V3(0.66f, 0.6f, 0.54f), rng.next()));
                appendBeam(frame, a + Vector3(0, 0.9f, 0), c + Vector3(0, 0.9f, 0), Vector3(0, 1, 0), 0.05f, 0.06f, woodLook(V3(0.75f, 0.66f, 0.56f), rng.next()));
            }
        }

        void buildPostsAndStairs(const Materials&) {
            using namespace inside;
            using namespace site;
            MeshBuild& frame = b(static_, interior_);
            const float stoneBeam = stoneFloor - boardT - joistD - beamD, binBeam = binFloor - boardT - joistD - beamD, garnerBeam = garnerFloor - boardT - joistD - beamD;
            uint32_t seed = 400;
            for (float z : {-beamZ, beamZ}) {
                appendPost(frame, midPostX, z, basementFloor, stoneBeam, seed++, true);
                appendPost(frame, midPostX, z, stoneFloor, binBeam, seed++, true);
                appendPost(frame, midPostX, z, binFloor, garnerBeam, seed++, true);
                for (float x : {6.3f + 0.5f * postW, 12.2f - 0.5f * postW}) {
                    appendPost(frame, x, z, stoneFloor, binBeam, seed++, false);
                    appendPost(frame, x, z, binFloor, garnerBeam, seed++, false);
                }
            }
            // stone pads under the basement posts
            MeshBuild& stones = b(static_, masonry_);
            for (float z : {-beamZ, beamZ}) {
                Frame f{Vector3(midPostX, basementFloor + 0.06f, z), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};
                StoneLook look;
                look.tint = V3(0.7f, 0.66f, 0.6f);
                look.seed = static_cast<uint32_t>(450 + (z > 0));
                appendStone(stones, f, 0.22f, 0.07f, 0.22f, look);
            }
            // the basement stair along the south wall, rising west; stair 1 along the north wall rising east; stair 2
            // beside it rising west
            appendStair(Vector3(11.6f, basementFloor, -3.5f), Vector3(9.35f, stoneFloor, -3.5f), 0.7f, 1.f, 460);
            appendStair(Vector3(8.6f, stoneFloor, 3.8f), Vector3(11.05f, binFloor, 3.8f), 0.7f, 1.f, 470);
            appendStair(Vector3(10.7f, binFloor, -3.8f), Vector3(8.25f, garnerFloor, -3.8f), 0.68f, 1.f, 480);
            // a rail round the basement stair's well on the stone floor (its open side faces south, its head west)
            const Rect w = basementWell();
            const WoodLook rail = woodLook(V3(0.7f, 0.63f, 0.55f), 490);
            for (float x : {w.x0 + 0.45f, 0.5f * (w.x0 + w.x1) + 0.2f, w.x1 - 0.04f}) appendBeam(frame, Vector3(x, stoneFloor, w.z1 + 0.03f), Vector3(x, stoneFloor + 0.92f, w.z1 + 0.03f), Vector3(1, 0, 0), 0.06f, 0.06f, rail);
            appendBeam(frame, Vector3(w.x0 + 0.42f, stoneFloor + 0.92f, w.z1 + 0.03f), Vector3(w.x1, stoneFloor + 0.92f, w.z1 + 0.03f), Vector3(0, 1, 0), 0.06f, 0.05f, rail);
            appendBeam(frame, Vector3(w.x0 + 0.45f, stoneFloor + 0.48f, w.z1 + 0.03f), Vector3(w.x1, stoneFloor + 0.48f, w.z1 + 0.03f), Vector3(0, 1, 0), 0.03f, 0.08f, rail);
            appendBeam(frame, Vector3(w.x1 - 0.04f, stoneFloor + 0.92f, w.z1 + 0.03f), Vector3(w.x1 - 0.04f, stoneFloor + 0.92f, w.z0), Vector3(0, 1, 0), 0.06f, 0.05f, rail);
            appendBeam(frame, Vector3(w.x1 - 0.04f, stoneFloor, w.z0 + 0.05f), Vector3(w.x1 - 0.04f, stoneFloor + 0.92f, w.z0 + 0.05f), Vector3(1, 0, 0), 0.06f, 0.06f, rail);
        }

        // ---- the drive: the wheel's shaft, the pit wheel, the wallower, the upright, the spur wheel ------------------------
        void buildDrive(const Materials&) {
            using namespace inside;
            using namespace site;
            using gear::Rot;
            // -- the waterwheel's shaft inside, its bearing and pier (the shaft's group sits on the axle)
            shaft_->position.set(0, axleY, axleZ);
            root_->add(shaft_);
            {
                MeshBuild& dark = b(shaftB_, black_, false);
                appendCylinder(dark, Vector3(millWest + 0.19f, 0, 0), Vector3(shaftEndX - 0.02f, 0, 0), 0.11f, 0.11f, 20, WoodLook{});
                appendCylinder(dark, Vector3(shaftEndX - 0.06f, 0, 0), Vector3(shaftEndX, 0, 0), 0.135f, 0.13f, 20, WoodLook{});// the end collar
                // the pit wheel: a cast iron mortise wheel, hornbeam cogs in its east face
                const Rot R = gear::aboutX(Vector3(pitX, 0, 0));
                MeshBuild& ironB = b(shaftB_, iron_);
                MeshBuild& wood = b(shaftB_, interior_);
                const WoodLook cast = woodLook(V3(0.4f, 0.4f, 0.42f), 501);
                for (int k = 0; k < 6; ++k) {
                    const float g0 = gear::twoPi * static_cast<float>(k) / 6.f;
                    gear::appendArc(ironB, R, pitRim0, pitRim1, -pitRimHalf, pitRimHalf, g0 + 0.004f, g0 + gear::twoPi / 6.f - 0.004f, 16, cast);
                    // the bolted flanges at the joints, on the back
                    gear::appendCog(ironB, R, g0, pitRim0 - 0.06f, pitRim0 + 0.02f, 0.09f, -pitRimHalf, pitRimHalf - 0.02f, cast);
                }
                // six arms, a cross section like a T, and the hub on the shaft
                for (int k = 0; k < 6; ++k) {
                    const float g = gear::twoPi * (static_cast<float>(k) + 0.5f) / 6.f;
                    gear::appendCog(ironB, R, g, 0.19f, pitRim0 + 0.01f, 0.045f, -0.06f, 0.06f, cast);
                    gear::appendCog(ironB, R, g, 0.21f, pitRim0 + 0.01f, 0.12f, -0.07f, -0.045f, cast);
                }
                gear::appendRound(ironB, R, 0.21f, -0.16f, 0.14f, 24, cast);
                gear::appendRound(ironB, R, 0.24f, -0.16f, -0.1f, 24, cast);
                // the cogs (cog 24 is at the bottom at angle 0), with their tails pinned behind the rim
                for (int k = 0; k < pitCogs; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(pitCogs);
                    WoodLook cog = woodLook(V3(0.82f, 0.72f, 0.6f) * (0.9f + 0.18f * unit(hash32(510u + k))), 510u + static_cast<uint32_t>(k));
                    gear::appendCog(wood, R, g, pitR - pitCogHalf, pitR + pitCogHalf, 0.046f, pitRimHalf, pitRimHalf + pitCogOut, cog);
                    gear::appendCog(wood, R, g, pitR - pitCogHalf + 0.01f, pitR + pitCogHalf - 0.01f, 0.04f, -pitRimHalf - 0.03f, -pitRimHalf + 0.002f, cog);
                }
            }
            flush(shaft_, shaftB_);
            // the bearing: a plummer block on an oak bearer on a stone pier against the wall
            {
                MeshBuild& stones = b(static_, masonry_);
                Rng rng(520);
                float y = basementFloor;
                int course = 0;
                while (y < axleY - 0.32f) {
                    const float h = std::min(rng.range(0.24f, 0.36f), axleY - 0.32f - y);
                    Frame f{Vector3(bearingX - 0.03f, y + 0.5f * h, axleZ + rng.range(-0.02f, 0.02f)), Vector3(0, 0, 1), Vector3(0, 1, 0), Vector3(1, 0, 0)};
                    StoneLook look;
                    look.tint = V3(0.78f, 0.74f, 0.68f) * rng.range(0.8f, 1.f);
                    look.seed = rng.next();
                    look.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                        return c * (0.75f + 0.25f * smooth(0.9f, 1.8f, p.y)) * (1.f + 0.25f * smooth(0.5f, 0.9f, n.y));
                    };
                    appendStone(stones, f, (course % 2 ? 0.34f : 0.3f) - 0.01f, 0.5f * h - 0.008f, 0.21f, look);
                    y += h;
                    ++course;
                }
                MeshBuild& frame = b(static_, interior_);
                appendBeam(frame, Vector3(bearingX - 0.2f, axleY - 0.28f, axleZ), Vector3(bearingX + 0.18f, axleY - 0.28f, axleZ), Vector3(0, 1, 0), 0.5f, 0.08f,
                           woodLook(V3(0.45f, 0.4f, 0.35f), 521));
                MeshBuild& dark = b(static_, black_, false);
                appendBeam(dark, Vector3(bearingX - 0.13f, axleY - 0.18f, axleZ), Vector3(bearingX + 0.13f, axleY - 0.18f, axleZ), Vector3(0, 1, 0), 0.42f, 0.12f, WoodLook{});
                appendCylinder(dark, Vector3(bearingX - 0.12f, 0, 0) + Vector3(0, axleY, axleZ), Vector3(bearingX + 0.12f, axleY, axleZ), 0.165f, 0.165f, 20, WoodLook{});
                for (float s : {-1.f, 1.f}) appendCylinder(dark, Vector3(bearingX, axleY - 0.12f, axleZ + s * 0.17f), Vector3(bearingX, axleY + 0.2f, axleZ + s * 0.17f), 0.022f, 0.022f, 6, WoodLook{});
            }

            // -- the upright shaft and what it carries
            upright_->position.set(uprightX, 0, uprightZ);
            root_->add(upright_);
            const Rot U = gear::aboutY(Vector3(0, 0, 0));
            {
                MeshBuild& wood = b(uprightB_, interior_);
                MeshBuild& ironB = b(uprightB_, iron_);
                MeshBuild& dark = b(uprightB_, black_, false);
                gear::appendPrism(wood, U, uprightR, uprightFoot, uprightHead, 8, woodLook(V3(0.52f, 0.45f, 0.38f), 530));
                const WoodLook band = woodLook(V3(0.33f, 0.34f, 0.36f), 531);
                for (float y : {1.3f, 1.95f, 2.66f, 3.2f, 4.25f, 4.62f, 5.85f, 6.3f, 6.85f, 7.2f, 9.4f, 9.7f, 10.0f})
                    gear::appendPrism(ironB, U, uprightR + 0.007f, y - 0.03f, y + 0.03f, 8, band);
                appendCylinder(dark, Vector3(0, uprightFoot - 0.1f, 0), Vector3(0, uprightFoot + 0.02f, 0), 0.06f, 0.06f, 12, WoodLook{});
                appendCylinder(dark, Vector3(0, uprightHead - 0.02f, 0), Vector3(0, uprightHead + 0.12f, 0), 0.06f, 0.06f, 12, WoodLook{});
                // the wallower: two elm heads bound with iron, fourteen iron staves
                const WoodLook head = woodLook(V3(0.55f, 0.47f, 0.4f), 532);
                gear::appendDisc(wood, U, 0.f, wallowerHeadR, wallowerY0 - wallowerHead, wallowerY0, 48, head);
                gear::appendDisc(wood, U, 0.f, wallowerHeadR, wallowerY1, wallowerY1 + wallowerHead, 48, head);
                for (float y0 : {wallowerY0 - wallowerHead, wallowerY1}) gear::appendArc(ironB, U, wallowerHeadR - 0.004f, wallowerHeadR + 0.008f, y0 + 0.008f, y0 + wallowerHead - 0.008f, 0.f, gear::twoPi, 48, band, false);
                for (int k = 0; k < wallowerStaves; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(wallowerStaves);
                    const Vector3 p = U.dir(g) * wallowerR;
                    appendCylinder(ironB, Vector3(p.x, wallowerY0 - 0.01f, p.z), Vector3(p.x, wallowerY1 + 0.01f, p.z), wallowerStave, wallowerStave, 10, band, false);
                }
                // the great spur wheel: clasp arms, eight felloes, 84 cogs
                const WoodLook felloe = woodLook(V3(0.58f, 0.5f, 0.42f), 533);
                gear::appendFelloes(wood, U, spurRim0, spurRim1, spurY0, spurY1, 8, 0.2f, felloe);
                for (int k = 0; k < spurCogs; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(spurCogs);
                    WoodLook cog = woodLook(V3(0.82f, 0.72f, 0.6f) * (0.9f + 0.18f * unit(hash32(540u + k))), 540u + static_cast<uint32_t>(k));
                    gear::appendTaperedCog(wood, U, g, spurRim1 - 0.004f, spurR + 0.032f, 0.044f, 0.028f, spurY0 + 0.012f, spurY1 - 0.012f, cog);
                    gear::appendCog(wood, U, g, spurRim0 - 0.025f, spurRim0 + 0.004f, 0.036f, spurY0 + 0.03f, spurY1 - 0.03f, cog);
                }
                clasp(wood, ironB, U, spurRim0, spurY0, spurY1, 534);
                // the crown wheel at the head: cogs facing up
                gear::appendFelloes(wood, U, crownR - crownCogHalf - 0.03f, crownR + crownCogHalf + 0.03f, crownFace - 0.12f, crownFace, 6, 0.1f, felloe);
                for (int k = 0; k < crownCogs; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(crownCogs);
                    WoodLook cog = woodLook(V3(0.82f, 0.72f, 0.6f) * (0.9f + 0.18f * unit(hash32(600u + k))), 600u + static_cast<uint32_t>(k));
                    gear::appendCog(wood, U, g, crownR - crownCogHalf, crownR + crownCogHalf, 0.032f, crownFace - 0.005f, crownFace + crownCogOut, cog);
                }
                clasp(wood, ironB, U, crownR - crownCogHalf - 0.03f, crownFace - 0.12f, crownFace, 535);
            }
            flush(upright_, uprightB_);
            // the footstep: a dressed stone with a brass step in it
            {
                MeshBuild& stones = b(static_, masonry_);
                Frame f{Vector3(uprightX, basementFloor + 0.05f, uprightZ), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};
                StoneLook look;
                look.tint = V3(0.72f, 0.68f, 0.62f);
                look.seed = 560;
                appendStone(stones, f, 0.32f, 0.05f, 0.32f, look);
                MeshBuild& dark = b(static_, black_, false);
                appendBeam(dark, Vector3(uprightX - 0.12f, basementFloor + 0.1f + 0.02f, uprightZ), Vector3(uprightX + 0.12f, basementFloor + 0.1f + 0.02f, uprightZ), Vector3(0, 1, 0), 0.24f, 0.04f, WoodLook{});
                // collars where the upright passes the floors, on short bearers
                MeshBuild& frame = b(static_, interior_);
                for (float y : {site::stoneFloor, site::binFloor, site::garnerFloor}) {
                    const float by = y - boardT - 0.06f;
                    for (float s : {-1.f, 1.f})
                        appendBeam(frame, Vector3(uprightX - 0.3f, by, uprightZ + s * 0.22f), Vector3(uprightX + 0.3f, by, uprightZ + s * 0.22f), Vector3(0, 1, 0), 0.1f, 0.12f,
                                   woodLook(V3(0.6f, 0.54f, 0.48f), static_cast<uint32_t>(570 + y)));
                }
            }
        }

        // clasp arms: two pairs of beams, halved into each other, gripping the octagonal shaft between them, from rim to
        // rim; iron wedges and plates where they cross
        void clasp(MeshBuild& wood, MeshBuild& ironB, const gear::Rot& U, float rimInner, float y0, float y1, uint32_t seed) {
            using namespace inside;
            const float off = uprightR + 0.065f, w = 0.13f;
            const float chord = std::sqrt(rimInner * rimInner - off * off) + 0.01f;
            const float ym = 0.5f * (y0 + y1);
            for (int pair = 0; pair < 2; ++pair) {
                const float g = pair * 0.5f * inside::pi;
                const Vector3 along = U.dir(g), perp = U.dir(g + 0.5f * inside::pi);
                for (float s : {-1.f, 1.f}) {
                    WoodLook look = woodLook(V3(0.56f, 0.48f, 0.41f), seed + static_cast<uint32_t>(pair * 2 + (s > 0)));
                    appendBeam(wood, along * -chord + perp * (s * off) + Vector3(0, ym, 0), along * chord + perp * (s * off) + Vector3(0, ym, 0), Vector3(0, 1, 0), w, y1 - y0 - 0.004f, look);
                }
            }
            // iron plates on the crossings
            const WoodLook plate = woodLook(V3(0.33f, 0.34f, 0.36f), seed + 9);
            for (int k = 0; k < 4; ++k) {
                const float g = 0.25f * inside::pi + static_cast<float>(k) * 0.5f * inside::pi;
                const Vector3 c = U.dir(g) * (off * 1.4142f);
                appendBeam(ironB, c + Vector3(0, y1 + 0.004f, 0) - U.dir(g) * 0.09f, c + Vector3(0, y1 + 0.004f, 0) + U.dir(g) * 0.09f, Vector3(0, 1, 0), 0.1f, 0.008f, plate);
            }
        }

        // ---- the stones, their tuns and feed --------------------------------------------------------------------------------
        void buildStones(const Materials&) {
            using namespace inside;
            using namespace site;
            using gear::Rot;
            MeshBuild& frame = b(static_, interior_);
            MeshBuild& boards = b(static_, boards_);
            MeshBuild& ironS = b(static_, iron_);
            MeshBuild& dark = b(static_, black_, false);
            const float sx = spindleX(0);
            // the bridge beam under the bin floor that holds the spindles' top bearings
            {
                const float by = binFloor - boardT - joistD - 0.5f * beamD;
                appendBeam(frame, Vector3(sx, by, -beamZ + 0.5f * beamW), Vector3(sx, by, beamZ - 0.5f * beamW), Vector3(0, 1, 0), 0.24f, beamD, woodLook(V3(0.6f, 0.54f, 0.48f), 600));
            }
            for (int i = 0; i < 2; ++i) {
                const Vector3 c(spindleX(i), 0, spindleZ(i));
                // the direction away from the upright, and across it
                Vector3 out(c.x - uprightX, 0, c.z - uprightZ);
                out.normalize();
                const Vector3 side = Vector3(0, 1, 0).cross(out);
                Rng rng(610 + i);
                // the hurst platform: a box of planks
                {
                    const float h = 0.875f;
                    for (float zz = -h + 0.1f; zz < h; zz += 0.2f) {
                        WoodLook look = woodLook(V3(0.9f, 0.82f, 0.72f) * rng.range(0.85f, 1.05f), rng.next());
                        look.weather = dustyFloor;
                        appendBeam(boards, c + Vector3(-h, hurstTop - 0.02f, zz), c + Vector3(h, hurstTop - 0.02f, zz), Vector3(0, 1, 0), 0.196f, 0.04f, look);
                    }
                    for (int k = 0; k < 4; ++k) {
                        const float g = static_cast<float>(k) * 0.5f * pi;
                        const Vector3 n(std::cos(g), 0, std::sin(g)), t(-std::sin(g), 0, std::cos(g));
                        appendBeam(frame, c + n * (h - 0.025f) - t * h + Vector3(0, stoneFloor + 0.5f * (hurstTop - 0.04f - stoneFloor), 0),
                                   c + n * (h - 0.025f) + t * h + Vector3(0, stoneFloor + 0.5f * (hurstTop - 0.04f - stoneFloor), 0), n, hurstTop - 0.04f - stoneFloor, 0.05f,
                                   woodLook(V3(0.7f, 0.62f, 0.55f), rng.next()));
                    }
                }
                // the bed stone, dressed face up
                const Rot S = gear::aboutY(c);
                appendMillstoneTo(S, hurstTop, bedTop, V3(1.0f, 0.94f, 0.84f), 620u + i, false);
                // the tun: staves round the stones, iron hoops; the south tun has lost three staves and half its lid
                const bool broken = i == 0;
                const int staves = 26;
                const float toEye = std::atan2(-(3.6f - c.z), 11.6f - c.x);// psi toward the stone floor's usual eye
                for (int k = 0; k < staves; ++k) {
                    const float g = gear::twoPi * (static_cast<float>(k) + 0.5f) / static_cast<float>(staves);
                    float dg = g - toEye;
                    dg -= gear::twoPi * std::floor((dg + pi) / gear::twoPi);
                    if (broken && std::abs(dg) < 0.32f) continue;
                    const Vector3 n = S.dir(g);
                    WoodLook look = woodLook(V3(0.7f, 0.62f, 0.54f) * rng.range(0.85f, 1.05f), rng.next());
                    appendBeam(frame, c + n * (tunR + 0.018f) + Vector3(0, hurstTop, 0), c + n * (tunR + 0.018f) + Vector3(0, tunTop, 0), n,
                               gear::twoPi * (tunR + 0.018f) / static_cast<float>(staves) - 0.003f, 0.034f, look);
                }
                for (float hy : {hurstTop + 0.12f, tunTop - 0.1f}) {
                    if (broken) {
                        // the hoops still go round over the gap
                        gear::appendArc(ironS, S, tunR + 0.036f, tunR + 0.042f, hy - 0.02f, hy + 0.02f, 0.f, gear::twoPi, 64, woodLook(V3(0.32f, 0.33f, 0.35f), 630), true);
                    } else {
                        gear::appendArc(ironS, S, tunR + 0.036f, tunR + 0.042f, hy - 0.02f, hy + 0.02f, 0.f, gear::twoPi, 64, woodLook(V3(0.32f, 0.33f, 0.35f), 630), false);
                    }
                }
                // the lid: boards across, cut round the eye; the broken tun's far half lies against the platform
                for (int k = -3; k <= 3; ++k) {
                    const float zz = static_cast<float>(k) * 0.205f;
                    const float half = std::sqrt(std::max(0.f, (tunR + 0.05f) * (tunR + 0.05f) - zz * zz));
                    if (half < 0.1f) continue;
                    if (broken && k < 0) continue;
                    std::vector<std::pair<float, float>> spans{{-half, half}};
                    if (std::abs(zz) < 0.2f) spans = {{-half, -0.18f}, {0.18f, half}};
                    for (auto [a, e] : spans) {
                        WoodLook look = woodLook(V3(0.9f, 0.82f, 0.72f) * rng.range(0.85f, 1.05f), rng.next());
                        // local frame: x along `side`, z along `out`
                        const Vector3 p0 = c + side * a + out * zz + Vector3(0, lidTop - 0.015f, 0), p1 = c + side * e + out * zz + Vector3(0, lidTop - 0.015f, 0);
                        appendBeam(boards, p0, p1, Vector3(0, 1, 0), 0.2f, 0.03f, look);
                    }
                }
                if (broken) {
                    // the lost half lid leans against the hurst
                    const Vector3 base = c - out * 1.05f + Vector3(0, stoneFloor, 0);
                    for (int k = 0; k < 3; ++k) {
                        const float a = (static_cast<float>(k) - 1.f) * 0.205f;
                        const float len = 1.3f - 0.3f * std::abs(static_cast<float>(k) - 1.f);
                        const Vector3 foot = base + side * a;
                        appendBeam(boards, foot, foot + Vector3(0, 0.36f, 0) + out * 0.17f, side, len * 0.55f, 0.03f, woodLook(V3(0.85f, 0.78f, 0.7f), rng.next()));
                    }
                }
                // the horse on the lid and the hopper on the horse, out from the spindle
                const Vector3 hc = c + out * hopperOut;
                for (float a : {-0.22f, 0.22f}) {
                    for (float o : {-0.22f, 0.22f}) {
                        const Vector3 foot = hc + side * a + out * o + Vector3(0, lidTop, 0);
                        appendBeam(frame, foot, foot + Vector3(0, 0.26f, 0), out, 0.045f, 0.045f, woodLook(V3(0.66f, 0.58f, 0.5f), rng.next()));
                    }
                    appendBeam(frame, hc + side * a - out * 0.3f + Vector3(0, lidTop + 0.28f, 0), hc + side * a + out * 0.3f + Vector3(0, lidTop + 0.28f, 0), Vector3(0, 1, 0), 0.05f, 0.05f,
                               woodLook(V3(0.66f, 0.58f, 0.5f), rng.next()));
                }
                {
                    const float y0 = lidTop + 0.3f, y1 = y0 + 0.48f, t0 = 0.07f, t1 = 0.33f;
                    const std::array<Vector3, 4> d{side, out, side * -1.f, out * -1.f};
                    for (int k = 0; k < 4; ++k) {
                        const Vector3 n = d[k], t = d[(k + 1) % 4];
                        const Vector3 lb = hc + n * t0 - t * t0 + Vector3(0, y0, 0), rb = hc + n * t0 + t * t0 + Vector3(0, y0, 0);
                        const Vector3 lt = hc + n * t1 - t * t1 + Vector3(0, y1, 0), rt = hc + n * t1 + t * t1 + Vector3(0, y1, 0);
                        parts::appendSlab(frame, lb, rb, rt, lt, -0.022f, woodLook(V3(0.72f, 0.64f, 0.55f) * rng.range(0.9f, 1.05f), rng.next()));
                    }
                }
                // the shoe: a trough from the hopper's mouth to the eye, hung so the damsel can shake it
                {
                    shoe_[i]->position.copy(hc + Vector3(0, lidTop + 0.25f, 0));
                    shoe_[i]->rotation.setOrder(threepp::Euler::RotationOrders::YXZ);
                    shoe_[i]->rotation.y = std::atan2(-out.z, out.x);// local +x along `out`
                    root_->add(shoe_[i]);
                    MeshBuild& sw = b(shoeB_[i], interior_);
                    const float x0 = 0.08f, x1 = -hopperOut + 0.11f;// from under the hopper back to the damsel
                    const float y0 = 0.f, y1 = -0.11f;
                    const WoodLook sl = woodLook(V3(0.7f, 0.62f, 0.54f), 640u + i);
                    parts::appendSlab(sw, Vector3(x0, y0, -0.08f), Vector3(x0, y0, 0.08f), Vector3(x1, y1, 0.08f), Vector3(x1, y1, -0.08f), 0.015f, sl);
                    for (float s : {-1.f, 1.f})
                        parts::appendSlab(sw, Vector3(x0, y0, s * 0.08f), Vector3(x1, y1, s * 0.08f), Vector3(x1, y1 + 0.07f, s * 0.08f), Vector3(x0, y0 + 0.07f, s * 0.08f), 0.012f * s, sl);
                    flush(shoe_[i], shoeB_[i]);
                }
                // the meal spout and the ark it fills
                {
                    const float g = std::atan2(-side.z, side.x);// psi of `side`
                    const Vector3 n = S.dir(g);
                    const Vector3 a = c + n * (tunR + 0.02f) + Vector3(0, hurstTop + 0.15f, 0), e = c + n * 1.15f + Vector3(0, hurstTop - 0.02f, 0);
                    appendBeam(frame, a, e, Vector3(0, 1, 0), 0.14f, 0.1f, woodLook(V3(0.7f, 0.62f, 0.54f), rng.next()));
                    const Vector3 ac = c + n * 1.32f;
                    const Vector3 t(-n.z, 0, n.x);
                    const float hx = 0.3f, hz = 0.24f, h = 0.4f;
                    for (int k = 0; k < 4; ++k) {
                        const Vector3 nn = k == 0 ? n : k == 1 ? t : k == 2 ? n * -1.f : t * -1.f;
                        const Vector3 tt = k == 0 ? t : k == 1 ? n * -1.f : k == 2 ? t * -1.f : n;
                        const float hn = (k % 2 == 0) ? hx : hz, ht = (k % 2 == 0) ? hz : hx;
                        const Vector3 f = ac + nn * hn;
                        parts::appendSlab(frame, f - tt * ht + Vector3(0, stoneFloor, 0), f + tt * ht + Vector3(0, stoneFloor, 0), f + tt * ht + Vector3(0, stoneFloor + h, 0),
                                          f - tt * ht + Vector3(0, stoneFloor + h, 0), 0.025f, woodLook(V3(0.72f, 0.63f, 0.54f), rng.next()));
                    }
                    // meal in the bottom of the ark, long since gone grey
                    MeshBuild& meal = b(static_, burrPlain_);
                    appendQuad(meal, ac + n * -0.27f + t * -0.21f + Vector3(0, stoneFloor + 0.09f, 0), ac + n * 0.27f + t * -0.21f + Vector3(0, stoneFloor + 0.09f, 0),
                               ac + n * 0.27f + t * 0.21f + Vector3(0, stoneFloor + 0.09f, 0), ac + n * -0.27f + t * 0.21f + Vector3(0, stoneFloor + 0.09f, 0), 0.6f, 0.6f, V3(1.05f, 1.02f, 0.96f));
                }
                // the turning parts: spindle, runner, rynd, damsel, stone nut
                nut_[i]->position.set(c.x, 0, c.z);
                root_->add(nut_[i]);
                {
                    const Rot N = gear::aboutY(Vector3(0, 0, 0));
                    MeshBuild& nb = b(nutB_[i], black_, false);
                    MeshBuild& ni = b(nutB_[i], iron_);
                    appendCylinder(nb, Vector3(0, runner0 + 0.05f, 0), Vector3(0, binFloor - boardT - joistD - beamD + 0.02f, 0), 0.04f, 0.04f, 12, WoodLook{});
                    // the runner: face down
                    MeshBuild& face = b(nutB_[i], burrFace_);
                    MeshBuild& plain = b(nutB_[i], burrPlain_);
                    parts::appendMillstone(face, plain, ni, N, stoneRadius, eyeRadius, runner0, runner1, V3(1.0f, 0.94f, 0.84f), true, 650u + i);
                    // the rynd across the eye
                    appendBeam(nb, Vector3(-eyeRadius - 0.03f, runner1 - 0.05f, 0), Vector3(eyeRadius + 0.03f, runner1 - 0.05f, 0), Vector3(0, 1, 0), 0.05f, 0.04f, WoodLook{});
                    // the damsel: a square on the spindle with four knocking arms
                    const WoodLook rust = woodLook(V3(0.4f, 0.4f, 0.42f), 660u + i);
                    gear::appendPrism(ni, N, 0.05f, damsel0, damsel1, 4, rust);
                    for (int k = 0; k < 4; ++k) gear::appendCog(ni, N, static_cast<float>(k) * 0.5f * pi + 0.25f * pi, 0.04f, 0.085f, 0.018f, damsel0 + 0.04f, damsel1 - 0.04f, rust);
                    // the stone nut: a cast iron pinion of 16 teeth (tooth 0 at angle 0)
                    gear::appendRound(ni, N, 0.19f, nutY0, nutY1, 32, rust);
                    gear::appendRound(ni, N, 0.1f, nutY0 - 0.08f, nutY1 + 0.08f, 20, rust);
                    for (int k = 0; k < nutTeeth; ++k) {
                        const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(nutTeeth);
                        gear::appendTooth(ni, N, g, nutR - 0.036f, nutR + 0.03f, 0.046f, 0.026f, nutY0, nutY1, rust);
                    }
                    flush(nut_[i], nutB_[i]);
                }
                // the spindle's top bearing under the bridge beam
                appendBeam(dark, c + Vector3(-0.12f, binFloor - boardT - joistD - beamD - 0.05f, 0), c + Vector3(0.12f, binFloor - boardT - joistD - beamD - 0.05f, 0), Vector3(0, 1, 0), 0.16f, 0.1f, WoodLook{});
            }
            // a third stone, worn out, turned face up on blocks for dressing and never put back
            {
                const Vector3 c(10.5f, 0, -2.45f);
                MeshBuild& stones = b(static_, masonry_);
                Rng rng(690);
                for (int k = 0; k < 3; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / 3.f + 0.4f;
                    Frame f{c + Vector3(std::cos(g) * 0.36f, stoneFloor + 0.07f, std::sin(g) * 0.36f), Vector3(std::cos(g), 0, std::sin(g)), Vector3(0, 1, 0), Vector3(-std::sin(g), 0, std::cos(g))};
                    StoneLook look;
                    look.tint = V3(0.6f, 0.52f, 0.44f);
                    look.seed = rng.next();
                    appendStone(stones, f, 0.14f, 0.07f, 0.09f, look);
                }
                appendMillstoneTo(gear::aboutY(c), stoneFloor + 0.14f, stoneFloor + 0.42f, V3(0.98f, 0.92f, 0.82f), 695, true);
            }
        }

        // a static millstone: face at hFace, back at hBack
        void appendMillstoneTo(const gear::Rot& S, float hBack, float hFace, const V3& tint, uint32_t seed, bool crack) {
            using namespace inside;
            MeshBuild& face = b(static_, burrFace_);
            MeshBuild& plain = b(static_, burrPlain_);
            MeshBuild& ironS = b(static_, iron_);
            const unsigned f0 = face.vertexCount();
            parts::appendMillstone(face, plain, ironS, S, stoneRadius, eyeRadius, hFace, hBack, tint, true, seed);
            if (crack) {
                // a dark crack across the face (the reason it was never put back)
                parts::reweather(face, f0, [&](const Vector3& p, const Vector3&, const V3& c) {
                    const float dx = p.x - S.o.x, dz = p.z - S.o.z;
                    const float d = std::abs(0.8f * dx + 0.6f * dz - 0.05f + 0.03f * std::sin(dx * 9.f));
                    return c * (0.55f + 0.45f * smooth(0.0f, 0.03f, d));
                });
            }
        }

        // ---- the head of the upright: the crown wheel drives the lay shaft; the sack hoist ----------------------------------
        void buildHead(const Materials&) {
            using namespace inside;
            using namespace site;
            MeshBuild& frame = b(static_, interior_);
            MeshBuild& dark = b(static_, black_, false);
            // the sprattle beam over the upright's head, on two posts
            appendBeam(frame, Vector3(uprightX, uprightHead + 0.1f, uprightZ - 1.35f), Vector3(uprightX, uprightHead + 0.1f, uprightZ + 1.35f), Vector3(0, 1, 0), 0.22f, 0.2f,
                       woodLook(V3(0.6f, 0.54f, 0.48f), 700));
            for (float s : {-1.f, 1.f}) appendPost(frame, uprightX, uprightZ + s * 1.2f, garnerFloor, uprightHead, 701 + (s > 0), false);
            appendBeam(dark, Vector3(uprightX - 0.1f, uprightHead - 0.02f, uprightZ), Vector3(uprightX + 0.1f, uprightHead - 0.02f, uprightZ), Vector3(0, 1, 0), 0.18f, 0.06f, WoodLook{});
            // the lay shaft: its lantern over the crown wheel, two bearings on trestles, the belt pulley
            lay_->position.set(0, layY, uprightZ);
            root_->add(lay_);
            {
                const gear::Rot L = gear::aboutX(Vector3(0, 0, 0));
                MeshBuild& lw = b(layB_, interior_);
                MeshBuild& li = b(layB_, iron_);
                MeshBuild& ld = b(layB_, black_, false);
                appendCylinder(ld, Vector3(layX0, 0, 0), Vector3(layX1, 0, 0), 0.055f, 0.055f, 16, WoodLook{});
                const WoodLook head = woodLook(V3(0.55f, 0.47f, 0.4f), 710);
                gear::appendDisc(lw, gear::aboutX(Vector3(layLantern0 - 0.04f, 0, 0)), 0.f, layHeadR, 0.f, 0.04f, 32, head);
                gear::appendDisc(lw, gear::aboutX(Vector3(layLantern1, 0, 0)), 0.f, layHeadR, 0.f, 0.04f, 32, head);
                for (int k = 0; k < layStaves; ++k) {
                    const float g = gear::twoPi * static_cast<float>(k) / static_cast<float>(layStaves);
                    const Vector3 p = L.dir(g) * layR;
                    appendCylinder(li, Vector3(layLantern0 - 0.01f, p.y, p.z), Vector3(layLantern1 + 0.01f, p.y, p.z), layStave, layStave, 8, woodLook(V3(0.33f, 0.34f, 0.36f), 712), false);
                }
                // the pulley: a wooden drum on four arms
                const gear::Rot P = gear::aboutX(Vector3(pulleyX, 0, 0));
                gear::appendArc(lw, P, layPulleyR - 0.04f, layPulleyR, -0.08f, 0.08f, 0.f, gear::twoPi, 48, woodLook(V3(0.6f, 0.52f, 0.45f), 711), false);
                for (int k = 0; k < 4; ++k) gear::appendCog(lw, P, static_cast<float>(k) * 0.5f * pi, 0.05f, layPulleyR - 0.035f, 0.05f, -0.025f, 0.025f, head);
                gear::appendRound(li, P, 0.08f, -0.06f, 0.06f, 16, head);
            }
            flush(lay_, layB_);
            for (float bx : {8.75f, 9.95f}) {
                // a trestle: two legs, a cap, the plummer block
                for (float s : {-1.f, 1.f})
                    appendBeam(frame, Vector3(bx, garnerFloor, uprightZ + s * 0.35f), Vector3(bx, layY - 0.13f, uprightZ + s * 0.12f), Vector3(1, 0, 0), 0.1f, 0.1f, woodLook(V3(0.62f, 0.56f, 0.5f), 720 + (s > 0)));
                appendBeam(frame, Vector3(bx, layY - 0.1f, uprightZ - 0.22f), Vector3(bx, layY - 0.1f, uprightZ + 0.22f), Vector3(0, 1, 0), 0.14f, 0.07f, woodLook(V3(0.62f, 0.56f, 0.5f), 722));
                appendBeam(dark, Vector3(bx - 0.07f, layY - 0.045f, uprightZ), Vector3(bx + 0.07f, layY - 0.045f, uprightZ), Vector3(0, 1, 0), 0.26f, 0.04f, WoodLook{});
                appendCylinder(dark, Vector3(bx - 0.06f, layY, uprightZ), Vector3(bx + 0.06f, layY, uprightZ), 0.085f, 0.085f, 16, WoodLook{});
            }
            // the sack hoist: a drum on a frame over the traps, its own pulley in line with the lay shaft's, a slack belt
            {
                const float d0 = pulleyX + 0.2f, d1 = 11.55f;
                const gear::Rot D = gear::aboutX(Vector3(0, hoistY, hoistZ));
                gear::appendPrism(frame, D, drumR, d0, d1, 10, woodLook(V3(0.55f, 0.48f, 0.42f), 730));
                appendCylinder(dark, Vector3(pulleyX - 0.25f, hoistY, hoistZ), Vector3(d1 + 0.25f, hoistY, hoistZ), 0.04f, 0.04f, 10, WoodLook{});
                const gear::Rot P = gear::aboutX(Vector3(pulleyX, hoistY, hoistZ));
                gear::appendArc(frame, P, hoistPulleyR - 0.05f, hoistPulleyR, -0.085f, 0.085f, 0.f, gear::twoPi, 48, woodLook(V3(0.6f, 0.52f, 0.45f), 731), false);
                gear::appendDisc(frame, P, 0.08f, hoistPulleyR - 0.04f, -0.02f, 0.02f, 48, woodLook(V3(0.6f, 0.52f, 0.45f), 732));
                for (float px : {pulleyX - 0.2f, d1 + 0.18f}) {
                    for (float s : {-1.f, 1.f})
                        appendBeam(frame, Vector3(px, garnerFloor, hoistZ + s * 0.45f), Vector3(px, hoistY + 0.06f, hoistZ + s * 0.1f), Vector3(1, 0, 0), 0.11f, 0.11f, woodLook(V3(0.62f, 0.56f, 0.5f), 733 + (s > 0)));
                    appendBeam(frame, Vector3(px, hoistY - 0.1f, hoistZ - 0.25f), Vector3(px, hoistY - 0.1f, hoistZ + 0.25f), Vector3(0, 1, 0), 0.13f, 0.08f, woodLook(V3(0.62f, 0.56f, 0.5f), 735));
                }
                // the rope wound on the drum, and its fall down to the trap with the chain and hook
                MeshBuild& rope = b(static_, rope_);
                for (int k = 0; k < 9; ++k) {
                    const float x = 10.95f + 0.032f * static_cast<float>(k);
                    gear::appendArc(rope, gear::aboutX(Vector3(x, hoistY, hoistZ)), drumR + 0.002f, drumR + 0.03f, -0.014f, 0.014f, 0.f, gear::twoPi, 20, WoodLook{V3(1, 1, 1), 736}, false);
                }
                const Vector3 top(11.25f, hoistY, hoistZ - drumR - 0.016f);
                appendCylinder(rope, top, Vector3(11.25f, garnerFloor + 0.32f, hoistZ - drumR - 0.016f), 0.014f, 0.014f, 8, WoodLook{V3(1, 1, 1), 737});
                for (int k = 0; k < 4; ++k) {
                    const float y = garnerFloor + 0.3f - 0.055f * static_cast<float>(k);
                    const Vector3 lc(11.25f, y, hoistZ - drumR - 0.016f);
                    const Vector3 ax = k % 2 ? Vector3(1, 0, 0) : Vector3(0, 0, 1);
                    gear::appendArc(ironS(), {lc, ax, Vector3(0, 1, 0), ax.clone().cross(Vector3(0, 1, 0))}, 0.018f, 0.028f, -0.005f, 0.005f, 0.f, gear::twoPi, 12,
                                    woodLook(V3(0.7f, 0.6f, 0.55f), 738), false);
                }
                // the belt: over the hoist pulley, down round the lay shaft's pulley, and hanging slack between
                buildBelt();
            }
        }

        MeshBuild& ironS() { return b(static_, iron_); }

        // a slack belt in the plane x = pulleyX: the hoist pulley (above, at hoistZ) and the lay shaft's (below, at the
        // upright's z); the top run lies tangent over both, the bottom run sags in a loose curve
        void buildBelt() {
            using namespace inside;
            MeshBuild& belt = b(static_, leather_);
            // 2D in (z, y)
            struct P2 {
                float z, y;
            };
            const P2 A{hoistZ, hoistY}, B{uprightZ, layY};
            const float rA = hoistPulleyR + 0.005f, rB = layPulleyR + 0.005f;
            const float dz = B.z - A.z, dy = B.y - A.y, dist = std::hypot(dz, dy);
            const float base = std::atan2(dy, dz);
            // the outer tangent on the upper side
            const float off = std::acos((rA - rB) / dist);
            const float aTop = base + off;// angle (in z,y) of the tangent point's normal on both circles
            std::vector<P2> path;
            // round the hoist pulley over its top from the far side to the tangent point
            const float aFar = base + pi;// the far side
            for (int k = 0; k <= 24; ++k) {
                const float a = aFar - (aFar - aTop) * static_cast<float>(k) / 24.f;
                path.push_back({A.z + rA * std::cos(a), A.y + rA * std::sin(a)});
            }
            // the straight top run to the lay pulley, and round it (it hangs on the lay pulley's far side)
            const float bEnd = base - 0.5f * pi - 0.4f;
            for (int k = 0; k <= 20; ++k) {
                const float a = aTop - (aTop - bEnd) * static_cast<float>(k) / 20.f;
                path.push_back({B.z + rB * std::cos(a), B.y + rB * std::sin(a)});
            }
            // the slack run back, sagging well below the line between them
            const P2 s0 = path.back(), s1 = {A.z + rA * std::cos(aFar), A.y + rA * std::sin(aFar)};
            for (int k = 1; k < 24; ++k) {
                const float t = static_cast<float>(k) / 24.f;
                const float sag = 0.42f * 4.f * t * (1.f - t);
                path.push_back({s0.z + (s1.z - s0.z) * t, s0.y + (s1.y - s0.y) * t - sag});
            }
            path.push_back(path.front());
            const float hw = 0.055f;
            for (size_t k = 0; k + 1 < path.size(); ++k) {
                const Vector3 p0(pulleyX, path[k].y, path[k].z), p1(pulleyX, path[k + 1].y, path[k + 1].z);
                appendQuad(belt, p0 + Vector3(-hw, 0, 0), p0 + Vector3(hw, 0, 0), p1 + Vector3(hw, 0, 0), p1 + Vector3(-hw, 0, 0), 1.f, 1.f, V3(1, 1, 1));
            }
        }

        // ---- bins on the bin floor and in the garner --------------------------------------------------------------------
        void appendBin(const Vector3& c, float hx, float hz, float h, uint32_t seed, const std::array<bool, 4>& walls) {
            using namespace inside;
            MeshBuild& frame = b(static_, interior_);
            Rng rng(seed);
            // corner posts, then boards laid horizontally on the open sides
            for (float sx : {-1.f, 1.f})
                for (float sz : {-1.f, 1.f}) appendBeam(frame, c + Vector3(sx * (hx - 0.04f), 0, sz * (hz - 0.04f)), c + Vector3(sx * (hx - 0.04f), h + 0.08f, sz * (hz - 0.04f)), Vector3(1, 0, 0), 0.08f, 0.08f, woodLook(V3(0.62f, 0.56f, 0.5f), rng.next()));
            for (int k = 0; k < 4; ++k) {
                if (!walls[k]) continue;
                const bool alongX = k % 2 == 0;
                const float s = k < 2 ? 1.f : -1.f;
                for (float y = 0.11f; y < h; y += 0.22f) {
                    const float yy = std::min(y, h - 0.11f);
                    WoodLook look = woodLook(V3(0.74f, 0.66f, 0.57f) * rng.range(0.85f, 1.06f), rng.next());
                    look.bow = rng.range(-0.008f, 0.008f);
                    if (alongX) appendBeam(frame, c + Vector3(-hx, yy, s * (hz + 0.012f)), c + Vector3(hx, yy, s * (hz + 0.012f)), Vector3(0, 1, 0), 0.025f, 0.21f, look);
                    else appendBeam(frame, c + Vector3(s * (hx + 0.012f), yy, -hz), c + Vector3(s * (hx + 0.012f), yy, hz), Vector3(0, 1, 0), 0.025f, 0.21f, look);
                }
            }
        }

        void buildBins(const Materials&) {
            using namespace site;
            // over the stones: a long bin against the south wall, and one against the west wall
            appendBin(Vector3(11.8f, binFloor, 0.3f), 0.5f, 1.2f, 1.5f, 750, {true, true, true, true});
            appendBin(Vector3(6.75f, binFloor, -0.88f), 0.6f, 0.88f, 1.45f, 751, {true, true, true, true});
            // in the garner, low bins under the eaves on the east side
            appendBin(Vector3(11.6f, garnerFloor, 2.6f), 0.55f, 1.0f, 0.9f, 752, {true, true, true, true});
        }

        // ---- what was left behind ------------------------------------------------------------------------------------
        void buildLeftBehind(const Materials&) {
            using namespace inside;
            using namespace site;
            using threepp::Matrix4;
            MeshBuild& sacks = b(static_, sacking_);
            const auto place = [](const Vector3& at, float yaw, float roll = 0.f, float pitch = 0.f) {
                Matrix4 m;
                m.makeRotationFromEuler(threepp::Euler(pitch, yaw, roll, threepp::Euler::RotationOrders::YXZ));
                m.setPosition(at);
                return m;
            };
            const auto sackLook = [](const Vector3& p, const Vector3& n, const V3& c) { return dusty(p, n, c) * V3(0.62f, 0.58f, 0.54f); };
            {
                const unsigned s0 = sacks.vertexCount();
                // by the loading doors: two full sacks, one slumped against the other, an empty one on the floor
                parts::appendSack(sacks, place(Vector3(11.55f, stoneFloor, 0.95f), 0.3f), 0.46f, 0.32f, 0.72f, 1.f, 0.f, 760);
                parts::appendSack(sacks, place(Vector3(11.25f, stoneFloor, 1.32f), 2.2f), 0.44f, 0.3f, 0.66f, 0.8f, 0.22f, 761);
                parts::appendEmptySack(sacks, place(Vector3(10.4f, stoneFloor, 0.65f), 0.5f), 0.9f, 0.55f, 762);
                // on the bin floor, by the bin
                parts::appendSack(sacks, place(Vector3(11.05f, binFloor, 1.9f), 1.1f), 0.46f, 0.3f, 0.7f, 0.9f, 0.1f, 763);
                parts::appendSack(sacks, place(Vector3(7.55f, binFloor, -2.45f), -0.4f), 0.46f, 0.32f, 0.5f, 0.45f, 0.35f, 764);
                // in the garner, one gone flat
                parts::appendEmptySack(sacks, place(Vector3(10.6f, garnerFloor, 2.4f), 1.4f), 0.85f, 0.5f, 765);
                parts::reweather(sacks, s0, sackLook);
            }
            MeshBuild& wood = b(static_, interior_);
            {
                const unsigned w0 = wood.vertexCount();
                // the chair on its side by the west window
                parts::appendBrokenChair(wood, place(Vector3(6.75f, stoneFloor + 0.215f, -2.55f), 0.35f, -0.5f * pi), 770);
                // its broken leg
                appendCylinder(wood, Vector3(7.25f, stoneFloor + 0.018f, -2.1f), Vector3(7.5f, stoneFloor + 0.018f, -2.02f), 0.017f, 0.016f, 8, WoodLook{V3(0.62f, 0.5f, 0.4f), 771});
                // a grain shovel leaning on the mid post
                const Vector3 foot(midPostX + 0.32f, stoneFloor + 0.01f, beamZ + 0.16f), grip(midPostX + 0.12f, stoneFloor + 1.18f, beamZ + 0.1f);
                appendCylinder(wood, foot + (grip - foot) * 0.27f, grip, 0.018f, 0.017f, 8, WoodLook{V3(0.7f, 0.58f, 0.46f), 772});
                {
                    // the blade: a wide wooden scoop
                    const Vector3 axis = (grip - foot).normalize();
                    const Vector3 b0 = foot, b1 = foot + axis * 0.36f;
                    parts::appendSlab(wood, b0 - Vector3(0, 0, 0.14f), b0 + Vector3(0, 0, 0.14f), b1 + Vector3(0, 0, 0.12f), b1 - Vector3(0, 0, 0.12f), 0.012f,
                                      WoodLook{V3(0.66f, 0.55f, 0.44f), 773});
                }
                parts::reweather(wood, w0, dusty);
            }
            // a barrel by the east wall, and a broken one in the basement on its side
            {
                MeshBuild& ironB = b(static_, iron_);
                const unsigned w0 = wood.vertexCount();
                parts::appendBarrel(wood, ironB, Vector3(11.75f, stoneFloor, 3.35f), 0.82f, 0.29f, 0.25f, 780);
                // in the basement: an old barrel, and two planks put by against the east wall
                parts::appendBarrel(wood, ironB, Vector3(11.35f, basementFloor, -2.25f), 0.72f, 0.27f, 0.23f, 781);
                appendBeam(wood, Vector3(11.62f, basementFloor + 0.01f, 0.9f), Vector3(11.86f, basementFloor + 2.3f, 0.98f), Vector3(-1, 0, 0), 0.24f, 0.035f, WoodLook{V3(0.6f, 0.52f, 0.44f), 782});
                appendBeam(wood, Vector3(11.55f, basementFloor + 0.01f, 1.25f), Vector3(11.86f, basementFloor + 2.1f, 1.18f), Vector3(-1, 0, 0), 0.2f, 0.035f, WoodLook{V3(0.52f, 0.46f, 0.4f), 783});
                parts::reweather(wood, w0, dusty);
            }
            // leaves blown in: under the south window by the stones, and in at the miller's door
            {
                MeshBuild& lv = b(static_, leaves_);
                Rng rng(790);
                const auto scatter = [&](const Vector3& c, float rx, float rz, int n, float y) {
                    for (int k = 0; k < n; ++k) {
                        // gathered toward the wall: more of them close to c
                        const float r = std::pow(rng(), 1.6f);
                        const float a = rng.range(0.f, gear::twoPi);
                        const Vector3 p = c + Vector3(std::cos(a) * r * rx, 0, std::sin(a) * r * rz);
                        const float size = rng.range(0.06f, 0.11f);
                        const float yaw = rng.range(0.f, gear::twoPi);
                        const int kind = static_cast<int>(rng() * 4.f) % 4;
                        const float u0 = (kind % 2) * 0.5f, v0 = (kind / 2) * 0.5f;
                        const Vector3 ax(std::cos(yaw), 0, std::sin(yaw)), sd(-std::sin(yaw), 0, std::cos(yaw));
                        // curled: the two halves lift along the midrib
                        const float curl = rng.range(0.005f, 0.02f);
                        const V3 tint = V3(1, 1, 1) * rng.range(0.7f, 1.05f);
                        const unsigned base = lv.vertexCount();
                        for (int j = 0; j < 3; ++j) {
                            for (int i = 0; i < 2; ++i) {
                                const float sx = (static_cast<float>(j) - 1.f) * 0.5f, ty = static_cast<float>(i);
                                const Vector3 q = p + sd * (sx * size) + ax * ((ty - 0.5f) * size) + Vector3(0, y + 0.004f + (j != 1 ? curl : 0.f) + 0.006f * ty, 0);
                                lv.vertex(q, Vector3(0, 1, 0), u0 + (sx + 0.5f) * 0.5f, v0 + ty * 0.5f, tint);
                            }
                        }
                        for (int j = 0; j < 2; ++j) {
                            const unsigned a0 = base + 2 * j, a1 = a0 + 1, b0 = a0 + 2, b1 = a0 + 3;
                            lv.index.insert(lv.index.end(), {a0, b0, a1, a1, b0, b1});
                        }
                    }
                };
                scatter(Vector3(7.55f, 0, 3.95f), 0.9f, 0.5f, 34, stoneFloor);
                scatter(Vector3(6.45f, 0, -3.6f), 0.6f, 0.7f, 22, stoneFloor);
                scatter(Vector3(7.6f, 0, 3.6f), 0.7f, 0.4f, 16, basementFloor);
                scatter(Vector3(7.7f, 0, 3.9f), 0.6f, 0.4f, 10, binFloor);
            }
            // cobwebs, thin and few: under the beams by the posts, the corners by the windows, under the stair, in the
            // hopper's horse, and in the basement between the pier and the joists
            {
                MeshBuild& web = b(static_, cobweb_);
                const float binBeam = binFloor - boardT - joistD - beamD;
                parts::appendCobweb(web, Vector3(midPostX + 0.11f, binBeam, beamZ - 0.15f), Vector3(1, 0, 0), Vector3(0, -1, 0), 0.55f, 0.03f);
                parts::appendCobweb(web, Vector3(6.32f, binFloor - boardT - joistD, 4.15f), Vector3(1, 0, 0), Vector3(0, -0.7f, -0.7f).normalize(), 0.6f, 0.04f);
                parts::appendCobweb(web, Vector3(12.18f, binFloor - boardT - joistD, -4.15f), Vector3(-1, 0, 0), Vector3(0, -1, 0), 0.5f, 0.02f);
                parts::appendCobweb(web, Vector3(midPostX - 0.11f, site::stoneFloor - boardT - joistD - beamD, -beamZ + 0.15f), Vector3(-1, 0, 0), Vector3(0, -1, 0), 0.5f, 0.03f);
                parts::appendCobweb(web, Vector3(6.62f, site::stoneFloor - boardT - joistD, -3.88f), Vector3(1, 0, 0), Vector3(0, -0.6f, 0.8f).normalize(), 0.7f, 0.05f);
                parts::appendCobweb(web, Vector3(12.18f, garnerFloor - boardT - joistD, 4.15f), Vector3(-0.7f, 0, -0.7f).normalize(), Vector3(0, -1, 0), 0.6f, 0.03f);
            }
        }

        // ---- dust motes, only in the shafts of sun that come in through the south and west windows ------------------------
        // (the sun's direction mirrors main.cpp's; setSun() moves the shafts if that changes)
    public:
        void setSun(const Vector3& towardSun) {
            sun_ = towardSun.clone().normalize();
            placeMotes();
        }

    private:
        Vector3 sun_ = Vector3(-0.52f, 0.72f, 0.46f).normalize();

        void buildMotes() {
            motes_ = threepp::ParticleSprites::create(160);
            motes_->setAnisotropy(0.55f);
            placeMotes();
            root_->add(motes_);
        }

        void placeMotes() {
            using namespace site;
            Rng rng(800);
            moteState_.clear();
            // the stone floor's and bin floor's windows on the sunny sides
            std::vector<WindowSpec> lit;
            for (const auto& w : millWindows())
                if ((w.facade == 0 || w.facade == 1) && w.y1 < garnerFloor) lit.push_back(w);
            int guard = 0;
            while (moteState_.size() < motes_->capacity() && ++guard < 20000) {
                const WindowSpec& w = lit[rng.next() % lit.size()];
                const Facade& f = facades()[w.facade];
                if (f.out.dot(sun_) < 0.05f) continue;
                // a point on the opening's inner face, then in along the sunbeam
                const Vector3 start = f.at(w.u + rng.range(-0.45f, 0.45f) * w.width, rng.range(w.y0 + 0.05f, w.y1 - 0.05f), -0.25f);
                const Vector3 p = start - sun_ * rng.range(0.15f, 2.6f);
                const float floorY = p.y < binFloor ? stoneFloor : binFloor;
                if (p.x < 6.4f || p.x > 12.1f || p.z < -4.1f || p.z > 4.1f || p.y < floorY + 0.1f) continue;
                Mote m;
                m.p = p;
                m.v = Vector3(rng.range(-0.008f, 0.008f), rng.range(-0.004f, 0.002f), rng.range(-0.008f, 0.008f));
                m.size = rng.range(0.003f, 0.006f);
                m.alpha = rng.range(0.2f, 0.45f);
                m.phase = rng.range(0.f, 6.28f);
                moteState_.push_back(m);
            }
            motes_->hideAll();
            updateMotes(0.f);
        }

        void updateMotes(float dt) {
            if (!motes_) return;
            moteTime_ += dt;
            for (size_t k = 0; k < moteState_.size(); ++k) {
                const Mote& m = moteState_[k];
                // a slow drift and a little eddying about where it hangs, so it never leaves its beam
                const float t = moteTime_ * 0.3f + m.phase;
                const Vector3 p = m.p + m.v * (4.f * std::sin(0.11f * moteTime_ + m.phase)) + Vector3(0.03f * std::sin(t), 0.02f * std::sin(1.3f * t + 1.f), 0.03f * std::cos(0.8f * t));
                motes_->set(k, p, m.size, m.alpha, threepp::Color(0.92f, 0.88f, 0.8f));
            }
            motes_->commit();
        }
    };

}// namespace mill

#endif
