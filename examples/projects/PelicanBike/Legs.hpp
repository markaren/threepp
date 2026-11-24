// The pelican's legs, driven by inverse kinematics from the pedals: a short femur hidden in the belly plumage, the
// feathered tibiotarsus, the bare orange tarsometatarsus and a totipalmate foot (all four toes joined by a web)
// planted on the pedal. Built in the bird's frame; the pedals' positions are given in it each frame.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_LEGS_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_LEGS_HPP

#include "Feathers.hpp"

namespace pb {

    struct Leg {
        float side = 1.f;
        Vector3 hip;
        float femur = 0.23f, tibia = 0.31f, tarsus = 0.115f;

        std::shared_ptr<Group> root = Group::create();
        std::shared_ptr<Group> femurGroup = Group::create(), tibiaGroup = Group::create(), tarsusGroup = Group::create(), footGroup = Group::create();
        static constexpr float femurInside = 0.04f;   // the thigh starts this far above the hip, inside the belly
        Vector3 knee, ankle, foot;   // the last solution, bird frame

        // a unit-length (along +Y) part
        static std::shared_ptr<Mesh> unitRod(const std::vector<std::pair<float, float>>& radius, const std::function<V3(float)>& colour, const std::shared_ptr<Material>& mat, float aspect, float length) {
            MeshBuild mb;
            std::vector<Vector3> pts;
            std::vector<float> r;
            const int n = 28;
            for (int i = 0; i <= n; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(n);
                pts.emplace_back(0.f, t * length, 0.f);
                // radius from the table, smooth
                size_t j = 0;
                while (j + 2 < radius.size() && t > radius[j + 1].first) ++j;
                const float u = clamp01((t - radius[j].first) / std::max(radius[j + 1].first - radius[j].first, 1e-6f));
                r.push_back(radius[j].second + (radius[j + 1].second - radius[j].second) * (u * u * (3.f - 2.f * u)));
            }
            SweepOptions o;
            o.seg = 18;
            o.aspect = aspect;
            o.up = Vector3(0, 0, 1);
            appendSweep(mb, pts, r, o);
            // paint
            const unsigned vcount = mb.vertexCount();
            for (unsigned i = 0; i < vcount; ++i) {
                const float y = mb.pos[3 * i + 1] / length;
                const V3 c = colour(y);
                mb.col[3 * i] = c.r;
                mb.col[3 * i + 1] = c.g;
                mb.col[3 * i + 2] = c.b;
            }
            return mb.mesh(mat);
        }

        // set up: the meshes; `thighFeathers` is a feather instance mesh to hang on the tibia, in its frame
        void build(const std::shared_ptr<Material>& skin, const std::shared_ptr<Material>& orange, const std::shared_ptr<Material>& web, const std::shared_ptr<Material>& ballMat, const std::shared_ptr<Material>& whiteMat) {
            const V3 white = srgb(247, 243, 234), leg = srgb(232, 128, 44), legDark = srgb(206, 100, 36);
            // the tibiotarsus: a feathered drumstick over the top two thirds, the bare shank below
            auto tib = unitRod({{0.f, 0.040f}, {0.18f, 0.043f}, {0.40f, 0.030f}, {0.50f, 0.0155f}, {1.f, 0.0115f}}, [&](float y) { return mill::mix(white, leg, smooth(0.44f, 0.52f, y)); }, skin, 0.92f, tibia);
            tibiaGroup->add(tib);
            // the thigh: from inside the belly to the knee, so the leg is one piece wherever the knee swings; a round knee
            // joins it to the drumstick
            femurGroup->add(unitRod({{0.f, 0.050f}, {0.45f, 0.058f}, {0.80f, 0.049f}, {1.f, 0.043f}}, [&](float) { return white; }, skin, 0.92f, femur + femurInside));
            tibiaGroup->add(ball(Vector3(0, 0, 0), 0.0425f, whiteMat, 18));
            // the tarsometatarsus: flattened, orange, reticulate
            auto tar = unitRod({{0.f, 0.0125f}, {0.5f, 0.0108f}, {1.f, 0.0125f}}, [&](float y) { return mill::mix(leg, legDark, 0.3f * y); }, orange, 0.62f, tarsus);
            tarsusGroup->add(tar);
            // a joint ball at the ankle and at the toes' root
            tarsusGroup->add(ball(Vector3(0, 0, 0), 0.0155f, ballMat, 12));
            footGroup->add(ball(Vector3(0, 0, 0), 0.0145f, ballMat, 12));
            buildFoot(*footGroup, orange, web);
            root->add(femurGroup);
            root->add(tibiaGroup);
            root->add(tarsusGroup);
            root->add(footGroup);
        }

        // the feathered thigh: short contour feathers lying down the tibia, in its own frame (+Y from the knee to the ankle)
        void addFeathers(const std::shared_ptr<BufferGeometry>& geo, const std::shared_ptr<Material>& mat, uint32_t seed) {
            Rng rng(seed);
            FeatherSet fs;
            const float zone = 0.5f * tibia;
            const int count = 520;
            for (int i = 0; i < count; ++i) {
                const float y = std::pow(rng(), 0.8f) * zone;
                const float a = rng.range(0.f, TAU);
                // the tube's radius there, as the skin has it
                const float r = y < 0.18f * tibia ? 0.040f + (0.043f - 0.040f) * y / (0.18f * tibia) : 0.043f - 0.013f * smooth(0.18f * tibia, 0.40f * tibia, y);
                const float rr = y < 0.4f * tibia ? r : 0.030f - 0.0145f * smooth(0.40f * tibia, 0.50f * tibia, y);
                const Vector3 n(std::cos(a), 0.f, std::sin(a));
                const Vector3 dir = Vector3(0.f, 1.f, 0.f);
                const float len = 0.05f * (1.f - 0.45f * y / zone) * rng.range(0.85f, 1.2f);
                fs.add(Vector3(n.x * (rr - 0.004f), y, n.z * (rr - 0.004f)), dir, n, len, 0.03f * rng.range(0.85f, 1.15f), 0.28f + 0.3f * rng(), rng.range(-0.25f, 0.25f), V3(0.97f, 0.97f, 0.97f));
            }
            tibiaGroup->add(fs.build(geo, mat));
            // and over the thigh, in its own frame (+Y from inside the belly down to the knee)
            FeatherSet ft;
            const float zl = femur + femurInside;
            for (int i = 0; i < 420; ++i) {
                const float y = femurInside * 0.5f + rng() * (zl - femurInside * 0.5f);
                const float a = rng.range(0.f, TAU);
                const float f = y / zl;
                const float r = f < 0.45f ? 0.050f + 0.008f * f / 0.45f : 0.058f - 0.015f * (f - 0.45f) / 0.55f;
                const Vector3 n(std::cos(a), 0.f, std::sin(a));
                const float len = 0.055f * rng.range(0.85f, 1.2f);
                ft.add(Vector3(n.x * (r - 0.004f), y, n.z * (r - 0.004f)), Vector3(0.f, 1.f, 0.f), n, len, 0.034f * rng.range(0.85f, 1.15f), 0.28f + 0.3f * rng(), rng.range(-0.25f, 0.25f), V3(0.97f, 0.97f, 0.97f));
            }
            femurGroup->add(ft.build(geo, mat));
        }

        void buildFoot(Group& g, const std::shared_ptr<Material>& orange, const std::shared_ptr<Material>& web) {
            struct Toe {
                float yaw, length, droop;
            };
            // yaw measured from forward toward outward (+z for the right foot)
            const Toe toes[4] = {{-0.30f, 0.105f, 0.004f}, {0.02f, 0.128f, 0.004f}, {0.34f, 0.112f, 0.004f}, {-2.55f, 0.052f, 0.0f}};
            const V3 colour = srgb(230, 124, 42);
            std::vector<std::vector<Vector3>> lines(4);
            for (int i = 0; i < 4; ++i) {
                const Toe& T = toes[i];
                const Vector3 dir(std::cos(T.yaw), 0.f, std::sin(T.yaw) * side);
                std::vector<Vector3> pts;
                const int n = 8;
                for (int k = 0; k <= n; ++k) {
                    const float f = static_cast<float>(k) / static_cast<float>(n);
                    pts.push_back(dir * (T.length * f) + Vector3(0.f, -0.006f - T.droop * f * f, 0.f));
                }
                lines[static_cast<size_t>(i)] = pts;
                MeshBuild mb;
                SweepOptions o;
                o.seg = 10;
                o.aspect = 0.62f;
                o.colour = colour;
                o.up = Vector3(0, 0, 1);
                appendSweep(mb, pts, radiiAlong(n + 1, {{0.f, 0.0105f}, {0.7f, 0.0072f}, {1.f, 0.0045f}}), o);
                g.add(mb.mesh(orange, true));
                // the claw
                const Vector3 tip = pts.back();
                auto claw = Mesh::create(ConeGeometry::create(0.0042f, 0.014f, 8), standard(lin(40, 34, 30), 0.4f));
                claw->position.copy(tip + dir * 0.005f - Vector3(0, 0.002f, 0));
                alignY(*claw, dir * 1.f + Vector3(0, -0.4f, 0));
                g.add(finish(claw));
            }
            // the web: a membrane between each neighbouring pair, from near the toes' roots out toward their tips
            const int pairs[3][2] = {{0, 1}, {1, 2}, {3, 0}};
            MeshBuild wb;
            for (const auto& pr : pairs) {
                const auto& A = lines[static_cast<size_t>(pr[0])];
                const auto& B = lines[static_cast<size_t>(pr[1])];
                const unsigned base = wb.vertexCount();
                const int n = static_cast<int>(A.size());
                const int steps = 6;
                for (int k = 0; k <= steps; ++k) {
                    const float f = 0.12f + 0.74f * static_cast<float>(k) / static_cast<float>(steps);
                    const auto pointOn = [&](const std::vector<Vector3>& L, float ff) {
                        const float x = ff * static_cast<float>(n - 1);
                        const int i0 = std::min(static_cast<int>(x), n - 2);
                        return lerp3(L[static_cast<size_t>(i0)], L[static_cast<size_t>(i0) + 1], x - static_cast<float>(i0));
                    };
                    // the web's free edge is scalloped: shorter in the middle
                    const Vector3 pa = pointOn(A, f), pb = pointOn(B, f);
                    for (int c = 0; c <= 4; ++c) {
                        const float u = static_cast<float>(c) / 4.f;
                        // sag the membrane a hair toward the middle
                        Vector3 q = lerp3(pa, pb, u);
                        q.y -= 0.0012f * std::sin(PI * u);
                        wb.vertex(q, Vector3(0, 1, 0), u, f, V3(0.93f, 0.52f, 0.2f));
                    }
                }
                const unsigned row = 5;
                for (int k = 0; k < steps; ++k)
                    for (int c = 0; c < 4; ++c) {
                        const unsigned a = base + k * row + c, b = a + 1, cc = a + row, d = cc + 1;
                        wb.index.insert(wb.index.end(), {a, b, cc, b, d, cc});
                    }
            }
            wb.smoothNormals();
            g.add(wb.mesh(web, true));
        }

        // ---- the pose ---------------------------------------------------------------------------------------------------
        // foot: where the toes' root goes (bird frame); heelBack: the direction the ankle lies from it (unit, bird frame);
        // forward: the foot's forward (unit, bird frame, horizontal in the world)
        void solve(const Vector3& footTarget, const Vector3& heelBack, const Vector3& forward, const Vector3& up) {
            foot = footTarget;
            ankle = foot + heelBack * tarsus;
            Vector3 hd = ankle - hip;
            float d = hd.length();
            const float dMax = femur + tibia - 0.004f, dMin = std::abs(femur - tibia) + 0.01f;
            Vector3 u = Vector3(hd).normalize();
            if (d > dMax) {
                ankle = hip + u * dMax;
                d = dMax;
            }
            if (d < dMin) {
                ankle = hip + u * dMin;
                d = dMin;
            }
            const float x = (femur * femur - tibia * tibia + d * d) / (2.f * d);
            const float h = std::sqrt(std::max(femur * femur - x * x, 0.f));
            // the knee points forward and out under the plumage, never up through the back
            Vector3 pole = forward * 1.0f - up * 0.25f + Vector3(0, 0, side * 0.55f);
            pole = (pole - u * pole.dot(u)).normalize();
            knee = hip + u * x + pole * h;
            // orient the parts: +Y along each bone
            femurGroup->position.copy(hip - Vector3(knee - hip).normalize() * femurInside);
            alignY(*femurGroup, knee - hip);
            tibiaGroup->position.copy(knee);
            alignY(*tibiaGroup, ankle - knee);
            tarsusGroup->position.copy(ankle);
            alignY(*tarsusGroup, foot - ankle);
            // the foot: x forward, y up
            footGroup->position.copy(foot);
            Vector3 fx = (forward - up * forward.dot(up)).normalize();
            Vector3 fz = Vector3(fx).cross(up).normalize();
            Vector3 fy = Vector3(fz).cross(fx).normalize();
            Matrix4 m;
            m.makeBasis(fx, fy, fz);
            footGroup->quaternion.setFromRotationMatrix(m);
        }
    };

}// namespace pb

#endif
