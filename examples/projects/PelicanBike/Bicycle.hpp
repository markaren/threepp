// A classic steel roadster, modelled part by part: the diamond frame with its lugs and fork, swept-back bars with
// leather grips, a sprung leather saddle, 36-spoke wheels (spokes laced three-cross, an instance each), rubber on
// alloy rims, a toothed chainring and cog with a chain of a hundred links that really runs round them, cranks and
// platform pedals, mud guards, lamp, bell, rear light and a kickstand.
//
// Coordinates: x forward, y up, z to the rider's right; the rear axle at (0, R, 0), the ground at y = 0.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_BICYCLE_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_BICYCLE_HPP

#include "Parts.hpp"
#include "Fish.hpp"
#include "Surfaces.hpp"

namespace pb {

    struct BikeMaterials {
        std::shared_ptr<MeshPhysicalMaterial> paint, paintCream;
        std::shared_ptr<MeshStandardMaterial> chrome, alloy, steel, spoke, rubber, tireTread, black, brass, leather, grip, glass, chain, enamel;
        std::shared_ptr<MeshStandardMaterial> reflectorRed, reflectorOrange, cable, wicker;

        BikeMaterials() {
            paint = physical(lin(86, 186, 172), 0.34f);
            paint->clearcoat = 1.f;
            paint->clearcoatRoughness = 0.06f;
            paintCream = physical(lin(236, 226, 196), 0.36f);
            paintCream->clearcoat = 1.f;
            paintCream->clearcoatRoughness = 0.08f;
            chrome = standard(lin(235, 236, 238), 0.1f, 1.f);
            alloy = standard(lin(200, 202, 206), 0.28f, 1.f);
            steel = standard(lin(150, 152, 156), 0.38f, 1.f);
            spoke = standard(lin(215, 217, 220), 0.3f, 1.f);
            rubber = standard(lin(26, 26, 28), 0.82f);
            tireTread = standard(lin(255, 255, 255), 0.88f);
            tireTread->vertexColors = true;
            black = standard(lin(22, 22, 24), 0.45f);
            brass = standard(lin(214, 168, 78), 0.28f, 1.f);
            leather = standard(lin(255, 255, 255), 0.55f);
            grip = standard(lin(255, 255, 255), 0.6f);
            glass = standard(lin(200, 215, 225), 0.04f);
            glass->transparent = true;
            glass->opacity = 0.35f;
            chain = standard(lin(92, 90, 88), 0.42f, 1.f);
            enamel = standard(lin(236, 226, 196), 0.4f);
            reflectorRed = standard(lin(190, 20, 20), 0.2f);
            reflectorRed->emissive = lin(60, 0, 0);
            reflectorOrange = standard(lin(230, 120, 20), 0.2f);
            cable = standard(lin(18, 18, 20), 0.5f);
            wicker = standard(lin(255, 255, 255), 0.85f);
        }
    };

    struct Bicycle {
        // ---- geometry (metres) -------------------------------------------------------------------------------------
        static constexpr float R = 0.335f;         // tyre outer radius
        static constexpr float tubeR = 0.0215f;    // tyre section radius
        static constexpr float crankLen = 0.145f;
        static constexpr int ringTeeth = 44, cogTeeth = 18;
        static constexpr float pitch = 0.0127f;
        static constexpr float chainLine = 0.047f; // z of the chain's plane
        static constexpr float dropoutZ = 0.0675f, forkZ = 0.05f;

        Vector3 bb, rearAxle, frontAxle, seatTop, headTop, headBottom, steerAxisUp;
        float ringR = 0, cogR = 0;
        float sinceRound = 0;

        std::shared_ptr<Group> root = Group::create();
        std::shared_ptr<Group> rearWheel = Group::create(), frontWheel = Group::create(), cranks = Group::create(), steering = Group::create();
        std::shared_ptr<Group> pedalR = Group::create(), pedalL = Group::create();
        std::shared_ptr<InstancedMesh> chainOuter, chainInner;
        Vector3 gripR, gripL;     // where the hands go
        Vector3 saddlePoint;      // the middle of the saddle's top
        Vector3 steerPivot;       // a point on the steering axis

        BikeMaterials M;

        // ---- the pose, set each frame -------------------------------------------------------------------------------
        // phase: the crank's angle, clockwise seen from the right, 0 with the right crank pointing forward
        void pose(float phase, float steer) {
            const float a = -phase;
            cranks->rotation.z = a;
            const float wheelAngle = -phase * static_cast<float>(ringTeeth) / static_cast<float>(cogTeeth);
            rearWheel->rotation.z = wheelAngle;
            frontWheel->rotation.z = wheelAngle;
            pedalR->rotation.z = -a;
            pedalL->rotation.z = -a;
            // the steering pivots about the fork axis
            steering->quaternion.setFromAxisAngle(steerAxisUp, steer);
            steering->position.copy(steerPivot);
            steering->position.sub(Vector3(steerPivot).applyQuaternion(steering->quaternion));
            updateChain(phase);
        }

        [[nodiscard]] Vector3 pedalCentre(int side, float phase) const {
            const float a = -phase + (side > 0 ? 0.f : PI);
            return Vector3(bb.x + crankLen * std::cos(a), bb.y + crankLen * std::sin(a), side > 0 ? 0.1295f : -0.1295f);
        }

        // ---- the chain -------------------------------------------------------------------------------------------------
        struct ChainPath {
            Vector3 c1, c2;
            float r1, r2, d, gamma, beta, run, arc1, arc2, length;
            float aTop, aBottom;
        } cp;

        void buildChainPath() {
            cp.c1 = Vector3(bb.x, bb.y, chainLine);
            cp.c2 = Vector3(rearAxle.x, rearAxle.y, chainLine);
            cp.r1 = ringR;
            cp.r2 = cogR;
            const Vector3 dv = cp.c2 - cp.c1;
            cp.d = std::hypot(dv.x, dv.y);
            cp.beta = std::atan2(dv.y, dv.x);
            cp.gamma = std::acos((cp.r1 - cp.r2) / cp.d);
            cp.aTop = cp.beta - cp.gamma;   // the normal of the upper common tangent
            cp.aBottom = cp.beta + cp.gamma;
            cp.run = std::sqrt(cp.d * cp.d - (cp.r1 - cp.r2) * (cp.r1 - cp.r2));
            // clockwise round the ring from top to bottom: the angle falls from aTop to aBottom - 2pi
            cp.arc1 = cp.aTop - (cp.aBottom - TAU);
            cp.arc2 = cp.aBottom - cp.aTop;
            cp.length = cp.arc1 * cp.r1 + cp.run + cp.arc2 * cp.r2 + cp.run;
        }

        // a point of the closed chain path, s from the ring's top tangent point, clockwise
        [[nodiscard]] Vector3 chainPoint(float s) const {
            s = std::fmod(s, cp.length);
            if (s < 0) s += cp.length;
            const auto onCircle = [&](const Vector3& c, float r, float a) { return Vector3(c.x + r * std::cos(a), c.y + r * std::sin(a), chainLine); };
            // 1. the ring's arc
            const float l1 = cp.arc1 * cp.r1;
            if (s < l1) return onCircle(cp.c1, cp.r1, cp.aTop - s / cp.r1);
            s -= l1;
            // 2. the lower run, back to the cog
            if (s < cp.run) {
                const Vector3 a = onCircle(cp.c1, cp.r1, cp.aBottom), b = onCircle(cp.c2, cp.r2, cp.aBottom);
                return lerp3(a, b, s / cp.run);
            }
            s -= cp.run;
            // 3. the cog's arc, clockwise from the bottom tangent to the top
            const float l2 = cp.arc2 * cp.r2;
            if (s < l2) return onCircle(cp.c2, cp.r2, cp.aBottom - s / cp.r2);
            s -= l2;
            // 4. the upper run, forward to the ring
            const Vector3 a = onCircle(cp.c2, cp.r2, cp.aTop), b = onCircle(cp.c1, cp.r1, cp.aTop);
            return lerp3(a, b, s / cp.run);
        }

        void updateChain(float phase) {
            const int links = 100;
            const float spacing = cp.length / static_cast<float>(links);
            const float shift = phase * cp.r1 + 0.f;
            Matrix4 m;
            std::vector<Vector3> pin(static_cast<size_t>(links) + 1);
            for (int i = 0; i <= links; ++i) pin[static_cast<size_t>(i)] = chainPoint(shift + spacing * static_cast<float>(i));
            for (int i = 0; i < links; ++i) {
                const Vector3 a = pin[static_cast<size_t>(i)], b = pin[static_cast<size_t>(i) + 1];
                const Vector3 mid = (a + b) * 0.5f;
                Vector3 x = (b - a);
                const float len = x.length();
                x.normalize();
                const Vector3 z(0, 0, 1);
                const Vector3 y = Vector3(z).cross(x);
                m.makeBasis(x, y, z);
                m.setPosition(mid);
                // a link a little shorter or longer than the standard keeps its plates between its pins
                Matrix4 s;
                s.makeScale(len / pitch, 1.f, 1.f);
                m.multiply(s);
                auto& target = (i % 2 == 0) ? chainOuter : chainInner;
                target->setMatrixAt(static_cast<size_t>(i / 2), m);
            }
            chainOuter->instanceMatrix()->needsUpdate();
            chainInner->instanceMatrix()->needsUpdate();
        }

        // ---- construction ----------------------------------------------------------------------------------------------
        Bicycle() {
            const SurfaceSet surf;
            M.leather->map = surf.leather.albedo;
            M.leather->normalMap = surf.leather.normal;
            M.leather->roughnessMap = surf.leather.orm;
            M.leather->aoMap = surf.leather.orm;
            M.grip->map = surf.grip.albedo;
            M.grip->normalMap = surf.grip.normal;
            M.grip->roughnessMap = surf.grip.orm;
            M.grip->aoMap = surf.grip.orm;
            M.wicker->map = surf.wicker.albedo;
            M.wicker->normalMap = surf.wicker.normal;
            M.wicker->roughnessMap = surf.wicker.orm;
            M.wicker->aoMap = surf.wicker.orm;
            M.wicker->normalScale.set(1.f, 1.f);

            // ---- the frame's points
            bb = Vector3(std::sqrt(0.44f * 0.44f - 0.03f * 0.03f), R - 0.03f, 0.f);
            rearAxle = Vector3(0.f, R, 0.f);
            ringR = static_cast<float>(ringTeeth) * pitch / TAU;
            cogR = static_cast<float>(cogTeeth) * pitch / TAU;
            const float seatAngle = 72.f * PI / 180.f;
            seatTop = bb + Vector3(-std::cos(seatAngle), std::sin(seatAngle), 0.f) * 0.32f;
            const float headAngle = 70.f * PI / 180.f;
            const Vector3 down(std::cos(headAngle), -std::sin(headAngle), 0.f);   // the head axis, top to bottom
            headTop = Vector3(0.755f, 0.800f, 0.f);
            headBottom = headTop + down * 0.11f;
            steerAxisUp = Vector3(-down.x, -down.y, 0.f);
            steerPivot = headTop;
            {
                const float rake = 0.05f;
                const Vector3 nrm(std::sin(headAngle), std::cos(headAngle), 0.f);
                const float t = (headBottom.y - (R - rake * nrm.y)) / std::sin(headAngle);
                frontAxle = headBottom + down * t + nrm * rake;
                frontAxle.y = R;
            }
            buildChainPath();

            buildFrame();
            buildFork();
            buildBars();
            buildBasket();
            buildSaddle();
            buildWheel(*rearWheel, rearAxle, true);
            buildWheel(*frontWheel, frontAxle - Vector3(0, 0, 0), false);
            buildDrivetrain();
            buildFenders();
            buildFittings();
            root->add(rearWheel);
            root->add(frontWheel);
            root->add(cranks);
            root->add(steering);
            pose(0.f, 0.f);
        }

        // ---- the frame -------------------------------------------------------------------------------------------------
        void buildFrame() {
            const auto& P = M.paint;
            auto g = Group::create();
            const Vector3 seatMid = bb + (seatTop - bb) * 0.93f;   // where the seat stays and top tube meet the seat tube
            const Vector3 headTT = headTop + (headBottom - headTop) * 0.14f;
            const Vector3 headDT = headTop + (headBottom - headTop) * 0.82f;
            // main triangle
            g->add(rod(bb - Vector3(0, 0, 0), seatTop + (seatTop - bb).normalize() * 0.03f, 0.0148f, 0.0148f, P, 24));
            g->add(rod(Vector3(seatTop.x + 0.004f, seatTop.y - 0.012f, 0), Vector3(headTT.x, headTT.y - 0.004f, 0), 0.0135f, 0.0135f, P, 24));
            g->add(rod(bb + Vector3(0.012f, 0.012f, 0), headDT, 0.0158f, 0.0158f, P, 24));
            g->add(rod(headTop + (headBottom - headTop) * 0.0f, headBottom, 0.0185f, 0.0185f, P, 28));
            // lug collars: rings over the joints
            const auto collar = [&](const Vector3& a, const Vector3& b, float r, const std::shared_ptr<Material>& m) { g->add(rod(a, b, r, r, m, 28)); };
            collar(headTop - down() * 0.0f, headTop + (headBottom - headTop) * 0.12f, 0.0205f, M.chrome);
            collar(headTop + (headBottom - headTop) * 0.88f, headBottom, 0.0205f, M.chrome);
            collar(seatTop - (seatTop - bb).normalize() * 0.028f, seatTop + (seatTop - bb).normalize() * 0.012f, 0.0178f, P);
            // bottom bracket shell and its cups
            {
                auto shell = Mesh::create(CylinderGeometry::create(0.0225f, 0.0225f, 0.072f, 28), P);
                shell->rotation.x = PI / 2.f;
                shell->position.copy(bb);
                g->add(finish(shell));
                for (int s : {1, -1}) {
                    auto cup = Mesh::create(CylinderGeometry::create(0.0245f, 0.0245f, 0.008f, 28), M.chrome);
                    cup->rotation.x = PI / 2.f;
                    cup->position.set(bb.x, bb.y, 0.04f * static_cast<float>(s));
                    g->add(finish(cup));
                }
            }
            // stays, both sides
            for (int s : {1, -1}) {
                const float z = dropoutZ * static_cast<float>(s);
                const Vector3 drop(rearAxle.x, rearAxle.y, z);
                g->add(rod(Vector3(bb.x, bb.y, 0.028f * static_cast<float>(s)), drop + Vector3(0.012f, 0.0f, 0), 0.0125f, 0.0085f, P, 20));
                g->add(rod(Vector3(seatMid.x, seatMid.y, 0.011f * static_cast<float>(s)), drop + Vector3(0.0f, 0.014f, 0), 0.0075f, 0.0065f, P, 16));
                // dropout plates
                g->add(box(Vector3(drop.x - 0.008f, drop.y + 0.0f, z), 0.04f, 0.036f, 0.0055f, M.steel));
            }
            // the seat stay bridge, above the tyre where the stays meet (the rear mud guard bolts to it), and the chain stay
            // bridge, ahead of the tyre behind the bottom bracket
            {
                const Vector3 eyelet(rearAxle.x, rearAxle.y + 0.014f, 0.f);
                const float t = 0.93f;
                const Vector3 p = lerp3(eyelet, seatMid, t);
                const float zz = dropoutZ + (0.011f - dropoutZ) * t;
                seatBridge = p;
                g->add(rod(p + Vector3(0, 0, zz), p - Vector3(0, 0, zz), 0.0058f, 0.0058f, P, 14));
                const float cx = 0.385f;
                const float f = (bb.x - cx) / bb.x;
                const float zc = 0.028f + f * (dropoutZ - 0.028f);
                const float yc = bb.y + f * (rearAxle.y - bb.y);
                chainBridge = Vector3(cx, yc, 0.f);
                g->add(rod(Vector3(cx, yc + 0.003f, zc), Vector3(cx, yc + 0.003f, -zc), 0.0058f, 0.0058f, P, 14));
            }
            // seat post, quick: a plain steel post with a clamp bolt
            {
                const Vector3 up = (seatTop - bb).normalize();
                const Vector3 top = seatTop + up * 0.035f;
                g->add(rod(seatTop - up * 0.05f, top, 0.0127f, 0.0127f, M.chrome, 20));
                saddleClamp = top;
                saddleUp = up;
            }
            // head badge
            {
                auto badge = Mesh::create(CylinderGeometry::create(0.0135f, 0.0135f, 0.002f, 24), M.brass);
                const Vector3 at = headTop + (headBottom - headTop) * 0.5f + Vector3(0.0f, 0.0f, 0.019f);
                badge->rotation.x = PI / 2.f;
                badge->position.copy(at + Vector3(0, 0, 0.0));
                badge->position.z = 0.0185f;
                g->add(finish(badge));
                auto badge2 = Mesh::create(CylinderGeometry::create(0.0135f, 0.0135f, 0.002f, 24), M.brass);
                badge2->rotation.x = PI / 2.f;
                badge2->position.copy(at);
                badge2->position.z = -0.0185f;
                g->add(finish(badge2));
            }
            // bottle-less downtube decal band in cream
            {
                const Vector3 a = bb + (headDT - bb) * 0.30f, b = bb + (headDT - bb) * 0.36f;
                g->add(rod(a, b, 0.0163f, 0.0163f, M.paintCream, 28));
                const Vector3 a2 = seatTop - (seatTop - bb).normalize() * 0.17f, b2 = seatTop - (seatTop - bb).normalize() * 0.21f;
                g->add(rod(a2, b2, 0.0152f, 0.0152f, M.paintCream, 28));
            }
            root->add(g);
        }
        Vector3 saddleClamp, saddleUp, seatBridge, chainBridge;

        [[nodiscard]] Vector3 down() const { return Vector3(-steerAxisUp.x, -steerAxisUp.y, 0.f); }

        // ---- fork, headset, stem and bars (they turn with the steering) --------------------------------------------------
        void buildFork() {
            const auto& P = M.paint;
            auto g = Group::create();
            // blades: from the crown's shoulders down the steering axis, bending forward to the dropouts
            for (int s : {1, -1}) {
                const float z = forkZ * static_cast<float>(s);
                const Vector3 crown = headBottom + down() * 0.012f + Vector3(0, 0, z);
                const Vector3 drop(frontAxle.x, frontAxle.y, z);
                const std::vector<Vector3> ctrl = {crown, lerp3(crown, drop, 0.28f), lerp3(crown, drop, 0.62f) + Vector3(0.0018f, 0, 0), drop + Vector3(0, 0.012f, 0)};
                const auto pts = splinePoints(ctrl, 14);
                MeshBuild mb;
                SweepOptions o;
                o.seg = 18;
                o.aspect = 0.8f;
                o.up = Vector3(0, 0, 1);
                appendSweep(mb, pts, radiiAlong(static_cast<int>(pts.size()), {{0.f, 0.0125f}, {0.6f, 0.0098f}, {1.f, 0.0072f}}), o);
                g->add(mb.mesh(P, false));
                g->add(box(drop + Vector3(-0.004f, -0.012f, 0), 0.034f, 0.034f, 0.005f, M.steel));
            }
            // crown
            {
                const Vector3 c = headBottom + down() * 0.006f;
                g->add(rod(c + Vector3(0, 0, forkZ), c - Vector3(0, 0, forkZ), 0.0145f, 0.0145f, P, 20));
            }
            // headset cups, steerer, quill stem
            const Vector3 up = steerAxisUp;
            g->add(rod(headTop, headTop + up * 0.02f, 0.0215f, 0.0215f, M.chrome, 28));
            g->add(rod(headBottom - up * 0.004f, headBottom + down() * 0.012f, 0.0215f, 0.0215f, M.chrome, 28));
            g->add(rod(headTop + up * 0.02f, headTop + up * 0.032f, 0.0172f, 0.0172f, M.chrome, 24));
            const Vector3 quillTop = headTop + up * 0.08f;
            g->add(rod(headTop + up * 0.032f, quillTop, 0.0105f, 0.0105f, M.chrome, 20));
            // the clamp, a short forward extension carrying the bars
            barCentre = quillTop + Vector3(0.026f, 0.014f, 0.f);
            g->add(rod(quillTop - Vector3(0, 0.004f, 0), barCentre, 0.0145f, 0.0145f, M.chrome, 20));
            g->add(rod(barCentre + Vector3(0, 0, 0.026f), barCentre - Vector3(0, 0, 0.026f), 0.0135f, 0.0135f, M.chrome, 20));
            root_steering = g;
            steering->add(g);
        }
        Vector3 barCentre;
        std::shared_ptr<Group> root_steering;

        void buildBars() {
            auto& g = *root_steering;
            for (int s : {1, -1}) {
                const float sg = static_cast<float>(s);
                // the swept bar: forward and out from the clamp, then back to the grip
                std::vector<Vector3> ctrl = {
                        barCentre + Vector3(0.0f, 0.0f, 0.f),
                        barCentre + Vector3(0.002f, 0.001f, 0.07f * sg),
                        barCentre + Vector3(0.006f, 0.004f, 0.14f * sg),
                        barCentre + Vector3(-0.004f, 0.012f, 0.20f * sg),
                        barCentre + Vector3(-0.045f, 0.022f, 0.252f * sg),
                        barCentre + Vector3(-0.105f, 0.028f, 0.287f * sg),
                        barCentre + Vector3(-0.165f, 0.027f, 0.300f * sg),
                        barCentre + Vector3(-0.235f, 0.024f, 0.304f * sg)};
                const auto pts = splinePoints(ctrl, 10);
                MeshBuild mb;
                SweepOptions o;
                o.seg = 18;
                o.up = Vector3(0, 1, 0);
                appendSweep(mb, pts, std::vector<float>(pts.size(), 0.0111f), o);
                g.add(mb.mesh(M.chrome, false));
                // the grip: a leather sleeve over the last stretch, with a flared end
                const Vector3 a = barCentre + Vector3(-0.115f, 0.0275f, 0.2915f * sg);
                const Vector3 e = pts.back();
                std::vector<Vector3> gp;
                const int n = 24;
                for (int i = 0; i <= n; ++i) gp.push_back(lerp3(a, e, static_cast<float>(i) / static_cast<float>(n)));
                MeshBuild gm;
                SweepOptions go;
                go.seg = 22;
                go.uvAlong = 6.f;
                const auto gr = radiiAlong(n + 1, {{0.f, 0.0145f}, {0.06f, 0.0158f}, {0.8f, 0.0168f}, {0.93f, 0.0172f}, {1.f, 0.0125f}});
                appendSweep(gm, gp, gr, go);
                g.add(gm.mesh(M.grip, false));
                (sg > 0 ? gripR : gripL) = lerp3(a, e, 0.5f);
            }
            // brake levers: a bracket on the bar, the blade reaching round the grip
            for (int s : {1, -1}) {
                const float sg = static_cast<float>(s);
                const Vector3 mount = barCentre + Vector3(-0.115f, 0.0275f, 0.2915f * sg) + Vector3(0.025f, 0.0f, 0.0f);
                g.add(box(mount + Vector3(0.0f, 0.005f, 0.0f), 0.034f, 0.016f, 0.026f, M.black));
                std::vector<Vector3> lp = {mount + Vector3(0.012f, -0.004f, 0.0f), mount + Vector3(0.050f, -0.016f, 0.0f), mount + Vector3(0.092f, -0.034f, 0.0f), mount + Vector3(0.122f, -0.068f, 0.0f)};
                // the lever is a flat blade on the bar's front, turned into the palm
                MeshBuild lv;
                SweepOptions lo;
                lo.seg = 10;
                lo.aspect = 1.9f;
                lo.up = Vector3(0, 0, 1);
                const auto pts = splinePoints(lp, 8);
                appendSweep(lv, pts, radiiAlong(static_cast<int>(pts.size()), {{0.f, 0.0052f}, {1.f, 0.0036f}}), lo);
                g.add(lv.mesh(M.alloy, false));
            }
            // a bell on the bar, brass
            {
                const Vector3 at = barCentre + Vector3(0.0f, 0.0f, 0.105f);
                std::vector<std::pair<float, float>> prof = {{0.0f, 0.0f}, {0.012f, 0.0f}, {0.026f, 0.004f}, {0.034f, 0.014f}, {0.036f, 0.022f}, {0.0f, 0.036f}};
                MeshBuild mb;
                appendLathe(mb, prof, 28, at + Vector3(0.0f, 0.016f, 0.0f), Vector3(0, 1, 0));
                g.add(mb.mesh(M.brass, false));
                g.add(rod(at, at + Vector3(0, 0.02f, 0), 0.0095f, 0.0095f, M.black, 14));
                g.add(rod(at + Vector3(0, 0.0f, 0.0f), at + Vector3(0.0f, 0.0f, 0.0f), 0.0f, 0.0f, M.black, 4));
                // the striker's lever
                g.add(rod(at + Vector3(0.0f, 0.012f, 0.028f), at + Vector3(0.0f, 0.016f, 0.05f), 0.0035f, 0.0035f, M.chrome, 10));
            }
            // headlamp: a chrome shell on a bracket from the fork crown
            {
                const Vector3 at = headBottom + down() * 0.035f + Vector3(0.062f, 0.095f, 0.0f);
                std::vector<std::pair<float, float>> prof = {{0.0f, -0.034f}, {0.026f, -0.032f}, {0.046f, -0.02f}, {0.052f, 0.0f}, {0.051f, 0.02f}, {0.046f, 0.036f}, {0.04f, 0.044f}};
                MeshBuild mb;
                appendLathe(mb, prof, 36, at, Vector3(1, 0, 0));
                g.add(mb.mesh(M.chrome, false));
                auto lens = Mesh::create(SphereGeometry::create(0.0415f, 28, 14, 0, TAU, 0, PI / 2.f), M.glass);
                lens->rotation.z = -PI / 2.f;
                lens->position.copy(at + Vector3(0.0425f, 0.0f, 0.0f));
                lens->scale.set(1.f, 1.f, 1.f);
                g.add(finish(lens, false, false));
                auto reflector = Mesh::create(SphereGeometry::create(0.034f, 24, 10, 0, TAU, 0, PI / 2.f), M.chrome);
                reflector->rotation.z = -PI / 2.f;
                reflector->position.copy(at + Vector3(0.012f, 0.0f, 0.0f));
                g.add(finish(reflector, false, false));
                auto bulb = Mesh::create(SphereGeometry::create(0.011f, 14, 8), standard(lin(255, 240, 200), 0.3f));
                bulb->position.copy(at + Vector3(0.028f, 0.0f, 0.0f));
                g.add(finish(bulb, false, false));
                // the stay to the crown and the bracket
                g.add(rod(headBottom + down() * 0.012f, at - Vector3(0.012f, 0.034f, 0.0f), 0.0045f, 0.0045f, M.chrome, 12));
                // the lamp's two ears
                for (int s : {1, -1}) g.add(ball(at + Vector3(-0.004f, -0.03f, 0.034f * static_cast<float>(s)), 0.006f, M.chrome, 12));
            }
        }


        // ---- a wicker basket on the bars, with the day's catch --------------------------------------------------------------
        void buildBasket() {
            auto& g = *root_steering;
            const Vector3 at(barCentre.x + 0.075f, barCentre.y - 0.035f, 0.f);   // the middle of its bottom
            auto basket = Group::create();
            basket->position.copy(at);
            // the shell: a rounded box widening upward, woven
            Loft L;
            const float H = 0.19f;
            L.spine.build({{0.f, 0.f, 0.f}, {0.f, 0.08f, 0.f}, {0.f, H, 0.f}}, 120);
            L.rings = 40;
            L.around = 64;
            L.stations = {{0.f, 0.128f, 0.095f, 0.095f, 3.4f, 0.f}, {0.5f, 0.142f, 0.105f, 0.105f, 3.2f, 0.f}, {1.f, 0.158f, 0.118f, 0.118f, 3.0f, 0.f}};
            MeshBuild shell;
            L.build(shell, nullptr, 1.f / 0.24f, false);
            for (size_t i = 0; i < shell.uv.size(); i += 2) shell.uv[i] *= 3.f;   // u runs three weaves round
            auto wick = M.wicker;
            wick->side = Side::Double;
            basket->add(shell.mesh(wick, false));
            // the bottom
            {
                auto bottom = Mesh::create(BoxGeometry::create(0.2f, 0.01f, 0.25f), wick);
                bottom->position.set(0.f, 0.005f, 0.f);
                basket->add(finish(bottom));
            }
            // the rim and base ring, a rolled cane
            for (float t : {0.f, 1.f}) {
                std::vector<Vector3> pts;
                for (int i = 0; i <= 64; ++i) pts.push_back(L.surface(t, TAU * static_cast<float>(i) / 64.f));
                MeshBuild ring;
                SweepOptions o;
                o.seg = 10;
                o.capStart = o.capEnd = false;
                appendSweep(ring, pts, std::vector<float>(pts.size(), t > 0.5f ? 0.0085f : 0.006f), o);
                for (size_t i = 0; i < ring.uv.size(); i += 2) ring.uv[i] *= 8.f;
                basket->add(ring.mesh(wick, false));
            }
            // a few upright stakes show through the weave: rods running up the outside
            for (int i = 0; i < 12; ++i) {
                const float a = TAU * static_cast<float>(i) / 12.f;
                std::vector<Vector3> pts;
                for (int k = 0; k <= 8; ++k) pts.push_back(L.surface(static_cast<float>(k) / 8.f, a) + Vector3(0.f, 0.f, 0.f));
                MeshBuild st;
                SweepOptions o;
                o.seg = 6;
                appendSweep(st, pts, std::vector<float>(pts.size(), 0.0042f), o);
                basket->add(st.mesh(wick, false));
            }
            // the catch: a mackerel with its nose on the floor, leaning back against the rear rim with its tail out over it,
            // so it is held at two points (the floor and the cane) and the tail points toward the bird
            auto fish = makeFish(0.34f);
            {
                const Vector3 head(0.0f, 0.032f, -0.05f), contact(-0.116f, 0.231f, 0.045f);
                const Vector3 dx = (contact - head).normalize();
                Vector3 dy = Vector3(0.f, 1.f, 0.f) - dx * dx.y;   // its back toward the sky
                dy.normalize();
                const Vector3 dz = Vector3(dx).cross(dy);
                Matrix4 m;
                m.makeBasis(dx, dy, dz);
                fish->quaternion.setFromRotationMatrix(m);
                fish->position.copy(head);
            }
            basket->add(fish);
            g.add(basket);
            // fixings: a rod from the basket's back to the stem and two stays down to the lamp bracket
            const Vector3 backMid = at + Vector3(-0.095f, 0.09f, 0.f);
            g.add(rod(backMid, barCentre + Vector3(0.f, -0.012f, 0.f), 0.0045f, 0.0045f, M.chrome, 10));
            for (int s : {1, -1}) {
                const float z = static_cast<float>(s) * 0.06f;
                g.add(rod(at + Vector3(-0.02f, 0.0f, z), headBottom + down() * 0.012f + Vector3(0.008f, 0.012f, static_cast<float>(s) * 0.03f), 0.0035f, 0.0035f, M.chrome, 8));
                // a strap round the bar
                g.add(rod(barCentre + Vector3(0.f, 0.018f, z * 0.6f), at + Vector3(-0.07f, 0.1f, z), 0.003f, 0.003f, M.chrome, 8));
            }
        }

        // ---- the saddle ------------------------------------------------------------------------------------------------
        void buildSaddle() {
            // a leather top, long and narrow at the nose, with a sag between the hips
            const Vector3 origin = saddleClamp + saddleUp * 0.02f + Vector3(0.018f, 0.0f, 0.0f);
            saddlePoint = origin + Vector3(0.0f, 0.027f, 0.0f);
            MeshBuild top;
            const int nx = 42, na = 34;
            const float x0 = -0.14f, x1 = 0.135f;
            for (int i = 0; i <= nx; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(nx);
                const float x = x0 + (x1 - x0) * u;
                // half width: 0.095 at the back, 0.075 mid, 0.016 at the nose
                const float hw = u < 0.55f ? 0.098f - 0.022f * (u / 0.55f) * (u / 0.55f) : 0.076f * std::pow(1.f - (u - 0.55f) / 0.45f, 0.62f) + 0.014f;
                // height of the top: the back lifted, the middle sagging, the nose lifting a little
                const float yy = 0.018f * std::pow(1.f - u, 1.8f) - 0.011f * smooth(0.18f, 0.5f, u) * (1.f - smooth(0.5f, 0.9f, u)) + 0.010f * smooth(0.7f, 1.f, u);
                for (int k = 0; k <= na; ++k) {
                    const float a = TAU * static_cast<float>(k) / static_cast<float>(na);
                    // an oblate ring: broad across, thin
                    const float c = std::cos(a), sn = std::sin(a);
                    const float ex = 3.f;
                    const float cx = sgnpow(c, 2.f / ex), cy = sgnpow(sn, 2.f / ex);
                    const float thick = 0.016f * (1.f - 0.45f * smooth(0.6f, 1.f, u));
                    const Vector3 p = origin + Vector3(x, yy + (cy > 0 ? thick : -0.012f) * cy - 0.0f, hw * cx);
                    Vector3 nrm(0, (cy > 0 ? 1.f / thick : 1.f / 0.012f) * std::abs(sgnpow(std::abs(cy), ex - 1.f)) * (cy >= 0 ? 1.f : -1.f), (hw > 1e-4f ? 1.f / hw : 0.f) * sgnpow(std::abs(cx), ex - 1.f) * (cx >= 0 ? 1.f : -1.f));
                    nrm.normalize();
                    top.vertex(p, nrm, u * 1.8f, static_cast<float>(k) / static_cast<float>(na) * 1.2f, V3(1, 1, 1));
                }
            }
            const unsigned row = static_cast<unsigned>(na + 1);
            for (int i = 0; i < nx; ++i)
                for (int k = 0; k < na; ++k) {
                    const unsigned a = i * row + k, b = a + 1, c = a + row, d = c + 1;
                    top.index.insert(top.index.end(), {a, c, b, b, c, d});
                }
            // the order of the section ring runs +z through +y: flip so the faces look out
            top.smoothNormals();
            auto mesh = top.mesh(M.leather, false);
            // the ring winding was chosen to match smoothNormals' outward test: verify and flip if the sign is wrong
            root->add(mesh);
            // the rails: two steel wires from the nose rivet back to the springs, joined by a clamp
            for (int s : {1, -1}) {
                const float sg = static_cast<float>(s);
                std::vector<Vector3> ctrl = {
                        origin + Vector3(0.115f, -0.016f, 0.012f * sg),
                        origin + Vector3(0.060f, -0.018f, 0.032f * sg),
                        origin + Vector3(0.005f, -0.020f, 0.036f * sg),
                        origin + Vector3(-0.055f, -0.017f, 0.050f * sg),
                        origin + Vector3(-0.108f, -0.010f, 0.070f * sg)};
                const auto pts = splinePoints(ctrl, 8);
                MeshBuild mb;
                SweepOptions o;
                o.seg = 10;
                appendSweep(mb, pts, std::vector<float>(pts.size(), 0.0034f), o);
                root->add(mb.mesh(M.chrome, false));
                // the coil spring under the rear
                const Vector3 top0 = origin + Vector3(-0.098f, -0.007f, 0.070f * sg);
                const Vector3 bot0 = origin + Vector3(-0.098f, -0.050f, 0.062f * sg);
                std::vector<Vector3> helix;
                const int turns = 7, per = 12;
                for (int i = 0; i <= turns * per; ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(turns * per);
                    const float a = t * turns * TAU;
                    const Vector3 c = lerp3(bot0, top0, t);
                    helix.push_back(c + Vector3(0.0125f * std::cos(a), 0.0f, 0.0125f * std::sin(a)));
                }
                MeshBuild sp;
                SweepOptions so;
                so.seg = 8;
                appendSweep(sp, helix, std::vector<float>(helix.size(), 0.0021f), so);
                root->add(sp.mesh(M.chrome, false));
            }
            // clamp block on the post
            root->add(box(saddleClamp + Vector3(0.0f, 0.016f, 0.0f), 0.05f, 0.016f, 0.07f, M.steel));
            root->add(rod(saddleClamp + Vector3(0.0f, 0.018f, 0.036f), saddleClamp + Vector3(0.0f, 0.018f, -0.036f), 0.0048f, 0.0048f, M.chrome, 12));
            // rivets along the saddle's edge
            Rng rr(7);
            for (int i = 0; i < 26; ++i) {
                const float u = 0.04f + 0.9f * static_cast<float>(i) / 25.f;
                const float hw = u < 0.55f ? 0.098f - 0.022f * (u / 0.55f) * (u / 0.55f) : 0.076f * std::pow(1.f - (u - 0.55f) / 0.45f, 0.62f) + 0.014f;
                const float yy = 0.018f * std::pow(1.f - u, 1.8f) - 0.011f * smooth(0.18f, 0.5f, u) * (1.f - smooth(0.5f, 0.9f, u)) + 0.010f * smooth(0.7f, 1.f, u);
                for (int s : {1, -1}) {
                    const float x = x0 + (x1 - x0) * u;
                    root->add(ball(origin + Vector3(x, yy + 0.001f, (hw * 0.93f) * static_cast<float>(s)), 0.0028f, M.brass, 8));
                }
            }
        }

        // ---- a wheel -------------------------------------------------------------------------------------------------------
        void buildWheel(Group& wheel, const Vector3& axle, bool rear) {
            wheel.position.copy(axle);
            const float flange = rear ? 0.040f : 0.037f;
            // the tyre: a ring with tan sidewalls and a black tread
            {
                MeshBuild mb;
                const int ringN = 144, tubeN = 30;
                const float major = R - tubeR;
                for (int j = 0; j <= tubeN; ++j) {
                    const float v = TAU * static_cast<float>(j) / static_cast<float>(tubeN);
                    for (int i = 0; i <= ringN; ++i) {
                        const float u = TAU * static_cast<float>(i) / static_cast<float>(ringN);
                        const float cv = std::cos(v), sv = std::sin(v);
                        const float rr = major + tubeR * cv;
                        const Vector3 p(rr * std::cos(u), rr * std::sin(u), tubeR * 0.97f * sv);
                        Vector3 n(cv * std::cos(u), cv * std::sin(u), sv / 0.97f);
                        n.normalize();
                        // tan sidewall where the section turns toward the side, black on the tread
                        const float side = smooth(0.58f, 0.74f, std::abs(sv));
                        const V3 tan = srgb(168, 134, 88), blk = srgb(24, 24, 26);
                        const V3 c = mill::mix(blk, tan, side * (0.9f + 0.1f * std::sin(u * 60.f)));
                        mb.vertex(p, n, static_cast<float>(i) / static_cast<float>(ringN) * 60.f, static_cast<float>(j) / static_cast<float>(tubeN), c);
                    }
                }
                const unsigned row = static_cast<unsigned>(ringN + 1);
                for (int j = 0; j < tubeN; ++j)
                    for (int i = 0; i < ringN; ++i) {
                        const unsigned a = j * row + i, b = a + 1, c = a + row, d = c + 1;
                        mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
                    }
                wheel.add(mb.mesh(M.tireTread));
            }
            // the rim: a polished alloy channel, a hollow section
            {
                MeshBuild mb;
                const int seg = 120;
                // the section as a polyline: inside floor, two walls rising to the bead hooks
                const std::vector<std::pair<float, float>> poly = {
                        {0.2965f, 0.0135f}, {0.2990f, 0.0125f}, {0.2990f, 0.0105f}, {0.2955f, 0.0100f}, {0.2860f, 0.0120f}, {0.2780f, 0.0125f},
                        {0.2755f, 0.0100f}, {0.2755f, -0.0100f}, {0.2780f, -0.0125f}, {0.2860f, -0.0120f}, {0.2955f, -0.0100f}, {0.2990f, -0.0105f},
                        {0.2990f, -0.0125f}, {0.2965f, -0.0135f}};
                const int m = static_cast<int>(poly.size());
                const unsigned base = mb.vertexCount();
                for (int k = 0; k < m; ++k)
                    for (int i = 0; i <= seg; ++i) {
                        const float a = TAU * static_cast<float>(i) / static_cast<float>(seg);
                        const auto [r, z] = poly[static_cast<size_t>(k)];
                        mb.vertex(Vector3(r * std::cos(a), r * std::sin(a), z), Vector3(std::cos(a), std::sin(a), 0.f), static_cast<float>(i) / static_cast<float>(seg) * 20.f, static_cast<float>(k) / static_cast<float>(m), V3(1, 1, 1));
                    }
                const unsigned row = static_cast<unsigned>(seg + 1);
                for (int k = 0; k + 1 < m; ++k)
                    for (int i = 0; i < seg; ++i) {
                        const unsigned a = base + k * row + i, b = a + 1, c = a + row, d = c + 1;
                        mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
                    }
                mb.smoothNormals(base);
                auto rimMat = M.alloy;
                wheel.add(mb.mesh(rimMat, false));
            }
            // hub: a barrel with two flanges, axle ends and locknuts
            {
                const Vector3 o(0, 0, 0);
                MeshBuild mb;
                const std::vector<std::pair<float, float>> prof = {
                        {0.0f, -0.06f}, {0.0060f, -0.06f}, {0.0060f, -0.052f}, {0.0115f, -0.050f}, {0.0115f, -0.040f}, {0.0285f, -0.038f}, {0.0295f, -0.034f}, {0.0285f, -0.0305f}, {0.0135f, -0.0295f},
                        {0.0125f, -0.02f}, {0.0125f, 0.02f}, {0.0135f, 0.0295f}, {0.0285f, 0.0305f}, {0.0295f, 0.034f}, {0.0285f, 0.038f}, {0.0115f, 0.040f}, {0.0115f, 0.050f}, {0.0060f, 0.052f}, {0.0060f, 0.06f}, {0.0f, 0.06f}};
                appendLathe(mb, prof, 36, o, Vector3(0, 0, 1));
                wheel.add(mb.mesh(M.chrome, false));
            }
            // spokes: 36, three-cross, as instances of a thin rod
            {
                auto geo = CylinderGeometry::create(0.00085f, 0.00085f, 1.f, 5, 1, true);
                auto spokes = InstancedMesh::create(geo, M.spoke, 36);
                Matrix4 m;
                for (int side = 0; side < 2; ++side) {
                    const float sg = side == 0 ? 1.f : -1.f;
                    const float zf = (side == 0 ? 1.f : -1.f) * (rear && side == 1 ? 0.034f : flange);
                    for (int i = 0; i < 18; ++i) {
                        const float hubA = TAU * (static_cast<float>(i) + (side == 0 ? 0.f : 0.5f)) / 18.f;
                        const int lead = (i % 2 == 0) ? 6 : -6;
                        const int rimHole = ((2 * i + side + lead) % 36 + 36) % 36;
                        const float rimA = TAU * static_cast<float>(rimHole) / 36.f;
                        const Vector3 a(0.0285f * std::cos(hubA), 0.0285f * std::sin(hubA), zf);
                        const Vector3 b(0.2765f * std::cos(rimA), 0.2765f * std::sin(rimA), 0.0f * sg + 0.0045f * (side == 0 ? 1.f : -1.f));
                        const Vector3 d = b - a;
                        const Vector3 mid = (a + b) * 0.5f;
                        Quaternion q;
                        q.setFromUnitVectors(Vector3(0, 1, 0), Vector3(d).normalize());
                        Vector3 s(1.f, d.length(), 1.f);
                        m.compose(mid, q, s);
                        spokes->setMatrixAt(static_cast<size_t>(side * 18 + i), m);
                    }
                }
                spokes->castShadow = true;
                wheel.add(spokes);
            }
            // the valve, and a wheel reflector clipped to the spokes
            {
                const float a = 0.37f;
                const Vector3 at(0.2975f * std::cos(a), 0.2975f * std::sin(a), 0.0f);
                wheel.add(rod(at, at - Vector3(std::cos(a), std::sin(a), 0.f) * 0.016f, 0.0034f, 0.0034f, M.chrome, 8));
                const float b = 2.2f;
                auto refl = Mesh::create(BoxGeometry::create(0.045f, 0.016f, 0.007f), M.reflectorOrange);
                refl->position.set(0.2f * std::cos(b), 0.2f * std::sin(b), 0.0f);
                refl->rotation.z = b + PI / 2.f;
                wheel.add(finish(refl, false, false));
            }
            if (rear) {
                // the cog on the right side
                MeshBuild mb;
                appendGearRing(mb, cogTeeth, cogR, 0.0032f, 0.0022f, 0.012f, 0.0035f, chainLine - 0.0018f, V3(1, 1, 1));
                wheel.add(mb.mesh(M.steel, false));
                wheel.add(rod(Vector3(0, 0, 0.034f), Vector3(0, 0, 0.0455f), 0.0145f, 0.0145f, M.chrome, 24));
            } else {
                // a dynamo roller
            }
        }

        // ---- cranks, chainring, pedals, chain ----------------------------------------------------------------------------------
        void buildDrivetrain() {
            auto& g = *cranks;
            g.position.copy(bb);
            // the chainring and its spider
            {
                MeshBuild mb;
                appendGearRing(mb, ringTeeth, ringR, 0.0032f, 0.0022f, 0.074f, 0.0035f, chainLine - 0.0018f, V3(1, 1, 1));
                g.add(mb.mesh(M.steel, false));
                // the spider: five arms from the hub of the crank to the ring
                for (int i = 0; i < 5; ++i) {
                    const float a = TAU * static_cast<float>(i) / 5.f + 0.3f;
                    const Vector3 u(std::cos(a), std::sin(a), 0.f);
                    g.add(rod(u * 0.02f + Vector3(0, 0, chainLine + 0.003f), u * 0.078f + Vector3(0, 0, chainLine + 0.003f), 0.0085f, 0.006f, M.alloy, 12));
                    g.add(ball(u * 0.078f + Vector3(0, 0, chainLine + 0.003f), 0.0078f, M.chrome, 10));
                }
            }
            // the right crank and the left
            for (int s : {1, -1}) {
                const float sg = static_cast<float>(s);
                const float zc = sg * 0.0735f + (s > 0 ? 0.0f : 0.0f);
                const float a = s > 0 ? 0.f : PI;
                const Vector3 u(std::cos(a), std::sin(a), 0.f);
                const Vector3 from = Vector3(0, 0, zc), to = u * crankLen + Vector3(0, 0, zc);
                g.add(rod(from, to, 0.0135f, 0.0105f, M.alloy, 20));
                g.add(rod(from - Vector3(0, 0, 0.0035f), from + Vector3(0, 0, 0.0035f), 0.019f, 0.019f, M.alloy, 24));
                // the spindle and its bolt
                g.add(rod(Vector3(0, 0, -0.095f), Vector3(0, 0, 0.095f), 0.0085f, 0.0085f, M.steel, 14));
                g.add(rod(from + Vector3(0, 0, sg * 0.003f), from + Vector3(0, 0, sg * 0.009f), 0.0075f, 0.0075f, M.chrome, 8));
                // the pedal, which stays level: its group counter-rotates
                auto& ped = (s > 0) ? pedalR : pedalL;
                ped->position.copy(to + Vector3(0, 0, sg * 0.0));
                g.add(ped);
                // pedal axle
                ped->add(rod(Vector3(0, 0, sg * -0.0), Vector3(0, 0, sg * 0.092f), 0.0052f, 0.0052f, M.steel, 10));
                // platform: rubber with cage edges
                ped->add(box(Vector3(0, 0, sg * 0.056f), 0.102f, 0.020f, 0.088f, M.rubber));
                ped->add(box(Vector3(0, 0.011f, sg * 0.056f), 0.108f, 0.003f, 0.092f, M.alloy));
                ped->add(box(Vector3(0, -0.011f, sg * 0.056f), 0.108f, 0.003f, 0.092f, M.alloy));
                for (int e : {1, -1}) ped->add(box(Vector3(0.0525f * static_cast<float>(e), 0.0f, sg * 0.056f), 0.004f, 0.02f, 0.092f, M.alloy));
                for (int e : {1, -1}) ped->add(box(Vector3(0.0f, 0.0f, sg * (0.056f + 0.0445f * static_cast<float>(e))), 0.108f, 0.02f, 0.004f, M.alloy));
            }
            // the chain: inner links hold rollers between narrow plates, outer links wear pin heads on wide ones
            {
                MeshBuild outer, inner;
                const float L = pitch;
                // a plate is a bar with a round end at each pin; the link is two of them
                const auto plate = [&](MeshBuild& mb, float z, float thick) {
                    for (int endI = -1; endI <= 1; endI += 2) appendCylinderZ(mb, 0.5f * L * static_cast<float>(endI), 0.f, 0.0054f, z - 0.5f * thick, z + 0.5f * thick, 14);
                    appendTileBar(mb, L, 0.0098f, thick, z);
                };
                for (float z : {-0.0044f, 0.0044f}) plate(outer, z, 0.0015f);
                for (float z : {-0.0027f, 0.0027f}) plate(inner, z, 0.0015f);
                // pin heads on the outer link's outside faces
                for (int e : {-1, 1})
                    for (float z : {-1.f, 1.f}) appendCylinderZ(outer, 0.5f * L * static_cast<float>(e), 0.f, 0.0027f, z * 0.0051f, z * 0.0062f, 10);
                // rollers between the inner plates
                for (int e : {-1, 1}) appendCylinderZ(inner, 0.5f * L * static_cast<float>(e), 0.f, 0.0042f, -0.0027f, 0.0027f, 14);
                chainOuter = InstancedMesh::create(outer.mesh(M.chain, false)->geometry(), M.chain, 50);
                chainInner = InstancedMesh::create(inner.mesh(M.chain, false)->geometry(), M.chain, 50);
                chainOuter->castShadow = chainInner->castShadow = true;
                chainOuter->frustumCulled = chainInner->frustumCulled = false;
                root->add(chainOuter);
                root->add(chainInner);
            }
        }

        static void appendTileBar(MeshBuild& mb, float length, float width, float thick, float z) {
            // a box between the two pins, narrower than the ends
            const float hx = length * 0.5f, hy = width * 0.5f * 0.82f, hz = thick * 0.5f;
            const float xs[2] = {-hx, hx}, ys[2] = {-hy, hy}, zs[2] = {z - hz, z + hz};
            const Vector3 nrm[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            for (int f = 0; f < 6; ++f) {
                const unsigned base = mb.vertexCount();
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) {
                        Vector3 p;
                        if (f < 2) p = Vector3(xs[f], ys[a], zs[b]);
                        else if (f < 4) p = Vector3(xs[a], ys[f - 2], zs[b]);
                        else p = Vector3(xs[a], ys[b], zs[f - 4]);
                        mb.vertex(p, nrm[f], 0, 0, V3(1, 1, 1));
                    }
                // orient so (b-a)x(c-a) matches the normal
                const Vector3 p0(mb.pos[3 * base], mb.pos[3 * base + 1], mb.pos[3 * base + 2]);
                const Vector3 p1(mb.pos[3 * (base + 1)], mb.pos[3 * (base + 1) + 1], mb.pos[3 * (base + 1) + 2]);
                const Vector3 p2(mb.pos[3 * (base + 2)], mb.pos[3 * (base + 2) + 1], mb.pos[3 * (base + 2) + 2]);
                if ((p1 - p0).cross(p2 - p0).dot(nrm[f]) > 0) mb.index.insert(mb.index.end(), {base, base + 1, base + 2, base + 1, base + 3, base + 2});
                else mb.index.insert(mb.index.end(), {base, base + 2, base + 1, base + 1, base + 2, base + 3});
            }
        }

        // ---- mud guards ----------------------------------------------------------------------------------------------------------
        // A guard is a shallow arch round the tyre from angle a0 to a1 (measured from +x, counter-clockwise, about the
        // axle), curled in at its edges, with a rolled bead along each edge.
        static constexpr float fenderClear = 0.0175f, fenderWidth = 0.068f, fenderCurl = 0.011f;

        static Vector3 fenderPoint(const Vector3& axle, float angle, float t, float radius) {
            const float r = radius - fenderCurl * std::pow(std::abs(t), 3.2f);
            return Vector3(axle.x + r * std::cos(angle), axle.y + r * std::sin(angle), 0.5f * fenderWidth * t);
        }

        std::shared_ptr<Group> buildFender(const Vector3& axle, float a0, float a1) {
            auto g = Group::create();
            const float rr = R + fenderClear;
            MeshBuild mb;
            const int na = 110, nw = 16;
            for (int i = 0; i <= na; ++i) {
                const float a = a0 + (a1 - a0) * static_cast<float>(i) / static_cast<float>(na);
                for (int k = 0; k <= nw; ++k) {
                    const float t = static_cast<float>(k) / static_cast<float>(nw) * 2.f - 1.f;
                    mb.vertex(fenderPoint(axle, a, t, rr), Vector3(std::cos(a), std::sin(a), 0.f), static_cast<float>(i) / static_cast<float>(na), t * 0.5f + 0.5f, V3(1, 1, 1));
                }
            }
            const unsigned row = static_cast<unsigned>(nw + 1);
            for (int i = 0; i < na; ++i)
                for (int k = 0; k < nw; ++k) {
                    const unsigned a = i * row + k, b = a + 1, c = a + row, d = c + 1;
                    mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
                }
            mb.smoothNormals();
            M.paint->side = Side::Double;
            g->add(mb.mesh(M.paint, false));
            // the rolled edges
            for (float t : {-1.f, 1.f}) {
                std::vector<Vector3> pts;
                for (int i = 0; i <= na; ++i) pts.push_back(fenderPoint(axle, a0 + (a1 - a0) * static_cast<float>(i) / static_cast<float>(na), t, rr) - Vector3(0, 0, 0));
                MeshBuild bead;
                SweepOptions o;
                o.seg = 8;
                appendSweep(bead, pts, std::vector<float>(pts.size(), 0.0021f), o);
                g->add(bead.mesh(M.paint, false));
            }
            return g;
        }

        // a stay from the dropout's eyelet out to the guard's edge at `angle`
        std::shared_ptr<Mesh> fenderStay(const Vector3& axle, float zDropout, float angle, float side) {
            const Vector3 eyelet(axle.x, axle.y, zDropout * side);
            const Vector3 edge = fenderPoint(axle, angle, side, R + fenderClear);
            return rod(eyelet, edge, 0.0022f, 0.0022f, M.chrome, 8);
        }

        void buildFenders() {
            const float rr = R + fenderClear;
            // ---- the rear guard: from behind the seat tube over the top and down behind the wheel
            {
                const float a0 = 0.72f, a1 = 3.45f;
                rearFender = buildFender(rearAxle, a0, a1);
                root->add(rearFender);
                // bolted to the seat stay bridge at its front end
                const Vector3 front = fenderPoint(rearAxle, a0 + 0.03f, 0.f, rr);
                root->add(rod(seatBridge, front + Vector3(0, 0.0035f, 0), 0.0032f, 0.0032f, M.chrome, 10));
                // and held at the chain stay bridge by a strap under the front end of the lower run
                // stays from the dropouts, both sides, two each
                for (int s : {1, -1})
                    for (float ang : {1.85f, 2.7f}) root->add(fenderStay(rearAxle, dropoutZ, ang, static_cast<float>(s)));
                // the mud flap: black rubber hung from the end of the guard, with a reflective strip
                {
                    const Vector3 end = Vector3(rearAxle.x + rr * std::cos(a1), rearAxle.y + rr * std::sin(a1), 0.f);
                    const Vector3 tangent(-std::sin(a1), std::cos(a1), 0.f);
                    auto flap = Mesh::create(BoxGeometry::create(0.003f, 0.085f, 0.064f), M.rubber);
                    flap->position.copy(end + tangent * 0.0405f + Vector3(0.0f, 0.0f, 0.0f));
                    alignY(*flap, tangent);
                    root->add(finish(flap));
                    auto strip = Mesh::create(BoxGeometry::create(0.0035f, 0.012f, 0.05f), M.reflectorRed);
                    strip->position.copy(end + tangent * 0.06f + Vector3(-0.0012f, 0.f, 0.f));
                    alignY(*strip, tangent);
                    root->add(finish(strip, false, false));
                }
                // the rear lamp, a bracketed housing on the guard's tail, lens facing back
                {
                    const float al = a1 - 0.2f;
                    const Vector3 base = Vector3(rearAxle.x + rr * std::cos(al), rearAxle.y + rr * std::sin(al), 0.f);
                    const Vector3 radial(std::cos(al), std::sin(al), 0.f);
                    const Vector3 tangent(-std::sin(al), std::cos(al), 0.f);
                    // a bracket plate screwed to the guard
                    auto plate = Mesh::create(BoxGeometry::create(0.004f, 0.05f, 0.03f), M.chrome);
                    plate->position.copy(base + radial * 0.002f);
                    alignY(*plate, tangent);
                    root->add(finish(plate));
                    // the housing, a rounded box standing out from it
                    auto housing = Mesh::create(BoxGeometry::create(0.022f, 0.056f, 0.046f), M.black);
                    housing->position.copy(base + radial * 0.014f);
                    alignY(*housing, tangent);
                    root->add(finish(housing));
                    // the red lens on the back of it
                    auto lens = Mesh::create(BoxGeometry::create(0.004f, 0.046f, 0.04f), M.reflectorRed);
                    lens->position.copy(base + radial * 0.014f - Vector3(0.0125f, 0.f, 0.f));
                    alignY(*lens, tangent);
                    root->add(finish(lens, false, false));
                }
            }
            // ---- the front guard, turning with the fork: from ahead of the wheel over the top to just past the crown
            {
                const Vector3 crown = headBottom + down() * 0.006f;
                const float aCrown = std::atan2(crown.y - frontAxle.y, crown.x - frontAxle.x);
                const float a0 = -0.5f, a1 = aCrown + 0.10f;
                auto fg = buildFender(frontAxle, a0, a1);
                steering->add(fg);
                // bolted up to the crown
                const Vector3 under = fenderPoint(frontAxle, aCrown, 0.f, rr);
                steering->add(rod(crown - Vector3(0, 0.006f, 0), under + Vector3(0, 0.0035f, 0), 0.0035f, 0.0035f, M.chrome, 10));
                for (int s : {1, -1})
                    for (float ang : {0.35f, 1.0f}) steering->add(fenderStay(frontAxle, forkZ, ang, static_cast<float>(s)));
                // a small white reflector on the front edge
                const Vector3 at = Vector3(frontAxle.x + (rr + 0.003f) * std::cos(a0 + 0.07f), frontAxle.y + (rr + 0.003f) * std::sin(a0 + 0.07f), 0.f);
                auto refl = Mesh::create(BoxGeometry::create(0.004f, 0.03f, 0.036f), standard(lin(230, 230, 232), 0.2f));
                refl->position.copy(at);
                alignY(*refl, Vector3(-std::sin(a0 + 0.07f), std::cos(a0 + 0.07f), 0.f));
                steering->add(finish(refl, false, false));
            }
            // the front wheel and fork turn together
            steering->add(frontWheel);
            frontWheel->position.copy(frontAxle);
        }
        std::shared_ptr<Group> rearFender;

        // ---- the kickstand -------------------------------------------------------------------------------------------------------------
        void buildFittings() {
            // a plate under the two chain stays, a hinge, and the stand folded back beside them, outside the chain
            const float x = 0.31f;
            const float f = (bb.x - x) / bb.x;
            const float yStay = bb.y + f * (rearAxle.y - bb.y);
            const float zStay = 0.028f + f * (dropoutZ - 0.028f);
            const float yPlate = yStay - 0.0125f;
            root->add(box(Vector3(x, yPlate, 0.f), 0.07f, 0.005f, 2.f * zStay + 0.03f, M.black));
            // hinge pins and the legs, which fold back along the guard's side
            for (int s : {1, -1}) {
                const float z = static_cast<float>(s) * (zStay + 0.022f);
                root->add(rod(Vector3(x, yPlate - 0.003f, 0.f), Vector3(x, yPlate - 0.003f, z + static_cast<float>(s) * 0.014f), 0.0035f, 0.0035f, M.steel, 8));
                std::vector<Vector3> leg = {Vector3(x, yPlate - 0.003f, z + static_cast<float>(s) * 0.014f), Vector3(x - 0.05f, yPlate - 0.01f, z + static_cast<float>(s) * 0.014f), Vector3(x - 0.15f, yPlate - 0.026f, z + static_cast<float>(s) * 0.022f)};
                const auto pts = splinePoints(leg, 6);
                MeshBuild mb;
                SweepOptions o;
                o.seg = 10;
                appendSweep(mb, pts, std::vector<float>(pts.size(), 0.0048f), o);
                root->add(mb.mesh(M.black, false));
            }
            // the foot bar across the legs' ends, in rubber
            {
                const float z = zStay + 0.044f;
                root->add(rod(Vector3(x - 0.15f, yPlate - 0.026f, z), Vector3(x - 0.15f, yPlate - 0.026f, -z), 0.0075f, 0.0075f, M.rubber, 12));
            }
        }
    };

}// namespace pb

#endif
