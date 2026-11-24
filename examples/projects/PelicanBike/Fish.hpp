// A mackerel, caught: a lofted body painted silver below and dark and barred above, a wet clear-coat, an eye, fins and a
// forked tail. Built with its head at the origin, the tail along +x, in metres.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_FISH_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_FISH_HPP

#include "Parts.hpp"

namespace pb {

    inline std::shared_ptr<Group> makeFish(float length = 0.30f) {
        auto g = Group::create();
        const float k = length / 0.30f;
        Loft L;
        L.spine.build({{0.f, 0.f, 0.f}, {0.08f * k, 0.003f * k, 0.f}, {0.18f * k, 0.f, 0.f}, {0.26f * k, -0.003f * k, 0.f}, {0.31f * k, 0.f, 0.f}}, 200);
        L.rings = 90;
        L.around = 32;
        L.stations = {
                {0.00f, 0.0015f * k, 0.002f * k, 0.002f * k, 2.2f, 0.f},
                {0.04f, 0.015f * k, 0.019f * k, 0.017f * k, 2.2f, 0.f},
                {0.14f, 0.022f * k, 0.030f * k, 0.027f * k, 2.2f, 0.f},
                {0.34f, 0.027f * k, 0.037f * k, 0.033f * k, 2.2f, 0.f},
                {0.62f, 0.019f * k, 0.027f * k, 0.024f * k, 2.2f, 0.f},
                {0.86f, 0.007f * k, 0.010f * k, 0.009f * k, 2.2f, 0.f},
                {1.00f, 0.004f * k, 0.006f * k, 0.006f * k, 2.2f, 0.f}};
        MeshBuild mb;
        L.build(mb, [](float t, float a) {
            // the angle runs from the right flank (0) through the back (pi/2) to the belly (3pi/2)
            const float back = smooth(-0.1f, 0.5f, std::sin(a));
            const float belly = smooth(0.1f, -0.6f, std::sin(a));
            const V3 silver = srgb(214, 222, 228), blue = srgb(28, 78, 92), black = srgb(14, 26, 30);
            V3 c = mill::mix(silver, blue, back);
            // the dark wavy bars on the back, slanting
            const float bars = smooth(0.35f, 0.65f, 0.5f + 0.5f * std::sin(t * 64.f + 3.f * std::cos(a) + 2.f * std::sin(t * 17.f)));
            c = mill::mix(c, black, bars * smooth(0.05f, 0.6f, std::sin(a)) * 0.85f * smooth(0.12f, 0.2f, t));
            // the gill, the dark snout and a golden flush along the side
            c = mill::mix(c, srgb(236, 214, 150), 0.25f * smooth(0.1f, 0.4f, t) * (1.f - back) * (1.f - belly) * 0.6f);
            c = mill::mix(c, black, smooth(0.04f, 0.0f, t) * 0.7f);
            return c;
        }, 3.f);
        auto mat = physical(Color(1, 1, 1), 0.22f);
        mat->vertexColors = true;
        mat->clearcoat = 0.6f;
        mat->clearcoatRoughness = 0.12f;
        mat->metalness = 0.15f;
        g->add(mb.mesh(mat));
        const auto dark = standard(lin(20, 40, 46), 0.5f);
        dark->side = Side::Double;
        // fins: thin triangles standing out of the back and the belly, the forked tail
        const auto fin = [&](const Vector3& at, float len, float height, float lean) {
            MeshBuild fb;
            fb.vertex(Vector3(0, 0, 0), Vector3(0, 0, 1), 0, 0, V3(1, 1, 1));
            fb.vertex(Vector3(len, 0, 0), Vector3(0, 0, 1), 1, 0, V3(1, 1, 1));
            fb.vertex(Vector3(len * 0.35f + lean, height, 0), Vector3(0, 0, 1), 0.4f, 1, V3(1, 1, 1));
            fb.index = {0, 1, 2};
            auto mesh = fb.mesh(dark, false);
            mesh->position.copy(at);
            g->add(finish(mesh, true, true));
        };
        fin(Vector3(0.10f * k, 0.030f * k, 0.f), 0.05f * k, 0.017f * k, 0.01f * k);
        fin(Vector3(0.17f * k, 0.031f * k, 0.f), 0.04f * k, 0.014f * k, 0.012f * k);
        fin(Vector3(0.20f * k, -0.028f * k, 0.f), 0.04f * k, -0.013f * k, 0.01f * k);
        for (int s : {1, -1}) {
            // pectoral
            auto pf = Mesh::create(BoxGeometry::create(0.032f * k, 0.002f, 0.012f * k), dark);
            pf->position.set(0.06f * k, -0.012f * k, static_cast<float>(s) * 0.023f * k);
            pf->rotation.y = static_cast<float>(s) * 0.5f;
            g->add(finish(pf));
        }
        // the tail: two lobes in the plane of the body's height
        for (int s : {1, -1}) {
            MeshBuild tb;
            const float fs = static_cast<float>(s);
            tb.vertex(Vector3(0, 0, 0), Vector3(0, 0, 1), 0, 0, V3(1, 1, 1));
            tb.vertex(Vector3(0.045f * k, fs * 0.040f * k, 0), Vector3(0, 0, 1), 1, 1, V3(1, 1, 1));
            tb.vertex(Vector3(0.038f * k, fs * 0.005f * k, 0), Vector3(0, 0, 1), 1, 0, V3(1, 1, 1));
            tb.index = {0, 1, 2};
            auto tail = tb.mesh(dark, false);
            tail->position.set(0.30f * k, 0.f, 0.f);
            g->add(finish(tail, true, true));
        }
        // the eye
        for (int s : {1, -1}) {
            const float fs = static_cast<float>(s);
            auto ring = ball(Vector3(0.022f * k, 0.007f * k, fs * 0.0145f * k), 0.0068f * k, standard(lin(220, 220, 214), 0.2f, 0.6f), 14);
            g->add(ring);
            g->add(ball(Vector3(0.0225f * k, 0.007f * k, fs * 0.0185f * k), 0.0042f * k, standard(lin(6, 6, 8), 0.1f), 12));
        }
        return g;
    }

}// namespace pb

#endif
