// Feathers: textures painted barb by barb (a shaft, a vane of barbs slanting to the tip, splits between barb
// groups, a downy base, a ragged or rounded end) and a card of a feather to lay them out with, thousands at a time,
// as instances of one shared mesh.
//
// A feather card is 1 long (+Y, the shaft, root at the origin), 1 wide (X, centred on the shaft), and lies in the
// XY plane with a curve along its length and across it toward -Z. Instances scale it to the feather's size: the
// matrix's columns are the feather's side * width, its direction * length and its outward normal.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_FEATHERS_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_FEATHERS_HPP

#include "Parts.hpp"

namespace pb {

    using mill::fbm2;
    using mill::hash2f;
    using mill::mix;

    enum class FeatherKind { Contour, Covert, Primary, Secondary, Crest, Tail, Down };

    struct FeatherStyle {
        float length = 0.085f, width = 0.05f;        // metres: the texture's physical size
        float tipRound = 0.8f;                       // 1 broad and round .. small pointed
        float baseDown = 0.22f;                      // the downy share of the length
        float barbAngle = 0.95f;                     // radians from the shaft
        float barbSpacing = 0.00045f;                // metres
        float innerWide = 1.f;                       // the inner vane (u > 0.5) times the outer's width
        V3 colour = srgb(250, 248, 244);
        V3 shadow = srgb(214, 222, 234);
        V3 edge = srgb(250, 248, 244);
        V3 shaft = srgb(246, 238, 218);
        float rough = 0.62f;
        float gap = 0.35f;                           // how often barb groups split
        float fringe = 0.f;                          // soft frayed end
    };

    inline FeatherStyle styleOf(FeatherKind k) {
        FeatherStyle s;
        switch (k) {
            case FeatherKind::Contour:
                break;
            case FeatherKind::Down:
                s.length = 0.05f;
                s.width = 0.035f;
                s.baseDown = 0.6f;
                s.gap = 0.6f;
                s.tipRound = 0.6f;
                break;
            case FeatherKind::Covert:
                s.length = 0.14f;
                s.width = 0.058f;
                s.baseDown = 0.12f;
                s.tipRound = 0.9f;
                s.edge = srgb(232, 236, 240);
                break;
            case FeatherKind::Primary:
                s.length = 0.40f;
                s.width = 0.050f;
                s.baseDown = 0.04f;
                s.tipRound = 0.55f;
                s.barbAngle = 0.62f;
                s.innerWide = 1.5f;
                s.colour = srgb(44, 40, 40);
                s.shadow = srgb(22, 20, 22);
                s.edge = srgb(70, 62, 58);
                s.shaft = srgb(118, 108, 94);
                s.rough = 0.5f;
                s.gap = 0.10f;
                break;
            case FeatherKind::Secondary:
                s.length = 0.34f;
                s.width = 0.064f;
                s.baseDown = 0.05f;
                s.tipRound = 0.8f;
                s.barbAngle = 0.7f;
                s.innerWide = 1.35f;
                s.colour = srgb(52, 48, 48);
                s.shadow = srgb(26, 24, 26);
                s.edge = srgb(120, 114, 108);   // the greyish tips of secondaries
                s.shaft = srgb(124, 114, 100);
                s.rough = 0.52f;
                s.gap = 0.12f;
                break;
            case FeatherKind::Tail:
                s.length = 0.22f;
                s.width = 0.052f;
                s.baseDown = 0.05f;
                s.tipRound = 0.85f;
                s.colour = srgb(244, 242, 236);
                s.shadow = srgb(198, 204, 214);
                s.edge = srgb(236, 234, 226);
                s.gap = 0.25f;
                break;
            case FeatherKind::Crest:
                s.length = 0.10f;
                s.width = 0.011f;
                s.baseDown = 0.1f;
                s.tipRound = 0.2f;
                s.colour = srgb(247, 226, 150);
                s.shadow = srgb(226, 196, 120);
                s.edge = srgb(250, 232, 160);
                s.barbAngle = 0.5f;
                s.gap = 0.1f;
                break;
        }
        return s;
    }

    // paint one feather into a texture set (u across, v from the root to the tip)
    inline TexSet featherTexture(FeatherKind kind, int W, int H, uint32_t seed) {
        const FeatherStyle st = styleOf(kind);
        Maps m(W, H, st.width, st.length);
        const float tanA = std::tan(st.barbAngle);
        mill::forRows(H, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            const float ym = v * st.length;
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const float xm = (u - 0.5f) * st.width;
                const bool inner = xm > 0.f;
                const float ax = std::abs(xm);
                // the vane's half width along the feather
                const float vv = std::pow(v, st.tipRound > 0.5f ? 0.78f : 0.9f);
                float hw = 0.5f * st.width * std::pow(std::max(std::sin(PI * std::clamp(vv, 0.f, 1.f)), 0.f), st.tipRound > 0.5f ? 0.55f : 0.9f);
                // the calamus: bare at the root
                hw *= smooth(0.0f, 0.06f, v);
                // inner and outer vanes differ
                const float sideScale = inner ? std::sqrt(st.innerWide) : 1.f / std::sqrt(st.innerWide);
                hw *= sideScale;
                if (kind == FeatherKind::Primary || kind == FeatherKind::Secondary) {
                    // a parallel-sided vane, narrowing to a rounded tip
                    const float body = smooth(0.04f, 0.2f, v) * (1.f - smooth(0.82f, 1.0f, v) * 0.8f);
                    hw = 0.5f * st.width * sideScale * body * std::sqrt(std::max(1.f - std::pow(smooth(0.9f, 1.f, v), 2.f), 0.f));
                }
                const size_t i = m.at(x, y);
                // the shaft
                const float shaftW = 0.00085f * (1.f - 0.7f * v) + 0.00018f;
                const float onShaft = smooth(shaftW, shaftW * 0.45f, ax);
                // barbs: they leave the shaft slanting toward the tip; their index is where they leave it
                const float barbIdx = (ym - ax / tanA) / st.barbSpacing;
                const float bi = std::floor(barbIdx), bf = barbIdx - bi;
                const float h = hash2f(static_cast<int>(bi) + (inner ? 7000 : 0), static_cast<int>(ax * 1000.f) / 40, seed);
                const float bh = hash2f(static_cast<int>(bi), inner ? 1 : 2, seed + 11);
                const float ridge = 0.5f + 0.5f * std::cos(bf * 6.2831853f);
                // splits: groups of barbs part, more toward the edge and the tip
                const float groupIdx = std::floor(barbIdx / 14.f);
                const float groupHash = hash2f(static_cast<int>(groupIdx) + (inner ? 900 : 0), 3, seed + 5);
                const float edgeNear = smooth(0.55f, 0.95f, ax / std::max(hw, 1e-5f));
                const float split = (groupHash < st.gap ? 1.f : 0.f) * edgeNear * smooth(0.25f, 0.6f, v);
                const float splitCut = split * smooth(0.5f, 0.9f, std::abs(bf - 0.5f) * 2.f + 0.5f * bh);
                float alpha = smooth(hw, hw - 0.00022f, ax);
                alpha *= 1.f - 0.85f * splitCut * 0.0f;
                // downy base: wispy, see-through
                const float down = smooth(st.baseDown, 0.f, v);
                const float wisp = smooth(0.35f, 0.75f, 0.5f + 0.5f * fbm2(xm * 700.f, ym * 160.f, 3, seed + 21));
                alpha *= 1.f - down * (1.f - wisp) * 0.9f;
                // ragged end: barb tips fray
                if (st.fringe > 0.f) alpha *= 1.f - st.fringe * smooth(0.9f, 1.f, v) * bh;
                // the split gaps: a notch in the vane's edge where a group stands apart
                alpha *= 1.f - smooth(0.3f, 0.9f, splitCut) * 0.9f;
                alpha = std::max(alpha, onShaft * (v > 0.001f ? 1.f : 0.f) * (ym < st.length * 0.985f ? 1.f : 0.f));
                // colour
                const float shade = 0.84f + 0.16f * ridge;
                V3 c = mix(st.shadow, st.colour, clamp01(shade * (0.9f + 0.1f * h)));
                // greying toward the edge of the vane (and the tip for secondaries)
                c = mix(c, st.edge, smooth(0.6f, 1.f, ax / std::max(hw, 1e-5f)) * 0.5f + smooth(0.86f, 1.f, v) * (kind == FeatherKind::Secondary ? 0.9f : 0.f));
                c = mix(c, st.shaft, onShaft);
                m.albedo[i] = c;
                m.alpha[i] = clamp01(alpha);
                m.height[i] = 0.000055f * ridge * (1.f - onShaft) + 0.00028f * onShaft;
                m.rough[i] = st.rough + 0.1f * (1.f - ridge) - 0.14f * onShaft;
                m.cavity[i] = 0.55f + 0.45f * (0.5f + 0.5f * ridge) - 0.2f * split;
            }
        });
        const bool flight = kind == FeatherKind::Primary || kind == FeatherKind::Secondary;
        return m.finish(flight ? 0.35f : 0.9f, false, false);
    }

    inline std::shared_ptr<MeshStandardMaterial> featherMaterial(const TexSet& t) {
        auto m = t.material(0.8f);
        m->side = Side::Double;
        m->alphaTest = 0.42f;
        m->alphaToCoverage = true;
        m->metalness = 0.f;
        m->metalnessMap = nullptr;
        m->roughness = 1.f;
        m->vertexColors = false;
        return m;
    }

    // the card: 1 long, 1 wide, curved along and across toward -Z
    inline std::shared_ptr<BufferGeometry> featherCard(float curlAlong, float curlAcross, int nx = 4, int ny = 10) {
        MeshBuild mb;
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(nx), v = static_cast<float>(j) / static_cast<float>(ny);
                const float x = u - 0.5f;
                const float z = -curlAlong * v * v - curlAcross * x * x * (0.3f + v);
                mb.vertex(Vector3(x, v, z), Vector3(0, 0, 1), u, v, V3(1, 1, 1));
            }
        const unsigned row = static_cast<unsigned>(nx + 1);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const unsigned a = j * row + i, b = a + 1, c = a + row, d = c + 1;
                mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
            }
        mb.smoothNormals();
        auto mesh = mb.mesh(standard(Color(1, 1, 1), 0.5f), false);
        return mesh->geometry();
    }

    // ---- a set of feathers of one kind, laid out by the caller and drawn as one instanced mesh ---------------------------
    struct FeatherSet {
        std::vector<Matrix4> m;
        std::vector<float> rgb;

        // place a feather: its root at `root`, running along `dir`, its side across `side`, `normal` outward;
        // `lift` raises the tip off the surface (radians, about the side axis), `roll` turns it about its own axis
        void add(const Vector3& root, const Vector3& dir, const Vector3& normal, float length, float width, float lift, float roll, const V3& tint = V3(1, 1, 1)) {
            Vector3 d = dir;
            d.normalize();
            Vector3 n = normal - d * normal.dot(d);
            if (n.lengthSq() < 1e-10f) n = Vector3(0, 1, 0);
            n.normalize();
            // lift: tilt the direction toward the normal
            const Vector3 dl = (d * std::cos(lift) + n * std::sin(lift)).normalize();
            const Vector3 nl = (n * std::cos(lift) - d * std::sin(lift)).normalize();
            Vector3 side = Vector3(dl).cross(nl);
            // roll about the feather's own axis
            const float cr = std::cos(roll), sr = std::sin(roll);
            const Vector3 s2 = side * cr + nl * sr;
            const Vector3 n2 = nl * cr - side * sr;
            Matrix4 mm;
            mm.makeBasis(s2 * width, dl * length, n2 * length);   // the card's curl is in units of the feather's length
            mm.setPosition(root);
            m.push_back(mm);
            rgb.insert(rgb.end(), {tint.r, tint.g, tint.b});
        }

        [[nodiscard]] std::shared_ptr<InstancedMesh> build(const std::shared_ptr<BufferGeometry>& geo, const std::shared_ptr<Material>& mat) const {
            auto inst = InstancedMesh::create(geo, mat, std::max<size_t>(m.size(), 1));
            for (size_t i = 0; i < m.size(); ++i) inst->setMatrixAt(i, m[i]);
            if (!m.empty()) inst->setColors(0, m.size(), rgb.data());
            inst->castShadow = true;
            inst->receiveShadow = true;
            inst->frustumCulled = false;
            return inst;
        }
    };

}// namespace pb

#endif
