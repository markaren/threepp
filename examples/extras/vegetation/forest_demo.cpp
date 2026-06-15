// Procedural forest demo.
//
// Combines the two procedural generators:
//   - terrain::TerrainGenerator  → a rolling heightfield with a slope/altitude
//                                   splat albedo map.
//   - vegetation::TreeGenerator  → a handful of prototype trees (trunk tubes +
//                                   alpha-cutout leaf cards) that are scattered
//                                   across the terrain with InstancedMesh, so a
//                                   few hundred trees cost only a few draw calls.
//
// Trees are placed on a jittered grid, dropped onto the terrain surface, and
// rejected on steep slopes.  Lit by an outdoor-sky HDR (IBL) + a shadow-casting
// sun, with distance fog for depth.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace threepp;

namespace {

    // One prototype tree: shared geometry + materials, instanced many times.
    struct TreeVariant {
        std::shared_ptr<BufferGeometry> trunkGeo;
        std::shared_ptr<BufferGeometry> leafGeo;
        std::shared_ptr<MeshStandardMaterial> barkMat;
        std::shared_ptr<MeshStandardMaterial> leafMat;
    };

    // ── Grass ────────────────────────────────────────────────────────────
    // A single tapered blade, instanced thousands of times. The sway is a
    // whole-blade tilt applied by rewriting each instance matrix on the CPU
    // every frame — this works on BOTH the GL and Vulkan backends (Vulkan has
    // no ShaderMaterial path, but it consumes InstancedMesh matrices directly).
    // A baked vertex-colour gradient + a standard lit material give the blades
    // shading and scene fog for free.
    std::shared_ptr<BufferGeometry> makeGrassBlade() {
        constexpr int seg = 4;
        constexpr float wBase = 0.05f;// half-width at the base
        // Kept deliberately dark: strong sun + sky IBL otherwise pushes the
        // green past 1.0 and ACES desaturates the tips toward white.
        const Vector3 bottom{0.06f, 0.13f, 0.04f};
        const Vector3 top{0.20f, 0.34f, 0.11f};
        std::vector<float> pos, nrm, uv, col;
        std::vector<unsigned int> idx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(seg);
            const float y = t;                 // unit height (scaled per instance)
            const float w = wBase * (1.f - t); // taper to a point at the tip
            const float r = bottom.x + (top.x - bottom.x) * t;
            const float g = bottom.y + (top.y - bottom.y) * t;
            const float b = bottom.z + (top.z - bottom.z) * t;
            for (int s = 0; s < 2; ++s) {
                const float x = (s == 0) ? -w : w;
                pos.push_back(x);
                pos.push_back(y);
                pos.push_back(0.f);
                // Up-biased normal: catches sky/sun light regardless of facing.
                nrm.push_back(0.f);
                nrm.push_back(0.85f);
                nrm.push_back(0.53f);
                uv.push_back(s == 0 ? 0.f : 1.f);
                uv.push_back(t);
                col.push_back(r);
                col.push_back(g);
                col.push_back(b);
            }
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            const unsigned int b = a + 1, c = a + 2, d = a + 3;
            idx.push_back(a); idx.push_back(b); idx.push_back(c);
            idx.push_back(b); idx.push_back(d); idx.push_back(c);
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        return geo;
    }

    // Per-blade static placement data. On WGPU it drives the CPU tilt; on
    // Vulkan it's baked into the merged GrassMesh geometry below.
    struct Blade {
        Vector3 pos;
        Vector3 scale;
        Quaternion yaw;// base orientation about +Y
        float phase;   // wind phase offset
    };

    // Bake every blade into ONE merged geometry for the Vulkan GrassMesh path:
    // each blade's verts are transformed by its placement (pos/yaw/scale) and
    // appended, with a per-vertex `heightFrac` (blade-local y, 0 base→1 tip)
    // for the wind compute shader. One mesh → one BLAS → one TLAS instance.
    std::shared_ptr<BufferGeometry> makeGrassField(const std::vector<Blade>& blades) {
        // Blade template (matches makeGrassBlade): 4 segments, tapered, vertex-
        // colour gradient, up-biased normal.
        constexpr int seg = 4;
        constexpr float wBase = 0.05f;
        const Vector3 bottom{0.06f, 0.13f, 0.04f};
        const Vector3 top{0.20f, 0.34f, 0.11f};
        struct V { Vector3 p; Vector3 n; float u, vy; Vector3 c; };
        std::vector<V> tmpl;
        std::vector<unsigned int> tidx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(seg);
            const float w = wBase * (1.f - t);
            Vector3 c{bottom.x + (top.x - bottom.x) * t,
                      bottom.y + (top.y - bottom.y) * t,
                      bottom.z + (top.z - bottom.z) * t};
            for (int s = 0; s < 2; ++s)
                tmpl.push_back({Vector3{(s == 0 ? -w : w), t, 0.f},
                                Vector3{0.f, 0.85f, 0.53f}, (s == 0 ? 0.f : 1.f), t, c});
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            tidx.insert(tidx.end(), {a, a + 1u, a + 2u, a + 1u, a + 3u, a + 2u});
        }

        std::vector<float> pos, nrm, uv, col, hfrac;
        std::vector<unsigned int> idx;
        const auto vpb = static_cast<unsigned int>(tmpl.size());
        pos.reserve(blades.size() * vpb * 3);
        Matrix4 m;
        for (const auto& bl : blades) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            m.compose(bl.pos, bl.yaw, bl.scale);
            for (const auto& tv : tmpl) {
                Vector3 p = tv.p;
                p.applyMatrix4(m);
                Vector3 n = tv.n;
                n.applyQuaternion(bl.yaw);
                n.normalize();
                pos.push_back(p.x); pos.push_back(p.y); pos.push_back(p.z);
                nrm.push_back(n.x); nrm.push_back(n.y); nrm.push_back(n.z);
                uv.push_back(tv.u); uv.push_back(tv.vy);
                col.push_back(tv.c.x); col.push_back(tv.c.y); col.push_back(tv.c.z);
                hfrac.push_back(tv.vy);// height fraction = blade-local y
            }
            for (unsigned int t : tidx) idx.push_back(base + t);
        }

        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        geo->setAttribute("heightFrac", FloatBufferAttribute::create(hfrac, 1));
        return geo;
    }

    // ── Preferred path: GPU vertex-shader wind (GL + WGPU via GLSL compat) ─
    // Cheaper (no per-frame CPU work) and nicer (per-vertex curve, base
    // planted) than the CPU tilt. ShaderMaterial supplies the #version,
    // modelViewMatrix / projectionMatrix / instanceMatrix / position / uv.
    const char* grassVertexShader() {
        return R"(
            uniform float time;
            uniform float windStrength;
            uniform vec2  windDir;
            varying float vHeight;
            varying float vFog;
            void main() {
                vHeight = uv.y;
                vec3 p = position;
            #ifdef USE_INSTANCING
                vec3 instPos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
            #else
                vec3 instPos = vec3(0.0);
            #endif
                float phase = time * 1.6 + instPos.x * 0.25 + instPos.z * 0.25;
                float gust  = sin(phase) * 0.6 + sin(phase * 2.3 + 1.7) * 0.25;
                float bend  = gust * windStrength * vHeight * vHeight;// base planted, tip sways
                p.x += windDir.x * bend;
                p.z += windDir.y * bend;
            #ifdef USE_INSTANCING
                vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
            #else
                vec4 mv = modelViewMatrix * vec4(p, 1.0);
            #endif
                vFog = -mv.z;
                gl_Position = projectionMatrix * mv;
            }
        )";
    }

    const char* grassFragmentShader() {
        // Self-contained shading (no IBL) keeps the blades from blowing out to
        // white the way a full standard material under the bright sky does.
        return R"(
            uniform vec3  topColor;
            uniform vec3  bottomColor;
            uniform vec3  sunDir;
            uniform vec3  sunColor;
            uniform vec3  ambient;
            uniform vec3  fogColor;
            uniform float fogNear;
            uniform float fogFar;
            varying float vHeight;
            varying float vFog;
            void main() {
                vec3 base = mix(bottomColor, topColor, vHeight);
                // Thin blades: a soft constant wrap + ambient, no harsh N·L term.
                vec3 lit = base * (ambient + sunColor * 0.7);
                float f = clamp((vFog - fogNear) / (fogFar - fogNear), 0.0, 1.0);
                gl_FragColor = vec4(mix(lit, fogColor, f), 1.0);
            }
        )";
    }

    TreeVariant makeVariant(int preset, unsigned int seed) {
        vegetation::TreeParams tp;
        vegetation::applyPreset(preset, tp);
        tp.seed = seed;

        if (preset == 2) {
            // Birch: the pure-white preset bark reads as a bare pole at forest
            // distance — mute it to a soft grey-birch and fill the crown more.
            tp.barkColor = {0.70f, 0.69f, 0.66f};
            tp.leafDensity = 0.97f;
            tp.leafClumping = 0.35f;
        }

        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);

        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);

        auto bark = vegetation::makeBarkTextures(256, seed, tp.barkColor);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;

        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        v.leafMat->map = vegetation::makeLeafClusterTexture(256, seed, tp.leafColor);
        v.leafMat->alphaTest = 0.5f;
        v.leafMat->side = Side::Double;
        v.leafMat->vertexColors = true;
        return v;
    }

    // Bush/shrub: short or no trunk, wide dense low crown.
    TreeVariant makeShrubVariant(unsigned int seed) {
        vegetation::TreeParams tp;
        tp.seed = seed;
        tp.trunkHeight = 0.5f;
        tp.trunkRadius = 0.06f;
        tp.crownShape = vegetation::CrownShape::Hemisphere;
        tp.crownRadiusX = 1.4f;
        tp.crownRadiusZ = 1.4f;
        tp.crownHeight = 1.6f;
        tp.attractorCount = 350;
        tp.influenceDistance = 2.0f;
        tp.killDistance = 0.45f;
        tp.segmentLength = 0.22f;
        tp.maxIterations = 160;
        tp.randomness = 0.12f;
        tp.tropism = -0.02f;
        tp.radiusExponent = 2.0f;
        tp.minBranchRadius = 0.004f;
        tp.radialSegments = 5;
        tp.leafStyle = vegetation::LeafStyle::CrossQuad;
        tp.leafSize = 0.5f;
        tp.leafDensity = 0.95f;
        tp.leavesPerCluster = 5;
        tp.leafSpread = 0.35f;
        tp.leafClumping = 0.3f;
        tp.barkColor = {0.28f, 0.22f, 0.15f};
        tp.leafColor = {0.20f, 0.40f, 0.13f};

        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);
        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);
        auto bark = vegetation::makeBarkTextures(128, seed, tp.barkColor);
        bark.first->repeat.set(2.f, 0.5f);
        bark.second->repeat.set(2.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;
        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        v.leafMat->map = vegetation::makeLeafClusterTexture(256, seed, tp.leafColor);
        v.leafMat->alphaTest = 0.5f;
        v.leafMat->side = Side::Double;
        v.leafMat->vertexColors = true;
        return v;
    }

    // A crossed-quad card (two perpendicular upright quads, uv 0..1, rooted at
    // y=0) for textured billboards like flowers.
    std::shared_ptr<BufferGeometry> makeFlowerCard() {
        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        const float hw = 0.5f;
        const Vector3 up{0.f, 1.f, 0.f};
        auto addQuad = [&](const Vector3& right, const Vector3& face) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            Vector3 n;
            n.copy(face).multiplyScalar(0.4f).add(up).normalize();
            Vector3 c[4];
            c[0].set(0.f, 0.f, 0.f).addScaledVector(right, -hw);
            c[1].set(0.f, 0.f, 0.f).addScaledVector(right, hw);
            c[2].set(0.f, 1.f, 0.f).addScaledVector(right, hw);
            c[3].set(0.f, 1.f, 0.f).addScaledVector(right, -hw);
            const float us[4] = {0.f, 1.f, 1.f, 0.f};
            const float vs[4] = {0.f, 0.f, 1.f, 1.f};
            for (int i = 0; i < 4; ++i) {
                pos.push_back(c[i].x); pos.push_back(c[i].y); pos.push_back(c[i].z);
                nrm.push_back(n.x); nrm.push_back(n.y); nrm.push_back(n.z);
                uv.push_back(us[i]); uv.push_back(vs[i]);
            }
            idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
            idx.push_back(base); idx.push_back(base + 2); idx.push_back(base + 3);
        };
        addQuad(Vector3(1.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f));
        addQuad(Vector3(0.f, 0.f, 1.f), Vector3(1.f, 0.f, 0.f));
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

    // Low-poly faceted boulder: a sphere displaced by a few smooth lumps.
    // Pair with a flat-shaded material for crisp facets.
    std::shared_ptr<BufferGeometry> makeRock(unsigned int seed) {
        constexpr int latSegs = 5, lonSegs = 7;
        constexpr float PI = 3.14159265358979f;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-PI, PI);
        const float p1 = u(rng), p2 = u(rng), p3 = u(rng);

        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        for (int lat = 0; lat <= latSegs; ++lat) {
            const float theta = static_cast<float>(lat) / latSegs * PI;
            const float sinT = std::sin(theta), cosT = std::cos(theta);
            for (int lon = 0; lon <= lonSegs; ++lon) {
                const float phi = static_cast<float>(lon) / lonSegs * 2.f * PI;
                const float nx = sinT * std::cos(phi);
                const float ny = cosT;
                const float nz = sinT * std::sin(phi);
                float disp = 1.f + 0.30f * std::sin(2.f * phi + p1) * sinT +
                             0.24f * std::cos(3.f * phi + p2) +
                             0.22f * std::sin(3.f * theta + p3) +
                             0.14f * std::cos(5.f * phi + 4.f * theta + p1);// finer irregularity
                disp = std::clamp(disp, 0.6f, 1.5f);
                pos.push_back(nx * disp);
                pos.push_back(ny * disp);
                pos.push_back(nz * disp);
                nrm.push_back(nx); nrm.push_back(ny); nrm.push_back(nz);
                uv.push_back(static_cast<float>(lon) / lonSegs);
                uv.push_back(static_cast<float>(lat) / latSegs);
            }
        }
        const int rowVerts = lonSegs + 1;
        for (int lat = 0; lat < latSegs; ++lat) {
            for (int lon = 0; lon < lonSegs; ++lon) {
                const auto a = static_cast<unsigned int>(lat * rowVerts + lon);
                const auto b = static_cast<unsigned int>(a + rowVerts);
                // CCW from outside (outward-facing normals for flat shading).
                idx.push_back(a); idx.push_back(a + 1); idx.push_back(b);
                idx.push_back(a + 1); idx.push_back(b + 1); idx.push_back(b);
            }
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

}// namespace

int main() {

    Canvas canvas("Procedural Forest", {{"vsync", true}, {"aa", 4}});
    auto renderer = createRenderer(canvas);

    // Grass wind path:
    //  - GL only: cheap GPU vertex-shader (ShaderMaterial). The WGPU GLSL→WGSL
    //    path does not render this shader, and Vulkan (a path tracer) has no
    //    ShaderMaterial path at all.
    //  - WGPU + Vulkan: CPU-tilt instance matrices on a standard lit material
    //    (the same material the flowers use — proven to render on both).
    const bool shaderGrass = (dynamic_cast<GLRenderer*>(renderer.get()) != nullptr);
    bool vulkanBackend = false;
#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
    if (vk) {
        vulkanBackend = true;
        // Path tracer: GPU cost (pathTrace + denoise) scales with pixel count,
        // so render below native and TAA-upsample.
        vk->setRenderScale(0.8f);
    }
#endif

    renderer->setClearColor(Color(0.62f, 0.72f, 0.84f));
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    // ── Lighting: sky HDR (IBL) + shadow-casting sun ─────────────────────
    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = hdr;
        scene.environment = hdr;
    }

    auto ambient = AmbientLight::create(Color::white, 0.25f);
    scene.add(ambient);

    auto sun = DirectionalLight::create(Color(1.0f, 0.97f, 0.90f), 2.8f);
    sun->position.set(60.f, 120.f, 80.f);
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -90.f;
        cam->right = cam->top = 90.f;
        cam->nearPlane = 1.f;
        cam->farPlane = 400.f;
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0005f;
    }
    scene.add(sun);

    // ── Terrain ──────────────────────────────────────────────────────────
    terrain::TerrainParams terr;
    terr.seed = 20260615u;
    terr.worldSize = 260.f;
    terr.resolution = 220;
    terr.noiseType = terrain::NoiseType::fBm;// gentle rolling hills (no erosion → heightAt matches mesh)
    terr.featureScale = 110.f;
    terr.octaves = 6;
    terr.amplitude = 16.f;
    terr.warp = 0.3f;
    terr.heightExponent = 1.0f;
    terr.erosion = terrain::ErosionType::None;
    // Grassy meadow palette.
    terr.grassColor = {0.27f, 0.34f, 0.17f};
    terr.screeColor = {0.45f, 0.42f, 0.34f};
    terr.rockColor = {0.40f, 0.37f, 0.33f};
    terr.snowLine = 2.0f;// effectively no snow
    terr.slopeGrassMax = 0.55f;

    terrain::TerrainGenerator terrainGen(terr.seed);
    auto terrainMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
    auto terrainMesh = Mesh::create(terrainGen.createGeometry(terr, false), terrainMat);
    {
        auto tex = DataTexture::create(ImageData{terrainGen.bakeSplatColors(terr)},
                                       static_cast<unsigned int>(terrainGen.dim()),
                                       static_cast<unsigned int>(terrainGen.dim()));
        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        terrainMat->map = tex;
    }
    terrainMesh->receiveShadow = true;
    scene.add(terrainMesh);

    // ── Tree prototypes ──────────────────────────────────────────────────
    std::vector<TreeVariant> variants;
    variants.push_back(makeVariant(0, 101u));// Oak
    variants.push_back(makeVariant(0, 202u));// Oak (alt)
    variants.push_back(makeVariant(1, 303u));// Pine
    variants.push_back(makeVariant(1, 505u));// Pine (alt)
    variants.push_back(makeVariant(2, 404u));// Birch (accent — 1 in 5)

    // ── Scatter trees onto the terrain (jittered grid + slope rejection) ──
    int treeCount = 0;
    std::vector<std::shared_ptr<InstancedMesh>> forest;

    auto buildForest = [&](unsigned int scatterSeed, float spacing, float fillProb) {
        for (auto& m : forest) scene.remove(*m);
        forest.clear();
        treeCount = 0;

        std::mt19937 rng(scatterSeed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        const float half = terr.worldSize * 0.5f * 0.92f;
        const int cells = std::max(1, static_cast<int>((2.f * half) / spacing));

        // Collect per-variant instance transforms.
        std::vector<std::vector<Matrix4>> xforms(variants.size());
        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};

        for (int cz = 0; cz < cells; ++cz) {
            for (int cx = 0; cx < cells; ++cx) {
                if (u01(rng) > fillProb) continue;
                const float jx = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float jz = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float x = -half + (static_cast<float>(cx) + 0.5f) * (2.f * half / static_cast<float>(cells)) + jx;
                const float z = -half + (static_cast<float>(cz) + 0.5f) * (2.f * half / static_cast<float>(cells)) + jz;

                // Surface height + slope (central differences on procedural height).
                const float e = 1.0f;
                const float h = terrainGen.heightAt(x, z, terr);
                const float hx = terrainGen.heightAt(x + e, z, terr) - terrainGen.heightAt(x - e, z, terr);
                const float hz = terrainGen.heightAt(x, z + e, terr) - terrainGen.heightAt(x, z - e, terr);
                const float ny = (2.f * e) / std::sqrt(hx * hx + hz * hz + (2.f * e) * (2.f * e));
                if (ny < 0.86f) continue;// too steep — bare slope

                const size_t vi = static_cast<size_t>(u01(rng) * static_cast<float>(variants.size())) % variants.size();
                const float s = 1.1f + u01(rng) * 0.9f;// scale variety
                q.setFromAxisAngle(up, u01(rng) * 6.28318530718f);
                Matrix4 m;
                m.compose(Vector3(x, h - 0.2f, z), q, Vector3(s, s, s));
                xforms[vi].push_back(m);
            }
        }

        // Build one trunk + one leaf InstancedMesh per variant.
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            const auto& xf = xforms[vi];
            if (xf.empty()) continue;
            treeCount += static_cast<int>(xf.size());

            auto trunks = InstancedMesh::create(variants[vi].trunkGeo, variants[vi].barkMat, xf.size());
            auto leaves = InstancedMesh::create(variants[vi].leafGeo, variants[vi].leafMat, xf.size());
            trunks->castShadow = true;
            trunks->receiveShadow = true;
            leaves->castShadow = true;
            for (size_t i = 0; i < xf.size(); ++i) {
                trunks->setMatrixAt(i, xf[i]);
                leaves->setMatrixAt(i, xf[i]);
            }
            trunks->instanceMatrix()->needsUpdate();
            leaves->instanceMatrix()->needsUpdate();
            scene.add(trunks);
            scene.add(leaves);
            forest.push_back(trunks);
            forest.push_back(leaves);
        }
    };

    float spacing = 11.f;
    float fillProb = 0.8f;
    unsigned int scatterSeed = 1u;
    buildForest(scatterSeed, spacing, fillProb);

    // ── Distance fog (shared by the scene and the grass shader) ──────────
    const Color fogColor(0.66f, 0.75f, 0.86f);
    const float fogNear = terr.worldSize * 0.35f;
    const float fogFar = terr.worldSize * 1.05f;
    scene.fog = Fog(fogColor, fogNear, fogFar);

    // ── Swaying grass (instanced) ────────────────────────────────────────
    // GL's GPU-shader grass is nearly free → dense. WGPU CPU-tilt rasterises
    // cheaply but pays per-instance CPU each frame → medium. Vulkan pays an
    // O(all-instances) TLAS refit per moving frame → sparse.
    const int bladeCount = shaderGrass ? 90000 : (vulkanBackend ? 9000 : 30000);
    const float grassRadius = shaderGrass ? 70.f : (vulkanBackend ? 42.f : 58.f);
    const Vector3 windAxis = Vector3(0.6f, 0.f, -0.8f).normalize();// CPU-path tilt axis ⟂ wind
    const Vector2 windDir2(0.8f, 0.6f);

    // Material: ShaderMaterial (GPU wind) when supported, else a standard lit
    // material driven by the CPU tilt below.
    std::shared_ptr<ShaderMaterial> grassShaderMat;
    std::shared_ptr<Material> grassMat;
    if (shaderGrass) {
        Vector3 sunDirN(60.f, 120.f, 80.f);
        sunDirN.normalize();
        grassShaderMat = ShaderMaterial::create();
        grassShaderMat->vertexShader = grassVertexShader();
        grassShaderMat->fragmentShader = grassFragmentShader();
        grassShaderMat->side = Side::Double;
        grassShaderMat->uniforms["time"].setValue(0.f);
        grassShaderMat->uniforms["windStrength"].setValue(0.18f);
        grassShaderMat->uniforms["windDir"].setValue(windDir2);
        grassShaderMat->uniforms["topColor"].setValue(Vector3(0.30f, 0.42f, 0.14f));
        grassShaderMat->uniforms["bottomColor"].setValue(Vector3(0.08f, 0.16f, 0.05f));
        grassShaderMat->uniforms["sunColor"].setValue(Vector3(0.55f, 0.55f, 0.50f));
        grassShaderMat->uniforms["ambient"].setValue(Vector3(0.30f, 0.34f, 0.30f));
        grassShaderMat->uniforms["fogColor"].setValue(Vector3(fogColor.r, fogColor.g, fogColor.b));
        grassShaderMat->uniforms["fogNear"].setValue(fogNear);
        grassShaderMat->uniforms["fogFar"].setValue(fogFar);
        grassMat = grassShaderMat;
    } else {
        auto std = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
        std->vertexColors = true;
        std->side = Side::Double;
        std->envMapIntensity = 0.45f;
        grassMat = std;
    }

    // Blade placements — filled once, then either instanced (GL/WGPU) or baked
    // into a merged GrassMesh (Vulkan GPU deform).
    std::vector<Blade> blades(static_cast<size_t>(bladeCount));
    {
        std::mt19937 grng(7u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const Vector3 up{0.f, 1.f, 0.f};
        for (int i = 0; i < bladeCount; ++i) {
            const float ang = u01(grng) * 6.28318530718f;
            const float rr = std::sqrt(u01(grng)) * grassRadius;// uniform over disk
            const float x = std::cos(ang) * rr;
            const float z = std::sin(ang) * rr;
            const float h = terrainGen.heightAt(x, z, terr);
            const float s = 0.5f + u01(grng) * 0.5f;  // width scale
            const float hgt = 0.3f + u01(grng) * 0.4f;// height 0.3–0.7 world units

            Blade& bl = blades[static_cast<size_t>(i)];
            bl.pos.set(x, h - 0.05f, z);
            bl.scale.set(s, hgt, s);
            bl.yaw.setFromAxisAngle(up, u01(grng) * 6.28318530718f);
            bl.phase = u01(grng) * 6.28318530718f;
        }
    }

    // Vulkan: one GPU-wind GrassMesh (compute-deform + BLAS refit → one TLAS
    // instance). GL/WGPU: an InstancedMesh (shader wind / CPU tilt below).
    std::shared_ptr<GrassMesh> grassFieldVk;
    std::shared_ptr<InstancedMesh> grass;
    if (vulkanBackend) {
        grassFieldVk = GrassMesh::create(makeGrassField(blades), grassMat);
        grassFieldVk->params.windDir = windDir2;
        grassFieldVk->params.windStrength = 0.18f;
        scene.add(grassFieldVk);
    } else {
        grass = InstancedMesh::create(makeGrassBlade(), grassMat, static_cast<size_t>(bladeCount));
        Matrix4 m;
        for (size_t i = 0; i < blades.size(); ++i) {
            m.compose(blades[i].pos, blades[i].yaw, blades[i].scale);
            grass->setMatrixAt(i, m);
        }
        grass->instanceMatrix()->needsUpdate();// static for the shader path
        scene.add(grass);
    }

    const float groundHalf = terr.worldSize * 0.5f * 0.9f;
    auto slopeOk = [&](float x, float z, float minNy) {
        const float e = 1.0f;
        const float hx = terrainGen.heightAt(x + e, z, terr) - terrainGen.heightAt(x - e, z, terr);
        const float hz = terrainGen.heightAt(x, z + e, terr) - terrainGen.heightAt(x, z - e, terr);
        return (2.f * e) / std::sqrt(hx * hx + hz * hz + (2.f * e) * (2.f * e)) >= minNy;
    };

    // ── Understory bushes (instanced TreeVariants) ───────────────────────
    {
        std::vector<TreeVariant> shrubs{makeShrubVariant(11u), makeShrubVariant(22u)};
        std::vector<std::vector<Matrix4>> xf(shrubs.size());
        std::mt19937 rng(55u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};
        Matrix4 m;
        for (int i = 0; i < 160; ++i) {
            const float x = (u01(rng) - 0.5f) * 2.f * groundHalf;
            const float z = (u01(rng) - 0.5f) * 2.f * groundHalf;
            if (!slopeOk(x, z, 0.85f)) continue;
            const float s = 0.7f + u01(rng) * 0.7f;
            q.setFromAxisAngle(up, u01(rng) * 6.28318530718f);
            m.compose(Vector3(x, terrainGen.heightAt(x, z, terr) - 0.05f, z), q, Vector3(s, s, s));
            xf[static_cast<size_t>(u01(rng) * static_cast<float>(shrubs.size())) % shrubs.size()].push_back(m);
        }
        for (size_t vi = 0; vi < shrubs.size(); ++vi) {
            if (xf[vi].empty()) continue;
            auto trunks = InstancedMesh::create(shrubs[vi].trunkGeo, shrubs[vi].barkMat, xf[vi].size());
            auto leaves = InstancedMesh::create(shrubs[vi].leafGeo, shrubs[vi].leafMat, xf[vi].size());
            trunks->castShadow = leaves->castShadow = true;
            trunks->receiveShadow = true;
            for (size_t i = 0; i < xf[vi].size(); ++i) {
                trunks->setMatrixAt(i, xf[vi][i]);
                leaves->setMatrixAt(i, xf[vi][i]);
            }
            trunks->instanceMatrix()->needsUpdate();
            leaves->instanceMatrix()->needsUpdate();
            scene.add(trunks);
            scene.add(leaves);
        }
    }

    // ── Wildflowers (instanced cards; gentle CPU-tilt sway, all backends) ─
    std::vector<std::shared_ptr<InstancedMesh>> flowerMeshes;
    std::vector<std::vector<Blade>> flowerBlades;
    {
        const int perVariant = shaderGrass ? 1100 : (vulkanBackend ? 400 : 900);
        const Vector3 up{0.f, 1.f, 0.f};
        for (int fv = 0; fv < 3; ++fv) {
            auto card = makeFlowerCard();
            auto mat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color::white).roughness(0.9f).metalness(0.f));
            mat->map = vegetation::makeFlowerTexture(128, static_cast<unsigned int>(1 + fv));
            mat->alphaTest = 0.5f;
            mat->side = Side::Double;
            mat->envMapIntensity = 0.6f;
            auto fm = InstancedMesh::create(card, mat, static_cast<size_t>(perVariant));
            std::vector<Blade> fb(static_cast<size_t>(perVariant));
            std::mt19937 rng(static_cast<unsigned int>(200 + fv));
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            Matrix4 m;
            for (int i = 0; i < perVariant; ++i) {
                const float ang = u01(rng) * 6.28318530718f;
                const float rr = std::sqrt(u01(rng)) * grassRadius;
                const float x = std::cos(ang) * rr, z = std::sin(ang) * rr;
                const float s = 0.25f + u01(rng) * 0.25f;
                Blade& bl = fb[static_cast<size_t>(i)];
                bl.pos.set(x, terrainGen.heightAt(x, z, terr) - 0.03f, z);
                bl.scale.set(s, s * (1.0f + u01(rng) * 0.6f), s);
                bl.yaw.setFromAxisAngle(up, u01(rng) * 6.28318530718f);
                bl.phase = u01(rng) * 6.28318530718f;
                m.compose(bl.pos, bl.yaw, bl.scale);
                fm->setMatrixAt(static_cast<size_t>(i), m);
            }
            fm->instanceMatrix()->needsUpdate();

            scene.add(fm);
            flowerMeshes.push_back(fm);
            flowerBlades.push_back(std::move(fb));
        }
    }

    // ── Rocks / boulders (instanced, flat-shaded) ────────────────────────
    {
        auto rockMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.30f, 0.28f, 0.25f)).roughness(1.f).metalness(0.f));
        rockMat->flatShading = true;
        rockMat->envMapIntensity = 0.3f;// keep boulders grey stone, not white blobs
        std::vector<std::shared_ptr<BufferGeometry>> rgeos{makeRock(1u), makeRock(2u), makeRock(3u)};
        std::vector<std::vector<Matrix4>> xf(rgeos.size());
        std::mt19937 rng(99u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        Quaternion q;
        Vector3 axis;
        Matrix4 m;
        for (int i = 0; i < 80; ++i) {
            const float x = (u01(rng) - 0.5f) * 2.f * groundHalf;
            const float z = (u01(rng) - 0.5f) * 2.f * groundHalf;
            const float sc = 0.4f + u01(rng) * 1.1f;
            axis.set(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f).normalize();
            q.setFromAxisAngle(axis, u01(rng) * 6.28318530718f);
            m.compose(Vector3(x, terrainGen.heightAt(x, z, terr) - sc * 0.25f, z), q,
                      Vector3(sc, sc * (0.75f + u01(rng) * 0.35f), sc));
            xf[static_cast<size_t>(u01(rng) * static_cast<float>(rgeos.size())) % rgeos.size()].push_back(m);
        }
        for (size_t gi = 0; gi < rgeos.size(); ++gi) {
            if (xf[gi].empty()) continue;
            auto rocks = InstancedMesh::create(rgeos[gi], rockMat, xf[gi].size());
            rocks->castShadow = true;
            rocks->receiveShadow = true;
            for (size_t i = 0; i < xf[gi].size(); ++i) rocks->setMatrixAt(i, xf[gi][i]);
            rocks->instanceMatrix()->needsUpdate();
            scene.add(rocks);
        }
    }

    // ── Camera ───────────────────────────────────────────────────────────
    PerspectiveCamera camera(55.f, canvas.aspect(), 0.1f, 1000.f);
    camera.position.set(70.f, 38.f, 70.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 6.f, 0.f);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── UI ───────────────────────────────────────────────────────────────
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;
    bool regen = false;
    float windStrength = 0.18f;
    float foliageUpdateMs = 0.f;// CPU cost of rewriting grass/flower matrices
    float uiRenderScale = 0.6f; // Vulkan render-scale (quadratic GPU lever)
    bool uiDenoise = true;      // Vulkan denoiser
    bool perfDirty = false;     // apply the above at the next frame top
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({320, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Forest");
        ImGui::Text("FPS: %.1f   trees: %d", fps, treeCount);
        ImGui::Text("grass path: %s", shaderGrass ? "GPU shader" : "CPU tilt");
        ImGui::Text("foliage CPU update: %.3f ms", foliageUpdateMs);
#ifdef THREEPP_WITH_VULKAN
        if (vk) {
            const auto t = vk->lastFrameTimings();
            ImGui::SeparatorText("Vulkan GPU (ms)");
            ImGui::Text("pathTrace %.2f  rasterGbuf %.2f", t.pathTraceMs, t.rasterGbufMs);
            ImGui::Text("overlay   %.2f  denoise    %.2f", t.overlayMs, t.denoiseMs);
            ImGui::Text("taa       %.2f  frame(cpu) %.2f", t.taaMs, t.cpuFrameMs);
            ImGui::Text("record(cpu) %.3f  ensureScene(cpu) %.3f", t.cpuRecordMs, t.cpuEnsureSceneMs);
            ImGui::SeparatorText("Vulkan perf");
            if (ImGui::SliderFloat("render scale", &uiRenderScale, 0.25f, 1.0f, "%.2f")) perfDirty = true;
            if (ImGui::Checkbox("denoise", &uiDenoise)) perfDirty = true;
        }
#endif
        ImGui::Separator();
        ImGui::SliderFloat("Spacing", &spacing, 6.f, 24.f, "%.1f");
        ImGui::SliderFloat("Fill", &fillProb, 0.1f, 1.0f, "%.2f");
        if (ImGui::Button("Re-scatter", ImVec2(-1, 0))) {
            ++scatterSeed;
            regen = true;
        }
        ImGui::SeparatorText("Wind");
        ImGui::SliderFloat("Strength", &windStrength, 0.f, 0.5f, "%.2f");
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    Clock clock;
    float tElapsed = 0.f;
    Matrix4 m;
    Quaternion qLean, qOut;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        tElapsed += dt;
        controls.update();

#ifdef THREEPP_WITH_VULKAN
        if (vk && perfDirty) {// applied here (outside render) — setRenderScale waits idle
            vk->setRenderScale(uiRenderScale);
            vk->setDenoise(uiDenoise);
            perfDirty = false;
        }
#endif

        const auto tFoliage0 = std::chrono::high_resolution_clock::now();

        // Wind sway.
        if (shaderGrass) {
            // GL: advance the GPU vertex-shader clock (no per-frame CPU work).
            grassShaderMat->uniforms["time"].setValue(tElapsed);
            grassShaderMat->uniforms["windStrength"].setValue(windStrength);
        } else if (vulkanBackend) {
            // Vulkan: GPU compute-deform GrassMesh — just hand it the clock;
            // the renderer runs grass_wind.comp + a one-instance BLAS refit.
            grassFieldVk->params.time = tElapsed;
            grassFieldVk->params.windStrength = windStrength;
        } else {
            // WGPU: CPU tilt — rewrite each instance matrix as a whole-blade
            // tilt toward the wind.
            for (size_t i = 0; i < blades.size(); ++i) {
                const Blade& bl = blades[i];
                const float gust = std::sin(tElapsed * 1.6f + bl.phase) * 0.6f +
                                   std::sin(tElapsed * 3.7f + bl.phase * 2.3f) * 0.25f;
                qLean.setFromAxisAngle(windAxis, gust * windStrength);
                qOut.multiplyQuaternions(qLean, bl.yaw);
                m.compose(bl.pos, qOut, bl.scale);
                grass->setMatrixAt(i, m);
            }
            grass->instanceMatrix()->needsUpdate();
        }

        // Wildflowers sway via CPU tilt on every backend (cheap at this count).
        for (size_t v = 0; v < flowerMeshes.size(); ++v) {
            const auto& fb = flowerBlades[v];
            for (size_t i = 0; i < fb.size(); ++i) {
                const Blade& bl = fb[i];
                const float gust = std::sin(tElapsed * 1.5f + bl.phase) * 0.6f +
                                   std::sin(tElapsed * 3.3f + bl.phase * 2.1f) * 0.2f;
                qLean.setFromAxisAngle(windAxis, gust * windStrength * 0.7f);
                qOut.multiplyQuaternions(qLean, bl.yaw);
                m.compose(bl.pos, qOut, bl.scale);
                flowerMeshes[v]->setMatrixAt(i, m);
            }
            flowerMeshes[v]->instanceMatrix()->needsUpdate();
        }

        {
            const auto tFoliage1 = std::chrono::high_resolution_clock::now();
            const float ms = std::chrono::duration<float, std::milli>(tFoliage1 - tFoliage0).count();
            foliageUpdateMs += (ms - foliageUpdateMs) * 0.1f;// smoothed
        }

        fpsAccum += dt;
        fpsFrames++;
        if (fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (regen && !ImGui::IsAnyItemActive()) {
            buildForest(scatterSeed, spacing, fillProb);
            regen = false;
        }

        renderer->render(scene, camera);
        ui.render();
    });
}
