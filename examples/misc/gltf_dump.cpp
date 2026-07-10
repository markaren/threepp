// Deterministic structural dump of a glTF/GLB file, plus load timing.
//
// Structural summary -> stdout (diff this across loader changes).
// Timing / memory      -> stderr (kept separate so structural diffs stay clean).
//
// Usage: gltf_dump <path.glb|.gltf> [--repeat N]
//   --repeat N : load N times, report per-load and median wall-clock (default 1)

#include "threepp/loaders/GLTFLoader.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

using namespace threepp;

namespace {

    std::string wrapStr(TextureWrapping w) {
        switch (w) {
            case TextureWrapping::Repeat: return "Repeat";
            case TextureWrapping::ClampToEdge: return "Clamp";
            case TextureWrapping::MirroredRepeat: return "Mirror";
        }
        return "?";
    }

    std::string filterStr(Filter f) {
        switch (f) {
            case Filter::Nearest: return "Nearest";
            case Filter::NearestMipmapNearest: return "NearestMipNearest";
            case Filter::NearestMipmapLinear: return "NearestMipLinear";
            case Filter::Linear: return "Linear";
            case Filter::LinearMipmapNearest: return "LinearMipNearest";
            case Filter::LinearMipmapLinear: return "LinearMipLinear";
        }
        return "?";
    }

    std::string csStr(ColorSpace c) {
        switch (c) {
            case ColorSpace::NoColorSpace: return "None";
            case ColorSpace::sRGB: return "sRGB";
            case ColorSpace::Linear: return "Linear";
            case ColorSpace::RGBE: return "RGBE";
        }
        return "?";
    }

    std::string f4(float v) {
        // Stable formatting: fixed 4 decimals, normalize -0.
        if (v == 0.0f) v = 0.0f;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.4f", v);
        return buf;
    }

    void dumpTexture(std::ostream& os, const std::string& slot, const std::shared_ptr<Texture>& t) {
        if (!t) return;
        os << "      " << slot << ": ";
        if (t->images().empty()) {
            os << "(no image) ";
        } else {
            os << t->image().width() << "x" << t->image().height() << " ";
        }
        os << "cs=" << csStr(t->colorSpace)
           << " wrap=" << wrapStr(t->wrapS) << "/" << wrapStr(t->wrapT)
           << " filt=" << filterStr(t->minFilter) << "/" << filterStr(t->magFilter)
           << " mips=" << (t->generateMipmaps ? 1 : 0)
           << " off=" << f4(t->offset.x) << "," << f4(t->offset.y)
           << " rep=" << f4(t->repeat.x) << "," << f4(t->repeat.y)
           << " rot=" << f4(t->rotation)
           << " uv=" << t->texCoord
           << "\n";
    }

    std::string sideStr(Side s) {
        switch (s) {
            case Side::Front: return "Front";
            case Side::Back: return "Back";
            case Side::Double: return "Double";
        }
        return "?";
    }

    // Dump a material's key params + populated texture slots. Dedup by pointer.
    void dumpMaterial(std::ostream& os, int idx, Material* m) {
        os << "  MAT[" << idx << "] \"" << m->name << "\" type=" << m->type()
           << " side=" << sideStr(m->side)
           << " transparent=" << (m->transparent ? 1 : 0)
           << " opacity=" << f4(m->opacity)
           << " alphaTest=" << f4(m->alphaTest) << "\n";

        if (auto* sm = dynamic_cast<MeshStandardMaterial*>(m)) {
            os << "    color=" << f4(sm->color.r) << "," << f4(sm->color.g) << "," << f4(sm->color.b)
               << " rough=" << f4(sm->roughness) << " metal=" << f4(sm->metalness)
               << " emissive=" << f4(sm->emissive.r) << "," << f4(sm->emissive.g) << "," << f4(sm->emissive.b)
               << " emissiveInt=" << f4(sm->emissiveIntensity)
               << " normalScale=" << f4(sm->normalScale.x) << "," << f4(sm->normalScale.y)
               << " aoInt=" << f4(sm->aoMapIntensity) << "\n";
            if (auto* pm = dynamic_cast<MeshPhysicalMaterial*>(m)) {
                os << "    phys transmission=" << f4(pm->transmission)
                   << " ior=" << f4(pm->ior)
                   << " clearcoat=" << f4(pm->clearcoat)
                   << " thickness=" << f4(pm->thickness)
                   << " sheenRough=" << f4(pm->sheenRoughness)
                   << " specInt=" << f4(pm->specularIntensity)
                   << " iridescence=" << f4(pm->iridescence) << "\n";
            }
            dumpTexture(os, "map", sm->map);
            dumpTexture(os, "normalMap", sm->normalMap);
            dumpTexture(os, "roughnessMap", sm->roughnessMap);
            dumpTexture(os, "metalnessMap", sm->metalnessMap);
            dumpTexture(os, "emissiveMap", sm->emissiveMap);
            dumpTexture(os, "aoMap", sm->aoMap);
        } else if (auto* bm = dynamic_cast<MeshBasicMaterial*>(m)) {
            os << "    color=" << f4(bm->color.r) << "," << f4(bm->color.g) << "," << f4(bm->color.b) << "\n";
            dumpTexture(os, "map", bm->map);
        }
    }

    struct Ctx {
        std::ostream& os;
        // Material dedup, printed in first-encounter (traversal) order.
        std::unordered_set<Material*> seenMats;
        std::vector<Material*> matOrder;
        // Aggregate geometry stats.
        std::unordered_set<const void*> seenGeom;
        long long totalVerts = 0;
        long long totalTris = 0;
        int uniqueGeoms = 0;
        int meshCount = 0;
    };

    void dumpGeometry(Ctx& ctx, BufferGeometry* g, const std::string& indent) {
        auto& os = ctx.os;
        auto* pos = g->getAttribute<float>("position");
        int vcount = pos ? pos->count() : 0;
        int icount = g->hasIndex() ? g->getIndex()->count() : 0;

        // Morph target count (position channel).
        size_t morphs = 0;
        if (auto* mp = g->getMorphAttribute("position")) morphs = mp->size();

        os << indent << "geom verts=" << vcount << " idx=" << icount << " morphs=" << morphs
           << " hasNormal=" << (g->hasAttribute("normal") ? 1 : 0)
           << " hasUV=" << (g->hasAttribute("uv") ? 1 : 0)
           << " hasUV2=" << (g->hasAttribute("uv2") ? 1 : 0)
           << " hasColor=" << (g->hasAttribute("color") ? 1 : 0)
           << " hasTangent=" << (g->hasAttribute("tangent") ? 1 : 0)
           << " hasSkin=" << (g->hasAttribute("skinIndex") ? 1 : 0) << "\n";

        if (pos && vcount > 0) {
            const auto& a = pos->array();
            double mn[3] = {1e300, 1e300, 1e300};
            double mx[3] = {-1e300, -1e300, -1e300};
            double sum[3] = {0, 0, 0};
            int is = pos->itemSize();
            for (int i = 0; i < vcount; ++i) {
                for (int c = 0; c < 3 && c < is; ++c) {
                    double v = a[static_cast<size_t>(i) * is + c];
                    mn[c] = std::min(mn[c], v);
                    mx[c] = std::max(mx[c], v);
                    sum[c] += v;
                }
            }
            os << indent << "pos min=" << f4((float) mn[0]) << "," << f4((float) mn[1]) << "," << f4((float) mn[2])
               << " max=" << f4((float) mx[0]) << "," << f4((float) mx[1]) << "," << f4((float) mx[2])
               << " sum=" << f4((float) sum[0]) << "," << f4((float) sum[1]) << "," << f4((float) sum[2]) << "\n";
        }

        // Aggregates (count each unique geometry once).
        if (ctx.seenGeom.insert(g).second) {
            ctx.uniqueGeoms++;
            ctx.totalVerts += vcount;
            ctx.totalTris += (icount > 0 ? icount : vcount) / 3;
        }
    }

    void dumpNode(Ctx& ctx, Object3D* obj, int depth) {
        auto& os = ctx.os;
        std::string indent(depth * 2, ' ');
        os << indent << "\"" << obj->name << "\" type=" << obj->type()
           << " children=" << obj->children.size();

        if (auto* inst = obj->as<InstancedMesh>()) {
            os << " [instances=" << inst->count() << "]";
        }
        os << "\n";

        if (auto* mesh = obj->as<Mesh>()) {
            ctx.meshCount++;
            if (auto geo = mesh->geometry()) {
                dumpGeometry(ctx, geo.get(), indent + "  ");
            }
            if (auto mat = mesh->material()) {
                if (ctx.seenMats.insert(mat.get()).second) ctx.matOrder.push_back(mat.get());
                os << indent + "  " << "mat=\"" << mat->name << "\" (" << mat->type() << ")\n";
            }
        }

        // Deterministic order = child insertion order.
        for (auto* c : obj->children) dumpNode(ctx, c, depth + 1);
    }

}// namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path.glb|.gltf> [--repeat N]\n";
        return 1;
    }
    std::filesystem::path path = argv[1];
    int repeat = 1;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
    }

    // Timing loop (stderr).
    std::vector<double> times;
    std::optional<GLTFResult> result;
    for (int r = 0; r < repeat; ++r) {
        GLTFLoader loader;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = loader.load(path);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        std::cerr << "load[" << r << "] " << f4((float) ms) << " ms\n";
        if (r == repeat - 1) result = std::move(res);
    }
    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    std::cerr << "median " << f4((float) median) << " ms  (" << repeat << " runs)\n";

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        std::cerr << "peakWorkingSet " << (pmc.PeakWorkingSetSize / (1024 * 1024)) << " MB\n";
    }
#endif

    // Structural dump (stdout).
    std::ostream& os = std::cout;
    if (!result) {
        os << "LOAD FAILED\n";
        return 2;
    }

    os << "FILE " << path.filename().string() << "\n";
    os << "scenes=" << result->scenes.size()
       << " animations=" << result->animations.size()
       << " variants=" << result->variants.names.size() << "\n";
    if (!result->variants.names.empty()) {
        os << "variantNames:";
        for (const auto& n : result->variants.names) os << " \"" << n << "\"";
        os << "\n";
    }

    Ctx ctx{os};
    for (size_t s = 0; s < result->scenes.size(); ++s) {
        os << "=== SCENE " << s << " \"" << (result->scenes[s] ? result->scenes[s]->name : "") << "\" ===\n";
        if (result->scenes[s]) dumpNode(ctx, result->scenes[s].get(), 0);
    }

    os << "=== MATERIALS (" << ctx.matOrder.size() << ") ===\n";
    for (size_t i = 0; i < ctx.matOrder.size(); ++i) {
        dumpMaterial(os, (int) i, ctx.matOrder[i]);
    }

    os << "=== ANIMATIONS (" << result->animations.size() << ") ===\n";
    // Sort by name for determinism (clip order can vary; content shouldn't).
    std::vector<std::pair<std::string, std::string>> clips;
    for (const auto& clip : result->animations) {
        if (!clip) continue;
        std::ostringstream line;
        line << "duration=" << f4(clip->getDuration());
        clips.emplace_back(clip->name(), line.str());
    }
    std::sort(clips.begin(), clips.end());
    for (const auto& [name, info] : clips) {
        os << "  \"" << name << "\" " << info << "\n";
    }

    os << "=== AGGREGATE ===\n";
    os << "meshNodes=" << ctx.meshCount
       << " uniqueGeoms=" << ctx.uniqueGeoms
       << " totalVerts=" << ctx.totalVerts
       << " totalTris=" << ctx.totalTris
       << " materials=" << ctx.matOrder.size() << "\n";

    return 0;
}
