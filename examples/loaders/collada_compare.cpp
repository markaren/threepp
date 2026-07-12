// Console tool: loads a .dae with both ColladaLoader and AssimpLoader and
// prints comparable scene summaries (node transforms, mesh/vertex totals,
// world AABBs, animation clips) so the two importers can be diffed.
//
// Usage: collada_compare [file.dae ...]
// With no arguments the bundled stormtrooper and youbot assets are used.

#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    struct Totals {
        int meshes{0};
        long long vertices{0};
    };

    void dumpNode(Object3D& obj, int depth, Totals& totals) {
        Vector3 p, s;
        Quaternion q;
        obj.matrixWorld->decompose(p, q, s);
        if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);// canonical sign

        std::string meshInfo;
        if (auto mesh = obj.as<Mesh>()) {
            const auto* pos = mesh->geometry() ? mesh->geometry()->getAttribute<float>("position") : nullptr;
            const int count = pos ? pos->count() : 0;
            totals.meshes += 1;
            totals.vertices += count;
            Box3 bb;
            bb.setFromObject(obj);
            char buf[160];
            std::snprintf(buf, sizeof(buf), " [mesh verts=%d aabb=(%.3f,%.3f,%.3f)-(%.3f,%.3f,%.3f)]",
                          count, bb.min().x, bb.min().y, bb.min().z, bb.max().x, bb.max().y, bb.max().z);
            meshInfo = buf;
        }

        std::printf("%*s%s p(%.3f,%.3f,%.3f) q(%.3f,%.3f,%.3f,%.3f) s(%.2f,%.2f,%.2f)%s\n",
                    depth * 2, "", obj.name.empty() ? "<unnamed>" : obj.name.c_str(),
                    p.x, p.y, p.z,
                    static_cast<float>(q.x), static_cast<float>(q.y), static_cast<float>(q.z), static_cast<float>(q.w),
                    s.x, s.y, s.z, meshInfo.c_str());

        for (const auto& child : obj.children) {
            dumpNode(*child, depth + 1, totals);
        }
    }

    void dumpScene(const char* loaderName, const std::shared_ptr<Group>& scene) {
        if (!scene) {
            std::printf("== %s: LOAD FAILED ==\n", loaderName);
            return;
        }
        scene->updateMatrixWorld(true);

        Box3 bb;
        bb.setFromObject(*scene);
        std::printf("== %s ==\n", loaderName);
        std::printf("scene aabb: (%.3f,%.3f,%.3f)-(%.3f,%.3f,%.3f)\n",
                    bb.min().x, bb.min().y, bb.min().z, bb.max().x, bb.max().y, bb.max().z);

        Totals totals;
        for (const auto& child : scene->children) {
            dumpNode(*child, 1, totals);
        }
        std::printf("meshes: %d, total verts: %lld\n", totals.meshes, totals.vertices);

        std::printf("animations: %zu\n", scene->animations.size());
        for (const auto& clip : scene->animations) {
            std::printf("  clip '%s' duration=%.3f\n", clip->name().c_str(), clip->getDuration());
        }
    }

}// namespace

int main(int argc, char** argv) {

    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) files.emplace_back(argv[i]);
    if (files.empty()) {
        files.emplace_back(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
        files.emplace_back(std::string(DATA_FOLDER) + "/models/collada/youbot.dae");
    }

    for (const auto& file : files) {
        std::printf("\n######## %s ########\n", file.c_str());

        ColladaLoader collada;
        dumpScene("ColladaLoader", collada.load(file));
        std::fflush(stdout);

        try {
            AssimpLoader assimp;
            dumpScene("AssimpLoader", assimp.load(file));
        } catch (const std::exception& ex) {
            std::printf("== AssimpLoader: EXCEPTION: %s ==\n", ex.what());
        }
        std::fflush(stdout);
    }

    return 0;
}
