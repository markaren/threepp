// The scene Play runs on: an in-memory copy of the authored scene.
//
// Instances are copied, assets are shared. Every node is a new object with the
// SAME uuid, name, transform, flags and userData; every distinct material is
// cloned once with the same uuid; geometry, textures, the environment and the
// animation clips are shared by pointer. Node-to-node pointers (skeleton bones,
// light targets, LOD levels, a robot's links and joints) are re-pointed at the
// copied nodes, so nothing in the copy reaches back into the authored graph.
//
// A SplatCloud cannot exist twice (a scan is up to gigabytes of host and GPU
// memory), so it is LENT: the one live node moves into the copy at the same
// child index and goes back to its authored parent, with its authored state,
// when the copy is given back.
//
// A ParticleField is runtime content (play sessions and overlays build them
// from userData) and not a document node, so the copy carries none, which is
// what the exporter writes for one as well.
//
// Two guards. G1, at build: every authored node and its copy agree on type()
// and child count, or the copy is refused with the node named, and the
// authored scene is untouched. G2, at give-back: the version of every
// attribute and index of every shared geometry is compared with the one
// recorded at build; a geometry written in place during play cannot be
// repaired (there is no second copy of its bytes) and is reported by mesh name.

#ifndef THREEPP_EDITOR_PLAYCOPY_HPP
#define THREEPP_EDITOR_PLAYCOPY_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    class BufferGeometry;
    class Object3D;
    class Scene;
    class SplatCloud;

}// namespace threepp

namespace threepp::editor {

    class PlayCopy {

    public:
        // Copy `authored`. Any editor-only object must already be detached
        // (SceneDocument does that). Returns nullptr with `error` set when G1
        // trips; the authored scene is then exactly as it was.
        [[nodiscard]] static std::unique_ptr<PlayCopy> build(Scene& authored, std::string* error = nullptr);

        PlayCopy(const PlayCopy&) = delete;
        PlayCopy& operator=(const PlayCopy&) = delete;

        // Gives back whatever is still lent.
        ~PlayCopy();

        [[nodiscard]] Scene& scene() const { return *copy_; }
        [[nodiscard]] std::shared_ptr<Scene> scenePtr() const { return copy_; }
        [[nodiscard]] Scene& authored() const { return *authored_; }

        // The copy of an authored node, or nullptr. A lent cloud is its own copy.
        [[nodiscard]] Object3D* copyOf(const Object3D& authored) const;

        // Hand the lent clouds back to their authored places and state, then run
        // G2. Returns one line per finding; empty when the authored geometry is
        // as it was at build. Idempotent: a second call returns nothing.
        std::vector<std::string> giveBack();

        // Replace `node`'s geometry by a private clone and return it: for a
        // session that writes vertices, so the write lands in the play copy and
        // not in the geometry the authored mesh shares. Idempotent per node per
        // play. A node outside any live play copy (a session started on a scene
        // that is not one) keeps its geometry, since there is no authored mesh
        // behind it to protect, and so does a lent cloud.
        static std::shared_ptr<BufferGeometry> ownGeometry(Object3D& node);

    private:
        PlayCopy() = default;

        struct Lent {
            std::shared_ptr<SplatCloud> cloud;
            Object3D* authoredParent = nullptr;
            std::size_t authoredIndex = 0;
            bool ownedByParent = true;
            // The node's own state at build (transform, flags, layers, name,
            // userData), held on a plain Object3D, plus the point-mode look and
            // the children it had: the state a reload would re-apply, and the
            // place in the authored tree.
            std::shared_ptr<Object3D> state;
            float pointMix = 0.f;
            float pointSize = 2.f;
            std::vector<Object3D*> children;
        };

        struct GeometryRecord {
            std::shared_ptr<BufferGeometry> geometry;
            std::string mesh;
            std::vector<std::pair<std::string, unsigned int>> versions;
            bool hasIndex = false;
            unsigned int indexVersion = 0;
        };

        std::shared_ptr<Scene> authored_;
        std::shared_ptr<Scene> copy_;
        std::unordered_map<const Object3D*, Object3D*> copies_;
        std::vector<Lent> lent_;
        std::vector<GeometryRecord> geometries_;
        std::unordered_set<const Object3D*> owned_;
        bool givenBack_ = false;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PLAYCOPY_HPP
