
#include "threepp/extras/editor/PlayCopy.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/light_interfaces.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <mutex>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // The live play copies, by the copy's root, so ownGeometry can tell a node
    // of a play copy from a node of any other scene without the caller having
    // to hand it the PlayCopy.
    std::mutex registryMutex;
    std::unordered_map<const Object3D*, PlayCopy*> registry;

    // Matched on type(), as the exporter matches it (ObjectExporter.cpp,
    // isUnexportable), so this file does not include the particle headers.
    bool isParticleField(const Object3D& object) {

        return object.type() == "ParticleField";
    }

    std::string describe(const Object3D& object) {

        return "'" + (object.name.empty() ? object.uuid : object.name) + "' (" + object.type() + ")";
    }

    // A node owned by a shared_ptr is referred to through it; one that is not
    // (a scene a test keeps on the stack) is referred to without being owned.
    template<class T>
    std::shared_ptr<T> refer(Object3D& node) {

        if (node.weak_from_this().expired()) {
            return std::shared_ptr<T>(std::shared_ptr<T>{}, dynamic_cast<T*>(&node));
        }
        return std::dynamic_pointer_cast<T>(node.shared_from_this());
    }

    // Move `child`, already a child of `parent`, to position `index` in its
    // list. Object3D has no insert-at; the list order is what the exporter and
    // the hierarchy panel show, so a lent node keeps its place.
    void placeAt(Object3D& parent, Object3D& child, std::size_t index) {

        auto& children = parent.children;
        const auto it = std::find(children.begin(), children.end(), &child);
        if (it == children.end()) return;
        children.erase(it);
        index = std::min(index, children.size());
        children.insert(children.begin() + static_cast<std::ptrdiff_t>(index), &child);
    }

    std::size_t indexIn(const Object3D& parent, const Object3D& child) {

        const auto& children = parent.children;
        return static_cast<std::size_t>(std::find(children.begin(), children.end(), &child) - children.begin());
    }

    // Copies the tree node by node: clone(false) of each node, then its
    // children by recursion, which pairs every authored node with its copy as
    // it goes. A recursive clone() would not: LOD::copy clones its levels
    // itself and drops its other children, and Robot and SplatCloud have no
    // createDefault() and would come back as a plain Object3D.
    struct Builder {

        std::unordered_map<const Object3D*, Object3D*>& copies;
        std::vector<std::pair<Object3D*, Object3D*>> pairs;// authored, copy
        // An authored cloud and the empty node standing in its place in the
        // copy until G1 has passed and the cloud itself can move over.
        std::vector<std::pair<Object3D*, Object3D*>> standIns;
        std::unordered_map<const Material*, std::shared_ptr<Material>> materials;

        std::shared_ptr<Material> material(const std::shared_ptr<Material>& source) {

            if (!source) return nullptr;
            auto& copy = materials[source.get()];
            if (!copy) {
                copy = source->clone();
                copy->setUuid(source->uuid());
            }
            return copy;
        }

        std::shared_ptr<Object3D> node(Object3D& source) {

            std::shared_ptr<Object3D> copy;
            if (dynamic_cast<Robot*>(&source)) {
                // The joint table is rebuilt over the copied nodes afterwards.
                copy = Robot::create();
                copy->Object3D::copy(source, false);
            } else if (auto* lod = source.as<LOD>()) {
                // LOD::copy clones the levels itself; the levels are copied
                // below with the other children and re-registered there.
                auto lodCopy = LOD::create();
                lodCopy->Object3D::copy(source, false);
                lodCopy->autoUpdate = lod->autoUpdate;
                copy = lodCopy;
            } else {
                copy = source.clone(false);
            }

            copy->uuid = source.uuid;
            // Object3D::copy leaves the clips out; they are assets, shared.
            copy->animations = source.animations;
            // LightShadow::copy copies the shadow camera's settings into the
            // copy's own camera, which has a uuid of its own; the document
            // names it.
            if (const auto* shadowed = dynamic_cast<const LightWithShadow*>(&source)) {
                auto* shadowCopy = dynamic_cast<LightWithShadow*>(copy.get());
                if (shadowed->shadow && shadowed->shadow->camera && shadowCopy && shadowCopy->shadow &&
                    shadowCopy->shadow->camera) {
                    shadowCopy->shadow->camera->uuid = shadowed->shadow->camera->uuid;
                }
            }

            if (auto* withMaterials = dynamic_cast<ObjectWithMaterials*>(copy.get())) {
                auto list = withMaterials->materials();
                for (auto& m : list) m = material(m);
                withMaterials->setMaterials(list);
            } else if (auto* sprite = copy->as<Sprite>()) {
                if (auto m = std::dynamic_pointer_cast<SpriteMaterial>(material(sprite->material()))) {
                    sprite->setMaterial(m);
                }
            }

            copies[&source] = copy.get();
            pairs.emplace_back(&source, copy.get());

            const auto* lodSource = source.as<LOD>();
            auto* lodCopy = copy->as<LOD>();

            for (auto* child : source.children) {
                if (!child || isParticleField(*child)) continue;

                if (child->as<SplatCloud>()) {
                    auto standIn = Object3D::create();
                    copy->add(standIn);
                    standIns.emplace_back(child, standIn.get());
                    continue;
                }

                auto childCopy = node(*child);
                const Level* level = nullptr;
                if (lodSource) {
                    for (const auto& l : lodSource->getLevels()) {
                        if (l.object == child) level = &l;
                    }
                }
                if (level && lodCopy) {
                    lodCopy->addLevel(childCopy, level->distance);
                } else {
                    copy->add(childCopy);
                }
            }

            return copy;
        }

        Object3D* find(const Object3D* authored) const {

            const auto it = copies.find(authored);
            return it == copies.end() ? nullptr : it->second;
        }

        // Skeletons, light targets and robots: everything that names another
        // node. Returns false with `error` set when a pointer leads out of the
        // scene, where following it would let the copy drive an authored node.
        bool repoint(std::string* error) {

            std::unordered_map<const Skeleton*, std::shared_ptr<Skeleton>> skeletons;

            for (const auto& [source, copy] : pairs) {

                if (auto* skinned = copy->as<SkinnedMesh>(); skinned && skinned->skeleton) {
                    auto& skeleton = skeletons[skinned->skeleton.get()];
                    if (!skeleton) {
                        const auto& authoredSkeleton = *skinned->skeleton;
                        std::vector<std::shared_ptr<Bone>> bones;
                        bones.reserve(authoredSkeleton.bones.size());
                        for (const auto& bone : authoredSkeleton.bones) {
                            auto* boneCopy = bone ? find(bone.get()) : nullptr;
                            if (!boneCopy || !boneCopy->as<Bone>()) {
                                if (error) {
                                    *error = "skinned mesh " + describe(*source) + " has a bone " +
                                             (bone ? describe(*bone) : std::string("(null)")) +
                                             " outside the scene, which a play copy cannot copy";
                                }
                                return false;
                            }
                            bones.push_back(refer<Bone>(*boneCopy));
                        }
                        skeleton = Skeleton::create(bones, authoredSkeleton.boneInverses);
                        skeleton->setUuid(authoredSkeleton.uuid());
                    }
                    skinned->skeleton = skeleton;
                }

                if (const auto* light = dynamic_cast<const LightWithTarget*>(source)) {
                    // A target elsewhere in the scene is re-pointed at its copy.
                    // The default target, or one outside the scene, is what
                    // the light's own copy() already chose.
                    if (auto* target = find(&light->target())) {
                        if (auto* lightCopy = dynamic_cast<LightWithTarget*>(copy)) lightCopy->setTarget(*target);
                    }
                }

                if (const auto* robot = dynamic_cast<const Robot*>(source)) {
                    auto* robotCopy = dynamic_cast<Robot*>(copy);
                    if (!robotCopy) continue;

                    for (const auto& link : robot->links()) {
                        if (auto* linkCopy = link ? find(link.get()) : nullptr) {
                            robotCopy->addLink(refer<Object3D>(*linkCopy));
                        }
                    }
                    const auto& joints = robot->jointNodes();
                    const auto& infos = robot->jointInfos();
                    for (std::size_t i = 0; i < joints.size() && i < infos.size(); ++i) {
                        auto* jointCopy = joints[i] ? find(joints[i].get()) : nullptr;
                        if (!jointCopy) continue;// frozen, as a reload leaves it
                        const auto [restPosition, restRotation] = robot->jointRestPose(i);
                        robotCopy->addJoint(refer<Object3D>(*jointCopy), infos[i], restPosition, restRotation);
                    }
                    robotCopy->finalizeInPlace();
                    if (!robot->endEffectorLink().empty()) robotCopy->setEndEffector(robot->endEffectorLink());

                    // setJointValue fills the joint table; it also re-poses the
                    // joint from its rest pose, so the copied pose goes back on
                    // afterwards, bit for bit.
                    const auto values = robot->jointValues();
                    for (std::size_t i = 0; i < values.size() && i < robotCopy->numDOF(); ++i) {
                        robotCopy->setJointValue(i, values[i]);
                    }
                    for (const auto& joint : joints) {
                        auto* jointCopy = joint ? find(joint.get()) : nullptr;
                        if (!jointCopy) continue;
                        jointCopy->position.copy(joint->position);
                        jointCopy->quaternion.copy(joint->quaternion);
                    }
                }
            }
            return true;
        }

        // G1: the same type and the same number of children on both sides
        // (less the particle fields, which the copy leaves out).
        bool agree(std::string* error) const {

            for (const auto& [source, copy] : pairs) {

                if (source->type() != copy->type()) {
                    if (error) {
                        *error = "node " + describe(*source) + " cannot be copied for play: its copy is a " +
                                 copy->type();
                    }
                    return false;
                }
                const auto expected = static_cast<std::size_t>(std::count_if(
                        source->children.begin(), source->children.end(),
                        [](const Object3D* c) { return c && !isParticleField(*c); }));
                if (copy->children.size() != expected) {
                    if (error) {
                        *error = "node " + describe(*source) + " has " + std::to_string(expected) +
                                 " children and its play copy " + std::to_string(copy->children.size());
                    }
                    return false;
                }
            }
            return true;
        }
    };

}// namespace


std::unique_ptr<PlayCopy> PlayCopy::build(Scene& authored, std::string* error) {

    std::unique_ptr<PlayCopy> play(new PlayCopy());
    play->authored_ = refer<Scene>(authored);

    Builder builder{play->copies_};
    play->copy_ = std::dynamic_pointer_cast<Scene>(builder.node(authored));

    if (!play->copy_) {
        if (error) *error = "the scene could not be copied for play";
        return nullptr;
    }
    if (!builder.repoint(error) || !builder.agree(error)) return nullptr;

    // G2's baseline: every geometry the copy shares with the authored scene.
    std::unordered_set<const BufferGeometry*> seen;
    authored.traverse([&](Object3D& object) {
        if (object.as<SplatCloud>() || isParticleField(object)) return;
        auto geometry = object.geometry();
        if (!geometry || !seen.insert(geometry.get()).second) return;

        GeometryRecord record;
        record.geometry = geometry;
        record.mesh = object.name.empty() ? object.uuid : object.name;
        for (const auto& [name, attribute] : geometry->getAttributes()) {
            if (attribute) record.versions.emplace_back(name, attribute->version);
        }
        if (const auto* index = geometry->getIndex()) {
            record.hasIndex = true;
            record.indexVersion = index->version;
        }
        play->geometries_.push_back(std::move(record));
    });

    // G1 passed: from here on nothing can fail, so the clouds move over.
    for (const auto& [cloudNode, standIn] : builder.standIns) {

        Lent lent;
        lent.cloud = refer<SplatCloud>(*cloudNode);
        lent.authoredParent = cloudNode->parent;
        lent.authoredIndex = indexIn(*cloudNode->parent, *cloudNode);
        lent.state = Object3D::create();
        lent.state->Object3D::copy(*cloudNode, false);
        lent.pointMix = lent.cloud->pointMix();
        lent.pointSize = lent.cloud->pointSize();
        lent.children = cloudNode->children;

        lent.ownedByParent = cloudNode->removeFromParent() != nullptr;

        auto* copyParent = standIn->parent;
        const auto index = indexIn(*copyParent, *standIn);
        copyParent->remove(*standIn);
        copyParent->add(lent.cloud);
        placeAt(*copyParent, *lent.cloud, index);

        play->copies_[cloudNode] = cloudNode;
        play->lent_.push_back(std::move(lent));
    }

    {
        std::lock_guard lock(registryMutex);
        registry[play->copy_.get()] = play.get();
    }
    return play;
}

PlayCopy::~PlayCopy() {

    giveBack();

    std::lock_guard lock(registryMutex);
    registry.erase(copy_.get());
}

Object3D* PlayCopy::copyOf(const Object3D& authored) const {

    const auto it = copies_.find(&authored);
    return it == copies_.end() ? nullptr : it->second;
}

std::vector<std::string> PlayCopy::giveBack() {

    if (givenBack_) return {};
    givenBack_ = true;

    // Last lent, first returned: each goes back to the index it had when the
    // ones after it in the same parent were still there.
    for (auto it = lent_.rbegin(); it != lent_.rend(); ++it) {

        auto& lent = *it;
        auto& cloud = *lent.cloud;

        cloud.removeFromParent();
        // A play session may hang nodes under a cloud (a sensor surface); they
        // belong to the play.
        const auto children = cloud.children;
        for (auto* child : children) {
            if (std::find(lent.children.begin(), lent.children.end(), child) == lent.children.end()) {
                cloud.remove(*child);
            }
        }
        cloud.Object3D::copy(*lent.state, false);
        cloud.setPointMix(lent.pointMix);
        cloud.setPointSize(lent.pointSize);

        if (lent.authoredParent) {
            if (lent.ownedByParent) {
                lent.authoredParent->add(lent.cloud);
            } else {
                lent.authoredParent->addRef(cloud);
            }
            placeAt(*lent.authoredParent, cloud, lent.authoredIndex);
        }
    }
    lent_.clear();

    // G2.
    std::vector<std::string> findings;
    for (const auto& record : geometries_) {

        std::vector<std::string> changed;
        const auto& attributes = record.geometry->getAttributes();
        for (const auto& [name, version] : record.versions) {
            const auto found = attributes.find(name);
            if (found == attributes.end() || !found->second || found->second->version != version) {
                changed.push_back(name);
            }
        }
        const auto* index = record.geometry->getIndex();
        if (record.hasIndex != (index != nullptr) || (index && index->version != record.indexVersion)) {
            changed.emplace_back("index");
        }
        if (changed.empty()) continue;

        std::string list;
        for (const auto& name : changed) list += (list.empty() ? "" : ", ") + name;
        findings.push_back("play wrote the geometry of mesh '" + record.mesh + "' in place (" + list +
                           "); the authored scene keeps the played data until it is reopened");
    }
    geometries_.clear();
    return findings;
}

std::shared_ptr<BufferGeometry> PlayCopy::ownGeometry(Object3D& node) {

    auto geometry = node.geometry();
    if (!geometry || node.as<SplatCloud>()) return geometry;

    const Object3D* root = &node;
    while (root->parent) root = root->parent;

    std::lock_guard lock(registryMutex);
    const auto it = registry.find(root);
    if (it == registry.end()) return geometry;
    auto& play = *it->second;
    if (!play.owned_.insert(&node).second) return geometry;

    auto own = geometry->clone();
    if (auto* mesh = node.as<Mesh>()) {
        mesh->setGeometry(own);
    } else if (auto* line = node.as<Line>()) {
        line->setGeometry(own);
    } else if (auto* points = node.as<Points>()) {
        points->setGeometry(own);
    } else {
        play.owned_.erase(&node);
        return geometry;
    }
    return own;
}
