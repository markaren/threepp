
#include "threepp/loaders/URDFLoader.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "pugixml.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/xacro/Expand.hpp"
#include "threepp/loaders/xacro/PackageResolver.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <unordered_map>


using namespace threepp;

namespace {

    Vector3 parseTupleString(const std::string& strValues) {
        if (strValues.empty()) return {};
        // URDF xyz/rpy/size attributes may separate components with runs of
        // whitespace (e.g. `xyz="0 1      0"`); collect the non-empty tokens so a
        // split on ' ' doesn't yield empty fields that fail float conversion.
        std::array<float, 3> v{};
        size_t n = 0;
        for (const auto& tok : utils::split(strValues, ' ')) {
            if (tok.empty() || n >= 3) continue;
            v[n++] = utils::parseFloat(tok);
        }
        return {v[0], v[1], v[2]};
    }

    void applyRotation(const std::shared_ptr<Object3D>& object, const Vector3& rotation) {

        static Quaternion tempQuaternion;
        static Euler tempEuler;

        object->rotation.set(0, 0, 0);

        tempEuler.set(rotation.x, rotation.y, rotation.z, Euler::RotationOrders::ZYX);
        tempQuaternion.setFromEuler(tempEuler);
        tempQuaternion.multiply(object->quaternion);
        object->quaternion.copy(tempQuaternion);
    }

    std::shared_ptr<Material> getMaterial(const pugi::xml_node& node) {
        const auto mtl = MeshStandardMaterial::create();

        // A <material> may be a named reference with no inline <color> (e.g.
        // <material name="grey"/>), and rgba may hold fewer than 4 components or
        // runs of whitespace between them.
        std::vector<float> rgba;
        for (const auto& tok : utils::split(node.child("color").attribute("rgba").value(), ' ')) {
            if (!tok.empty()) rgba.push_back(utils::parseFloat(tok));
        }
        if (rgba.size() < 4) return mtl;

        mtl->color.setRGB(rgba[0], rgba[1], rgba[2]);
        if (rgba[3] < 1) {
            mtl->transparent = true;
            mtl->opacity = rgba[3];
        }

        return mtl;
    }

    Robot::JointType getType(const std::string& type) {
        if (type == "revolute" || type == "continuous") {
            return Robot::JointType::Revolute;
        }
        if (type == "prismatic") {
            return Robot::JointType::Prismatic;
        }
        return Robot::JointType::Fixed;
    }

    std::optional<Robot::JointRange> getRange(const pugi::xml_node& node) {
        const auto limit = node.child("limit");
        if (!limit || !limit.attribute("lower") || !limit.attribute("upper")) return {};
        const auto min = utils::parseFloat(limit.attribute("lower").value());
        const auto max = utils::parseFloat(limit.attribute("upper").value());
        return Robot::JointRange{
                .min = min,
                .max = max};
    }

    Robot::JointInfo parseInfo(const pugi::xml_node& node) {

        auto axis = parseTupleString(node.child("axis")
                                             .attribute("xyz")
                                             .as_string("1 0 0"));

        return {
                .axis = axis.normalize(),
                .type = getType(node.attribute("type").value()),
                .name = node.attribute("name").value(),
                .range = getRange(node),
                .parent = node.child("parent").attribute("link").value(),
                .child = node.child("child").attribute("link").value()};
    }

    // Robot models reference the same mesh file from <visual> and <collision>
    // (and across repeated links); load each file once per parse and hand out
    // clones that share geometry/materials, so per-use origin/scale stay
    // independent while the expensive parse/decode happens once.
    class CachingLoader final: public Loader<Group> {

    public:
        explicit CachingLoader(Loader<Group>* inner): inner_(inner) {}

        std::shared_ptr<Group> load(const std::filesystem::path& path) override {
            if (!inner_) return nullptr;

            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(path, ec);
            const std::string key = (ec ? path : canonical).string();

            if (const auto it = cache_.find(key); it != cache_.end()) {
                return it->second->clone<Group>();
            }

            auto loaded = inner_->load(path);
            if (!loaded) return nullptr;

            // clone() shares geometry/materials but has no SkinnedMesh overload and
            // drops animations — hand such (rare for robot links) models out
            // uncached rather than degrade them.
            bool cloneable = loaded->animations.empty();
            if (cloneable) {
                loaded->traverseType<SkinnedMesh>([&cloneable](SkinnedMesh&) { cloneable = false; });
            }
            if (!cloneable) return loaded;

            cache_[key] = loaded;
            return loaded->clone<Group>();
        }

    private:
        Loader<Group>* inner_;
        std::unordered_map<std::string, std::shared_ptr<Group>> cache_;
    };

    std::filesystem::path findPackageRoot(const std::filesystem::path& start) {
        for (auto path = start; path != path.parent_path(); path = path.parent_path()) {
            if (exists(path / "package.xml")) {
                return path;
            }

        }
        return {};
    }

    // package://pkg/rel, resolved by the same PackageResolver $(find pkg) uses, with the
    // basePath-relative and walk-up attempts kept as fallbacks.
    std::filesystem::path getModelPath(xacro::PackageResolver* packages,
                                       const std::filesystem::path& basePath, std::string_view fileName) {
        if (!fileName.starts_with("package://")) {
            return basePath / fileName;
        }

        const std::string uri(fileName.substr(10));
        const auto relative = std::filesystem::path(uri);

        if (packages) {
            const auto slash = uri.find('/');
            const std::string package = uri.substr(0, slash);
            const std::filesystem::path tail =
                    slash == std::string::npos ? std::filesystem::path{} : std::filesystem::path(uri.substr(slash + 1));
            if (const auto dir = packages->resolve(package, basePath)) {
                if (auto p = *dir / tail; exists(p)) return p;
            }
        }

        if (auto p = basePath / relative; exists(p)) return p;

        const auto pkgRoot = findPackageRoot(basePath);
        if (pkgRoot.empty()) return {};

        if (auto p = pkgRoot / relative; exists(p)) return p;
        if (auto p = pkgRoot.parent_path() / relative; exists(p)) return p;

        return {};
    }

    std::shared_ptr<Object3D> parseGeometryNode(xacro::PackageResolver* packages, const std::filesystem::path& path,
                                                Loader<Group>* loader, const pugi::xml_node& geometry) {
        if (const auto mesh = geometry.child("mesh")) {
            const auto fileName = getModelPath(packages, path.parent_path(), mesh.attribute("filename").value());
            if (fileName.empty()) {
                return nullptr;
            }

            if (!loader) {
                std::cerr << "[URDFLoader] No geometry loader set, cannot load " << fileName << std::endl;
                return nullptr;
            }

            if (auto obj = loader->load(fileName)) {
                if (const auto scale = mesh.attribute("scale")) {
                    obj->scale.copy(parseTupleString(scale.value()));
                }

                return obj;
            }
        }
        if (const auto box = geometry.child("box")) {
            const auto size = parseTupleString(box.attribute("size").value());
            auto obj = Mesh::create(BoxGeometry::create(1, 1, 1));
            obj->scale.copy(size);

            return obj;
        }
        if (const auto sphere = geometry.child("sphere")) {
            const auto radius = utils::parseFloat(sphere.attribute("radius").value());
            auto obj = Mesh::create(SphereGeometry::create(radius));

            return obj;
        }
        if (const auto cylinder = geometry.child("cylinder")) {
            const auto radius = utils::parseFloat(cylinder.attribute("radius").value());
            const auto length = utils::parseFloat(cylinder.attribute("length").value());
            auto obj = Mesh::create(CylinderGeometry::create(radius, radius, length));
            obj->rotateX(math::PI / 2);
            return obj;
        }

        return nullptr;
    }

    // <origin xyz rpy> -> a 4x4 transform (URDF rpy is extrinsic XYZ == threepp Euler order ZYX,
    // matching applyRotation above).
    Matrix4 originMatrix(const pugi::xml_node& node) {
        Matrix4 m;
        const auto origin = node.child("origin");
        if (!origin) return m;
        const Vector3 xyz = parseTupleString(origin.attribute("xyz").value());
        const Vector3 rpy = parseTupleString(origin.attribute("rpy").value());
        Euler euler(rpy.x, rpy.y, rpy.z, Euler::RotationOrders::ZYX);
        Quaternion q;
        q.setFromEuler(euler);
        m.compose(xyz, q, Vector3(1, 1, 1));
        return m;
    }

    // A link's <collision> as a PhysX-buildable primitive: box/sphere directly, cylinder approximated by
    // a capsule, mesh approximated by its bounding box (articulation links take primitive shapes).
    URDFArticulationDesc::Collision parseCollisionShape(xacro::PackageResolver* packages,
                                                        const pugi::xml_node& collisionNode,
                                                        const std::filesystem::path& path, Loader<Group>* loader) {
        using Coll = URDFArticulationDesc::Collision;
        Coll c;
        if (!collisionNode) return c;
        const auto geometry = collisionNode.child("geometry");
        if (!geometry) return c;
        c.origin = originMatrix(collisionNode);
        if (const auto box = geometry.child("box")) {
            c.shape = Coll::Shape::Box;
            c.halfExtents.copy(parseTupleString(box.attribute("size").value())).multiplyScalar(0.5f);
            return c;
        }
        if (const auto sphere = geometry.child("sphere")) {
            c.shape = Coll::Shape::Sphere;
            c.radius = utils::parseFloat(sphere.attribute("radius").value());
            return c;
        }
        if (const auto cylinder = geometry.child("cylinder")) {
            c.shape = Coll::Shape::Capsule;// cylinder -> capsule (articulation links don't take raw cylinders)
            c.radius = utils::parseFloat(cylinder.attribute("radius").value());
            c.halfHeight = utils::parseFloat(cylinder.attribute("length").value()) * 0.5f;
            // A URDF cylinder is Z-aligned; a threepp Capsule is Y-aligned. Fold the
            // difference into the origin, so `origin` means "where to put a threepp
            // capsule" and every consumer gets a collider that agrees with the spec —
            // and with the VISUAL cylinder, which applyGeometry already rotates the
            // same way. Without this a spec-conforming leg or strut collided across
            // its own axis: ninety degrees wrong, in the right place, silently.
            Matrix4 yToZ;
            yToZ.makeRotationX(math::PI / 2);
            c.origin.multiply(yToZ);
            return c;
        }
        if (const auto mesh = geometry.child("mesh")) {// trimesh -> ONE convex hull
            const auto fileName = getModelPath(packages, path.parent_path(), mesh.attribute("filename").value());
            if (!fileName.empty() && loader) {
                if (auto obj = loader->load(fileName)) {
                    if (const auto scale = mesh.attribute("scale")) {
                        obj->scale.copy(parseTupleString(scale.value()));
                    }
                    obj->updateMatrixWorld(true);
                    // Gather every sub-mesh's vertices in the collision frame,
                    // scale baked in via each sub-mesh's world matrix. The
                    // articulation builder cooks the union into one convex hull —
                    // a far better collider than the bounding box this used to
                    // produce (a chair leg no longer collides as a solid slab).
                    std::vector<float> points;
                    obj->traverseType<Mesh>([&](Mesh& m) {
                        const auto* pos = m.geometry() ? m.geometry()->getAttribute<float>("position") : nullptr;
                        if (!pos) return;
                        m.updateMatrixWorld();
                        for (unsigned i = 0; i < pos->count(); ++i) {
                            Vector3 v(pos->getX(i), pos->getY(i), pos->getZ(i));
                            v.applyMatrix4(*m.matrixWorld);
                            points.push_back(v.x);
                            points.push_back(v.y);
                            points.push_back(v.z);
                        }
                    });
                    if (points.size() >= 12) {// >= 4 vertices
                        c.shape = Coll::Shape::Hull;
                        c.hullPoints = std::move(points);
                        return c;
                    }
                    // Too few vertices to cook: fall back to the bounding box so
                    // the link still becomes a body.
                    Box3 bb;
                    bb.setFromObject(*obj);
                    if (!bb.isEmpty()) {
                        Vector3 size, center;
                        bb.getSize(size);
                        bb.getCenter(center);
                        c.shape = Coll::Shape::Box;
                        c.halfExtents.copy(size).multiplyScalar(0.5f);
                        Matrix4 shift;
                        shift.setPosition(center);
                        c.origin.multiply(shift);// collider centred on the bbox centre
                        return c;
                    }
                }
            }
        }
        return c;
    }

    std::optional<float> parseInertialMass(const pugi::xml_node& linkNode) {
        const auto inertial = linkNode.child("inertial");
        if (!inertial) return {};
        const auto mass = inertial.child("mass");
        if (!mass || !mass.attribute("value")) return {};
        const float m = utils::parseFloat(mass.attribute("value").value());
        if (m <= 0.f) return {};
        return m;
    }

    // The link's first <visual> as a link-frame subtree (origin-positioned Group + geometry), for the
    // articulation builder to parent under the collider so the robot renders as its real meshes.
    std::shared_ptr<Object3D> parseFirstVisual(xacro::PackageResolver* packages, const pugi::xml_node& linkNode,
                                               const std::filesystem::path& path, Loader<Group>* loader) {
        const auto visual = linkNode.child("visual");
        if (!visual) return nullptr;
        auto group = Group::create();
        if (const auto origin = visual.child("origin")) {
            group->position.copy(parseTupleString(origin.attribute("xyz").value()));
            applyRotation(group, parseTupleString(origin.attribute("rpy").value()));
        }
        if (auto obj = parseGeometryNode(packages, path, loader, visual.child("geometry"))) {
            if (const auto material = visual.child("material"); material && material.child("color")) {
                const auto mtl = getMaterial(material);
                obj->traverseType<Mesh>([mtl](Mesh& mesh) { mesh.setMaterial(mtl); });
            }
            group->add(obj);
        }
        return group;
    }

    // A link's <visual> and <collision> both collapse to one in the articulation description
    // (Link holds a single Collision and a single visual subtree, and Articulation::addLink
    // attaches one shape). Keeping the first in document order is the model, not a failure —
    // but silently is expensive: the FR3's fingers are four collision boxes each and the first
    // is the proximal mount block, not the pad at the grasp plane, so a grasp built on it fails
    // with nothing to explain why. The renderer's Robot path (loadFromXml) draws every one of
    // them, so the collider wireframe and the physics genuinely disagree. Say what was ignored.
    void warnExtraElements(std::vector<std::string>& diagnostics, const pugi::xml_node& linkNode,
                           const std::string& name, const char* tag, const char* model) {

        const auto nodes = linkNode.children(tag);
        const auto count = std::distance(nodes.begin(), nodes.end());
        if (count < 2) return;

        const std::string message =
                "[URDFLoader] warning: link '" + name + "' has " + std::to_string(count) + " <" + tag +
                "> elements; the articulation uses the first and ignores " + std::to_string(count - 1) +
                " (" + model + ")";
        std::cerr << message << std::endl;
        diagnostics.push_back(message);
    }

    // Walk the URDF link/joint tree (root first) into a flat articulation description.
    URDFArticulationDesc buildArticulationDesc(xacro::PackageResolver* packages, const pugi::xml_node& robotNode,
                                               const std::filesystem::path& path, Loader<Group>* loader,
                                               bool loadVisuals, std::vector<std::string>& diagnostics) {
        URDFArticulationDesc desc;
        std::map<std::string, pugi::xml_node> linkNodes;
        for (const auto link : robotNode.children("link"))
            linkNodes[link.attribute("name").value()] = link;

        std::map<std::string, pugi::xml_node> inboundJoint;        // child link -> its inbound joint
        std::map<std::string, std::vector<std::string>> childLinks;// parent link -> child links
        std::set<std::string> hasParent;
        for (const auto joint : robotNode.children("joint")) {
            const std::string parent = joint.child("parent").attribute("link").value();
            const std::string child = joint.child("child").attribute("link").value();
            inboundJoint[child] = joint;
            childLinks[parent].push_back(child);
            hasParent.insert(child);
        }
        std::string root;// the link no joint names as a child
        for (const auto& [name, node] : linkNodes)
            if (!hasParent.count(name)) {
                root = name;
                break;
            }
        if (root.empty()) return desc;

        std::list<std::pair<std::string, int>> queue;// (link name, parent index in desc.links)
        queue.emplace_back(root, -1);
        while (!queue.empty()) {
            const auto [name, parentIdx] = queue.front();
            queue.pop_front();
            const auto& linkNode = linkNodes[name];
            URDFArticulationDesc::Link L;
            L.name = name;
            L.parent = parentIdx;
            if (const auto jit = inboundJoint.find(name); jit != inboundJoint.end()) {
                const auto& joint = jit->second;
                L.jointName = joint.attribute("name").value();
                L.jointType = getType(joint.attribute("type").value());
                L.jointAxis = parseTupleString(joint.child("axis").attribute("xyz").as_string("1 0 0")).normalize();
                L.range = getRange(joint);
                L.jointOrigin = originMatrix(joint);
            }
            warnExtraElements(diagnostics, linkNode, name, "collision", "one collider per link");
            L.collision = parseCollisionShape(packages, linkNode.child("collision"), path, loader);
            if (const auto m = parseInertialMass(linkNode)) {
                L.hasMass = true;
                L.mass = *m;
            }
            // Only when visuals are actually being built: with loadVisuals=false the caller
            // asked for no visual subtree at all, so reporting the ones it "ignored" is noise.
            if (loadVisuals) {
                warnExtraElements(diagnostics, linkNode, name, "visual", "one visual subtree per link");
                L.visual = parseFirstVisual(packages, linkNode, path, loader);
            }
            const int myIdx = static_cast<int>(desc.links.size());
            desc.links.push_back(std::move(L));
            for (const auto& child : childLinks[name])
                queue.emplace_back(child, myIdx);
        }
        return desc;
    }

}// namespace


struct URDFLoader::Impl {

    std::shared_ptr<Loader<Group>> loader;
    std::map<std::string, std::string> xacroArgs;
    xacro::PackageResolver packages;

    // Everything the last call had to say, and the errors on their own. Two lists
    // rather than one filtered on read, because a caller that wants the reason a load
    // failed must not have to guess which lines are the reason.
    std::vector<std::string> diagnostics;
    std::vector<std::string> errors;

    Impl() {
        auto ml = std::make_shared<ModelLoader>();
        ml->setIgnoreUpDirection(true);
        loader = std::move(ml);
    }

    void beginCall() {
        diagnostics.clear();
        errors.clear();
    }

    void fail(const std::string& message) {
        diagnostics.push_back(message);
        errors.push_back(message);
    }

    static bool needsProcessing(const pugi::xml_document& doc, const std::filesystem::path& path) {
        return utils::toLower(path.extension().string()) == ".xacro" || xacro::needsProcessing(doc);
    }

    // `document` is the file the XML came from — includes, $(dirname) and load_yaml all
    // resolve against its directory, so a string gets the base directory it was parsed with.
    bool expandXacro(const pugi::xml_document& doc, const std::filesystem::path& document,
                     pugi::xml_document& out, std::string_view source = {}) {

        xacro::ExpandInputs inputs;
        for (const auto& [name, value] : xacroArgs) inputs.args[name] = xacro::Value(value);
        inputs.packages = &packages;
        inputs.document = document;
        inputs.source = source;
        inputs.argsAsProperties = true;

        xacro::Diagnostics diags;
        const bool ok = xacro::expand(doc, out, inputs, diags);

        for (const auto& warning : diags.warnings()) {
            std::cerr << "[xacro] warning: " << warning << std::endl;
            diagnostics.push_back("[xacro] warning: " + warning);
        }
        if (!ok) {
            for (const auto& error : diags.errors()) {
                std::cerr << "[xacro] " << error << std::endl;
                fail(error);
            }
            // expand() only returns false with something in diags.errors(), but a caller
            // asking why must never be told "no reason".
            if (diags.errors().empty()) fail("xacro expansion of '" + document.string() + "' failed");
        }
        return ok;
    }

    std::shared_ptr<Robot> load(const std::filesystem::path& path) {
        beginCall();

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(path.string().c_str());

        if (!result) {
            fail("cannot read '" + path.string() + "': " + result.description());
            return nullptr;
        }

        if (needsProcessing(doc, path)) {
            pugi::xml_document processed;
            if (!expandXacro(doc, path, processed)) return nullptr;
            return loadFromXml(processed, path);
        }

        return loadFromXml(doc, path);
    }

    std::shared_ptr<Robot> parse(const std::filesystem::path& baseDir, const std::string& urdf) {
        beginCall();

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_string(urdf.c_str());

        if (!result) {
            fail(std::string("cannot parse the document: ") + result.description());
            return nullptr;
        }

        if (xacro::needsProcessing(doc)) {
            pugi::xml_document processed;
            if (!expandXacro(doc, baseDir / "(string)", processed, urdf)) return nullptr;
            return loadFromXml(processed, baseDir);
        }

        return loadFromXml(doc, baseDir);
    }

    URDFArticulationDesc parseArticulation(const std::filesystem::path& path, bool loadVisuals = true) {
        beginCall();

        pugi::xml_document doc;
        if (const auto result = doc.load_file(path.string().c_str()); !result) {
            fail("cannot read '" + path.string() + "': " + result.description());
            return {};
        }

        CachingLoader cachingLoader(loader.get());
        if (needsProcessing(doc, path)) {
            pugi::xml_document processed;
            if (!expandXacro(doc, path, processed)) return {};
            const auto robotNode = processed.child("robot");
            if (!robotNode) {
                fail("'" + path.string() + "' has no <robot> element");
                return {};
            }
            return buildArticulationDesc(&packages, robotNode, path, &cachingLoader, loadVisuals, diagnostics);
        }
        const auto robotNode = doc.child("robot");
        if (!robotNode) {
            fail("'" + path.string() + "' has no <robot> element");
            return {};
        }
        return buildArticulationDesc(&packages, robotNode, path, &cachingLoader, loadVisuals, diagnostics);
    }

    std::shared_ptr<Robot> loadFromXml(const pugi::xml_document& doc, const std::filesystem::path& path) {

        const auto root = doc.child("robot");
        if (!root) {
            fail("'" + path.string() + "' has no <robot> element");
            return nullptr;
        }

        CachingLoader cachingLoader(loader.get());

        auto robot = std::make_shared<Robot>();
        robot->name = root.attribute("name").as_string("robot");

        for (const auto link : root.children("link")) {

            const auto linkObject = std::make_shared<Object3D>();
            linkObject->name = link.attribute("name").value();

            for (const auto visual : link.children("visual")) {

                auto group = Group::create();
                if (const auto origin = visual.child("origin")) {
                    group->position.copy(parseTupleString(origin.attribute("xyz").value()));
                    applyRotation(group, parseTupleString(origin.attribute("rpy").value()));
                }

                if (auto visualObject = parseGeometryNode(&packages, path, &cachingLoader, visual.child("geometry"))) {
                    group->add(visualObject);
                }

                // Only override the mesh's own materials when the URDF supplies an
                // inline color; a bare named reference keeps the loaded appearance.
                if (const auto material = visual.child("material"); material && material.child("color")) {

                    const auto mtl = getMaterial(material);

                    group->traverseType<Mesh>([mtl](Mesh& mesh) {
                        mesh.setMaterial(mtl);
                    });
                }

                linkObject->add(group);
            }

            for (const auto collider : link.children("collision")) {

                auto group = Group::create();
                group->userData["collider"] = true;
                if (const auto origin = collider.child("origin")) {
                    group->position.copy(parseTupleString(origin.attribute("xyz").value()));
                    applyRotation(group, parseTupleString(origin.attribute("rpy").value()));
                }

                const auto material = MeshBasicMaterial::create();
                material->wireframe = true;
                material->color = Color::white;

                if (auto colliderObject = parseGeometryNode(&packages, path, &cachingLoader, collider.child("geometry"))) {
                    group->add(colliderObject);

                    colliderObject->traverseType<Mesh>([material](Mesh& mesh) {
                        mesh.setMaterial(material);
                    });
                }

                linkObject->add(group);
            }

            robot->addLink(linkObject);
        }

        for (const auto joint : root.children("joint")) {

            const auto jointObject = std::make_shared<Object3D>();
            jointObject->name = joint.attribute("name").value();

            if (const auto origin = joint.child("origin")) {
                jointObject->position.copy(parseTupleString(origin.attribute("xyz").value()));
                applyRotation(jointObject, parseTupleString(origin.attribute("rpy").value()));
            }

            robot->addJoint(jointObject, parseInfo(joint));
        }

        robot->finalize();

        return robot;
    }
};

URDFLoader::URDFLoader()
    : pimpl_(std::make_unique<Impl>()) {}

URDFLoader& URDFLoader::setGeometryLoader(std::shared_ptr<Loader<Group>> loader) {
    pimpl_->loader = std::move(loader);

    return *this;
}

URDFLoader& URDFLoader::setArgs(std::map<std::string, std::string> args) {
    pimpl_->xacroArgs = std::move(args);
    return *this;
}

URDFLoader& URDFLoader::addPackagePath(const std::string& package, const std::filesystem::path& dir) {
    pimpl_->packages.addPackagePath(package, dir);
    return *this;
}

std::shared_ptr<Robot> URDFLoader::load(const std::filesystem::path& path) {

    return pimpl_->load(path);
}

std::shared_ptr<Robot> URDFLoader::parse(const std::filesystem::path& baseDir, const std::string& xml) {

    return pimpl_->parse(baseDir, xml);
}

URDFArticulationDesc URDFLoader::parseArticulation(const std::filesystem::path& path, bool loadVisuals) {

    return pimpl_->parseArticulation(path, loadVisuals);
}

const std::vector<std::string>& URDFLoader::diagnostics() const {

    return pimpl_->diagnostics;
}

std::string URDFLoader::lastError() const {

    std::string joined;
    for (const auto& error : pimpl_->errors) {
        if (!joined.empty()) joined += '\n';
        joined += error;
    }
    return joined;
}

URDFLoader::~URDFLoader() = default;

void threepp::scaleArticulationDesc(URDFArticulationDesc& desc, float scale) {

    if (!(scale > 0.f) || scale == 1.f) return;

    // Translation only. The rotation columns are direction cosines and the
    // matrices carry no scale of their own (originMatrix composes with unit
    // scale), so the position column is the whole length content of a frame.
    const auto scaleTranslation = [scale](Matrix4& m) {
        m.elements[12] *= scale;
        m.elements[13] *= scale;
        m.elements[14] *= scale;
    };

    for (auto& link : desc.links) {

        scaleTranslation(link.jointOrigin);

        auto& collision = link.collision;
        collision.halfExtents.multiplyScalar(scale);
        collision.radius *= scale;
        collision.halfHeight *= scale;
        scaleTranslation(collision.origin);
        // Already in the collision frame with the <mesh scale> baked in, so the
        // points are plain lengths.
        for (auto& component : collision.hullPoints) component *= scale;

        // A prismatic limit is a distance; a revolute one is an angle.
        if (link.jointType == Robot::JointType::Prismatic && link.range) {
            link.range->min *= scale;
            link.range->max *= scale;
        }

        // The visual subtree is geometry in the URDF's units hanging off a
        // link-frame node. Scaling that node (and the origin offset that places
        // it) rescales the meshes without touching the geometry itself, which
        // CachingLoader shares between links — mutating vertices here would
        // resize every other link that references the same file.
        if (link.visual) {
            link.visual->position.multiplyScalar(scale);
            link.visual->scale.multiplyScalar(scale);
        }
    }
}
