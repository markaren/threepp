
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
#include "threepp/utils/StringUtils.hpp"

#include "pugixml.hpp"
#include "threepp/loaders/ModelLoader.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <set>


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
        const auto color = node.child("color");
        const auto diffuse = color.attribute("rgba").value();
        const auto diffuseArray = utils::split(diffuse, ' ');

        const auto mtl = MeshStandardMaterial::create();
        mtl->color.setRGB(
                utils::parseFloat(diffuseArray[0]),
                utils::parseFloat(diffuseArray[1]),
                utils::parseFloat(diffuseArray[2]));

        const float alpha = utils::parseFloat(diffuseArray[3]);
        if (alpha < 1) {
            mtl->transparent = true;
            mtl->opacity = alpha;
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

    std::filesystem::path findPackageRoot(const std::filesystem::path& start) {
        for (auto path = start; path != path.parent_path(); path = path.parent_path()) {
            if (exists(path / "package.xml")) {
                return path;
            }

        }
        return {};
    }

    std::filesystem::path getModelPath(const std::filesystem::path& basePath, std::string_view fileName) {
        if (!fileName.starts_with("package://")) {
            return basePath / fileName;
        }

        const auto relative = std::filesystem::path(fileName.substr(10));

        if (auto p = basePath / relative; exists(p)) return p;

        const auto pkgRoot = findPackageRoot(basePath);
        if (pkgRoot.empty()) return {};

        if (auto p = pkgRoot / relative; exists(p)) return p;
        if (auto p = pkgRoot.parent_path() / relative; exists(p)) return p;

        return {};
    }

    std::shared_ptr<Object3D> parseGeometryNode(const std::filesystem::path& path, Loader<Group>* loader, const pugi::xml_node& geometry) {
        if (const auto mesh = geometry.child("mesh")) {
            const auto fileName = getModelPath(path.parent_path(), mesh.attribute("filename").value());
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
    URDFArticulationDesc::Collision parseCollisionShape(const pugi::xml_node& collisionNode,
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
            return c;
        }
        if (const auto mesh = geometry.child("mesh")) {// trimesh -> bounding box
            const auto fileName = getModelPath(path.parent_path(), mesh.attribute("filename").value());
            if (!fileName.empty() && loader) {
                if (auto obj = loader->load(fileName)) {
                    if (const auto scale = mesh.attribute("scale")) {
                        obj->scale.copy(parseTupleString(scale.value()));
                    }
                    obj->updateMatrixWorld(true);
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
    std::shared_ptr<Object3D> parseFirstVisual(const pugi::xml_node& linkNode,
                                               const std::filesystem::path& path, Loader<Group>* loader) {
        const auto visual = linkNode.child("visual");
        if (!visual) return nullptr;
        auto group = Group::create();
        if (const auto origin = visual.child("origin")) {
            group->position.copy(parseTupleString(origin.attribute("xyz").value()));
            applyRotation(group, parseTupleString(origin.attribute("rpy").value()));
        }
        if (auto obj = parseGeometryNode(path, loader, visual.child("geometry"))) {
            if (const auto material = visual.child("material"); material && material.child("color")) {
                const auto mtl = getMaterial(material);
                obj->traverseType<Mesh>([mtl](Mesh& mesh) { mesh.setMaterial(mtl); });
            }
            group->add(obj);
        }
        return group;
    }

    // Walk the URDF link/joint tree (root first) into a flat articulation description.
    URDFArticulationDesc buildArticulationDesc(const pugi::xml_node& robotNode,
                                               const std::filesystem::path& path, Loader<Group>* loader,
                                               bool loadVisuals) {
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
            L.collision = parseCollisionShape(linkNode.child("collision"), path, loader);
            if (const auto m = parseInertialMass(linkNode)) {
                L.hasMass = true;
                L.mass = *m;
            }
            if (loadVisuals) L.visual = parseFirstVisual(linkNode, path, loader);
            const int myIdx = static_cast<int>(desc.links.size());
            desc.links.push_back(std::move(L));
            for (const auto& child : childLinks[name])
                queue.emplace_back(child, myIdx);
        }
        return desc;
    }

}// namespace

#include "XacroProcessor.hpp"


struct URDFLoader::Impl {

    std::shared_ptr<Loader<Group>> loader;
    std::map<std::string, std::string> xacroArgs;

    Impl() {
        auto ml = std::make_shared<ModelLoader>();
        ml->setIgnoreUpDirection(true);
        loader = std::move(ml);
    }

    std::shared_ptr<Robot> load(const std::filesystem::path& path) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(path.string().c_str());

        if (!result) return nullptr;

        const std::string ext = utils::toLower(path.extension().string());
        if (ext == ".xacro" || xacro::Processor::needsProcessing(doc)) {
            xacro::Processor proc(path.parent_path(), xacroArgs);
            pugi::xml_document processed;
            proc.process(doc, processed);
            return loadFromXml(processed, path);
        }

        return loadFromXml(doc, path);
    }

    std::shared_ptr<Robot> parse(const std::filesystem::path& baseDir, const std::string& urdf) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_string(urdf.c_str());

        if (!result) return nullptr;

        if (xacro::Processor::needsProcessing(doc)) {
            xacro::Processor proc(baseDir, xacroArgs);
            pugi::xml_document processed;
            proc.process(doc, processed);
            return loadFromXml(processed, baseDir);
        }

        return loadFromXml(doc, baseDir);
    }

    URDFArticulationDesc parseArticulation(const std::filesystem::path& path, bool loadVisuals = true) {
        pugi::xml_document doc;
        if (!doc.load_file(path.string().c_str())) return {};

        const std::string ext = utils::toLower(path.extension().string());
        if (ext == ".xacro" || xacro::Processor::needsProcessing(doc)) {
            xacro::Processor proc(path.parent_path(), xacroArgs);
            pugi::xml_document processed;
            proc.process(doc, processed);
            const auto robotNode = processed.child("robot");
            return robotNode ? buildArticulationDesc(robotNode, path, loader.get(), loadVisuals) : URDFArticulationDesc{};
        }
        const auto robotNode = doc.child("robot");
        return robotNode ? buildArticulationDesc(robotNode, path, loader.get(), loadVisuals) : URDFArticulationDesc{};
    }

    std::shared_ptr<Robot> loadFromXml(const pugi::xml_document& doc, const std::filesystem::path& path) {

        const auto root = doc.child("robot");
        if (!root) return nullptr;

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

                if (auto visualObject = parseGeometryNode(path, loader.get(), visual.child("geometry"))) {
                    group->add(visualObject);
                }

                if (const auto material = visual.child("material")) {

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

                if (auto colliderObject = parseGeometryNode(path, loader.get(), collider.child("geometry"))) {
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

std::shared_ptr<Robot> URDFLoader::load(const std::filesystem::path& path) {

    return pimpl_->load(path);
}

std::shared_ptr<Robot> URDFLoader::parse(const std::filesystem::path& baseDir, const std::string& xml) {

    return pimpl_->parse(baseDir, xml);
}

URDFArticulationDesc URDFLoader::parseArticulation(const std::filesystem::path& path, bool loadVisuals) {

    return pimpl_->parseArticulation(path, loadVisuals);
}

URDFLoader::~URDFLoader() = default;
