
#include "threepp/loaders/URDFLoader.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "pugixml.hpp"

#include <iostream>

using namespace threepp;

namespace {

    Vector3 parseTupleString(const std::string& strValues) {
        if (strValues.empty()) return {};
        const auto values = utils::split(strValues, ' ');
        return {utils::parseFloat(values[0]),
                utils::parseFloat(values[1]),
                utils::parseFloat(values[2])};
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
                .range = getRange(node),
                .parent = node.child("parent").attribute("link").value(),
                .child = node.child("child").attribute("link").value()};
    }

    std::filesystem::path getModelPath(const std::filesystem::path& basePath, std::string fileName) {

        if (fileName.find("package://") != std::string::npos) {

            fileName = fileName.substr(10);

            //find parent path with package.xml
            bool packageFound = false;
            auto packagePath = basePath;
            for (int i = 0; i < 10; ++i) {
                if (exists(packagePath / "package.xml") || exists(packagePath / "source-information.json")) {
                    packageFound = true;
                    break;
                }
                packagePath = packagePath.parent_path();
            }
            if (!packageFound) {
                return "";
            }

            std::filesystem::path path = packagePath / fileName;
            if (exists(path)) {
                return path;
            }
            path = packagePath.parent_path() / fileName;
            if (exists(path)) {
                return path;
            }

            return "";
        }

        return basePath / fileName;
    }

    std::shared_ptr<Object3D> parseGeometryNode(const std::filesystem::path& path, Loader<Group>& loader, const pugi::xml_node& geometry) {
        if (const auto mesh = geometry.child("mesh")) {
            const auto fileName = getModelPath(path.parent_path(), mesh.attribute("filename").value());
            if (fileName.empty()) {
                return nullptr;
            }

            if (auto obj = loader.load(fileName)) {
                if (const auto scale = mesh.attribute("scale")) {
                    obj->scale.copy(parseTupleString(scale.value()));
                }

                if (utils::toLower(fileName.extension().string()) == ".dae") {
                    obj->traverseType<Mesh>([](const Mesh& mesh) {
                        mesh.geometry()->applyMatrix4(Matrix4().makeRotationX(math::PI / 2));
                    });
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

            return obj;
        }

        return nullptr;
    }

}// namespace

struct URDFLoader::Impl {

    static std::shared_ptr<Robot> load(Loader<Group>& loader, const std::filesystem::path& path) {

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(path.string().c_str());

        if (!result) return nullptr;

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

                if (auto visualObject = parseGeometryNode(path, loader, visual.child("geometry"))) {
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

                if (auto colliderObject = parseGeometryNode(path, loader, collider.child("geometry"))) {
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

std::shared_ptr<Robot> URDFLoader::load(Loader<Group>& loader, const std::filesystem::path& path) {

    return pimpl_->load(loader, path);
}

URDFLoader::~URDFLoader() = default;
