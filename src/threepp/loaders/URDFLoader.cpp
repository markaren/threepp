
#include "threepp/loaders/URDFLoader.hpp"

#include "pugixml.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/StringUtils.hpp"

using namespace threepp;

namespace {

    Vector3 getXYZ(const pugi::xml_node& node) {
        if (!node.child("origin")) return {};
        const auto xyz = utils::split(node.child("origin").attribute("xyz").value(), ' ');
        return Vector3(
                utils::parseFloat(xyz[0]),
                utils::parseFloat(xyz[2]),
                -utils::parseFloat(xyz[1]));
    }

    Vector3 getAxis(const std::string& axis) {
        if (axis.empty()) return {};
        const auto xyz = utils::split(axis, ' ');
        return Vector3(
                       utils::parseFloat(xyz[0]),
                       utils::parseFloat(xyz[2]),
                       utils::parseFloat(xyz[1]))
                .normalize();
    }

    Vector3 getRPY(const pugi::xml_node& node) {
        if (!node.child("origin")) return {};
        const auto xyz = utils::split(node.child("origin").attribute("rpy").value(), ' ');
        return Vector3(
                utils::parseFloat(xyz[0]),
                utils::parseFloat(xyz[1]),
                utils::parseFloat(xyz[2]));
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

        float alpha = utils::parseFloat(diffuseArray[3]);
        if (alpha < 1) {
            mtl->transparent = true;
            mtl->opacity = alpha;
        }

        return mtl;
    }

    JointType getType(const std::string& type) {
        if (type == "revolute" || type == "continuous") {
            return JointType::Revolute;
        }
        if (type == "prismatic") {
            return JointType::Prismatic;
        }
        return JointType::Fixed;
    }

    std::optional<JointRange> getRange(const pugi::xml_node& node) {
        const auto limit = node.child("limit");
        if (!limit || !limit.attribute("lower") || !limit.attribute("upper")) return {};
        const auto min =  utils::parseFloat(limit.attribute("lower").value());
        const auto max =  utils::parseFloat(limit.attribute("upper").value());
        return JointRange{
                .min = min,
                .max = max};
    }

    JointInfo parseInfo(const pugi::xml_node& node) {
        return {
                .axis = getAxis(node.child("axis").attribute("xyz").value()),
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
                if (exists(packagePath / "package.xml")) {
                    packageFound = true;
                    break;
                }
                packagePath = packagePath.parent_path();
            }
            if (!packageFound) {
                return "";
            }

            return packagePath.parent_path().string() + "/" + fileName;
        }

        return basePath / fileName;
    }

}// namespace

struct URDFLoader::Impl {

    std::shared_ptr<Robot> load(Loader<Group>& loader, const std::filesystem::path& path) {

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

            if (const auto visual = link.child("visual")) {

                auto group = Group::create();
                group->position.copy(getXYZ(visual));
                applyRotation(group, getRPY(visual));

                for (const auto geometry : visual.children("geometry")) {

                    const auto mesh = geometry.child("mesh");
                    const auto fileName = getModelPath(path.parent_path(), mesh.attribute("filename").value());

                    if (auto visualObject = loader.load(fileName)) {

                        visualObject->traverseType<Mesh>([fileName](const Mesh& mesh) {
                            if (fileName.extension().string() == ".stl" || fileName.extension().string() == ".STL") {
                                mesh.geometry()->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));
                            }
                        });

                        group->add(visualObject);
                    }
                }

                if (const auto material = visual.child("material")) {

                    const auto mtl = getMaterial(material);

                    group->traverseType<Mesh>([mtl](Mesh& mesh) {
                        mesh.setMaterial(mtl);
                    });
                }

                linkObject->add(group);
            }

            if (const auto collider = link.child("collision")) {

                auto group = Group::create();
                group->userData["collider"] = true;
                group->position.copy(getXYZ(collider));
                applyRotation(group, getRPY(collider));

                const auto material = MeshBasicMaterial::create();
                material->wireframe = true;

                for (const auto geometry : collider.children("geometry")) {

                    const auto mesh = geometry.child("mesh");
                    const auto fileName = getModelPath(path.parent_path(), mesh.attribute("filename").value());

                    if (auto colliderObject = loader.load(fileName)) {

                        colliderObject->traverseType<Mesh>([fileName, material](Mesh& mesh) {
                            mesh.setMaterial(material);
                            if (fileName.extension().string() == ".stl" || fileName.extension().string() == ".STL") {
                                mesh.geometry()->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2));
                            }
                        });

                        group->add(colliderObject);
                    }
                }

                linkObject->add(group);
            }

            robot->addLink(linkObject);
        }

        for (const auto joint : root.children("joint")) {

            const auto jointObject = std::make_shared<Object3D>();
            jointObject->name = joint.attribute("name").value();

            jointObject->position.copy(getXYZ(joint));
            applyRotation(jointObject, getRPY(joint));

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
