
#include "threepp/loaders/URDFLoader.hpp"

#include "pugixml.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <iostream>

using namespace threepp;

namespace {

    Vector3 getXYZ(const pugi::xml_node& node) {
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
                utils::parseFloat(xyz[1]));
    }

    Euler getRPY(const pugi::xml_node& node) {
        const auto xyz = utils::split(node.child("origin").attribute("rpy").value(), ' ');
        return Euler(
                utils::parseFloat(xyz[0]),
                utils::parseFloat(xyz[1]),
                -utils::parseFloat(xyz[2]));
    }

    JointType getType(const std::string& type) {
        if (type == "revolute") {
            return JointType::Revolute;
        } else if (type == "continuous") {
            return JointType::Continuous;
        } else if (type == "prismatic") {
            return JointType::Prismatic;
        } else {
            return JointType::Fixed;
        }
    }

    std::optional<JointRange> getRange(const pugi::xml_node& node) {
        const auto limit = node.child("limit");
        if (!limit) return {};
        return JointRange{
                .min = utils::parseFloat(limit.attribute("lower").value()),
                .max = utils::parseFloat(limit.attribute("upper").value())
        };
    }

    JointInfo parseInfo(const pugi::xml_node& node) {
        return {
                .axis = getAxis(node.child("axis").attribute("xyz").value()),
                .type = getType(node.attribute("type").value()),
                .range = getRange(node),
                .parent = node.child("parent").attribute("link").value(),
                .child = node.child("child").attribute("link").value()
        };
    }


    std::filesystem::path getModelPath(const std::filesystem::path& basePath, std::string fileName) {

        if (fileName.find("package://") != std::string::npos) {

            fileName = fileName.substr(10);

            //find parent path with package.xml
            bool packageFound = false;
            auto packagePath = basePath.parent_path();
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

        return fileName;
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

            auto linkObject = std::make_shared<Object3D>();
            linkObject->name = link.attribute("name").value();

            for (const auto visual : link.children("visual")) {

                auto visualObject = std::make_shared<Object3D>();
                visualObject->name = visual.attribute("name").value();

                visualObject->position.copy(getXYZ(visual));
                visualObject->rotation.copy(getRPY(visual));

                for (const auto geometry : visual.children("geometry")) {

                    auto mesh = geometry.child("mesh");
                    auto fileName = getModelPath(path, mesh.attribute("filename").value());

                    if (auto load = loader.load(fileName)) {
                        visualObject->add(load);
                    }
                }

                // for (auto material : visual.children("material")) {
                //
                //     auto color = material.child("color");
                //     auto diffuse = color.attribute("rgba").value();
                //     auto diffuseArray = utils::split(diffuse, ' ');
                //
                //     auto mtl = MeshBasicMaterial::create();
                //     mtl->color.setRGB(
                //             utils::parseFloat(diffuseArray[0]),
                //             utils::parseFloat(diffuseArray[1]),
                //             utils::parseFloat(diffuseArray[2]));
                // }

                linkObject->add(visualObject);
            }

            robot->addLink(linkObject);
        }

        for (const auto joint : root.children("joint")) {

            auto jointObject = std::make_shared<Object3D>();
            jointObject->name = joint.attribute("name").value();

            jointObject->position.copy(getXYZ(joint));
            jointObject->rotation.copy(getRPY(joint));

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
