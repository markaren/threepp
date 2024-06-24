
#include "threepp/loaders/URDFLoader.hpp"

#include "pugixml.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/stringUtils.hpp"

#include <iostream>
#include <numbers>

using namespace threepp;

namespace {

    Vector3 getXYZ(const pugi::xml_node& node) {
        const auto xyz = utils::split(node.child("origin").attribute("xyz").value(), ' ');
        return Vector3(
                utils::parseFloat(xyz[0]),
                utils::parseFloat(xyz[2]),
                -utils::parseFloat(xyz[1]));
    }

    Euler getRPY(const pugi::xml_node& node) {
        const auto xyz = utils::split(node.child("origin").attribute("rpy").value(), ' ');
        return Euler(
                utils::parseFloat(xyz[0]),
                utils::parseFloat(xyz[1]),
                -utils::parseFloat(xyz[2]));
    }

}// namespace

struct URDFLoader::Impl {

    std::shared_ptr<Group> load(Loader<Group>& loader, const std::filesystem::path& path) {

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(path.string().c_str());

        if (!result) return nullptr;

        const auto root = doc.child("robot");
        if (!root) return nullptr;


        auto object = std::make_shared<Group>();
        object->name = root.attribute("name").as_string("robot");

        std::vector<std::shared_ptr<Object3D>> links;
        for (auto link : root.children("link")) {

            auto linkObject = std::make_shared<Object3D>();
            linkObject->name = link.attribute("name").value();

            if (linkObject->name == "base_link") {
                object->add(linkObject);
            }

            for (auto visual : link.children("visual")) {

                auto visualObject = std::make_shared<Object3D>();
                visualObject->name = visual.attribute("name").value();

                visualObject->position.copy(getXYZ(visual));
                visualObject->rotation.copy(getRPY(visual));

                for (auto geometry : visual.children("geometry")) {

                    auto mesh = geometry.child("mesh");
                    auto fileName = std::string(mesh.attribute("filename").value());

                    if (fileName.find("package://") == 0) {
                        fileName = fileName.substr(10);
                        //find parent path with package.xml
                        bool packageFound = false;
                        auto packagePath = path.parent_path();
                        for (int i = 0; i < 10; ++i) {
                            if (exists(packagePath / "package.xml")) {
                                packageFound = true;
                                break;
                            }
                            packagePath = packagePath.parent_path();
                        }
                        if (!packageFound) {
                            return nullptr;
                        }

                        fileName = packagePath.parent_path().string() + "/" + fileName;
                    }

                    auto load = loader.load(fileName);
                    if (load) {
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

            links.emplace_back(linkObject);
        }

        for (auto joint : root.children("joint")) {

            auto jointObject = std::make_shared<Object3D>();
            jointObject->name = joint.attribute("name").value();

            jointObject->position.copy(getXYZ(joint));
            jointObject->rotation.copy(getRPY(joint));

            const auto parent = joint.child("parent").attribute("link").value();
            if (const auto parentIt = std::ranges::find_if(links, [&](auto& o) {
                    return o->name == parent;
                });
                parentIt != links.end()) {
                (*parentIt)->add(jointObject);
            }

            const auto child = joint.child("child").attribute("link").value();
            if (const auto childIt = std::ranges::find_if(links, [&](auto& o) {
                    return o->name == child;
                });
                childIt != links.end()) {
                jointObject->add(*childIt);
            }
        }

        for (const auto& l : links) {

            // if (!l->parent) object->add(l);
        }

        // auto o = object->getObjectByName("joint_1");
        // o->rotation.z = -std::numbers::phi / 2;

        // object->traverse([&](auto& o) {
        //     if (o.name.find("joint_") != std::string::npos) {
        //         std::cout << o.name << std::endl;
        //         // std::cout << o.parent->name << std::endl;
        //         std::cout << std::endl;
        //         o.rotation.y = std::numbers::phi/6;
        //     }
        // });

        return object;
    }
};

URDFLoader::URDFLoader()
    : pimpl_(std::make_unique<Impl>()) {}

std::shared_ptr<Group> URDFLoader::load(Loader<Group>& loader, const std::filesystem::path& path) {

    return pimpl_->load(loader, path);
}

URDFLoader::~URDFLoader() = default;
