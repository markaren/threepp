
#include "threepp/loaders/URDFLoader.hpp"

#include "pugixml.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/stringUtils.hpp"

using namespace threepp;

struct URDFLoader::Impl {

    std::shared_ptr<Object3D> load(const std::filesystem::path& path) {

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(path.string().c_str());

        if (!result) {

            return nullptr;
        }

        auto root = doc.child("robot");

        if (!root) {

            return nullptr;
        }

        auto loader = AssimpLoader();

        auto object = std::make_shared<Object3D>();
        object->name = root.attribute("name").value();

        for (auto link : root.children("link")) {

            auto linkObject = std::make_shared<Object3D>();
            linkObject->name = link.attribute("name").value();

            for (auto visual : link.children("visual")) {

                auto visualObject = std::make_shared<Object3D>();
                visualObject->name = visual.attribute("name").value();

                auto xyz = utils::split(visual.child("origin").attribute("xyz").value(), ' ');
                visualObject->position.set(
                        utils::parseFloat(xyz[0]),
                        utils::parseFloat(xyz[1]),
                        utils::parseFloat(xyz[2]));

                auto rpy = utils::split(visual.child("origin").attribute("rpy").value(), ' ');
                visualObject->rotation.set(
                        utils::parseFloat(rpy[0]),
                        utils::parseFloat(rpy[1]),
                        utils::parseFloat(rpy[2]));

                for (auto geometry : visual.children("geometry")) {

                    auto mesh = geometry.child("mesh");
                    auto fileName = std::string(mesh.attribute("filename").value());

                    //if filename startswith package://
                    if (fileName.find("package://") == 0) {
                        fileName = fileName.substr(10);
                        //find parent path with package.xml
                        bool packageFound = false;
                        auto packagePath = path.parent_path();
                        for (int i = 0; i < 10; ++i) {
                            if (!std::filesystem::exists(packagePath / "package.xml")) {
                                packagePath = packagePath.parent_path();
                                packageFound = true;
                            }
                        }
                        if (!packageFound) {
                            return nullptr;
                        }

                        fileName = packagePath.string() + "/" + fileName;
                    }

                    auto load = loader.load(fileName);
                    if (load) {
                        visualObject->add(load);
                    }

                }

                for (auto material : visual.children("material")) {

                    auto color = material.child("color");
                    auto diffuse = color.attribute("rgba").value();
                    auto diffuseArray = utils::split(diffuse, ' ');

                    auto mtl = MeshBasicMaterial::create();
                    mtl->color.setRGB(
                            utils::parseFloat(diffuseArray[0]),
                            utils::parseFloat(diffuseArray[1]),
                            utils::parseFloat(diffuseArray[2]));

//                    auto loader = AssimpLoader();
//                    auto meshObject = loader.load(fileName);
//                    if (meshObject) {
//                        meshObject->material = mtl;
//                        visualObject->add(meshObject);
//                    }
                }

                linkObject->add(visualObject);
            }

            object->add(linkObject);
        }

        return nullptr;
    }
};

URDFLoader::URDFLoader()
    : pimpl_(std::make_unique<Impl>()) {}

std::shared_ptr<Object3D> URDFLoader::load(const std::filesystem::path& path) {

    return pimpl_->load(path);
}

URDFLoader::~URDFLoader() = default;
