
#include "threepp/loaders/URDFLoader.hpp"

#include "pugixml.hpp"

using namespace threepp;

struct URDFLoader::Impl {

    std::shared_ptr<Object3D> load(const std::filesystem::path& path) {



        return nullptr;
    }

};

URDFLoader::URDFLoader()
    : pimpl_(std::make_unique<Impl>()) {}

std::shared_ptr<Object3D> URDFLoader::load(const std::filesystem::path& path) {

    return pimpl_-> load(path);
}

URDFLoader::~URDFLoader() = default;
