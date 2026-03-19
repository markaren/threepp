
#ifndef THREEPP_COLLADALOADER_HPP
#define THREEPP_COLLADALOADER_HPP

#include "threepp/loaders/Loader.hpp"
#include "threepp/objects/Group.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class ColladaLoader: public Loader<Group> {

    public:
        ColladaLoader();

        std::shared_ptr<Group> load(const std::filesystem::path& path) override;

        ~ColladaLoader() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_COLLADALOADER_HPP