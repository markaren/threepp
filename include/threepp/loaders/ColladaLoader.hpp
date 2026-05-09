
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

        // When true, the file-level <up_axis> metadata is ignored and the
        // returned scene is left in its file-native orientation. Use this when
        // the mesh is being composed into an outer system (URDF/SDF/MJCF) that
        // owns the coordinate frame.
        ColladaLoader& setIgnoreUpDirection(bool ignore);

        std::shared_ptr<Group> load(const std::filesystem::path& path) override;

        ~ColladaLoader() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_COLLADALOADER_HPP