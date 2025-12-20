
#ifndef THREEPP_OBJLOADER_HPP
#define THREEPP_OBJLOADER_HPP

#include <filesystem>
#include <memory>

#include "threepp/objects/Group.hpp"
#include "threepp/loaders/Loader.hpp"

namespace threepp {

    class MaterialCreator;

    class OBJLoader: public Loader<Group> {

    public:
        bool useCache = true;

        OBJLoader();

        std::shared_ptr<Group> load(const std::filesystem::path& path) override {
            return load(path, true);
        }

        std::shared_ptr<Group> load(const std::filesystem::path& path, bool tryLoadMtl);

        OBJLoader& setMaterials(const std::shared_ptr<MaterialCreator>& materials);

        void clearCache();

        ~OBJLoader() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_OBJLOADER_HPP
