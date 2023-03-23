
#ifndef THREEPP_OBJLOADER_HPP
#define THREEPP_OBJLOADER_HPP

#include <filesystem>
#include <memory>

#include "threepp/objects/Group.hpp"

namespace threepp {

    class MaterialCreator;

    class OBJLoader {

    public:
        bool useCache = true;

        OBJLoader();

        std::shared_ptr<Group> load(const std::filesystem::path& path, bool tryLoadMtl = true);

        OBJLoader& setMaterials(const std::shared_ptr<MaterialCreator>& materials);

        void clearCache();

        ~OBJLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_OBJLOADER_HPP
