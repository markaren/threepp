
#ifndef THREEPP_STLLOADER_HPP
#define THREEPP_STLLOADER_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <filesystem>

namespace threepp {

    class STLLoader {

    public:
        [[nodiscard]] std::shared_ptr<BufferGeometry> load(const std::filesystem::path& path) const;
    };

}// namespace threepp

#endif//THREEPP_STLLOADER_HPP
