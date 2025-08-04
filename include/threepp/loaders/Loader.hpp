
#ifndef THREEPP_LOADER_HPP
#define THREEPP_LOADER_HPP

#include <filesystem>
#include <memory>

namespace threepp {

    template<class T>
    class Loader {
    public:
        virtual std::shared_ptr<T> load(const std::filesystem::path& path) = 0;

        virtual ~Loader() = default;
    };

}// namespace threepp

#endif//THREEPP_LOADER_HPP
