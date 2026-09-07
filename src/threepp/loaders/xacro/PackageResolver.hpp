// Maps a ROS package name to a directory, without requiring a ROS installation.

#ifndef THREEPP_XACRO_PACKAGERESOLVER_HPP
#define THREEPP_XACRO_PACKAGERESOLVER_HPP

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace threepp::xacro {

    class PackageResolver {

    public:
        void addPackagePath(const std::string& package, const std::filesystem::path& dir);

        // fromDoc is the document asking; its ancestors are searched before the environment.
        // tried, when given, collects the locations that were rejected (for error messages).
        [[nodiscard]] std::optional<std::filesystem::path> resolve(const std::string& package,
                                                                   const std::filesystem::path& fromDoc,
                                                                   std::vector<std::string>* tried = nullptr);

        void clearCache();

        // the <name> of the package manifest in dir, if there is one
        [[nodiscard]] static std::optional<std::string> manifestName(const std::filesystem::path& dir);

    private:
        std::map<std::string, std::filesystem::path> registry_;
        std::map<std::string, std::filesystem::path> cache_;
    };

}// namespace threepp::xacro

#endif
