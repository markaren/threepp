// A deliberately small YAML reader: block mappings and sequences, flow mappings and
// sequences, typed scalars and comments. Anything beyond that (anchors, aliases, tags,
// block scalars, multiple documents) is reported as unsupported rather than guessed at.

#ifndef THREEPP_XACRO_YAMLLITE_HPP
#define THREEPP_XACRO_YAMLLITE_HPP

#include "threepp/loaders/xacro/Value.hpp"

#include <filesystem>
#include <string_view>

namespace threepp::xacro {

    [[nodiscard]] Value parseYaml(std::string_view text, const std::filesystem::path& document = {});

    [[nodiscard]] Value loadYamlFile(const std::filesystem::path& path);

}// namespace threepp::xacro

#endif
