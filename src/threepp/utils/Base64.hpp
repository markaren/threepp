
#ifndef THREEPP_BASE64_HPP
#define THREEPP_BASE64_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace threepp::utils {

    // Decodes standard base64 (RFC 4648). Stops at the first character outside
    // the alphabet, so a data-URI payload may be passed with or without padding.
    std::vector<uint8_t> base64Decode(const std::string& encoded);

    std::string base64Encode(const uint8_t* data, size_t size);

    // The payload of a data: URI ("data:[<mediatype>][;base64],<data>"),
    // base64-decoded: the only encoding the loaders meet. nullopt when `uri`
    // is not a data: URI or has no comma before its payload.
    std::optional<std::vector<uint8_t>> decodeDataUri(const std::string& uri);

    inline std::string base64Encode(const std::vector<uint8_t>& data) {

        return base64Encode(data.data(), data.size());
    }

}// namespace threepp::utils

#endif//THREEPP_BASE64_HPP
