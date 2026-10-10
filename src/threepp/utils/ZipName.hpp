// How a ZIP entry name is spelled, shared by the reader and the writer so
// that what one stores the other finds.

#ifndef THREEPP_ZIPNAME_HPP
#define THREEPP_ZIPNAME_HPP

#include <algorithm>
#include <string>

namespace threepp::utils {

    // Entry names are stored with forward slashes by every writer that follows
    // the spec, but not by every writer, and a caller should not have to care.
    // Applied to both sides of a lookup, so has("0_0\\meta.json"),
    // has("./0_0/meta.json") and has("0_0/meta.json") are one question.
    inline std::string normalizeZipName(std::string name) {

        std::replace(name.begin(), name.end(), '\\', '/');
        while (name.rfind("./", 0) == 0) name.erase(0, 2);
        return name;
    }

}// namespace threepp::utils

#endif//THREEPP_ZIPNAME_HPP
