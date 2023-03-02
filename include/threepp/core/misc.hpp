
#ifndef THREEPP_MISC_HPP
#define THREEPP_MISC_HPP

namespace threepp {

    struct GeometryGroup {

        int start{};
        int count{};
        unsigned int materialIndex{};
    };

    struct UpdateRange {

        int offset{};
        int count{};
    };

    struct DrawRange {

        int start{};
        int count{};
    };

}// namespace threepp

#endif//THREEPP_MISC_HPP
