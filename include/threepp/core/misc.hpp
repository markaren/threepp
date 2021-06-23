
#ifndef THREEPP_MISC_HPP
#define THREEPP_MISC_HPP

namespace threepp {

    struct GeometryGroup {

        const int start;
        const int count;
        const int materialIndex;

    };

    struct UpdateRange {

        const int offset;
        int count;

    };

    struct DrawRange {

        int start;
        int count;

    };

}

#endif//THREEPP_MISC_HPP
