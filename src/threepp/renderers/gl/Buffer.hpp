
#ifndef THREEPP_BUFFER_HPP
#define THREEPP_BUFFER_HPP

namespace threepp::gl {

    struct Buffer {
        unsigned int buffer{};
        int type{};
        int bytesPerElement{};
        unsigned int version{};
    };

}// namespace threepp::gl

#endif//THREEPP_BUFFER_HPP
