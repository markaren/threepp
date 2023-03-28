
#ifndef THREEPP_AXESHELPER_HPP
#define THREEPP_AXESHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

#include <memory>

namespace threepp {

    class AxesHelper: public LineSegments {

    public:
        ~AxesHelper() override;

        static std::shared_ptr<AxesHelper> create(float size);

    protected:
        explicit AxesHelper(float size);
    };

}// namespace threepp

#endif//THREEPP_AXESHELPER_HPP
