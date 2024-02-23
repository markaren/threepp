
#ifndef THREEPP_KINECOMPONENT_HPP
#define THREEPP_KINECOMPONENT_HPP

#include <memory>

#include "threepp/math/Matrix4.hpp"

namespace kine {

    class KineComponent {

    public:
        [[nodiscard]] virtual threepp::Matrix4 getTransformation() const = 0;

        virtual ~KineComponent() = default;
    };

}// namespace kine

#endif//THREEPP_KINECOMPONENT_HPP
