
#ifndef THREEPP_KINECOMPONENT_HPP
#define THREEPP_KINECOMPONENT_HPP

#include <memory>

#include "threepp/math/Matrix4.hpp"

namespace kine {

    class KineComponent {

    public:
        [[nodiscard]] bool isFirst() const {
            return previous_ == nullptr;
        }

        [[nodiscard]] bool isLast() const {
            return next_ == nullptr;
        }

        void setPrevious(KineComponent* c) {
            previous_ = c;
        }

        void setNext(KineComponent* c) {
            next_ = c;
        }

        KineComponent* next() {
            return next_;
        }

        [[nodiscard]] virtual threepp::Matrix4 getTransformation() const = 0;

    private:
        KineComponent* previous_ = nullptr;
        KineComponent* next_ = nullptr;

    };

}

#endif//THREEPP_KINECOMPONENT_HPP
