
#ifndef THREEPP_LINK_HPP
#define THREEPP_LINK_HPP

#include "KineComponent.hpp"
#include "threepp/math/Vector3.hpp"


namespace kine {

    class KineLink: public KineComponent {

    public:
        KineLink(const threepp::Vector3& link)
            : transformation_(threepp::Matrix4().setPosition(link)) {}

        [[nodiscard]] threepp::Matrix4 getTransformation() const override {
            return transformation_;
        }

    private:
        threepp::Matrix4 transformation_;
    };

}// namespace kine

#endif//THREEPP_LINK_HPP
