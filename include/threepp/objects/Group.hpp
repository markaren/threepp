
#ifndef THREEPP_GROUP_HPP
#define THREEPP_GROUP_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp {

    class Group: public Object3D {

    public:
        [[nodiscard]] std::string type() const override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override;

        static std::shared_ptr<Group> create();

    protected:
        Group();
    };

}// namespace threepp

#endif//THREEPP_GROUP_HPP
