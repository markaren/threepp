
#ifndef THREEPP_GROUP_HPP
#define THREEPP_GROUP_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp {

    class Group: public Object3D {

    public:

        std::shared_ptr<Object3D> clone(bool recursive = false) override {
            auto clone = create();
            clone->copy(*this, recursive);

            return clone;
        }

        static std::shared_ptr<Group> create() {
            return std::shared_ptr<Group>(new Group());
        }

    protected:
        Group() = default;
    };

}// namespace threepp

#endif//THREEPP_GROUP_HPP
