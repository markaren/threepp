
#ifndef THREEPP_GROUP_HPP
#define THREEPP_GROUP_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp {

    // This class is almost identical to an Object3D. Its purpose is to make working with groups of objects syntactically clearer.
    class Group: public Object3D {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<Group> create();

        ~Group() override = default;

    protected:
        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_GROUP_HPP
