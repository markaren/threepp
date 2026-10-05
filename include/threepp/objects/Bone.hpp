
#ifndef THREEPP_BONE_HPP
#define THREEPP_BONE_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp {

    class Bone: public Object3D {

    public:
        [[nodiscard]] const std::string& type() const override {

            static const std::string typeName = "Bone";
            return typeName;
        }

        static std::shared_ptr<Bone> create() {

            return std::make_shared<Bone>();
        }

    protected:
        std::shared_ptr<Object3D> createDefault() override {

            return create();
        }
    };

}// namespace threepp

#endif//THREEPP_BONE_HPP
