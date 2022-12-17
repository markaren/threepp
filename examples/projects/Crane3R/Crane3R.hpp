
#ifndef THREEPP_CRANE3R_HPP
#define THREEPP_CRANE3R_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/loaders/OBJLoader.hpp"

class Crane3R : public threepp::Group {

public:

    void setAngles(float j1, float j2, float j3) {
        p1_->rotation.z = threepp::math::DEG2RAD * j1;
        p2_->rotation.y = threepp::math::DEG2RAD * j2;
        p3_->rotation.y = threepp::math::DEG2RAD * j3;
    }

    static std::shared_ptr<Crane3R> create() {

        threepp::OBJLoader loader;
        auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
        auto part2 = loader.load("data/models/obj/Crane3R/7000/7000.obj");
        part2->position.set(0, 0, 4.2);
        auto part3 = loader.load("data/models/obj/Crane3R/5200/5200.obj");
        part3->position.set(7, 0, 0);

        part1->add(part2);
        part2->add(part3);

        part1->rotateX(-threepp::math::PI / 2);

        part1->name = "part1";
        part2->name = "part2";
        part3->name = "part3";

        return std::shared_ptr<Crane3R>(new Crane3R(part1));
    }

private:
    threepp::Object3D* p1_;
    threepp::Object3D* p2_;
    threepp::Object3D* p3_;

    explicit Crane3R(const std::shared_ptr<threepp::Group> &obj) {

        p1_ = obj->getObjectByName("part1");
        p2_ = obj->getObjectByName("part2");
        p3_ = obj->getObjectByName("part3");

        add(obj);
    }
};


#endif//THREEPP_CRANE3R_HPP
