
#include "Crane3R.hpp"

#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include "linalg.h"

using namespace threepp;

namespace {

    std::shared_ptr<Group> make_house_attachment(int i, float len, const Vector3 &p) {

        auto material = MeshPhongMaterial::create();
        material->color = threepp::Color::gray;

        float radius = 0.1f;
        auto geometry = threepp::CylinderGeometry::create(radius, radius, len);
        geometry->translate(0, len / 2, 0);
        geometry->rotateX(threepp::math::PI / 2);

        auto group = threepp::Group::create();
        group->name = "house" + std::to_string(i);
        group->position.copy(p);

        auto cyl1 = threepp::Mesh::create(geometry, material);
        auto cyl2 = threepp::Mesh::create(geometry, material);
        cyl1->position.x = 0.34f;
        cyl2->position.x = -0.34f;
        group->add(cyl1);
        group->add(cyl2);

        return group;
    }

    std::shared_ptr<Group> make_rod_attachment(int i, float len, const Vector3 &p) {

        auto material = threepp::MeshPhongMaterial::create();
        material->color = threepp::Color::grey;

        float radius = 0.075f;
        auto geometry = threepp::CylinderGeometry::create(radius, radius, len);
        geometry->translate(0, len / 2, 0);
        geometry->rotateX(threepp::math::PI / 2);

        auto group = threepp::Group::create();
        group->name = "rod" + std::to_string(i);
        group->position.copy(p);

        auto cyl1 = threepp::Mesh::create(geometry, material);
        auto cyl2 = threepp::Mesh::create(geometry, material);
        cyl1->position.x = 0.34f;
        cyl2->position.x = -0.34f;
        group->add(cyl1);
        group->add(cyl2);

        return group;
    }

    linalg::mat<float, 3, 3> jacobian(const std::array<float, 3> values) {
        constexpr float h = 0.0001f;//some low value

        linalg::mat<float, 3, 3> jacobian{};

        for (int i = 0; i < 3; ++i) {
            auto vals = values; //copy
            vals[i] += h;
            auto d1 = Crane3R::calculateEndEffectorPosition(vals);
            auto d2 = Crane3R::calculateEndEffectorPosition(values);

            jacobian.x[i] = (d1.x - d2.x) / h;
            jacobian.y[i] = (d1.y - d2.y) / h;
            jacobian.z[i] = (d1.z - d2.z) / h;
        }
        return jacobian;
    }

    linalg::mat<float, 3, 3> DLS(const linalg::mat<float, 3, 3> &j, float lambda) {

        linalg::mat<float, 3, 3> I = linalg::identity;

        auto jt = linalg::transpose(j);
        auto jjt = linalg::mul(j, jt);
        auto plus = linalg::cmul(I, lambda*lambda);

        return linalg::mul(jt, linalg::inverse(jjt + plus));
    }


}// namespace

Crane3R::Crane3R(const std::shared_ptr<threepp::Group> &obj) {

    parts_[0] = obj->getObjectByName("part1");
    parts_[1] = obj->getObjectByName("part2");
    parts_[2] = obj->getObjectByName("part3");

    cylinders_[0] = {
            obj->getObjectByName("house1"),
            obj->getObjectByName("rod1")};

    cylinders_[1] = {
            obj->getObjectByName("house2"),
            obj->getObjectByName("rod2")};


    add(obj);

    update();
}

std::shared_ptr<Crane3R> Crane3R::create() {

    threepp::OBJLoader loader;
    auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
    part1->rotateX(-threepp::math::PI / 2);
    auto part2 = loader.load("data/models/obj/Crane3R/7000/7000.obj");
    part2->position.set(0, 0, 4.2);
    auto part3 = loader.load("data/models/obj/Crane3R/5200/5200.obj");
    part3->position.set(7, 0, 0);

    part1->add(part2);
    part2->add(part3);

    part1->name = "part1";
    part2->name = "part2";
    part3->name = "part3";

    part1->add(make_house_attachment(1, 2.05f, {0.68, 0, 2.85}));
    part2->add(make_house_attachment(2, 2.05f, {4.2, 0, -0.78}));

    part2->add(make_rod_attachment(1, 2.05f, {2.695, 0, -0.85}));
    part3->add(make_rod_attachment(2, 2.05f, {0.98, 0, 0.2}));

    return std::shared_ptr<Crane3R>(new Crane3R(part1));
}


void Crane3R::setTargetAngles(const std::array<float, 3> &values, bool degrees) {

    if (degrees) {
        targetangles = {values[0] * math::DEG2RAD, values[1] * math::DEG2RAD, values[2] * math::DEG2RAD};
    } else {
        targetangles = values;
    }
}

void Crane3R::update() {

    parts_[0]->rotation.z = targetangles[0];
    parts_[1]->rotation.y = targetangles[1];
    parts_[2]->rotation.y = targetangles[2];

    Vector3 tmp;
    for (auto &cylinder : cylinders_) {

        auto house = cylinder.first;
        auto rod = cylinder.second;

        house->lookAt(rod->getWorldPosition(tmp));
        rod->lookAt(house->getWorldPosition(tmp));
    }
}

std::array<float, 3> Crane3R::getAngles(bool degrees) const {

    std::array<float, 3> angles{parts_[0]->rotation.z(), parts_[1]->rotation.y(), parts_[2]->rotation.y()};
    if (degrees) {
        for (auto &a : angles) {
            a *= math::RAD2DEG;
        }
    }

    return angles;
}

Vector3 Crane3R::calculateEndEffectorPosition(const std::array<float, 3> &values, bool degrees) {

    float j1 = degrees ? math::DEG2RAD * values[0] : values[0];
    float j2 = degrees ? math::DEG2RAD * values[1] : values[1];
    float j3 = degrees ? math::DEG2RAD * values[2] : values[1];

    auto t1 = Matrix4().makeRotationY(j1);
    auto t2 = Matrix4().makeTranslation(0, 4.2, 0).multiply(Matrix4().makeRotationZ(-j2));
    auto t3 = Matrix4().makeTranslation(7, 0, 0).multiply(Matrix4().makeRotationZ(-j3));
    auto t4 = Matrix4().makeTranslation(5.2, 0, 0);

    auto T = Matrix4().copy(t1).multiply(t2).multiply(t3).multiply(t4);
    return {T.elements[12], T.elements[13], T.elements[14]};
}

std::array<float, 3> Crane3R::computeAngles(const Vector3 &target) const {

    auto vals = getAngles();

    for (int i = 0; i < 100; ++i) {

        auto j = jacobian(vals);
        auto actual = calculateEndEffectorPosition(vals);

        auto delta = linalg::vec<float, 3>{target.x - actual.x, target.y - actual.y, target.z - actual.z};
        linalg::mat<float, 3, 3> inv = DLS(j, 0.5f);

        linalg::vec<float, 3> theta_dot = linalg::mul(inv, delta);

        for (int k = 0; k < 3; ++k) {

            vals[k] += theta_dot[k];
            if (vals[k] < limMin[k]) {
                vals[k] = limMin[k];
            } else if (vals[k] > limMax[k]) {
                vals[k] = limMax[k];
            }
        }
    }


    return vals;
}
