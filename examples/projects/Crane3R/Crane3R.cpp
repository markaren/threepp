
#include "Crane3R.hpp"

#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

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

    auto parent = Group::create();

    auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
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

    parent->add(part1);
    parent->rotateY(-math::PI/2);
    parent->rotateX(-math::PI/2);

    return std::shared_ptr<Crane3R>(new Crane3R(parent));
}


void Crane3R::setTargetValues(const std::vector<float> &values, bool degrees) {

    if (degrees) {
        targetValues = {values[0] * math::DEG2RAD, values[1] * math::DEG2RAD, values[2] * math::DEG2RAD};
    } else {
        std::copy_n(values.begin(), 3, targetValues.begin());
    }
}

void Crane3R::update() {

    parts_[0]->rotation.z = targetValues[0];
    parts_[1]->rotation.y = targetValues[1];
    parts_[2]->rotation.y = targetValues[2];

    Vector3 tmp;
    for (const auto &cylinder : cylinders_) {

        auto house = cylinder.first;
        auto rod = cylinder.second;

        house->lookAt(rod->getWorldPosition(tmp));
        rod->lookAt(house->getWorldPosition(tmp));
    }
}

std::vector<float> Crane3R::getValues(bool degrees) const {

    std::vector<float> angles{parts_[0]->rotation.z(), parts_[1]->rotation.y(), parts_[2]->rotation.y()};
    if (degrees) {
        for (auto &a : angles) {
            a *= math::RAD2DEG;
        }
    }

    return angles;
}
