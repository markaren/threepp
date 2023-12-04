
#include "Crane3R.hpp"

#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<Group> make_house_attachment(int i, float len, const Vector3& p) {

        auto material = MeshPhongMaterial::create();
        material->color = Color::gray;

        float radius = 0.1f;
        auto geometry = CylinderGeometry::create(radius, radius, len);
        geometry->translate(0, len / 2, 0);
        geometry->rotateX(math::PI / 2);

        auto group = Group::create();
        group->name = "house" + std::to_string(i);
        group->position.copy(p);

        auto cyl1 = Mesh::create(geometry, material);
        auto cyl2 = Mesh::create(geometry, material);
        cyl1->position.x = 0.34f;
        cyl2->position.x = -0.34f;
        group->add(cyl1);
        group->add(cyl2);

        return group;
    }

    std::shared_ptr<Group> make_rod_attachment(int i, float len, const Vector3& p) {

        auto material = MeshPhongMaterial::create();
        material->color = Color::grey;

        float radius = 0.075f;
        auto geometry = CylinderGeometry::create(radius, radius, len);
        geometry->translate(0, len / 2, 0);
        geometry->rotateX(math::PI / 2);

        auto group = Group::create();
        group->name = "rod" + std::to_string(i);
        group->position.copy(p);

        auto cyl1 = Mesh::create(geometry, material);
        auto cyl2 = Mesh::create(geometry, material);
        cyl1->position.x = 0.34f;
        cyl2->position.x = -0.34f;
        group->add(cyl1);
        group->add(cyl2);

        return group;
    }

    void updateCylinders(std::array<std::pair<Object3D*, Object3D*>, 2> cylinders) {
        Vector3 tmp;
        for (const auto& cylinder : cylinders) {

            auto house = cylinder.first;
            auto rod = cylinder.second;

            rod->getWorldPosition(tmp);
            house->lookAt(tmp);

            house->getWorldPosition(tmp);
            rod->lookAt(tmp);
        }
    }

}// namespace

Crane3R::Crane3R(const std::shared_ptr<threepp::Group>& obj) {

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

    controller_ = std::make_unique<Controller>(*this);

    updateCylinders(cylinders_);
}

std::shared_ptr<Crane3R> Crane3R::create() {

    auto parent = Group::create();

    OBJLoader loader;
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
    parent->rotateY(-math::PI / 2);
    parent->rotateX(-math::PI / 2);

    return std::shared_ptr<Crane3R>(new Crane3R(parent));
}


std::vector<Angle> Crane3R::getValues() const {

    std::vector<Angle> angles{
            Angle::radians(parts_[0]->rotation.z()),
            Angle::radians(parts_[1]->rotation.y()),
            Angle::radians(parts_[2]->rotation.y()),
    };

    return angles;
}

void Crane3R::setTargetValues(const std::vector<Angle>& values) {

    if (controllerEnabled) {
        controller_->setTargetValues(values);
    } else {
        parts_[0]->rotation.z = values[0].inRadians();
        parts_[1]->rotation.y = values[1].inRadians();
        parts_[2]->rotation.y = values[2].inRadians();
    }
}

void Crane3R::update(float dt) {

    updateCylinders(cylinders_);

    if (controllerEnabled) {
        controller_->update(dt);
    }
}

Crane3R::Controller::Controller(const Crane3R& c) {
    actuators_[0] = std::make_unique<Object3DActuator>(c.parts_[0], Object3DActuator::Axis::Z, 1 * math::DEG2RAD, std::make_pair<float, float>(-90.f, 90.f));
    actuators_[1] = std::make_unique<Object3DActuator>(c.parts_[1], Object3DActuator::Axis::Y, 1 * math::DEG2RAD, std::make_pair<float, float>(-80.f, 0.f));
    actuators_[2] = std::make_unique<Object3DActuator>(c.parts_[2], Object3DActuator::Axis::Y, 1 * math::DEG2RAD, std::make_pair<float, float>(-140.f, 40.f));
}

void Crane3R::Controller::setGains(const std::vector<float>& values) {
    mode_ = DIRECT;
    for (unsigned i = 0; i < actuators_.size(); ++i) {
        actuators_[i]->setGain(values[i]);
    }
}

void Crane3R::Controller::setTargetValues(const std::vector<Angle>& values) {
    mode_ = POSITION;
    for (unsigned i = 0; i < 3; ++i) {
        targetValues[i] = values[i].inRadians();
    }
}

void Crane3R::Controller::update(float dt) {

    if (mode_ == POSITION) {
        for (unsigned i = 0; i < 3; ++i) {
            auto& pid = pids_[i];
            auto& act = actuators_[i];
            auto v = pid.regulate(targetValues[i], act->getProcessOutput(), dt);
            act->setGain(v);
        }
    } else {
        for (unsigned i = 0; i < 3; ++i) {
            targetValues[i] = actuators_[i]->getProcessOutput();
        }
    }

    for (auto& actuator : actuators_) {
        actuator->update();
    }
}
