
#include "threepp/objects/Group.hpp"

using namespace threepp;

Group::Group() = default;

std::string Group::type() const {

    return "Group";
}

std::shared_ptr<Object3D> Group::clone(bool recursive) {
    auto clone = create();
    clone->copy(*this, recursive);

    return clone;
}

std::shared_ptr<Group> Group::create() {
    return std::shared_ptr<Group>(new Group());
}
