
#include <memory>

#include "threepp/objects/Group.hpp"

using namespace threepp;

Group::Group() {

    typeMap_["Group"] = true;
}


std::string Group::type() const {

    return "Group";
}

std::shared_ptr<Object3D> Group::clone(bool recursive) {
    auto clone = create();
    clone->copy(*this, recursive);

    return clone;
}

std::shared_ptr<Group> Group::create() {

    return std::make_shared<Group>();
}
