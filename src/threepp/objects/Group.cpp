
#include <memory>

#include "threepp/objects/Group.hpp"

using namespace threepp;

const std::string& Group::type() const {

    static const std::string typeName = "Group";
    return typeName;
}

std::shared_ptr<Group> Group::create() {

    return std::make_shared<Group>();
}

std::shared_ptr<Object3D> Group::createDefault() {

    return create();
}
