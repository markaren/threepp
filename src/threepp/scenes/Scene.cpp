
#include "threepp/scenes/Scene.hpp"

using namespace threepp;

Scene::Scene() {

    this->typeMap_["Scene"] = true;
}

std::shared_ptr<Scene> Scene::create() {

    return std::make_shared<Scene>();
}
