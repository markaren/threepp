
#include "threepp/scenes/Scene.hpp"

using namespace threepp;


std::shared_ptr<Scene> Scene::create() {

    return std::make_shared<Scene>();
}
