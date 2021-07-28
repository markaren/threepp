
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct MyEventListener : EventListener {

        void onEvent(Event &e) override {
            std::cout << "Event type:" << e.type << std::endl;
        }
    };

    struct OnMaterialDispose : EventListener {

        void onEvent(Event &event) override {

            std::cout << "Material disposed" << std::endl;

            auto *material = static_cast<Material *>(event.target);
            material->removeEventListener("dispose", this);
        }
    };

}// namespace

int main() {

    EventDispatcher evt;

    MyEventListener l;

    LambdaEventListener l1([](Event &e) {
        std::cout << "Event type:" << e.type << std::endl;
    });

    evt.addEventListener("per", &l);
    evt.addEventListener("truls", &l1);

    evt.dispatchEvent("per");
    evt.dispatchEvent("per");

    evt.removeEventListener("per", &l);

    evt.dispatchEvent("per");
    evt.dispatchEvent("truls");

    std::cout << "has per evt:" << evt.hasEventListener("per", &l) << std::endl;
    std::cout << "has truls evt:" << evt.hasEventListener("truls", &l1) << std::endl;

    auto onDispose = OnMaterialDispose();
    auto material = MeshBasicMaterial::create();
    material->addEventListener("dispose", &onDispose);

    std::cout << "Has listener should be 1: " << material->hasEventListener("dispose", &onDispose) << std::endl;
    material->dispose();
    std::cout << "Has listener should be 0: " << material->hasEventListener("dispose", &onDispose) << std::endl;

    return 0;
}
