
#include "threepp/core/EventDispatcher.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct MyEventListener : EventListener {

        void onEvent(Event &e) override {
            std::cout << "Event type:" << e.type << std::endl;
        }
    };

}

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

    return 0;

}
