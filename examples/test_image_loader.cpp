
#include "threepp/loaders/ImageLoader.hpp"

#include <iostream>

using namespace threepp;

int main() {

    auto image = ImageLoader().load(R"(C:\Users\laht\Downloads\LarsIvar.png)");

    std::cout << image.getData() << std::endl;

}
