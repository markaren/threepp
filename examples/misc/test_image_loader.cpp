
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"

#include <iostream>
#include <fstream>

using namespace threepp;

int main() {

    const char* path = R"(C:\Users\laht\Downloads\LarsIvar.png)";

    auto image = ImageLoader().load(path);
    std::cout << "Image size: (" << image.width << ", " << image.height << ")" << std::endl;

    auto texture = TextureLoader().loadTexture(path);
    std::cout << "Image size: (" << texture.image->width << ", " << texture.image->height << ")" << std::endl;

    return 0;

}
