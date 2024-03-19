
#include "threepp/scenes/Scene.hpp"

#include "threepp/textures/Texture.hpp"
#include "threepp/textures/CubeTexture.hpp"

using namespace threepp;


std::shared_ptr<Scene> Scene::create() {

    return std::make_shared<Scene>();
}

Background::Background(): hasValue_(false) {}

Background::Background(int color): Background(Color(color)) {}

Background::Background(const Color& color): color_(color), hasValue_(true) {}

Background::Background(const std::shared_ptr<Texture>& texture): texture_(texture), hasValue_(texture) {}

Background::Background(const std::shared_ptr<CubeTexture>& texture): texture_(texture), hasValue_(texture) {}

bool Background::isColor() const {

    return color_.has_value();
}

bool Background::isTexture() const {

    return texture_ != nullptr;
}

Color& Background::color() {

    return *color_;
}

std::shared_ptr<Texture> Background::texture() const {

    return texture_;
}

bool Background::empty() const{

    return !hasValue_;
}
