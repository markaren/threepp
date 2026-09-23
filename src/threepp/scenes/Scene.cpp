
#include "threepp/scenes/Scene.hpp"

#include "threepp/textures/CubeTexture.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;


std::string Scene::type() const {

    return "Scene";
}

void Scene::copy(const Object3D& source, bool recursive) {
    Object3D::copy(source, recursive);

    if (const auto s = source.as<Scene>()) {

        background = s->background;
        environment = s->environment;
        fog = s->fog;
        overrideMaterial = s->overrideMaterial;
        autoUpdate = s->autoUpdate;
    }
}

std::shared_ptr<Scene> Scene::create() {

    return std::make_shared<Scene>();
}

std::shared_ptr<Object3D> Scene::createDefault() {

    return create();
}

Background::Background() {}

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

bool Background::empty() const {

    return !hasValue_;
}
