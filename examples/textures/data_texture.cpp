
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    void updateSpritePosition(Sprite& sprite, WindowSize windowSize, unsigned int textureSize) {

        const auto halfWidth = static_cast<float>(windowSize.width) / 2;
        const auto halfHeight = static_cast<float>(windowSize.height) / 2;

        const auto halfImageWidth = static_cast<float>(textureSize) / 2;
        const auto halfImageHeight = static_cast<float>(textureSize) / 2;

        sprite.position.set(-halfWidth + halfImageWidth, halfHeight - halfImageHeight, 1);
    }

}// namespace

int main() {

    Canvas canvas("Data texture", {{"antialiasing", 4}});
    GLRenderer renderer{canvas};
    renderer.autoClear = false;
    renderer.setClearColor(Color::aliceblue);

    const auto& size = canvas.getSize();

    auto scene = Scene::create();
    auto orthoScene = Scene::create();

    auto camera = PerspectiveCamera::create(70, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 10;

    auto orthoCamera = OrthographicCamera::create(-size.width / 2, size.width / 2, size.height / 2, -size.height / 2, 1, 10);
    orthoCamera->position.z = 10;

    OrbitControls controls{camera, canvas};

    unsigned int textureSize = 128;
    std::vector<unsigned char> data(textureSize * textureSize * 3);
    auto texture = DataTexture::create(data, textureSize, textureSize);
    texture->format = RGBFormat;
    texture->minFilter = NearestFilter;
    texture->magFilter = NearestFilter;

    auto spriteMaterial = SpriteMaterial::create({{"map", texture}});
    spriteMaterial->map->offset.set(0.5, 0.5);
    auto sprite = Sprite::create(spriteMaterial);
    sprite->scale.set(textureSize, textureSize, 1);
    orthoScene->add(sprite);

    updateSpritePosition(*sprite, size, textureSize);

    TextureLoader tl;

    const auto sphereGeometry = SphereGeometry::create(0.5f, 16, 16);
    const auto sphereMaterial = MeshBasicMaterial::create({{"map", tl.load("data/textures/checker.png")}});
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.x = 1;
    scene->add(sphere);

    const auto boxGeometry = BoxGeometry::create(1, 1, 1);
    const auto boxMaterial = MeshBasicMaterial::create({{"map", tl.load("data/textures/crate.gif")}});
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.x = -1;
    scene->add(box);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();

        orthoCamera->left = -size.width / 2;
        orthoCamera->right = size.width / 2;
        orthoCamera->top = size.height / 2;
        orthoCamera->bottom = -size.height / 2;
        orthoCamera->updateProjectionMatrix();

        renderer.setSize(size);

        updateSpritePosition(*sprite, size, textureSize);
    });

    Vector2 vector;
    canvas.animate([&](float dt) {
        box->rotation.y += 0.5f * dt;
        sphere->rotation.x += 0.5f * dt;

        renderer.clear();
        renderer.render(scene, camera);

        vector.x = (size.width / 2) - (textureSize / 2);
        vector.y = (size.height / 2) - (textureSize / 2);

        renderer.copyFramebufferToTexture(vector, *texture);

        renderer.clearDepth();
        renderer.render(orthoScene, orthoCamera);
    });
}
