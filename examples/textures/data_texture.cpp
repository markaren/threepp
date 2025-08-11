
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    void updateSpritePosition(Sprite& sprite, std::pair<int, int> windowSize, unsigned int textureSize) {

        const auto halfWidth = static_cast<float>(windowSize.first) / 2;
        const auto halfHeight = static_cast<float>(windowSize.second) / 2;

        const auto halfImageWidth = static_cast<float>(textureSize) / 2;
        const auto halfImageHeight = static_cast<float>(textureSize) / 2;

        sprite.position.set(-halfWidth + halfImageWidth, halfHeight - halfImageHeight, 1);
    }

    void addGrid(Scene& scene) {
        auto grid1 = GridHelper::create(5);
        grid1->rotateX(math::PI / 2);
        grid1->position.z = -2.5;
        scene.add(grid1);

        auto grid2 = GridHelper::create(5);
        grid2->rotateX(math::PI / 2).rotateZ(math::PI / 2);
        grid2->position.x = -2.5;
        scene.add(grid2);

        auto grid3 = GridHelper::create(5);
        grid3->position.y = -2.5;
        scene.add(grid3);
    }

}// namespace

int main() {

    Canvas canvas("Data texture", {{"aa", 4}});
    auto size = canvas.size();
    GLRenderer renderer{size};
    renderer.autoClear = false;
    renderer.setClearColor(Color::aliceblue);

    Scene scene;
    Scene orthoScene;

    PerspectiveCamera camera(70, canvas.aspect(), 0.1f, 1000);
    camera.position.z = 10;

    OrthographicCamera orthoCamera(-size.width() / 2, size.width() / 2, size.height() / 2, -size.height() / 2, 1, 10);
    orthoCamera.position.z = 10;

    OrbitControls controls{camera, canvas};

    unsigned int textureSize = 128;
    auto texture = DataTexture::create(3, textureSize, textureSize);
    texture->format = Format::RGB;
    texture->minFilter = Filter::Nearest;
    texture->magFilter = Filter::Nearest;

    auto spriteMaterial = SpriteMaterial::create({{"map", texture}});
    spriteMaterial->map->offset.set(0.5, 0.5);
    Sprite sprite(spriteMaterial);
    sprite.scale.set(textureSize, textureSize, 1);
    orthoScene.add(sprite);

    updateSpritePosition(sprite, size, textureSize);

    TextureLoader tl;

    const auto sphereGeometry = SphereGeometry::create(0.5f, 16, 16);
    const auto sphereMaterial = MeshBasicMaterial::create({{"map", tl.load(std::string(DATA_FOLDER) + "/textures/checker.png")}});
    Mesh sphere(sphereGeometry, sphereMaterial);
    sphere.position.x = 1;
    scene.add(sphere);

    const auto boxGeometry = BoxGeometry::create(1, 1, 1);
    const auto boxMaterial = MeshBasicMaterial::create(
            {{"map", tl.load(std::string(DATA_FOLDER) + "/textures/crate.gif")}});
    Mesh box(boxGeometry, boxMaterial);
    box.position.x = -1;
    scene.add(box);

    addGrid(scene);

    canvas.onWindowResize([&](WindowSize newSize) {
        camera.aspect = newSize.aspect();
        camera.updateProjectionMatrix();

        orthoCamera.left = -newSize.width() / 2;
        orthoCamera.right = newSize.width() / 2;
        orthoCamera.top = newSize.height() / 2;
        orthoCamera.bottom = -newSize.height() / 2;
        orthoCamera.updateProjectionMatrix();

        renderer.setSize(newSize);
        size = newSize;

        updateSpritePosition(sprite, newSize, textureSize);
    });

    Clock clock;
    Vector2 vector;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        box.rotation.y += 0.5f * dt;
        sphere.rotation.x += 0.5f * dt;

        renderer.clear();
        renderer.render(scene, camera);

        vector.x = (size.width() / 2) - (textureSize / 2);
        vector.y = (size.height() / 2) - (textureSize / 2);

        renderer.copyFramebufferToTexture(vector, *texture);

        renderer.clearDepth();
        renderer.render(orthoScene, orthoCamera);
    });
}
