
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(8));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = 0x050505;
    scene->fog = Fog(0x050505, 2000, 3500);
    auto camera = PerspectiveCamera::create(27, canvas.getAspect(), 5, 3500);
    camera->position.z = 2750;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    const int numParticles = 500000;
    std::vector<float> positions(numParticles * 3);
    std::vector<float> colors(numParticles * 3);

    const float n = 1000;
    const float n2 = n / 2;

    for (int i = 0; i < numParticles; i += 3) {
        positions[i] = (math::random() * n - n2);
        positions[i + 1] = (math::random() * n - n2);
        positions[i + 2] = (math::random() * n - n2);

        colors[i] = ((positions[i] / n) + 0.5f);
        colors[i + 1] = ((positions[i + 1] / n) + 0.5f);
        colors[i + 2] = ((positions[i + 2] / n) + 0.5f);
    }

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    geometry->computeBoundingSphere();

    auto material = PointsMaterial::create();
    material->size = 2;
    material->vertexColors = true;

    auto points = Points::create(geometry, material);
    scene->add(points);

    canvas.animate([&](float t, float dt) {

        points->rotation.x = t * 0.25f;
        points->rotation.y = t * 0.5f;

        renderer.render(scene, camera);
    });
}
