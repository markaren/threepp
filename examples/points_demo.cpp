
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

    int numParticles = 500000;
    std::vector<float> positions;
    std::vector<float> colors;

    float n = 1000;
    float n2 = n / 2;

    for (int i = 0; i < numParticles; i += 3) {
        positions.emplace_back(math::randomInRange(0, 1) * n - n2);
        positions.emplace_back(math::randomInRange(0, 1) * n - n2);
        positions.emplace_back(math::randomInRange(0, 1) * n - n2);

        colors.emplace_back((positions[i] / n) + 0.5f);
        colors.emplace_back((positions[i + 1] / n) + 0.5f);
        colors.emplace_back((positions[i + 2] / n) + 0.5f);
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

    float time = 0;
    points->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        time += dt;
        points->rotation.x = time * 0.25f;
        points->rotation.y = time * 0.5f;

        renderer.render(scene, camera);
    });
}
