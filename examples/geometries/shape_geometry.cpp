#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

auto createHeartShape() {

    float x = 0, y = 0;

    auto heartShape = Shape();
    heartShape.moveTo(x + 5, y + 5)
            .bezierCurveTo(x + 5, y + 5, x + 4, y, x, y)
            .bezierCurveTo(x - 6, y, x - 6, y + 7, x - 6, y + 7)
            .bezierCurveTo(x - 6, y + 11, x - 3, y + 15.4f, x + 5, y + 19)
            .bezierCurveTo(x + 12, y + 15.4f, x + 16, y + 11, x + 16, y + 7)
            .bezierCurveTo(x + 16, y + 7, x + 16, y, x + 10, y)
            .bezierCurveTo(x + 7, y, x + 5, y + 5, x + 5, y + 5);

    return heartShape;
}

auto createFish() {

    float x = 0, y = 0;

    auto fishShape = Shape();
    fishShape.moveTo(x, y)
            .quadraticCurveTo(x + 50, y - 80, x + 90, y - 10)
            .quadraticCurveTo(x + 100, y - 10, x + 115, y - 40)
            .quadraticCurveTo(x + 115, y, x + 115, y + 40)
            .quadraticCurveTo(x + 100, y + 10, x + 90, y + 10)
            .quadraticCurveTo(x + 50, y + 80, x, y);

    return fishShape;
}

auto createSmiley() {

    auto smileyShape = Shape();
    smileyShape.moveTo(80, 40)
            .absarc(40, 40, 40, 0, math::PI * 2, false);

    auto smileyEye1Path = std::make_shared<Path>();
    smileyEye1Path->moveTo(35, 20)
            .absellipse(25, 20, 10, 10, 0, math::PI * 2, true);

    auto smileyEye2Path = std::make_shared<Path>();
    smileyEye2Path->moveTo(65, 20)
            .absarc(55, 20, 10, 0, math::PI * 2, true);

    auto smileyMouthPath = std::make_shared<Path>();
    smileyMouthPath->moveTo(20, 40)
            .quadraticCurveTo(40, 60, 60, 40)
            .bezierCurveTo(70, 45, 70, 50, 60, 60)
            .quadraticCurveTo(40, 80, 20, 60)
            .quadraticCurveTo(5, 50, 20, 40);

    smileyShape.holes.emplace_back(smileyEye1Path);
    smileyShape.holes.emplace_back(smileyEye2Path);
    smileyShape.holes.emplace_back(smileyMouthPath);

    return smileyShape;
}

std::shared_ptr<Mesh> createMesh(Shape shape, float scale = 1) {
    auto shapeGeometry = ShapeGeometry::create(shape);
    shapeGeometry->center();
    shapeGeometry->scale(scale, scale, scale);
    auto material = MeshBasicMaterial::create({{"color", 0x00ff00}, {"side", DoubleSide}});

    auto shapeMesh = Mesh::create(shapeGeometry, material);
    shapeMesh->add(LineSegments::create(WireframeGeometry::create(*shapeGeometry)));

    ExtrudeGeometry::Options opts;
    opts.depth = 3;
    auto extrudeGeometry = ExtrudeGeometry::create(shape, opts);
    extrudeGeometry->center();
    extrudeGeometry->scale(scale, scale, scale);
    auto extrudeMesh = Mesh::create(extrudeGeometry, material);
    extrudeMesh->add(LineSegments::create(WireframeGeometry::create(*extrudeMesh->geometry())));
    extrudeMesh->position.z = 10;

    shapeMesh->add(extrudeMesh);

    return shapeMesh;
}

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(65, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 5, 40);

    OrbitControls controls{camera, canvas};

    auto group = Group::create();
    group->rotateX(-math::PI);
    group->position.y = 5;

    auto heart = createMesh(createHeartShape());
    heart->position.x = 15;
    auto fish = createMesh(createFish(), 0.2f);
    fish->position.x = -15;
    auto smiley = createMesh(createSmiley(), 0.2f);
    smiley->position.y = 15;

    group->add(heart);
    group->add(fish);
    group->add(smiley);

    scene->add(group);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        group->rotation.y += 0.8f * dt;

        renderer.render(scene, camera);
    });
}
