#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::gray;
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 50;

    OrbitControls controls{camera, canvas};

    std::vector<Vector2> points;
    for ( unsigned i = 0; i < 10; i ++ ) {
        points.emplace_back( std::sin( i * 0.2f ) * 10 + 5, ( i - 5.f ) * 2.f );
    }
    auto geometry = LatheGeometry::create( points );
    auto material = MeshNormalMaterial::create();
    material->side = DoubleSide;
    auto lathe = Mesh::create( geometry, material );
    scene->add( lathe );

    auto line = LineSegments::create(WireframeGeometry::create(*geometry));
    line->material()->as<LineBasicMaterial>()->alphaTest = false;
    lathe->add(line);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {

        lathe->rotation.y += 0.8f * dt;
        lathe->rotation.x += 0.5f * dt;

        renderer.render(scene, camera);
    });

}