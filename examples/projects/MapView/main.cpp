
#include "threepp/threepp.hpp"

#include "geo/MapView.hpp"
#include "geo/providers/OpenStreetMapsProvider.hpp"
#include "geo/utils/UnitUtils.hpp"
#include "geo/lod/LODRadial.hpp"

using namespace threepp;

int main() {

    Canvas canvas("MapView");
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    scene.add(AmbientLight::create(0x777777));

    PerspectiveCamera camera(80, canvas.aspect(), 0.1, 1e12);
    camera.layers.enable(0);

    OrbitControls controls{camera, canvas};

    MapView map(std::make_unique<OpenStreetMapProvider>(), std::make_unique<LODRadial>());
    scene.add(map);
    map.updateMatrixWorld(true);

    const auto coords = utils::datumsToSpherical(62.50094228364612, 6.09138498277564);
    controls.target.set(coords.x, 0, -coords.y);
    camera.position.set(coords.x, 100000, -coords.y);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
