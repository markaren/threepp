// Wgpu/WebGPU renderer example: lit scene with multiple material types.

#include "threepp/threepp.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

using namespace threepp;

int main() {

    Canvas::Parameters params;
    params.title("Wgpu Lit Scene")
          .size(800, 600);

    Canvas canvas(params);

    WgpuRenderer renderer(canvas);
    renderer.setClearColor(0x1a1a2e);

    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100.f);
    camera->position.z = 5;
    camera->position.y = 2;

    auto scene = Scene::create();

    // Lights
    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(Color(0xffffff), 0.8f);
    dirLight->position.set(3, 5, 4);
    scene->add(dirLight);

    auto pointLight = PointLight::create(Color(0xff4444), 1.0f, 20.0f);
    pointLight->position.set(-3, 2, 0);
    scene->add(pointLight);

    // MeshBasicMaterial (unlit, green)
    auto basicGeo = BoxGeometry::create(1, 1, 1);
    auto basicMat = MeshBasicMaterial::create();
    basicMat->color = Color(0x00ff88);
    auto basicMesh = Mesh::create(basicGeo, basicMat);
    basicMesh->position.x = -3;
    scene->add(basicMesh);

    // MeshLambertMaterial (diffuse only, blue)
    auto lambertGeo = SphereGeometry::create(0.7f, 32, 16);
    auto lambertMat = MeshLambertMaterial::create();
    lambertMat->color = Color(0x4488ff);
    auto lambertMesh = Mesh::create(lambertGeo, lambertMat);
    lambertMesh->position.x = -1;
    scene->add(lambertMesh);

    // MeshPhongMaterial (specular highlights, orange)
    auto phongGeo = SphereGeometry::create(0.7f, 32, 16);
    auto phongMat = MeshPhongMaterial::create();
    phongMat->color = Color(0xff8844);
    phongMat->specular = Color(0xffffff);
    phongMat->shininess = 60.0f;
    auto phongMesh = Mesh::create(phongGeo, phongMat);
    phongMesh->position.x = 1;
    scene->add(phongMesh);

    // MeshStandardMaterial (PBR, metallic silver)
    auto stdGeo = SphereGeometry::create(0.7f, 32, 16);
    auto stdMat = MeshStandardMaterial::create();
    stdMat->color = Color(0xcccccc);
    stdMat->roughness = 0.3f;
    stdMat->metalness = 0.8f;
    auto stdMesh = Mesh::create(stdGeo, stdMat);
    stdMesh->position.x = 3;
    scene->add(stdMesh);

    canvas.animate([&] {
        basicMesh->rotation.x += 0.01f;
        basicMesh->rotation.y += 0.02f;
        lambertMesh->rotation.y += 0.01f;
        phongMesh->rotation.y += 0.01f;
        stdMesh->rotation.y += 0.01f;

        renderer.render(*scene, *camera);
    });

    return 0;
}
