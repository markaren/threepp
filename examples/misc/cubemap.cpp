
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"
#include "threepp/threepp.hpp"
#include <iostream>

using namespace threepp;

std::string vertexSource() {

    return R"(
            // Output to fragment shader
            out vec3 FragPos;

            void main()
            {
                // Transform the vertex position
                FragPos = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            })";
}
std::string fragmentSource() {

    return R"(

            // Input from vertex shader
            in vec3 FragPos;


            // Uniform variables
            uniform samplerCube skybox;

            void main()
            {
                // Perform the texture lookup in the cube map
                 gl_FragColor = texture2D(skybox, FragPos);
            })";
}

int main() {

    Canvas canvas("Cubemap");
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

    PerspectiveCamera camera(50, canvas.aspect(), 0.1, 1000);
    camera.position.z = 10;

    std::filesystem::path path("data/textures");
    std::array<std::filesystem::path, 6> urls{
            // clang-format off
            path / "crate.gif", path / "crate.gif",
            path / "crate.gif", path / "crate.gif",
            path / "crate.gif", path / "crate.gif"
            // clang-format on
    };

    CubeTextureLoader loader{};
    auto reflectionCube = loader.load(urls);

    auto im = reflectionCube->image[5];
    auto tex = Texture::create(im);
    tex->needsUpdate();

    Scene scene;
    scene.background = Color::aliceblue;

    auto material = MeshLambertMaterial::create();
    material->map = tex;
    auto mesh = Mesh::create(SphereGeometry::create(0.5), material);
    mesh->position.x = 2;
    scene.add(mesh);

    auto shaderMaterial = ShaderMaterial::create();
    shaderMaterial->name = "BackgroundCubeMaterial";
    shaderMaterial->uniforms = {
            {"skybox", Uniform()},
    };
    shaderMaterial->vertexShader = vertexSource();
    shaderMaterial->fragmentShader = fragmentSource();
    shaderMaterial->side = Side::Front;
    shaderMaterial->depthTest = false;
    shaderMaterial->depthWrite = false;
    shaderMaterial->fog = false;

    shaderMaterial->uniforms["skybox"].setValue(reflectionCube.get());
    //    shaderMaterial->uniforms["flipEnvMap"].setValue(true);

    shaderMaterial->uniformsNeedUpdate = true;

    auto geometry = BoxGeometry::create(1, 1, 1);
    geometry->deleteAttribute("normal");
    geometry->deleteAttribute("uv");

    auto boxMesh = Mesh::create(geometry, shaderMaterial);
    scene.add(boxMesh);

    boxMesh->onBeforeRender = [&](void*, Object3D*, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
//        boxMesh->matrixWorld->copyPosition(*camera->matrixWorld);
    };

    OrbitControls controls{camera, canvas};

    //lights
    const auto ambient = AmbientLight::create(0xffffff);
    scene.add(ambient);

    const auto directionalLight = DirectionalLight::create(0xffffff);
    directionalLight->position.set(1, 1, 1);
    scene.add(directionalLight);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}