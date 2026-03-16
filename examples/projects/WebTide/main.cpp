// WebTide: FFT-based ocean simulation using threepp's DawnRenderer (WebGPU).
// Ported from the WebTide BabylonJS/TypeScript project.
//
// Credits & References:
//   - Original project: WebTide by Barth Paleologue
//     https://github.com/BarthPaleologue/WebTide
//   - "Simulating Ocean Water" by Jerry Tessendorf
//   - GPU-based FFT from BabylonJS Ocean Demo by Popov72
//   - Specular coefficients from Shadertoy by afl_ext
//   - Acerola's video breakdown of Tessendorf's paper
//   - Tangent calculations by Rikard Olajos
//   - Tropical sunny day skybox from the BabylonJS Asset Library
//   - Sand texture by Engin Akyurt

#include "threepp/threepp.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/loaders/TextureLoader.hpp"

#include "threepp/renderers/dawn/GPUTexture.hpp"

#include "shaders.hpp"
#include "PhillipsSpectrum.hpp"
#include "DynamicSpectrum.hpp"
#include "IFFT.hpp"

#include <chrono>

using namespace threepp;

int main() {

    constexpr uint32_t textureSize = 256;
    constexpr float tileSize = 10.0f;
    constexpr int tileRadius = 1; // 3x3 grid (-1..1)

    // Create window and renderer
    Canvas::Parameters params;
    params.title("WebTide Ocean")
          .size(1280, 720)
          .graphicsApi(GraphicsAPI::WebGPU);

    Canvas canvas(params);

    DawnRenderer renderer(canvas);
    renderer.setClearColor(Color(0x87ceeb)); // Sky blue

    // Camera
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000.f);
    camera->position.set(0, 5, 15);
    camera->lookAt({0, 0, 0});

    OrbitControls controls(*camera, canvas);
    controls.target.set(0, 1.5f, 0);
    controls.minDistance = 2;
    controls.update();

    // Scene
    auto scene = Scene::create();

    // Lights
    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(Color(0xffffff), 0.8f);
    dirLight->position.set(1, 1, -3);
    scene->add(dirLight);

    // Initialize ocean compute pipelines
    webtide::PhillipsSpectrum phillipsSpectrum(renderer, textureSize, tileSize);
    webtide::DynamicSpectrum dynamicSpectrum(renderer, phillipsSpectrum, textureSize, tileSize);
    webtide::IFFT ifft(renderer, textureSize);

    // Output textures (spatial domain, written by IFFT)
    GPUTexture heightMap(renderer, textureSize, textureSize, GPUTexture::Format::RG32Float);
    GPUTexture gradientMap(renderer, textureSize, textureSize, GPUTexture::Format::RG32Float);
    GPUTexture displacementMap(renderer, textureSize, textureSize, GPUTexture::Format::RG32Float);

    // Create water ShaderMaterial with custom WGSL shaders
    auto waterMaterial = ShaderMaterial::create();
    waterMaterial->vertexShader = webtide::waterVertexWGSL;
    waterMaterial->fragmentShader = webtide::waterFragmentWGSL;
    waterMaterial->side = Side::Double;

    // Set custom uniforms (ocean params at binding 2)
    waterMaterial->uniforms["tileSize"] = Uniform(tileSize);

    // Bind GPU textures (sorted alphabetically for deterministic binding order)
    // Order: displacementMap, gradientMap, heightMap
    // Binding layout: 2=oceanUniforms, 3/4=displacementMap, 5/6=gradientMap (swapped with height for alphabetical), 7/8=heightMap
    // Actually alphabetical: "displacementMap" < "gradientMap" < "heightMap"
    waterMaterial->customTextures["displacementMap"] = &displacementMap;
    waterMaterial->customTextures["gradientMap"] = &gradientMap;
    waterMaterial->customTextures["heightMap"] = &heightMap;

    // Create water geometry (XZ plane with subdivisions)
    auto waterGeo = PlaneGeometry::create(tileSize, tileSize, textureSize, textureSize);
    // PlaneGeometry creates an XY plane, rotate to XZ
    waterGeo->rotateX(-math::PI / 2.0f);

    // Create 3x3 grid of water tiles
    for (int x = -tileRadius; x <= tileRadius; x++) {
        for (int z = -tileRadius; z <= tileRadius; z++) {
            auto waterMesh = Mesh::create(waterGeo, waterMaterial);
            waterMesh->position.set(
                static_cast<float>(x) * tileSize,
                0,
                static_cast<float>(z) * tileSize
            );
            scene->add(waterMesh);
        }
    }

    // Ground plane (sand)
    auto groundGeo = PlaneGeometry::create(tileSize * tileRadius * 6.0f, tileSize * tileRadius * 6.0f);
    groundGeo->rotateX(-math::PI / 2.0f);
    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color(0xc2b280); // Sand color

    // Try to load sand texture
    TextureLoader textureLoader;
    try {
        auto sandTex = textureLoader.load(std::string(DATA_FOLDER) + "/textures/sand.jpg");
        if (sandTex) {
            groundMat->map = sandTex;
        }
    } catch (...) {
        // Sand texture not found, use solid color
    }

    auto ground = Mesh::create(groundGeo, groundMat);
    ground->position.y = -2.0f;
    scene->add(ground);

    // Animation state
    float elapsedSeconds = 60.0f; // Start at 60s to avoid startup artifacts
    auto lastTime = std::chrono::high_resolution_clock::now();

    canvas.animate([&] {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        elapsedSeconds += dt;

        // Update ocean simulation (compute shaders)
        dynamicSpectrum.generate(elapsedSeconds);
        ifft.applyToTexture(dynamicSpectrum.ht, heightMap);
        ifft.applyToTexture(dynamicSpectrum.dht, gradientMap);
        ifft.applyToTexture(dynamicSpectrum.displacement, displacementMap);

        controls.update();
        renderer.render(*scene, *camera);
    });

    return 0;
}
