// WebTide: FFT-based ocean simulation using threepp's WgpuRenderer (WebGPU).
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
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/CubeTextureLoader.hpp"

#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include "shaders.hpp"
#include "PhillipsSpectrum.hpp"
#include "DynamicSpectrum.hpp"
#include "IFFT.hpp"

#include <chrono>

using namespace threepp;

int main() {

    constexpr uint32_t textureSize = 256;
    constexpr float tileSize = 40.0f;
    constexpr int tileRadius = 6; // 13x13 grid (-6..6)

    // Create window and renderer
    Canvas::Parameters params;
    params.title("WebTide Ocean")
          .size(1280, 720)
          .graphicsApi(GraphicsAPI::WebGPU);

    Canvas canvas(params);

    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color::black); // Will be fully covered by skybox

    // Camera
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 2000.f);
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
    dirLight->position.set(-1, 1, -3);
    scene->add(dirLight);

    // Initialize ocean compute pipelines
    webtide::PhillipsSpectrum phillipsSpectrum(renderer, textureSize, tileSize);
    webtide::DynamicSpectrum dynamicSpectrum(renderer, phillipsSpectrum, textureSize, tileSize);
    webtide::IFFT ifft(renderer, textureSize);

    // Output textures (spatial domain, written by IFFT)
    WgpuTexture heightMap(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float);
    WgpuTexture gradientMap(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float);
    WgpuTexture displacementMap(renderer, textureSize, textureSize, WgpuTexture::Format::RG32Float);

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

    // LOD geometries — inner tiles keep detail, outer tiles are coarser.
    // All share the same material and FFT textures; only vertex density differs.
    // Chebyshev distance from centre determines LOD:
    //   dist <= 1 : 128x128  (9 tiles,  ~147k verts)
    //   dist <= 3 : 64x64    (40 tiles, ~164k verts)
    //   dist <= 6 : 32x32    (120 tiles, ~123k verts)
    // Total: ~434k verts vs ~3.2M previously; ocean 520x520 vs 140x140 units.
    auto makeWaterGeo = [&](int subdiv) {
        auto g = PlaneGeometry::create(tileSize, tileSize, subdiv, subdiv);
        g->rotateX(-math::PI / 2.0f);
        return g;
    };
    auto geoInner = makeWaterGeo(256);
    auto geoMid   = makeWaterGeo(256);
    auto geoOuter = makeWaterGeo(128);

    for (int x = -tileRadius; x <= tileRadius; x++) {
        for (int z = -tileRadius; z <= tileRadius; z++) {
            int dist = std::max(std::abs(x), std::abs(z)); // Chebyshev distance
            auto& geo = dist <= 1 ? geoInner : dist <= 3 ? geoMid : geoOuter;
            auto waterMesh = Mesh::create(geo, waterMaterial);
            waterMesh->position.set(
                static_cast<float>(x) * tileSize,
                0,
                static_cast<float>(z) * tileSize
            );
            scene->add(waterMesh);
        }
    }

    // Ground plane (sand)
    auto groundGeo = PlaneGeometry::create((1+tileRadius * 2.0f) * tileSize, (1+tileRadius * 2.0f) * tileSize);
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
    ground->position.y = -5.0f;
    scene->add(ground);

    // Skybox (Tropical Sunny Day) – 6 individual textures mapped to box faces
    std::filesystem::path skyboxPath(std::string(DATA_FOLDER) + "/textures/skybox");

    TextureLoader skyTexLoader;
    auto texPx = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_px.jpg").string());
    auto texNx = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_nx.jpg").string());
    auto texPy = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_py.jpg").string());
    auto texNy = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_ny.jpg").string());
    auto texPz = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_pz.jpg").string());
    auto texNz = skyTexLoader.load((skyboxPath / "TropicalSunnyDay_nz.jpg").string());

    auto skyboxGeo = BoxGeometry::create(1800.f, 1800.f, 1800.f);

    auto matPx = MeshBasicMaterial::create();
    matPx->map = texPx;
    matPx->side = Side::Back;

    auto matNx = MeshBasicMaterial::create();
    matNx->map = texNx;
    matNx->side = Side::Back;

    auto matPy = MeshBasicMaterial::create();
    matPy->map = texPy;
    matPy->side = Side::Back;

    auto matNy = MeshBasicMaterial::create();
    matNy->map = texNy;
    matNy->side = Side::Back;

    auto matPz = MeshBasicMaterial::create();
    matPz->map = texPz;
    matPz->side = Side::Back;

    auto matNz = MeshBasicMaterial::create();
    matNz->map = texNz;
    matNz->side = Side::Back;

    std::vector<std::shared_ptr<Material>> skyboxMats{matPx, matNx, matPy, matNy, matPz, matNz};
    auto skybox = Mesh::create(skyboxGeo, skyboxMats);
    scene->add(skybox);

    // Create GPU cubemap from skybox images for water reflections
    auto& faceImg = texPx->image();
    WgpuTexture reflectionMap(renderer, faceImg.width, faceImg.height,
                             WgpuTexture::Format::RGBA8Unorm, WgpuTexture::Dimension::Cube);
    // Write each face (+X, -X, +Y, -Y, +Z, -Z)
    auto& dataPx = texPx->image().data<unsigned char>();
    auto& dataNx = texNx->image().data<unsigned char>();
    auto& dataPy = texPy->image().data<unsigned char>();
    auto& dataNy = texNy->image().data<unsigned char>();
    auto& dataPz = texPz->image().data<unsigned char>();
    auto& dataNz = texNz->image().data<unsigned char>();
    reflectionMap.writeFace(0, dataPx.data(), dataPx.size());
    reflectionMap.writeFace(1, dataNx.data(), dataNx.size());
    reflectionMap.writeFace(2, dataPy.data(), dataPy.size());
    reflectionMap.writeFace(3, dataNy.data(), dataNy.size());
    reflectionMap.writeFace(4, dataPz.data(), dataPz.size());
    reflectionMap.writeFace(5, dataNz.data(), dataNz.size());

    waterMaterial->customTextures["reflectionMap"] = &reflectionMap;

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
