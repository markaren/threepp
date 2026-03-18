
#include "threepp/objects/Water.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/threepp.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include <cmath>

using namespace threepp;

struct WaterScene {
    std::shared_ptr<Scene> scene;
    std::shared_ptr<DirectionalLight> light;
    std::shared_ptr<Mesh> sphere;
    std::shared_ptr<Water> water;
    std::shared_ptr<Sky> sky;

    float elevation = 2;
    float azimuth = 180;

    void build() {
        scene = Scene::create();

        light = DirectionalLight::create(0xffffff);
        scene->add(light);

        auto sphereGeometry = SphereGeometry::create(30);
        auto sphereMaterial = MeshBasicMaterial::create();
        sphereMaterial->color.setHex(0x0000ff);
        sphereMaterial->wireframe = true;
        sphere = Mesh::create(sphereGeometry, sphereMaterial);
        scene->add(sphere);

        TextureLoader textureLoader{};
        auto texture = textureLoader.load(std::string(DATA_FOLDER) + "/textures/waternormals.jpg");
        texture->wrapS = TextureWrapping::Repeat;
        texture->wrapT = TextureWrapping::Repeat;

        Water::Options opt;
        opt.textureHeight = 512;
        opt.textureWidth = 512;
        opt.alpha = 0.8f;
        opt.waterNormals = texture;
        opt.distortionScale = 3.7f;
        opt.sunDirection = light->position.clone().normalize();
        opt.sunColor = light->color;
        opt.waterColor = 0x001e0f;
        opt.fog = false;

        auto waterGeometry = PlaneGeometry::create(10000, 10000);
        water = Water::create(waterGeometry, opt);
        water->rotateX(math::degToRad(-90));
        scene->add(water);

        sky = Sky::create();
        sky->scale.setScalar(10000);
        auto& su = sky->material()->as<ShaderMaterial>()->uniforms;
        su.at("turbidity").value<float>() = 10;
        su.at("rayleigh").value<float>() = 1;
        su.at("mieCoefficient").value<float>() = 0.005;
        su.at("mieDirectionalG").value<float>() = 0.8;
        scene->add(sky);

        computeSunPosition();
    }

    void computeSunPosition() {
        float phi = math::degToRad(90 - elevation);
        float theta = math::degToRad(azimuth);
        light->position.setFromSphericalCoords(1, phi, theta);
        auto& su = sky->material()->as<ShaderMaterial>()->uniforms;
        su.at("sunPosition").value<Vector3>().copy(light->position);
    }

    void animate(float t) {
        sphere->position.y = std::sin(t) * 20 + 5;
        sphere->rotation.x = t * 0.05f;
        sphere->rotation.z = t * 0.051f;
        water->material()->as<ShaderMaterial>()->uniforms.at("time").setValue(t);
    }
};

int main() {

    const int halfW = 640;
    const int halfH = 480;

    // Single visible GL canvas (double width)
    Canvas canvas(Canvas::Parameters()
                          .title("Water — OpenGL (left) vs WebGPU/Dawn (right)")
                          .size(halfW * 2, halfH)
                          .antialiasing(4));
    auto glRenderer = createRenderer(canvas);
    glRenderer->checkShaderErrors = true;
    glRenderer->toneMapping = ToneMapping::ACESFilmic;
    glRenderer->setScissorTest(true);

    // Headless Dawn canvas for offscreen rendering
    Canvas dawnCanvas(Canvas::Parameters()
                              .size(halfW, halfH)
                              .headless(true)
                              .graphicsApi(GraphicsAPI::WebGPU));
    DawnRenderer dawnRenderer(dawnCanvas);
    dawnRenderer.checkShaderErrors = true;
    dawnRenderer.toneMapping = ToneMapping::ACESFilmic;

    // Dawn render target for offscreen rendering + readback
    auto dawnRT = RenderTarget::create(halfW, halfH, RenderTarget::Options{});

    // DataTexture to receive Dawn readback pixels, displayed on right half
    auto dawnTex = DataTexture::create(3, halfW, halfH);
    dawnTex->format = Format::RGB;
    dawnTex->magFilter = Filter::Linear;
    dawnTex->minFilter = Filter::Linear;

    // Preview scene: fullscreen quad showing the Dawn texture
    auto previewScene = Scene::create();
    auto previewCam = OrthographicCamera::create(-1, 1, 1, -1, 0, 1);
    auto previewMat = MeshBasicMaterial::create();
    previewMat->map = dawnTex;
    auto previewMesh = Mesh::create(PlaneGeometry::create(2, 2), previewMat);
    previewScene->add(previewMesh);

    // Shared perspective camera
    auto camera = PerspectiveCamera::create(55, static_cast<float>(halfW) / halfH, 1, 2000);
    camera->position.set(-50, 120, 500);

    OrbitControls controls{*camera, canvas};
    controls.maxPolarAngle = math::PI * 0.495f;
    controls.target.set(0, 10, 0);
    controls.minDistance = 40;
    controls.maxDistance = 400;
    controls.update();

    // Two separate scene graphs (Water's onBeforeRender captures its own renderer)
    WaterScene sceneGL, sceneDawn;
    sceneGL.build();
    sceneDawn.build();

    auto& suGL = sceneGL.sky->material()->as<ShaderMaterial>()->uniforms;
    auto& suDawn = sceneDawn.sky->material()->as<ShaderMaterial>()->uniforms;

    float elevation = 2;
    float azimuth = 180;

    auto syncSun = [&] {
        sceneGL.elevation = elevation;
        sceneGL.azimuth = azimuth;
        sceneGL.computeSunPosition();
        sceneDawn.elevation = elevation;
        sceneDawn.azimuth = azimuth;
        sceneDawn.computeSunPosition();
    };

    auto syncSkyParams = [&] {
        suDawn.at("turbidity").value<float>() = suGL.at("turbidity").value<float>();
        suDawn.at("rayleigh").value<float>() = suGL.at("rayleigh").value<float>();
        suDawn.at("mieCoefficient").value<float>() = suGL.at("mieCoefficient").value<float>();
        suDawn.at("mieDirectionalG").value<float>() = suGL.at("mieDirectionalG").value<float>();
    };

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Controls");
        ImGui::SliderFloat("turbidity", &suGL.at("turbidity").value<float>(), 0, 20);
        ImGui::SliderFloat("rayleigh", &suGL.at("rayleigh").value<float>(), 0, 4);
        ImGui::SliderFloat("mieCoefficient", &suGL.at("mieCoefficient").value<float>(), 0, 0.1);
        ImGui::SliderFloat("mieDirectionalG", &suGL.at("mieDirectionalG").value<float>(), 0, 1);
        if (ImGui::SliderFloat("elevation", &elevation, 0, 90)) {
            syncSun();
        }
        if (ImGui::SliderFloat("azimuth", &azimuth, -180, 180)) {
            syncSun();
        }
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = static_cast<float>(size.width() / 2) / size.height();
        camera->updateProjectionMatrix();
        glRenderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        const auto t = clock.getElapsedTime();
        const auto size = canvas.size();
        const int hw = size.width() / 2;
        const int h = size.height();

        sceneGL.animate(t);
        sceneDawn.animate(t);
        syncSkyParams();

        // --- Dawn offscreen render ---
        dawnRenderer.setRenderTarget(dawnRT.get());
        dawnRenderer.render(*sceneDawn.scene, *camera);
        auto pixels = dawnRenderer.readRGBPixels();

        if (!pixels.empty()) {
            // Flip vertically: Dawn readback is top-down, GL textures are bottom-up
            const int rowBytes = halfW * 3;
            std::vector<unsigned char> flipped(pixels.size());
            for (int row = 0; row < halfH; row++) {
                std::memcpy(&flipped[row * rowBytes],
                            &pixels[(halfH - 1 - row) * rowBytes],
                            rowBytes);
            }
            dawnTex->image().setData(std::move(flipped));
            dawnTex->needsUpdate();
        }

        // --- GL left half: render scene directly ---
        glRenderer->setViewport(0, 0, hw, h);
        glRenderer->setScissor(0, 0, hw, h);
        glRenderer->render(*sceneGL.scene, *camera);

        // --- GL right half: display Dawn readback texture ---
        // Disable tone mapping — the Dawn render already applied it.
        auto savedToneMapping = glRenderer->toneMapping;
        glRenderer->toneMapping = ToneMapping::None;
        glRenderer->setViewport(hw, 0, hw, h);
        glRenderer->setScissor(hw, 0, hw, h);
        glRenderer->render(*previewScene, *previewCam);
        glRenderer->toneMapping = savedToneMapping;

        // ImGui overlay (drawn over the full window after scissor reset)
        glRenderer->setViewport(0, 0, size.width(), h);
        glRenderer->setScissor(0, 0, size.width(), h);
        ui.render();
    });
}
