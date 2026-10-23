// Live point cloud from an Intel RealSense L515, read through Media Foundation
// (no librealsense). Depth is deprojected with the nominal intrinsics, so the
// cloud has the right shape but is not metrically calibrated. The RGB frame is
// shown on a plane spanning the RGB field of view, and can optionally colour
// the points; depth and RGB are not registered, so that overlay is offset by
// the few centimetres between the two lenses.
//
//     l515_pointcloud [--sensor-view] [--rgb] [--snapshot out.png]
//
// --sensor-view starts with the view placed at the sensor, --rgb starts with
// RGB point colours, and --snapshot renders until 60 depth frames have
// arrived, writes the frame and exits.

#include "renderer_factory.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/sensors/RealSenseL515.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    // Near-to-far colour ramp (blue -> cyan -> yellow -> red).
    Color depthColor(float t) {
        t = std::clamp(t, 0.f, 1.f);
        const float r = std::clamp(1.5f - std::abs(4.f * t - 3.f), 0.f, 1.f);
        const float g = std::clamp(1.5f - std::abs(4.f * t - 2.f), 0.f, 1.f);
        const float b = std::clamp(1.5f - std::abs(4.f * t - 1.f), 0.f, 1.f);
        return {r, g, b};
    }

}// namespace

int main(int argc, char** argv) {

    std::string snapshot;
    bool sensorView = false;
    bool colourFromRgb = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--snapshot" && i + 1 < argc) snapshot = argv[++i];
        else if (arg == "--sensor-view") sensorView = true;
        else if (arg == "--rgb") colourFromRgb = true;
    }

    std::cout << "L515 devices found: " << RealSenseL515::deviceCount() << std::endl;
    RealSenseL515 cam;
    if (!cam.depthOpen() || !cam.colorOpen()) {
        std::cerr << "L515: " << cam.lastError() << std::endl;
        if (!cam.depthOpen()) return 1;
    }

    Canvas canvas("RealSense L515", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x101014);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01f, 100);
    OrbitControls controls(*camera, canvas);

    // Orbit view, or the eye at the sensor looking along its axis.
    auto resetView = [&](bool fromSensor) {
        if (fromSensor) camera->position.set(0, 0, 0.001f);
        else camera->position.set(1.2f, 0.8f, 1.0f);
        controls.target.set(0, 0, -1.5f);
        controls.update();
    };
    resetView(sensorView);

    // The sensor sits at the origin looking down -Z (threepp camera convention).
    scene->add(AxesHelper::create(0.1f));

    RealSenseL515::DepthFrame depth;
    RealSenseL515::ColorFrame color;

    const int maxPoints = 1024 * 768;
    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    geometry->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    geometry->setDrawRange(0, 0);

    auto pointsMaterial = PointsMaterial::create();
    pointsMaterial->size = 2;
    pointsMaterial->sizeAttenuation = false;
    pointsMaterial->vertexColors = true;
    auto points = Points::create(geometry, pointsMaterial);
    points->frustumCulled = false;
    scene->add(points);

    // RGB image on a plane spanning the RGB field of view at `rgbDistance`.
    float rgbDistance = 4.f;
    auto colorTexture = DataTexture::create(4, 1280, 720);
    colorTexture->colorSpace = ColorSpace::sRGB;
    colorTexture->magFilter = Filter::Linear;
    colorTexture->minFilter = Filter::Linear;
    auto planeGeometry = PlaneGeometry::create(1, 1);
    {
        // DataTexture rows are uploaded bottom-up; the frame is top-down.
        auto& uv = planeGeometry->getAttribute<float>("uv")->array();
        for (size_t i = 1; i < uv.size(); i += 2) uv[i] = 1.f - uv[i];
    }
    auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->map = colorTexture;
    auto rgbPlane = Mesh::create(planeGeometry, planeMaterial);
    rgbPlane->visible = cam.colorOpen();
    scene->add(rgbPlane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    float maxRange = 6.f;
    bool haveColor = false;
    int validPoints = 0;
    float depthFps = 0;
    double lastDepthTime = 0;

    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::Text("depth: %s, RGB: %s", cam.depthOpen() ? "streaming" : "closed",
                    cam.colorOpen() ? "streaming" : "closed");
        if (const auto err = cam.lastError(); !err.empty()) ImGui::TextWrapped("%s", err.c_str());
        ImGui::Text("%d points, %.1f fps", validPoints, depthFps);
        ImGui::SliderFloat("Point size", &pointsMaterial->size, 1.f, 6.f);
        ImGui::SliderFloat("Max range [m]", &maxRange, 0.5f, 9.f);
        ImGui::Checkbox("Colour from RGB (unregistered)", &colourFromRgb);
        ImGui::Checkbox("Show RGB plane", &rgbPlane->visible);
        ImGui::SliderFloat("RGB plane distance [m]", &rgbDistance, 0.5f, 9.f);
        if (ImGui::Button("Sensor view")) resetView(true);
        ImGui::SameLine();
        if (ImGui::Button("Orbit view")) resetView(false);
    }, "L515");

    canvas.animate([&] {
        if (cam.pollColor(color)) {
            if (colorTexture->image().width() != static_cast<unsigned>(color.width) ||
                colorTexture->image().height() != static_cast<unsigned>(color.height)) {
                colorTexture = DataTexture::create(4, color.width, color.height);
                colorTexture->colorSpace = ColorSpace::sRGB;
                colorTexture->magFilter = Filter::Linear;
                colorTexture->minFilter = Filter::Linear;
                planeMaterial->map = colorTexture;
            }
            colorTexture->image().data() = color.rgba;
            colorTexture->needsUpdate();
            haveColor = true;
        }

        if (cam.pollDepth(depth)) {
            if (lastDepthTime > 0 && depth.timestamp > lastDepthTime) {
                const float inst = static_cast<float>(1.0 / (depth.timestamp - lastDepthTime));
                depthFps = depthFps == 0 ? inst : depthFps + 0.1f * (inst - depthFps);
            }
            lastDepthTime = depth.timestamp;

            auto& pos = geometry->getAttribute<float>("position")->array();
            auto& col = geometry->getAttribute<float>("color")->array();
            const auto& k = depth.intrinsics;
            const auto& ck = color.intrinsics;
            const bool rgb = colourFromRgb && haveColor;
            int n = 0;
            for (int v = 0; v < depth.height; ++v) {
                for (int u = 0; u < depth.width; ++u) {
                    const float z = depth.metres(u, v);
                    if (z <= 0 || z > maxRange) continue;
                    const auto p = k.deproject(static_cast<float>(u) + 0.5f, static_cast<float>(v) + 0.5f, z);
                    // Camera frame (x right, y down, z forward) -> threepp (y up, -z forward).
                    pos[n * 3 + 0] = p[0];
                    pos[n * 3 + 1] = -p[1];
                    pos[n * 3 + 2] = -p[2];

                    Color c = depthColor(z / maxRange);
                    if (rgb) {
                        const int cu = static_cast<int>(ck.fx * p[0] / z + ck.ppx);
                        const int cv = static_cast<int>(ck.fy * p[1] / z + ck.ppy);
                        if (cu >= 0 && cu < color.width && cv >= 0 && cv < color.height) {
                            const auto* px = &color.rgba[(static_cast<size_t>(cv) * color.width + cu) * 4];
                            c.setRGB(px[0] / 255.f, px[1] / 255.f, px[2] / 255.f);
                            c.convertSRGBToLinear();
                        }
                    }
                    col[n * 3 + 0] = c.r;
                    col[n * 3 + 1] = c.g;
                    col[n * 3 + 2] = c.b;
                    ++n;
                }
            }
            validPoints = n;
            geometry->setDrawRange(0, n);
            geometry->getAttribute<float>("position")->needsUpdate();
            geometry->getAttribute<float>("color")->needsUpdate();
        }

        if (haveColor) {
            const auto& ck = color.intrinsics;
            rgbPlane->position.set(0, 0, -rgbDistance);
            rgbPlane->scale.set(rgbDistance * static_cast<float>(ck.width) / ck.fx,
                                rgbDistance * static_cast<float>(ck.height) / ck.fy, 1);
        }

        renderer->render(*scene, *camera);

        if (!snapshot.empty()) {
            if (depth.frameNumber >= 60) {
                renderer->writeFramebuffer(snapshot);
                std::cout << "wrote " << snapshot << " (" << validPoints << " points, "
                          << depthFps << " fps)" << std::endl;
                canvas.close();
            }
            return;
        }

        ui.render();
    });
}
