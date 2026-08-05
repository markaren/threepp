// 3D Gaussian Splats on the GL backend.
//
//   gaussian_splats [<file.ply>]           an INRIA-convention splat file
//   gaussian_splats                        three interpenetrating procedural clouds
//
//   --shot <out.png> [--frames N]          headless capture, then exit
//   --screenshot=<out.png>                 same thing, spelled the other way
//   --cam x,y,z --look x,y,z               reframe a capture without rebuilding
//
// The procedural scene is built so that a rendering mistake is visible rather
// than plausible. Three ellipsoidal shells of flattened, randomly oriented
// Gaussians interpenetrate at different depths, in three saturated hues: if the
// back-to-front sort is wrong the shells stop looking like they pass through
// each other and start looking like whichever one happens to be drawn last is
// in front, from every angle. Orbiting is the test — a still frame of a broken
// sort can look perfectly reasonable.
//
// Press D to paint non-finite fragments magenta instead of discarding them.

#include "capture_util.hpp"

#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    // Reshapes a generated cloud into a shell of flat, randomly oriented discs
    // in one hue. The generator supplies the determinism (and the rotations,
    // which stay as they are — randomly oriented discs are exactly what makes
    // the anisotropy visible).
    SplatData shell(const Vector3& centre, const Vector3& radii, const Vector3& hue,
                    size_t count, unsigned int seed, int degree) {

        SplatGenerator::Options options;
        options.count = count;
        options.seed = seed;
        options.shDegree = degree;
        options.extent.set(2.f, 2.f, 2.f);
        options.minScale = 0.03f;
        options.maxScale = 0.07f;
        options.anisotropy = 5.f;
        options.minOpacity = 0.10f;
        options.maxOpacity = 0.30f;
        options.higherOrderAmplitude = 0.f;

        auto data = SplatGenerator::generate(options);

        for (size_t i = 0; i < data.count(); ++i) {

            // The generated mean is a point in a box; use its direction, and
            // its length as the shell thickness. Points near the centre would
            // divide by ~0, so nudge them outward first.
            Vector3 dir = data.means[i];
            if (dir.length() < 1e-3f) dir.set(0.f, 1.f, 0.f);
            const float thickness = 0.86f + 0.14f * std::min(1.f, dir.length());
            dir.normalize();

            data.means[i].set(centre.x + dir.x * radii.x * thickness,
                              centre.y + dir.y * radii.y * thickness,
                              centre.z + dir.z * radii.z * thickness);

            // Flatten one axis: discs, not blobs. Aligned with the splat's own
            // (random) rotation, so the cloud has visible grain.
            data.scales[i].z *= 0.15f;

            // Hue with a little per-splat variation, so it does not read as a
            // flat wash.
            const float jitter = 0.85f + 0.3f * static_cast<float>(i % 7) / 6.f;
            data.setDcColor(i, Vector3{hue.x * jitter, hue.y * jitter, hue.z * jitter});

            if (degree >= 1) {

                // One band-1 term per cloud: a directional sheen that only
                // appears while orbiting. Nothing else in the demo exercises
                // the view-dependent half of the shader.
                float* c = data.shAt(i);
                c[1 * 3 + 0] = 0.25f * hue.z;
                c[1 * 3 + 1] = 0.25f * hue.x;
                c[1 * 3 + 2] = 0.25f * hue.y;
            }
        }

        return data;
    }

    void append(SplatData& dst, const SplatData& src) {

        if (dst.count() == 0) dst.shDegree = src.shDegree;
        if (dst.shDegree != src.shDegree) throw std::runtime_error("SH degree mismatch");

        dst.means.insert(dst.means.end(), src.means.begin(), src.means.end());
        dst.scales.insert(dst.scales.end(), src.scales.begin(), src.scales.end());
        dst.rotations.insert(dst.rotations.end(), src.rotations.begin(), src.rotations.end());
        dst.opacities.insert(dst.opacities.end(), src.opacities.begin(), src.opacities.end());
        dst.sh.insert(dst.sh.end(), src.sh.begin(), src.sh.end());
    }

    // The clouds share one SplatCloud on purpose. Three separate objects would
    // be ordered against each other by the renderer's per-object transparency
    // sort, which is far too coarse for shells that interpenetrate — the whole
    // point is that the ordering is per splat.
    SplatData proceduralScene() {

        constexpr int DEGREE = 1;

        // Spread mostly along x, so the default camera (mostly along +z) sees
        // three overlapping shells side by side rather than one behind another,
        // and staggered in z so which one is in front actually depends on where
        // you are standing.
        SplatData data;
        append(data, shell({-1.75f, 0.f, -1.3f}, {1.25f, 1.45f, 1.25f}, {1.0f, 0.22f, 0.16f}, 900, 101u, DEGREE));
        append(data, shell({0.f, 0.f, 0.f}, {1.45f, 1.15f, 1.45f}, {0.18f, 0.95f, 0.32f}, 900, 202u, DEGREE));
        append(data, shell({1.75f, 0.f, 1.3f}, {1.25f, 1.45f, 1.25f}, {0.20f, 0.40f, 1.0f}, 900, 303u, DEGREE));

        return data;
    }

}// namespace


int main(int argc, char** argv) {

    // --cam / --look / --frames / --out come from the shared capture harness.
    const auto args = capture::parseArgs(argc, argv);

    std::string plyPath;
    std::string shotPath = args.out.value_or("");
    int shotFrames = args.frames.value_or(10);
    int shotFrame = 0;

    for (int i = 1; i < argc; ++i) {

        const std::string arg = argv[i];
        if (arg == "--shot" && i + 1 < argc) {
            shotPath = argv[++i];
        } else if (arg.rfind("--screenshot=", 0) == 0) {
            shotPath = arg.substr(13);
        } else if (arg == "--cam" || arg == "--look" || arg == "--frames" || arg == "--out") {
            ++i;// consumed by parseArgs
        } else if (arg.rfind("--", 0) != 0) {
            plyPath = arg;
        }
    }

    SplatData data;
    if (!plyPath.empty()) {

        std::cout << "loading " << plyPath << " ..." << std::endl;
        data = SplatLoader::loadPly(plyPath);
        std::cout << "  " << data.count() << " splats, SH degree " << data.shDegree;
        for (const auto& [name, values] : data.extras) std::cout << ", extra '" << name << "'";
        std::cout << std::endl;

    } else {

        data = proceduralScene();
        std::cout << "procedural scene: " << data.count() << " splats, SH degree "
                  << data.shDegree << std::endl;
    }

    // Framing uses 1 sigma, not the renderer's 3: the 3-sigma sphere is the
    // conservative bound the culler needs, and using it here backs the camera
    // off far enough that the cloud sits in the middle of the frame like a coin.
    Sphere fit;
    data.computeBounds(1.f).getBoundingSphere(fit);

    Canvas canvas("Gaussian splats", {{"aa", 0}});
    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;

    auto scene = Scene::create();
    scene->background = Color(0x101014);

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.05f, 500.f);

    auto cloud = SplatCloud::create(std::move(data));
    scene->add(cloud);

    // Frame the cloud: back off far enough that its bounding sphere fits the
    // vertical field of view, with a little margin. Mostly along +z with a
    // slight rise, so the three shells read side by side.
    const float distance = std::max(0.5f, fit.radius / std::tan(math::degToRad(camera->fov * 0.5f)) * 1.05f);
    Vector3 eye{fit.center.x + distance * 0.10f,
                fit.center.y + distance * 0.28f,
                fit.center.z + distance * 0.95f};
    Vector3 target = fit.center;

    if (args.camPos) eye = *args.camPos;
    if (args.camTarget) target = *args.camTarget;

    camera->position.copy(eye);
    camera->lookAt(target);

    OrbitControls controls{*camera, canvas};
    controls.target.copy(target);
    controls.update();

    bool debugNonFinite = false;
    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::D) {
            debugNonFinite = !debugNonFinite;
            cloud->setDebugNonFinite(debugNonFinite);
            std::cout << "non-finite debug " << (debugNonFinite ? "on (magenta)" : "off") << std::endl;
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        // Before render(), not inside onBeforeRender: the renderer uploads the
        // sorted-index attribute while it builds the render list, which is
        // earlier than the hook, so the hook's sort would show up a frame late.
        cloud->update(*camera);

        renderer.render(*scene, *camera);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {

            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
