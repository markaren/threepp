// A Sphero RVR in a walled arena, driven through the same two commands as the
// real robot: driveWithHeading and driveRaw. It carries a camera and a
// time-of-flight sensor at each end, and can use them to follow the ball that
// rolls about the arena (BallFollower.hpp), which is the same loop one would
// run on the real thing: read a camera frame, find the ball, send a drive command.
//
//   W / S        drive forwards / backwards
//   A / D        turn the heading the robot holds
//   Space        boost
//   R            raw (tank) mode on-off: W / S run the left track, Up / Down the right
//   F            follow the ball on-off
//   B            time-of-flight beams on-off
//   mouse        orbit and zoom the camera, which rides along with the robot
//
// Headless (dev):
//   SpheroControl --shot out.png [--time T] [--follow] [--cam X Y Z] [--size WxH]
//   SpheroControl --sequence <dir> [--time T] [--fps F] [--follow] [--cam X Y Z] [--size WxH]
//   (--follow starts out following the ball, --cam is the camera's offset from the robot)

#include "threepp/threepp.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/textures/DataTexture.hpp"

#include "BallFollower.hpp"
#include "Sphero.hpp"
#include "SpheroKeyController.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // The arena floor, half its size along x and z [m].
    constexpr float arenaX = 1.5f;
    constexpr float arenaZ = 1.1f;

    constexpr float ballRadius = 0.06f;
    constexpr float ballSpeed = 0.22f;// [m/s]

    constexpr int feedWidth = 320;
    constexpr int feedHeight = 240;

    // What stands on the arena floor, as the shapes a disc can run into.
    class Arena {

    public:
        void addDisc(float x, float z, float radius) {
            discs_.push_back({x, z, radius});
        }

        void addBox(float x, float z, float halfX, float halfZ, float yaw) {
            boxes_.push_back({x, z, halfX, halfZ, std::cos(yaw), std::sin(yaw)});
        }

        // Moves a disc at (x, z) out of the walls and of everything on the
        // floor. Returns whether it touched anything, with (nx, nz) the
        // direction it was pushed in.
        bool resolve(float& x, float& z, float radius, float& nx, float& nz) const {
            nx = nz = 0;
            const auto push = [&](float dx, float dz) {
                x += dx;
                z += dz;
                nx += dx;
                nz += dz;
            };

            if (x > arenaX - radius) push(arenaX - radius - x, 0);
            if (x < -arenaX + radius) push(-arenaX + radius - x, 0);
            if (z > arenaZ - radius) push(0, arenaZ - radius - z);
            if (z < -arenaZ + radius) push(0, -arenaZ + radius - z);

            for (const auto& disc : discs_) {
                const float dx = x - disc.x, dz = z - disc.z;
                const float distance = std::hypot(dx, dz);
                const float overlap = radius + disc.radius - distance;
                if (overlap > 0 && distance > 0) push(dx / distance * overlap, dz / distance * overlap);
            }

            for (const auto& box : boxes_) {
                // into the box's own frame
                const float wx = x - box.x, wz = z - box.z;
                const float lx = box.cos * wx - box.sin * wz;
                const float lz = box.sin * wx + box.cos * wz;

                const float dx = lx - std::clamp(lx, -box.halfX, box.halfX);
                const float dz = lz - std::clamp(lz, -box.halfZ, box.halfZ);
                const float distance = std::hypot(dx, dz);
                if (distance >= radius) continue;

                float px, pz;
                if (distance > 0) {
                    px = dx / distance * (radius - distance);
                    pz = dz / distance * (radius - distance);
                } else {
                    // the centre is inside: out through the nearest face
                    const float toX = box.halfX - std::abs(lx), toZ = box.halfZ - std::abs(lz);
                    px = toX < toZ ? std::copysign(toX + radius, lx) : 0;
                    pz = toX < toZ ? 0 : std::copysign(toZ + radius, lz);
                }
                push(box.cos * px + box.sin * pz, -box.sin * px + box.cos * pz);
            }

            const float length = std::hypot(nx, nz);
            if (length == 0) return false;
            nx /= length;
            nz /= length;

            return true;
        }

    private:
        struct Disc {
            float x, z, radius;
        };
        struct Box {
            float x, z, halfX, halfZ, cos, sin;
        };

        std::vector<Disc> discs_;
        std::vector<Box> boxes_;
    };

    // The ball rolls about the arena on its own, bouncing off whatever it meets.
    struct Ball {
        std::shared_ptr<Mesh> mesh;
        float vx = -0.6f * ballSpeed, vz = 0.8f * ballSpeed;

        void update(float dt, float time, const Arena& arena, Sphero& sphero) {
            // a slow weave, so it does not settle into one loop
            const float turn = 0.35f * std::sin(0.4f * time) * dt;
            const float c = std::cos(turn), s = std::sin(turn);
            const float wx = c * vx + s * vz, wz = -s * vx + c * vz;
            vx = wx;
            vz = wz;

            auto& p = mesh->position;
            const Vector3 before = p;
            p.x += vx * dt;
            p.z += vz * dt;

            // the robot shoves it along
            const float reach = Sphero::footprintRadius + ballRadius;
            float dx = p.x - sphero.position.x, dz = p.z - sphero.position.z;
            float distance = std::hypot(dx, dz);
            if (distance < reach && distance > 0) {
                p.x += dx / distance * (reach - distance);
                p.z += dz / distance * (reach - distance);
                bounce(dx / distance, dz / distance);
            }

            float nx, nz;
            if (arena.resolve(p.x, p.z, ballRadius, nx, nz)) bounce(nx, nz);

            // pinned between the robot and something solid, it is the robot that gives way
            dx = p.x - sphero.position.x;
            dz = p.z - sphero.position.z;
            distance = std::hypot(dx, dz);
            if (distance < reach && distance > 0) {
                sphero.position.x -= dx / distance * (reach - distance);
                sphero.position.z -= dz / distance * (reach - distance);
            }

            const Vector3 moved = Vector3().subVectors(p, before);
            const float rolled = moved.length();
            if (rolled > 0) {
                mesh->rotateOnWorldAxis(Vector3(moved.z, 0, -moved.x).normalize(), rolled / ballRadius);
            }
        }

    private:
        void bounce(float nx, float nz) {
            const float along = vx * nx + vz * nz;
            if (along < 0) {
                vx -= 2 * along * nx;
                vz -= 2 * along * nz;
            }
        }
    };

    std::shared_ptr<DataTexture> createTexture(int width, int height, const std::function<Color(float, float)>& colorAt) {
        std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Color c = colorAt((static_cast<float>(x) + 0.5f) / static_cast<float>(width), (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
                auto* p = &pixels[(static_cast<size_t>(y) * width + x) * 4];
                p[0] = static_cast<unsigned char>(std::clamp(c.r, 0.f, 1.f) * 255);
                p[1] = static_cast<unsigned char>(std::clamp(c.g, 0.f, 1.f) * 255);
                p[2] = static_cast<unsigned char>(std::clamp(c.b, 0.f, 1.f) * 255);
                p[3] = 255;
            }
        }

        auto texture = DataTexture::create(ImageData{std::move(pixels)}, width, height);
        texture->magFilter = Filter::Linear;
        texture->minFilter = Filter::LinearMipmapLinear;
        texture->generateMipmaps = true;
        texture->anisotropy = 16;
        texture->colorSpace = ColorSpace::sRGB;
        texture->needsUpdate();

        return texture;
    }

    // The arena mat: slate, with the white lines of a playing field.
    std::shared_ptr<Texture> createMatTexture() {
        constexpr int width = 1200, height = 880;
        constexpr float lineHalfWidth = 0.007f;       // [m]
        constexpr float blur = 1.5f * 2 * arenaX / width;// antialiasing, about a texel and a half

        const Color slate(0x40566b), white(0xe6eaed);

        return createTexture(width, height, [&](float u, float v) {
            const float x = (u - 0.5f) * 2 * arenaX, z = (v - 0.5f) * 2 * arenaZ;

            // distance to the nearest line [m]
            const float qx = std::abs(x) - (arenaX - 0.12f), qz = std::abs(z) - (arenaZ - 0.12f);
            const float border = std::abs(std::hypot(std::max(qx, 0.f), std::max(qz, 0.f)) + std::min(std::max(qx, qz), 0.f));
            const float halfway = qz < 0 ? std::abs(x) : 1.f;
            const float circle = std::abs(std::hypot(x, z) - 0.3f);
            const float spot = std::hypot(x, z) - 0.02f + lineHalfWidth;
            const float distance = std::min({border, halfway, circle, spot});

            // a little grain, so the mat is not one flat colour
            const float grain = std::fmod(std::abs(std::sin(u * 7919.f + v * 10007.f) * 43758.5453f), 1.f) - 0.5f;

            Color c = slate;
            c.lerp(white, 1 - math::smoothstep(lineHalfWidth - blur, lineHalfWidth + blur, distance));
            c.r += grain * 0.018f;
            c.g += grain * 0.018f;
            c.b += grain * 0.018f;

            return c;
        });
    }

    // Magenta, in two tones so that one can see the ball roll.
    std::shared_ptr<Texture> createBallTexture() {

        return createTexture(64, 32, [](float u, float) {
            return static_cast<int>(u * 6) % 2 == 0 ? Color(0xff2fc4) : Color(0xd9189f);
        });
    }

    std::shared_ptr<MeshStandardMaterial> matte(const Color& color, float roughness = 0.7f) {
        auto material = MeshStandardMaterial::create();
        material->color = color;
        material->roughness = roughness;

        return material;
    }

    template<class T>
    std::shared_ptr<T> shadowed(std::shared_ptr<T> object) {
        object->castShadow = true;
        object->receiveShadow = true;

        return object;
    }

    std::shared_ptr<Object3D> createCone() {
        const auto orange = matte(0xff6a13, 0.55f);

        auto cone = Group::create();
        auto base = shadowed(Mesh::create(BoxGeometry::create(0.1f, 0.008f, 0.1f), orange));
        base->position.y = 0.004f;
        cone->add(base);
        auto body = shadowed(Mesh::create(CylinderGeometry::create(0.008f, 0.04f, 0.12f, 32), orange));
        body->position.y = 0.068f;
        cone->add(body);
        auto band = shadowed(Mesh::create(CylinderGeometry::create(0.0206f, 0.0273f, 0.025f, 32), matte(0xf2f2f2, 0.5f)));
        band->position.y = 0.0705f;
        cone->add(band);

        return cone;
    }

    void setupArena(Scene& scene, Arena& arena) {

        auto room = Mesh::create(PlaneGeometry::create(60, 60), matte(0x20252b, 0.9f));
        room->rotation.x = -math::PI / 2;
        room->position.y = -0.002f;
        room->receiveShadow = true;
        scene.add(room);

        auto matMaterial = matte(Color::white, 0.85f);
        matMaterial->map = createMatTexture();
        auto mat = Mesh::create(PlaneGeometry::create(2 * arenaX, 2 * arenaZ), matMaterial);
        mat->rotation.x = -math::PI / 2;
        mat->receiveShadow = true;
        scene.add(mat);

        constexpr float wallHeight = 0.12f, wallThickness = 0.03f;
        const auto board = matte(0xe9e6df);
        for (const float side : {1.f, -1.f}) {
            auto alongX = shadowed(Mesh::create(BoxGeometry::create(2 * arenaX + 2 * wallThickness, wallHeight, wallThickness), board));
            alongX->position.set(0, wallHeight / 2, side * (arenaZ + wallThickness / 2));
            scene.add(alongX);
            auto alongZ = shadowed(Mesh::create(BoxGeometry::create(wallThickness, wallHeight, 2 * arenaZ), board));
            alongZ->position.set(side * (arenaX + wallThickness / 2), wallHeight / 2, 0);
            scene.add(alongZ);
        }

        for (const auto& [x, z] : std::vector<std::pair<float, float>>{{0.55f, -0.45f}, {-0.35f, 0.6f}, {0.95f, 0.6f}, {-0.2f, -0.7f}}) {
            auto cone = createCone();
            cone->position.set(x, 0, z);
            scene.add(cone);
            arena.addDisc(x, z, 0.055f);
        }

        struct Block {
            float x, z, halfX, halfY, halfZ, yaw;
            Color color;
        };
        for (const auto& b : {Block{1.05f, -0.4f, 0.06f, 0.06f, 0.06f, -0.4f, 0x2f6fde},
                              Block{-0.85f, -0.7f, 0.1f, 0.04f, 0.05f, 0.15f, 0xf2c230},
                              Block{0.2f, 0.78f, 0.07f, 0.07f, 0.07f, 0.6f, 0xd8a35a}}) {
            auto block = shadowed(Mesh::create(BoxGeometry::create(2 * b.halfX, 2 * b.halfY, 2 * b.halfZ), matte(b.color, 0.6f)));
            block->position.set(b.x, b.halfY, b.z);
            block->rotation.y = b.yaw;
            scene.add(block);
            arena.addBox(b.x, b.z, b.halfX, b.halfZ, b.yaw);
        }
    }

    void setupLights(Scene& scene) {

        scene.add(HemisphereLight::create(0xdfe9ff, 0x30343a, 0.7f));

        auto sun = DirectionalLight::create(0xfff4e6, 1.6f);
        sun->position.set(1.4f, 3.f, 1.8f);
        sun->castShadow = true;
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0004f;
        sun->shadow->radius = 3.f;
        auto shadowCamera = sun->shadow->camera->as<OrthographicCamera>();
        shadowCamera->left = shadowCamera->bottom = -2.2f;
        shadowCamera->right = shadowCamera->top = 2.2f;
        shadowCamera->nearPlane = 0.5f;
        shadowCamera->farPlane = 8.f;
        shadowCamera->updateProjectionMatrix();
        scene.add(sun);
    }

    // A ring, for marking the ball in the camera view.
    std::shared_ptr<Texture> createRingTexture() {
        constexpr int size = 64;
        std::vector<unsigned char> pixels(size * size * 4);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float d = std::hypot(static_cast<float>(x) + 0.5f - size / 2.f, static_cast<float>(y) + 0.5f - size / 2.f) / (size / 2.f);
                const float alpha = math::smoothstep(0.8f, 0.86f, d) * (1 - math::smoothstep(0.94f, 1.f, d));
                auto* p = &pixels[(y * size + x) * 4];
                p[0] = p[1] = p[2] = 255;
                p[3] = static_cast<unsigned char>(alpha * 255);
            }
        }

        auto texture = DataTexture::create(ImageData{std::move(pixels)}, size, size);
        texture->magFilter = texture->minFilter = Filter::Linear;
        texture->needsUpdate();

        return texture;
    }

    // The overlay: telemetry, key help, and the onboard camera's view with the
    // ball marked in it. All of it sits on layer 1, which the onboard camera
    // does not see.
    struct Hud {
        std::shared_ptr<TextSprite> values;
        std::shared_ptr<Sprite> marker;
        float scale;

        Hud(Scene& scene, const std::shared_ptr<Texture>& feed) {
            scale = monitor::contentScale().first;
            const float s = scale;

            FontLoader fontLoader;
            const auto font = fontLoader.defaultFont();

            const auto text = [&](const std::string& string, float size, const Color& color, float anchorX, float anchorY, float x, float y) {
                auto sprite = TextSprite::create(font, size * s, TextBox::Line);
                sprite->setText(string);
                sprite->setColor(color);
                sprite->setHorizontalAlignment(anchorX > 0.5f ? TextSprite::HorizontalAlignment::Right : TextSprite::HorizontalAlignment::Left);
                sprite->setVerticalAlignment(anchorY > 0.5f ? TextSprite::VerticalAlignment::Below : TextSprite::VerticalAlignment::Above);
                sprite->screenSpace = true;
                sprite->screenAnchor.set(anchorX, anchorY);
                sprite->position.set(x * s, y * s, 0);
                sprite->layers.set(1);
                scene.add(sprite);

                return sprite;
            };

            // A rectangle, placed by the corner that its anchor names.
            const auto quad = [&](const std::shared_ptr<SpriteMaterial>& material, float anchorX, float anchorY, float x, float y, float width, float height) {
                auto sprite = Sprite::create(material);
                sprite->center.set(anchorX, anchorY);
                sprite->screenSpace = true;
                sprite->screenAnchor.set(anchorX, anchorY);
                sprite->position.set(x * s, y * s, 0);
                sprite->scale.set(width * s, height * s, 1);
                sprite->layers.set(1);
                scene.add(sprite);

                return sprite;
            };

            // The panels the text is read against.
            auto panel = SpriteMaterial::create();
            panel->color = Color(0x0b0d10);
            panel->transparent = true;
            panel->opacity = 0.72f;
            panel->toneMapped = false;

            const Color dim(0x9fb0c0);
            quad(panel, 0, 1, 12, -12, 268, 122);
            text("mode\nheading\nspeed\nToF front\nToF rear\nball", 17, dim, 0, 1, 24, -22);
            values = text(" ", 17, Color::white, 0, 1, 116, -22);

            quad(panel, 1, 0, -12, 12, 348, 62);
            text("W/S drive   A/D turn   Space boost\n"
                 "R raw mode: W/S left track, Up/Down right track\n"
                 "F follow the ball   B ToF beams",
                 14, dim, 1, 0, -22, 22);

            quad(panel, 0, 0, 12, 12, feedWidth + 10, feedHeight + 34);
            text("onboard camera", 14, dim, 0, 0, 19, feedHeight + 24);

            auto feedMaterial = SpriteMaterial::create();
            feedMaterial->map = feed;
            feedMaterial->toneMapped = false;
            quad(feedMaterial, 0, 0, 17, 17, feedWidth, feedHeight);

            auto markerMaterial = SpriteMaterial::create();
            markerMaterial->map = createRingTexture();
            markerMaterial->color = Color(0x35ff6b);
            markerMaterial->transparent = true;
            markerMaterial->toneMapped = false;
            marker = quad(markerMaterial, 0, 0, 0, 0, 1, 1);
            marker->center.set(0.5f, 0.5f);
            marker->visible = false;
        }

        void update(const Sphero& sphero, const std::string& mode, const BallFollower::Detection* ball) {
            const auto range = [](float metres) {
                char buffer[16];
                std::snprintf(buffer, sizeof(buffer), "%.2f m", metres);
                return std::isfinite(metres) ? std::string(buffer) : std::string("--");
            };

            const auto [front, rear] = sphero.getTofMeasurements();
            char buffer[160];
            std::snprintf(buffer, sizeof(buffer), "%s\n%03d deg\n%+.2f m/s\n%s\n%s\n", mode.c_str(),
                          static_cast<int>(sphero.heading()) % 360, sphero.speed(), range(front).c_str(), range(rear).c_str());
            std::string string = buffer;
            if (ball && ball->found) {
                std::snprintf(buffer, sizeof(buffer), "%.2f m, %+.0f deg", ball->range, ball->bearing);
                string += buffer;
            } else {
                string += ball ? "not in view" : "--";
            }
            if (string != values->getText()) values->setText(string);

            marker->visible = ball && ball->found;
            if (marker->visible) {
                const float size = (ball->diameter + 14) * scale;
                marker->scale.set(size, size, 1);
                marker->position.set((17 + ball->x) * scale, (17 + static_cast<float>(feedHeight) - ball->y) * scale, 0);
            }
        }
    };

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, sequenceDir;
    bool following = false;
    float shotTime = 4;
    int sequenceFps = 30;
    int width = 1280, height = 720;
    Vector3 cameraOffset(-0.55f, 0.42f, 0.38f);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--sequence" && i + 1 < argc) sequenceDir = argv[++i];
        else if (a == "--fps" && i + 1 < argc) sequenceFps = std::atoi(argv[++i]);
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--follow") following = true;
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--cam" && i + 3 < argc) {
            cameraOffset.set(std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3]));
            i += 3;
        }
    }
    const bool headless = !shotPath.empty() || !sequenceDir.empty();
    if (!sequenceDir.empty()) std::filesystem::create_directories(sequenceDir);

    Canvas canvas(Canvas::Parameters()
                          .title("Sphero RVR")
                          .size(width, height)
                          .antialiasing(8)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    const Color backdrop(0x1a1e23);
    Scene scene;
    scene.background = backdrop;
    scene.fog = Fog(backdrop, 4, 16);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene.environment = pmrem.fromScene(room, 0.04f);
        room.dispose();
    }
    scene.environmentIntensity = 0.5f;

    Arena arena;
    setupArena(scene, arena);
    setupLights(scene);

    Ball ball;
    auto ballMaterial = matte(Color::white, 0.45f);
    ballMaterial->map = createBallTexture();
    ball.mesh = shadowed(Mesh::create(SphereGeometry::create(ballRadius, 48, 24), ballMaterial));
    ball.mesh->position.set(0.5f, ballRadius, 0.1f);
    scene.add(ball.mesh);

    Sphero sphero;
    sphero.position.set(-0.95f, 0, 0.25f);
    sphero.rotation.y = math::PI / 2;
    scene.addRef(sphero);

    // What the onboard camera sees, as the robot's own code gets it: pixels.
    RenderTarget::Options feedOptions;
    feedOptions.encoding = ColorSpace::sRGB;
    feedOptions.samples = 4;
    RenderTarget feed(feedWidth, feedHeight, feedOptions);

    Hud hud(scene, feed.texture);

    PerspectiveCamera camera(45, canvas.aspect(), 0.01f, 100);
    camera.position.copy(sphero.position).add(cameraOffset);
    camera.layers.enableAll();

    OrbitControls controls(camera, canvas);
    controls.enableKeys = false;
    controls.enablePan = false;
    controls.minDistance = 0.25f;
    controls.maxDistance = 6.f;
    controls.maxPolarAngle = math::PI / 2 - 0.03f;

    SpheroKeyController keys(sphero, canvas);
    keys.reset();
    BallFollower follower(sphero, 2 * ballRadius);

    bool beams = true;
    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::F) {
            following = !following;
            keys.reset();
        } else if (evt.key == Key::R) {
            following = false;
        } else if (evt.key == Key::B) {
            beams = !beams;
            sphero.showTofBeams(beams);
        }
    });

    canvas.onWindowResize([&](WindowSize newSize) {
        camera.aspect = newSize.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(newSize);
    });

    Clock clock;
    float time = 0, sinceVision = 1;
    int frame = 0, written = 0;
    Vector3 lastPosition = sphero.position;
    canvas.animate([&] {
        // a hitch must not throw the robot through a wall
        const float dt = headless ? 1.f / 60 : std::min(clock.getDelta(), 0.05f);
        time += dt;

        if (!following) keys.update(dt);
        sphero.update(dt);

        float nx, nz;
        arena.resolve(sphero.position.x, sphero.position.z, Sphero::footprintRadius, nx, nz);
        ball.update(dt, time, arena, sphero);

        // the camera rides along with the robot
        camera.position.add(Vector3().subVectors(sphero.position, lastPosition));
        lastPosition.copy(sphero.position);
        controls.target.set(sphero.position.x, 0.05f, sphero.position.z);
        controls.update();

        renderer.setRenderTarget(&feed);
        renderer.render(scene, sphero.camera());
        renderer.setRenderTarget(nullptr);

        // The robot's own loop runs at 10 Hz: a frame in, a drive command out.
        sinceVision += dt;
        if (sinceVision >= 0.1f) {
            sinceVision = 0;

            if (following) {
                renderer.copyTextureToImage(*feed.texture);
                const auto& pixels = feed.texture->image().data();
                follower.update(pixels.data(), feedWidth, feedHeight, static_cast<int>(pixels.size()) / (feedWidth * feedHeight));
            }

            sphero.setAllLeds(following ? Color(0x2bd9ff) : keys.raw() ? Color(0xffa51f)
                                                                       : Color(0x35ff6b));
            hud.update(sphero, following ? "follow the ball" : keys.raw() ? "raw"
                                                                         : "heading",
                       following ? &follower.detection() : nullptr);
        }

        renderer.render(scene, camera);

        if (!sequenceDir.empty() && frame++ % std::max(1, 60 / sequenceFps) == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04d.png", written++);
            renderer.writeFramebuffer(std::filesystem::path(sequenceDir) / name);
        }
        if (headless && time >= shotTime) {
            if (!shotPath.empty()) {
                renderer.writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << std::endl;
            }
            std::exit(0);
        }
    });
}
