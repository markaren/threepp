// Two spiral galaxies fall into each other. Every star is simulated and drawn
// on the GPU: four million of them by default, sixty-seven million on key 4.
//
// The physics is the restricted N-body method of Toomre & Toomre (1972), the
// one that first explained the tails of the Antennae and the Mice: the stars
// are massless test particles, each pulled by both galactic cores, and only the
// two cores pull on each other. That makes the cost linear in the star count,
// which is what lets it run on every star, every frame.
//
// How it is done with the GL renderer:
//
// - The state lives in two float textures, one texel per star: position and
//   velocity. A step is two full-screen passes that read one pair of textures
//   and write the other (kick the velocities, then drift the positions), so no
//   star ever crosses to the CPU. The starting state is written by a shader
//   too: an exponential disk with two trailing arms and a Plummer bulge, each
//   star close to the circular orbit of its own core.
// - The stars are one Points draw. Its "position" attribute is not a position
//   but the texel that holds the star; the vertex shader fetches the real one.
//   A star smaller than a pixel is drawn one pixel wide and dimmed by the area
//   it lost, so a distant galaxy keeps its brightness instead of gaining it.
// - Everything adds up in the composer's half-float target (a pixel of the core
//   holds thousands of stars), blooms there, and is tone-mapped once on the way
//   to the screen.
// - The two cores are integrated on the CPU. A drag that grows as they overlap
//   stands in for dynamical friction, so the pair merges after a few passes.
//
//   mouse    orbit / zoom          Space  pause        R  restart
//   1 2 3 4  1 / 4.2 / 16.8 / 67 million stars (the last one takes about 5 GB of video memory)
//   - =      slower / faster       [ ]    exposure     B  bloom
//   O        camera circles and frames the pair, on / off (the wheel takes the zoom over)
//
// Flags: --stars <texture side>  --uncapped (no vsync, to read the real frame time)
// Headless (dev): Galaxy --shot <prefix> [--time T] [--every E] [--size WxH] [--switch T <side>] [--film <every n frames>]

#include "Hud.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    const std::string hashGlsl = R"(
        // Dave Hoskins' hash without sine: stable across drivers, no state.
        vec4 hash42(vec2 p) {
            vec4 p4 = fract(vec4(p.xyxy) * vec4(.1031, .1030, .0973, .1099));
            p4 += dot(p4, p4.wzxy + 33.33);
            return fract((p4.xxyz + p4.yzzw) * p4.zywx);
        })";

    const std::string quadVertex = R"(
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        })";

    // Writes the starting state. uMode 0: position (w = a random seed),
    // uMode 1: velocity (w = what kind of star, and which galaxy it began in).
    const std::string initFragment = hashGlsl + R"(
        uniform float uMode;
        uniform float uSize;
        uniform float uSplit;
        uniform vec4 uCore0;
        uniform vec4 uCore1;// xyz centre, w = G * mass
        uniform vec3 uCoreVel0;
        uniform vec3 uCoreVel1;
        uniform mat4 uTilt0;
        uniform mat4 uTilt1;
        uniform vec3 uShape0;
        uniform vec3 uShape1;// disk scale length, softening, outer radius

        const float PI = 3.14159265;

        float circularSpeed(float r, float gm, float eps) {
            float s = r * r + eps * eps;
            return sqrt(gm * r * r / (s * sqrt(s)));
        }

        vec3 onSphere(vec2 u) {
            float y = 2.0 * u.x - 1.0;
            float s = sqrt(max(0.0, 1.0 - y * y));
            return vec3(s * cos(2.0 * PI * u.y), y, s * sin(2.0 * PI * u.y));
        }

        void main() {
            vec2 id = floor(gl_FragCoord.xy);
            vec4 a = hash42(id + 0.5);
            vec4 b = hash42(id + vec2(311.7, 127.1));
            vec4 c = hash42(id + vec2(74.7, 913.3));

            bool second = id.y >= floor(uSplit * uSize);
            vec4 core = second ? uCore1 : uCore0;
            vec3 coreVel = second ? uCoreVel1 : uCoreVel0;
            mat3 tilt = second ? mat3(uTilt1) : mat3(uTilt0);
            vec3 shape = second ? uShape1 : uShape0;
            float gm = core.w, scaleLength = shape.x, eps = shape.y, outer = shape.z;

            vec3 p, v;
            float kind;
            if (a.x < 0.14) {
                // The bulge: a Plummer sphere of old stars, each on a circular
                // orbit in a plane of its own.
                float u = clamp(a.y, 0.002, 0.998);
                float r = min(0.4 * eps / sqrt(pow(u, -2.0 / 3.0) - 1.0), 2.5 * eps);
                vec3 dir = onSphere(b.xy);
                vec3 tangent = normalize(cross(dir, onSphere(c.xy)) + 1e-5);
                p = dir * r;
                v = tangent * circularSpeed(r, gm, eps);
                kind = 2.0;
            } else {
                // The disk: surface density falls as exp(-r / scaleLength); the
                // sum of two exponential variates has exactly that radial law.
                float r = -scaleLength * log(max(a.y * a.z, 1e-7));
                if (r > outer) r = outer * sqrt(c.z);
                r = max(r, 0.02);

                float m = sqrt(-2.0 * log(max(b.x, 1e-6)));
                float g1 = m * cos(2.0 * PI * b.y), g2 = m * sin(2.0 * PI * b.y);

                float theta = 2.0 * PI * b.z;
                kind = 1.0;
                if (a.w < 0.62) {
                    // two trailing logarithmic arms, where the young stars are
                    theta = floor(b.z * 2.0) * PI - 1.6 * log(r / scaleLength + 0.25) + 0.3 * g1;
                    kind = 0.0;
                }

                p = vec3(r * cos(theta), 0.05 * scaleLength * g2, r * sin(theta));
                float speed = circularSpeed(r, gm, eps);
                // Some random motion on top of the rotation: without it the
                // arms shear into rings as sharp as the grooves of a record.
                vec3 dispersion = (vec3(c.x, c.y, a.w) - 0.5) * vec3(0.12, 0.05, 0.12);
                v = (vec3(-sin(theta), 0.0, cos(theta)) + dispersion) * speed;
            }

            p = core.xyz + tilt * p;
            v = coreVel + tilt * v;
            if (second) kind += 4.0;

            gl_FragColor = uMode < 0.5 ? vec4(p, c.w) : vec4(v, kind);
        })";

    // Kick: every star is accelerated by both cores (softened point masses).
    const std::string kickFragment = R"(
        uniform sampler2D tPos;
        uniform sampler2D tVel;
        uniform vec4 uCore0;
        uniform vec4 uCore1;
        uniform vec2 uSoftening2;
        uniform float uDt;

        vec3 pull(vec3 p, vec4 core, float eps2) {
            vec3 d = core.xyz - p;
            float r2 = dot(d, d) + eps2;
            return core.w * d / (r2 * sqrt(r2));
        }

        void main() {
            ivec2 texel = ivec2(gl_FragCoord.xy);
            vec4 p = texelFetch(tPos, texel, 0);
            vec4 v = texelFetch(tVel, texel, 0);
            vec3 acceleration = pull(p.xyz, uCore0, uSoftening2.x) + pull(p.xyz, uCore1, uSoftening2.y);
            gl_FragColor = vec4(v.xyz + acceleration * uDt, v.w);
        })";

    // Drift, with the velocity the kick just wrote: symplectic Euler, which
    // keeps a closed orbit closed over thousands of steps.
    const std::string driftFragment = R"(
        uniform sampler2D tPos;
        uniform sampler2D tVel;
        uniform float uDt;

        void main() {
            ivec2 texel = ivec2(gl_FragCoord.xy);
            vec4 p = texelFetch(tPos, texel, 0);
            vec4 v = texelFetch(tVel, texel, 0);
            gl_FragColor = vec4(p.xyz + v.xyz * uDt, p.w);
        })";

    const std::string starVertex = hashGlsl + R"(
        uniform sampler2D tPos;
        uniform sampler2D tVel;
        uniform float uPixelsPerUnit;
        uniform float uStarSize;
        uniform float uBrightness;

        varying vec3 vColor;

        void main() {
            // "position" is the texel that holds this star
            ivec2 texel = ivec2(position.xy);
            vec4 p = texelFetch(tPos, texel, 0);
            vec4 v = texelFetch(tVel, texel, 0);
            vec4 rnd = hash42(position.xy + vec2(17.3, 59.1));

            float kind = v.w;
            bool second = kind > 3.5;
            if (second) kind -= 4.0;

            // many dim stars and a few that outshine hundreds of them
            float luminosity = 0.3 + 12.0 * pow(rnd.x, 8.0);
            float size = 1.0;

            vec3 color;
            if (kind < 0.5) {
                color = mix(vec3(0.50, 0.68, 1.0), vec3(0.95, 0.96, 1.0), rnd.y);
                if (rnd.z < 0.02) {
                    // a star-forming cloud: wide, faint, hydrogen pink or dusty blue
                    color = rnd.w < 0.6 ? vec3(1.0, 0.22, 0.42) : vec3(0.30, 0.42, 1.0);
                    luminosity = 0.03;
                    size = 26.0;
                }
            } else if (kind < 1.5) {
                color = mix(vec3(1.0, 0.84, 0.64), vec3(0.80, 0.88, 1.0), rnd.y);
            } else {
                color = mix(vec3(1.0, 0.58, 0.28), vec3(1.0, 0.84, 0.58), rnd.y);
            }
            // each galaxy keeps a cast of its own, so the mixing can be followed
            color *= second ? vec3(1.0, 0.76, 0.56) : vec3(0.66, 0.84, 1.0);

            vec4 mvPosition = modelViewMatrix * vec4(p.xyz, 1.0);
            gl_Position = projectionMatrix * mvPosition;

            float ideal = uStarSize * size * uPixelsPerUnit / max(-mvPosition.z, 0.01);
            float drawn = clamp(ideal, 1.0, 48.0);
            gl_PointSize = drawn;

            float kept = min(1.0, (ideal * ideal) / (drawn * drawn));
            vColor = color * (luminosity * uBrightness * kept);
        })";

    const std::string starFragment = R"(
        varying vec3 vColor;

        void main() {
            vec2 c = gl_PointCoord * 2.0 - 1.0;
            float d2 = dot(c, c);
            float falloff = exp(-3.5 * d2) * (1.0 - smoothstep(0.7, 1.0, d2));
            gl_FragColor = vec4(vColor * falloff, 1.0);
        })";

    struct Galaxy {
        float gm;
        float scaleLength, softening, outer;
        Matrix4 tilt;
        Vector3 position, velocity;
    };

    std::shared_ptr<ShaderMaterial> passMaterial(const std::string& fragment) {

        auto material = ShaderMaterial::create();
        material->vertexShader = quadVertex;
        material->fragmentShader = fragment;
        material->depthTest = false;
        material->depthWrite = false;
        material->blending = Blending::None;

        return material;
    }

    // The whole simulation: the two cores on the CPU, the stars in textures.
    class Simulation {

    public:
        std::array<Galaxy, 2> galaxies;
        float time = 0;

        Simulation(GLRenderer& renderer, int size)
            : renderer_(renderer),
              init_(passMaterial(initFragment)),
              kick_(passMaterial(kickFragment)),
              drift_(passMaterial(driftFragment)),
              quad_(init_) {

            for (const auto name : {"uMode", "uSize", "uSplit", "uCore0", "uCore1", "uCoreVel0", "uCoreVel1", "uTilt0", "uTilt1", "uShape0", "uShape1"}) {
                init_->uniforms[name] = Uniform();
            }
            for (const auto name : {"tPos", "tVel", "uCore0", "uCore1", "uSoftening2", "uDt"}) {
                kick_->uniforms[name] = Uniform();
            }
            for (const auto name : {"tPos", "tVel", "uDt"}) {
                drift_->uniforms[name] = Uniform();
            }

            galaxies[0].gm = 1.f;
            galaxies[0].scaleLength = 0.5f;
            galaxies[0].softening = 0.35f;
            galaxies[0].outer = 2.6f;
            galaxies[0].tilt.makeRotationFromEuler(Euler(0.5f, 0, 0.3f));

            galaxies[1].gm = 0.8f;
            galaxies[1].scaleLength = 0.42f;
            galaxies[1].softening = 0.3f;
            galaxies[1].outer = 2.2f;
            galaxies[1].tilt.makeRotationFromEuler(Euler(-0.75f, 0, -0.45f));

            resize(size);
        }

        [[nodiscard]] int size() const { return size_; }
        [[nodiscard]] Texture* positions() const { return pos_[current_]->texture.get(); }
        [[nodiscard]] Texture* velocities() const { return vel_[current_]->texture.get(); }
        [[nodiscard]] float separation() const { return galaxies[0].position.distanceTo(galaxies[1].position); }

        void resize(int size) {

            size_ = size;

            RenderTarget::Options options;
            options.type = Type::Float;
            options.format = Format::RGBA;
            options.minFilter = Filter::Nearest;
            options.magFilter = Filter::Nearest;
            options.depthBuffer = false;

            for (int i = 0; i < 2; ++i) {
                pos_[i] = RenderTarget::create(size, size, options);
                vel_[i] = RenderTarget::create(size, size, options);
            }

            reset();
        }

        // Puts the pair on a slightly bound, nearly parabolic orbit about their
        // common centre, both disks turning the way the orbit does: the
        // prograde case, which is the one that throws long tails.
        void reset() {

            auto& [a, b] = galaxies;

            constexpr float distance = 9.f, pericentre = 1.25f;
            const float gm = a.gm + b.gm;
            const float speed = 0.96f * std::sqrt(2 * gm / distance);
            const float tangential = std::sqrt(2 * gm * pericentre) / distance;
            const float radial = -std::sqrt(std::max(0.f, speed * speed - tangential * tangential));

            const float fa = b.gm / gm, fb = a.gm / gm;
            a.position.set(-distance * fa, 0, 0);
            b.position.set(distance * fb, 0, 0);
            a.velocity.set(-radial * fa, 0, -tangential * fa);
            b.velocity.set(radial * fb, 0, tangential * fb);

            time = 0;
            current_ = 0;

            auto& u = init_->uniforms;
            u.at("uSize").setValue(static_cast<float>(size_));
            u.at("uSplit").setValue(a.gm / gm);
            u.at("uCore0").setValue(Vector4(a.position.x, a.position.y, a.position.z, a.gm));
            u.at("uCore1").setValue(Vector4(b.position.x, b.position.y, b.position.z, b.gm));
            u.at("uCoreVel0").setValue(a.velocity);
            u.at("uCoreVel1").setValue(b.velocity);
            u.at("uTilt0").setValue(a.tilt);
            u.at("uTilt1").setValue(b.tilt);
            u.at("uShape0").setValue(Vector3(a.scaleLength, a.softening, a.outer));
            u.at("uShape1").setValue(Vector3(b.scaleLength, b.softening, b.outer));

            quad_.setMaterial(init_);
            u.at("uMode").setValue(0.f);
            draw(*pos_[0]);
            u.at("uMode").setValue(1.f);
            draw(*vel_[0]);
            renderer_.setRenderTarget(nullptr);
        }

        void step(float dt) {

            auto& [a, b] = galaxies;
            const int next = 1 - current_;

            auto& k = kick_->uniforms;
            k.at("tPos").setValue(pos_[current_]->texture.get());
            k.at("tVel").setValue(vel_[current_]->texture.get());
            k.at("uCore0").setValue(Vector4(a.position.x, a.position.y, a.position.z, a.gm));
            k.at("uCore1").setValue(Vector4(b.position.x, b.position.y, b.position.z, b.gm));
            k.at("uSoftening2").setValue(Vector2(a.softening * a.softening, b.softening * b.softening));
            k.at("uDt").setValue(dt);
            quad_.setMaterial(kick_);
            draw(*vel_[next]);

            auto& d = drift_->uniforms;
            d.at("tPos").setValue(pos_[current_]->texture.get());
            d.at("tVel").setValue(vel_[next]->texture.get());
            d.at("uDt").setValue(dt);
            quad_.setMaterial(drift_);
            draw(*pos_[next]);

            renderer_.setRenderTarget(nullptr);
            current_ = next;

            // the cores: the same softened pull on each other, plus the drag
            Vector3 delta = b.position.clone().sub(a.position);
            const float r2 = delta.lengthSq() + a.softening * a.softening + b.softening * b.softening;
            const float inv = 1.f / (r2 * std::sqrt(r2));
            const float gm = a.gm + b.gm;

            Vector3 relative = b.velocity.clone().sub(a.velocity);
            const float drag = 0.16f * std::exp(-r2 / (1.4f * 1.4f));

            a.velocity.addScaledVector(delta, b.gm * inv * dt);
            b.velocity.addScaledVector(delta, -a.gm * inv * dt);
            a.velocity.addScaledVector(relative, drag * dt * b.gm / gm);
            b.velocity.addScaledVector(relative, -drag * dt * a.gm / gm);

            a.position.addScaledVector(a.velocity, dt);
            b.position.addScaledVector(b.velocity, dt);

            time += dt;
        }

    private:
        GLRenderer& renderer_;
        std::shared_ptr<ShaderMaterial> init_, kick_, drift_;
        FullScreenQuad quad_;
        std::unique_ptr<RenderTarget> pos_[2], vel_[2];
        int size_ = 0;
        int current_ = 0;

        void draw(RenderTarget& target) {
            renderer_.setRenderTarget(&target);
            quad_.render(renderer_);
        }
    };

    // One vertex per star, carrying the texel its state is stored in.
    std::shared_ptr<BufferGeometry> starGeometry(int size) {

        std::vector<float> texels(static_cast<size_t>(size) * size * 3);
        size_t i = 0;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                texels[i++] = static_cast<float>(x);
                texels[i++] = static_cast<float>(y);
                texels[i++] = 0;
            }
        }

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(std::move(texels), 3));

        return geometry;
    }

    std::string grouped(long long n) {

        std::string s = std::to_string(n);
        for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(i, " ");

        return s;
    }

    // The 2D layer: a status card in the top left corner and a strip of key
    // caps along the bottom. Sizes are pixels at 100 % display scaling, each
    // multiplied by the display's scale factor.
    class Overlay {

    public:
        Overlay(): s_(monitor::contentScale().first) {
            const float s = s_;

            // The order of the calls is the order of drawing: a card first, then what is on it.
            card_ = &hud_.panel(cardW * s, cardH * s, 13 * s, card, cardAlpha);
            cardEdge_ = &hud_.frame(cardW * s, cardH * s, 13 * s, 1.2f * s, edge, 0.10f);
            kicker_ = &hud_.text("GALAXY COLLISION, SIMULATED AND DRAWN ON THE GPU", 11.5f * s, accent, Hud::Face::Semibold);
            headline_ = &hud_.text(" ", 27 * s, ink, Hud::Face::Numeric);
            const char* names[] = {"FRAME", "FPS", "STAR STEPS / S", "TIME", "CORES APART"};
            const float xs[] = {0, 92, 160, 296, 372};
            for (size_t i = 0; i < tiles_.size(); ++i) {
                tiles_[i] = {&hud_.text(names[i], 10.5f * s, dim, Hud::Face::Semibold),
                             &hud_.text(" ", 17 * s, ink, Hud::Face::Numeric),
                             xs[i] * s};
            }

            keys_ = &hud_.panel(10, 10, 12 * s, card, cardAlpha);// its width is known once the hints are
            keysEdge_ = &hud_.frame(10, 10, 12 * s, 1.2f * s, edge, 0.10f);
            addHint("Mouse", "orbit");
            addHint("O", "circle");
            addHint("Space", "pause");
            addHint("R", "restart");
            addHint("1  2  3  4", "1 / 4 / 17 / 67 million stars");
            addHint("-  =", "time");
            addHint("[  ]", "exposure");
            addHint("B", "bloom");
            keysW_ = 6 * s;
            for (const Hint& hint : hints_) keysW_ += hint.slot;
            Hud::reshapePanel(*keys_, keysW_, keysH * s, 12 * s);
            Hud::reshapeFrame(*keysEdge_, keysW_, keysH * s, 12 * s, 1.2f * s);
        }

        // A text is placed by the left end of its baseline, a card by its bottom
        // left corner, and y counts up from the bottom of the window.
        void layout(int windowWidth, int windowHeight) {
            hud_.resize(windowWidth, windowHeight);
            const float s = s_;
            const float m = margin * s;
            const float top = static_cast<float>(windowHeight) - m;

            card_->position.set(m, top - cardH * s, 0);
            cardEdge_->position.copy(card_->position);
            const float left = m + 18 * s;
            kicker_->place(left, top - 26 * s);
            headline_->place(left, top - 58 * s);
            for (const Tile& tile : tiles_) {
                tile.name->place(left + tile.x, top - 84 * s);
                tile.value->place(left + tile.x, top - 105 * s);
            }

            keys_->position.set(m, m, 0);
            keysEdge_->position.copy(keys_->position);
            float x = m + 10 * s;
            for (const Hint& hint : hints_) {
                hint.cap->position.set(x, m + (keysH - capH) / 2 * s, 0);
                hint.capEdge->position.copy(hint.cap->position);
                hint.key->place(x + 7 * s, m + (keysH / 2 - 4) * s);
                hint.what->place(x + hint.capWidth + 7 * s, m + (keysH / 2 - 4.5f) * s);
                x += hint.slot;
            }
        }

        void show(const std::string& stars, float frameSeconds, double starSteps, float time, bool paused, float separation) {
            headline_->set(stars + " stars");

            char buffer[48];
            std::snprintf(buffer, sizeof(buffer), "%.1f ms", frameSeconds * 1000);
            tiles_[0].value->set(buffer);
            std::snprintf(buffer, sizeof(buffer), "%.0f", 1 / frameSeconds);
            tiles_[1].value->set(buffer);
            if (starSteps >= 1e9) std::snprintf(buffer, sizeof(buffer), "%.2f billion", starSteps / 1e9);
            else std::snprintf(buffer, sizeof(buffer), "%.0f million", starSteps / 1e6);
            tiles_[2].value->set(buffer);
            if (paused) {
                tiles_[3].value->set("paused");
            } else {
                std::snprintf(buffer, sizeof(buffer), "%.1f", time);
                tiles_[3].value->set(buffer);
            }
            std::snprintf(buffer, sizeof(buffer), "%.2f", separation);
            tiles_[4].value->set(buffer);
        }

        void draw(GLRenderer& renderer) { hud_.draw(renderer); }

    private:
        // The palette: one dark card colour, two greys for text, one accent.
        inline static const Color ink{0xf2f5fa};
        inline static const Color dim{0x9fb2cc};
        inline static const Color accent{0x19d3ff};
        inline static const Color card{0x0d131e};
        inline static const Color edge{0xffffff};
        static constexpr float cardAlpha = 0.80f;
        static constexpr float cardW = 500, cardH = 122, keysH = 38, capH = 22, margin = 16;

        struct Tile {
            Hud::Label* name;
            Hud::Label* value;
            float x;
        };
        struct Hint {
            Mesh* cap;
            Mesh* capEdge;
            Hud::Label* key;
            Hud::Label* what;
            float capWidth, slot;
        };

        Hud hud_;
        float s_;
        Mesh *card_, *cardEdge_, *keys_, *keysEdge_;
        Hud::Label *kicker_, *headline_;
        std::array<Tile, 5> tiles_{};
        std::vector<Hint> hints_;
        float keysW_ = 0;

        void addHint(const std::string& key, const std::string& what) {
            const float s = s_;
            Hint hint{};
            // The cap is created before the letter on it (drawing order), so its
            // width is measured, not taken from the label.
            hint.capWidth = hud_.measure(key, 11.5f * s, Hud::Face::Semibold) + 14 * s;
            hint.cap = &hud_.panel(hint.capWidth, capH * s, 5 * s, Color(0x2a3648), 1.f);
            hint.capEdge = &hud_.frame(hint.capWidth, capH * s, 5 * s, 1.f * s, edge, 0.22f);
            hint.key = &hud_.text(key, 11.5f * s, ink, Hud::Face::Semibold);
            hint.what = &hud_.text(what, 12.5f * s, dim);
            hint.slot = hint.capWidth + 7 * s + hud_.measure(what, 12.5f * s) + 20 * s;
            hints_.push_back(hint);
        }
    };

}// namespace


int main(int argc, char** argv) {

    std::string shotPrefix;
    float shotTime = 20, shotEvery = 0, exposure = 1, speed = 1.2f, switchTime = -1;
    int width = 1600, height = 900, size = 2048, switchSize = 0, filmEvery = 0;
    bool uncapped = false;
    Vector3 cameraPosition(0, 8.5f, 13.5f);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPrefix = argv[++i];
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--every" && i + 1 < argc) shotEvery = std::stof(argv[++i]);
        else if (a == "--exposure" && i + 1 < argc) exposure = std::stof(argv[++i]);
        else if (a == "--speed" && i + 1 < argc) speed = std::stof(argv[++i]);
        else if (a == "--stars" && i + 1 < argc) size = std::clamp(std::atoi(argv[++i]), 64, 8192);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--uncapped") uncapped = true;
        else if (a == "--film" && i + 1 < argc) filmEvery = std::atoi(argv[++i]);
        else if (a == "--switch" && i + 2 < argc) {
            switchTime = std::stof(argv[i + 1]);
            switchSize = std::atoi(argv[i + 2]);
            i += 2;
        }
        else if (a == "--cam" && i + 3 < argc) {
            cameraPosition.set(std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3]));
            i += 3;
        }
    }
    const bool headless = !shotPrefix.empty();

    Canvas canvas(Canvas::Parameters()
                          .title("Galaxy collision")
                          .size(width, height)
                          .antialiasing(0)
                          .vsync(!uncapped)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = exposure;

    Scene scene;
    scene.background = Color(0x000000);

    constexpr float fov = 45;
    PerspectiveCamera camera(fov, canvas.aspect(), 0.05f, 500.f);
    camera.position.copy(cameraPosition);
    camera.lookAt(Vector3{0, 0, 0});

    OrbitControls controls{camera, canvas};
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5f;

    Simulation sim(renderer, size);

    auto starMaterial = ShaderMaterial::create();
    starMaterial->vertexShader = starVertex;
    starMaterial->fragmentShader = starFragment;
    starMaterial->blending = Blending::Additive;
    starMaterial->transparent = true;
    starMaterial->depthTest = false;
    starMaterial->depthWrite = false;
    starMaterial->uniforms["tPos"] = Uniform();
    starMaterial->uniforms["tVel"] = Uniform();
    starMaterial->uniforms["uPixelsPerUnit"] = Uniform(1.f);
    starMaterial->uniforms["uStarSize"] = Uniform(0.0045f);
    starMaterial->uniforms["uBrightness"] = Uniform(1.f);

    auto stars = Points::create(starGeometry(size), starMaterial);
    stars->frustumCulled = false;
    scene.add(stars);

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(scene, camera));
    auto bloom = std::make_shared<UnrealBloomPass>(
            Vector2(static_cast<float>(canvas.size().width()), static_cast<float>(canvas.size().height())),
            0.3f, 0.2f, 1.2f);
    composer.addPass(bloom);

    Overlay overlay;
    overlay.layout(canvas.size().width(), canvas.size().height());

    bool paused = false;
    int pendingSize = 0;

    // Zooming by hand takes the framing over; O hands it back.
    bool framing = true;
    canvas.onMouseWheel([&](Vector2) { framing = false; });

    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::SPACE: paused = !paused; break;
            case Key::R: sim.reset(); break;
            case Key::NUM_1: pendingSize = 1024; break;
            case Key::NUM_2: pendingSize = 2048; break;
            case Key::NUM_3: pendingSize = 4096; break;
            case Key::NUM_4: pendingSize = 8192; break;
            case Key::MINUS: speed = std::max(0.15f, speed / 1.3f); break;
            case Key::EQUAL: speed = std::min(6.f, speed * 1.3f); break;
            case Key::LEFT_BRACKET: renderer.toneMappingExposure /= 1.2f; break;
            case Key::RIGHT_BRACKET: renderer.toneMappingExposure *= 1.2f; break;
            case Key::B: bloom->enabled = !bloom->enabled; break;
            case Key::O:
                controls.autoRotate = !controls.autoRotate;
                framing = controls.autoRotate;
                break;
            default: break;
        }
    });

    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(s);
        composer.setSize(s.width(), s.height());
        overlay.layout(s.width(), s.height());
    });

    Clock clock;
    float frameTime = 1.f / 60, nextShot = 0;
    int frame = 0, shots = 0, substeps = 1;
    double total = 0;
    canvas.animate([&] {
        const float elapsed = clock.getDelta();
        const float dt = headless ? 1.f / 60 : std::min(elapsed, 0.05f);
        frameTime += 0.05f * (elapsed - frameTime);
        if (frame > 10) total += elapsed;

        if (switchTime >= 0 && sim.time >= switchTime) {
            pendingSize = switchSize;
            switchTime = -1;
        }
        if (pendingSize != 0 && pendingSize != sim.size()) {
            sim.resize(pendingSize);
            stars->setGeometry(starGeometry(pendingSize));
        }
        pendingSize = 0;

        if (!paused) {
            const float simDt = dt * speed;
            substeps = std::clamp(static_cast<int>(std::ceil(simDt / 0.012f)), 1, 8);
            for (int i = 0; i < substeps; ++i) sim.step(simDt / static_cast<float>(substeps));
            // once the remnant has settled, start over
            if (!headless && sim.time > 72) sim.reset();
        }

        // The look is tuned at 2048 x 2048 stars; with more of them each one
        // carries less light, so the galaxies keep their brightness.
        const float perStar = (2048.f * 2048.f) / (static_cast<float>(sim.size()) * static_cast<float>(sim.size()));
        auto& u = starMaterial->uniforms;
        u.at("tPos").setValue(sim.positions());
        u.at("tVel").setValue(sim.velocities());
        u.at("uPixelsPerUnit").setValue(0.5f * static_cast<float>(canvas.size().height()) / std::tan(math::degToRad(fov) / 2));
        u.at("uBrightness").setValue(0.085f * perStar);

        // While the camera circles it also keeps the pair framed: far out for
        // the approach, close in once the cores have met.
        if (framing && controls.autoRotate) {
            const float wanted = std::clamp(5.f + 1.25f * sim.separation(), 6.5f, 17.f);
            const float distance = camera.position.length();
            camera.position.multiplyScalar(1 + (wanted / distance - 1) * (1 - std::exp(-0.5f * dt)));
        }
        controls.update();
        composer.render(dt);

        if (frame % 15 == 0) {
            const auto n = static_cast<long long>(sim.size()) * sim.size();
            overlay.show(grouped(n), frameTime, paused ? 0 : static_cast<double>(n) * substeps / frameTime, sim.time, paused, sim.separation());
        }
        overlay.draw(renderer);

        ++frame;
        if (headless) {
            // Reading a pixel back waits for the GPU: the frame time on the
            // card is then the frame's own, and not just the time it took to
            // queue it.
            unsigned char pixel[4];
            renderer.readPixels(Vector2(0, 0), {1, 1}, Format::RGBA, pixel);

            if (filmEvery > 0 ? frame % filmEvery == 0 : shotEvery > 0 && sim.time >= nextShot) {
                char name[32];
                std::snprintf(name, sizeof(name), "_%04d.png", shots++);
                renderer.writeFramebuffer(shotPrefix + name);
                nextShot += shotEvery;
                clock.getDelta();// writing the file is not part of the frame
            }
            if (sim.time >= shotTime) {
                if (shotEvery <= 0 && filmEvery <= 0) renderer.writeFramebuffer(shotPrefix + ".png");
                std::cout << "frames " << frame << "  mean frame " << 1000 * total / std::max(1, frame - 11) << " ms" << std::endl;
                std::exit(0);
            }
        }
    });
}
