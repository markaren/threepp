// A sand table: a box of sand and water to play in. Pile sand up, dig it out,
// pour water on it, and the water runs downhill, pools, spills and cuts its own
// channels. It starts with a mountain that holds a crater lake: dig through the
// rim and the lake drains across the plain.
//
// The physics is in SandTable.hpp. Here the two height fields become two
// meshes that are rewritten every frame, and the water is a transmissive
// surface whose depth is handed to the material as a texture: the deeper the
// water, the more of the light it takes, so shallows are clear and the lake is
// blue-green, and where it runs white it stops being transparent at all.
//
// The tools and the switches are buttons under the readout, each with its key
// on it and an icon that is an SVG path written out in this file.
//
//   left mouse     use the tool where the pointer is, or press a button
//   right mouse    orbit        wheel: closer / further
//   1 2 3 4        the tool: pile sand, dig, pour water, soak water up
//   Up / Down      a larger / smaller brush
//   T              the tap over the crater on / off
//   D              the overflow at the sea's end open / shut
//   R              rain on / off
//   E              erosion on / off: whether running water takes sand with it
//   B              the balls back to the lake
//   Space          pause
//   Backspace      start over
//
// Headless (dev):
//   SandTable --shot out.png [--time T] [--orbit AZ POLAR DISTANCE] [--size WxH] [--demo] [--tap] [--drain] [--rain] [--plain]
//   SandTable --sequence <dir> [--fps F] ...
//   (--demo digs the lake open by itself, through the same brush as the mouse;
//   --orbit is in degrees and metres; --plain turns the post chain off)

#include "threepp/threepp.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/input/MouseListener.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/textures/DataTexture.hpp"

#include "SandTable.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;
using namespace sandtable;

namespace {

    constexpr float baseTop = 0.05f;   // the slab the box stands on [m]
    constexpr float deepest = 0.1f;    // the depth at which the water's texture runs out of range [m]

    template<class T>
    std::shared_ptr<T> shadowed(std::shared_ptr<T> object) {
        object->castShadow = true;
        object->receiveShadow = true;

        return object;
    }

    // Fine grain, so that the sand is not one flat colour up close.
    std::shared_ptr<Texture> createGrainTexture() {
        constexpr int size = 256;

        std::vector<unsigned char> pixels(size * size * 4);
        unsigned int state = 12345u;
        for (int k = 0; k < size * size; ++k) {
            state = state * 1664525u + 1013904223u;
            const auto grain = static_cast<unsigned char>(205 + (state >> 24) % 50);
            pixels[k * 4] = pixels[k * 4 + 1] = pixels[k * 4 + 2] = grain;
            pixels[k * 4 + 3] = 255;
        }

        auto texture = DataTexture::create(ImageData{std::move(pixels)}, size, size);
        texture->magFilter = Filter::Linear;
        texture->minFilter = Filter::LinearMipmapLinear;
        texture->generateMipmaps = true;
        texture->wrapS = texture->wrapT = TextureWrapping::Repeat;
        texture->repeat.set(24, 16);
        texture->colorSpace = ColorSpace::sRGB;
        texture->needsUpdate();

        return texture;
    }

    // The sand and the water, as two meshes over the same grid.
    class Surface {

    public:
        std::shared_ptr<Mesh> sand, water;

        Surface(): level_(columns * rows), damp_(columns * rows), known_(columns * rows) {
            constexpr int count = columns * rows;

            std::vector<float> positions(count * 3), normals(count * 3), uvs(count * 2);
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    positions[c * 3] = xOf(i);
                    positions[c * 3 + 2] = zOf(j);
                    normals[c * 3 + 1] = 1;
                    // the middle of the texel that belongs to this cell
                    uvs[c * 2] = (static_cast<float>(i) + 0.5f) / columns;
                    uvs[c * 2 + 1] = (static_cast<float>(j) + 0.5f) / rows;
                }
            }
            std::vector<unsigned int> indices;
            indices.reserve((columns - 1) * (rows - 1) * 6);
            for (int j = 0; j < rows - 1; ++j) {
                for (int i = 0; i < columns - 1; ++i) {
                    const unsigned int a = cellIndex(i, j), b = cellIndex(i, j + 1), c = cellIndex(i + 1, j), d = cellIndex(i + 1, j + 1);
                    indices.insert(indices.end(), {a, b, c, b, d, c});
                }
            }

            const auto grid = [&](bool colored) {
                auto geometry = BufferGeometry::create();
                geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
                geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
                geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
                if (colored) geometry->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(count * 3, 1.f), 3));
                geometry->setIndex(indices);
                for (const auto& name : {"position", "normal", "color"}) {
                    if (geometry->hasAttribute(name)) geometry->getAttribute(name)->setUsage(DrawUsage::Dynamic);
                }
                return geometry;
            };

            auto sandMaterial = MeshStandardMaterial::create();
            sandMaterial->vertexColors = true;
            sandMaterial->roughness = 0.95f;
            sandMaterial->map = createGrainTexture();
            sand = shadowed(Mesh::create(grid(true), sandMaterial));
            sand->frustumCulled = false;

            // Red: how much of the light gets through (none, where it foams).
            // Green: the depth, as a fraction of `deepest`. The material reads
            // its transmission and its thickness from those two channels.
            depths_ = DataTexture::create(ImageData{std::vector<unsigned char>(count * 4, 255)}, columns, rows);
            depths_->magFilter = depths_->minFilter = Filter::Linear;
            depths_->needsUpdate();

            auto waterMaterial = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                                      .color(Color(0xf4fbff))
                                                                      .roughness(0.05f)
                                                                      .transmission(1.f)
                                                                      .ior(1.33f)
                                                                      .thickness(deepest)
                                                                      .attenuationColor(Color(0x1f8fc0))
                                                                      .attenuationDistance(0.02f));
            waterMaterial->transmissionMap = depths_;
            waterMaterial->thicknessMap = depths_;
            water = Mesh::create(grid(false), waterMaterial);
            water->frustumCulled = false;
        }

        void refresh(const Simulation& simulation) {
            refreshSand(simulation);
            refreshWater(simulation);
        }

    private:
        std::shared_ptr<DataTexture> depths_;
        std::vector<float> level_, damp_;
        std::vector<unsigned char> known_;

        // The normals of a height field, from its neighbours on either side.
        static void writeNormals(const std::vector<float>& heights, std::vector<float>& normals) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = cellIndex(i, j);
                    const float dx = (heights[cellIndex(std::min(i + 1, columns - 1), j)] - heights[cellIndex(std::max(i - 1, 0), j)]) / (2 * cellSize);
                    const float dz = (heights[cellIndex(i, std::min(j + 1, rows - 1))] - heights[cellIndex(i, std::max(j - 1, 0))]) / (2 * cellSize);
                    const float scale = 1 / std::sqrt(dx * dx + dz * dz + 1);
                    normals[c * 3] = -dx * scale;
                    normals[c * 3 + 1] = scale;
                    normals[c * 3 + 2] = -dz * scale;
                }
            }
        }

        void refreshSand(const Simulation& simulation) {
            static const Color dry(0xd9c6a2), wet(0xa08a62), floor(0x2b6cb0);

            auto& geometry = *sand->geometry();
            auto& positions = geometry.getAttribute<float>("position")->array();
            auto& colors = geometry.getAttribute<float>("color")->array();

            // The damp spreads a little way into the sand around the water:
            // the wetness, averaged over five cells each way.
            constexpr int reach = 2;
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    float sum = 0;
                    for (int k = -reach; k <= reach; ++k) sum += simulation.wetness[cellIndex(std::clamp(i + k, 0, columns - 1), j)];
                    level_[cellIndex(i, j)] = sum / (2 * reach + 1);
                }
            }
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    float sum = 0;
                    for (int k = -reach; k <= reach; ++k) sum += level_[cellIndex(i, std::clamp(j + k, 0, rows - 1))];
                    damp_[cellIndex(i, j)] = sum / (2 * reach + 1);
                }
            }

            for (int c = 0; c < columns * rows; ++c) {
                const float height = simulation.ground[c];
                positions[c * 3 + 1] = height;

                // dark where it is or has been wet, and the blue floor of the box where it is dug away
                Color color = dry;
                color.lerp(wet, damp_[c]);
                color.lerp(floor, 1 - math::smoothstep(0.0008f, 0.004f, height));
                colors[c * 3] = color.r;
                colors[c * 3 + 1] = color.g;
                colors[c * 3 + 2] = color.b;
            }
            writeNormals(simulation.ground, geometry.getAttribute<float>("normal")->array());

            for (const auto& name : {"position", "normal", "color"}) geometry.getAttribute(name)->needsUpdate();
        }

        void refreshWater(const Simulation& simulation) {
            constexpr int count = columns * rows;
            constexpr float wetEnough = 0.0001f;
            constexpr unsigned char rings = 4;

            // Where there is water, its surface. Around it, that same level
            // carried a few cells on into the dry sand, so that the water's
            // edge is where a level surface meets the shore, and flat right up
            // to it. (A sheet that stopped at the last wet cell would end in a
            // staircase, a cell at a time, wherever the beach is nearly flat.)
            for (int c = 0; c < count; ++c) {
                known_[c] = simulation.water[c] > wetEnough;
                level_[c] = known_[c] ? simulation.ground[c] + simulation.water[c] : simulation.ground[c] - 0.004f;
            }
            for (unsigned char ring = 1; ring <= rings; ++ring) {
                for (int j = 0; j < rows; ++j) {
                    for (int i = 0; i < columns; ++i) {
                        const int c = cellIndex(i, j);
                        if (known_[c]) continue;

                        float sum = 0;
                        int found = 0;
                        const auto look = [&](int n) {
                            if (known_[n] && known_[n] <= ring) {
                                sum += level_[n];
                                ++found;
                            }
                        };
                        if (i > 0) look(c - 1);
                        if (i < columns - 1) look(c + 1);
                        if (j > 0) look(c - columns);
                        if (j < rows - 1) look(c + columns);
                        if (found) {
                            // Up a shore the level carries on, out of sight
                            // under the sand. Down a slope there is nothing
                            // to hold it up: the sheet comes down to the sand
                            // and ends there, where the water does.
                            const float carried = sum / static_cast<float>(found);
                            const float sandHere = simulation.ground[c];
                            level_[c] = sandHere >= carried - 0.0005f ? carried : sandHere - 0.0004f;
                            known_[c] = ring + 1;// not seen by the rest of this pass
                        }
                    }
                }
            }

            auto& geometry = *water->geometry();
            auto& positions = geometry.getAttribute<float>("position")->array();
            auto& pixels = depths_->image().data();
            for (int c = 0; c < count; ++c) {
                positions[c * 3 + 1] = level_[c];

                const float depth = std::clamp((level_[c] - simulation.ground[c]) / deepest, 0.f, 1.f);
                pixels[c * 4] = static_cast<unsigned char>(255 * (1 - simulation.foam[c]));
                pixels[c * 4 + 1] = static_cast<unsigned char>(255 * depth);
            }
            writeNormals(level_, geometry.getAttribute<float>("normal")->array());

            geometry.getAttribute("position")->needsUpdate();
            geometry.getAttribute("normal")->needsUpdate();
            depths_->needsUpdate();
        }
    };

    // The box the sand is in, and the slab it stands on.
    void createBox(Object3D& table, Scene& scene) {
        constexpr float thickness = 0.022f, overlap = 0.0015f;
        constexpr float top = wallHeight + 0.012f, bottom = -0.02f;

        auto wood = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                         .color(Color(0x4a2d18))
                                                         .roughness(0.55f)
                                                         .clearcoat(0.35f)
                                                         .clearcoatRoughness(0.25f));

        const float height = top - bottom, middle = (top + bottom) / 2;
        for (const float side : {1.f, -1.f}) {
            auto alongX = shadowed(Mesh::create(BoxGeometry::create(2 * (halfWidth + thickness), height, thickness), wood));
            alongX->position.set(0, middle, side * (halfDepth - overlap + thickness / 2));
            table.add(alongX);

            auto alongZ = shadowed(Mesh::create(BoxGeometry::create(thickness, height, 2 * halfDepth), wood));
            alongZ->position.set(side * (halfWidth - overlap + thickness / 2), middle, 0);
            table.add(alongZ);
        }
        auto floor = shadowed(Mesh::create(BoxGeometry::create(2 * (halfWidth + thickness), 0.018f, 2 * (halfDepth + thickness)), wood));
        floor->position.y = -0.0095f;
        table.add(floor);

        auto lacquer = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                                                            .color(Color(0x0c0d10))
                                                            .roughness(0.55f)
                                                            .clearcoat(1.f)
                                                            .clearcoatRoughness(0.08f));
        auto slab = shadowed(Mesh::create(BoxGeometry::create(2 * halfWidth + 0.3f, baseTop - 0.02f, 2 * halfDepth + 0.3f), lacquer));
        slab->position.y = (baseTop - 0.02f) / 2;
        scene.add(slab);
    }

    // The tap over the crater and the overflow in the wall at the sea's end.
    class Plumbing {

    public:
        explicit Plumbing(Object3D& table) {
            auto brass = MeshStandardMaterial::create();
            brass->color = Color(0xc9953a);
            brass->metalness = 1;
            brass->roughness = 0.3f;

            const auto pipe = [&](const std::vector<Vector3>& line, float radius) {
                table.add(shadowed(Mesh::create(TubeGeometry::create(std::make_shared<CatmullRomCurve3>(line), 96, radius, 14), brass)));
            };

            // The tap: up beside the box, and over in an arch to above the lake.
            constexpr float bore = 0.008f, shoulder = 0.2f, rise = 0.085f;
            const float foot = -halfWidth - 0.05f, reach = (tapX - foot) / 2;
            std::vector<Vector3> neck;
            for (int k = 0; k <= 8; ++k) neck.emplace_back(foot, math::lerp(-0.02f, shoulder, static_cast<float>(k) / 8), tapZ);
            for (int k = 1; k <= 24; ++k) {
                const float a = math::PI * (1 - static_cast<float>(k) / 24);
                neck.emplace_back(foot + reach + reach * std::cos(a), shoulder + rise * std::sin(a), tapZ);
            }
            for (int k = 1; k <= 3; ++k) neck.emplace_back(tapX, shoulder - 0.012f * static_cast<float>(k), tapZ);
            mouth_ = neck.back().y;
            pipe(neck, bore);

            auto nozzle = shadowed(Mesh::create(CylinderGeometry::create(bore * 1.4f, bore * 1.4f, 0.014f, 24), brass));
            nozzle->position.set(tapX, mouth_ + 0.007f, tapZ);
            table.add(nozzle);

            auto valve = shadowed(Mesh::create(CylinderGeometry::create(0.014f, 0.014f, 0.032f, 24), brass));
            valve->position.set(foot, 0.09f, tapZ);
            table.add(valve);

            // its lever: across the pipe when shut, along it when open
            lever_ = shadowed(Mesh::create(BoxGeometry::create(0.007f, 0.007f, 0.07f), brass));
            lever_->position.set(foot - 0.02f, 0.09f, tapZ);
            table.add(lever_);

            auto falling = MeshStandardMaterial::create();
            falling->color = Color(0xd4efff);
            falling->roughness = 0.08f;
            falling->transparent = true;
            falling->opacity = 0.6f;
            stream_ = Mesh::create(CylinderGeometry::create(0.0045f, 0.006f, 1.f, 14), falling);
            stream_->visible = false;
            table.add(stream_);

            // The overflow: a port in the wall, the gate that slides over it,
            // and the pipe that takes the water away under the table.
            const float wall = halfWidth - 0.0015f;// the wall's inner face
            const float centre = drainSill + portRadius;

            auto rim = shadowed(Mesh::create(TorusGeometry::create(portRadius, 0.005f, 10, 48), brass));
            rim->rotation.y = math::PI / 2;
            rim->position.set(wall - 0.002f, centre, drainZ);
            table.add(rim);

            auto dark = MeshBasicMaterial::create();
            dark->color = Color(0x050505);
            auto hole = Mesh::create(CircleGeometry::create(portRadius, 40), dark);
            hole->rotation.y = -math::PI / 2;
            hole->position.set(wall - 0.0006f, centre, drainZ);
            table.add(hole);

            gate_ = shadowed(Mesh::create(CylinderGeometry::create(portRadius * 1.04f, portRadius * 1.04f, 0.003f, 40), brass));
            gate_->rotation.z = math::PI / 2;
            gate_->position.set(wall - 0.003f, centre, drainZ);
            table.add(gate_);

            const float outside = halfWidth + 0.02f;
            pipe({{outside, centre, drainZ}, {outside + 0.03f, centre, drainZ}, {outside + 0.052f, centre - 0.022f, drainZ}, {outside + 0.052f, -0.02f, drainZ}}, 0.013f);
        }

        void update(const Simulation& simulation, float dt) {
            const float ease = std::min(1.f, 8.f * dt);
            tapOpen_ += ((simulation.tap ? 1.f : 0.f) - tapOpen_) * ease;
            drainOpen_ += ((simulation.drain ? 1.f : 0.f) - drainOpen_) * ease;

            lever_->rotation.x = tapOpen_ * math::PI / 2;
            gate_->position.y = drainSill + portRadius * (1 + 2.3f * drainOpen_);

            // the falling water, from the mouth down to whatever is under it
            stream_->visible = simulation.tap;
            const float foot = simulation.groundAt(tapX, tapZ) + simulation.waterAt(tapX, tapZ);
            stream_->scale.y = std::max(0.001f, mouth_ - foot);
            stream_->position.set(tapX, (mouth_ + foot) / 2, tapZ);
        }

    private:
        static constexpr float portRadius = 0.03f;

        std::shared_ptr<Mesh> lever_, gate_, stream_;
        float mouth_ = 0;
        float tapOpen_ = 0, drainOpen_ = 0;
    };

    void setupStudio(Scene& scene) {

        auto material = MeshStandardMaterial::create();
        material->color = Color(0x1c2025);
        material->roughness = 0.82f;

        auto floor = Mesh::create(PlaneGeometry::create(40, 40), material);
        floor->rotation.x = -math::PI / 2;
        floor->receiveShadow = true;
        scene.add(floor);

        scene.add(HemisphereLight::create(0xdfe9ff, 0x2a2d33, 0.2f));

        // low, from one side: relief only shows in raking light
        auto key = DirectionalLight::create(0xfff1df, 1.7f);
        key->position.set(-1.5f, 0.95f, 0.75f);
        key->castShadow = true;
        key->shadow->mapSize.set(2048, 2048);
        key->shadow->bias = -0.0006f;
        key->shadow->radius = 2.5f;
        auto shadowCamera = key->shadow->camera->as<OrthographicCamera>();
        shadowCamera->left = shadowCamera->bottom = -0.85f;
        shadowCamera->right = shadowCamera->top = 0.85f;
        shadowCamera->nearPlane = 0.3f;
        shadowCamera->farPlane = 5.f;
        shadowCamera->updateProjectionMatrix();
        scene.add(key);

        auto rim = DirectionalLight::create(0x9fc4ff, 0.5f);
        rim->position.set(1.2f, 0.9f, -1.3f);
        scene.add(rim);
    }

    // The camera goes round a point above the table. (The left button is the
    // tool's, so this is not OrbitControls.)
    struct Orbit {
        float azimuth = 0.3f, polar = 0.86f, distance = 1.6f;
        Vector3 target{0.02f, baseTop + 0.06f, 0.03f};

        void apply(Camera& camera) const {
            camera.position.set(target.x + distance * std::sin(polar) * std::sin(azimuth),
                                target.y + distance * std::cos(polar),
                                target.z + distance * std::sin(polar) * std::cos(azimuth));
            camera.lookAt(target);
        }
    };

    struct Pointer: MouseListener {
        Vector2 position{-1, -1};
        Vector2 dragged;
        float wheel = 0;
        bool left = false, right = false;
        bool pressed = false;// the left button went down since this was last cleared

        void onMouseDown(int button, const Vector2& pos) override {
            position = pos;
            if (button == 0) left = pressed = true;
            if (button == 1) right = true;
        }

        void onMouseUp(int button, const Vector2& pos) override {
            position = pos;
            if (button == 0) left = false;
            if (button == 1) right = false;
        }

        void onMouseMove(const Vector2& pos) override {
            if (right) dragged.add(Vector2().subVectors(pos, position));
            position = pos;
        }

        void onMouseWheel(const Vector2& delta) override {
            wheel += delta.y;
        }
    };

    struct Tool {
        const char* name;
        bool water;
        float rate;// height at the middle of the brush [m/s]
    };
    constexpr std::array<Tool, 4> tools{{{"pile sand", false, 0.05f},
                                         {"dig", false, -0.09f},
                                         {"pour water", true, 0.035f},
                                         {"soak up", true, -0.09f}}};

    // The icons on the buttons: line drawings on a square of 24 units, each
    // written out as the path of an SVG and read by SVGLoader.
    namespace icons {
        constexpr const char* pile = "M3 20 C7 20 8 13 12 13 C16 13 17 20 21 20 M12 9 V3 M9 6 L12 3 L15 6";
        constexpr const char* dig = "M3 9 H7 C8.5 9 8 19 12 19 C16 19 15.5 9 17 9 H21 M12 3 V13 M9 10 L12 13 L15 10";
        constexpr const char* pour = "M12 3 C9 7 6 10.5 6 14.5 C6 18 8.7 21 12 21 C15.3 21 18 18 18 14.5 C18 10.5 15 7 12 3 Z";
        constexpr const char* soak = "M5 10 H19 C20.1 10 21 10.9 21 12 V18 C21 19.1 20.1 20 19 20 H5 C3.9 20 3 19.1 3 18 V12 C3 10.9 3.9 10 5 10 Z "
                                     "M7.5 14 H8.5 M12 16.5 H13 M16 14 H17 M12 7 V2.5 M9.5 5 L12 2.5 L14.5 5";
        constexpr const char* tap = "M3 10 H14 C16.2 10 18 11.8 18 14 V16 M9 10 V6 M6 6 H12 M18 19.5 V21";
        constexpr const char* drain = "M12 4 C16.4 4 20 7.6 20 12 C20 16.4 16.4 20 12 20 C7.6 20 4 16.4 4 12 C4 7.6 7.6 4 12 4 Z M9 9 H15 M8 12 H16 M9 15 H15";
        constexpr const char* rain = "M7 15 C4.6 15 3 13.4 3 11.4 C3 9.4 4.6 7.8 6.6 7.8 C7.3 5.6 9.3 4 11.8 4 C14.9 4 17.4 6.4 17.5 9.4 "
                                     "C19.5 9.6 21 10.8 21 12.4 C21 13.9 19.8 15 18 15 Z M8 18 L7 21 M12 18 L11 21 M16 18 L15 21";
        constexpr const char* erosion = "M3 8 C5 6 7 6 9 8 C11 10 13 10 15 8 C17 6 19 6 21 8 M3 13 C5 11 7 11 9 13 C11 15 13 15 15 13 C17 11 19 11 21 13 "
                                        "M5.5 18.5 H6.5 M11.5 19 H12.5 M17.5 18.5 H18.5";
    }// namespace icons

    // What is drawn over the finished frame: the readout, the buttons with
    // their icons, the help, and the ring that shows where the brush is.
    class Hud {

    public:
        std::shared_ptr<TextSprite> values;
        std::shared_ptr<Mesh> ring;

        Hud(): scale_(monitor::contentScale().first), font_(FontLoader().defaultFont()) {

            plate(0, 1, 12, -12, readoutWidth, readoutHeight);
            text("tool\nbrush\nwater\nphysics\nframe", 17, dim_, 0, 1, 24, -22);
            values = text(" ", 17, Color::white, 0, 1, 106, -22);

            plate(1, 0, -12, 12, 420, 44);
            text("left mouse: use the tool   right mouse: orbit   wheel: zoom\n"
                 "Up / Down brush   B balls   Space pause   Backspace start over",
                 14, dim_, 1, 0, -22, 22);

            auto ringMaterial = MeshBasicMaterial::create();
            ringMaterial->color = Color(0xffffff);
            ringMaterial->transparent = true;
            ringMaterial->opacity = 0.85f;
            ringMaterial->depthTest = false;
            ringMaterial->toneMapped = false;
            ring = Mesh::create(TorusGeometry::create(1.f, 0.025f, 6, 64), ringMaterial);
            ring->rotation.x = math::PI / 2;
            ring->visible = false;
            scene_.add(ring);

            pixels_.position.z = 5;
        }

        // A button in the grid under the readout. `art` is one of the icons,
        // `active` says whether to show it lit, `press` is what a click on it does.
        void addButton(int row, int column, const std::string& key, const std::string& art, std::function<bool()> active, std::function<void()> press) {
            Button button;
            button.x = 12 + static_cast<float>(column) * (buttonSize + buttonGap);
            button.y = 12 + readoutHeight + 8 + static_cast<float>(row) * (buttonSize + buttonGap);
            button.active = std::move(active);
            button.press = std::move(press);

            button.plate = plate(0, 1, button.x, -button.y, buttonSize, buttonSize);
            button.key = text(key, 11, dim_, 0, 1, button.x + 5, -(button.y + 3));

            // The icon: SVGLoader turns each stroke of the path into a strip of triangles.
            button.ink = MeshBasicMaterial::create();
            button.ink->side = Side::Double;
            button.ink->depthTest = false;
            button.ink->toneMapped = false;
            button.icon = Group::create();
            const std::string svg = R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="none" stroke="#ffffff" stroke-width="2" )"
                                    R"(stroke-linecap="round" stroke-linejoin="round" d=")" +
                                    art + R"("/></svg>)";
            SVGLoader loader;
            for (const auto& data : loader.parse(svg)) {
                for (const auto& subPath : data.path.subPaths) {
                    if (auto geometry = SVGLoader::pointsToStroke(subPath->getPoints(), data.style)) {
                        button.icon->add(Mesh::create(geometry, button.ink));
                    }
                }
            }
            icons_.add(button.icon);

            buttons_.push_back(std::move(button));
        }

        // Puts the icons where their buttons are. They are drawn by a camera
        // that counts in pixels from the window's lower left corner.
        void layout(WindowSize size) {
            height_ = static_cast<float>(size.height());
            pixels_.left = pixels_.bottom = 0;
            pixels_.right = static_cast<float>(size.width());
            pixels_.top = height_;
            pixels_.updateProjectionMatrix();

            const float inset = (buttonSize - iconSize) / 2;
            for (auto& button : buttons_) {
                // an SVG counts its y downwards, hence the flip
                button.icon->scale.set(iconSize * scale_ / 24, -iconSize * scale_ / 24, 1);
                button.icon->position.set((button.x + inset) * scale_, height_ - (button.y + inset + 3) * scale_, 0);
            }
        }

        // Lights the buttons that are on.
        void update() {
            for (auto& button : buttons_) {
                const bool on = button.active();
                button.plate->color = on ? Color(0x2f7fb8) : Color(0x0b0d10);
                button.plate->opacity = on ? 0.92f : 0.72f;
                button.ink->color = on ? Color(0xffffff) : dim_;
                button.key->setColor(on ? Color(0xffffff) : dim_);
            }
        }

        // Whether a pixel of the window, counted from its upper left, is on a button.
        [[nodiscard]] bool covers(const Vector2& pixel) const {
            return find(pixel) != nullptr;
        }

        [[nodiscard]] size_t buttonCount() const {
            return buttons_.size();
        }

        // The middle of a button, as a pixel of the window counted from its upper left.
        [[nodiscard]] Vector2 buttonCentre(size_t index) const {
            const auto& button = buttons_[index];
            return {(button.x + buttonSize / 2) * scale_, (button.y + buttonSize / 2) * scale_};
        }

        // Presses the button under a pixel, if there is one.
        bool click(const Vector2& pixel) {
            const auto* button = find(pixel);
            if (button) button->press();

            return button != nullptr;
        }

        void render(GLRenderer& renderer, Camera& camera) {
            renderer.autoClear = false;
            renderer.clearDepth();
            renderer.render(scene_, camera);
            renderer.render(icons_, pixels_);
            renderer.autoClear = true;
        }

    private:
        static constexpr float readoutWidth = 202, readoutHeight = 106;
        static constexpr float buttonSize = 46, buttonGap = 6, iconSize = 28;

        struct Button {
            float x = 0, y = 0;// its upper left corner, from the window's
            std::shared_ptr<SpriteMaterial> plate;
            std::shared_ptr<MeshBasicMaterial> ink;
            std::shared_ptr<TextSprite> key;
            std::shared_ptr<Group> icon;
            std::function<bool()> active;
            std::function<void()> press;
        };

        float scale_;
        Font font_;
        const Color dim_{0x9fb0c0};
        Scene scene_; // text, plates and the ring, through the scene's own camera
        Scene icons_;
        OrthographicCamera pixels_{0, 1, 1, 0, 0.1f, 10};
        float height_ = 1;
        std::vector<Button> buttons_;

        [[nodiscard]] const Button* find(const Vector2& pixel) const {
            for (const auto& button : buttons_) {
                const float x = pixel.x / scale_ - button.x, y = pixel.y / scale_ - button.y;
                if (x >= 0 && x <= buttonSize && y >= 0 && y <= buttonSize) return &button;
            }

            return nullptr;
        }

        std::shared_ptr<TextSprite> text(const std::string& string, float size, const Color& color, float anchorX, float anchorY, float x, float y) {
            auto sprite = TextSprite::create(font_, size * scale_, TextBox::Line);
            sprite->setText(string);
            sprite->setColor(color);
            sprite->setHorizontalAlignment(anchorX > 0.5f ? TextSprite::HorizontalAlignment::Right : TextSprite::HorizontalAlignment::Left);
            sprite->setVerticalAlignment(anchorY > 0.5f ? TextSprite::VerticalAlignment::Below : TextSprite::VerticalAlignment::Above);
            sprite->screenSpace = true;
            sprite->screenAnchor.set(anchorX, anchorY);
            sprite->position.set(x * scale_, y * scale_, 0);
            scene_.add(sprite);

            return sprite;
        }

        // A dark rectangle, placed by the corner that its anchor names.
        std::shared_ptr<SpriteMaterial> plate(float anchorX, float anchorY, float x, float y, float width, float height) {
            auto material = SpriteMaterial::create();
            material->color = Color(0x0b0d10);
            material->transparent = true;
            material->opacity = 0.72f;
            material->toneMapped = false;

            auto sprite = Sprite::create(material);
            sprite->center.set(anchorX, anchorY);
            sprite->screenSpace = true;
            sprite->screenAnchor.set(anchorX, anchorY);
            sprite->position.set(x * scale_, y * scale_, 0);
            sprite->scale.set(width * scale_, height * scale_, 1);
            scene_.add(sprite);

            return material;
        }
    };

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, sequenceDir;
    bool plain = false, demo = false, startRaining = false, startTap = false, startDrain = false;
    float shotTime = 2;
    int sequenceFps = 30;
    int width = 1280, height = 720;
    Orbit orbit;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--sequence" && i + 1 < argc) sequenceDir = argv[++i];
        else if (a == "--fps" && i + 1 < argc) sequenceFps = std::atoi(argv[++i]);
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--plain") plain = true;
        else if (a == "--demo") demo = true;
        else if (a == "--rain") startRaining = true;
        else if (a == "--tap") startTap = true;
        else if (a == "--drain") startDrain = true;
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--orbit" && i + 3 < argc) {
            orbit.azimuth = std::stof(argv[i + 1]) * math::DEG2RAD;
            orbit.polar = std::stof(argv[i + 2]) * math::DEG2RAD;
            orbit.distance = std::stof(argv[i + 3]);
            i += 3;
        }
    }
    const bool headless = !shotPath.empty() || !sequenceDir.empty();
    if (!sequenceDir.empty()) std::filesystem::create_directories(sequenceDir);

    Canvas canvas(Canvas::Parameters()
                          .title("Sand table")
                          .size(width, height)
                          .antialiasing(4)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;
    // a frame draws the scene more than once; the shadows are the same in all of them
    renderer.shadowMapAutoUpdate = false;

    const Color backdrop(0x14171b);
    Scene scene;
    scene.background = backdrop;
    scene.fog = Fog(backdrop, 3, 11);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene.environment = pmrem.fromScene(room, 0.04f);
        room.dispose();
    }
    scene.environmentIntensity = 0.4f;

    setupStudio(scene);

    auto table = Group::create();
    table->position.y = baseTop;
    scene.add(table);
    createBox(*table, scene);
    Plumbing plumbing(*table);

    Simulation simulation;
    simulation.rain = startRaining;
    simulation.tap = startTap;
    simulation.drain = startDrain;

    Surface surface;
    table->add(surface.sand);
    table->add(surface.water);

    Floats floats(16, simulation);
    static const std::array<Color, 6> ballColors{Color(0xff5a36), Color(0xffc21f), Color(0x35d46b), Color(0x2bb8ff), Color(0xb07cff), Color(0xf2f2f2)};
    auto ballMaterial = MeshStandardMaterial::create();
    ballMaterial->roughness = 0.35f;
    auto balls = shadowed(InstancedMesh::create(SphereGeometry::create(Floats::radius, 24, 12), ballMaterial, floats.balls.size()));
    for (size_t k = 0; k < floats.balls.size(); ++k) balls->setColorAt(k, ballColors[k % ballColors.size()]);
    balls->frustumCulled = false;
    table->add(balls);

    PerspectiveCamera camera(34, canvas.aspect(), 0.02f, 40);
    orbit.apply(camera);

    Hud hud;

    EffectComposer::Options composerOptions;
    composerOptions.samples = 4;
    EffectComposer composer(renderer, composerOptions);
    composer.addPass(std::make_shared<RenderPass>(scene, camera));

    const auto size = canvas.size();

    GTAOParameters aoParameters;
    aoParameters.radius = 0.03f;
    aoParameters.distanceExponent = 1.f;
    aoParameters.thickness = 1.f;
    aoParameters.scale = 1.f;
    aoParameters.samples = 16;
    aoParameters.distanceFallOff = 1.f;
    aoParameters.screenSpaceRadius = false;
    PoissonDenoiseParameters denoiseParameters;
    denoiseParameters.lumaPhi = 10.f;
    denoiseParameters.depthPhi = 2.f;
    denoiseParameters.normalPhi = 3.f;
    denoiseParameters.radius = 4.f;
    denoiseParameters.radiusExponent = 1.f;
    denoiseParameters.rings = 2.f;
    denoiseParameters.samples = 16;
    auto ao = std::make_shared<GTAOPass>(scene, camera, size.width(), size.height(), aoParameters, denoiseParameters);
    composer.addPass(ao);

    auto dof = std::make_shared<BokehPass>(scene, camera, orbit.distance, 0.005f, 0.0035f);
    composer.addPass(dof);

    auto bloom = std::make_shared<UnrealBloomPass>(Vector2(static_cast<float>(size.width()), static_cast<float>(size.height())), 0.1f, 0.1f, 1.3f);
    composer.addPass(bloom);

    if (plain) ao->enabled = dof->enabled = bloom->enabled = false;

    Pointer pointer;
    canvas.addMouseListener(pointer);

    int tool = 1;
    float brushRadius = 0.03f;
    bool paused = false;
    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::NUM_1: tool = 0; break;
            case Key::NUM_2: tool = 1; break;
            case Key::NUM_3: tool = 2; break;
            case Key::NUM_4: tool = 3; break;
            case Key::UP: brushRadius = std::min(0.1f, brushRadius * 1.25f); break;
            case Key::DOWN: brushRadius = std::max(0.012f, brushRadius / 1.25f); break;
            case Key::T: simulation.tap = !simulation.tap; break;
            case Key::D: simulation.drain = !simulation.drain; break;
            case Key::R: simulation.rain = !simulation.rain; break;
            case Key::E: simulation.erosion = !simulation.erosion; break;
            case Key::B: floats.scatter(simulation); break;
            case Key::SPACE: paused = !paused; break;
            case Key::BACKSPACE:
                simulation.reset();
                floats.scatter(simulation);
                break;
            default: break;
        }
    });

    const std::array<const char*, 4> toolIcons{icons::pile, icons::dig, icons::pour, icons::soak};
    for (int k = 0; k < 4; ++k) {
        hud.addButton(0, k, std::to_string(k + 1), toolIcons[k], [&, k] { return (demo ? 1 : tool) == k; }, [&, k] { tool = k; });
    }
    hud.addButton(1, 0, "T", icons::tap, [&] { return simulation.tap; }, [&] { simulation.tap = !simulation.tap; });
    hud.addButton(1, 1, "D", icons::drain, [&] { return simulation.drain; }, [&] { simulation.drain = !simulation.drain; });
    hud.addButton(1, 2, "R", icons::rain, [&] { return simulation.rain; }, [&] { simulation.rain = !simulation.rain; });
    hud.addButton(1, 3, "E", icons::erosion, [&] { return simulation.erosion; }, [&] { simulation.erosion = !simulation.erosion; });
    hud.layout(canvas.size());

    canvas.onWindowResize([&](WindowSize newSize) {
        camera.aspect = newSize.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(newSize);
        composer.setSize(newSize.width(), newSize.height());
        hud.layout(newSize);
    });

    // Uses the tool at a point given in the table's frame.
    const auto use = [&](const Tool& t, float x, float z, float dt) {
        if (t.water) simulation.pour(x, z, brushRadius, t.rate * dt);
        else
            simulation.sculpt(x, z, brushRadius, t.rate * dt);
    };

    // Where on the sand or the water a pixel of the window is, in the table's frame.
    Raycaster raycaster;
    const auto pickAt = [&](const Vector2& pixel, Vector3& hit) {
        const auto window = canvas.size();
        raycaster.setFromCamera(Vector2(2 * pixel.x / static_cast<float>(window.width()) - 1, 1 - 2 * pixel.y / static_cast<float>(window.height())), camera);
        Ray ray = raycaster.ray;
        ray.origin.sub(table->position);

        return simulation.pick(ray, hit);
    };

    using SteadyClock = std::chrono::steady_clock;
    const auto milliseconds = [](SteadyClock::duration d) { return std::chrono::duration<float, std::milli>(d).count(); };

    Clock clock;
    float time = 0, sinceHud = 1, frameTime = 1.f / 60, physicsTime = 0, meshTime = 0;
    float focus = orbit.distance;
    bool onButtons = false;
    double physicsTotal = 0, meshTotal = 0;
    int frame = 0, written = 0;
    canvas.animate([&] {
        const float elapsed = std::min(clock.getDelta(), 0.05f);
        const float dt = headless ? 1.f / 60 : elapsed;
        time += dt;
        frameTime += 0.05f * (elapsed - frameTime);

        // the camera
        if (pointer.dragged.x != 0 || pointer.dragged.y != 0) {
            orbit.azimuth -= 0.006f * pointer.dragged.x;
            orbit.polar = std::clamp(orbit.polar - 0.006f * pointer.dragged.y, 0.08f, 1.45f);
            pointer.dragged.set(0, 0);
        }
        if (pointer.wheel != 0) {
            orbit.distance = std::clamp(orbit.distance * std::pow(0.9f, pointer.wheel), 0.3f, 4.f);
            pointer.wheel = 0;
        }
        orbit.apply(camera);
        camera.updateMatrixWorld();

        // where the pointer is on the sand
        Vector3 hit;
        bool onTable = false;
        if (demo) {
            // digging the lake open, from inside the rim out onto the plain
            const float along = std::clamp((time - 1.f) / 4.f, 0.f, 1.f);
            hit.set(math::lerp(lakeX + 0.07f, lakeX + 0.36f, along), 0, math::lerp(lakeZ, 0.03f, along));
            hit.y = simulation.groundAt(hit.x, hit.z) + simulation.waterAt(hit.x, hit.z);
            onTable = true;
            if (time > 1 && time < 5) use(tools[1], hit.x, hit.z, dt);
        } else if (pointer.position.x >= 0) {
            // A press that lands on a button is the button's, until the mouse
            // button comes up again: dragging off it does not start digging.
            if (pointer.pressed) onButtons = hud.click(pointer.position);
            if (!pointer.left) onButtons = false;

            onTable = !onButtons && !hud.covers(pointer.position) && pickAt(pointer.position, hit);
            if (onTable && pointer.left) use(tools[tool], hit.x, hit.z, dt);
        }
        pointer.pressed = false;

        auto start = SteadyClock::now();
        if (!paused) {
            simulation.step(dt);
            floats.update(dt, simulation);
        }
        auto lap = SteadyClock::now();
        surface.refresh(simulation);
        plumbing.update(simulation, dt);
        Matrix4 matrix;
        for (size_t k = 0; k < floats.balls.size(); ++k) {
            matrix.setPosition(floats.balls[k].position);
            balls->setMatrixAt(k, matrix);
        }
        balls->instanceMatrix()->needsUpdate();
        const auto done = SteadyClock::now();
        physicsTime += 0.05f * (milliseconds(lap - start) - physicsTime);
        meshTime += 0.05f * (milliseconds(done - lap) - meshTime);
        physicsTotal += milliseconds(lap - start);
        meshTotal += milliseconds(done - lap);

        hud.ring->visible = onTable;
        if (onTable) {
            hud.ring->position.copy(hit).add(table->position);
            hud.ring->scale.setScalar(brushRadius);
        }

        // in focus: where the work is
        const float wanted = onTable ? camera.position.distanceTo(hud.ring->position) : orbit.distance;
        focus += (wanted - focus) * std::min(1.f, 4.f * dt);
        dof->focus = focus;

        renderer.shadowMap().needsUpdate = true;
        composer.render(dt);

        sinceHud += elapsed;
        if (sinceHud >= 0.25f || headless) {
            sinceHud = 0;

            char buffer[200];
            std::snprintf(buffer, sizeof(buffer), "%s\n%.0f mm\n%.2f l\n%.2f ms\n%.1f ms", demo ? tools[1].name : tools[tool].name,
                          brushRadius * 2000, simulation.volume(), physicsTime + meshTime, frameTime * 1000);
            if (buffer != hud.values->getText()) hud.values->setText(buffer);
        }

        hud.update();
        hud.render(renderer, camera);

        if (!sequenceDir.empty() && frame % std::max(1, 60 / sequenceFps) == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04d.png", written++);
            renderer.writeFramebuffer(std::filesystem::path(sequenceDir) / name);
        }
        ++frame;
        if (headless && time >= shotTime) {
            if (!shotPath.empty()) {
                renderer.writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << std::endl;
            }
            std::cout << "per frame: physics " << physicsTotal / frame << " ms, meshes " << meshTotal / frame << " ms, water "
                      << simulation.volume() << " l" << std::endl;

            // The pointer's path, checked without a pointer: points on the
            // surface go to the screen and come back through pickAt.
            float worst = 0;
            for (const auto& [x, z] : {std::pair{lakeX, lakeZ}, std::pair{0.f, 0.f}, std::pair{0.4f, -0.2f}, std::pair{-0.1f, 0.3f}}) {
                const Vector3 point(x, simulation.groundAt(x, z) + simulation.waterAt(x, z), z);
                Vector3 onScreen = Vector3(point).add(table->position).project(camera);
                const auto window = canvas.size();
                Vector3 found;
                const bool ok = pickAt(Vector2((onScreen.x + 1) / 2 * static_cast<float>(window.width()), (1 - onScreen.y) / 2 * static_cast<float>(window.height())), found);
                worst = std::max(worst, ok ? found.distanceTo(point) : 1.f);
            }
            std::cout << "pointer check: worst miss " << worst * 1000 << " mm" << std::endl;

            // And the buttons: a click in the middle of each has to change what it stands for.
            const auto state = [&] { return tool + 8 * (simulation.tap + 2 * simulation.drain + 4 * simulation.rain + 8 * simulation.erosion); };
            size_t responding = 0;
            tool = -1;
            for (size_t k = 0; k < hud.buttonCount(); ++k) {
                const int before = state();
                responding += hud.click(hud.buttonCentre(k)) && state() != before;
            }
            std::cout << "button check: " << responding << " of " << hud.buttonCount() << " respond" << std::endl;
            std::exit(0);
        }
    });
}
