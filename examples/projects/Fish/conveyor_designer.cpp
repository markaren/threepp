// Conveyor System Designer — assemble Isaac Sim conveyor pieces into a layout and
// save it to JSON for the physics sim to rebuild.
//
// Models are whatever .usd files have been downloaded into data/models/conveyor
// (see scripts/fetch_isaac_conveyors.sh). Add pieces from the palette, orbit with
// the mouse, drag the gizmo to position a piece (W = move, E = rotate, hold SHIFT
// to snap), and set yaw / scale / belt speed per piece in the panel. Save/Load
// round-trips the layout to JSON.

#include "threepp/threepp.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/USDLoader.hpp"

#include "ConveyorAssets.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Raw GL 1.1 texture upload so the thumbnails can be shown in ImGui (its GL3
// backend takes a GL texture id as ImTextureID). Windows-only; elsewhere the
// palette falls back to text buttons. Included last so <windows.h> macros don't
// leak into the threepp/imgui headers.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/gl.h>
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#endif

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    // Decode a PNG via threepp's TextureLoader and upload it to a GL texture.
    // Returns the GL id (0 on failure). Must be called with the GL context current.
    unsigned int uploadThumbnail(const fs::path& png) {
#ifdef _WIN32
        if (!fs::exists(png)) return 0;
        TextureLoader tl;
        // flipY=false: keep top-down PNG row order. TextureLoader flips by default
        // for 3D GL texture-coord convention, which would show ImGui images upside down.
        auto tex = tl.load(png.string(), false);
        if (!tex) return 0;
        auto& img = tex->image();
        auto& data = img.data<unsigned char>();
        if (data.empty() || img.width() == 0 || img.height() == 0) return 0;
        const std::size_t channels = data.size() / (static_cast<std::size_t>(img.width()) * img.height());
        const GLenum fmt = channels == 4 ? GL_RGBA : (channels == 3 ? GL_RGB : 0);
        if (fmt == 0) return 0;

        GLuint id = 0;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(fmt),
                     static_cast<GLsizei>(img.width()), static_cast<GLsizei>(img.height()),
                     0, fmt, GL_UNSIGNED_BYTE, data.data());
        return id;
#else
        (void) png;
        return 0;
#endif
    }

}// namespace

int main(int argc, char** argv) {

    // Conveyor model/layout folder from the command line (optional):
    //   conveyor_designer [conveyorDir]
    const fs::path modelDir = argc > 1 ? fs::path(argv[1])
                                       : fs::path(std::string(PROJECT_FOLDER) + "/data/models/conveyor");
    auto models = conveyor::discoverModels(modelDir);
    if (models.empty()) {
        std::cerr << "No conveyor .usd models in " << modelDir
                  << " — run scripts/fetch_isaac_conveyors.sh --models" << std::endl;
    }

    Canvas canvas("Conveyor System Designer", {{"aa", 4}, {"vsync", true}});
    auto renderer = GLRenderer(canvas);

    Scene scene;
    scene.background = Color(0x20242b);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.05f, 200);
    camera->position.set(4, 5, 6);

    OrbitControls orbit{*camera, canvas};
    orbit.target.set(0, 0.3f, 0);

    scene.add(AmbientLight::create(0xffffff, 0.5f));
    auto sun = DirectionalLight::create(0xffffff, 1.5f);
    sun->position.set(6, 10, 4);
    scene.add(sun);

    auto grid = GridHelper::create(20, 20);
    scene.add(grid);

    USDLoader loader;
    auto typeOverrides = conveyor::loadTypeOverrides(modelDir / "conveyor_types.json");
    std::unordered_map<std::string, conveyor::ModelTemplate> templates;
    auto getTemplate = [&](const std::string& name) -> conveyor::ModelTemplate& {
        auto it = templates.find(name);
        if (it == templates.end()) {
            std::cout << "Loading model " << name << "..." << std::endl;
            it = templates.emplace(name, conveyor::prepareModel(loader, modelDir / (name + ".usd"))).first;
            if (auto ov = typeOverrides.find(name); ov != typeOverrides.end()) {
                it->second.kind = (ov->second == "curve") ? conveyor::BeltKind::Curve
                                                          : conveyor::BeltKind::Straight;
            }
        }
        return it->second;
    };

    struct Placed {
        std::shared_ptr<Object3D> obj;
        std::string model;
        float yawDeg = 0.f;
        float scale = 1.f;
        float beltSpeed = 0.6f;
    };
    std::vector<Placed> placed;
    int selected = -1;

    // Multiple parallel belt paths. Each is a list of draggable waypoint markers
    // (spline control points) + its own belt settings, drawn in a distinct colour.
    auto wpGeom = SphereGeometry::create(0.08f, 12, 8);
    auto pathColor = [](int i) {
        static const int cols[] = {0xff8800, 0x33cc66, 0xcc44ff, 0x44aaff, 0xffcc33, 0xff4466};
        return Color(cols[((i % 6) + 6) % 6]);
    };
    struct DesignPath {
        std::vector<std::shared_ptr<Object3D>> markers;
        std::vector<char> arcCenter;// parallel to markers: 1 = this is an arc centre
        std::vector<int> segKind;   // parallel to markers: surface of the segment leaving each
        float beltWidth = 1.0f, beltSpeed = 0.6f;
        bool reverse = false, smooth = true;
        // Separator: a collision-only vertical wall along the centerline instead of a
        // belt. beltWidth is reused as the wall height when this is set.
        bool separator = false;
        // Shared tuning for whichever segments opt into rollers / cleats (segKind, per
        // segment). rollerRadius = cylinder radius; cleatHeight/cleatSpacing size the flights.
        float rollerRadius = 0.05f;
        float cleatHeight = 0.15f;
        float cleatSpacing = 0.6f;
        std::shared_ptr<MeshBasicMaterial> markerMat, planeMat;
        std::shared_ptr<MeshStandardMaterial> rollerMat, cleatMat;
        std::shared_ptr<LineBasicMaterial> lineMat;
        std::shared_ptr<Line> line;
        std::vector<std::shared_ptr<Mesh>> preview;
        // Roller previews to spin each frame (decoupled from `preview`, which also holds
        // ribbons / cleat bars). rollerBase[i] is roller i's base axis orientation; the spin
        // is an extra turn about its own axis. rollerAngle accumulates that spin.
        std::vector<std::shared_ptr<Mesh>> rollerMeshes;
        std::vector<Quaternion> rollerBase;
        float rollerAngle = 0.f;
        // Rebuild cache so the (spline-dense) preview only updates on change.
        bool cacheValid = false;
        std::vector<Vector3> cacheCtrl;
        std::vector<int> cacheSegKind;
        float cacheWidth = -1.f;
        bool cacheSmooth = true;
        bool cacheSeparator = false;
        float cacheRollerRadius = -1.f;
        float cacheCleatHeight = -1.f, cacheCleatSpacing = -1.f;
    };
    std::vector<DesignPath> paths;
    int activePath = -1;
    int selectedWp = -1;

    TransformControls controls(*camera, canvas);
    controls.setMode("translate");
    scene.addRef(controls);

    LambdaEventListener draggingListener([&](Event& e) {
        orbit.enabled = !std::any_cast<bool>(e.target);
    });
    controls.addEventListener("dragging-changed", draggingListener);

    auto selectPiece = [&](int i) {
        selected = i;
        selectedWp = -1;
        if (i >= 0 && i < static_cast<int>(placed.size())) {
            controls.attach(*placed[i].obj);
        } else {
            controls.detach();
        }
    };

    auto selectWaypoint = [&](int i) {
        selectedWp = i;
        selected = -1;
        if (activePath >= 0 && i >= 0 && i < static_cast<int>(paths[activePath].markers.size())) {
            controls.attach(*paths[activePath].markers[i]);
        } else {
            controls.detach();
        }
    };

    auto addPath = [&] {
        DesignPath p;
        const Color c = pathColor(static_cast<int>(paths.size()));
        p.markerMat = MeshBasicMaterial::create();
        p.markerMat->color = c;
        p.lineMat = LineBasicMaterial::create();
        p.lineMat->color = c;
        p.planeMat = MeshBasicMaterial::create();
        p.planeMat->color = c;
        p.planeMat->transparent = true;
        p.planeMat->opacity = 0.35f;
        p.planeMat->side = Side::Double;
        // Metallic + flat-shaded so the facets catch the light and the spin reads clearly;
        // tinted by the path colour so roller paths are still colour-coded like the others.
        p.rollerMat = MeshStandardMaterial::create();
        p.rollerMat->color = c;
        p.rollerMat->metalness = 0.6f;
        p.rollerMat->roughness = 0.4f;
        p.rollerMat->flatShading = true;
        // Cleats: solid bars, path-colour tinted so they read against the translucent belt.
        p.cleatMat = MeshStandardMaterial::create();
        p.cleatMat->color = c;
        p.cleatMat->metalness = 0.3f;
        p.cleatMat->roughness = 0.6f;
        paths.push_back(std::move(p));
        activePath = static_cast<int>(paths.size()) - 1;
        selectedWp = -1;
    };

    auto addWaypoint = [&] {
        if (activePath < 0) return;
        auto& path = paths[activePath];
        auto m = Mesh::create(wpGeom, path.markerMat);
        if (!path.markers.empty()) {
            m->position.copy(path.markers.back()->position);
            m->position.x += 0.5f;
        } else {
            m->position.set(0.f, 0.05f, 0.f);
        }
        scene.add(m);
        path.markers.push_back(m);
        path.arcCenter.push_back(0);
        path.segKind.push_back(conveyor::SegFlat);
        selectWaypoint(static_cast<int>(path.markers.size()) - 1);
    };

    auto deleteWaypoint = [&](int i) {
        if (activePath < 0) return;
        auto& path = paths[activePath];
        if (i < 0 || i >= static_cast<int>(path.markers.size())) return;
        controls.detach();
        if (path.markers[i]->parent) path.markers[i]->removeFromParent();
        path.markers.erase(path.markers.begin() + i);
        path.arcCenter.erase(path.arcCenter.begin() + i);
        path.segKind.erase(path.segKind.begin() + i);
        selectedWp = -1;
    };

    // Insert a waypoint after index i (midpoint of the i..i+1 gap) in the active path.
    auto insertWaypointAfter = [&](int i) {
        if (activePath < 0) return;
        auto& path = paths[activePath];
        if (i < 0 || i >= static_cast<int>(path.markers.size())) {
            addWaypoint();
            return;
        }
        auto m = Mesh::create(wpGeom, path.markerMat);
        if (i + 1 < static_cast<int>(path.markers.size())) {
            const auto& p0 = path.markers[i]->position;
            const auto& p1 = path.markers[i + 1]->position;
            m->position.set((p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f, (p0.z + p1.z) * 0.5f);
        } else {
            m->position.copy(path.markers[i]->position);
            m->position.x += 0.5f;
        }
        scene.add(m);
        path.markers.insert(path.markers.begin() + (i + 1), m);
        path.arcCenter.insert(path.arcCenter.begin() + (i + 1), 0);
        // New segment inherits the kind of the one it splits, so the surface stays uniform.
        path.segKind.insert(path.segKind.begin() + (i + 1), path.segKind[i]);
        selectWaypoint(i + 1);
    };

    auto deletePath = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(paths.size())) return;
        controls.detach();
        auto& path = paths[idx];
        for (auto& m : path.markers) {
            if (m->parent) m->removeFromParent();
        }
        if (path.line && path.line->parent) path.line->removeFromParent();
        for (auto& m : path.preview) {
            if (m->parent) m->removeFromParent();
        }
        paths.erase(paths.begin() + idx);
        activePath = paths.empty() ? -1 : std::min(activePath, static_cast<int>(paths.size()) - 1);
        selectedWp = -1;
    };

    auto addPiece = [&](const std::string& name) {
        auto& t = getTemplate(name);
        if (!t.group) {
            std::cerr << "Failed to load model " << name << std::endl;
            return;
        }
        auto inst = t.group->clone();
        scene.add(inst);
        placed.push_back({inst, name, 0.f, 1.f, 0.6f});
        selectPiece(static_cast<int>(placed.size()) - 1);
    };

    auto deletePiece = [&](int i) {
        if (i < 0 || i >= static_cast<int>(placed.size())) return;
        controls.detach();
        if (placed[i].obj->parent) placed[i].obj->removeFromParent();
        placed.erase(placed.begin() + i);
        selected = -1;
    };

    auto clearAll = [&] {
        controls.detach();
        for (auto& p : placed) {
            if (p.obj->parent) p.obj->removeFromParent();
        }
        placed.clear();
        for (auto& path : paths) {
            for (auto& m : path.markers) {
                if (m->parent) m->removeFromParent();
            }
            if (path.line && path.line->parent) path.line->removeFromParent();
            for (auto& m : path.preview) {
                if (m->parent) m->removeFromParent();
            }
        }
        paths.clear();
        selected = -1;
        selectedWp = -1;
        addPath();// always keep one active path
    };

    addPath();// seed the first path

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    // Gizmo is translate-only: yaw is authoritative per-piece (set from the panel
    // and re-applied each frame), so a gizmo rotate would just get overwritten.
    KeyAdapter onPress(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::LEFT_SHIFT) controls.setTranslationSnap(0.25f);
    });
    KeyAdapter onRelease(KeyAdapter::KEY_RELEASED, [&](KeyEvent evt) {
        if (evt.key == Key::LEFT_SHIFT) controls.setTranslationSnap(std::nullopt);
    });
    canvas.addKeyListener(onPress);
    canvas.addKeyListener(onRelease);

    std::unordered_map<std::string, unsigned int> thumbCache;
    auto thumbId = [&](const std::string& model) -> unsigned int {
        auto it = thumbCache.find(model);
        if (it != thumbCache.end()) return it->second;
        const unsigned int id = uploadThumbnail(modelDir / "thumbs" / (model + ".usd.png"));
        thumbCache.emplace(model, id);
        return id;
    };

    char pathBuf[260];
    std::strncpy(pathBuf, (modelDir / "layout.json").string().c_str(), sizeof(pathBuf) - 1);
    pathBuf[sizeof(pathBuf) - 1] = '\0';
    std::string status = models.empty() ? "No models found" : "";

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({330, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Conveyor Designer");

        if (!models.empty()) {
            ImGui::TextDisabled("Palette — click a thumbnail to add:");
            ImGui::BeginChild("palette", ImVec2(0, 200), ImGuiChildFlags_Borders);
            const int perRow = 4;
            for (int i = 0; i < static_cast<int>(models.size()); ++i) {
                ImGui::PushID(i);
                const unsigned int tid = thumbId(models[i]);
                bool clicked;
                if (tid) {
                    clicked = ImGui::ImageButton("thumb", static_cast<ImTextureID>(tid), ImVec2(64, 64));
                } else {
                    clicked = ImGui::Button(models[i].c_str(), ImVec2(72, 72));
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", models[i].c_str());
                if (clicked) addPiece(models[i]);
                ImGui::PopID();
                if ((i % perRow) != (perRow - 1)) ImGui::SameLine();
            }
            ImGui::EndChild();
        } else {
            ImGui::TextWrapped("No .usd in data/models/conveyor. Run "
                               "scripts/fetch_isaac_conveyors.sh --models");
        }

        ImGui::Separator();
        ImGui::Text("Pieces: %d", static_cast<int>(placed.size()));
        for (int i = 0; i < static_cast<int>(placed.size()); ++i) {
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "%d: %s##sel%d", i, placed[i].model.c_str(), i);
            if (ImGui::Selectable(lbl, i == selected)) selectPiece(i);
        }

        if (selected >= 0 && selected < static_cast<int>(placed.size())) {
            auto& p = placed[selected];
            const auto& tt = getTemplate(p.model);
            ImGui::Separator();
            ImGui::Text("Selected: %s", p.model.c_str());
            ImGui::Text("Detected: %s", tt.kind == conveyor::BeltKind::Curve
                                                ? (tt.arcSweepDeg > 135.f ? "curve ~180\xC2\xB0" : "curve ~90\xC2\xB0")
                                                : "straight");
            ImGui::DragFloat3("Position", &p.obj->position.x, 0.05f);
            ImGui::SliderFloat("Yaw (deg)", &p.yawDeg, -180.f, 180.f);
            if (ImGui::Button("Yaw -90")) p.yawDeg = std::fmod(p.yawDeg - 90.f + 540.f, 360.f) - 180.f;
            ImGui::SameLine();
            if (ImGui::Button("Yaw +90")) p.yawDeg = std::fmod(p.yawDeg + 90.f + 540.f, 360.f) - 180.f;
            ImGui::SliderFloat("Scale", &p.scale, 0.25f, 3.f);
            ImGui::SliderFloat("Belt speed (m/s)", &p.beltSpeed, 0.f, 3.f);
            if (ImGui::Button("Delete piece")) deletePiece(selected);
        }

        ImGui::Separator();
        ImGui::Text("Belt paths: %d", static_cast<int>(paths.size()));
        if (ImGui::Button("Add path")) addPath();
        ImGui::SameLine();
        if (ImGui::Button("Delete path")) deletePath(activePath);
        for (int i = 0; i < static_cast<int>(paths.size()); ++i) {
            char lbl[40];
            std::snprintf(lbl, sizeof(lbl), "Path %d (%d wp)##path%d", i,
                          static_cast<int>(paths[i].markers.size()), i);
            if (ImGui::Selectable(lbl, i == activePath)) {
                activePath = i;
                selectedWp = -1;
                controls.detach();
            }
        }
        if (activePath >= 0 && activePath < static_cast<int>(paths.size())) {
            auto& path = paths[activePath];
            ImGui::Separator();
            ImGui::Text("Active path %d — %d waypoints", activePath, static_cast<int>(path.markers.size()));
            if (ImGui::Button("Add waypoint")) addWaypoint();
            for (int i = 0; i < static_cast<int>(path.markers.size()); ++i) {
                const bool hasSeg = i + 1 < static_cast<int>(path.markers.size()) && !path.arcCenter[i];
                const char* kindTag = !hasSeg ? "" : path.segKind[i] == conveyor::SegRollers ? " [rollers]"
                                              : path.segKind[i] == conveyor::SegCleats        ? " [cleats]"
                                                                                              : "";
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "wp %d%s%s##wp%d", i,
                              path.arcCenter[i] ? " [arc centre]" : "", kindTag, i);
                if (ImGui::Selectable(lbl, i == selectedWp)) selectWaypoint(i);
            }
            if (selectedWp >= 0 && selectedWp < static_cast<int>(path.markers.size())) {
                ImGui::DragFloat3("WP pos", &path.markers[selectedWp]->position.x, 0.05f);
                bool isArc = path.arcCenter[selectedWp] != 0;
                if (ImGui::Checkbox("Arc centre (bend between neighbours)", &isArc)) {
                    path.arcCenter[selectedWp] = isArc ? 1 : 0;
                    const float s = isArc ? 1.8f : 1.f;
                    path.markers[selectedWp]->scale.set(s, s, s);
                    path.cacheValid = false;// force preview rebuild
                }
                // Per-segment surface: the kind of the span leaving this waypoint (flat by
                // default; arc spans are always flat so the selector is hidden for them).
                if (!path.separator && selectedWp + 1 < static_cast<int>(path.markers.size()) &&
                    !path.arcCenter[selectedWp]) {
                    int* kp = &path.segKind[selectedWp];
                    ImGui::Text("Segment to next:");
                    ImGui::RadioButton("Flat", kp, conveyor::SegFlat);
                    ImGui::SameLine();
                    ImGui::RadioButton("Rollers", kp, conveyor::SegRollers);
                    ImGui::SameLine();
                    ImGui::RadioButton("Cleats", kp, conveyor::SegCleats);
                }
                if (ImGui::Button("Insert after")) insertWaypointAfter(selectedWp);
                ImGui::SameLine();
                if (ImGui::Button("Delete waypoint")) deleteWaypoint(selectedWp);
            }
            ImGui::Checkbox("Separator (vertical wall, collision-only)", &path.separator);
            if (path.separator) {
                ImGui::SliderFloat("Wall height", &path.beltWidth, 0.1f, 3.f);
                ImGui::Checkbox("Smooth (spline)", &path.smooth);
            } else {
                ImGui::SliderFloat("Belt width", &path.beltWidth, 0.1f, 3.f);
                ImGui::SliderFloat("Belt speed (m/s)", &path.beltSpeed, 0.f, 3.f);
                ImGui::Checkbox("Reverse flow", &path.reverse);
                ImGui::SameLine();
                ImGui::Checkbox("Smooth (spline)", &path.smooth);
                // Tuning for whichever segments opt into rollers / cleats (set per segment in
                // the waypoint list above). Shown only when at least one segment uses them.
                bool anyRollers = false, anyCleats = false;
                for (int k = 0; k + 1 < static_cast<int>(path.markers.size()); ++k) {
                    if (path.arcCenter[k]) continue;
                    if (path.segKind[k] == conveyor::SegRollers) anyRollers = true;
                    else if (path.segKind[k] == conveyor::SegCleats) anyCleats = true;
                }
                if (anyRollers) ImGui::SliderFloat("Roller radius", &path.rollerRadius, 0.02f, 0.2f);
                if (anyCleats) {
                    ImGui::SliderFloat("Cleat height", &path.cleatHeight, 0.03f, 0.5f);
                    ImGui::SliderFloat("Cleat spacing", &path.cleatSpacing, 0.15f, 2.f);
                }
                ImGui::TextDisabled("Per segment: pick a waypoint, set Flat / Rollers / Cleats.");
            }
        }

        ImGui::Separator();
        ImGui::InputText("File", pathBuf, sizeof(pathBuf));
        if (ImGui::Button("Save")) {
            conveyor::Layout layout;
            for (auto& p : placed) {
                conveyor::Piece piece;
                piece.model = p.model;
                piece.position = p.obj->position;
                piece.yawDeg = p.yawDeg;
                piece.scale = p.scale;
                piece.beltSpeed = p.beltSpeed;
                layout.pieces.push_back(piece);
            }
            for (auto& path : paths) {
                conveyor::Path cp;
                cp.beltWidth = path.beltWidth;
                cp.beltSpeed = path.beltSpeed;
                cp.reverse = path.reverse;
                cp.smooth = path.smooth;
                cp.separator = path.separator;
                cp.wallHeight = path.beltWidth;// designer reuses the width slider as wall height
                cp.rollerRadius = path.rollerRadius;
                cp.cleatHeight = path.cleatHeight;
                cp.cleatSpacing = path.cleatSpacing;
                for (size_t k = 0; k < path.markers.size(); ++k) {
                    conveyor::Waypoint w;
                    w.pos = path.markers[k]->position;
                    w.arcCenter = path.arcCenter[k] != 0;
                    w.segKind = path.segKind[k];
                    cp.waypoints.push_back(w);
                }
                layout.paths.push_back(cp);
            }
            conveyor::saveLayout(pathBuf, layout);
            status = "Saved " + std::to_string(layout.pieces.size()) + " pieces, " +
                     std::to_string(layout.paths.size()) + " paths";
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            if (auto loaded = conveyor::loadLayout(pathBuf)) {
                clearAll();   // pieces gone; one empty path seeded
                deletePath(0);// remove that seeded path; rebuild from file
                for (auto& piece : loaded->pieces) {
                    addPiece(piece.model);
                    auto& p = placed.back();
                    p.obj->position.copy(piece.position);
                    p.yawDeg = piece.yawDeg;
                    p.scale = piece.scale;
                    p.beltSpeed = piece.beltSpeed;
                }
                for (auto& lp : loaded->paths) {
                    addPath();
                    auto& path = paths.back();
                    path.separator = lp.separator;
                    path.beltWidth = lp.separator ? lp.wallHeight : lp.beltWidth;// width slider doubles as height
                    path.beltSpeed = lp.beltSpeed;
                    path.reverse = lp.reverse;
                    path.smooth = lp.smooth;
                    path.rollerRadius = lp.rollerRadius;
                    path.cleatHeight = lp.cleatHeight;
                    path.cleatSpacing = lp.cleatSpacing;
                    for (auto& wp : lp.waypoints) {
                        auto m = Mesh::create(wpGeom, path.markerMat);
                        m->position.copy(wp.pos);
                        if (wp.arcCenter) m->scale.set(1.8f, 1.8f, 1.8f);
                        scene.add(m);
                        path.markers.push_back(m);
                        path.arcCenter.push_back(wp.arcCenter ? 1 : 0);
                        path.segKind.push_back(wp.segKind);
                    }
                }
                if (paths.empty()) addPath();
                activePath = 0;
                selectPiece(-1);
                status = "Loaded " + std::to_string(loaded->pieces.size()) + " pieces, " +
                         std::to_string(loaded->paths.size()) + " paths";
            } else {
                status = "Load failed";
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) clearAll();
        if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());

        ImGui::Separator();
        ImGui::TextDisabled("Add pieces (visual) + a belt path (waypoints) the fish ride.");
        ImGui::TextDisabled("Drag gizmo to move selected piece/waypoint (SHIFT=snap).");
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        // Yaw/scale are authoritative per-piece fields; position is gizmo/ImGui-driven.
        for (auto& p : placed) {
            p.obj->rotation.set(0, math::degToRad(p.yawDeg), 0);
            p.obj->scale.set(p.scale, p.scale, p.scale);
        }
        // Rebuild each path's line + velocity-plane preview, but only when its control
        // points / width / smoothing change (the spline is dense — rebuilding every
        // frame would be wasteful). While a waypoint is being dragged it stays dirty.
        for (auto& path : paths) {
            bool dirty = !path.cacheValid || path.cacheWidth != path.beltWidth ||
                         path.cacheSmooth != path.smooth || path.cacheSeparator != path.separator ||
                         path.cacheRollerRadius != path.rollerRadius ||
                         path.cacheCleatHeight != path.cleatHeight || path.cacheCleatSpacing != path.cleatSpacing ||
                         path.cacheSegKind != path.segKind ||
                         path.cacheCtrl.size() != path.markers.size();
            if (!dirty) {
                for (size_t i = 0; i < path.markers.size(); ++i) {
                    const Vector3& q = path.markers[i]->position;
                    const Vector3& c = path.cacheCtrl[i];
                    if (std::abs(q.x - c.x) > 1e-5f || std::abs(q.y - c.y) > 1e-5f ||
                        std::abs(q.z - c.z) > 1e-5f) {
                        dirty = true;
                        break;
                    }
                }
            }
            if (!dirty) continue;

            path.cacheCtrl.clear();
            for (auto& m : path.markers) path.cacheCtrl.push_back(m->position);
            path.cacheWidth = path.beltWidth;
            path.cacheSmooth = path.smooth;
            path.cacheSeparator = path.separator;
            path.cacheRollerRadius = path.rollerRadius;
            path.cacheCleatHeight = path.cleatHeight;
            path.cacheCleatSpacing = path.cleatSpacing;
            path.cacheSegKind = path.segKind;
            path.cacheValid = true;

            if (path.line) {
                if (path.line->parent) path.line->removeFromParent();
                path.line.reset();
            }
            for (auto& m : path.preview) {
                if (m->parent) m->removeFromParent();
            }
            path.preview.clear();
            path.rollerMeshes.clear();
            path.rollerBase.clear();

            std::vector<conveyor::Waypoint> wps;
            wps.reserve(path.markers.size());
            for (size_t k = 0; k < path.markers.size(); ++k) {
                conveyor::Waypoint w;
                w.pos = path.markers[k]->position;
                w.arcCenter = path.arcCenter[k] != 0;
                w.segKind = path.segKind[k];
                wps.push_back(w);
            }
            const std::vector<Vector3> pts = conveyor::resamplePath(wps, path.smooth);
            if (pts.size() >= 2) {
                auto lineGeom = BufferGeometry::create();
                lineGeom->setFromPoints(pts);
                path.line = Line::create(lineGeom, path.lineMat);
                scene.add(path.line);
                if (path.separator) {
                    // A separator previews as one continuous vertical wall (beltWidth = height).
                    auto wall = Mesh::create(conveyor::wallGeometry(pts, path.beltWidth), path.planeMat);
                    scene.add(wall);
                    path.preview.push_back(wall);
                } else {
                    // Build each segment's surface by its kind. Runs share boundary points so a
                    // flat→rollers/cleats change meets gap-free; consecutive flat segments are
                    // one continuous ribbon (so bends stay watertight).
                    std::shared_ptr<CylinderGeometry> cyl;
                    std::shared_ptr<BoxGeometry> box;
                    for (const auto& run : conveyor::resamplePathByKind(wps, path.smooth)) {
                        if (run.pts.size() < 2) continue;
                        if (run.kind == conveyor::SegRollers) {
                            if (!cyl) cyl = CylinderGeometry::create(path.rollerRadius, path.rollerRadius,
                                                                     path.beltWidth, 12);
                            const float spacing = conveyor::rollerSpacing(path.rollerRadius);
                            for (const auto& r : conveyor::rollerTransforms(run.pts, path.rollerRadius, spacing)) {
                                auto roller = Mesh::create(cyl, path.rollerMat);
                                roller->position.copy(r.center);
                                roller->quaternion.copy(r.orientation);
                                scene.add(roller);
                                path.preview.push_back(roller);
                                path.rollerMeshes.push_back(roller);
                                path.rollerBase.push_back(r.orientation);
                            }
                        } else {
                            auto ribbon = Mesh::create(conveyor::ribbonGeometry(run.pts, path.beltWidth), path.planeMat);
                            scene.add(ribbon);
                            path.preview.push_back(ribbon);
                            if (run.kind == conveyor::SegCleats) {
                                if (!box) box = BoxGeometry::create(conveyor::kCleatThickness,
                                                                    path.cleatHeight, path.beltWidth);
                                for (const auto& cl : conveyor::cleatTransforms(run.pts, path.cleatHeight, path.cleatSpacing)) {
                                    auto bar = Mesh::create(box, path.cleatMat);
                                    bar->position.copy(cl.center);
                                    bar->quaternion.copy(cl.orientation);
                                    scene.add(bar);
                                    path.preview.push_back(bar);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Spin roller previews so roller segments read as moving. The cylinder's own axis
        // (local +Y) is aligned to the belt-width direction by rollerBase; the extra turn
        // about that axis is the surface rolling. omega = surface speed / radius; negative
        // so the top surface flows along travel (reverse flips it).
        for (auto& path : paths) {
            if (path.rollerMeshes.empty()) continue;
            const float omega = (path.reverse ? 1.f : -1.f) * path.beltSpeed /
                                std::max(path.rollerRadius, 1e-3f);
            path.rollerAngle += omega * dt;
            Quaternion spin;
            spin.setFromAxisAngle(Vector3(0, 1, 0), path.rollerAngle);
            for (size_t i = 0; i < path.rollerMeshes.size() && i < path.rollerBase.size(); ++i) {
                Quaternion q;
                q.multiplyQuaternions(path.rollerBase[i], spin);
                path.rollerMeshes[i]->quaternion.copy(q);
            }
        }

        renderer.render(scene, *camera);
        ui.render();
    });
}
