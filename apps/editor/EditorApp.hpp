// The editor application: one window, one document, one selection.
//
// EditorApp owns the platform-facing pieces (canvas, renderer, editor camera,
// gizmo, ImGui context) and the reusable editor core from
// threepp/extras/editor (document, selection, command stack, play controller).
// The panels are member functions split across apps/editor/panels/*.cpp — they
// are all views onto this one object, so keeping them members avoids a web of
// back-pointers.
//
// Editor-only scene content (grid, axes, selection outline, transform gizmo)
// hangs off a single overlay Group that is registered with the SceneDocument
// as editor-only, so it renders but is detached from the scene for the duration
// of every save and every play snapshot.

#ifndef THREEPP_EDITOR_EDITORAPP_HPP
#define THREEPP_EDITOR_EDITORAPP_HPP

#include "FileBrowser.hpp"
#include "VulkanViewPane.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"
#endif

#include "threepp/extras/editor/Command.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/EditorSettings.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/RenderConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/Selection.hpp"
#include "threepp/extras/editor/TerrainConfig.hpp"
#include "threepp/extras/editor/TerrainSculpt.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/constants.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/helpers/Box3Helper.hpp"
#include "threepp/helpers/BoxHelper.hpp"
#include "threepp/input/IOCapture.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class ImguiContext;

namespace threepp {

    class Audio;
    class AudioListener;
    class CameraHelper;
    class Line;
    class LineBasicMaterial;
    class LineSegments;
    class Material;
    class MeshBasicMaterial;
    class ObjectWithMorphTargetInfluences;
    // The edit-mode particle previews. Forward-declared like the play sessions
    // below: every panel includes this header and none of them has any business
    // recompiling against the renderer-facing particle type.
    class ParticleField;
    class Points;
    class Robot;
    class Texture;

}// namespace threepp

namespace threepp::editor {

    // Held by the app so the collider overlay can reach its PhysX world.
    // Forward-declared: PhysicsPlaySession pulls in the whole PhysX SDK, and
    // every panel includes this header.
    class PhysicsPlaySession;
    // Forward-declared for the same reason (it includes PhysicsPlaySession).
    class ConveyorPlaySession;
    // The play-mode particle fields. PhysX-free, but it includes the
    // renderer-facing particle type and the Vulkan renderer — neither of which
    // a panel has any business recompiling against.
    class ParticleFieldPlaySession;
    // The play-mode grain piles. Forward-declared for both reasons at once: it
    // includes PhysicsPlaySession (the whole PhysX SDK) and the particle types.
    class GranularPlaySession;
    // The play-mode character controllers. Forward-declared for the first
    // reason (it includes PhysicsPlaySession, hence the whole PhysX SDK).
    class CharacterPlaySession;
    // PhysX-free (the PhysX half is PhysxSensorPlaySession, constructed in
    // EditorApp.cpp), but still heavy — it pulls in the depth/lidar sensors and
    // the renderer, which the panels have no business recompiling against.
    class SensorPlaySession;
    // Baked scan surfaces during Play. Forward-declared for both reasons at
    // once (PhysX SDK + the Vulkan renderer), and the memo beside it because it
    // carries SplatCloud and the bake.
    class SplatSurfacePlaySession;
    class SplatSurfaceCache;
    // Sounds during Play. Only exists in a THREEPP_WITH_AUDIO build — that
    // macro is PUBLIC on the threepp target, so every TU that includes this
    // header agrees on whether the member below is there.
    class AudioPlaySession;

    class EditorApp {

    public:
        struct Options {
            bool vulkan = false;
            std::filesystem::path openOnStart;
            // Run this many frames and exit (0 = until the window closes).
            // Makes the editor scriptable enough for a smoke test.
            int maxFrames = 0;
            // Drive select/delete/undo through the UI code paths and exit
            // non-zero on failure. Diagnostic, not part of the test suite.
            bool selfTest = false;
            // Run only the selftest sections matching these comma-separated,
            // case-insensitive terms (--selftest=terrain). An exact section
            // name selects that section; anything else matches as a substring.
            // Empty = the full suite.
            std::string selfTestFilter;
            // Press Play as soon as the scene is open. With --frames this
            // makes a play session scriptable end to end: open, play, exit.
            bool play = false;
            // Optional robot for the selftest's URDF pass.
            std::filesystem::path urdf;
            // Optional rigged model for the selftest's CHARACTER pass. Without
            // it that section builds a synthetic rig with known clip speeds
            // (which is what CI runs); with it, the same checks run against a
            // real asset — threepp_data's xbot.glb is the one it was written
            // for — and the pass writes a photograph of the thing walking.
            std::filesystem::path character;
            // Light the scene from this .hdr / .exr on start, as File > Set Environment does.
            // A document cannot carry a float environment of its own (its images go through the
            // 8-bit ImageLoader), so without this a --screenshot run could never be lit the way
            // the app lights it. Set as the background too, matching the menu's default.
            std::filesystem::path environment;
            // Render the road-acceptance scene to this PNG and exit. Exists so
            // a person — or a tool — can LOOK at what the geometry does before
            // anyone claims it works.
            std::filesystem::path screenshot;
            // Open a shipped example on start, by slug (see ExampleScenes.hpp).
            // Composes with --screenshot: with a document of its own to shoot,
            // --screenshot skips its built-in spline scenario and photographs
            // what is loaded, which is what makes an arbitrary scene reviewable
            // without a new code path per scene.
            std::string example;
            // Camera placements for that pass, as position/target pairs. Empty
            // means one automatically framed three-quarter view. On the command
            // line: --shot=px,py,pz@tx,ty,tz, repeatable.
            struct Shot {
                Vector3 position;
                Vector3 target;
                // Suffix on the file name; the first shot writes the bare path.
                std::string label;
            };
            std::vector<Shot> shots;
            // Seconds of play before the first shot of that pass.
            float settle = 3.f;
            // Keys held for that settle, by name ("W", "SPACE", ...), then
            // released before the shots. A scene whose whole point is that you
            // drive it cannot be reviewed standing still, and pressing a key by
            // hand is not something a capture script can do.
            std::vector<std::string> keys;
            // Keep holding them THROUGH the shots instead of letting go and
            // waiting for the scene to settle. What a manoeuvre looks like
            // halfway through it — a drone mid-turn, and the chase camera coming
            // round with it — is a picture the settled pose cannot show.
            bool holdKeys = false;
            // Timed pass: warm up, then measure this many frames and print
            // median/p95 frame time (plus the Vulkan per-pass GPU breakdown)
            // instead of running interactively. Implies vsync OFF — a
            // present-capped frame time measures the display, not the renderer.
            int bench = 0;
        };

        // The standard editor viewpoints. `User` is any freely orbited angle;
        // the other six are the axis-aligned views every 3D editor puts on the
        // numpad. Public because it names the argument of setViewPreset().
        enum class ViewPreset {
            User,
            Front,
            Back,
            Left,
            Right,
            Top,
            Bottom
        };

        // "px,py,pz@tx,ty,tz" -> two points. The format a document's
        // userData["editorView"] and the command line's --shot both speak,
        // parsed in one place so the two cannot drift. False on anything it
        // cannot read, leaving both outputs untouched.
        [[nodiscard]] static bool parseViewSpec(const std::string& text,
                                                Vector3& position, Vector3& target);

        explicit EditorApp(const Options& options);
        ~EditorApp();

        EditorApp(const EditorApp&) = delete;
        EditorApp& operator=(const EditorApp&) = delete;

        // Runs until the window closes. Returns a process exit code.
        int run();

    private:
        // --- frame ---------------------------------------------------------
        void frame(float dt);
        void drawUi();
        void updateWindowTitle();

        // --- panels (apps/editor/panels/*.cpp) ------------------------------
        void drawMenuBar();
        void drawHierarchy();
        void drawInspector();
        void drawBottomPanel();
        void drawStatusBar();

        // Hierarchy helpers
        void drawHierarchyNode(Object3D& object);
        void drawAddMenu(Object3D& parent);

        // Right-click in the viewport: object actions on whatever was picked,
        // or the Add menu when the click landed on empty space.
        void drawViewportContextMenu();

        // Inspector sections
        void drawObjectSection(Object3D& object);
        void drawTransformSection(Object3D& object);
        void drawMaterialSection(Object3D& object);
        void drawGeometrySection(Object3D& object);
        void drawGeneratorSection(Object3D& object);
        // What the cloud IS (splats, SH degree), where it came from, and the
        // fact that none of it is saved yet — plus the one authoring verb a
        // scan has: the surface bake that makes it collide and return beams.
        // Placement stays the transform section's job like everything else's.
        void drawSplatSection(Object3D& object);
        // Bake the selected cloud's surface NOW, straight into the memo the
        // play session reads. Blocking and visibly so: the bake reparents the
        // cloud and renders the pose loop through the primary view, so the
        // viewport flashes for the ~0.4 s it takes. Reports through the log and
        // the status flash.
        void bakeSplatSurface(Object3D& object);
        void drawInstancingSection(Object3D& object);
        void drawLightSection(Object3D& object);
        void drawLightShadowSection(Object3D& object);
        void drawCameraSection(Object3D& object);
        void drawAnimationSection(Object3D& object);
        void drawJointsSection(Object3D& object);
        // Whether Play simulates this robot as a PhysX articulation, written into
        // userData["articulation"]. Drawn inside the Robot section, PhysX-free.
        void drawArticulationBlock(Object3D& object, Robot& robot);
        void drawScriptSection(Object3D& object);
        void drawPhysicsSection(Object3D& object);
        // Joint authoring: shown for a node carrying a JointConfig (its own
        // scene node — the transform is the joint frame, the parent chain is
        // body A, the other body is picked here by name). NOT the Robot
        // section's joint sliders — that is drawJointsSection above.
        void drawJointAuthoringSection(Object3D& object);
        // Vehicle authoring: shown for a node carrying a VehicleConfig, and —
        // as an invitation — for any node with enough descendant meshes to
        // pick four wheels from. Point at the wheels, press Play, drive; the
        // geometry is derived from the picks unless overridden. See
        // VehicleConfig.
        void drawVehicleSection(Object3D& object);
        // Offered on any node with a SkinnedMesh under it; open once the node
        // is an authored character. Everything it shows about clips is a
        // READOUT of what CharacterConfig::derived matched, with a combo to
        // override the one it got wrong.
        void drawCharacterSection(Object3D& object);
        // Sensor authoring: type, rate, seed and the per-type noise model, all
        // written into userData["sensor"]. Fields for the types you are not on
        // are hidden, never dropped — see SensorConfig. The host gates the type
        // list: a Camera hosts the pinhole sensors (its frustum is theirs),
        // everything else hosts the rest.
        void drawSensorSection(Object3D& object);
        // The migration behind the legacy hint in that section: a pinhole
        // sensor authored on a plain object moves onto a new camera child whose
        // frustum is stamped from the config, in one undo step. The child sits
        // at the host's origin with identity rotation, so the aim the host's
        // transform encoded is exactly the aim the camera wakes up with.
        void moveSensorToCameraChild(Object3D& host);
        // Shown for a spline and, in its point form, for one of its control
        // points — both are ordinary scene nodes, so the section is what tells
        // them apart.
        void drawSplineSection(Object3D& object);
        // Shown for a node carrying a SoundConfig: the file, the playback
        // parameters and — in edit mode only — the audition button.
        void drawSoundSection(Object3D& object);
        // --- sound audition (edit mode only) --------------------------------
        // Hears the authored file without pressing Play. Its own listener and
        // its own Audio, rebuilt from the config on every start, and FLAT: it
        // answers "is this the right file at the right volume", not "how does
        // it sound from over there". The uuid is what identifies the target,
        // since a play/stop or a scene load replaces the whole graph.
        void startAudition(const Object3D& object);
        void stopAudition();
        [[nodiscard]] bool isAuditioning(const Object3D& object) const;
        // Shown for a mesh: whether sound has to get THROUGH it, and what it
        // eats when sound bounces off it. Play traces rays against every
        // flagged mesh (AcousticSurfaceConfig), so this is authored on the
        // walls rather than on the sound.
        void drawAcousticsSection(Object3D& object);
        // Shown for a text mesh (TextConfig): the content and the type
        // parameters, each edit rebuilding the geometry through the same
        // undoable property write every other config section uses.
        void drawTextSection(Object3D& object);
        // Shown for a procedural tree (TreeConfig): the species presets, the
        // seed and the generator's parameters. Edits write the CONFIG only —
        // the trunk and foliage meshes are derived state that syncTreeOverlays
        // regrows to follow it, which is also what makes undo cheap.
        void drawTreeSection(Object3D& object);
        // Shown for a terrain mesh (TerrainConfig): the generator's knobs, the
        // erosion pass behind its own button, and the splat bands. Unlike the
        // tree, the MESH is the truth here — every commit goes through
        // commitTerrain, which re-bakes and carries the sculpt layer across.
        void drawTerrainSection(Object3D& object);
        // One undo entry per terrain parameter edit, holding both generations
        // of the geometry so undo restores byte-identical heights rather than
        // re-deriving them (see the command in InspectorPanel.cpp).
        void commitTerrain(Object3D& object, const editor::TerrainConfig& before,
                           const editor::TerrainConfig& after, const std::string& label);
        // The conveyor twin: shown for a conveyor group and, in its waypoint
        // form, for one of its waypoints (arc centre / segment surface).
        void drawConveyorSection(Object3D& object);
        // Shown for a node carrying a ParticleFieldConfig: the four presets,
        // the structural block that rebuilds the preview field, and one group
        // per representation. Edits write the CONFIG only — the preview field
        // is derived state syncParticleOverlays follows.
        void drawParticleFieldSection(Object3D& object);
        // The PhysX PBD twin: shown for a node carrying a GranularConfig. The
        // grains only exist while playing, so this section is the chute and
        // nothing else.
        void drawGranularSection(Object3D& object);
        // Shown for a group carrying FlockConfig. Authoring only — the birds
        // exist while playing (FlockPlaySession), so this section is the
        // territory and the species knobs and nothing else.
        void drawFlockSection(Object3D& object);
        // `owner` is the object the material hangs off; the slot is identified
        // by (owner uuid, label) whenever it has to outlive the frame.
        void drawTextureSlot(const Object3D& owner, Material& material, const char* label,
                             const std::shared_ptr<Texture>& current,
                             const std::function<void(const std::shared_ptr<Texture>&)>& setter,
                             bool srgb);
        // Tiling / offset / rotation, drawn once for the whole material rather
        // than once per slot. GL feeds a SINGLE uvTransform uniform for every
        // uv1 map (GLMaterials picks the first assigned map by priority and
        // copies ITS matrix), so a per-slot transform would silently not render
        // there. `textures` is every distinct map on the material and every one
        // of them is written.
        void drawUvTransformBlock(Material& material,
                                  const std::vector<std::shared_ptr<Texture>>& textures);
        // The slot row's "..." button: wrap, filtering, anisotropy and colour
        // space for that one texture. Textures are shared instances, so this
        // deliberately changes every material using it - there is no
        // clone-on-edit.
        void drawTextureSettingsPopup(Material& material, const std::shared_ptr<Texture>& texture);

        // What the UV transform block writes. The command stores one of these
        // PER texture, so undo restores each map's own prior transform - a
        // loaded document may well arrive with unequal ones.
        struct UvTransform {
            Vector2 repeat{1, 1};
            Vector2 offset{0, 0};
            Vector2 center{0, 0};
            float rotation = 0;// radians, as Texture stores it
        };
        // What the per-slot popup writes, as one value: undo puts the whole
        // sampling state back, and each widget names the entry it pushed.
        struct TextureSampling {
            TextureWrapping wrapS{TextureWrapping::ClampToEdge};
            TextureWrapping wrapT{TextureWrapping::ClampToEdge};
            Filter minFilter{Filter::LinearMipmapLinear};
            Filter magFilter{Filter::Linear};
            int anisotropy = 1;
            ColorSpace colorSpace{ColorSpace::NoColorSpace};
        };
        [[nodiscard]] static UvTransform uvTransformOf(const Texture& texture);
        [[nodiscard]] static TextureSampling samplingOf(const Texture& texture);
        // The undoable writes behind those two, factored out of the widgets so
        // the self-test drives exactly what a click drives. `label` names the
        // undo entry.
        void applyUvTransform(Material& material,
                              const std::vector<std::shared_ptr<Texture>>& textures,
                              const UvTransform& after, const char* label);
        void applyTextureSampling(Material& material, const std::shared_ptr<Texture>& texture,
                                  const TextureSampling& after, const char* label);
        // Thumbnails are cached per texture; drop the lot. Called when the scene
        // is replaced, alongside every other cache that points into the old one.
        void clearThumbnailCache();

        // Assets / console / sensors tabs
        void drawAssetsTab();
        void drawConsoleTab();
        // Live sensor readout (apps/editor/panels/SensorsPanel.cpp): what is
        // measuring, what it last read, plots of the scalar channels, and the
        // Record toggle. A sensor is invisible without this.
        void drawSensorsTab();

        // Script Editor (apps/editor/panels/ScriptEditorPanel.cpp): a Scripts
        // tab in the bottom panel, holding one inner tab per open script.
        //
        // Several at once, because editing one object's script while reading
        // another's is the normal way to write two things that talk to each
        // other. One per object — the object still carries a single script.
        struct ScriptEditorState;

        // The unsaved marker, the raise-on-selection rule and the
        // close-when-the-script-is-gone rule are per-frame state, so they run
        // here rather than in a tab body — a collapsed panel, or a tab that is
        // not the visible one, must not freeze them. Called before
        // drawBottomPanel(), which draws the tabs.
        void updateScriptEditors();
        // The Scripts tab body: the inner tab bar, and the visible script.
        void drawScriptsTab();
        void drawScriptTab(ScriptEditorState& state);
        // Labels: the outer tab, and one script's inner tab.
        [[nodiscard]] std::string scriptsTabLabel() const;
        [[nodiscard]] static std::string scriptTabLabel(const ScriptEditorState& state);
        // The editor open on `uuid`, or nullptr. Invalidated by any open, so do
        // not hold it across one.
        [[nodiscard]] ScriptEditorState* scriptEditorFor(const std::string& uuid);
        // The script the user is looking at: what Apply, Revert and Ctrl+Enter
        // act on. Null with no scripts open.
        [[nodiscard]] ScriptEditorState* activeScriptEditor();
        // Opens a tab on `object`, or reuses the one already on it. `reveal`
        // brings that tab forward (and the bottom panel with it); an external
        // session syncing in the background passes false, since it must not
        // pull the panel out from under whatever the user was looking at.
        //
        // Reopening an object that already has a tab keeps whatever was being
        // typed — closing a tab is not a decision to throw text away.
        void openScriptEditor(const Object3D& object, bool reveal = true);
        // Normalizes, syntax-checks and commits the buffer as one undo step.
        void applyScriptEditor(ScriptEditorState& state);
        // The same, on whichever script is visible.
        void applyScriptEditor();

        // --- external editing (apps/editor/ExternalScriptEdit.cpp) ----------
        // Tier 1: hand a .py to VS Code. No watcher — every Play recompiles the
        // file, so a file script is already hot.
        void openScriptFileExternally(const std::filesystem::path& file);
        // Which inline source an external session is editing. A behaviour script
        // and a generator live on the same object under different keys, so the
        // session has to know which one it exported — and they commit through
        // different paths (the Script Editor tab vs the Generator's own
        // property write).
        enum class ExternalEditKind {
            Script,
            Generator
        };
        // Tier 2: export the inline source to a scratch .py, open it, and poll
        // it back in through applyScriptEditor() on every save.
        void startExternalEdit(Object3D& object, ExternalEditKind kind = ExternalEditKind::Script);
        // The undoable write behind a generator sync. Returns the normalized text
        // as committed, which is what the poll compares against next time.
        std::string applyGeneratorSource(Object3D& target, const std::string& text);
        void stopExternalEdit(const std::string& why = {});
        void pollExternalEdit(float dt);
        [[nodiscard]] bool externalEditActive() const { return externalEdit_.active; }
        [[nodiscard]] bool externalEditActive(const Object3D& object) const;
        // Writes `<dir>/.vscode/settings.json` when it is absent, so Pylance
        // completes `import threepp` in whatever folder the script lives in.
        void ensureScriptWorkspace(const std::filesystem::path& dir);
        // Where the threepp type stubs are: $THREEPP_PYTHON_STUBS, else the
        // source tree this binary was built from.
        [[nodiscard]] static std::filesystem::path pythonStubDir();
        // Detached, non-blocking and without a console flash. Suppressed in the
        // self-test, which drives everything else about a session.
        void launchExternalEditor(const std::filesystem::path& dir,
                                  const std::filesystem::path& file);

        // A material slot resolved down to the pointers needed to write it.
        // Valid for the frame that built it and no longer — see
        // PendingTextureSlot for the form that survives a file dialog.
        struct TextureSlotTarget {
            Material* material = nullptr;
            std::function<void(const std::shared_ptr<Texture>&)> setter;
            std::shared_ptr<Texture> current;
            std::string slot;
            bool srgb = true;
        };

        // --- editing operations --------------------------------------------
        void newScene();
        void buildTemplateScene();
        void openScene(const std::filesystem::path& path);
        // Open one of the scenes compiled into the binary (ExampleScenes.hpp).
        // Everything openScene does, minus the two things that are about a file:
        // the document keeps no path (so Save prompts Save As) and the slug is
        // not a recent file. Framing is added instead, because an example is
        // something you asked to LOOK at.
        void openExample(const std::string& slug);
        // Put the viewport where the whole document is visible, from wherever
        // the camera currently stands. focusSelected() with the scene as the
        // subject; separate because it must work with nothing selected.
        void frameDocument();
        // How a document asks to be SEEN when it is opened, off the scene root:
        //
        //   userData["editorView"]   "px,py,pz@tx,ty,tz" - camera and orbit target
        //   userData["editorFollow"] "<object name>"     - select it, chase it
        //
        // Called from the OPEN paths only (openScene, openExample, the file on
        // the command line) and never from the scene-replaced listener: Stop
        // restores a snapshot, and a Stop that teleports the camera away from
        // wherever the user drove it is worse than no framing at all.
        //
        // Returns whether a view was applied, which is what tells openExample
        // whether it still has to frame the document itself. A malformed value
        // is a console line and nothing else.
        bool applyDocumentView();
        // How a document asks to be RENDERED, off the same scene root:
        //
        //   userData["render"]  "key=value;key=value" - see RenderConfig
        //
        // Called from the OPEN paths beside applyDocumentView(), and from New,
        // which is a document too. A document that says nothing gets
        // renderDefaults_ — NOT whatever the last document left on the renderer,
        // which is the difference between opening a scene and inheriting one.
        void applyDocumentRender();
        void saveScene();
        void saveSceneAs(const std::filesystem::path& path);
        void importModel(const std::filesystem::path& path);
        // Save-time storage choices. Applied to the document and remembered in
        // the settings file, so the preference outlives the session.
        void setImageStorage(ImageStorage storage);
        void setModelStorage(ModelStorage storage);
        // Turns the selected linked subtree into ordinary scene content, so a
        // save writes it in full. Undoable — it is a userData edit.
        void unlinkSelectedAsset();
        void setEnvironment(const std::filesystem::path& path, bool alsoBackground);
        void clearEnvironment();
        // Where a dropped image goes when nothing pointed at a specific slot:
        // inferred from the file name, else the material's base colour map.
        void assignTextureToSelection(const std::filesystem::path& path);
        void assignTextureToSlot(const std::filesystem::path& path);
        // Loads `path` in the slot's colour space and assigns it as one
        // undoable step. `note` is appended to the console line.
        void applyTextureToSlot(const std::filesystem::path& path,
                                const TextureSlotTarget& target,
                                const std::string& note);
        // Consumes the frame's dropped images against the texture slot rows the
        // inspector just drew, hit-testing them at (mouseX, mouseY). Called at
        // the end of drawUi() with the cursor position; the position is a
        // parameter so the self-test can aim a drop without an OS drag.
        void resolveTextureDrops(float mouseX, float mouseY);
        // Attaches (or clears, with an empty path) a .py on `object`, as one
        // undoable step. Field values already stored for the same file are kept.
        void assignScript(Object3D& object, const std::filesystem::path& path);
        // The same for an audio file (userData["soundFile"]). Authors a default
        // SoundConfig alongside it if the object had none, so a file dropped on
        // an ordinary mesh makes it a sound source in one step.
        void assignSound(Object3D& object, const std::filesystem::path& path);
        // Stores inline source on `object` as one undoable step, clearing any
        // file reference — an object carries one script, in one form. Parameter
        // values survive an edit to the same inline script and are dropped when
        // the form changes, since they belong to the class that exposed them.
        void setInlineScript(Object3D& object, const std::string& source, const std::string& label);
        // Starting point for "New Inline Script": one class, one exposed field,
        // and a header saying what the shape is.
        [[nodiscard]] static std::string inlineScriptTemplate();

        void addObject(const std::shared_ptr<Object3D>& object, Object3D& parent, const std::string& label);

        // Run the generator script `carrier` holds and replace its output with
        // what the script built. One undoable step; nothing is committed if the
        // script raises, because it fills a detached node that is only attached
        // on success. Refused while playing, like every other document edit.
        // False (with a console line) when there is no generator, the build has
        // no Python, or the script failed.
        bool regenerate(Object3D& carrier);
        // Remove a generator AND the output it produced, as one undoable step.
        void clearGenerator(Object3D& carrier);
        // Starting source for a scene generator, offered by the Generator section
        // when nothing is authored yet. Deliberately NOT inlineScriptTemplate():
        // a behaviour script is a class with update(dt) on one object, an
        // authoring script is a module body that builds content.
        [[nodiscard]] static std::string generatorTemplate();
        // A new control point for `spline`, at `index` among its siblings
        // (AddObjectCommand::atEnd appends). Placed midway to the neighbour it
        // is inserted next to, or past the end along the last segment, so the
        // curve visibly changes. Undoable, and the point becomes the selection
        // so the gizmo is already on it.
        void addSplinePoint(Object3D& spline, std::size_t index, const std::string& label);
        // Same contract for a conveyor's waypoints.
        void addConveyorPoint(Object3D& conveyor, std::size_t index, const std::string& label);
        // A wall (diverter / side guide) attached to `conveyor`, and a point
        // inserted into an existing wall at `index` (AddObjectCommand::atEnd
        // appends, extending the last span — the grow-it-point-by-point verb).
        // Both undoable, both select the result.
        void addConveyorWall(Object3D& conveyor, const std::string& label);
        void addConveyorWallPoint(Object3D& wall, std::size_t index, const std::string& label);
        void deleteSelected();
        void duplicateSelected();

        // --- prefabs ---------------------------------------------------------
        // A prefab is an ordinary scene document whose root happens to be a
        // saved subtree. No second format, no registry: every scene the editor
        // writes already doubles as one, and ObjectExporter writes exactly the
        // geometries, materials and textures the subtree references.
        //
        // The begin* pair only opens the dialog; the work happens when it
        // confirms, frames later, against whatever is in the scene by then.
        void beginSavePrefab(Object3D& object);
        void savePrefab(Object3D& object, const std::filesystem::path& path);
        void beginAddPrefab(Object3D& parent);
        // Loads `path`, gives the subtree fresh identity and adds it under
        // `parent` through addObject — so undo, selection and the play gate are
        // the ones every other Add gets.
        void addPrefab(const std::filesystem::path& path, Object3D& parent);
        // Where a prefab dialog opens: the prefab library once one has been
        // used, and the scene directory until then.
        [[nodiscard]] std::string prefabStartDir() const;

        void focusSelected();
        void reparent(Object3D& object, Object3D& newParent);
        // Undo/redo as editor operations rather than raw stack calls: they are
        // document mutations and go through the same Play gate as the rest.
        // The self-test drives commands_ directly where it wants the stack.
        void undo();
        void redo();

        // The gate every document-mutating operation passes through. Play runs
        // on a snapshot and Stop rebuilds the authored scene from it, so an edit
        // made while playing is thrown away on Stop while leaving an undo entry
        // rebound against the restored graph. Worse, PlaySession's contract is
        // that "the editor does not touch the graph while playing" — a delete
        // pulls a node out from under a live PhysX actor. Refusing centrally is
        // what makes that a contract rather than a hope; the disabled menu items
        // are only the visible half. `what` names the action in the console line.
        [[nodiscard]] bool rejectWhilePlaying(const char* what);

        // --- selection / picking -------------------------------------------
        // `instance` is the InstancedMesh sub-instance the pick landed on, when
        // the selection IS an InstancedMesh. It only changes which instance the
        // outline boxes — the selection, the gizmo and every edit still address
        // the whole object, because one instance is not an Object3D and has
        // nothing to carry a transform edit on.
        void selectObject(Object3D* object, std::optional<int> instance = std::nullopt);
        // `dt` is the frame's own delta: the particle previews advance a clock
        // here, and everything else in the pass ignores it.
        void refreshSelectionHelpers(float dt = 0.f);
        // World-space bounds of `instance` of `mesh`: the geometry's own box
        // through matrixWorld * instanceMatrix[instance]. Empty when the index
        // is out of range or the geometry has no bounds to take.
        [[nodiscard]] static Box3 instanceWorldBox(const InstancedMesh& mesh, int instance);

        // --- viewport markers ----------------------------------------------
        // Billboarded SVG icons standing in for objects that draw nothing
        // (cameras, lights), plus the frustum helper for a selected camera.
        // Re-drives loaded Robots from their saved joint values, and revives the
        // frozen placeholders a pre-articulation-block document leaves behind.
        // See RobotConfig.
        void rearticulateRobots(Scene& scene);
        // Drives one joint and records the new pose in userData, so the scene
        // carries the pose it is showing.
        void setJointValue(Robot& robot, std::size_t index, float radians);

        void syncViewportMarkers();
        void syncCameraHelper();
        // Min/max distance circles for the SELECTED positional sound, in the
        // same file and for the same reason as the camera frustum: an authored
        // falloff is otherwise a pair of numbers with no picture.
        void syncSoundRings();
        void clearSoundRings();
        // Anchor cross + hinge/slide axis for the SELECTED joint node, same
        // file and same selected-only rule as the sound rings: the node's
        // transform IS the joint frame, and an axis you cannot see is an axis
        // authored by trial and error.
        void syncJointHelper();
        void clearJointHelper();
        // Wheel rings for the SELECTED vehicle: a circle of the derived radius
        // at each picked wheel, same selected-only rule as the joint helper —
        // the picture that says which meshes the config resolved and what
        // radius it read off them.
        void syncVehicleHelper();
        void clearVehicleHelper();
        void clearViewportMarkers();
        // --- spline overlay (apps/editor/SplineOverlay.cpp) -----------------
        // One Line per spline, sampled from the CatmullRomCurve3 its control
        // points describe. Editor furniture: it lives under the overlay, so it
        // is never saved and never picked.
        void syncSplineOverlays();
        void clearSplineOverlays();
        // --- conveyor overlay (apps/editor/ConveyorOverlay.cpp) -------------
        // One Line per conveyor path, plus the derived-group regeneration —
        // the conveyor twin of the spline overlay pass.
        void syncConveyorOverlays();
        void clearConveyorOverlays();
        // --- particle previews (apps/editor/ParticleOverlay.cpp) ------------
        // One ParticleField per authored node, built through the same
        // ParticleFieldBuild the play session uses and advanced on the editor's
        // own clock so the weather falls while it is authored. Two-tier change
        // detection: the structural key rebuilds, everything else is pushed in
        // place. Vulkan only — the map stays empty on OpenGL.
        void syncParticleOverlays(float dt);
        void clearParticleOverlays();
        // The spawn slab (or a chute's pour mouth) and the flight direction for
        // the SELECTED node, same selected-only rule as the sound rings. Drawn
        // on every backend: where particles are born is an authoring fact.
        void syncParticleHelper();
        // Whether this session can draw a particle field at all, i.e. whether
        // it is running the Vulkan backend. Read by the inspector too, which
        // says so rather than showing an empty preview.
        [[nodiscard]] bool particlePreviewAvailable() const;
        // Density volumes authored in the whole document, counted by the last
        // sync — the inspector's budget warning (ParticleFieldConfig::
        // maxDensityFields) reads it.
        [[nodiscard]] int particleDensityCount() const { return particleDensityCount_; }
        // --- procedural trees (apps/editor/TreeOverlay.cpp) -----------------
        // Regrows the trunk and foliage meshes an authored TreeConfig
        // describes. No editor furniture of its own: unlike the two passes
        // above, everything a tree has to show IS the generated geometry.
        void syncTreeOverlays();
        void clearTreeOverlays();
        // The corner-radius handle: a draggable ball on the selected corner's
        // arc midpoint. Interaction runs in the ImGui frame (it reads the same
        // mouse state picking does); placement rides syncConveyorOverlays.
        // Returns true while a drag owns the mouse, so picking stands down.
        bool updateConveyorRadiusDrag();
        // The drag core, separated so the selftest can drive it without a
        // mouse: maps a world-space ray to a radius via the corner's bisector.
        void applyConveyorRadiusDrag(const Vector3& rayOrigin, const Vector3& rayDirection);
        void beginConveyorRadiusDrag(const Vector3& rayOrigin, const Vector3& rayDirection);
        void endConveyorRadiusDrag();
        // --- baked scan surfaces (apps/editor/SplatSurfaceOverlay.cpp) -------
        // The wireframe of what a scan's surface bake captured, over the scan,
        // while it is authored. Asks the memo and never bakes; hidden for the
        // duration of a Play, which adds its own twin of the same triangles.
        void syncSplatSurfacePreviews();
        void clearSplatSurfacePreviews();
        // --- physics collider overlay (apps/editor/PhysicsDebugOverlay.cpp) --
        // PhysX's own debug lines for every collider in the playing world,
        // drawn as one LineSegments under the overlay. The answer to "where is
        // my collider" being unanswerable without leaving the editor.
        void syncPhysicsDebug();
        void clearPhysicsDebug();
        // --- script debug draw (apps/editor/DebugDrawOverlay.cpp) -----------
        // threepp.editor.draw_line and friends: the segments a playing script
        // asked to see this frame, drained from scripting::debugDraw() into one
        // LineSegments under the overlay. Immediate mode - drained is gone, a
        // paused frame keeps the last picture.
        void syncDebugDraw();
        void clearDebugDraw();
        // --- sensor point cloud (apps/editor/SensorOverlay.cpp) --------------
        // Every playing depth camera's and LIDAR's returns, as one Points under
        // the overlay, coloured by range. Same in-place attribute contract as
        // the collider lines above and for the same reason.
        void syncSensorOverlay();
        void clearSensorOverlay();
        // The object a marker stands for, or nullptr when `hit` is not part of
        // one. Lets a click on an icon select its owner.
        [[nodiscard]] Object3D* markerOwnerOf(Object3D* hit) const;
        // The dock the docked camera renders into: the band beside the bottom
        // panel, under the inspector. False when there is no room for it (the
        // bottom panel is collapsed, or the window is tiny).
        [[nodiscard]] bool cameraDockRect(float& x, float& y, float& w, float& h) const;
        // The selected object as a Camera, perspective or orthographic. Aims
        // the dock when it changes; it is NOT what the dock renders — see
        // dockCamera().
        [[nodiscard]] Camera* selectedCamera() const;
        // The camera the dock renders. Resolved from dockCamera_ every frame
        // rather than cached, which is what carries it across a scene replace
        // (play/stop rebuilds every camera behind the same uuid) and lets a
        // deleted camera fall back to another without any bookkeeping.
        [[nodiscard]] Camera* dockCamera() const;
        // Every camera in the scene, in hierarchy order: what the dock's picker
        // offers, and where dockCamera() finds its default.
        [[nodiscard]] std::vector<Camera*> sceneCameras() const;
        // Aims the dock. nullptr is the explicit "None" — the dock stays empty
        // instead of falling back to the first camera in the scene.
        void setDockCamera(Camera* camera);
        // The picker in the dock's corner. Drawn as a real ImGui window (the
        // rest of the dock is background-drawlist), so it also stops a click on
        // the dock from picking through into the scene behind it.
        void drawCameraDockPicker();
        // Points the dock's Vulkan secondary view at the dock camera and at
        // this frame's dock rect. Must run BEFORE Renderer::render(), which is
        // where the view is actually recorded and composited. A no-op on OpenGL,
        // whose dock is a scissored second render in renderCameraPreview().
        void syncCameraDockPane();
        // Renders the docked scene camera into that dock; drawUi frames it and
        // draws the picker via preview_. On Vulkan the pixels are already there
        // (see syncCameraDockPane) and this only fills preview_ in.
        void renderCameraPreview();
        // Selects what the ray under (mouseX, mouseY) hits and returns it, or
        // nullptr on a miss. A miss deselects unless deselectOnMiss is false —
        // the context menu keeps the selection and shows the Add menu instead.
        Object3D* pickAt(float mouseX, float mouseY, bool deselectOnMiss = true);
        [[nodiscard]] Object3D* resolveSelectable(Object3D* hit) const;

        // --- viewport camera -------------------------------------------------
        // The camera the viewport is currently seen through: the perspective
        // one, or the orthographic one while an ortho view is on. Everything
        // that projects or unprojects (render, pick, gizmo, markers) goes
        // through this rather than naming camera_ directly.
        [[nodiscard]] Camera& viewCamera();
        // Swaps the projection, preserving what is framed: the ortho frustum is
        // sized to what the perspective camera sees at the orbit distance, and
        // the reverse on the way back.
        void setOrthographic(bool ortho);
        [[nodiscard]] bool orthographic() const { return orthographic_; }
        // Points the view down a world axis without changing what it looks at.
        // `User` only clears the label — an axis view is not a mode, it is a
        // place the camera happens to be standing.
        void setViewPreset(ViewPreset preset);
        [[nodiscard]] ViewPreset viewPreset() const { return viewPreset_; }
        [[nodiscard]] static const char* viewPresetLabel(ViewPreset preset);
        // Unit vector from the orbit target towards where the camera stands in
        // that view. The pole views carry a hair of tilt so `lookAt` and the
        // orbit spherical never hit their degenerate case.
        [[nodiscard]] static Vector3 viewPresetDirection(ViewPreset preset);
        // Drops the label back to `User` once the view has been orbited off its
        // axis, and keeps the grid facing the viewer in the axis views.
        void updateViewPreset();
        void updateGridPlacement();
        // Rebuilds the orbit and transform controls against whichever camera is
        // active. Both hold a Camera& for life, so switching projection means
        // building new ones.
        void bindViewportControls();
        // --- view gizmo (apps/editor/ViewGizmo.cpp) --------------------------
        // The camera-orientation gizmo every 3D editor keeps in a viewport
        // corner, after three.js editor's: a ball at each end of the three
        // world axes, drawn where the axes actually point. Clicking a ball
        // swings the view onto that axis; clicking the axis the camera already
        // stands on heads for the far end, the same idiom as Ctrl on the
        // numpad keys. Drawn with ImGui's background draw list - no second
        // scene, no second render pass, the same picture on either backend.
        void drawViewGizmo();
        // Starts the swing. The projection is left alone here - the gizmo's
        // click handler forces orthographic first, the numpad's policy - so
        // the tween itself is usable from either projection.
        void startViewTween(ViewPreset preset);
        // One frame of it: slerp the orbit direction, keep the distance and
        // target. Runs before orbit_->update(), which re-derives its spherical
        // from wherever the camera stands - the same contract setViewPreset
        // leans on, just spread over a third of a second.
        void updateViewTween(float dt);
        // --- viewport chrome (apps/editor/ToolPalette.cpp) -------------------
        // The controls that used to be a toolbar, as viewport furniture drawn
        // with the view gizmo's background-draw-list brush and interaction
        // rules. The palette: Select/Move/Rotate/Scale, the space toggle and
        // Snap, stacked top-left. The transport: Play/Pause/Stop as a pill
        // top-centre. The viewpoint picker: a small real-ImGui window under
        // the view gizmo (a combo is not worth reinventing in a draw list).
        void drawToolPalette();
        void drawTransportBar();
        void drawViewpointPicker();
        // --- follow selection -----------------------------------------------
        // Chase camera. While it is on and something is selected, every frame
        // walks the orbit target towards the selection's world position and
        // carries the camera with it, keeping the offset the user orbited to —
        // in the SUBJECT'S HEADING FRAME, so a body that turns is chased round
        // the corner rather than watched flying sideways out of frame. Works in
        // both projections (a parallel projection translates, see updateFollow)
        // and, above all, while playing — chasing a body the physics is moving
        // is the point.
        void setFollowSelection(bool follow);
        [[nodiscard]] bool followSelection() const { return followSelection_; }
        // One frame of that chase. Deselecting pauses it (there is nothing to
        // chase) and reselecting resumes; the approach is exponential rather
        // than a hard lock, because a hard lock on a physics body reads as
        // jitter.
        void updateFollow(float dt);
        // Sets the ortho frustum to `height` world units tall at the current
        // aspect, with zoom reset.
        void setOrthoHeight(float height);
        // How far back from `from` an ortho camera has to stand to keep the
        // whole document in front of its near plane. Ortho framing does not
        // depend on the distance, only the clipping does.
        [[nodiscard]] float sceneClearDistance(const Vector3& from) const;
        // World units per screen pixel at `world`, for whichever projection is
        // active. Constant-screen-size overlays size themselves with this.
        [[nodiscard]] float viewportWorldPerPixel(const Vector3& world) const;

        // --- play ----------------------------------------------------------
        void startPlay();
        void togglePause();
        void stopPlay();
        [[nodiscard]] bool isPlaying() const;

        // --- vehicle teleop -------------------------------------------------
        // While a played scene has vehicles, W/S/A/D and SPACE drive them:
        // polled every frame before the sessions step, pushed through
        // PhysicsPlaySession::driveVehicles. Only writes controls while a key
        // is actually held (plus one release), so a script driving the same
        // vehicle through its handle is not overwritten by silence. The
        // transmission stays automatic — teleop only ever selects
        // forward/reverse.
        void updateVehicleTeleop(float dt);

        // --- character teleop ------------------------------------------------
        // While a played scene has characters, W/S walk and backpedal along the
        // VIEW's forward, A/D strafe across it, Shift runs and Space jumps —
        // and the character turns to face the way the camera looks, which is
        // what makes a locomotion pack's strafe and backward clips play at all.
        // Polled every frame before the sessions step, pushed through
        // CharacterPlaySession::drive.
        void updateCharacterTeleop(float dt);
        // Keeps the played character in frame: the orbit target chases it and
        // the camera rides along rigidly, so the user's own orbiting, panning
        // and zooming still compose on top. Deliberately NOT the heading-
        // rotating updateFollow — a character that faces the camera and a
        // camera that rotates with the character chase each other in a circle.
        // Yields entirely while the user's own Follow Selection is on.
        void updateCharacterCamera(float dt);
        // Where the view stood before the chase above took it over, and put
        // back by Stop. The chase engages BY ITSELF (a character in the played
        // scene is enough), so unlike Follow Selection it is not a standing
        // choice the user made — leaving the camera wherever the character
        // wandered to would be an automatic behaviour quietly rewriting editor
        // state. Stop restores the scene; it restores the view with it.
        void restoreCharacterCamera();
        // The yaw the viewport camera is looking along, three.js convention
        // (atan2(x, z)). What "forward" means to the character teleop.
        // Non-const only because viewCamera() is (it picks the live camera).
        [[nodiscard]] float viewYaw();

        // --- animation preview ---------------------------------------------
        // Edit-mode preview of one clip on one subtree. Every touched value
        // is recorded up front and put back on stop; no undo entries appear.
        void startAnimationPreview(Object3D& root, const std::string& clipName,
                                   bool loop, float speed);
        // restore = false discards without writing (the nodes are gone, e.g.
        // after a scene replace).
        void stopAnimationPreview(bool restore = true);
        [[nodiscard]] bool isPreviewing(const Object3D& root) const;

        // --- async import --------------------------------------------------
        // importModel only enqueues; pollImports runs one background load at
        // a time (the loaders share global image-decoder state) and finalizes
        // finished ones on the main thread.
        void pollImports(float dt);
        void drawImportToast();
        // The modal that asks for a xacro's declared arguments. Drawn with the
        // other dialogs; it has to answer before pollImports can launch the
        // worker, which is why the queue entry waits in argPrompt_ meanwhile.
        void drawArgPrompt();
        void flashStatus(std::string message);

        // --- misc ----------------------------------------------------------
        int runSelfTest();
        int runScreenshot();
        // --screenshot over the document that is already open (a scene.json on
        // the command line, or --example). Honours --play / --seconds / --shot.
        int runSceneScreenshot();
        // --bench: a timed pass over whatever is open. Honours --play/--keys,
        // warms up for --seconds, then measures --bench frames and prints
        // median/p95 CPU frame time and (on Vulkan) the per-pass GPU medians.
        int runBench();
        // One PNG of whatever the renderer last produced, right way up for the
        // backend in use.
        bool shootTo(const std::filesystem::path& path);
        void handleShortcuts();
        void handleFileDrop(const std::vector<std::string>& paths);
        void log(const std::string& message);
        void logWarnings();
        // Says, in the console and the status bar, that any splat cloud in the
        // scene is about to be dropped by `action` ("Play", "Save"). Splat
        // clouds are not serialized yet, and both operations go through
        // ObjectExporter — Play because Stop restores from a snapshot in the
        // same format the save file uses. Losing the object is accepted for
        // now; losing it silently is not. Returns how many were found.

        void applyGizmoMode();
        // Whether the transform handles belong on screen at all: something is
        // selected, the toolbar is not in Select mode, and play is stopped.
        // Play runs on a snapshot the simulation owns, so a gizmo inviting a
        // transform edit on a moving body is an invitation to nothing — the
        // edit would be refused, and dragging handles across a falling crate
        // is not a thing anyone means to do. One predicate, because
        // applyGizmoMode() and refreshSelectionHelpers() both decide it and a
        // gizmo that comes back for one frame is a gizmo that came back.
        [[nodiscard]] bool gizmoActive() const;
        // Whether the AUTHORING LAYER belongs on screen at all: the selection
        // outline, the outlined instance, the marker icons and the selected
        // camera's frustum. They are editor concepts — they say what you are
        // editing — so Play takes them away with the gizmo and the viewport
        // shows the scene, which is what the button is for. Deliberately NOT
        // everything the overlay holds: the sensor point cloud is play DATA, the
        // collider lines are a debug view that only means anything while
        // playing, and the grid and the origin axes are View-menu preferences a
        // user set on purpose.
        [[nodiscard]] bool authoringVisible() const;
        // Applies it to every node that carries it. Called from the frame loop
        // and from selectObject rather than toggled once at Play, because
        // picking stays live while playing: a selection made mid-play builds
        // NEW outline nodes, and they have to arrive hidden.
        void applyAuthoringVisibility();
        void loadSettings();
        void persistSettings();
        [[nodiscard]] float scale() const { return contentScale_; }

        // Panel sizes in device pixels. The unscaled values are a user
        // preference (draggable, persisted); everything that lays out against
        // a panel goes through these.
        [[nodiscard]] float hierarchyPx() const;
        [[nodiscard]] float inspectorPx() const;
        // Height of the open bottom panel, clamped to what the window can
        // actually spare (bottomHeightLimit()) so shrinking the window cannot
        // leave the editor all panel and no viewport.
        [[nodiscard]] float bottomPanelPx() const;
        // The tab strip that is left when the panel is collapsed.
        [[nodiscard]] float collapsedBottomPx() const;
        // What the side panels have to keep clear above the status bar: the
        // panel plus its splitter when open, the collapsed strip when not.
        [[nodiscard]] float bottomBandPx() const;
        // Largest bottom panel height, unscaled, for the current window.
        [[nodiscard]] float bottomHeightLimit() const;

        // The grab strip both splitters below are made of. `sign` is +1 when
        // dragging along the axis grows the value, -1 when it shrinks it.
        void drawSplitterStrip(const char* id, float x, float top, float width, float height,
                               bool horizontal, float& value, float sign, float lo, float hi);
        // Vertical drag handle beside a side panel. `x` is the strip's left
        // edge and `sign` is +1 when dragging right widens the panel (left-hand
        // panels), -1 when it narrows it (right-hand panels).
        void drawSplitter(const char* id, float x, float top, float height,
                          float& width, float sign);
        // The horizontal twin, along the top edge of the bottom panel: dragging
        // up makes it taller.
        void drawHeightSplitter(const char* id, float x, float top, float width,
                                float& height);

        // Undo-friendly ImGui helpers: begin a transaction when a widget is
        // activated and close it when the edit finishes, so a drag collapses to
        // one undo step.
        void beginEditIfActivated();
        void endEditIfDeactivated();

        // --- state ---------------------------------------------------------
        Options options_;

        Canvas canvas_;
        std::unique_ptr<Renderer> renderer_;
        // Two cameras, one viewport: only one is ever rendered with, and every
        // projection-aware path asks viewCamera() which. They look at the same
        // orbit target and hand their framing over on each switch. The ortho
        // frustum is sized in world units rather than by a negative near plane,
        // so the camera is pushed clear of the scene when an axis view is
        // entered — in a parallel projection the distance only sets clipping.
        PerspectiveCamera camera_;
        OrthographicCamera ortho_;
        std::unique_ptr<OrbitControls> orbit_;
        bool orthographic_ = false;
        ViewPreset viewPreset_ = ViewPreset::User;
        // The view gizmo's animated snap. `from` is the unit orbit direction
        // at the click; the flight slerps it onto the preset's axis while the
        // distance and target stay the orbit's own. Retired by any explicit
        // view set (setViewPreset) or by a viewport drag mid-flight.
        struct ViewTween {
            bool active = false;
            float t = 0.f;
            Vector3 from;
            ViewPreset preset = ViewPreset::Front;
        };
        ViewTween viewTween_;
        // Whether the pointer is on the gizmo this frame. The pick gate and
        // the orbit's mouse capture both stand down while it is - the gizmo
        // is background furniture ImGui's own capture knows nothing about.
        bool viewGizmoHovered_ = false;
        // Its twin for the tool palette in the opposite corner, consumed by
        // the same two gates (ioCapture_ and the pick), for the same reason.
        bool toolPaletteHovered_ = false;

        // --- terrain sculpt (TerrainSculptTool.cpp) --------------------------
        // The Sculpt tool is a mode of the palette, not of the transform gizmo:
        // it edits the MESH, not the transform, so it sits beside gizmoMode_
        // rather than inside it.
        bool sculptTool_ = false;
        editor::TerrainBrush brush_;
        // Whether the pointer is over the selected terrain this frame. Consumed
        // by ioCapture on the NEXT event, which is how the press is owned before
        // OrbitControls ever sees it — the palette's idiom exactly.
        bool sculptHover_ = false;
        Vector3 sculptHoverLocal_;
        struct SculptStroke {
            bool active = false;
            std::string uuid;
            int dim = 0;
            // The whole Y column as it stood at press. One stroke, one undo
            // entry; the release diffs this to a tight rect.
            std::vector<float> before;
            std::vector<float> heights;
            float flattenTarget = 0.f;
            editor::TerrainSculpt::Rect rect;
        } sculptStroke_;

        [[nodiscard]] bool sculptArmed() const;
        [[nodiscard]] Object3D* sculptTarget() const;
        // True while a stroke owns the pointer, or the frame before one starts.
        [[nodiscard]] bool sculptOwnsMouse() const;
        void updateSculpt();
        void endSculptStroke();
        void syncBrushRing();
        // Follow Selection (View menu, Shift+F). Session state on purpose: it
        // belongs to what is open and what is selected, not to the editor's
        // saved preferences.
        bool followSelection_ = false;
        // The heading the chase last placed the camera with: the followed
        // object's yaw about world up, exponentially smoothed. Kept as state
        // because it is BOTH ends of a frame — the offset is read back out of
        // the camera through this angle and written back through the new one,
        // which is what lets the user orbit while it follows.
        //
        // followHeadingFor_ is the subject it belongs to, by uuid: a different
        // subject SNAPS (reading the offset through the new heading leaves the
        // camera exactly where it stands, so selecting something that happens to
        // face east does not fling the view). By uuid rather than by pointer
        // because Stop rebuilds the graph, and the same subject across that swap
        // is the same chase.
        float followHeading_ = 0.f;
        std::string followHeadingFor_;
        // Whether the document that is open placed the camera itself, through
        // userData["editorView"]. Only the --screenshot pass asks: a considered
        // vantage is not something to overwrite with an automatic framing.
        bool documentView_ = false;
        // The renderer as the editor set it up, captured once in the
        // constructor. Two jobs, both about keeping a saved document honest: it
        // is what a document that carries no render block opens with, and it is
        // the baseline a save is written as a difference FROM — so a scene that
        // never touched the Renderer Settings panel saves no render block at
        // all, and one that dialled in fog saves the fog and nothing else.
        RenderConfig renderDefaults_;

        SceneDocument document_;
        Selection selection_;
        CommandStack commands_;
        PlayController play_;
        EditorSettings settings_;
        std::filesystem::path settingsPath_;

        // Editor-only scene content.
        std::shared_ptr<Group> overlay_;
        std::shared_ptr<Object3D> grid_;
        // The brush cursor: a polyline conformed to the terrain, updated in
        // place. Lives in overlay_ so it is excluded from export.
        std::shared_ptr<Line> brushRing_;
        std::shared_ptr<Object3D> axes_;
        std::shared_ptr<BoxHelper> selectionBox_;
        // A selected InstancedMesh outlines the ONE instance that was picked,
        // not the whole cloud — a box around 500 scattered rocks says nothing.
        // Box3Helper keeps a reference to the box, so instanceBox_ must outlive
        // the helper (both are members; the helper is also dropped whenever the
        // selection changes) and is refreshed each frame in
        // refreshSelectionHelpers so the outline tracks a moving instance.
        // Set by the Generator section's button, consumed once at the top of the
        // next frame. Regenerate replaces the node the panel is drawing from, and
        // the selection re-resolve that follows must not run inside the ImGui tree
        // that is reading it. By uuid, because a play/stop in between replaces the
        // graph — the same reason scriptTargetUuid_ is one.
        std::string pendingRegenerate_;
        // Same deferral, for Clear — it removes the output node outright.
        std::string pendingGeneratorClear_;
        std::optional<int> selectedInstance_;
        Box3 instanceBox_;
        std::shared_ptr<Box3Helper> instanceOutline_;
        std::unique_ptr<TransformControls> gizmo_;

        // Marker icons. One node per owner, all parented to markers_, which is
        // itself part of the editor-only overlay.
        struct ViewportMarker {
            Object3D* owner = nullptr;
            std::shared_ptr<Object3D> node;
            std::vector<std::shared_ptr<MeshBasicMaterial>> materials;
            // Which glyph this was built from (ViewportMarkers.cpp's file-local
            // Icon, as an int so this header stays free of it). What an object IS
            // can change under a live marker — authoring a sensor on a camera
            // changes its icon — so the marker is rebuilt when it no longer
            // matches rather than showing yesterday's kind.
            int icon = -1;
        };
        std::shared_ptr<Group> markers_;
        std::vector<ViewportMarker> viewportMarkers_;
        // Frustum of the selected camera. Holds a reference to that camera, so
        // it is torn down whenever the selection or the scene changes.
        std::shared_ptr<CameraHelper> cameraHelper_;
        Object3D* cameraHelperFor_ = nullptr;
        // Distance rings for the selected positional sound. One LineSegments
        // under the overlay, rebuilt when the key below changes — the sound's
        // UUID (not its address: a play/stop replaces the graph) plus the two
        // radii it is drawn from.
        std::shared_ptr<LineSegments> soundRings_;
        std::string soundRingsKey_;
        // Axis + anchor helper for the selected joint node. Same lifetime and
        // keying rules as the sound rings above (uuid + the fields the picture
        // is built from); placed every frame at constant screen size.
        std::shared_ptr<LineSegments> jointHelper_;
        std::string jointHelperKey_;
        // Wheel rings for the selected vehicle. Same lifetime and keying rules
        // as the joint helper (uuid + the numbers the picture is built from).
        std::shared_ptr<LineSegments> vehicleHelper_;
        std::string vehicleHelperKey_;
        // The other half of authoringVisible(): a --screenshot pass over a
        // document has no user and nothing being authored, so the whole layer is
        // off for its duration. One flag instead of the four hand-hidden nodes
        // it used to set, and the same flag the play gate reads.
        bool hideAuthoring_ = false;

        // Curve overlays. Same lifetime rules as the markers above: the owner
        // is a raw pointer into the current graph, the whole list is dropped
        // when the scene is replaced, and entries whose owner is absent from
        // this frame's walk are retired. `hash` is over everything the curve is
        // built from (point count, positions, config), so a dragged control
        // point rebuilds the line and nothing else does.
        struct SplineOverlay {
            Object3D* owner = nullptr;
            std::shared_ptr<Line> line;
            std::shared_ptr<LineBasicMaterial> material;
            std::size_t hash = 0;
            // Vertices the position attribute can hold. The attribute is only
            // replaced when the curve outgrows it — see writeSamples().
            int capacity = 0;
        };
        std::shared_ptr<Group> splines_;
        std::vector<SplineOverlay> splineOverlays_;

        // Conveyor path overlays — same lifetime rules as the spline ones. The
        // hash additionally covers each waypoint's own config (corner radius,
        // segment surface), which lives on the waypoint node rather than the
        // owner. `aids` carries the design helpers — travel-direction chevrons
        // and the selected rounded corner's derived centre + tangent spokes —
        // rebuilt on its own key since selection is not part of the hash.
        struct ConveyorOverlay {
            Object3D* owner = nullptr;
            std::shared_ptr<Line> line;
            std::shared_ptr<LineBasicMaterial> material;
            std::size_t hash = 0;
            int capacity = 0;
            std::shared_ptr<LineSegments> aids;
            std::size_t aidsKey = 0;
            int aidsCapacity = 0;
        };
        std::vector<ConveyorOverlay> conveyorOverlays_;

        // Procedural trees. Same lifetime rules as the two above, but pure
        // bookkeeping — a tree owns no editor-side node, so there is nothing
        // to retire but the record. Strings rather than a decoded config so
        // this header stays clear of the vegetation generator: `encoded` is
        // the userData entry the meshes were last grown from, and comparing it
        // is how a frame where nothing moved costs nothing. The texture key is
        // tracked apart because the atlases are redrawn on a slower clock than
        // the geometry — see TreeOverlay.cpp.
        struct TreeOverlay {
            Object3D* owner = nullptr;
            std::string encoded;
            std::string textureKey;
            std::string wantedTextureKey;
        };
        std::vector<TreeOverlay> treeOverlays_;
        std::shared_ptr<Group> conveyors_;

        // Particle-field previews, keyed by the AUTHORED NODE'S uuid rather
        // than by pointer: unlike the overlays above, an entry has to survive
        // being looked up from a frame in which the graph was rebuilt. The two
        // keys are the two tiers — `structural` is what forces a destroy and
        // rebuild (the churn contract), `mutable` is the encoded config and
        // decides whether new parameters have to be pushed at all.
        struct ParticlePreview {
            std::shared_ptr<ParticleField> field;
            std::string structuralKey;
            std::string mutableKey;
        };
        std::shared_ptr<Group> particles_;
        std::unordered_map<std::string, ParticlePreview> particlePreviews_;
        // The clock the previews run on. Wall time while editing, frozen for
        // the duration of a Play (the previews are parked then, and the play
        // session owns its own deterministic clock).
        float particleTime_ = 0.f;
        int particleDensityCount_ = 0;
        // Spawn slab + flight arrow for the selected particle or granular node.
        // Same lifetime and keying rules as the joint helper.
        std::shared_ptr<LineSegments> particleHelper_;
        std::string particleHelperKey_;

        // The corner-radius handle: one ball, re-aimed at whichever corner
        // waypoint is selected (see ConveyorOverlay.cpp). Dragging it along
        // the corner's bisector writes the waypoint's cornerRadius through the
        // same property command the inspector uses, transaction-coalesced so
        // the whole drag is one undo entry.
        std::shared_ptr<Mesh> conveyorRadiusHandle_;
        struct ConveyorRadiusDrag {
            bool active = false;
            std::string conveyorUuid;
            std::string waypointUuid;
            float grabOffset = 0.f;// bisector distance at grab minus the handle's
        };
        ConveyorRadiusDrag radiusDrag_;

        // Physics collider overlay. The line buffer PhysX hands out changes
        // size every frame, so the attribute is rewritten in place and only
        // replaced when it is outgrown — same contract as the spline curves
        // above, for the same reason (the renderer caches GPU buffers by
        // attribute identity). Null whenever the view is off, play is stopped,
        // or the scene was replaced under it.
        std::shared_ptr<PhysicsPlaySession> physics_;
        // Belts in the world physics_ builds + the visual motion on the derived
        // meshes. Registered right after physics (start order), which also puts
        // its stop BEFORE physics' — the world is still alive to unregister from.
        std::shared_ptr<ConveyorPlaySession> conveyorSession_;
        // The fields an authored particle node runs during Play — the previews
        // above are parked while it does. Kept as a member for the selftest and
        // the status readout, like the sessions beside it; registered in every
        // build, since a field wants a renderer rather than a world.
        std::shared_ptr<ParticleFieldPlaySession> particleSession_;
        // The PBD grains an authored chute pours during Play, in the world
        // physics_ built. Registered right after the fields above and inside
        // the PhysX guard, so its stop comes before physics' — the world is
        // still alive to release the particle actor into.
        std::shared_ptr<GranularPlaySession> granularSession_;
        // The capsule controllers an authored character walks on during Play,
        // in the world physics_ built. Registered inside the PhysX guard after
        // the grains, so its stop comes first and the controller manager is
        // released while its scene is still alive.
        std::shared_ptr<CharacterPlaySession> characterSession_;
        // Scanned surfaces made solid and sensable during Play. Registered
        // inside the PhysX guard right after the grains, for the same reason:
        // its colliders are cooked into the world physics_ built, and stopping
        // in reverse order tears them down first.
        std::shared_ptr<SplatSurfacePlaySession> splatSurfaceSession_;
        // The bake memo, shared between that session and the inspector's "Bake
        // now" — which is why it is the APP's and not the session's (see
        // SplatSurfaceCache.hpp). shared_ptr so this header can leave the type
        // incomplete.
        std::shared_ptr<SplatSurfaceCache> splatSurfaces_;
        // The last bake's stats line and the node it was for, so the inspector
        // can show what it produced without re-reading a mesh it does not own.
        std::string splatBakeNode_;
        std::string splatBakeStats_;
        // Draw the baked surface over the scans that carry one. A view state
        // like physicsDebug_ — never serialized, never on the undo stack, and
        // not a document property (SplatSurfaceOverlay.cpp).
        bool splatSurfacePreview_ = false;
        // One wireframe per previewed node, keyed by the node's uuid and by the
        // MEMO's key: a hit says the mesh is current, not that it is the same
        // mesh, so the geometry is rebuilt whenever the key moves.
        struct SplatSurfacePreview {
            std::shared_ptr<LineSegments> mesh;
            std::string key;
            std::size_t triangles = 0;// what the memo held when it was built
        };
        std::unordered_map<std::string, SplatSurfacePreview> splatSurfacePreviews_;
        std::shared_ptr<LineSegments> physicsDebugLines_;
        int physicsDebugCapacity_ = 0;
        bool physicsDebug_ = false;
        // Has a script polled the keyboard during THIS play session? Set by the
        // is_key_down provider, cleared by startPlay. While it is set, the plain
        // (unmodified) editor shortcuts yield to the script — teleop keys and
        // the gizmo/viewpoint bindings are the same keys, and both acting on one
        // press means driving the robot also retargets the gizmo. Declared
        // outside the Python guard because startPlay and handleShortcuts read it
        // unconditionally; always false without Python, so the yield never fires.
        bool scriptsPolledKeys_ = false;
        // A vehicle is being played this frame: the teleop keys are live, so
        // the plain editor shortcuts yield until Stop — the same rule (and the
        // same keys) as a script that polls the keyboard. Recomputed every
        // frame by updateVehicleTeleop; always false without PhysX.
        bool vehicleDriving_ = false;
        // Teleop wrote controls last frame, so one all-keys-released frame
        // still writes the zeros (and nothing after it does).
        bool vehicleTeleopActive_ = false;
        // A character is being played this frame: same rule and same keys as
        // the vehicle above, plus Shift to run and Space to jump. Unlike the
        // vehicle's, the demand is written EVERY frame — letting go of W has to
        // decelerate the character to a stop, and silence would leave it
        // sprinting. Recomputed every frame; always false without PhysX.
        bool characterDriving_ = false;
        // Teleop wrote a demand last frame, so one all-keys-released frame
        // still writes the zeros — and nothing after it does. Without the
        // second half, a silent frame would overwrite whatever else is driving
        // the character (a script, a test) with "stand still" forever.
        bool characterTeleopActive_ = false;
        // Set the first frame the chase camera actually moves the view, so a
        // played scene with no character never touches it (see
        // restoreCharacterCamera).
        bool characterCameraSaved_ = false;
        Vector3 characterCameraPosition_;
        Vector3 characterCameraTarget_;

        // Script debug draw. Same in-place-rewrite contract as the collider
        // lines above, plus a colour attribute (each call picks its own). No
        // toggle: the view is on exactly when a playing script draws, and an
        // empty frame hides it.
        std::shared_ptr<LineSegments> debugDrawLines_;
        int debugDrawCapacity_ = 0;
        bool debugDrawWarned_ = false;

        // Sensors authored on scene objects. The session is kept as a member for
        // the same reason physics_ is: the readout and the point-cloud overlay
        // both read what it built.
        //
        // sensorRig_ is where the vision sensors' nodes are parented. It is
        // editor-only (so they are never saved, picked or given a marker) and
        // deliberately NOT a child of overlay_, because the overlay is hidden for
        // the duration of every scan — a depth camera aimed at the viewport grid
        // otherwise measures the grid. The point cloud, being furniture, IS under
        // the overlay and so is correctly invisible to the sensors.
        std::shared_ptr<SensorPlaySession> sensors_;
        std::shared_ptr<Group> sensorRig_;
        std::shared_ptr<Points> sensorCloud_;
        int sensorCloudCapacity_ = 0;
        bool sensorCloudVisible_ = true;

#ifdef THREEPP_WITH_AUDIO
        // Sounds authored on scene objects, played back during Play. Kept as a
        // member for the status readout and the selftest, like physics_.
        std::shared_ptr<AudioPlaySession> audio_;
        // The audition's own engine, opened on the first audition and kept for
        // the rest of the session (opening a device costs tens of ms). DECLARED
        // BEFORE the sound so it is DESTROYED AFTER it — an ma_sound must not
        // outlive its ma_engine.
        std::unique_ptr<AudioListener> auditionListener_;
        std::unique_ptr<Audio> auditionSound_;
        // No device on this machine: said once, and the button stays disabled
        // rather than re-trying (and re-logging) on every click.
        bool auditionUnavailable_ = false;
#endif
        // Which object is being auditioned, by uuid — outside the #ifdef so the
        // inspector's "nothing is auditioning" branch needs no second gate.
        std::string auditionUuid_;
        // --bench with THREEPP_BENCH_DISABLE=ui: skip the ImGui pass so the
        // frame time measures the renderer alone. Never set outside runBench().
        bool benchSkipUi_ = false;

        Raycaster raycaster_;
        std::unique_ptr<ImguiContext> ui_;
        IOCapture ioCapture_;
        // The handler outlives the gizmo (rebuilt on every projection change);
        // each rebuild takes a fresh subscription on it.
        std::function<void(Event&)> gizmoDragHandler_;
        Subscription gizmoDragSub_;

        // Gizmo state
        std::string gizmoMode_ = "translate";
        bool gizmoWorldSpace_ = false;
        bool snapEnabled_ = false;
        bool gizmoDragging_ = false;
        SetTransformCommand::Trs gizmoBefore_;

        // File dialogs
        FileBrowser fileBrowser_;
        enum class PendingDialog {
            None,
            Open,
            SaveAs,
            ImportModel,
            Environment,
            Texture,
            Script,
            Sound,
            // A subtree on its way out to a document of its own, and one on its
            // way back in. Both carry their subject as a uuid below, because
            // either dialog can outlive the object it was opened for.
            SavePrefab,
            AddPrefab,
            // Where sensor recordings go. The browser has no directory mode, so
            // this is a Save dialog whose PARENT directory is what gets used —
            // the file name the user types is ignored (one CSV per sensor, named
            // after the sensor).
            RecordDir
        };
        PendingDialog pendingDialog_ = PendingDialog::None;
        bool environmentAsBackground_ = true;
        // Which object a script file dialog is filling. Resolved by uuid rather
        // than pointer: the dialog spans frames, and a Play/Stop in between
        // replaces the whole graph.
        std::string scriptTargetUuid_;
        // Same contract, for the Sound section's "Load..." button.
        std::string soundTargetUuid_;
        // Which subtree "Save as Prefab..." is writing out.
        std::string prefabSourceUuid_;
        // Where an instantiated prefab will be parented. EMPTY means the scene
        // root — not the scene's own uuid, because Play/Stop replaces the whole
        // scene object and the root of the moment is the one that is meant.
        std::string prefabTargetUuid_;

        // Which material slot an inspector "Load..." button is filling. This
        // cannot be the raw TextureSlotTarget the drop path below uses: that one
        // lives for a single frame, while a file dialog spans many, and by the
        // time it comes back the material may be gone (a Play/Stop rebuilds the
        // whole graph, a delete takes the object with it). Recorded the way the
        // script dialog records its target — by uuid — and re-resolved against
        // the live scene on confirmation.
        struct PendingTextureSlot {
            std::string objectUuid;
            std::string slot;
        };
        PendingTextureSlot pendingTextureSlot_;

        // Where each texture slot row landed on screen this frame, so a file
        // dropped from the OS can be hit-tested against them. Rebuilt every
        // frame and consumed at the end of the same one — the raw Material* in
        // the target must never outlive the frame that recorded it.
        //
        // Screen-space rect as plain floats: this header is included by every
        // panel and has no business dragging imgui's types in for four numbers.
        struct FrameTextureSlot {
            TextureSlotTarget target;
            float minX, minY, maxX, maxY;

            [[nodiscard]] bool contains(float x, float y) const {
                return x >= minX && x <= maxX && y >= minY && y <= maxY;
            }
        };
        std::vector<FrameTextureSlot> frameTextureSlots_;
        // Opens the inspector's Textures section for one frame. The self-test
        // sets it to reach the slot rows; a user does the same by clicking.
        bool openTextureSectionOnce_ = false;
        // Image paths dropped this frame, resolved after the UI has drawn (and
        // so after the rows above are known).
        std::vector<std::filesystem::path> pendingTextureDrops_;

        // Confirmation for a destructive action on a dirty document.
        enum class PendingAction {
            None,
            New,
            Open,
            OpenPath,
            OpenExample,
            Quit
        };
        PendingAction pendingAction_ = PendingAction::None;
        std::filesystem::path pendingPath_;
        // Which shipped example File ▸ Open Example picked, held across the
        // unsaved-changes modal exactly as pendingPath_ is.
        std::string pendingExample_;

        // UI state
        float contentScale_ = 1.f;
        // Heights of the fixed chrome, measured as it is drawn — the side
        // panels need them to size themselves in the same frame.
        float menuHeight_ = 0.f;
        float statusHeight_ = 0.f;
        bool bottomPanelOpen_ = true;
        // One-shot request to bring the Sensors tab forward. Consumed by the tab
        // bar on the next frame it draws. Exists for --screenshot: a live readout
        // that nobody has looked at is a readout nobody knows is right.
        bool selectSensorsTab_ = false;
        // The same, for the Scripts tab, on an explicit open — and, separately,
        // for which script inside it. They are two bars: raising a script the
        // selection moved onto must not also drag the panel off the Console.
        bool selectScriptsTab_ = false;
        std::string selectScriptUuid_;
        std::deque<std::string> console_;
        std::string renameBuffer_;
        // Name as it was when the inspector's name field gained focus, so the
        // whole edit becomes one undo entry.
        std::string nameBeforeEdit_;
        Object3D* renaming_ = nullptr;
        std::filesystem::path assetDir_;
        bool consoleScrollToBottom_ = false;
        float fps_ = 0.f;
        std::size_t objectCount_ = 0;
        std::string lastWindowTitle_;
        // Edit-mode animation preview state.
        struct AnimPreview {
            Object3D* root = nullptr;
            std::string clip;
            std::unique_ptr<AnimationMixer> mixer;
            struct SavedNode {
                Object3D* node;
                Vector3 position;
                Quaternion quaternion;
                Vector3 scale;
            };
            std::vector<SavedNode> saved;
            std::vector<std::pair<ObjectWithMorphTargetInfluences*, std::vector<float>>> savedMorphs;
        };
        std::unique_ptr<AnimPreview> animPreview_;

        // One open script — a text buffer that is not the document until Apply
        // commits it, which is what makes the unsaved marker on its tab mean
        // something.
        //
        // Not gated on Python: a build without it still edits and saves the
        // source, it just cannot check or run it.
        struct ScriptEditorState {
            // Presence in scriptEditors_ IS being open; the flag exists because
            // ImGui's tab close button writes through a bool*. Cleared entries
            // are erased at the top of the next updateScriptEditors().
            bool open = true;
            // The object being edited, by uuid: a play/stop replaces the whole
            // graph, and the tabs have to survive that.
            std::string uuid;
            std::string label;
            // What the text box holds, and what the document holds. Different
            // means unsaved.
            std::string buffer;
            std::string committed;
            // Syntax error from the last Apply, shown in red until the next.
            std::string status;
            // Take the keyboard on the frame after an explicit open (but never
            // when a tab is merely raised or synced under the user).
            bool focus = false;
            // The object is gone from the scene, as of the last update. Resolved
            // there so the tab body does not walk the graph a second time.
            bool missing = false;
        };
        // In tab order, one entry per open script. A vector and not a map: the
        // order on screen is the order they were opened in, and there are never
        // enough of them for the lookup to be worth a hash.
        std::vector<ScriptEditorState> scriptEditors_;
        // Which one is visible, by uuid — set by the inner tab bar as it draws.
        std::string activeScriptUuid_;
        // The selection as the raise-on-selection rule last saw it, so that the
        // rule fires on a change rather than on a mismatch.
        std::string lastScriptSelection_;

        // "Edit in VS Code" on an inline script. The source is exported to a
        // scratch .py, VS Code is pointed at it, and the file is polled once a
        // second from the frame loop — no thread, no watcher API, nothing that
        // can call back into the editor from anywhere but a frame.
        //
        // The object is held by UUID and re-resolved on every poll: a play/stop
        // replaces the whole graph, and surviving exactly that is what makes
        // "edit externally while iterating on Play" work at all.
        struct ExternalEditState {

            static constexpr float pollSeconds = 1.f;

            bool active = false;
            ExternalEditKind kind = ExternalEditKind::Script;
            std::string uuid;
            std::string label;
            std::filesystem::path file;
            // The normalized text last committed, and the only thing a poll
            // compares against — write times tie between two quick saves, and a
            // save that only rewrote the line endings (which is every save from
            // a Windows editor) must not become an undo entry either way.
            std::string synced;
            float poll = 0.f;
            // Open for the session's life. Coalescing is the command stack's
            // transactions: ten saves collapse into one undo step, and the
            // first still starts a fresh entry, because a transaction never
            // merges into what was on the stack before it opened.
            bool transaction = false;
            // A save arrived while playing and is waiting for Stop.
            bool waiting = false;
            // Syncs applied this session (the self-test's counter).
            int syncs = 0;
        };
        ExternalEditState externalEdit_;

#ifdef THREEPP_EDITOR_WITH_PYTHON
        // Registered with the play controller; also the inspector's source for
        // "what went wrong in this script last session".
        std::shared_ptr<ScriptPlaySession> scripts_;

        // Discovering a script's fields means importing the file, which the
        // inspector cannot do sixty times a second. Cached per path and
        // invalidated by the file's write time, so saving the .py in an editor
        // refreshes the section without any reload button being pressed.
        struct CachedInspection {
            std::filesystem::file_time_type stamp{};
            scripting::Inspection inspection;
        };
        std::unordered_map<std::string, CachedInspection> scriptInspections_;
        // Returns the cached inspection for `path`, re-running it when the file
        // changed on disk.
        const scripting::Inspection& inspectScript(const std::string& path);

        // The same for inline source, which has no write time to key on: the
        // stamp is a hash of the text, so editing it in the Script Editor
        // refreshes the parameters exactly as saving a .py does. Keyed by
        // object, because the synthetic module is named after the object too.
        struct CachedSourceInspection {
            std::size_t hash = 0;
            scripting::Inspection inspection;
        };
        std::unordered_map<std::string, CachedSourceInspection> scriptSourceInspections_;
        const scripting::Inspection& inspectScriptSource(const std::string& uuid,
                                                         const std::string& label,
                                                         const std::string& source);
#endif

        // Async model imports. One worker at a time; the rest wait in the
        // queue. importError_ non-empty keeps the failure modal open.
        // Object3D rather than Group: a URDF import yields a Robot, which is not
        // a Group.
        struct ActiveImport {
            std::filesystem::path path;
            std::future<std::shared_ptr<Object3D>> future;
            float elapsed = 0.f;
            // What the arg prompt collected, kept so the RobotConfig written on
            // success records it and every later rebuild can reuse it.
            std::vector<std::pair<std::string, std::string>> args;
        };
        // A queued file, and the xacro arguments it is to be expanded with.
        // `argsResolved` separates "nobody has looked yet" from "looked, and the
        // answer was none" — the second is the normal case for every asset type
        // that is not an argument-driven description.
        struct PendingImport {
            std::filesystem::path path;
            std::vector<std::pair<std::string, std::string>> args;
            bool argsResolved = false;
        };
        std::deque<PendingImport> importQueue_;
        std::unique_ptr<ActiveImport> activeImport_;
        std::string importError_;
        // A popped entry parked between "this file declares arguments" and the
        // user answering. One editable buffer per declaration, left EMPTY on
        // purpose: an untouched field must send no override at all, so that the
        // file's own default — which may be derived from another argument —
        // still applies. The declared text is shown as a hint instead.
        struct ArgPrompt {
            std::filesystem::path path;
            std::vector<xacro::ArgDecl> declared;
            std::vector<std::vector<char>> values;
            bool opened = false;
        };
        std::unique_ptr<ArgPrompt> argPrompt_;
        // Transient status-bar message (import results and similar).
        std::string statusFlash_;
        float statusFlashRemaining_ = 0.f;
        // Accumulated frame time driving the import spinner.
        float uiTime_ = 0.f;

        // Camera dock, filled by renderCameraPreview each frame and read by
        // drawUi, which draws the frame and the picker over it. `visible` is
        // whether the dock has room this frame; `active` whether a camera
        // rendered into it — an empty dock still paints itself, so the corner
        // never reverts to a sliver of unreachable viewport.
        // `pending` is the Vulkan-only middle state: a camera IS docked but its
        // secondary view has not composited yet (it is allocated at the next
        // frame boundary), so the dock still holds primary-viewport pixels and
        // has to be painted over — without the empty-dock hint, which would be
        // a lie for the frame it appeared in.
        struct {
            float x = 0, y = 0, w = 0, h = 0;
            bool visible = false;
            bool active = false;
            bool pending = false;
            // Whether the scene has any camera at all: an empty dock in a scene
            // with cameras is a choice, in a scene without them a fact.
            bool hasCameras = false;
        } preview_;
        // Which camera the dock shows, as a uuid because the scene it lives in
        // is replaced wholesale by Play and Stop — same uuids, new objects, so
        // a Camera* here would dangle and a "same uuid, skip" check would not
        // notice. Deliberately NOT the selection: the dock is a view you set up
        // and then work against, and the object you are framing is exactly what
        // you select next.
        //   nullopt      - nothing chosen yet; follow the scene's first camera,
        //                  so a scene that has one never opens to a blank dock.
        //   empty string - "None" chosen in the picker. Stays empty; the point
        //                  of choosing None is that nothing takes its place.
        //   a uuid       - that camera, falling back to the first while it is
        //                  out of the scene (deleted, or an undo away).
        std::optional<std::string> dockCamera_;
        // The dock's exact-pixel path on Vulkan: one persistent secondary view,
        // re-pointed at whichever camera the dock holds. Empty and inert on
        // OpenGL, where the dock is a scissored second render of the same scene.
        VulkanViewPane dockPane_;
        // Object3D* the hierarchy wants to scroll into view next frame.
        Object3D* scrollTo_ = nullptr;
        // What the viewport context menu was opened on (nullptr = empty space).
        // Checked against the selection every frame the popup is up: anything
        // that re-points the selection (a delete, Play swapping the graph)
        // closes the menu rather than acting through a stale pointer.
        Object3D* viewportCtx_ = nullptr;
        // Structural edits requested from inside a tree walk (delete, reparent,
        // add) run here, after the walk, so nothing mutates the children vector
        // we are iterating.
        std::function<void()> deferred_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORAPP_HPP
