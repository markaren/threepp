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

#include "threepp/extras/editor/Command.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/EditorSettings.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/Selection.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/helpers/BoxHelper.hpp"
#include "threepp/input/IOCapture.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

class ImguiContext;

namespace threepp {

    class CameraHelper;
    class Material;
    class MeshBasicMaterial;
    class ObjectWithMorphTargetInfluences;
    class Texture;

}// namespace threepp

namespace threepp::editor {

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
        };

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
        void drawToolbar();
        void drawHierarchy();
        void drawInspector();
        void drawBottomPanel();
        void drawStatusBar();
        void drawPlayBanner();

        // Hierarchy helpers
        void drawHierarchyNode(Object3D& object);
        void drawAddMenu(Object3D& parent);

        // Inspector sections
        void drawObjectSection(Object3D& object);
        void drawTransformSection(Object3D& object);
        void drawMaterialSection(Object3D& object);
        void drawGeometrySection(Object3D& object);
        void drawLightSection(Object3D& object);
        void drawCameraSection(Object3D& object);
        void drawAnimationSection(Object3D& object);
        void drawPhysicsSection(Object3D& object);
        void drawTextureSlot(Material& material, const char* label,
                             const std::shared_ptr<Texture>& current,
                             const std::function<void(const std::shared_ptr<Texture>&)>& setter,
                             bool srgb);

        // Assets / console tabs
        void drawAssetsTab();
        void drawConsoleTab();

        // --- editing operations --------------------------------------------
        void newScene();
        void buildTemplateScene();
        void openScene(const std::filesystem::path& path);
        void saveScene();
        void saveSceneAs(const std::filesystem::path& path);
        void importModel(const std::filesystem::path& path);
        void setEnvironment(const std::filesystem::path& path, bool alsoBackground);
        void clearEnvironment();
        void assignTextureToSelection(const std::filesystem::path& path);
        void assignTextureToSlot(const std::filesystem::path& path);

        void addObject(const std::shared_ptr<Object3D>& object, Object3D& parent, const std::string& label);
        void deleteSelected();
        void duplicateSelected();
        void focusSelected();
        void reparent(Object3D& object, Object3D& newParent);

        // --- selection / picking -------------------------------------------
        void selectObject(Object3D* object);
        void refreshSelectionHelpers();

        // --- viewport markers ----------------------------------------------
        // Billboarded SVG icons standing in for objects that draw nothing
        // (cameras, lights), plus the frustum helper for a selected camera.
        void syncViewportMarkers();
        void syncCameraHelper();
        void clearViewportMarkers();
        // The object a marker stands for, or nullptr when `hit` is not part of
        // one. Lets a click on an icon select its owner.
        [[nodiscard]] Object3D* markerOwnerOf(Object3D* hit) const;
        // Renders the selected scene camera into a bottom-right inset of the
        // viewport; drawUi frames and labels it via preview_.
        void renderCameraPreview();
        void pickAt(float mouseX, float mouseY);
        [[nodiscard]] Object3D* resolveSelectable(Object3D* hit) const;

        // --- play ----------------------------------------------------------
        void startPlay();
        void togglePause();
        void stopPlay();
        [[nodiscard]] bool isPlaying() const;

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
        void flashStatus(std::string message);

        // --- misc ----------------------------------------------------------
        int runSelfTest();
        void handleShortcuts();
        void handleFileDrop(const std::vector<std::string>& paths);
        void log(const std::string& message);
        void logWarnings();
        void applyGizmoMode();
        void loadSettings();
        void persistSettings();
        [[nodiscard]] float scale() const { return contentScale_; }

        // Undo-friendly ImGui helpers: begin a transaction when a widget is
        // activated and close it when the edit finishes, so a drag collapses to
        // one undo step.
        void beginEditIfActivated();
        void endEditIfDeactivated();

        // --- state ---------------------------------------------------------
        Options options_;

        Canvas canvas_;
        std::unique_ptr<Renderer> renderer_;
        PerspectiveCamera camera_;
        OrbitControls orbit_;

        SceneDocument document_;
        Selection selection_;
        CommandStack commands_;
        PlayController play_;
        EditorSettings settings_;
        std::filesystem::path settingsPath_;

        // Editor-only scene content.
        std::shared_ptr<Group> overlay_;
        std::shared_ptr<Object3D> grid_;
        std::shared_ptr<Object3D> axes_;
        std::shared_ptr<BoxHelper> selectionBox_;
        std::unique_ptr<TransformControls> gizmo_;

        // Marker icons. One node per owner, all parented to markers_, which is
        // itself part of the editor-only overlay.
        struct ViewportMarker {
            Object3D* owner = nullptr;
            std::shared_ptr<Object3D> node;
            std::vector<std::shared_ptr<MeshBasicMaterial>> materials;
        };
        std::shared_ptr<Group> markers_;
        std::vector<ViewportMarker> viewportMarkers_;
        // Frustum of the selected camera. Holds a reference to that camera, so
        // it is torn down whenever the selection or the scene changes.
        std::shared_ptr<CameraHelper> cameraHelper_;
        Object3D* cameraHelperFor_ = nullptr;

        Raycaster raycaster_;
        std::unique_ptr<ImguiContext> ui_;
        IOCapture ioCapture_;
        // The gizmo's "dragging-changed" listener is referenced, not owned, by
        // the dispatcher — it has to outlive the subscription.
        std::unique_ptr<LambdaEventListener> gizmoDragListener_;

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
            Texture
        };
        PendingDialog pendingDialog_ = PendingDialog::None;
        bool environmentAsBackground_ = true;

        // Which material slot a "Load..." button in the inspector is filling.
        // The file dialog resolves a frame or more later, so the target has to
        // survive across frames.
        struct TextureSlotTarget {
            Material* material = nullptr;
            std::function<void(const std::shared_ptr<Texture>&)> setter;
            std::shared_ptr<Texture> current;
            std::string slot;
            bool srgb = true;
        };
        TextureSlotTarget textureSlotTarget_;

        // Confirmation for a destructive action on a dirty document.
        enum class PendingAction {
            None,
            New,
            Open,
            OpenPath,
            Quit
        };
        PendingAction pendingAction_ = PendingAction::None;
        std::filesystem::path pendingPath_;

        // UI state
        float contentScale_ = 1.f;
        // Heights of the fixed chrome, measured as it is drawn — the side
        // panels need them to size themselves in the same frame.
        float menuHeight_ = 0.f;
        float toolbarHeight_ = 0.f;
        float statusHeight_ = 0.f;
        bool bottomPanelOpen_ = true;
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

        // Async model imports. One worker at a time; the rest wait in the
        // queue. importError_ non-empty keeps the failure modal open.
        struct ActiveImport {
            std::filesystem::path path;
            std::future<std::shared_ptr<Group>> future;
            float elapsed = 0.f;
        };
        std::deque<std::filesystem::path> importQueue_;
        std::unique_ptr<ActiveImport> activeImport_;
        std::string importError_;
        // Transient status-bar message (import results and similar).
        std::string statusFlash_;
        float statusFlashRemaining_ = 0.f;
        // Accumulated frame time driving the import spinner.
        float uiTime_ = 0.f;

        // Camera-preview inset, filled by renderCameraPreview each frame and
        // read by drawUi to draw the border and label on top.
        struct {
            float x = 0, y = 0, w = 0, h = 0;
            bool active = false;
            std::string label;
        } preview_;
        // Object3D* the hierarchy wants to scroll into view next frame.
        Object3D* scrollTo_ = nullptr;
        // Structural edits requested from inside a tree walk (delete, reparent,
        // add) run here, after the walk, so nothing mutates the children vector
        // we are iterating.
        std::function<void()> deferred_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORAPP_HPP
