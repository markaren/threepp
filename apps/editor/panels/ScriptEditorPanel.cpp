// The Script Editor: the bottom panel's Scripts tab, holding one inner tab per
// open script.
//
// Deliberately a plain text box. No syntax highlighting, no autocomplete, no
// third-party editor widget — v1 is "type Python, see whether it parses, press
// Play", and the one thing it does add over Notepad is that the source lives in
// the scene rather than in a file beside it.
//
// It used to be a floating window, and floating over the viewport is exactly
// what was wrong with it: a text box big enough to write in covered the thing
// the script was being written about. Docked beside the console it takes room
// from nothing — the bottom panel is draggable now, so "big enough" is a drag
// on its top edge rather than a window parked over the scene.
//
// It also used to be ONE editor that retargeted itself onto the selection,
// which meant writing two scripts that talk to each other was a matter of
// clicking back and forth and hoping nothing was dropped in transit. Now every
// script you open keeps its own tab, its own buffer and its own syntax error
// until you close it. Still one script per object — that is ScriptConfig's
// rule, not this file's.
//
// Nothing in here runs the script. Apply compiles the source to find syntax
// errors and stops there; the code executes when the user presses Play, and
// never before.

#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // std::string-backed InputText, the way misc/cpp/imgui_stdlib does it (not
    // vendored here). ImGui asks for the buffer to grow and we hand back the
    // string's own storage, so the text never lives in a fixed-size scratch
    // buffer — a script is exactly as long as it is.
    int growBuffer(ImGuiInputTextCallbackData* data) {

        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
            auto* buffer = static_cast<std::string*>(data->UserData);
            buffer->resize(static_cast<std::size_t>(data->BufTextLen));
            data->Buf = buffer->data();
        }
        return 0;
    }

    std::string firstLineOf(const std::string& text) {

        return text.substr(0, text.find('\n'));
    }

}// namespace


std::string EditorApp::inlineScriptTemplate() {

    // One class, one exposed parameter, and the three method names spelled out
    // — the header is the documentation most users will ever read about the
    // shape a script has to have.
    //
    // `import threepp` and the Object3D annotation are here for the IDE, not
    // the runtime: they are what makes `self.obj.` complete in VS Code the
    // moment the file opens, instead of after the user reinvents both lines.
    // At Play they resolve against the embedded module, so they are also
    // correct. Narrow the annotation (threepp.Mesh, threepp.Robot) for the
    // type-specific API.
    return "# Inline Python script. Stored in this scene, runs when you press Play.\n"
           "#\n"
           "# One class with any of start(obj) / update(dt) / fixed_update(dt) /\n"
           "# stop() - fixed_update runs on the physics clock. Plain class\n"
           "# attributes (int, float, bool, str) appear as parameters in the\n"
           "# inspector, where their values are saved with the scene.\n"
           "#\n"
           "# The annotation below is what gives `self.obj.` completion in an IDE;\n"
           "# narrow it (threepp.Mesh, threepp.Robot, ...) for the full API of\n"
           "# what this script is attached to.\n"
           "\n"
           "import threepp\n"
           "\n"
           "\n"
           "class Behaviour:\n"
           "\n"
           "    speed = 1.0\n"
           "\n"
           "    def start(self, obj: threepp.Object3D):\n"
           "        self.obj = obj\n"
           "\n"
           "    def update(self, dt: float):\n"
           "        self.obj.rotation.y += self.speed * dt\n";
}

EditorApp::ScriptEditorState* EditorApp::scriptEditorFor(const std::string& uuid) {

    for (auto& state : scriptEditors_) {
        if (state.uuid == uuid) return &state;
    }
    return nullptr;
}

EditorApp::ScriptEditorState* EditorApp::activeScriptEditor() {

    if (auto* state = scriptEditorFor(activeScriptUuid_)) return state;
    // Nothing has drawn yet this session, or the active one was just closed:
    // the first tab is what the bar will show.
    return scriptEditors_.empty() ? nullptr : &scriptEditors_.front();
}

void EditorApp::openScriptEditor(const Object3D& object, bool reveal) {

    const auto config = ScriptConfig::read(object).value_or(ScriptConfig{});

    auto* state = scriptEditorFor(object.uuid);
    if (!state) {
        state = &scriptEditors_.emplace_back();
        state->uuid = object.uuid;
        state->buffer = config.source;
    }
    // An object that already has a tab keeps its working text and its syntax
    // error: both describe the buffer, and the buffer is still there. Closing a
    // tab is not a decision to discard an edit, and Revert is right there for
    // when it is.
    state->committed = config.source;
    state->label = object.name.empty() ? object.type() : object.name;
    state->missing = false;
    state->open = true;

    if (reveal) {
        // A tab nobody can see is not an editor. Opening one is an explicit
        // request to look at it, so the panel comes up with it.
        bottomPanelOpen_ = true;
        selectScriptsTab_ = true;
        selectScriptUuid_ = state->uuid;
        activeScriptUuid_ = state->uuid;
        state->focus = true;
        // Whatever is selected has now been accounted for: an explicit open
        // outranks the raise-on-selection rule, which would otherwise pull the
        // bar back to the selected object on the very next frame.
        lastScriptSelection_ = selection_.get() ? selection_.get()->uuid : std::string{};
    }
}

void EditorApp::applyScriptEditor() {

    if (auto* state = activeScriptEditor()) applyScriptEditor(*state);
}

void EditorApp::applyScriptEditor(ScriptEditorState& state) {

    auto* target = findByUuid(document_.scene(), state.uuid);
    if (!target || isPlaying()) return;

    // Tabs become four spaces, carriage returns are dropped — see
    // ScriptWorkspace::normalize. The box takes real tabs (a Python editor that
    // cannot indent is not one) and Python raises TabError on source that mixes
    // them, and an external save arrives CRLF-terminated.
    state.buffer = ScriptWorkspace::normalize(state.buffer);
    state.status.clear();

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Shown, recorded — and applied anyway. Half-written code is a normal thing
    // to want to save; an editor that refuses to let go of what you typed
    // because it does not parse yet is a worse editor.
    state.status = scripting::checkSyntax(state.buffer, state.label);
    if (!state.status.empty()) {
        log("script syntax error on " + state.label + " - " + firstLineOf(state.status));
    }
#endif

    setInlineScript(*target, state.buffer, "Edit Script");
    state.committed = state.buffer;
}

void EditorApp::updateScriptEditors() {

    const auto closed = [](const ScriptEditorState& state) { return !state.open; };

    // Tabs closed by their × last frame. Buried here rather than mid-draw: the
    // inner tab bar holds a pointer into this vector for the length of a frame.
    std::erase_if(scriptEditors_, closed);

    // Selecting an object that already has a tab raises it. That is what
    // following the selection means once there can be more than one: the single
    // editor used to RETARGET itself onto the selection, which is exactly what
    // tabs exist to stop — and it could only ever do it with nothing to lose,
    // because retargeting a buffer somebody is typing into drops it. Only the
    // inner bar moves; the panel stays on whatever tab you were reading.
    //
    // On the CHANGE, not on the mismatch. A rule that fired whenever the
    // selection and the visible tab disagreed would fight every explicit open —
    // opening a script on one object while another is selected would bounce
    // straight back on the next frame — and would pin the bar so that clicking
    // between two open scripts did nothing.
    const std::string selected = selection_.get() ? selection_.get()->uuid : std::string{};
    if (selected != lastScriptSelection_) {
        lastScriptSelection_ = selected;
        if (!selected.empty() && scriptEditorFor(selected)) {
            selectScriptUuid_ = selected;
            // Active immediately, not when the bar catches up: ImGui applies a
            // SetSelected at the NEXT BeginTabBar, and Apply must not act on
            // the script the user just navigated away from in between.
            activeScriptUuid_ = selected;
        }
    }

    if (scriptEditors_.empty()) return;

    for (auto& state : scriptEditors_) {
        // By uuid, not by pointer: a play/stop replaces the whole graph.
        auto* target = findByUuid(document_.scene(), state.uuid);
        state.missing = target == nullptr;
        if (!target) continue;

        // The document is the truth. An undo, or Clear in the inspector, moves
        // the committed text under the tab, and the unsaved marker has to
        // reflect that rather than the state at open time.
        state.committed = ScriptConfig::read(*target).value_or(ScriptConfig{}).source;
        state.label = target->name.empty() ? target->type() : target->name;

        // The object's inline script is gone (Clear in the inspector, an undo,
        // or an Apply of an empty buffer): there is nothing left to edit, so
        // the tab goes away. Nothing is lost by it — the removal is one Ctrl+Z
        // away and the text stays in the buffer, so reopening the object brings
        // it back.
        if (state.committed.empty()) state.open = false;
    }

    // And the ones that just lost their script, in the same frame the document
    // lost it: "the tab is gone" has to be true the moment Clear or Ctrl+Z
    // says so, not a frame later.
    std::erase_if(scriptEditors_, closed);
}

std::string EditorApp::scriptsTabLabel() const {

    // No object name and no count: this tab sits beside Console and Assets in
    // a bar only as wide as the window minus the camera dock, and a label that
    // grows pushes them off the end of it. Which scripts are open is the inner
    // bar's job. The ### id keeps the tab's identity stable while the unsaved
    // marker comes and goes — otherwise every keystroke would close the tab and
    // open a new one beside it.
    const bool unsaved = std::any_of(scriptEditors_.begin(), scriptEditors_.end(),
                                     [](const ScriptEditorState& state) {
                                         return state.buffer != state.committed;
                                     });

    return std::string("Scripts") + (unsaved ? "*" : "") + "###scriptsTab";
}

std::string EditorApp::scriptTabLabel(const ScriptEditorState& state) {

    // Here the object name IS the label — this bar holds nothing else, and
    // which object a script belongs to is the only thing that tells two of them
    // apart. Keyed by uuid so a rename moves the label rather than the tab.
    return state.label + (state.buffer == state.committed ? "" : "*") +
           "###script:" + state.uuid;
}

void EditorApp::drawScriptsTab() {

    // A switch asked for this frame lands on the next one — ImGui reads
    // SetSelected at the following BeginTabBar — so for one frame this bar is
    // still drawing the tab we are leaving. It must not write that back as the
    // active script: that would undo the switch and point Apply at the script
    // the user just navigated away from.
    const bool switching = !selectScriptUuid_.empty();

    if (ImGui::BeginTabBar("##openScripts",
                           // The list button is not decoration here: a handful
                           // of open scripts overflows a panel this wide, and
                           // scroll arrows alone make you hunt for one by name.
                           ImGuiTabBarFlags_TabListPopupButton |
                                   ImGuiTabBarFlags_FittingPolicyScroll)) {

        for (auto& state : scriptEditors_) {

            ImGuiTabItemFlags flags = 0;
            if (selectScriptUuid_ == state.uuid) {
                flags = ImGuiTabItemFlags_SetSelected;
                selectScriptUuid_.clear();
            }

            const auto label = scriptTabLabel(state);
            if (ImGui::BeginTabItem(label.c_str(), &state.open, flags)) {
                if (!switching) activeScriptUuid_ = state.uuid;
                drawScriptTab(state);
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
}

void EditorApp::drawScriptTab(ScriptEditorState& state) {

    const float s = contentScale_;
    const bool external = externalEdit_.active && externalEdit_.uuid == state.uuid;
    const bool playing = isPlaying();
    // Read-only during an external session too, and for the same reason as
    // during Play: the next save would overwrite whatever was typed here.
    const bool locked = state.missing || playing || external;
    const bool unsaved = state.buffer != state.committed;
    const float buttonWidth = 110 * s;

    // Every widget below is per script, and the text box above all: ImGui keys
    // an InputText's cursor, selection and undo stack by id, so two scripts
    // sharing one would carry all three across when you switch tabs.
    ImGui::PushID(state.uuid.c_str());

    ImGui::BeginDisabled(locked);
    if (ImGui::Button("Apply", {buttonWidth, 0})) applyScriptEditor(state);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Ctrl+Enter. Saves the source into the scene as one undoable step.\n"
                          "Tabs become four spaces, because Python will not mix them.\n"
                          "The source is compiled to check that it parses - nothing runs\n"
                          "until you press Play.");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(locked || !unsaved);
    if (ImGui::Button("Revert", {buttonWidth, 0})) {
        state.buffer = state.committed;
        state.status.clear();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Throw away the edits since the last Apply.");
    }
    ImGui::EndDisabled();

    // Everything that says why the box is locked shares the button row: the
    // panel is as tall as the user dragged it, and a banner on its own line is
    // a line the script does not get. Which object this is does NOT need a
    // place here — it is the tab's own label, right above.
    ImGui::SameLine();
    if (state.missing) {
        ImGui::TextColored(theme::danger(), "The object is no longer in the scene.");
    } else if (external) {
        if (ImGui::Button("Stop external edit", {buttonWidth * 1.7f, 0})) {
            stopExternalEdit("stopped from the Script Editor");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stop watching the file and delete it. Everything saved so far\n"
                              "is already in the scene, and the box becomes editable again.");
        }
        ImGui::SameLine();
        ImGui::TextColored(theme::accent(), "Editing externally - syncing on save");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", externalEdit_.file.generic_string().c_str());
        }
        if (playing) {
            ImGui::SameLine();
            ImGui::TextColored(theme::warning(), "- applied when you press Stop");
        }
    } else if (playing) {
        // Same rule as the inspector: the play snapshot is put back on Stop, so
        // an edit made now would be thrown away without a word.
        ImGui::TextColored(theme::warning(), "Read-only while playing");
    } else {
        ImGui::TextColored(theme::muted(), "%s", unsaved ? "unsaved" : "saved in the scene");
    }

    if (!state.status.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, theme::danger());
        ImGui::TextWrapped("%s", state.status.c_str());
        ImGui::PopStyleColor();
    }

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput |
                                ImGuiInputTextFlags_CallbackResize |
                                // Multiline turns Ctrl+Enter into a validate
                                // rather than a newline, which is exactly the
                                // Apply shortcut.
                                ImGuiInputTextFlags_EnterReturnsTrue;
    if (locked) flags |= ImGuiInputTextFlags_ReadOnly;

    if (state.focus) {
        ImGui::SetKeyboardFocusHere();
        state.focus = false;
    }

    // Two widget ids, one per lock state, and that is not cosmetic:
    // flipping ReadOnly on a LIVE InputText crashes ImGui. A widget
    // activated while read-only never fills its editing copy (read-only
    // reads the user buffer directly), so the frame the flag comes off,
    // the widget dereferences that null copy with the read-only length —
    // which is exactly what pressing Play with the keyboard in this box
    // and then pressing Stop used to do. Changing the id instead means the
    // active widget simply stops being submitted, which ImGui retires
    // cleanly, and the other one starts from the buffer.
    const char* boxId = locked ? "##scriptSourceLocked" : "##scriptSource";

    const float footer = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    if (ImGui::InputTextMultiline(boxId, state.buffer.data(), state.buffer.capacity() + 1,
                                  {-FLT_MIN, -footer}, flags, growBuffer, &state.buffer)) {
        if (!locked) applyScriptEditor(state);
    }

#ifdef THREEPP_EDITOR_WITH_PYTHON
    ImGui::TextColored(theme::muted(),
                       "Stored in userData[\"scriptSource\"] - it runs when you press Play. "
                       "Drag the panel's top edge for more room.");
#else
    ImGui::TextColored(theme::muted(),
                       "Built without Python scripting - the source is saved, not checked or run.");
#endif

    ImGui::PopID();
}
