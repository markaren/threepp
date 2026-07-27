// The Script Editor: a floating window over one object's inline script source.
//
// Deliberately a plain text box. No syntax highlighting, no autocomplete, no
// third-party editor widget — v1 is "type Python, see whether it parses, press
// Play", and the one thing it does add over Notepad is that the source lives in
// the scene rather than in a file beside it.
//
// It is a normal ImGui window, not one of the fixed panels: movable, resizable
// and closable, because it is a tool you open over your work rather than part
// of the frame around it.
//
// Nothing in here runs the script. Apply compiles the source to find syntax
// errors and stops there; the code executes when the user presses Play, and
// never before.

#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cfloat>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Tabs become four spaces, carriage returns are dropped.
    //
    // The text box takes real tabs (AllowTabInput, because a Python editor
    // that cannot indent is not one), and Python raises TabError on source
    // that mixes them with spaces — which is exactly what happens when a user
    // types Tab on one line and the editor's own template supplied spaces on
    // the next. Normalizing on Apply makes that impossible instead of
    // diagnosing it later.
    std::string normalized(const std::string& text) {

        std::string out;
        out.reserve(text.size());
        for (const char c : text) {
            if (c == '\t') {
                out.append(4, ' ');
            } else if (c != '\r') {
                out.push_back(c);
            }
        }
        return out;
    }

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
    return "# Inline Python script. Stored in this scene, runs when you press Play.\n"
           "#\n"
           "# One class with any of start(obj) / update(dt) / stop(). Plain class\n"
           "# attributes (int, float, bool, str) appear as parameters in the\n"
           "# inspector, where their values are saved with the scene.\n"
           "\n"
           "class Behaviour:\n"
           "\n"
           "    speed = 1.0\n"
           "\n"
           "    def start(self, obj):\n"
           "        self.obj = obj\n"
           "\n"
           "    def update(self, dt):\n"
           "        self.obj.rotation.y += self.speed * dt\n";
}

void EditorApp::openScriptEditor(const Object3D& object) {

    auto& state = scriptEditor_;
    const auto config = ScriptConfig::read(object).value_or(ScriptConfig{});

    // Reopening the object we were already on keeps the working text: the
    // window's Close button is not a decision to discard an edit, and Revert
    // is right there for when it is.
    const bool same = state.uuid == object.uuid && !state.buffer.empty();
    state.committed = config.source;
    if (!same) state.buffer = config.source;

    state.uuid = object.uuid;
    state.label = object.name.empty() ? object.type() : object.name;
    state.status.clear();
    state.open = true;
    state.focus = true;
}

void EditorApp::applyScriptEditor() {

    auto& state = scriptEditor_;
    auto* target = findByUuid(document_.scene(), state.uuid);
    if (!target || isPlaying()) return;

    state.buffer = normalized(state.buffer);
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

void EditorApp::drawScriptEditor() {

    auto& state = scriptEditor_;
    if (!state.open) return;

    const float s = contentScale_;

    // Follow the selection — but only with nothing to lose. Retargeting a
    // buffer someone is still typing into would silently drop it, so an unsaved
    // window stays on the object it was opened for however the selection moves.
    if (state.buffer == state.committed) {
        if (auto* selected = selection_.get(); selected && selected->uuid != state.uuid) {
            if (const auto config = ScriptConfig::read(*selected); config && config->isInline()) {
                openScriptEditor(*selected);
                // Following the selection must not steal the keyboard from
                // whatever the user was actually doing.
                state.focus = false;
            }
        }
    }

    // By uuid, not by pointer: a play/stop replaces the whole graph.
    auto* target = findByUuid(document_.scene(), state.uuid);
    if (target) {
        // The document is the truth. An undo, or Clear in the inspector, moves
        // the committed text under the window, and the unsaved marker has to
        // reflect that rather than the state at open time.
        state.committed = ScriptConfig::read(*target).value_or(ScriptConfig{}).source;
        state.label = target->name.empty() ? target->type() : target->name;
    }

    const bool unsaved = state.buffer != state.committed;

    // The object's inline script is gone (Clear in the inspector, an undo, or
    // an Apply of an empty buffer): there is nothing left to edit, so the window
    // goes away. Nothing is lost by it — the removal is one Ctrl+Z away and the
    // text stays in the buffer, so reopening the object brings it back.
    if (target && state.committed.empty()) {
        state.open = false;
        return;
    }

    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize({700 * s, 500 * s}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos({viewport->Pos.x + (viewport->Size.x - 700 * s) * 0.5f,
                             viewport->Pos.y + (viewport->Size.y - 500 * s) * 0.5f},
                            ImGuiCond_FirstUseEver);

    // The ### id keeps the window identity (and therefore its position and
    // size) stable while the visible title gains and loses its unsaved marker
    // and follows the object's name.
    const std::string title = "Script Editor" + std::string(unsaved ? "*" : "") +
                              " - " + state.label + "###scriptEditor";

    bool keepOpen = true;
    if (ImGui::Begin(title.c_str(), &keepOpen, ImGuiWindowFlags_NoSavedSettings)) {

        const bool missing = target == nullptr;
        const bool playing = isPlaying();
        const bool locked = missing || playing;
        const float buttonWidth = 110 * s;

        if (missing) {
            ImGui::TextColored(theme::danger(), "The object is no longer in the scene.");
        } else if (playing) {
            // Same rule as the inspector: the play snapshot is put back on
            // Stop, so an edit made now would be thrown away without a word.
            ImGui::TextColored(theme::warning(), "Read-only while playing");
        }

        ImGui::BeginDisabled(locked);
        if (ImGui::Button("Apply", {buttonWidth, 0})) applyScriptEditor();
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

        ImGui::SameLine();
        if (ImGui::Button("Close", {buttonWidth, 0})) keepOpen = false;

        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "%s", unsaved ? "unsaved" : "saved in the scene");

        if (!state.status.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, theme::danger());
            ImGui::TextWrapped("%s", state.status.c_str());
            ImGui::PopStyleColor();
        }

        ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput |
                                    ImGuiInputTextFlags_CallbackResize |
                                    // Multiline turns Ctrl+Enter into a
                                    // validate rather than a newline, which is
                                    // exactly the Apply shortcut.
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
            if (!locked) applyScriptEditor();
        }

#ifdef THREEPP_EDITOR_WITH_PYTHON
        ImGui::TextColored(theme::muted(),
                           "Stored in userData[\"scriptSource\"] - it runs when you press Play.");
#else
        ImGui::TextColored(theme::muted(),
                           "Built without Python scripting - the source is saved, not checked or run.");
#endif
    }
    ImGui::End();

    if (!keepOpen) state.open = false;
}
