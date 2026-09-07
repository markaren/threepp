// The undo ceremony every inspector field performs, written once.
//
// A scalar widget in this editor is never just a widget. Dragging one has to
// open an undo transaction on the frame the drag STARTS, execute a merging
// PropertyCommand on every frame the value changes, and close the transaction
// when the mouse comes up — otherwise one drag of a slider lands in the undo
// stack as fifty steps. That three-line dance was hand-written at ~67 sites in
// InspectorPanel.cpp, and the per-Config `commit` and `dragFloat` lambdas
// around it were re-declared, near-identically, in eighteen sections. This is
// the one copy.
//
// The shape below is not invented here: `drawGranularSection` had already
// grown a local pointer-to-member `dragFloat` with a section label prefix, and
// that is the shape that generalized. Addressing a field as
// `&PhysicsConfig::friction` rather than as a setter lambda is not only
// shorter — it cannot silently assign to the wrong member the way a
// hand-written `[](PhysicsConfig& c, float v) { c.mass = v; }` pasted under a
// "Friction" label can, and that copy-paste slip is exactly what a table of
// near-identical fields invites.
//
// Two levels, because the panel needs both:
//
//   committed(commands, changed, apply)  — the dance alone, for a widget whose
//       change is not an assignment into a Config: a field on the Object3D
//       itself, a material property, a combo that also rebuilds geometry.
//
//   ConfigFields<Config>                 — the whole pattern, holding the
//       decoded config, the before-snapshot the undo step rewinds to, and the
//       merge key that coalesces a drag into one step.

#ifndef THREEPP_EDITOR_CONFIGFIELDS_HPP
#define THREEPP_EDITOR_CONFIGFIELDS_HPP

#include "threepp/extras/editor/Command.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>

namespace threepp::editor {

    // threepp keeps colors in the renderer's LINEAR working space; a color
    // picker must show what the user typed in, so both directions go through
    // the sRGB hex accessors rather than touching r/g/b.
    inline void toSrgbFloats(const Color& color, float out[3]) {

        const unsigned int hex = color.getHex();
        out[0] = static_cast<float>((hex >> 16) & 0xff) / 255.f;
        out[1] = static_cast<float>((hex >> 8) & 0xff) / 255.f;
        out[2] = static_cast<float>(hex & 0xff) / 255.f;
    }

    inline Color fromSrgbFloats(const float in[3]) {

        const auto channel = [](float v) {
            return static_cast<unsigned int>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
        };
        Color color;
        color.setHex((channel(in[0]) << 16) | (channel(in[1]) << 8) | channel(in[2]));
        return color;
    }

    // Wrap the ImGui item that was JUST drawn: `changed` is that widget's own
    // return value, so the item-state queries below refer to it.
    //
    // Evaluation order is the point — the widget call is the argument, so it
    // runs before this body and ImGui's "last item" is the one we mean.
    template<class Apply>
    void committed(CommandStack& commands, bool changed, Apply&& apply) {

        if (ImGui::IsItemActivated()) commands.beginTransaction();
        if (changed) apply();
        if (ImGui::IsItemDeactivated()) commands.endTransaction();
    }

    // A float that lives on the object itself — a material's roughness, a
    // light's intensity, a camera's fov — rather than inside a userData Config.
    // Same ceremony, different destination: the widget shows the live value and
    // one merging PropertyCommand records the step, so undo and redo replay
    // `setter` alone.
    inline void dragProperty(CommandStack& commands, const char* label, std::string mergeKey,
                             float* value, float speed, float min, float max,
                             std::function<void(const float&)> setter,
                             const std::function<void()>& onEdit) {

        float edited = *value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max);
        committed(commands, changed, [&] {
            commands.execute(makeProperty<float>(label, std::move(mergeKey), std::move(setter),
                                                 *value, edited));
            if (onEdit) onEdit();
        });
    }

    // Editing surface for one userData Config on one object.
    //
    // Every field call drives a widget and, on change, executes one
    // PropertyCommand that writes the WHOLE config back. Writing the whole
    // config (rather than one member) is what lets the merge key coalesce a
    // drag into a single undo step, and it keeps the erase-at-defaults write
    // rule in the one place that owns it — the Config's own write().
    template<class Config>
    class ConfigFields {

    public:
        // `mergeKey` is the PropertyCommand coalescing key, conventionally
        // "<name>:" + object.uuid. `labelPrefix` names the section in the undo
        // menu: a field with no explicit action reads "<prefix> <label>", so
        // "Friction" under prefix "Physics" undoes as "Physics Friction".
        // `onEdit` runs after every commit — the panel passes its
        // document-dirty marker.
        ConfigFields(CommandStack& commands, Object3D& object, Config config,
                     std::string mergeKey, std::string labelPrefix,
                     std::function<void()> onEdit = {})
            : commands_(commands), object_(&object), config_(std::move(config)),
              before_(config_), mergeKey_(std::move(mergeKey)),
              prefix_(std::move(labelPrefix)), onEdit_(std::move(onEdit)) {}

        // What the widgets are showing. Sections read this to decide which
        // fields to draw at all ("only when enabled").
        [[nodiscard]] const Config& value() const { return config_; }
        const Config* operator->() const { return &config_; }

        // The snapshot every undo step rewinds to — constant for the life of
        // the section, which is what makes a coalesced drag one step.
        [[nodiscard]] const Config& before() const { return before_; }

        // ----------------------------------------------------------- widgets

        // `lo < hi` bounds the value; lo >= hi means unbounded, ImGui's own
        // convention (a DragFloat3 position passes 0, 0 and must not pin to 0).
        void dragFloat(const char* label, float Config::* field, float step, float lo, float hi,
                       const char* format = "%.3f", const char* action = nullptr,
                       ImGuiSliderFlags flags = 0) {

            float value = config_.*field;
            const bool changed = ImGui::DragFloat(label, &value, step, lo, hi, format, flags);
            committed(commands_, changed, [&] {
                Config after = config_;
                after.*field = lo < hi ? std::clamp(value, lo, hi) : value;
                commitField(label, action, std::move(after));
            });
        }

        void dragInt(const char* label, int Config::* field, float step, int lo, int hi,
                     const char* action = nullptr) {

            int value = config_.*field;
            const bool changed = ImGui::DragInt(label, &value, step, lo, hi);
            committed(commands_, changed, [&] {
                Config after = config_;
                after.*field = lo < hi ? std::clamp(value, lo, hi) : value;
                commitField(label, action, std::move(after));
            });
        }

        void slider(const char* label, float Config::* field, float lo, float hi,
                    const char* format = "%.3f", const char* action = nullptr) {

            float value = config_.*field;
            const bool changed = ImGui::SliderFloat(label, &value, lo, hi, format);
            committed(commands_, changed, [&] {
                Config after = config_;
                after.*field = std::clamp(value, lo, hi);
                commitField(label, action, std::move(after));
            });
        }

        void dragVector3(const char* label, Vector3 Config::* field, float step, float lo, float hi,
                         const char* format = "%.3f", const char* action = nullptr) {

            const Vector3& current = config_.*field;
            float value[3]{current.x, current.y, current.z};
            const bool changed = ImGui::DragFloat3(label, value, step, lo, hi, format);
            committed(commands_, changed, [&] {
                const auto bound = [lo, hi](float v) { return lo < hi ? std::clamp(v, lo, hi) : v; };
                Config after = config_;
                (after.*field).set(bound(value[0]), bound(value[1]), bound(value[2]));
                commitField(label, action, std::move(after));
            });
        }

        // A checkbox cannot be dragged, so it opens no transaction — it either
        // changed this frame or it did not.
        void check(const char* label, bool Config::* field, const char* action = nullptr) {

            check(label, field, action, action);
        }

        // Two actions for the two directions, so the undo menu reads "Undo
        // Enable Acoustic Surface" rather than "Undo Acoustic Surface".
        void check(const char* label, bool Config::* field, const char* onAction,
                   const char* offAction) {

            bool value = config_.*field;
            if (!ImGui::Checkbox(label, &value)) return;
            Config after = config_;
            after.*field = value;
            commitField(label, value ? onAction : offAction, std::move(after));
        }

        // Enum fields, stored as the enum and edited as its index. `items` is
        // an array of names with `count` entries.
        template<class Enum>
        void combo(const char* label, Enum Config::* field, const char* const items[], int count,
                   const char* action = nullptr) {

            int value = static_cast<int>(config_.*field);
            if (!ImGui::Combo(label, &value, items, count)) return;
            if (value < 0 || value >= count) return;
            Config after = config_;
            after.*field = static_cast<Enum>(value);
            commitField(label, action, std::move(after));
        }

        // The "A\0B\0C\0" form ImGui also accepts. `count` bounds the cast back
        // to the enum; ImGui itself will not return an out-of-range index, so
        // it is a guard rather than a check.
        template<class Enum>
        void combo(const char* label, Enum Config::* field, const char* items, int count,
                   const char* action = nullptr) {

            int value = static_cast<int>(config_.*field);
            if (!ImGui::Combo(label, &value, items)) return;
            if (value < 0 || value >= count) return;
            Config after = config_;
            after.*field = static_cast<Enum>(value);
            commitField(label, action, std::move(after));
        }

        void color(const char* label, Color Config::* field, const char* action = nullptr) {

            float value[3];
            toSrgbFloats(config_.*field, value);
            const bool changed = ImGui::ColorEdit3(label, value);
            committed(commands_, changed, [&] {
                Config after = config_;
                after.*field = fromSrgbFloats(value);
                commitField(label, action, std::move(after));
            });
        }

        // ------------------------------------------------------- the escapes

        // Everything the widgets above do not cover — a text box, a file
        // picker, a field that has to fix up a second field to stay consistent
        // — builds its own `after` and lands here.
        void commit(Config after, std::string action) {

            commit(std::move(after), std::move(action), mergeKey_);
        }

        // As above with a per-field merge key, so two independent drags in one
        // section stay two undo steps rather than coalescing into one.
        void commit(Config after, std::string action, std::string mergeKey) {

            if (normalize_) normalize_(after);
            auto* target = object_;
            commands_.execute(makeProperty<Config>(
                    std::move(action), std::move(mergeKey),
                    [target, writer = writer_](const Config& value) { writer(value, *target); },
                    before_, after));
            // The widgets below this one in the same frame should show what was
            // just committed, not the value the section opened with.
            config_ = std::move(after);
            if (onEdit_) onEdit_();
        }

        // Give every field its OWN merge key, suffixed with its label, so that
        // dragging one field and then another stays two undo steps instead of
        // coalescing into one. Sections that edit several unrelated knobs of a
        // single config (joints, vehicles, sensors) want this; a section whose
        // knobs describe one thing is fine sharing a key. Opt-in, because it
        // changes what a sequence of edits undoes as.
        ConfigFields& mergePerField(bool on = true) {

            perField_ = on;
            return *this;
        }

        // Fix up a config on its way to being committed. The sensor section
        // stamps its host camera's frustum into every write, so the flat string
        // in userData cannot drift from the frustum Play will actually read.
        // Runs on the `after` value only — `before` is what was read.
        ConfigFields& normalizeWith(std::function<void(Config&)> normalize) {

            normalize_ = std::move(normalize);
            return *this;
        }

        // Configs whose undo step is not a plain write() swap the writer here:
        // TextConfig has to rebuild its geometry, so execute/undo/redo all go
        // through apply() instead. Returns *this so it chains onto construction.
        ConfigFields& writeWith(std::function<void(const Config&, Object3D&)> writer) {

            writer_ = std::move(writer);
            return *this;
        }

        [[nodiscard]] Object3D& object() const { return *object_; }
        [[nodiscard]] const std::string& mergeKey() const { return mergeKey_; }
        [[nodiscard]] CommandStack& commands() const { return commands_; }

    private:
        // Every widget above lands here, so the undo label and the merge key
        // are derived in exactly one place.
        void commitField(const char* label, const char* action, Config after) {

            commit(std::move(after), actionFor(label, action),
                   perField_ ? mergeKey_ + ":" + label : mergeKey_);
        }

        [[nodiscard]] std::string actionFor(const char* label, const char* action) const {

            if (action) return action;
            return prefix_.empty() ? std::string(label) : prefix_ + " " + label;
        }

        CommandStack& commands_;
        Object3D* object_;
        Config config_;
        Config before_;
        std::string mergeKey_;
        std::string prefix_;
        bool perField_ = false;
        std::function<void()> onEdit_;
        std::function<void(Config&)> normalize_;
        std::function<void(const Config&, Object3D&)> writer_ =
                [](const Config& value, Object3D& object) { value.write(object); };
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_CONFIGFIELDS_HPP
