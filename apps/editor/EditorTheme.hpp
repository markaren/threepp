// The editor's ImGui look.
//
// Deliberately not the ImGui default: flatter, darker, tighter, with a single
// accent colour used only for selection and active controls so the eye always
// knows what is live. Applied every frame because ImguiContext rebuilds the
// style from scratch whenever the DPI scale changes.

#ifndef THREEPP_EDITOR_EDITORTHEME_HPP
#define THREEPP_EDITOR_EDITORTHEME_HPP

struct ImVec4;

namespace threepp::editor::theme {

    // `scale` is the monitor content scale (DPI), already applied to fonts by
    // ImguiContext — every size here is multiplied by it.
    void apply(float scale);

    // Palette, exposed so panels can tint text consistently.
    ImVec4 accent();
    ImVec4 warning();
    ImVec4 danger();
    ImVec4 muted();
    ImVec4 playing();

}// namespace threepp::editor::theme

#endif//THREEPP_EDITOR_EDITORTHEME_HPP
