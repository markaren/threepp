// Text authoring, stored on the mesh itself.
//
// A text object is an ordinary Mesh whose geometry is BUILT from what its
// userData says, so everything the editor already has — the gizmo, materials,
// physics, undo, serialization — applies to it untouched. The mesh saves with
// its geometry baked in like any other mesh, which means a saved scene renders
// (and collides) with no editor and no font present; the config is only needed
// to EDIT the text again, and it travels in the same file.
//
// Storage follows GeneratorConfig's reasoning: the content is free text
// (newlines and all), so it cannot ride in the flat key=value format the
// parameters use. Two entries:
//
//   userData["text"]        Hello\nWorld          - the content, verbatim
//   userData["textParams"]  size=0.5;depth=0.1    - the numbers
//
// The "text" entry's presence is what makes a mesh a text object. There is no
// "enabled" flag and write() never erases it, for SplineConfig's reason: the
// entry IS the thing.
//
// The glyphs come from the embedded default font (FontLoader::defaultFont),
// deliberately: a font FILE path in the document would stop the scene being
// self-contained, which is the same line GeneratorConfig holds against
// script paths.

#ifndef THREEPP_EDITOR_TEXTCONFIG_HPP
#define THREEPP_EDITOR_TEXTCONFIG_HPP

#include <memory>
#include <optional>
#include <string>

namespace threepp {

    class BufferGeometry;
    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct TextConfig {

        // Where the mesh's origin sits on the text block — what the gizmo
        // grabs and what position means. Center is the default: a title is
        // placed by its middle. Left and Right put the origin on that edge,
        // for text that grows away from an anchor as it is edited.
        // Vertically the origin is always the block's middle.
        enum class Align {
            Left,
            Center,
            Right
        };

        std::string text = "Text";
        // Glyph height in world units.
        float size = 0.5f;
        // Extrusion depth; 0 is flat (a sign), anything else is solid type.
        float depth = 0.1f;
        // Curve subdivisions per glyph outline segment.
        int curveSegments = 4;
        Align align = Align::Center;

        static constexpr const char* textKey = "text";
        static constexpr const char* paramsKey = "textParams";
        // Beyond this a glyph costs more than it shows.
        static constexpr int maxCurveSegments = 12;

        [[nodiscard]] std::string encodeParams() const;

        // nullopt when the object is not a text mesh.
        [[nodiscard]] static std::optional<TextConfig> read(const Object3D& object);

        // Writes both entries — see the header note.
        void write(Object3D& object) const;

        // Carrying the "text" entry is the whole definition.
        [[nodiscard]] static bool isText(const Object3D& object);

        // The geometry this config describes: flat shapes at depth 0, an
        // extrusion otherwise, anchored per `align` with its cached bounds
        // refreshed — the selection outline and the raycast read those, so a
        // box left where the glyphs used to be is a text nobody can click.
        // Empty content builds an empty geometry — a text object with nothing
        // to say draws nothing, it does not disappear from the document.
        [[nodiscard]] std::shared_ptr<BufferGeometry> buildGeometry() const;

        static const char* label(Align align);

        // write() plus the geometry swap on a Mesh, as one call: the factory
        // and every inspector edit (execute, undo and redo alike) go through
        // this, so the entries and the triangles cannot disagree.
        void apply(Object3D& object) const;

        bool operator==(const TextConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_TEXTCONFIG_HPP
