// Text + 2D vector overlay: fonts (FontLoader/Font), flat & extruded text
// (Text2D/Text3D, both Meshes), world-anchored billboard labels (TextSprite),
// and SVG -> flat fill and stroke meshes (SVGLoader). FontLoader.default_font() ships an
// embedded font, so text needs no asset. Pairs with an OrthographicCamera +
// auto_clear=False overlay pass for HUDs, or place text anywhere in the scene.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/constants.hpp"
#include "threepp/extras/core/Font.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Text.hpp"
#include "threepp/objects/TextSprite.hpp"

#include <optional>

using namespace threepp;

namespace threepp_py {

    // Turn parsed SVG paths into a Group of flat meshes (z=0), in document order: per
    // element its fill, then one mesh per stroked sub-path, each named after the
    // element's id. Shapes that overlap share a depth, so the SVG's painter's order is
    // carried by renderOrder with depth writes off. The group is y-flipped so SVG's
    // y-down coordinates come out upright in the scene.
    static std::shared_ptr<Group> svg_to_group(const std::vector<SVGLoader::SVGData>& datas, unsigned int curveSegments) {
        auto group = Group::create();
        int order = 0;
        const auto add = [&](const std::shared_ptr<BufferGeometry>& geom, const std::string& paint, float opacity, const std::string& id) {
            auto mat = MeshBasicMaterial::create();
            mat->side = Side::Double;
            mat->color.setStyle(paint);
            mat->opacity = opacity;
            mat->transparent = true;// one render list, so renderOrder alone decides
            mat->depthWrite = false;
            auto mesh = Mesh::create(geom, mat);
            mesh->name = id;
            mesh->renderOrder = order++;
            group->add(mesh);
        };
        const auto painted = [](const std::optional<std::string>& paint) {
            return paint && !paint->empty() && *paint != "none";
        };
        for (const auto& d : datas) {
            const auto& style = d.style;
            if (painted(style.fill)) {
                auto shapes = SVGLoader::createShapes(d);
                if (!shapes.empty()) add(ShapeGeometry::create(shapes, curveSegments), *style.fill, style.fillOpacity, style.id);
            }
            if (painted(style.stroke) && style.strokeWidth > 0) {
                for (const auto& subPath : d.path.subPaths) {
                    if (auto geom = SVGLoader::pointsToStroke(subPath->getPoints(curveSegments), style)) {
                        add(geom, *style.stroke, style.strokeOpacity, style.id);
                    }
                }
            }
        }
        group->scale.y = -1;
        return group;
    }

    void init_text(py::module_& m) {

        // ---- Font / FontLoader ----------------------------------------------
        py::class_<Font>(m, "Font")
                .def_readonly("family_name", &Font::familyName)
                // Metrics, scaled to a text size (the `size` Text2D/Text3D take), so
                // text can be aligned before it is built: right-aligned x is
                // x - advance(text, size); the baseline sits ascender(size) below
                // the top of the line box.
                .def("advance", &Font::advance, py::arg("text"), py::arg("size") = 1.f,
                     "Width of `text` at `size`, exactly as Text2D lays it out (widest line).")
                .def("ascender", [](const Font& f, float size) {
                    return f.resolution > 0 ? static_cast<float>(f.ascender) * size / static_cast<float>(f.resolution) : 0.f;
                }, py::arg("size") = 1.f, "Height above the baseline at `size` (positive).")
                .def("descender", [](const Font& f, float size) {
                    return f.resolution > 0 ? static_cast<float>(f.descender) * size / static_cast<float>(f.resolution) : 0.f;
                }, py::arg("size") = 1.f, "Depth below the baseline at `size` (negative).")
                .def("__repr__", [](const Font& f) { return "<threepp.Font '" + f.familyName + "'>"; });

        py::class_<FontLoader>(m, "FontLoader")
                .def(py::init<>())
                .def("default_font", &FontLoader::defaultFont,
                     "The built-in embedded font (no file needed).")
                .def("load", [](FontLoader& l, const std::string& path) -> py::object {
                    auto f = l.load(path);
                    return f ? py::cast(*f) : py::none();
                }, py::arg("path"), "Load a typeface (.json) or TrueType (.ttf) font; None on failure.");

        // ---- Text2D (flat filled text, a Mesh) ------------------------------
        py::class_<Text2D, Mesh, std::shared_ptr<Text2D>>(m, "Text2D")
                .def(py::init([](const Font& font, const std::string& text, float size,
                                 unsigned int curve_segments, const py::object& matObj) {
                    TextGeometry::Options opts(font, size, curve_segments);
                    auto mat = as_material(matObj);
                    if (!mat) mat = MeshBasicMaterial::create();
                    return Text2D::create(opts, text, mat);
                }),
                     py::arg("font"), py::arg("text") = "", py::arg("size") = 1.0f,
                     py::arg("curve_segments") = 3, py::arg("material") = py::none())
                .def("set_text", [](Text2D& t, const std::string& s) { t.setText(s); }, py::arg("text"))
                .def("set_color", [](Text2D& t, const Color& c) { t.setColor(c); }, py::arg("color"));

        // ---- Text3D (extruded 3D text, a Mesh) ------------------------------
        py::class_<Text3D, Mesh, std::shared_ptr<Text3D>>(m, "Text3D")
                .def(py::init([](const Font& font, const std::string& text, float size,
                                 float height, bool bevel, const py::object& matObj) {
                    ExtrudeTextGeometry::Options opts(font, size, height);
                    // The C++ defaults (bevelThickness=10, bevelSize=8) are sized for
                    // three.js' ~100-unit text; scale them to `size` so a size=1 glyph
                    // gets a subtle bevel instead of an 8x blob.
                    opts.bevelEnabled = bevel;
                    opts.bevelThickness = size * 0.03f;
                    opts.bevelSize = size * 0.02f;
                    auto mat = as_material(matObj);
                    if (!mat) mat = MeshBasicMaterial::create();
                    return Text3D::create(opts, text, mat);
                }),
                     py::arg("font"), py::arg("text") = "", py::arg("size") = 1.0f,
                     py::arg("height") = 0.2f, py::arg("bevel") = false, py::arg("material") = py::none())
                .def("set_color", [](Text3D& t, const Color& c) { t.setColor(c); }, py::arg("color"));

        // ---- TextSprite (billboard label that always faces the camera) ------
        py::enum_<TextSprite::HorizontalAlignment>(m, "HorizontalAlignment")
                .value("Left", TextSprite::HorizontalAlignment::Left)
                .value("Center", TextSprite::HorizontalAlignment::Center)
                .value("Right", TextSprite::HorizontalAlignment::Right);
        py::enum_<TextSprite::VerticalAlignment>(m, "VerticalAlignment")
                .value("Above", TextSprite::VerticalAlignment::Above)
                .value("Center", TextSprite::VerticalAlignment::Center)
                .value("Below", TextSprite::VerticalAlignment::Below);

        // What a TextSprite's world_scale is the height of.
        py::enum_<TextBox>(m, "TextBox")
                .value("Ink", TextBox::Ink, "The string's own ink: the glyph size depends on the string.")
                .value("Line", TextBox::Line, "One line of the font, ascender to descender: the same glyph size and baseline for every string.");

        py::class_<TextSprite, Sprite, std::shared_ptr<TextSprite>>(m, "TextSprite")
                .def(py::init([](const Font& font, const py::object& world_scale, TextBox box) {
                    std::optional<float> ws;
                    if (!world_scale.is_none()) ws = world_scale.cast<float>();
                    return TextSprite::create(font, ws, box);
                }), py::arg("font"), py::arg("world_scale") = py::none(), py::arg("box") = TextBox::Ink,
                     "world_scale is the sprite's height. With box=TextBox.Ink (the default) that is the height "
                     "of the string's own ink, so capitals come out larger than a string with a descender. With "
                     "box=TextBox.Line it is the height of one line of the font, the same for every string: the "
                     "baseline lies -descender / (ascender - descender) of world_scale above the bottom edge "
                     "(Font.ascender, Font.descender).")
                .def("set_text", [](TextSprite& t, const std::string& s) { t.setText(s); }, py::arg("text"))
                .def("set_color", [](TextSprite& t, const Color& c) { t.setColor(c); }, py::arg("color"))
                .def("set_world_scale", [](TextSprite& t, float s) { t.setWorldScale(s); }, py::arg("scale"))
                .def("set_text_box", [](TextSprite& t, TextBox b) { t.setTextBox(b); }, py::arg("box"),
                     "What world_scale is the height of: the string's ink, or a line of the font.")
                .def("get_text_box", [](const TextSprite& t) { return t.getTextBox(); })
                .def("set_horizontal_alignment", [](TextSprite& t, TextSprite::HorizontalAlignment a) { t.setHorizontalAlignment(a); }, py::arg("alignment"))
                .def("set_vertical_alignment", [](TextSprite& t, TextSprite::VerticalAlignment a) { t.setVerticalAlignment(a); }, py::arg("alignment"))
                .def("get_text", [](const TextSprite& t) { return t.getText(); });

        // ---- SVGLoader (SVG -> Group of fill and stroke meshes) ---------------
        py::class_<SVGLoader>(m, "SVGLoader")
                .def(py::init<>())
                .def("load", [](SVGLoader& l, const std::string& path, unsigned int curveSegments) { return svg_to_group(l.load(path), curveSegments); },
                     py::arg("path"), py::arg("curve_segments") = 12,
                     "Load an .svg file as a Group of flat meshes: fills and strokes in document order "
                     "(render_order), each named after its element's id. curve_segments: points per curve.")
                .def("parse", [](SVGLoader& l, const std::string& text, unsigned int curveSegments) { return svg_to_group(l.parse(text), curveSegments); },
                     py::arg("text"), py::arg("curve_segments") = 12,
                     "Parse SVG XML into a Group of flat meshes: fills and strokes in document order "
                     "(render_order), each named after its element's id. curve_segments: points per curve.");
    }

}// namespace threepp_py
