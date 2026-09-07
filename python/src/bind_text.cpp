// Text + 2D vector overlay: fonts (FontLoader/Font), flat & extruded text
// (Text2D/Text3D, both Meshes), world-anchored billboard labels (TextSprite),
// and SVG -> filled meshes (SVGLoader). FontLoader.default_font() ships an
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

    // Turn parsed SVG paths into a Group of flat filled meshes (z=0). The group is
    // y-flipped so SVG's y-down coordinates come out upright in the scene.
    static std::shared_ptr<Group> svg_to_group(const std::vector<SVGLoader::SVGData>& datas) {
        auto group = Group::create();
        for (const auto& d : datas) {
            auto shapes = SVGLoader::createShapes(d);
            if (shapes.empty()) continue;
            auto geom = ShapeGeometry::create(shapes);
            auto mat = MeshBasicMaterial::create();
            mat->side = Side::Double;
            if (d.style.fill && *d.style.fill != "none") {
                Color c;
                c.setStyle(*d.style.fill);
                mat->color = c;
            }
            group->add(Mesh::create(geom, mat));
        }
        group->scale.y = -1;
        return group;
    }

    void init_text(py::module_& m) {

        // ---- Font / FontLoader ----------------------------------------------
        py::class_<Font>(m, "Font")
                .def_readonly("family_name", &Font::familyName)
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

        py::class_<TextSprite, Sprite, std::shared_ptr<TextSprite>>(m, "TextSprite")
                .def(py::init([](const Font& font, const py::object& world_scale) {
                    std::optional<float> ws;
                    if (!world_scale.is_none()) ws = world_scale.cast<float>();
                    return TextSprite::create(font, ws);
                }), py::arg("font"), py::arg("world_scale") = py::none())
                .def("set_text", [](TextSprite& t, const std::string& s) { t.setText(s); }, py::arg("text"))
                .def("set_color", [](TextSprite& t, const Color& c) { t.setColor(c); }, py::arg("color"))
                .def("set_world_scale", [](TextSprite& t, float s) { t.setWorldScale(s); }, py::arg("scale"))
                .def("set_horizontal_alignment", [](TextSprite& t, TextSprite::HorizontalAlignment a) { t.setHorizontalAlignment(a); }, py::arg("alignment"))
                .def("set_vertical_alignment", [](TextSprite& t, TextSprite::VerticalAlignment a) { t.setVerticalAlignment(a); }, py::arg("alignment"))
                .def("get_text", [](const TextSprite& t) { return t.getText(); });

        // ---- SVGLoader (SVG -> Group of filled meshes) ----------------------
        py::class_<SVGLoader>(m, "SVGLoader")
                .def(py::init<>())
                .def("load", [](SVGLoader& l, const std::string& path) { return svg_to_group(l.load(path)); },
                     py::arg("path"), "Load an .svg file as a Group of filled meshes.")
                .def("parse", [](SVGLoader& l, const std::string& text) { return svg_to_group(l.parse(text)); },
                     py::arg("text"), "Parse SVG XML into a Group of filled meshes.");
    }

}// namespace threepp_py
