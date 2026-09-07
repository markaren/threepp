// threepp.editor — the runtime face of the editor's spline authoring.
//
// One function and one class: spline_from_object(obj) turns an authored spline
// (a Group carrying userData["spline"] whose untagged children are control
// points) into a SplinePath, so a script samples the path the editor draws
// without re-deriving any of that — no child filtering, no config parsing.
//
// The contract, stated once: the LOCAL-SPACE curve (points, closed, type,
// tension) is captured when the path is created or refresh()ed; the spline's
// WORLD transform is applied live on every sample. A spline riding a moving
// platform is followable for free; editing control points mid-play needs
// refresh(). Sampling runs off the curve's arc-length table, independent of
// the editor overlay's samples-per-segment density.
#include "bindings.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include <stdexcept>

using namespace threepp;

namespace {

    CatmullRomCurve3::CurveType curveTypeOf(editor::SplineConfig::Type type) {

        switch (type) {
            case editor::SplineConfig::Type::Chordal: return CatmullRomCurve3::chordal;
            case editor::SplineConfig::Type::CatmullRom: return CatmullRomCurve3::catmullrom;
            default: return CatmullRomCurve3::centripetal;
        }
    }

    class SplinePath {

    public:
        explicit SplinePath(std::shared_ptr<Object3D> spline)
            : spline_(std::move(spline)) {

            refresh();
        }

        // Re-captures the control points and config. The world transform never
        // needs this — it is read fresh on every sample.
        void refresh() {

            config_ = editor::SplineConfig::read(*spline_).value_or(editor::SplineConfig{});
            curve_ = config_.curve(*spline_);
        }

        [[nodiscard]] bool valid() const { return curve_ != nullptr; }

        [[nodiscard]] Vector3 pointAt(float u) const {

            Vector3 point;
            sampled()->getPointAt(u, point);
            spline_->updateMatrixWorld();
            point.applyMatrix4(*spline_->matrixWorld);
            return point;
        }

        [[nodiscard]] Vector3 tangentAt(float u) const {

            Vector3 tangent;
            sampled()->getTangentAt(u, tangent);
            spline_->updateMatrixWorld();
            // Rotation and scale only, renormalized — a direction has no
            // business being translated.
            tangent.transformDirection(*spline_->matrixWorld);
            return tangent;
        }

        // Local-space length: scale the spline and distances scale with it.
        [[nodiscard]] float length() const { return sampled()->getLength(); }

        [[nodiscard]] bool closed() const { return config_.closed; }
        [[nodiscard]] float tension() const { return config_.tension; }
        [[nodiscard]] CatmullRomCurve3::CurveType curveType() const { return curveTypeOf(config_.type); }
        // The captured local-space curve, for anything the surface above does
        // not cover (get_points for drawing, arc_length_divisions, ...).
        [[nodiscard]] std::shared_ptr<CatmullRomCurve3> curve() const { return curve_; }

    private:
        std::shared_ptr<Object3D> spline_;
        editor::SplineConfig config_;
        std::shared_ptr<CatmullRomCurve3> curve_;

        // A path is only handed out valid, but refresh() can invalidate it — a
        // script whose spline lost its points gets told, not a zero vector.
        [[nodiscard]] const CatmullRomCurve3* sampled() const {

            if (!curve_) throw std::runtime_error("the spline has fewer than two control points");
            return curve_.get();
        }
    };

}// namespace

namespace threepp_py {

    void init_editor(py::module_& m) {

        auto sub = m.def_submodule("editor", "The runtime face of editor-authored data.");

        py::class_<SplinePath, std::shared_ptr<SplinePath>>(sub, "SplinePath")
                .def("get_point_at", &SplinePath::pointAt, py::arg("u"),
                     "WORLD-SPACE point at fraction u in [0, 1] of the arc length.")
                .def("get_tangent_at", &SplinePath::tangentAt, py::arg("u"),
                     "WORLD-SPACE unit tangent at fraction u of the arc length.")
                .def("get_length", &SplinePath::length,
                     "Arc length in the spline's LOCAL space.")
                .def("refresh", &SplinePath::refresh,
                     "Re-capture the control points and config. The world transform is live regardless.")
                .def_property_readonly("closed", &SplinePath::closed)
                .def_property_readonly("tension", &SplinePath::tension)
                .def_property_readonly("curve_type", &SplinePath::curveType)
                .def_property_readonly("curve", &SplinePath::curve,
                                       "The captured LOCAL-SPACE CatmullRomCurve3.");

        sub.def(
                "spline_from_object", [](const py::handle& h) -> py::object {
                    auto object = as_object3d(h);
                    if (!object || !editor::SplineConfig::isSpline(*object)) return py::none();
                    auto path = std::make_shared<SplinePath>(std::move(object));
                    // A one-point spline is not followable; None now beats a
                    // RuntimeError on the first update().
                    if (!path->valid()) return py::none();
                    return py::cast(path);
                },
                py::arg("object"),
                "The SplinePath an authored spline describes, or None when `object` is not a "
                "spline or has fewer than two control points.");
    }

}// namespace threepp_py
