
#ifndef THREEPP_TRANSFORMCONTROLSGIZMO_HPP
#define THREEPP_TRANSFORMCONTROLSGIZMO_HPP

#include "TransformControlsState.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>


using namespace threepp;

using GizmoMap = std::unordered_map<std::string, std::vector<std::tuple<std::shared_ptr<Object3D>, std::optional<Vector3>, std::optional<Euler>, std::optional<Vector3>, std::optional<std::string>>>>;

class TransformControlsGizmo: public Object3D {

public:

    std::unordered_map<std::string, Object3D*> gizmo;
    std::unordered_map<std::string, Object3D*> picker;
    std::unordered_map<std::string, Object3D*> helper;

    explicit TransformControlsGizmo(State& state): state(state) {

        auto gizmoMaterial = MeshBasicMaterial::create();
        gizmoMaterial->depthTest = false;
        gizmoMaterial->depthWrite = false;
        gizmoMaterial->transparent = true;
        gizmoMaterial->side = Side::Double;
        gizmoMaterial->fog = false;
        gizmoMaterial->toneMapped = false;

        auto gizmoLineMaterial = LineBasicMaterial::create();
        gizmoLineMaterial->depthTest = false;
        gizmoLineMaterial->depthWrite = false;
        gizmoLineMaterial->transparent = true;
        gizmoMaterial->fog = false;
        gizmoMaterial->toneMapped = false;

        // Make unique material for each axis/color

        const auto matInvisible = gizmoMaterial->clone();
        matInvisible->opacity = 0.15f;

        const auto matHelper = gizmoMaterial->clone();
        matHelper->opacity = 0.33f;

        const auto matRed = gizmoMaterial->clone();
        matRed->as<MaterialWithColor>()->color.setHex(0xff0000);

        const auto matGreen = gizmoMaterial->clone();
        matGreen->as<MaterialWithColor>()->color.setHex(0x00ff00);

        const auto matBlue = gizmoMaterial->clone();
        matBlue->as<MaterialWithColor>()->color.setHex(0x0000ff);

        const auto matWhiteTransparent = gizmoMaterial->clone();
        matWhiteTransparent->opacity = 0.25f;

        const auto matYellowTransparent = matWhiteTransparent->clone();
        matYellowTransparent->as<MaterialWithColor>()->color.setHex(0xffff00);

        const auto matCyanTransparent = matWhiteTransparent->clone();
        matCyanTransparent->as<MaterialWithColor>()->color.setHex(0x00ffff);

        const auto matMagentaTransparent = matWhiteTransparent->clone();
        matMagentaTransparent->as<MaterialWithColor>()->color.setHex(0xff00ff);

        const auto matYellow = gizmoMaterial->clone();
        matYellow->as<MaterialWithColor>()->color.setHex(0xffff00);

        const auto matLineRed = gizmoLineMaterial->clone();
        matLineRed->as<MaterialWithColor>()->color.setHex(0xff0000);

        const auto matLineGreen = gizmoLineMaterial->clone();
        matLineGreen->as<MaterialWithColor>()->color.setHex(0x00ff00);

        const auto matLineBlue = gizmoLineMaterial->clone();
        matLineBlue->as<MaterialWithColor>()->color.setHex(0x0000ff);

        const auto matLineCyan = gizmoLineMaterial->clone();
        matLineCyan->as<MaterialWithColor>()->color.setHex(0x00ffff);

        const auto matLineMagenta = gizmoLineMaterial->clone();
        matLineMagenta->as<MaterialWithColor>()->color.setHex(0xff00ff);

        const auto matLineYellow = gizmoLineMaterial->clone();
        matLineYellow->as<MaterialWithColor>()->color.setHex(0xffff00);

        const auto matLineGray = gizmoLineMaterial->clone();
        matLineGray->as<MaterialWithColor>()->color.setHex(0x787878);

        const auto matLineYellowTransparent = matLineYellow->clone();
        matLineYellowTransparent->opacity = 0.25f;

        // reusable geometry

        const auto arrowGeometry = CylinderGeometry::create(0, 0.05, 0.2, 12, 1, false);

        const auto scaleHandleGeometry = BoxGeometry::create(0.125, 0.125, 0.125);

        const auto lineGeometry = BufferGeometry::create();
        lineGeometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>{0, 0, 0, 1, 0, 0}, 3));

        // Gizmo definitions - custom hierarchy definitions for setupGizmo() function

        // clang-format off
        GizmoMap gizmoTranslate {
                {"X", {
                              {Mesh::create(arrowGeometry, matRed), Vector3{1,0,0}, Euler{0,0,-math::PI/2}, std::nullopt, "fwd"},
                              {Mesh::create(arrowGeometry, matRed), Vector3{1,0,0}, Euler{0,0,math::PI/2}, std::nullopt, "bwd"},
                              {Line::create(lineGeometry, matLineRed), std::nullopt, std::nullopt, std::nullopt, std::nullopt}
                      }},
                {"Y", {
                              {Mesh::create(arrowGeometry, matGreen), Vector3{0,1,0}, std::nullopt, std::nullopt, "fwd"},
                              {Mesh::create(arrowGeometry, matGreen), Vector3{0,1,0}, Euler{math::PI, 0, 0}, std::nullopt, "bwd"},
                              {Line::create(lineGeometry, matLineGreen), std::nullopt, Euler{0,0,math::PI/2}, std::nullopt, std::nullopt}
                      }},
                {"Z", {
                              {Mesh::create(arrowGeometry, matBlue), Vector3{0,0,1}, Euler{math::PI/2, 0,0}, std::nullopt, "fwd"},
                              {Mesh::create(arrowGeometry, matBlue), Vector3{0,0,1}, Euler{-math::PI/2, 0, 0}, std::nullopt, "bwd"},
                              {Line::create(lineGeometry, matLineBlue), std::nullopt, Euler{0,-math::PI/2,0}, std::nullopt, std::nullopt}
                      }},
                {"XYZ", {
                             {Mesh::create(OctahedronGeometry::create(0.1, 0), matWhiteTransparent->clone()), Vector3{0,0,0}, Euler{0,0,0}, std::nullopt, std::nullopt}
                        }},
                {"XY", {
                             {Mesh::create(PlaneGeometry::create(0.295, 0.295), matYellowTransparent->clone()), Vector3{0.15,0.15,0}, std::nullopt, std::nullopt, std::nullopt},
                             {Line::create(lineGeometry, matLineYellow), Vector3{0.18, 0.3, 0}, std::nullopt, Vector3{0.125, 1, 1}, std::nullopt},
                             {Line::create(lineGeometry, matLineYellow), Vector3{0.3, 0.18, 0}, Euler{0, 0, math::PI/2}, Vector3{0.125, 1, 1}, std::nullopt}
                        }},
                {"YZ", {
                            {Mesh::create(PlaneGeometry::create(0.295, 0.295), matCyanTransparent->clone()), Vector3{0, 0.15,0.15}, Euler{0, math::PI/2, 0}, std::nullopt, std::nullopt},
                            {Line::create(lineGeometry, matLineCyan), Vector3{0, 0.18, 0.3}, Euler{0, 0, math::PI/2}, Vector3{0.125, 1, 1}, std::nullopt},
                            {Line::create(lineGeometry, matLineCyan), Vector3{0, 0.3, 0.18}, Euler{0, -math::PI/2, 0}, Vector3{0.125, 1, 1}, std::nullopt}
                       }},
                {"XZ", {
                            {Mesh::create(PlaneGeometry::create(0.295, 0.295), matMagentaTransparent->clone()), Vector3{0.15,0,0.15}, Euler{-math::PI/2, 0, 0}, std::nullopt, std::nullopt},
                            {Line::create(lineGeometry, matLineMagenta), Vector3{0.18, 0, 0.3}, std::nullopt, Vector3{0.125, 1, 1}, std::nullopt},
                            {Line::create(lineGeometry, matLineMagenta), Vector3{0.3, 0, 0.18}, Euler{0, -math::PI/2, 0}, Vector3{0.125, 1, 1}, std::nullopt}
                       }}
        };

        GizmoMap pickerTranslate {
                {"X", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 1, 4, 1, false), matInvisible), Vector3{0.6, 0, 0}, Euler{0, 0, -math::PI/2}, std::nullopt, std::nullopt}
                      }},
                {"Y", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 1, 4, 1, false), matInvisible), Vector3{0, 0.6, 0}, std::nullopt, std::nullopt, std::nullopt}
                      }},
                {"Z", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 1, 4, 1, false), matInvisible), Vector3{0, 0, 0.6}, Euler{math::PI/2, 0, 0}, std::nullopt, std::nullopt}
                      }},
                {"XYZ", {
                            {Mesh::create(OctahedronGeometry::create(0.2, 0), matInvisible), std::nullopt, std::nullopt, std::nullopt, std::nullopt}
                      }},
                {"XY", {
                            {Mesh::create(PlaneGeometry::create(0.4, 0.4), matInvisible), Vector3{0.2, 0.2, 0}, std::nullopt, std::nullopt, std::nullopt}
                      }},
                {"YZ", {
                           {Mesh::create(PlaneGeometry::create(0.4, 0.4), matInvisible), Vector3{0, 0.2, 0.2}, Euler{0, math::PI/2, 0}, std::nullopt, std::nullopt}
                     }},
                {"XZ", {
                           {Mesh::create(PlaneGeometry::create(0.4, 0.4), matInvisible), Vector3{0.2, 0, 0.2}, Euler{-math::PI/2, 0, 0}, std::nullopt, std::nullopt}
                     }}
        };

        GizmoMap helperTranslate {
                {"START", {
                            {Mesh::create(OctahedronGeometry::create(0.01, 2), matHelper), std::nullopt, std::nullopt, std::nullopt, "helper"}
                      }},
                {"END", {
                            {Mesh::create(OctahedronGeometry::create(0.01, 2), matHelper), std::nullopt, std::nullopt, std::nullopt, "helper"}
                      }},
                {"DELTA", {
                            {Line::create(TranslateHelperGeometry(), matHelper), std::nullopt, std::nullopt, std::nullopt, "helper"}
                      }},
                {"X", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{-1e3, 0, 0}, std::nullopt, Vector3{1e6, 1, 1}, "helper"}
                      }},
                {"Y", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{0, -1e3, 0}, Euler{0, 0, math::PI/2}, Vector3{1e6, 1, 1}, "helper"}
                      }},
                {"Z", {
                           {Line::create(lineGeometry, matHelper->clone()), Vector3{0, 0, -1e3}, Euler{0, -math::PI/2, 0}, Vector3{1e6, 1, 1}, "helper"}
                     }}
                };

        GizmoMap gizmoRotate {
                {"X", {
                            {Line::create(CircleGeometry(1, 0.5), matLineRed), std::nullopt, std::nullopt, std::nullopt, std::nullopt},
                            {Mesh::create(OctahedronGeometry::create(0.04, 0), matRed), Vector3{0, 0, 0.99}, std::nullopt, Vector3{1, 3, 1}, std::nullopt}
                      }},
                {"Y", {
                            {Line::create(CircleGeometry(1, 0.5), matLineGreen), std::nullopt, Euler{0, 0, -math::PI/2}, std::nullopt, std::nullopt},
                            {Mesh::create(OctahedronGeometry::create(0.04, 0), matGreen), Vector3{0, 0, 0.99}, std::nullopt, Vector3{3, 1, 1}, std::nullopt}
                      }},
                {"Z", {
                            {Line::create(CircleGeometry(1, 0.5), matLineBlue), std::nullopt, Euler{0, math::PI/2, 0}, std::nullopt, std::nullopt},
                            {Mesh::create(OctahedronGeometry::create(0.04, 0), matBlue), Vector3{0.99, 0, 0}, std::nullopt, Vector3{1, 3, 1}, std::nullopt},
                      }},
                {"E", {
                            {Line::create(CircleGeometry(1.25, 1), matLineYellowTransparent), std::nullopt, Euler{0, math::PI/2, 0}, std::nullopt, std::nullopt},
                            {Mesh::create(CylinderGeometry::create(0.03, 0, 0.15, 4, 1, false), matLineYellowTransparent), Vector3{1.17, 0, 0}, Euler{0, 0, -math::PI/2}, Vector3{1, 1, 0.001}, std::nullopt},
                            {Mesh::create(CylinderGeometry::create(0.03, 0, 0.15, 4, 1, false), matLineYellowTransparent), Vector3{-1.17, 0, 0}, Euler{0, 0, math::PI/2}, Vector3{1, 1, 0.001}, std::nullopt},
                            {Mesh::create(CylinderGeometry::create(0.03, 0, 0.15, 4, 1, false), matLineYellowTransparent), Vector3{0, -1.17, 0}, Euler{math::PI, 0, 0}, Vector3{1, 1, 0.001}, std::nullopt},
                            {Mesh::create(CylinderGeometry::create(0.03, 0, 0.15, 4, 1, false), matLineYellowTransparent), Vector3{0, 1.17, 0}, Euler{0, 0, 0}, Vector3{1, 1, 0.001}, std::nullopt},
                      }},
                {"XYZE", {
                            {Line::create(CircleGeometry(1, 1), matLineGray), std::nullopt, Euler{0, math::PI/2, 0}, std::nullopt, std::nullopt}
                      }}
                };

        GizmoMap helperRotate {
                {"AXIS", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{-1e3, 0, 0}, std::nullopt, Vector3{1e6, 1, 1}, "helper"}
                     }}
                };

        GizmoMap pickerRotate {
                {"X", {
                            {Mesh::create(TorusGeometry::create(1, 0.1, 4, 24), matInvisible), Vector3{0, 0, 0}, Euler{0, -math::PI/2, -math::PI/2}, std::nullopt, std::nullopt}
                      }},
                {"Y", {
                            {Mesh::create(TorusGeometry::create(1, 0.1, 4, 24), matInvisible), Vector3{0, 0, 0}, Euler{math::PI/2, 0, 0}, std::nullopt, std::nullopt}
                      }},
                {"Z", {
                            {Mesh::create(TorusGeometry::create(1, 0.1, 4, 24), matInvisible), Vector3{0, 0, 0}, Euler{0, 0, -math::PI/2}, std::nullopt, std::nullopt},
                      }},
                {"E", {
                            {Mesh::create(TorusGeometry::create(1.25, 0.1, 2, 24), matInvisible), std::nullopt, std::nullopt, std::nullopt, std::nullopt},
                      }},
                {"XYZE", {
                            {Mesh::create(SphereGeometry::create(0.7, 10, 8), matInvisible), std::nullopt, std::nullopt, std::nullopt, std::nullopt}
                      }}
                };

        GizmoMap gizmoScale {
                {"X", {
                            {Mesh::create(scaleHandleGeometry, matRed), Vector3{0.8, 0, 0}, Euler{0, 0, -math::PI/2}, std::nullopt, std::nullopt},
                            {Line::create(lineGeometry, matLineRed), std::nullopt, std::nullopt, Vector3{0.8, 1, 1}, std::nullopt}
                      }},
                {"Y", {
                            {Mesh::create(scaleHandleGeometry, matGreen), Vector3{0, 0.8, 0}, std::nullopt, std::nullopt, std::nullopt},
                            {Line::create(lineGeometry, matLineGreen), std::nullopt, Euler{0, 0, math::PI/2}, Vector3{0.8, 1, 1}, std::nullopt}
                      }},
                {"Z", {
                            {Mesh::create(scaleHandleGeometry, matBlue), Vector3{0, 0, 0.8}, Euler{math::PI/2, 0, 0}, std::nullopt, std::nullopt},
                            {Line::create(lineGeometry, matLineBlue), std::nullopt, Euler{0, -math::PI/2, 0}, Vector3{0.8, 1, 1}, std::nullopt}
                      }},
                {"XY", {
                            {Mesh::create(scaleHandleGeometry, matYellowTransparent), Vector3{0.85, 0.85, 0}, std::nullopt, Vector3{2, 2, 0.2}, std::nullopt},
                            {Line::create(lineGeometry, matLineYellow), Vector3{0.855, 0.98, 0}, std::nullopt, Vector3{0.125, 1, 1}, std::nullopt},
                            {Line::create(lineGeometry, matLineYellow), Vector3{0.98, 0.855, 0}, Euler{0, 0, math::PI/2}, Vector3{0.125, 1, 1}, std::nullopt}
                      }},
                {"YZ", {
                            {Mesh::create(scaleHandleGeometry, matCyanTransparent), Vector3{0, 0.85, 0.85}, std::nullopt, Vector3{0.2, 2, 2}, std::nullopt},
                            {Line::create(lineGeometry, matLineCyan), Vector3{0, 0.855, 0.98}, Euler{0, 0, math::PI/2}, Vector3{0.125, 1, 1}, std::nullopt},
                            {Line::create(lineGeometry, matLineCyan), Vector3{0, 0.98, 0.855}, Euler{0, -math::PI/2, 0}, Vector3{0.125, 1, 1}, std::nullopt}
                      }},
                {"XZ", {
                           {Mesh::create(scaleHandleGeometry, matMagentaTransparent), Vector3{0.85, 0, 0.85}, std::nullopt, Vector3{2, 0.2, 2}, std::nullopt},
                           {Line::create(lineGeometry, matLineMagenta), Vector3{0.855, 0, 0.98}, Euler{0, 0, math::PI/2}, Vector3{0.125, 1, 1}, std::nullopt},
                           {Line::create(lineGeometry, matLineMagenta), Vector3{0.98, 0, 0.855}, Euler{0, -math::PI/2, 0}, Vector3{0.125, 1, 1}, std::nullopt}
                     }},
                {"XYZX", {
                           {Mesh::create(BoxGeometry::create(0.125, 0.125, 0.125), matWhiteTransparent->clone()), Vector3{1.1, 0, 0}, std::nullopt, std::nullopt, std::nullopt}
                     }},
                {"XYZY", {
                           {Mesh::create(BoxGeometry::create(0.125, 0.125, 0.125), matWhiteTransparent->clone()), Vector3{0, 1.1, 0}, std::nullopt, std::nullopt, std::nullopt}
                     }},
                {"XYZZ", {
                           {Mesh::create(BoxGeometry::create(0.125, 0.125, 0.125), matWhiteTransparent->clone()), Vector3{0, 0, 1.1}, std::nullopt, std::nullopt, std::nullopt}
                     }}
        };

        GizmoMap pickerScale {
                {"X", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 0.8, 4, 1, false), matInvisible), Vector3{0.5, 0, 0}, Euler{0, 0, -math::PI/2}, std::nullopt, std::nullopt}
                      }},
                {"Y", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 0.8, 4, 1, false), matInvisible), Vector3{0, 0.5, 0}, std::nullopt, std::nullopt, std::nullopt}
                      }},
                {"Z", {
                            {Mesh::create(CylinderGeometry::create(0.2, 0, 0.8, 4, 1, false), matInvisible), Vector3{0, 0, 0.5}, Euler{math::PI/2, 0, 0}, std::nullopt, std::nullopt}
                      }},
                {"XY", {
                            {Mesh::create(scaleHandleGeometry, matInvisible), Vector3{0.85, 0.85, 0}, std::nullopt, Vector3{3, 3, 0.2}, std::nullopt}
                      }},
                {"YZ", {
                            {Mesh::create(scaleHandleGeometry, matInvisible), Vector3{0, 0.85, 0.85}, std::nullopt, Vector3{0.2, 3, 3}, std::nullopt}
                      }},
                {"XZ", {
                           {Mesh::create(scaleHandleGeometry, matInvisible), Vector3{0.85, 0, 0.85}, std::nullopt, Vector3{3, 0.2, 3}, std::nullopt}
                     }},
                {"XYZX", {
                           {Mesh::create(BoxGeometry::create(0.2, 0.2, 0.2), matInvisible), Vector3{1.1, 0, 0}, std::nullopt, std::nullopt, std::nullopt}
                     }},
                {"XYZY", {
                           {Mesh::create(BoxGeometry::create(0.2, 0.2, 0.2), matInvisible), Vector3{0, 1.1, 0}, std::nullopt, std::nullopt, std::nullopt}
                     }},
                {"XYZZ", {
                           {Mesh::create(BoxGeometry::create(0.2, 0.2, 0.2), matInvisible), Vector3{0, 0, 1.1}, std::nullopt, std::nullopt, std::nullopt}
                     }}
        };

        GizmoMap helperScale {
                {"X", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{-1e3, 0, 0}, std::nullopt, Vector3{1e6, 1, 1}, "helper"}
                     }},
                {"Y", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{0, -1e3, 0}, Euler{0, 0, math::PI/2}, Vector3{1e6, 1, 1}, "helper"}
                     }},
                {"Z", {
                            {Line::create(lineGeometry, matHelper->clone()), Vector3{0, 0, -1e3}, Euler{0, -math::PI/2, 0}, Vector3{1e6, 1, 1}, "helper"}
                     }}
                };

        // clang-format on

        {
            auto translate = setupGizmo(gizmoTranslate);
            this->gizmo["translate"] = translate.get();
            add(translate);

            auto rotate = setupGizmo(gizmoRotate);
            this->gizmo["rotate"] = rotate.get();
            add(rotate);

            auto scale = setupGizmo(gizmoScale);
            this->gizmo["scale"] = scale.get();
            add(scale);
        }

        {
            auto translate = setupGizmo(pickerTranslate);
            this->picker["translate"] = translate.get();
            add(translate);

            auto rotate = setupGizmo(pickerRotate);
            this->picker["rotate"] = rotate.get();
            add(rotate);

            auto scale = setupGizmo(pickerScale);
            this->picker["scale"] = scale.get();
            add(scale);
        }

        {
            auto translate = setupGizmo(helperTranslate);
            this->helper["translate"] = translate.get();
            add(translate);

            auto rotate = setupGizmo(helperRotate);
            this->helper["rotate"] = rotate.get();
            add(rotate);

            auto scale = setupGizmo(helperScale);
            this->helper["scale"] = scale.get();
            add(scale);
        }

        this->picker["translate"]->visible = false;
        this->picker["rotate"]->visible = false;
        this->picker["scale"]->visible = false;
    }

    std::shared_ptr<Object3D> setupGizmo(const GizmoMap& gizmoMap) {

        const auto gizmo = Object3D::create();

        for (const auto& [name, value] : gizmoMap) {

            for (unsigned i = value.size(); i--;) {

                auto object = std::get<0>(value[i])->clone();
                const auto position = std::get<1>(value[i]);
                const auto rotation = std::get<2>(value[i]);
                const auto scale = std::get<3>(value[i]);
                const auto tag = std::get<4>(value[i]);

                // name and tag properties are essential for picking and updating logic.
                object->name = name;
                if (tag) object->userData["tag"] = *tag;

                if (position) {

                    object->position.copy(*position);
                }

                if (rotation) {

                    object->rotation.copy(*rotation);
                }

                if (scale) {

                    object->scale.copy(*scale);
                }

                object->updateMatrix();

                const auto tempGeometry = object->geometry()->clone();
                tempGeometry->applyMatrix4(*object->matrix);
                if (auto mesh = object->as<Mesh>()) {
                    mesh->setGeometry(tempGeometry);
                } else if (auto line = object->as<Line>()) {
                    line->setGeometry(tempGeometry);
                } else {
                    throw std::runtime_error("GizmoObject::setupGizmo: invalid type");
                }

                object->renderOrder = std::numeric_limits<int>::infinity();

                object->position.set(0, 0, 0);
                object->rotation.set(0, 0, 0);
                object->scale.set(1, 1, 1);

                gizmo->add(object);
            }
        }

        return gizmo;
    }

    void updateMatrixWorld(bool force) override {

        const auto space = (state.mode == "scale") ? "local" : state.space;// scale always oriented to local rotation

        const auto quaternion = (space == "local") ? state.worldQuaternion : _identityQuaternion;

        // Show only gizmos for current transform mode

        this->gizmo["translate"]->visible = state.mode == "translate";
        this->gizmo["rotate"]->visible = state.mode == "rotate";
        this->gizmo["scale"]->visible = state.mode == "scale";

        this->helper["translate"]->visible = state.mode == "translate";
        this->helper["rotate"]->visible = state.mode == "rotate";
        this->helper["scale"]->visible = state.mode == "scale";


        std::vector<Object3D*> handles;
        for (auto obj : this->picker[state.mode]->children) {
            handles.emplace_back(obj);
        }
        for (auto obj : this->gizmo[state.mode]->children) {
            handles.emplace_back(obj);
        }
        for (auto obj : this->helper[state.mode]->children) {
            handles.emplace_back(obj);
        }


        for (auto handle : handles) {

            // hide aligned to camera

            handle->visible = true;
            handle->rotation.set(0, 0, 0);
            handle->position.copy(state.worldPosition);

            float factor;

            if (auto orthoCam = this->state.camera->as<OrthographicCamera>()) {

                factor = (orthoCam->top - orthoCam->bottom) / orthoCam->zoom;

            } else {

                auto perspCam = this->state.camera->as<PerspectiveCamera>();
                factor = state.worldPosition.distanceTo(this->state.cameraPosition) * std::min(1.9f * std::tan(math::PI * perspCam->fov / 360.f) / perspCam->zoom, 7.f);
            }

            handle->scale.set(1.f, 1.f, 1.f).multiplyScalar(factor * this->state.size / 7);

            // TODO: simplify helpers and consider decoupling from gizmo

            if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "helper") {

                handle->visible = false;

                if (handle->name == "AXIS") {

                    handle->position.copy(this->state.worldPositionStart);
                    handle->visible = state.axis.has_value();

                    if (state.axis == "X") {

                        _tempQuaternion.setFromEuler(_tempEuler.set(0, 0, 0));
                        handle->quaternion.copy(quaternion).multiply(_tempQuaternion);

                        if (std::abs(_alignVector.copy(_unitX).applyQuaternion(quaternion).dot(this->state.eye)) > 0.9) {

                            handle->visible = false;
                        }
                    }

                    if (this->state.axis == "Y") {

                        _tempQuaternion.setFromEuler(_tempEuler.set(0, 0, math::PI / 2));
                        handle->quaternion.copy(quaternion).multiply(_tempQuaternion);

                        if (std::abs(_alignVector.copy(_unitY).applyQuaternion(quaternion).dot(this->state.eye)) > 0.9) {

                            handle->visible = false;
                        }
                    }

                    if (this->state.axis == "Z") {

                        _tempQuaternion.setFromEuler(_tempEuler.set(0, math::PI / 2, 0));
                        handle->quaternion.copy(quaternion).multiply(_tempQuaternion);

                        if (std::abs(_alignVector.copy(_unitZ).applyQuaternion(quaternion).dot(this->state.eye)) > 0.9f) {

                            handle->visible = false;
                        }
                    }

                    if (this->state.axis == "XYZE") {

                        _tempQuaternion.setFromEuler(_tempEuler.set(0, math::PI / 2, 0));
                        _alignVector.copy(this->state.rotationAxis);
                        handle->quaternion.setFromRotationMatrix(_lookAtMatrix.lookAt(_zeroVector, _alignVector, _unitY));
                        handle->quaternion.multiply(_tempQuaternion);
                        handle->visible = this->state.dragging;
                    }

                    if (this->state.axis == "E") {

                        handle->visible = false;
                    }


                } else if (handle->name == "START") {

                    handle->position.copy(this->state.worldPositionStart);
                    handle->visible = this->state.dragging;

                } else if (handle->name == "END") {

                    handle->position.copy(this->state.worldPosition);
                    handle->visible = this->state.dragging;

                } else if (handle->name == "DELTA") {

                    handle->position.copy(this->state.worldPositionStart);
                    handle->quaternion.copy(this->state.worldQuaternionStart);
                    _tempVector.set(1e-10, 1e-10, 1e-10).add(this->state.worldPositionStart).sub(this->state.worldPosition).multiplyScalar(-1.f);
                    _tempVector.applyQuaternion(this->state.worldQuaternionStart.clone().invert());
                    handle->scale.copy(_tempVector);
                    handle->visible = this->state.dragging;

                } else {

                    handle->quaternion.copy(quaternion);

                    if (this->state.dragging) {

                        handle->position.copy(this->state.worldPositionStart);

                    } else {

                        handle->position.copy(this->state.worldPosition);
                    }

                    if (this->state.axis) {

                        handle->visible = this->state.axis->find(handle->name) != std::string::npos;
                    }
                }// If updating helper, skip rest of the loop

                continue;

            }// Align handles to current local or world rotation

            handle->quaternion.copy(quaternion);

            if (this->state.mode == "translate" || this->state.mode == "scale") {

                // Hide translate and scale axis facing the camera

                const auto AXIS_HIDE_TRESHOLD = 0.99;
                const auto PLANE_HIDE_TRESHOLD = 0.2;
                const auto AXIS_FLIP_TRESHOLD = 0.0;

                if (handle->name == "X" || handle->name == "XYZX") {

                    if (std::abs(_alignVector.copy(_unitX).applyQuaternion(quaternion).dot(this->state.eye)) > AXIS_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                if (handle->name == "Y" || handle->name == "XYZY") {

                    if (std::abs(_alignVector.copy(_unitY).applyQuaternion(quaternion).dot(this->state.eye)) > AXIS_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                if (handle->name == "Z" || handle->name == "XYZZ") {

                    if (std::abs(_alignVector.copy(_unitZ).applyQuaternion(quaternion).dot(this->state.eye)) > AXIS_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                if (handle->name == "XY") {

                    if (std::abs(_alignVector.copy(_unitZ).applyQuaternion(quaternion).dot(this->state.eye)) < PLANE_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                if (handle->name == "YZ") {

                    if (std::abs(_alignVector.copy(_unitX).applyQuaternion(quaternion).dot(this->state.eye)) < PLANE_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                if (handle->name == "XZ") {

                    if (std::abs(_alignVector.copy(_unitY).applyQuaternion(quaternion).dot(this->state.eye)) < PLANE_HIDE_TRESHOLD) {

                        handle->scale.set(1e-10, 1e-10, 1e-10);
                        handle->visible = false;
                    }
                }

                // Flip translate and scale axis ocluded behind another axis

                if (handle->name.find('X') != std::string::npos) {

                    if (_alignVector.copy(_unitX).applyQuaternion(quaternion).dot(this->state.eye) < AXIS_FLIP_TRESHOLD) {

                        if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "fwd") {

                            handle->visible = false;

                        } else {

                            handle->scale.x *= -1;
                        }

                    } else if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "bwd") {

                        handle->visible = false;
                    }
                }

                if (handle->name.find('Y') != std::string::npos) {

                    if (_alignVector.copy(_unitY).applyQuaternion(quaternion).dot(this->state.eye) < AXIS_FLIP_TRESHOLD) {

                        if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "fwd") {

                            handle->visible = false;

                        } else {

                            handle->scale.y *= -1;
                        }

                    } else if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "bwd") {

                        handle->visible = false;
                    }
                }

                if (handle->name.find('Z') != std::string::npos) {

                    if (_alignVector.copy(_unitZ).applyQuaternion(quaternion).dot(this->state.eye) < AXIS_FLIP_TRESHOLD) {

                        if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "fwd") {

                            handle->visible = false;

                        } else {

                            handle->scale.z *= -1;
                        }

                    } else if (handle->userData.contains("tag") && std::any_cast<std::string>(handle->userData["tag"]) == "bwd") {

                        handle->visible = false;
                    }
                }

            } else if (this->state.mode == "rotate") {

                // Align handles to current local or world rotation

                _tempQuaternion2.copy(quaternion);
                _alignVector.copy(this->state.eye).applyQuaternion(_tempQuaternion.copy(quaternion).invert());

                if (handle->name.find('E') != std::string::npos) {

                    handle->quaternion.setFromRotationMatrix(_lookAtMatrix.lookAt(this->state.eye, _zeroVector, _unitY));
                }

                if (handle->name == "X") {

                    _tempQuaternion.setFromAxisAngle(_unitX, std::atan2(-_alignVector.y, _alignVector.z));
                    _tempQuaternion.multiplyQuaternions(_tempQuaternion2, _tempQuaternion);
                    handle->quaternion.copy(_tempQuaternion);
                }

                if (handle->name == "Y") {

                    _tempQuaternion.setFromAxisAngle(_unitY, std::atan2(_alignVector.x, _alignVector.z));
                    _tempQuaternion.multiplyQuaternions(_tempQuaternion2, _tempQuaternion);
                    handle->quaternion.copy(_tempQuaternion);
                }

                if (handle->name == "Z") {

                    _tempQuaternion.setFromAxisAngle(_unitZ, std::atan2(_alignVector.y, _alignVector.x));
                    _tempQuaternion.multiplyQuaternions(_tempQuaternion2, _tempQuaternion);
                    handle->quaternion.copy(_tempQuaternion);
                }
            }

            // Hide disabled axes
            handle->visible = handle->visible && (handle->name.find('X') == std::string::npos || this->state.showX);
            handle->visible = handle->visible && (handle->name.find('Y') == std::string::npos || this->state.showY);
            handle->visible = handle->visible && (handle->name.find('Z') == std::string::npos || this->state.showZ);
            handle->visible = handle->visible && (handle->name.find('E') == std::string::npos || (this->state.showX && this->state.showY && this->state.showZ));

            // highlight selected axis

            if (auto mat = handle->material()) {

                // Save originals on first encounter
                if (!handle->userData.contains("__orig_opacity")) {
                    handle->userData["__orig_opacity"] = mat->opacity;
                }

                if (!handle->userData.contains("__orig_color")) {
                    if (auto mwc = mat->as<MaterialWithColor>()) {
                        handle->userData["__orig_color"] = mwc->color;// copy stored
                    }
                }

                // Restore original color (if material supports color)
                if (auto mwc = mat->as<MaterialWithColor>()) {
                    if (handle->userData.contains("__orig_color")) {
                        mwc->color.copy(std::any_cast<Color>(handle->userData["__orig_color"]));
                    }
                }

                // Restore original opacity
                if (handle->userData.contains("__orig_opacity")) {
                    mat->opacity = std::any_cast<float>(handle->userData["__orig_opacity"]);
                }
            }

            if (!this->state.enabled) {

                handle->material()->opacity *= 0.5;
                handle->materialAs<MaterialWithColor>()->color.lerp(Color(1, 1, 1), 0.5);

            } else if (this->state.axis) {

                if (handle->name == this->state.axis) {

                    handle->material()->opacity = 1.0;
                    handle->materialAs<MaterialWithColor>()->color.lerp(Color(1, 1, 1), 0.5);

                } else if (std::ranges::any_of(this->state.axis.value(),
                                               [&](char a) {
                                                   return handle->name.size() == 1 && handle->name[0] == a;
                                               })) {

                    handle->material()->opacity = 1.0;
                    handle->materialAs<MaterialWithColor>()->color.lerp(Color(1, 1, 1), 0.5f);

                } else {

                    handle->material()->opacity *= 0.25;
                    handle->materialAs<MaterialWithColor>()->color.lerp(Color(1, 1, 1), 0.5f);
                }
            }
        }


        Object3D::updateMatrixWorld(force);
    }
private:

    State& state;

    Vector3 _tempVector;
    Vector3 _zeroVector;

    Euler _tempEuler;

    Quaternion _identityQuaternion;
    Quaternion _tempQuaternion;
    Quaternion _tempQuaternion2;

    Matrix4 _tempMatrix;
    Matrix4 _lookAtMatrix;

    Vector3 _unitX = Vector3(1, 0, 0), _unitY = Vector3(0, 1, 0), _unitZ = Vector3(0, 0, 1);
    Vector3 _dirVector, _alignVector;


    static std::shared_ptr<BufferGeometry> CircleGeometry(float radius, float arc) {

        auto geometry = BufferGeometry::create();
        std::vector<float> vertices;

        for (auto i = 0; i <= 64 * arc; ++i) {

            vertices.emplace_back(0.f);
            vertices.emplace_back(std::cos(static_cast<float>(i) / 32 * math::PI) * radius);
            vertices.emplace_back(std::sin(static_cast<float>(i) / 32 * math::PI) * radius);
        }

        geometry->setAttribute("position", FloatBufferAttribute::create(vertices, 3));

        return geometry;
    }

    static std::shared_ptr<BufferGeometry> TranslateHelperGeometry() {

        auto geometry = BufferGeometry::create();

        geometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>{0, 0, 0, 1, 1, 1}, 3));

        return geometry;
    }
};


#endif //THREEPP_TRANSFORMCONTROLSGIZMO_HPP
