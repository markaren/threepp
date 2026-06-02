// Flat screen-space SVG user interface — a "bridge HMI" overlay.
//
// Demonstrates how to build a full, interactive 2D UI out of SVG on top of a
// live 3D scene, reusing threepp's existing overlay infrastructure:
//
//   * SVG vector art (compass, gauges, panels, buttons) is parsed by SVGLoader
//     into ShapeGeometry meshes and drawn in a second, ortho-camera overlay
//     pass (autoClear=false + clearDepth), so it composites on top of the 3D
//     world.
//   * All text uses screen-space TextSprite (Sprite::screenSpace) — the
//     renderer overlays these automatically via its internal ortho cam.
//   * Interaction reuses SpriteInteractor: each button carries a transparent
//     screen-space Sprite hit-target whose onMouseDown/onMouseUp fire the
//     button action. Hover is mirrored from the same pixel-AABB convention.
//
// Coordinate convention (matches the renderer's screen-space sprite pass):
// the overlay ortho camera is (0,W,H,0) -> y-up, origin bottom-left. SVG is
// authored y-down, so each SVG widget group is flipped via scale.y = -1; its
// position is then the widget's TOP edge in y-up pixel space.

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // ---- palette -----------------------------------------------------------
    constexpr int kPanel = 0x0e1b2a;     // panel fill
    constexpr int kPanelEdge = 0x1d3b57; // panel border
    constexpr int kAccent = 0x35c2ff;    // cyan accent
    constexpr int kBtn = 0x16314a;       // button idle
    constexpr int kBtnHover = 0x21496b;  // button hover
    constexpr int kBtnActive = 0x35c2ff; // button active
    constexpr int kGood = 0x47e07a;      // green LED
    constexpr int kWarn = 0xff4d4d;      // red LED
    constexpr int kDim = 0x254056;       // dim LED

    // ---- SVG -> mesh helpers ----------------------------------------------

    // Build a Group of fill + stroke meshes from parsed SVG data. Mirrors
    // examples/loaders/svg_loader.cpp. mesh->name is set to the SVG element id
    // so sub-elements (e.g. compass needles) can be found via getObjectByName.
    std::shared_ptr<Group> buildSvg(const std::vector<SVGLoader::SVGData>& svgData) {

        auto group = Group::create();

        for (const auto& data : svgData) {

            const auto& fill = data.style.fill;
            if (fill && *fill != "none") {

                auto material = MeshBasicMaterial::create();
                material->color.copy(data.path.color);
                material->opacity = data.style.fillOpacity;
                material->transparent = true;
                material->depthTest = false;
                material->depthWrite = false;
                material->side = Side::Double;

                auto geometry = ShapeGeometry::create(SVGLoader::createShapes(data));
                auto mesh = Mesh::create(geometry, material);
                mesh->name = data.style.id;
                mesh->visible = data.style.visibility;
                group->add(mesh);
            }

            const auto& stroke = data.style.stroke;
            if (stroke && *stroke != "none") {

                auto sMat = MeshBasicMaterial::create();
                sMat->color.setStyle(*stroke);
                sMat->opacity = data.style.strokeOpacity;
                sMat->transparent = true;
                sMat->depthTest = false;
                sMat->depthWrite = false;
                sMat->side = Side::Double;

                for (const auto& subPath : data.path.subPaths) {
                    auto sg = SVGLoader::pointsToStroke(subPath->getPoints(), data.style);
                    if (sg) group->add(Mesh::create(sg, sMat));
                }
            }
        }

        return group;
    }

    std::shared_ptr<Group> svgFromString(const std::string& svg) {
        SVGLoader loader;
        return buildSvg(loader.parse(svg));
    }

    std::string hex(int rgb) {
        std::ostringstream os;
        os << '#' << std::hex << std::setfill('0') << std::setw(6) << (rgb & 0xffffff);
        return os.str();
    }

    // A rounded-rect mesh whose fill material we own (so it can be recoloured
    // for hover / active states). Built from an SVG <rect rx>.
    struct RectMesh {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<MeshBasicMaterial> material;
    };

    RectMesh roundedRect(float w, float h, float r, int color, float opacity = 1.f) {

        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg">)"
            << R"(<rect x="0" y="0" width=")" << w << R"(" height=")" << h
            << R"(" rx=")" << r << R"(" ry=")" << r << R"(" fill="#ffffff"/></svg>)";

        SVGLoader loader;
        auto data = loader.parse(svg.str());

        auto mat = MeshBasicMaterial::create();
        mat->color = Color(color);
        mat->opacity = opacity;
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;

        auto geo = ShapeGeometry::create(SVGLoader::createShapes(data.front()));
        return {Mesh::create(geo, mat), mat};
    }

    // ---- responsive layout -------------------------------------------------
    // Anchor a widget group in y-up pixel space: position = anchor*viewport +
    // pixel offset (the group's TOP edge, since SVG groups are y-flipped).
    // Recorded so it can be re-applied on resize.
    struct Layout {
        std::vector<std::function<void(float, float)>> fns;

        void add(const std::shared_ptr<Object3D>& g, float ax, float ay, float ox, float oy, float z) {
            fns.emplace_back([=](float W, float H) {
                g->position.set(ax * W + ox, ay * H + oy, z);
            });
        }
        // For widgets that need custom resize logic (e.g. stretching to width).
        void addRaw(std::function<void(float, float)> fn) {
            fns.emplace_back(std::move(fn));
        }
        void apply(float W, float H) {
            for (auto& f : fns) f(W, H);
        }
    };

    // ---- text readout ------------------------------------------------------
    // A screen-space TextSprite that only re-rasterises when its text changes.
    struct Readout {
        std::shared_ptr<TextSprite> sprite;
        std::string last;

        void set(const std::string& s) {
            if (s == last) return;
            last = s;
            sprite->setText(s);
        }
    };

    std::shared_ptr<TextSprite> makeText(const Font& font, const std::string& text, int color,
                                         float px, float ax, float ay, float ox, float oy,
                                         TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Left,
                                         TextSprite::VerticalAlignment v = TextSprite::VerticalAlignment::Center) {
        auto t = TextSprite::create(font, px);
        t->setColor(Color(color));
        t->setText(text);
        t->setHorizontalAlignment(h);
        t->setVerticalAlignment(v);
        t->screenSpace = true;
        t->screenAnchor.set(ax, ay);
        t->position.set(ox, oy, 0.f);
        return t;
    }

    // ---- interactive button ------------------------------------------------
    // Visual = SVG rounded-rect mesh (y-flipped group) + a centred TextSprite
    // label. Interaction = a transparent screen-space Sprite hit-target driven
    // by SpriteInteractor. The pixel-AABB matches SpriteInteractor's convention
    // (center=(0,1): covers [px, px+w] x [py-h, py] in y-up space).
    struct Button {
        std::shared_ptr<Group> group;
        std::shared_ptr<MeshBasicMaterial> bg;
        std::shared_ptr<Sprite> hit;
        std::shared_ptr<TextSprite> label;
        float ax, ay, ox, oy, w, h;
        bool active = false, hovered = false, pressed = false;
        std::function<void()> onClick;

        void refresh() const {
            int c = active ? kBtnActive : (pressed ? kBtnHover : (hovered ? kBtnHover : kBtn));
            bg->color = Color(c);
            // label flips to dark when the button is lit, for contrast
            label->setColor(Color(active ? 0x06121d : 0xcfe8ff));
        }

        // y-up pixel AABB hit test (cursor already flipped to y-up).
        [[nodiscard]] bool contains(float mx, float myUp, float W, float H) const {
            const float px = ax * W + ox;
            const float py = ay * H + oy;
            return mx >= px && mx <= px + w && myUp >= py - h && myUp <= py;
        }
    };

    std::shared_ptr<Button> makeButton(Scene& ui, Layout& layout, const Font& font,
                                       const std::string& text, float w, float h,
                                       float ax, float ay, float ox, float oy,
                                       std::function<void()> onClick) {
        auto b = std::make_shared<Button>();
        b->ax = ax;
        b->ay = ay;
        b->ox = ox;
        b->oy = oy;
        b->w = w;
        b->h = h;
        b->onClick = std::move(onClick);

        auto rect = roundedRect(w, h, 8.f, kBtn);
        b->bg = rect.material;
        b->group = Group::create();
        b->group->add(rect.mesh);
        b->group->scale.y = -1.f;// flip SVG (y-down) to upright in y-up cam
        ui.add(b->group);
        layout.add(b->group, ax, ay, ox, oy, 0.4f);

        // centred label, auto-shrunk to fit the button width (screen-space
        // sprite scale.x == rendered pixel width, same units as the button)
        const float labelPx = h * 0.42f;
        b->label = makeText(font, text, 0xcfe8ff, labelPx, ax, ay, ox + w / 2.f, oy - h / 2.f,
                            TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center);
        const float maxLabelW = w - 18.f;// 9px padding each side
        if (b->label->scale.x > maxLabelW) {
            b->label->setWorldScale(labelPx * maxLabelW / static_cast<float>(b->label->scale.x));
        }
        ui.add(b->label);

        // transparent hit target
        auto hitMat = SpriteMaterial::create();
        hitMat->visible = false;// not drawn, but still hit-tested
        b->hit = Sprite::create(hitMat);
        b->hit->screenSpace = true;
        b->hit->screenAnchor.set(ax, ay);
        b->hit->position.set(ox, oy, 0.f);
        b->hit->scale.set(w, h, 1.f);
        b->hit->center.set(0.f, 1.f);
        Button* raw = b.get();
        b->hit->onMouseDown = [raw](int) { raw->pressed = true; };
        b->hit->onMouseUp = [raw](int) {
            raw->pressed = false;
            if (raw->onClick) raw->onClick();
        };
        ui.add(b->hit);

        return b;
    }

    // ---- circular gauge ----------------------------------------------------
    // SVG dial (face + tick ring) with a needle child rotated by value. The
    // dial is authored in a 2R x 2R box centred at (R,R).
    struct Gauge {
        std::shared_ptr<Group> group;// y-flipped widget group
        std::shared_ptr<Object3D> needle;
        float R;
        float a0, a1;// sweep start / end angle (radians, 0 = up)

        void setValue(float t) const {// t in [0,1]
            t = std::clamp(t, 0.f, 1.f);
            // sweeps clockwise: min (t=0) at lower-left, max (t=1) at lower-right
            needle->rotation.z = a0 + (a1 - a0) * t;
        }
    };

    Gauge makeGauge(float R, int ringColor) {

        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><g>)";
        // face
        svg << R"(<circle cx=")" << R << R"(" cy=")" << R << R"(" r=")" << R
            << R"(" fill=")" << hex(kPanel) << R"(" stroke=")" << hex(kPanelEdge)
            << R"(" stroke-width="2"/>)";
        // tick ring: 27 ticks over a 270-degree sweep (-135..+135 from top)
        const int ticks = 27;
        for (int i = 0; i < ticks; ++i) {
            const float t = static_cast<float>(i) / (ticks - 1);
            const float ang = (-135.f + 270.f * t) * math::PI / 180.f;// 0 = up
            const float dx = std::sin(ang), dy = -std::cos(ang);
            const bool major = (i % 6 == 0);
            const float r0 = R - (major ? 16.f : 9.f);
            const float r1 = R - 4.f;
            svg << R"(<line x1=")" << R + dx * r0 << R"(" y1=")" << R + dy * r0
                << R"(" x2=")" << R + dx * r1 << R"(" y2=")" << R + dy * r1
                << R"(" stroke=")" << hex(major ? ringColor : kPanelEdge)
                << R"(" stroke-width=")" << (major ? 3.f : 1.5f) << R"("/>)";
        }
        svg << "</g></svg>";

        Gauge g;
        g.R = R;
        g.a0 = -135.f * math::PI / 180.f;
        g.a1 = 135.f * math::PI / 180.f;
        g.group = svgFromString(svg.str());
        g.group->scale.y = -1.f;

        // needle (points up at value 0), pivot at dial centre (R,R)
        std::ostringstream nsvg;
        nsvg << R"(<svg xmlns="http://www.w3.org/2000/svg"><polygon points=")"
             << "0," << -(R - 12.f) << " " << -4.f << ",0 " << 4.f << ",0"
             << R"(" fill=")" << hex(kAccent) << R"("/></svg>)";
        g.needle = svgFromString(nsvg.str());
        g.needle->position.set(R, R, 0.1f);
        g.group->add(g.needle);

        // hub
        std::ostringstream hub;
        hub << R"(<svg xmlns="http://www.w3.org/2000/svg"><circle cx=")" << R << R"(" cy=")" << R
            << R"(" r="6" fill=")" << hex(kAccent) << R"("/></svg>)";
        g.group->add(svgFromString(hub.str()));

        return g;
    }

    // ---- status LED --------------------------------------------------------
    struct Led {
        std::shared_ptr<Group> group;
        std::shared_ptr<MeshBasicMaterial> mat;
        void set(int color) const { mat->color = Color(color); }
    };

    Led makeLed(float r) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><circle cx=")" << r << R"(" cy=")" << r
            << R"(" r=")" << r << R"(" fill="#ffffff"/></svg>)";
        SVGLoader loader;
        auto data = loader.parse(svg.str());
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(kDim);
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;
        auto mesh = Mesh::create(ShapeGeometry::create(SVGLoader::createShapes(data.front())), mat);
        auto g = Group::create();
        g->add(mesh);
        g->scale.y = -1.f;
        return {g, mat};
    }

    // ---- radar PPI scope ---------------------------------------------------
    // Fully runtime-generated SVG: face + range rings + crosshairs + bearing
    // ticks (static), with a rotating sweep wedge and a heading marker as
    // origin-pivoted children translated to the dial centre (R,R) — same
    // trick as the gauge needle, so rotation.z spins about the centre.
    struct Radar {
        std::shared_ptr<Group> group;
        std::shared_ptr<Object3D> sweep;
        std::shared_ptr<Object3D> headingMark;
        float R;
        void setSweep(float ang) const { sweep->rotation.z = ang; }
        // heading 0 = north (top); increasing heading sweeps clockwise (east).
        // If it turns the wrong way on screen, negate here.
        void setHeading(float ang) const { headingMark->rotation.z = ang; }
    };

    Radar makeRadar(float R) {

        const float D = 2 * R;
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><g>)";
        // face + accent rim
        svg << R"(<circle cx=")" << R << R"(" cy=")" << R << R"(" r=")" << R
            << R"(" fill=")" << hex(kPanel) << R"(" stroke=")" << hex(kAccent) << R"(" stroke-width="2"/>)";
        // range rings
        for (float f : {0.66f, 0.33f})
            svg << R"(<circle cx=")" << R << R"(" cy=")" << R << R"(" r=")" << R * f
                << R"(" fill="none" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1"/>)";
        // crosshairs
        svg << R"(<line x1=")" << R << R"(" y1="0" x2=")" << R << R"(" y2=")" << D
            << R"(" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1"/>)";
        svg << R"(<line x1="0" y1=")" << R << R"(" x2=")" << D << R"(" y2=")" << R
            << R"(" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1"/>)";
        // bearing ticks every 30 degrees
        for (int i = 0; i < 12; ++i) {
            const float a = static_cast<float>(i) * 30.f * math::PI / 180.f;
            const float dx = std::sin(a), dy = -std::cos(a);
            svg << R"(<line x1=")" << R + dx * (R - 8.f) << R"(" y1=")" << R + dy * (R - 8.f)
                << R"(" x2=")" << R + dx * (R - 2.f) << R"(" y2=")" << R + dy * (R - 2.f)
                << R"(" stroke=")" << hex(kAccent) << R"(" stroke-width="1.5"/>)";
        }
        svg << "</g></svg>";

        Radar rd;
        rd.R = R;
        rd.group = svgFromString(svg.str());
        rd.group->scale.y = -1.f;

        // sweep wedge (apex at origin, pointing up), translucent accent
        {
            const float rr = R - 3.f, beam = 18.f * math::PI / 180.f;
            std::ostringstream s;
            s << R"(<svg xmlns="http://www.w3.org/2000/svg"><polygon points="0,0 0,)" << -rr
              << " " << std::sin(beam) * rr << "," << -std::cos(beam) * rr
              << R"(" fill=")" << hex(kAccent) << R"(" fill-opacity="0.30"/></svg>)";
            rd.sweep = svgFromString(s.str());
            rd.sweep->position.set(R, R, 0.1f);
            rd.group->add(rd.sweep);
        }
        // heading marker: triangle on the ring pointing inward (origin-pivoted)
        {
            std::ostringstream s;
            s << R"(<svg xmlns="http://www.w3.org/2000/svg"><polygon points="0,)" << -(R - 14.f)
              << " " << -7.f << "," << -R << " " << 7.f << "," << -R
              << R"(" fill=")" << hex(kWarn) << R"("/></svg>)";
            rd.headingMark = svgFromString(s.str());
            rd.headingMark->position.set(R, R, 0.15f);
            rd.group->add(rd.headingMark);
        }
        // centre hub
        {
            std::ostringstream s;
            s << R"(<svg xmlns="http://www.w3.org/2000/svg"><circle cx=")" << R << R"(" cy=")" << R
              << R"(" r="4" fill=")" << hex(kAccent) << R"("/></svg>)";
            rd.group->add(svgFromString(s.str()));
        }
        return rd;
    }

}// namespace

int main() {

    Canvas canvas(Canvas::Parameters().title("threepp - SVG Bridge HMI").size(1366, 820).antialiasing(4));
    auto renderer = createRenderer(canvas);

    const float dpi = monitor::contentScale().first;

    // ===== 3D world (background) ===========================================
    auto scene = Scene::create();
    int dayBg = 0x0a0f16, nightBg = 0x05080c;
    scene->background = Color(nightBg);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 2.6f, 6.5f);

    OrbitControls controls(*camera, canvas);
    controls.enableDamping = true;
    controls.target.set(0, 0.6f, 0);

    auto hemi = HemisphereLight::create(0x88bbff, 0x10202c);
    hemi->intensity = 0.8f;
    scene->add(hemi);
    auto dir = DirectionalLight::create(0xffffff, 1.4f);
    dir->position.set(4, 6, 3);
    scene->add(dir);

    auto hero = Group::create();
    scene->add(hero);

    auto knotMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(kAccent)).roughness(0.35f).metalness(0.6f));
    auto knot = Mesh::create(TorusKnotGeometry::create(1.0f, 0.32f, 160, 24), knotMat);
    knot->position.y = 0.7f;
    hero->add(knot);

    // bow indicator — a bright cone pointing forward (-Z = heading 0) so the
    // craft's heading is visible as it yaws.
    auto bowMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x9fe0ff)).roughness(0.4f).metalness(0.3f));
    auto bow = Mesh::create(ConeGeometry::create(0.32f, 1.1f, 20), bowMat);
    bow->position.set(0, 0.7f, -1.5f);
    bow->rotation.x = -math::PI / 2.f;
    hero->add(bow);

    // sea grid — large so the craft can roam; scrolled each frame to convey travel
    auto grid = GridHelper::create(160, 40, Color(kPanelEdge), Color(0x12222f));
    scene->add(grid);

    // ===== UI overlay scene + ortho camera =================================
    auto ui = Scene::create();
    auto size = canvas.size();
    auto uiCam = OrthographicCamera::create(0, size.width(), size.height(), 0, 0.1f, 100);
    uiCam->position.z = 10;

    Layout layout;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    // --- top status bar (authored 1px wide, x-scaled to full window width) --
    {
        std::ostringstream bar;
        bar << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="1" height="44" fill=")"
            << hex(kPanel) << R"(" fill-opacity="0.85"/>)"
            << R"(<rect x="0" y="42" width="1" height="2" fill=")" << hex(kAccent) << R"("/></svg>)";
        auto g = svgFromString(bar.str());
        ui->add(g);
        layout.addRaw([g](float W, float H) {
            g->position.set(0.f, H, 0.1f);
            g->scale.set(W, -1.f, 1.f);// stretch across the full width on resize
        });
    }
    auto titleTxt = makeText(font, "THREEPP  //  SVG BRIDGE HMI", kAccent, 20 * dpi, 0.f, 1.f, 18.f, -22.f,
                             TextSprite::HorizontalAlignment::Left, TextSprite::VerticalAlignment::Center);
    ui->add(titleTxt);

    Readout clockTxt{makeText(font, "", 0xcfe8ff, 18 * dpi, 0.5f, 1.f, 0.f, -22.f,
                              TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center)};
    ui->add(clockTxt.sprite);
    Readout fpsTxt{makeText(font, "", 0x8fb6cf, 16 * dpi, 1.f, 1.f, -18.f, -22.f,
                            TextSprite::HorizontalAlignment::Right, TextSprite::VerticalAlignment::Center)};
    ui->add(fpsTxt.sprite);

    // --- radar PPI scope (runtime-generated SVG) ----------------------------
    auto scope = makeRadar(110.f);
    ui->add(scope.group);
    layout.add(scope.group, 1.f, 1.f, -240.f, -66.f, 0.3f);

    Readout hdgTxt{makeText(font, "", 0xffffff, 26 * dpi, 1.f, 1.f, -130.f, -250.f,
                            TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center)};
    ui->add(hdgTxt.sprite);
    ui->add(makeText(font, "HEADING", 0x8fb6cf, 13 * dpi, 1.f, 1.f, -130.f, -284.f,
                     TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center));

    // --- gauges (speed + rpm) ----------------------------------------------
    auto spdGauge = makeGauge(70.f, kGood);
    ui->add(spdGauge.group);
    layout.add(spdGauge.group, 1.f, 0.f, -320.f, 175.f, 0.3f);
    Readout spdTxt{makeText(font, "", 0xffffff, 24 * dpi, 1.f, 0.f, -250.f, 95.f,
                            TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center)};
    ui->add(spdTxt.sprite);
    ui->add(makeText(font, "SPEED kn", 0x8fb6cf, 12 * dpi, 1.f, 0.f, -250.f, 70.f,
                     TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center));

    auto rpmGauge = makeGauge(70.f, kWarn);
    ui->add(rpmGauge.group);
    layout.add(rpmGauge.group, 1.f, 0.f, -160.f, 175.f, 0.3f);
    Readout rpmTxt{makeText(font, "", 0xffffff, 24 * dpi, 1.f, 0.f, -90.f, 95.f,
                            TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center)};
    ui->add(rpmTxt.sprite);
    ui->add(makeText(font, "RPM x100", 0x8fb6cf, 12 * dpi, 1.f, 0.f, -90.f, 70.f,
                     TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center));

    // --- left status panel + LEDs -------------------------------------------
    {
        auto panel = roundedRect(220, 150, 12, kPanel, 0.85f);
        auto g = Group::create();
        g->add(panel.mesh);
        g->scale.y = -1.f;
        ui->add(g);
        layout.add(g, 0.f, 1.f, 18.f, -64.f, 0.1f);
    }
    ui->add(makeText(font, "SYSTEMS", kAccent, 15 * dpi, 0.f, 1.f, 34.f, -86.f,
                     TextSprite::HorizontalAlignment::Left, TextSprite::VerticalAlignment::Center));

    struct StatusRow {
        Led led;
        const char* name;
    };
    std::vector<StatusRow> rows{
            {makeLed(8.f), "AUTOPILOT"},
            {makeLed(8.f), "ENGINE"},
            {makeLed(8.f), "RADAR"},
            {makeLed(8.f), "OVERSPEED"}};
    for (size_t i = 0; i < rows.size(); ++i) {
        const float y = -112.f - static_cast<float>(i) * 24.f;
        ui->add(rows[i].led.group);
        layout.add(rows[i].led.group, 0.f, 1.f, 36.f, y + 8.f, 0.2f);
        ui->add(makeText(font, rows[i].name, 0xcfe8ff, 14 * dpi, 0.f, 1.f, 58.f, y,
                         TextSprite::HorizontalAlignment::Left, TextSprite::VerticalAlignment::Center));
    }

    // ===== vessel / simulation state ========================================
    // This is the single source of truth. The HMI (gauges, radar, readouts)
    // is purely a view of it, and the 3D scene moves according to it.
    bool autoHelm = true;
    bool wireframe = false;
    bool night = true;
    bool radar = true;
    float targetSpeed = 12.f;// knots (commanded)
    float speed = 0.f;       // knots (actual, eased toward target)
    float heading = 0.f;     // degrees true
    float scrollX = 0.f, scrollZ = 0.f;// accumulated travel → sea-scroll offset

    // ===== buttons ==========================================================
    std::vector<std::shared_ptr<Button>> buttons;
    const float bw = 132.f, bh = 40.f, gap = 12.f;
    float bx = 18.f;
    auto addBtn = [&](const std::string& label, std::function<void()> fn) {
        auto b = makeButton(*ui, layout, font, label, bw, bh, 0.f, 0.f, bx, bh + 18.f, std::move(fn));
        buttons.push_back(b);
        bx += bw + gap;
    };

    addBtn("AUTO-HELM", [&] { autoHelm = !autoHelm; });
    addBtn("WIREFRAME", [&] {
        wireframe = !wireframe;
        knotMat->wireframe = wireframe;
    });
    addBtn("NIGHT/DAY", [&] {
        night = !night;
        scene->background = Color(night ? nightBg : dayBg);
        hemi->intensity = night ? 0.8f : 1.4f;
        dir->intensity = night ? 1.4f : 2.2f;
    });
    addBtn("RADAR", [&] {
        radar = !radar;
        grid->visible = radar;
    });

    // throttle - / +
    auto minus = makeButton(*ui, layout, font, "-", 40.f, bh, 1.f, 0.f, -250.f, 30.f, [&] {
        targetSpeed = std::clamp(targetSpeed - 2.f, 0.f, 30.f);
    });
    auto plus = makeButton(*ui, layout, font, "+", 40.f, bh, 1.f, 0.f, -60.f, 30.f, [&] {
        targetSpeed = std::clamp(targetSpeed + 2.f, 0.f, 30.f);
    });
    buttons.push_back(minus);
    buttons.push_back(plus);
    ui->add(makeText(font, "THROTTLE", 0x8fb6cf, 13 * dpi, 1.f, 0.f, -155.f, 50.f,
                     TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center));

    // sync initial button "active" states
    buttons[0]->active = autoHelm;
    buttons[3]->active = radar;

    // control hint (bottom-centre)
    ui->add(makeText(font, "A / D  STEER       W / S  THROTTLE", 0x5f7a90, 12 * dpi, 0.5f, 0.f, 0.f, 22.f,
                     TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center));

    // ===== interaction ======================================================
    SpriteInteractor interactor(canvas, *ui);// dispatches clicks to hit-sprites

    // hover: mirror SpriteInteractor's y-up pixel-AABB test; also gate orbit
    Vector2 cursor{-1e9f, -1e9f};
    MouseMoveListener moveL([&](const Vector2& p) { cursor = p; });
    canvas.addMouseListener(moveL);

    // ===== resize ===========================================================
    auto relayout = [&](WindowSize s) {
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(s);
        uiCam->right = W;
        uiCam->top = H;
        uiCam->updateProjectionMatrix();
        layout.apply(W, H);
    };
    canvas.onWindowResize([&](WindowSize s) { relayout(s); });
    relayout(size);

    // ===== render loop ======================================================
    Clock clock;
    float fps = 0.f, fpsTimer = 0.f;
    int frames = 0;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        const auto s = canvas.size();
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());

        // --- vessel kinematics: the world drives the HMI, not the reverse ---
        // throttle: W/S ramps commanded speed (the +/- buttons also set it)
        if (canvas.isKeyDown(Key::W)) targetSpeed = std::clamp(targetSpeed + dt * 8.f, 0.f, 30.f);
        if (canvas.isKeyDown(Key::S)) targetSpeed = std::clamp(targetSpeed - dt * 8.f, 0.f, 30.f);
        speed += (targetSpeed - speed) * std::min(1.f, dt * 1.5f);
        const float rpm = speed * 0.28f;// x100 engine rpm, derived from speed

        // steering: A/D turn the wheel; AUTO-HELM gently weaves on its own
        float steer = 0.f;
        if (canvas.isKeyDown(Key::A)) steer -= 1.f;
        if (canvas.isKeyDown(Key::D)) steer += 1.f;
        if (steer != 0.f) autoHelm = false;// taking the wheel disengages autopilot
        const float turnRate = autoHelm ? 12.f * std::sin(clock.getElapsedTime() * 0.25f)
                                        : steer * 45.f;// deg/s
        heading = std::fmod(heading + turnRate * dt + 360.f, 360.f);
        const float hRad = heading * math::PI / 180.f;

        // ownship stays at the origin; integrate travel into a sea-scroll offset
        scrollX += std::sin(hRad) * speed * 0.2f * dt;
        scrollZ += -std::cos(hRad) * speed * 0.2f * dt;

        // --- hover / orbit gating ---
        const float myUp = H - cursor.y;
        bool anyHover = false;
        Button* top = nullptr;
        for (auto& b : buttons) {
            b->hovered = b->contains(cursor.x, myUp, W, H);
            if (b->hovered) {
                anyHover = true;
                top = b.get();// last wins == visually on top
            }
        }
        for (auto& b : buttons) b->hovered = (b.get() == top);
        controls.enabled = !anyHover;
        for (auto& b : buttons) b->refresh();

        // --- drive instruments (all read straight from vessel state) ---
        scope.setSweep(clock.getElapsedTime() * 1.6f);
        scope.setHeading(hRad);
        spdGauge.setValue(speed / 30.f);
        rpmGauge.setValue(rpm / 10.f);

        rows[0].led.set(autoHelm ? kGood : kDim);
        rows[1].led.set(speed > 0.2f ? kGood : kDim);
        rows[2].led.set(radar ? kGood : kDim);
        scope.sweep->visible = radar;
        const bool overspeed = speed > 24.f;
        rows[3].led.set(overspeed ? (std::fmod(clock.getElapsedTime(), 0.6f) < 0.3f ? kWarn : kDim) : kDim);

        // --- readouts ---
        {
            std::ostringstream os;
            os << std::setfill('0') << std::setw(3) << static_cast<int>(heading) << " DEG";
            hdgTxt.set(os.str());
        }
        {
            std::ostringstream os;
            os << std::fixed << std::setprecision(1) << speed;
            spdTxt.set(os.str());
        }
        {
            std::ostringstream os;
            os << std::fixed << std::setprecision(1) << rpm;
            rpmTxt.set(os.str());
        }
        {
            const int sec = static_cast<int>(clock.getElapsedTime());
            std::ostringstream os;
            os << "T+ " << std::setfill('0') << std::setw(2) << (sec / 60) << ":" << std::setw(2) << (sec % 60);
            clockTxt.set(os.str());
        }
        frames++;
        fpsTimer += dt;
        if (fpsTimer >= 0.5f) {
            fps = static_cast<float>(frames) / fpsTimer;
            frames = 0;
            fpsTimer = 0;
            std::ostringstream os;
            os << static_cast<int>(fps + 0.5f) << " FPS";
            fpsTxt.set(os.str());
        }

        buttons[0]->active = autoHelm;
        buttons[1]->active = wireframe;
        buttons[2]->active = !night;
        buttons[3]->active = radar;

        // --- 3D motion: yaw the craft to heading, scroll the sea beneath it ---
        hero->rotation.y = -hRad;
        hero->position.y = std::sin(clock.getElapsedTime() * 1.2f) * 0.05f;// sea bob
        constexpr float cell = 4.f;// grid cell; wrap scroll so the sea looks infinite
        grid->position.x = -(scrollX - std::floor(scrollX / cell) * cell);
        grid->position.z = -(scrollZ - std::floor(scrollZ / cell) * cell);
        controls.update();

        // ===== two-pass render =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);// 3D world

        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);// SVG overlay + screen-space sprites
    });
}
