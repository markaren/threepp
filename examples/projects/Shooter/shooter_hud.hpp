// ============================================================================
//  Shooter — SVG HUD toolkit (condensed from examples/loaders/svg_ui.cpp)
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: shooter_constants.hpp (uiScale)
// ============================================================================

std::shared_ptr<Group> buildSvg(const std::vector<SVGLoader::SVGData>& svgData) {
    auto group = Group::create();
    for (const auto& data : svgData) {
        const auto& fill = data.style.fill;
        if (fill && *fill != "none") {
            auto m = MeshBasicMaterial::create();
            m->color.copy(data.path.color);
            m->opacity = data.style.fillOpacity;
            m->transparent = true;
            m->depthTest = false;
            m->depthWrite = false;
            m->side = Side::Double;
            auto mesh = Mesh::create(ShapeGeometry::create(SVGLoader::createShapes(data)), m);
            mesh->name = data.style.id;
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

// Annular wedge (donut slice) path, polyline-sampled — no SVG arc-flag
// pitfalls. Centred on the origin, opening toward +y (screen-up in the
// y-up UI overlay), spanning ±halfDeg. rInner = 0 gives a pie slice.
std::string wedgePath(float rInner, float rOuter, float halfDeg) {
    std::ostringstream d;
    const int N = 18;
    auto ang = [&](int i) { return (90.f - halfDeg + 2.f * halfDeg * i / N) * math::DEG2RAD; };
    for (int i = 0; i <= N; ++i)
        d << (i == 0 ? "M" : "L") << rOuter * std::cos(ang(i)) << "," << rOuter * std::sin(ang(i)) << " ";
    if (rInner > 0.f)
        for (int i = N; i >= 0; --i)
            d << "L" << rInner * std::cos(ang(i)) << "," << rInner * std::sin(ang(i)) << " ";
    else
        d << "L0,0 ";
    d << "Z";
    return d.str();
}

// Owned-material rect (so we can recolour / rescale it). Built from <rect>.
struct RectMesh {
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<MeshBasicMaterial> material;
};
RectMesh rect(float w, float h, int color, float opacity = 1.f) {
    std::ostringstream svg;
    svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width=")" << w
        << R"(" height=")" << h << R"(" fill="#ffffff"/></svg>)";
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

// Rounded, border-stroked panel — the HUD framing primitive (<rect rx> + stroke).
std::shared_ptr<Group> panel(float w, float h, float rx, int fill, float fillOp,
                             int edge, float edgeW) {
    std::ostringstream svg;
    svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width=")" << w
        << R"(" height=")" << h << R"(" rx=")" << rx << R"(" fill=")" << hex(fill)
        << R"(" fill-opacity=")" << fillOp << R"(" stroke=")" << hex(edge)
        << R"(" stroke-width=")" << edgeW << R"("/></svg>)";
    return svgFromString(svg.str());
}

// Collect every MeshBasicMaterial under an svgFromString group, so HUD
// pieces built from full SVG documents stay restylable per frame.
std::vector<std::shared_ptr<MeshBasicMaterial>> svgMats(const std::shared_ptr<Group>& g) {
    std::vector<std::shared_ptr<MeshBasicMaterial>> out;
    g->traverseType<Mesh>([&](Mesh& m) {
        if (auto mat = std::dynamic_pointer_cast<MeshBasicMaterial>(m.material())) out.push_back(mat);
    });
    return out;
}

std::shared_ptr<TextSprite> makeText(const Font& font, const std::string& text, int color,
                                     float px, float ax, float ay, float ox, float oy,
                                     TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Left,
                                     TextSprite::VerticalAlignment v = TextSprite::VerticalAlignment::Center) {
    // screen-space sprites bypass the scene graph, so scale px size AND
    // pixel offsets here (uiScale, not a parent group transform)
    auto t = TextSprite::create(font, px * uiScale);
    t->setColor(Color(color));
    t->setText(text);
    t->setHorizontalAlignment(h);
    t->setVerticalAlignment(v);
    t->screenSpace = true;
    t->screenAnchor.set(ax, ay);
    t->position.set(ox * uiScale, oy * uiScale, 0.f);
    return t;
}

// A text readout that only re-rasterises when its content changes.
struct Readout {
    std::shared_ptr<TextSprite> sprite;
    std::string last;
    void set(const std::string& s) {
        if (s == last) return;
        last = s;
        sprite->setText(s);
    }
};

// Responsive anchoring: position = anchor*viewport + pixel offset, re-applied
// on resize. Mirrors the svg_ui layout helper.
struct Layout {
    std::vector<std::function<void(float, float)>> fns;
    void add(const std::shared_ptr<Object3D>& g, float ax, float ay, float ox, float oy, float z) {
        fns.emplace_back([=](float W, float H) { g->position.set(ax * W + ox * uiScale, ay * H + oy * uiScale, z); });
    }
    void addRaw(std::function<void(float, float)> fn) { fns.emplace_back(std::move(fn)); }
    void apply(float W, float H) {
        for (auto& f : fns) f(W, H);
    }
};
