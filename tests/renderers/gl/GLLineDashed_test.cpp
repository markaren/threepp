// LineDashedMaterial must reach the shader.
//
// The GL material-uniform refresh dispatches on the exact string Material::type().
// "LineDashedMaterial" had no case, so a dashed line was drawn with the ShaderLib
// defaults — white, opaque, dashSize 1, totalSize 2, scale 1 — and every field the
// user set was silently discarded. Nothing failed and nothing warned; the line just
// came out the wrong colour with the wrong dash pattern.
//
// Each case below fails on that code and passes once refreshUniformsDash runs:
// the colour case pins `diffuse`, the coverage case pins `dashSize`/`totalSize`,
// and the scale case pins `scale` (which changes the dash COUNT at fixed coverage,
// so it cannot be satisfied by the other two).

#include "gl_test_helpers.hpp"

#include "threepp/materials/LineDashedMaterial.hpp"
#include "threepp/objects/Line.hpp"

namespace {

    // A horizontal line across the middle of the frame, long enough that several
    // dash periods fit inside the 64 px viewport.
    std::shared_ptr<Line> makeDashedLine(const Color& color, float dashSize, float gapSize, float scale) {
        std::vector<float> pts{-2.f, 0.f, 0.f, 2.f, 0.f, 0.f};
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(pts, 3));

        auto mat = LineDashedMaterial::create();
        mat->color = color;
        mat->dashSize = dashSize;
        mat->gapSize = gapSize;
        mat->scale = scale;

        auto line = Line::create(geo, mat);
        line->computeLineDistances();
        return line;
    }

    std::vector<unsigned char> renderLine(const std::shared_ptr<Line>& line) {
        auto scene = Scene::create();
        scene->add(line);
        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 5);
        camera.lookAt({0, 0, 0});
        return renderWithGL(*scene, camera, Color(0x000000));
    }

    // The row the line lands on, and the lit runs along it. A run is a maximal
    // stretch of lit pixels, i.e. one dash.
    struct RowProfile {
        int lit = 0;
        int runs = 0;
    };

    RowProfile profileLine(const std::vector<unsigned char>& px) {
        RowProfile best;
        // The line is at y = 0, but find the brightest row rather than assuming
        // which pixel row that is after projection and viewport rounding.
        for (int y = 0; y < RT_HEIGHT; ++y) {
            RowProfile p;
            bool inRun = false;
            for (int x = 0; x < RT_WIDTH; ++x) {
                const int i = (y * RT_WIDTH + x) * 3;
                const bool on = px[i] > 30 || px[i + 1] > 30 || px[i + 2] > 30;
                if (on) {
                    ++p.lit;
                    if (!inRun) ++p.runs;
                }
                inRun = on;
            }
            if (p.lit > best.lit) best = p;
        }
        return best;
    }

}// namespace

TEST_CASE("LineDashedMaterial colour reaches the shader") {

    const auto px = renderLine(makeDashedLine(Color(0xff0000), 0.5f, 0.5f, 1.f));

    // Find the brightest lit pixel and check its hue. Left inert, the dashed
    // shader draws the ShaderLib default diffuse, which is white.
    int bestSum = -1, br = 0, bg = 0, bb = 0;
    for (size_t i = 0; i < px.size(); i += 3) {
        const int sum = px[i] + px[i + 1] + px[i + 2];
        if (sum > bestSum) {
            bestSum = sum;
            br = px[i];
            bg = px[i + 1];
            bb = px[i + 2];
        }
    }

    INFO("brightest lit pixel rgb = " << br << "," << bg << "," << bb);
    REQUIRE(br > 150);      // the line is drawn at all
    CHECK(bg < br / 3);     // and it is red,
    CHECK(bb < br / 3);     // not the default white
}

TEST_CASE("LineDashedMaterial dash and gap sizes reach the shader") {

    // Same total period, opposite duty cycles: 90% lit versus 10% lit.
    const auto mostlyDash = profileLine(renderLine(makeDashedLine(Color(0xffffff), 0.9f, 0.1f, 1.f)));
    const auto mostlyGap = profileLine(renderLine(makeDashedLine(Color(0xffffff), 0.1f, 0.9f, 1.f)));

    INFO("lit pixels: dash-heavy " << mostlyDash.lit << ", gap-heavy " << mostlyGap.lit);
    REQUIRE(mostlyDash.lit > 0);
    REQUIRE(mostlyGap.lit > 0);
    // Inert, both render the identical default pattern and this ratio is 1.
    CHECK(mostlyDash.lit > mostlyGap.lit * 3);
}

TEST_CASE("LineDashedMaterial scale reaches the shader") {

    // Identical duty cycle, so coverage is unchanged and only the dash COUNT can
    // distinguish them: scale multiplies the line distance, packing more periods
    // into the same line.
    const auto coarse = profileLine(renderLine(makeDashedLine(Color(0xffffff), 0.5f, 0.5f, 1.f)));
    const auto fine = profileLine(renderLine(makeDashedLine(Color(0xffffff), 0.5f, 0.5f, 4.f)));

    INFO("dash runs: scale 1 -> " << coarse.runs << ", scale 4 -> " << fine.runs);
    REQUIRE(coarse.runs > 0);
    CHECK(fine.runs > coarse.runs);
}
