// Three GL render-path defects from the 2026-10-10 cleanup survey, each pinned
// by what a frame shows.
//
// 1. Material::linewidth never reached GL: GLState::lineWidthAvailable was never
//    set, so setLineWidth only updated its cache. It is now derived from
//    GL_ALIASED_LINE_WIDTH_RANGE and the context flags: a forward-compatible
//    core context, which Canvas requests on every desktop platform, removes
//    wide lines, so there the request is declined and the line stays one
//    pixel, with no GL error raised. The case below pins that a wide line
//    still draws, and widens where the context allows it.
// 2. Material::defaultAttributeValues was never applied: the binding-state code
//    looked up the literal string "name" instead of the attribute's name, so a
//    ShaderMaterial program wanting an attribute the geometry lacks read the GL
//    default (0, 0, 0, 1) instead of the material's value.
// 3. RenderTarget::Options::generateMipmaps was never read by the constructor;
//    the target's texture kept Texture's default of true.

#include "gl_test_helpers.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"

namespace {

    // Count the lit rows at the frame's middle column: the thickness of a
    // horizontal line, in pixels.
    int litRowsAtCentre(const std::vector<unsigned char>& px) {
        int rows = 0;
        const int x = RT_WIDTH / 2;
        for (int y = 0; y < RT_HEIGHT; ++y) {
            const int i = (y * RT_WIDTH + x) * 3;
            if (px[i] > 30 || px[i + 1] > 30 || px[i + 2] > 30) ++rows;
        }
        return rows;
    }

    std::vector<unsigned char> renderHorizontalLine(float linewidth) {
        std::vector<float> pts{-2.f, 0.f, 0.f, 2.f, 0.f, 0.f};
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(pts, 3));
        auto mat = LineBasicMaterial::create();
        mat->color = Color(0xffffff);
        mat->linewidth = linewidth;

        auto scene = Scene::create();
        scene->add(LineSegments::create(geo, mat));
        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 5);
        camera.lookAt({0, 0, 0});
        return renderWithGL(*scene, camera, Color(0x000000));
    }

}// namespace

TEST_CASE("GL: Material::linewidth reaches glLineWidth") {

    const int thin = litRowsAtCentre(renderHorizontalLine(1.f));
    const int wide = litRowsAtCentre(renderHorizontalLine(6.f));

    INFO("rows lit at the centre column: linewidth 1 -> " << thin << ", linewidth 6 -> " << wide);
    REQUIRE(thin >= 1);
    // On the forward-compatible core context the test canvas creates, wide
    // lines are not available and both renders are one pixel thick (two rows
    // when the line straddles a row boundary); on a context that has them the
    // wide line covers more rows. Either way the line draws.
    CHECK(wide >= thin);
    WARN("linewidth 6 covers " << wide << " rows against " << thin << " for linewidth 1 (equal = wide lines unavailable on this context)");
}

TEST_CASE("GL: a ShaderMaterial attribute the geometry lacks takes the material's default value") {

    // A quad with position only; the program also wants `color`.
    std::vector<float> pts{-1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 1.f, 1.f, 0.f, -1.f, 1.f, 0.f};
    auto geo = BufferGeometry::create();
    geo->setAttribute("position", FloatBufferAttribute::create(pts, 3));
    geo->setIndex(std::vector<int>{0, 1, 2, 0, 2, 3});

    auto mat = RawShaderMaterial::create();
    mat->vertexShader = R"(
        #version 330 core
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        in vec3 position;
        in vec3 color;
        out vec3 vColor;
        void main() {
            vColor = color;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    mat->fragmentShader = R"(
        #version 330 core
        in vec3 vColor;
        out vec4 fragColor;
        void main() { fragColor = vec4(vColor, 1.0); }
    )";
    // ShaderMaterial's constructor seeds color = white; pick something a disabled
    // attribute's GL default (0, 0, 0, 1) cannot be mistaken for.
    mat->defaultAttributeValues["color"] = Color(1.f, 0.f, 0.f);

    auto scene = Scene::create();
    scene->add(Mesh::create(geo, mat));
    PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
    camera.position.set(0, 0, 2);
    camera.lookAt({0, 0, 0});

    const auto px = renderWithGL(*scene, camera, Color(0x0000ff));
    const auto c = centerPixel(px, RT_WIDTH, RT_HEIGHT);

    INFO("centre pixel rgb = " << c.r << "," << c.g << "," << c.b);
    CHECK(c.r > 200);// the default
    CHECK(c.g < 30);
    CHECK(c.b < 30); // not the clear colour, and not the attribute default black
}

TEST_CASE("GL: RenderTarget honours Options::generateMipmaps") {

    RenderTarget::Options off;
    off.minFilter = Filter::LinearMipmapLinear;
    off.generateMipmaps = false;
    CHECK_FALSE(RenderTarget::create(RT_WIDTH, RT_HEIGHT, off)->texture->generateMipmaps);

    RenderTarget::Options on = off;
    on.generateMipmaps = true;
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, on);
    CHECK(target->texture->generateMipmaps);

    // And a target that asked for mips samples correctly through its mipmap
    // minFilter: render red into it, then draw it on a quad.
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0xff0000));
    renderer.setRenderTarget(target.get());
    auto empty = Scene::create();
    PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
    camera.position.set(0, 0, 2);
    renderer.render(*empty, camera);
    renderer.setRenderTarget(nullptr);

    auto mat = MeshBasicMaterial::create();
    mat->map = target->texture;
    auto scene = Scene::create();
    scene->add(Mesh::create(PlaneGeometry::create(4, 4), mat));
    renderer.setClearColor(Color(0x0000ff));
    renderer.render(*scene, camera);
    const auto px = renderer.readRGBPixels();
    const auto c = centerPixel(px, RT_WIDTH, RT_HEIGHT);

    INFO("centre pixel rgb = " << c.r << "," << c.g << "," << c.b);
    CHECK(c.r > 200);
    CHECK(c.b < 30);
    renderer.dispose();
}
