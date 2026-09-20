// Blending::Custom with only the colour factors set.
//
// Material::blendSrc / blendDst / blendEquation are plain values, but the three
// ALPHA variants are std::optional and default to nullopt — a material that asks
// for additive blending sets the colour factors and leaves the alpha ones alone,
// which is the ordinary case, not an exotic one.
//
// GLState::setBlending computed the fallbacks for those three into new locals
// (blendEquationAlpha_, blendSrcAlpha_, blendDstAlpha_) and then called
// glBlendEquationSeparate / glBlendFuncSeparate dereferencing the ORIGINAL
// optionals, which are empty. r129 reassigns the parameters instead of making
// copies, which is why the port lost it.
//
// Additive over an opaque backdrop is the cheapest observable: the overlap has to
// come out brighter than the backdrop alone.

#include "gl_test_helpers.hpp"

namespace {

    // Backdrop quad plus an additively-blended quad in front of it.
    std::vector<unsigned char> renderAdditive(bool setAlphaFactorsExplicitly) {
        auto scene = Scene::create();

        auto backMat = MeshBasicMaterial::create();
        backMat->color = Color(0.25f, 0.25f, 0.25f);
        auto back = Mesh::create(PlaneGeometry::create(4.f, 4.f), backMat);
        back->position.set(0, 0, -0.5f);
        scene->add(back);

        auto glowMat = MeshBasicMaterial::create();
        glowMat->color = Color(0.5f, 0.5f, 0.5f);
        glowMat->transparent = true;
        glowMat->blending = Blending::Custom;
        glowMat->blendEquation = BlendEquation::Add;
        glowMat->blendSrc = BlendFactor::SrcAlpha;
        glowMat->blendDst = BlendFactor::One;// additive
        if (setAlphaFactorsExplicitly) {
            glowMat->blendEquationAlpha = BlendEquation::Add;
            glowMat->blendSrcAlpha = BlendFactor::SrcAlpha;
            glowMat->blendDstAlpha = BlendFactor::One;
        }
        auto glow = Mesh::create(PlaneGeometry::create(4.f, 4.f), glowMat);
        scene->add(glow);

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 3.f);
        camera.lookAt({0, 0, 0});

        return renderWithGL(*scene, camera, Color(0x000000));
    }

    double backdropOnly() {
        auto scene = Scene::create();
        auto backMat = MeshBasicMaterial::create();
        backMat->color = Color(0.25f, 0.25f, 0.25f);
        scene->add(Mesh::create(PlaneGeometry::create(4.f, 4.f), backMat));

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 3.f);
        camera.lookAt({0, 0, 0});

        return avgBrightness(renderWithGL(*scene, camera, Color(0x000000)));
    }

}// namespace

TEST_CASE("custom blending works with the alpha factors left unset") {

    const double plain = backdropOnly();

    double additive = 0;
    REQUIRE_NOTHROW(additive = avgBrightness(renderAdditive(false)));

    INFO("backdrop alone " << plain << ", with an additive layer " << additive);
    CHECK(additive > plain + 10.0);
}

TEST_CASE("custom blending agrees whether or not the alpha factors are spelled out") {

    // Setting the alpha factors to exactly what the fallback should supply must
    // produce the same image. If the unset path is reading an empty optional,
    // these two disagree (or the first one throws).
    std::vector<unsigned char> implicitPx, explicitPx;
    REQUIRE_NOTHROW(implicitPx = renderAdditive(false));
    REQUIRE_NOTHROW(explicitPx = renderAdditive(true));

    REQUIRE(implicitPx.size() == explicitPx.size());

    int differing = 0;
    for (size_t i = 0; i < implicitPx.size(); ++i) {
        if (std::abs(int(implicitPx[i]) - int(explicitPx[i])) > 2) ++differing;
    }
    INFO("channels differing between the implicit and explicit spellings: " << differing);
    CHECK(differing == 0);
}
