// Morph targets: negative influences, and an influence array longer than the
// geometry's morph list.
//
// Two defects in GLMorphTargets::update, both from the same place — r129 tests a
// slot for TRUTHINESS (`influences[i][1]`, and later `value`), while the port
// wrote `> 0`:
//
//   - a NEGATIVE influence was dropped, so a morph asked to extrapolate AWAY
//     from its target rendered exactly as if the influence were zero. With
//     morphTargetsRelative false the shader computes
//     base * (1 - sum) + sum(target * influence), so influence -1 is a
//     well-defined reflection, 2*base - target, and is the reason to write a
//     negative one at all;
//   - the per-geometry influence cache was COPIED out of the map and never grew,
//     and the morph attribute list was indexed with the OBJECT's influence index
//     without a bounds check. The cache is keyed on the geometry, so two objects
//     sharing one geometry with different influence counts — or one object whose
//     influence array grows — walked off the end of both.
//
// The quad below sits at x = 0 and its single morph target moves it to x = +1.2,
// so the influence maps directly to a screen position the test can measure.

#include "gl_test_helpers.hpp"

namespace {

    constexpr float kBaseX = 0.f;
    constexpr float kTargetX = 1.2f;

    std::vector<float> quadAt(float cx) {
        return {cx - 0.35f, -0.35f, 0.f, cx + 0.35f, -0.35f, 0.f, cx + 0.35f, 0.35f, 0.f,
                cx - 0.35f, -0.35f, 0.f, cx + 0.35f, 0.35f, 0.f, cx - 0.35f, 0.35f, 0.f};
    }

    std::shared_ptr<BufferGeometry> morphingQuad() {
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(quadAt(kBaseX), 3));
        geo->morphTargetsRelative = false;

        auto* targets = geo->getOrCreateMorphAttribute("position");
        targets->push_back(FloatBufferAttribute::create(quadAt(kTargetX), 3));
        return geo;
    }

    // Mean x of the lit pixels, in pixels from the left edge. The quad is the
    // only thing drawn, so this is its centre.
    double quadCentreX(const std::vector<unsigned char>& px) {
        return avgXPosition(px, RT_WIDTH, RT_HEIGHT);
    }

    double renderAtInfluence(float influence, size_t influenceCount = 1) {
        auto scene = Scene::create();

        auto mat = MeshBasicMaterial::create();
        mat->color = Color(0xffffff);
        mat->morphTargets = true;

        auto mesh = Mesh::create(morphingQuad(), mat);
        mesh->morphTargetInfluences() = std::vector<float>(influenceCount, 0.f);
        mesh->morphTargetInfluences()[0] = influence;
        scene->add(mesh);

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 4.f);
        camera.lookAt({0, 0, 0});

        return quadCentreX(renderWithGL(*scene, camera, Color(0x000000)));
    }

}// namespace

TEST_CASE("a positive morph influence moves the geometry toward its target") {

    const double at0 = renderAtInfluence(0.f);
    const double at1 = renderAtInfluence(1.f);

    INFO("influence 0 -> centre x " << at0 << ", influence 1 -> " << at1);
    CHECK(at1 > at0 + 5.0);// the target is to the RIGHT of the base
}

TEST_CASE("a negative morph influence extrapolates away from the target") {

    const double at0 = renderAtInfluence(0.f);
    const double atMinus1 = renderAtInfluence(-1.f);

    // base*(1 - (-1)) + target*(-1) = 2*base - target, i.e. as far to the LEFT of
    // the base as the target is to its right. Dropped, this reads identical to
    // influence 0.
    INFO("influence 0 -> centre x " << at0 << ", influence -1 -> " << atMinus1);
    CHECK(atMinus1 < at0 - 5.0);
}

TEST_CASE("more influences than the geometry has morph targets is not fatal") {

    // One morph target on the geometry, four influences on the object. The extra
    // three index nothing; they must be ignored rather than read off the end.
    double centre = 0;
    REQUIRE_NOTHROW(centre = renderAtInfluence(1.f, 4));

    const double at1 = renderAtInfluence(1.f, 1);
    INFO("4 influences -> centre x " << centre << ", 1 influence -> " << at1);
    CHECK(std::abs(centre - at1) < 2.0);
}
