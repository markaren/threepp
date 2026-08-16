// DownwashEffect: the entrainment gate, the parcel envelope, and determinism
// under a reproduced update sequence. The LOOK of the cloud is judged on
// captures (house rule); what a unit test can hold is the contract.

#include "threepp/extras/uav/DownwashEffect.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>

using namespace threepp;
using namespace threepp::uav;

namespace {

    // A deterministic scripted descent: hover at altitude, then sink to the
    // ground. Mirrors how the demo drives the effect: fixed dt, sim clock.
    void drive(DownwashEffect& fx, float t0, float t1, float agl, float thrust,
               const Vector3& groundPos = {0.f, 0.f, 0.f}) {
        constexpr float dt = 1.f / 60.f;
        for (float t = t0; t < t1; t += dt) {
            // setWind every frame, exactly as the demo does — a re-derive that
            // feeds its own output back (the box-climbs-4-m-per-frame bug)
            // only shows up under per-frame calls.
            fx.setWind(fx.wind());
            fx.update(t, Vector3(groundPos.x, groundPos.y + agl, groundPos.z),
                      thrust, agl);
        }
    }

}// namespace

TEST_CASE("high hover entrains nothing; landing fills the field") {
    auto fx = DownwashEffect::create();

    // 3 s at 20 m AGL: every slot retries and parks; the field costs nothing.
    drive(*fx, 0.f, 3.f, 20.f, 0.41f);
    CHECK(fx->dustiness() == 0.f);
    CHECK(fx->field()->liveCount() == 0);

    // 4 s at 0.6 m AGL under hover thrust: the brownout.
    drive(*fx, 3.f, 7.f, 0.6f, 0.41f);
    CHECK(fx->dustiness() > 0.9f);
    // submit() sets liveCount to the submitted span; occupancy is judged on
    // the live (w >= 0) parcels below.
    REQUIRE(fx->field()->liveCount() > 0);

    // Every live parcel sits inside the physical envelope: above ground
    // (clamped), below the roll-up ceiling, within the jet's reach + wobble.
    const auto& p = fx->params();
    const auto& host = fx->field()->hostPositions();
    std::size_t live = 0;
    bool envelopeOk = true;
    for (const auto& pp : host) {
        if (pp.w < 0.f) continue;
        ++live;
        const float r = std::hypot(pp.x, pp.z);
        // Ground here is y = 0 (drone at 0.6, AGL 0.6) and parcels clamp to
        // ground + 0.05.
        if (pp.y < 0.04f || pp.y > p.ringHeight * 2.5f) envelopeOk = false;
        if (r > p.maxRadius * 1.6f) envelopeOk = false;
    }
    CHECK(live > p.capacity / 4);
    CHECK(envelopeOk);

    // The density box sits on the ground (y = 0 here), not in the sky: its
    // centre must be within one box-height of the cloud it contains.
    const auto& c = fx->field()->densityRepr().center;
    CHECK(std::abs(c.y) < fx->field()->densityRepr().halfExtent.y * 2.f);
}

TEST_CASE("dust is conserved: eroded from the pad, deposited downwind") {
    auto fx = DownwashEffect::create();
    fx->setWind(Vector3(1.4f, 0.f, 0.5f));

    // A dusty low hover: quanta leave the ground for the air, exactly 1:1.
    drive(*fx, 0.f, 5.f, 0.7f, 0.42f);
    const double initial = fx->groundDustInitial();
    REQUIRE(initial > 0.);
    CHECK(fx->airborne() > 0);
    CHECK(fx->groundDustQuanta() + static_cast<double>(fx->airborne()) == initial);

    // The pad's source cell has been mined.
    const float padAfterHover = fx->groundDustAt(0.f, 0.f);
    CHECK(padAfterHover < fx->params().groundDustPerM2 *
                                  fx->params().gridCell * fx->params().gridCell);

    // Climb away and let every parcel settle: all mass returns to the GROUND,
    // but not to where it came from — the wind carried it downwind.
    drive(*fx, 5.f, 5.f + fx->params().life * 2.5f, 30.f, 0.41f);
    CHECK(fx->airborne() == 0);
    CHECK(fx->groundDustQuanta() == initial);
}

TEST_CASE("same update sequence, same bytes") {
    auto a = DownwashEffect::create();
    auto b = DownwashEffect::create();
    for (auto* fx : {a.get(), b.get()}) {
        fx->setWind(Vector3(1.2f, 0.f, 0.4f));
        drive(*fx, 0.f, 1.f, 8.f, 0.5f); // approach
        drive(*fx, 1.f, 4.f, 0.8f, 0.42f);// flare
    }
    const auto& ha = a->field()->hostPositions();
    const auto& hb = b->field()->hostPositions();
    REQUIRE(ha.size() == hb.size());
    CHECK(std::memcmp(ha.data(), hb.data(), ha.size() * sizeof(ParticlePos)) == 0);
    const auto& ca = a->field()->densityRepr().center;
    const auto& cb = b->field()->densityRepr().center;
    CHECK((ca.x == cb.x && ca.y == cb.y && ca.z == cb.z));
}
