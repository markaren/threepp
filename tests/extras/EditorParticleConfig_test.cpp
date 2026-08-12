// The whole of the particle-authoring schema: ParticleFieldConfig and
// GranularConfig. Both are just strings, so neither needs PhysX, Vulkan or a
// renderer — which is the point being pinned down here as much as the values
// are. What a saved document carries for a particle field is one userData
// entry, and everything downstream (overlay preview, play session, player)
// rebuilds from exactly that.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"

#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/scenes/Scene.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;


TEST_CASE("ParticleFieldConfig round-trips every field byte-identically") {

    // Every preset, not just one: the round trip has to hold for the values a
    // user actually gets, and the four presets are where the odd magnitudes are
    // (1900 K, 0.070, 0.1375, 300000).
    for (const auto& config : {ParticleFieldConfig{}, ParticleFieldConfig::snow(),
                               ParticleFieldConfig::rain(), ParticleFieldConfig::embers(),
                               ParticleFieldConfig::motes()}) {

        const auto text = config.encode();
        const auto decoded = ParticleFieldConfig::decode(text);
        REQUIRE(decoded.has_value());
        CHECK(*decoded == config);
        // The one that keeps saved documents diff-clean: re-encoding what was
        // decoded must produce the same bytes, or every load/save cycle churns
        // the file.
        CHECK(decoded->encode() == text);
    }
}

TEST_CASE("ParticleFieldConfig decodes an empty entry to the defaults") {

    const auto empty = ParticleFieldConfig::decode("");
    REQUIRE(empty.has_value());
    CHECK(*empty == ParticleFieldConfig{});

    // Unknown keys are ignored, known ones still land — the forwards
    // compatibility rule every flat config in the editor follows.
    const auto partial = ParticleFieldConfig::decode("life=9;futureKey=7;vely=-3;proxy=sphere");
    REQUIRE(partial.has_value());
    CHECK(partial->lifetime == Catch::Approx(9.f));
    CHECK(partial->velocity.y == Catch::Approx(-3.f));
    CHECK(partial->proxy == ParticleFieldConfig::Proxy::Sphere);
    CHECK(partial->dutyCycle == Catch::Approx(1.f));// untouched default

    // An unknown enum token falls back rather than corrupting the field.
    const auto bogus = ParticleFieldConfig::decode("proxy=hexagon");
    REQUIRE(bogus.has_value());
    CHECK(bogus->proxy == ParticleFieldConfig{}.proxy);
}

TEST_CASE("ParticleFieldConfig clamps a hand-edited document into the setters' ranges") {

    // Every bound here is one ParticleField's own setters clamp or throw on. A
    // document that got past this would take the overlay down at load.
    const auto config = ParticleFieldConfig::decode(
            "capacity=0;radius=0;densityres=4000;surfaceres=2;"
            "life=-5;lifejitter=3;duty=0;sizejitter=-1;size=-0.5;driftgrowth=9;"
            "surfacerestjitter=4;bbintensity=-2;bbsize=0;"
            "sigma=0;dextx=0;dexty=-3;dextz=0");
    REQUIRE(config.has_value());

    CHECK(config->capacity == 1);
    CHECK(config->radius > 0.f);
    CHECK(config->densityResolution == 256);
    CHECK(config->surfaceResolution == 16);

    CHECK(config->lifetime >= 1e-3f);
    CHECK(config->lifetimeJitter == Catch::Approx(1.f));
    CHECK(config->dutyCycle > 0.f);
    CHECK(config->dutyCycle <= 1.f);
    CHECK(config->sizeJitter == Catch::Approx(0.f));
    CHECK(config->size == Catch::Approx(0.f));
    CHECK(config->driftGrowth == Catch::Approx(1.f));
    CHECK(config->surfaceRestJitter == Catch::Approx(1.f));

    CHECK(config->billboardIntensity == Catch::Approx(0.f));
    CHECK(config->billboardSize > 0.f);

    CHECK(config->sigma > 0.f);
    CHECK(config->densityHalfExtent.x > 0.f);
    CHECK(config->densityHalfExtent.y > 0.f);
    CHECK(config->densityHalfExtent.z > 0.f);

    // A low density resolution clamps UP, not down — 8 is the floor.
    const auto low = ParticleFieldConfig::decode("densityres=1");
    REQUIRE(low.has_value());
    CHECK(low->densityResolution == 8);
}

TEST_CASE("ParticleFieldConfig presets enable the representations their look needs") {

    // Snow is a solid the eye resolves as a shape, so it keeps the mesh proxy —
    // plus the far billboard, the haze, and the surface landing.
    const auto snow = ParticleFieldConfig::snow();
    CHECK(snow.mesh);
    CHECK(snow.proxy == ParticleFieldConfig::Proxy::Flake);
    CHECK(snow.billboard);
    CHECK(snow.density);
    CHECK(snow.surface);
    CHECK(snow.velocity.y < 0.f);
    CHECK(snow.accel.y == Catch::Approx(0.f));// terminal velocity, not falling
    CHECK(snow.meshLodFar > 0.f);

    // A drop is a STREAK, not a shape: billboard only, stretched along its own
    // analytic velocity.
    const auto rain = ParticleFieldConfig::rain();
    CHECK_FALSE(rain.mesh);
    CHECK(rain.billboard);
    CHECK(rain.density);
    CHECK(rain.billboardStretch > 0.f);
    CHECK(rain.velocity.y < snow.velocity.y);// faster than snow, by a lot

    // Embers rise, glow, and breathe (duty < 1).
    const auto embers = ParticleFieldConfig::embers();
    CHECK(embers.billboard);
    CHECK_FALSE(embers.mesh);
    CHECK_FALSE(embers.density);
    CHECK(embers.velocity.y > 0.f);
    CHECK(embers.accel.y > 0.f);
    CHECK(embers.dutyCycle < 1.f);
    CHECK(embers.driftGrowth == Catch::Approx(1.f));
    CHECK(embers.billboardGlow > 0.f);

    // Motes hang: velocity ~0, a wide slow wobble, and a spawn box that is the
    // room rather than a thin slab.
    const auto motes = ParticleFieldConfig::motes();
    CHECK(motes.billboard);
    CHECK_FALSE(motes.mesh);
    CHECK_FALSE(motes.density);
    CHECK(std::abs(motes.velocity.y) < 0.1f);
    CHECK(motes.driftAmplitude > 0.f);
    CHECK(motes.spawnHalfExtent.y > 1.f);

    // Every preset survives the codec unchanged — a preset button and a reload
    // must land on the same config.
    for (const auto& preset : {snow, rain, embers, motes}) {
        CHECK(*ParticleFieldConfig::decode(preset.encode()) == preset);
    }
}

TEST_CASE("structuralKey moves for the four fields that force a rebuild and no others") {

    const ParticleFieldConfig base;
    const auto key = base.structuralKey();

    {
        auto c = base;
        c.capacity = 40000;
        CHECK(c.structuralKey() != key);
    }
    {
        auto c = base;
        c.radius = 0.02f;
        CHECK(c.structuralKey() != key);
    }
    {
        auto c = base;
        c.proxy = ParticleFieldConfig::Proxy::Sphere;
        CHECK(c.structuralKey() != key);
        c.proxy = ParticleFieldConfig::Proxy::None;
        CHECK(c.structuralKey() != key);
    }
    {
        auto c = base;
        c.densityResolution = 64;
        CHECK(c.structuralKey() != key);
    }

    // Everything else is a parameter push: setEmitter, or a repr struct mutated
    // in place. Rebuilding the field for any of these would mean a
    // vkDeviceWaitIdle and a cleared TAA history per slider tick.
    auto mutable_ = base;
    mutable_.velocity.set(1.f, -9.f, 2.f);
    mutable_.speedSpread = 0.4f;
    mutable_.accel.set(0.f, 0.5f, 0.f);
    mutable_.wind.set(1.f, 0.f, 1.f);
    mutable_.spawnHalfExtent.set(30.f, 1.f, 30.f);
    mutable_.driftAmplitude = 0.3f;
    mutable_.driftFrequency = 0.2f;
    mutable_.driftGrowth = 1.f;
    mutable_.driftScale = 8.f;
    mutable_.lifetime = 17.f;
    mutable_.lifetimeJitter = 0.5f;
    mutable_.dutyCycle = 0.8f;
    mutable_.size = 0.05f;
    mutable_.sizeJitter = 0.4f;
    mutable_.follow = true;
    mutable_.followSnap = 8.f;
    mutable_.seed = 7;
    mutable_.surface = true;
    mutable_.surfaceRest = 1.f;
    mutable_.surfaceRestJitter = 0.2f;
    mutable_.surfaceFade = 2.f;
    mutable_.surfaceSplash = 0.3f;
    mutable_.surfaceSplashGrow = 12.f;
    mutable_.surfaceBias = 0.01f;
    mutable_.surfaceResolution = 512;
    mutable_.billboard = true;
    mutable_.billboardSize = 0.3f;
    mutable_.colorHot = Color(0.1f, 0.2f, 0.3f);
    mutable_.colorCool = Color(0.3f, 0.2f, 0.1f);
    mutable_.billboardIntensity = 2.f;
    mutable_.billboardSoftness = 0.9f;
    mutable_.billboardFade = 1.f;
    mutable_.billboardJitter = 0.1f;
    mutable_.billboardTaper = 0.2f;
    mutable_.billboardStretch = 0.02f;
    mutable_.billboardStretchMax = 30.f;
    mutable_.billboardNearFade = 1.f;
    mutable_.billboardGlow = 4.f;
    mutable_.billboardGlowThreshold = 0.5f;
    mutable_.mesh = true;
    mutable_.meshLodFar = 20.f;
    mutable_.meshLodFade = 5.f;
    mutable_.meshNearCull = 3.f;
    mutable_.density = true;
    mutable_.sigma = 0.05f;
    mutable_.albedo = Color(0.4f, 0.5f, 0.6f);
    mutable_.anisotropy = 0.35f;
    mutable_.densityHalfExtent.set(26.f, 7.5f, 26.f);
    mutable_.emissiveIntensity = 20.f;
    mutable_.tempBottom = 1500.f;
    mutable_.tempTop = 700.f;
    mutable_.tempFalloff = 2.f;

    CHECK(mutable_ != base);// the walk above really did touch everything
    CHECK(mutable_.structuralKey() == key);

    // The presets differ structurally from each other, which is what makes
    // pressing a preset button a rebuild rather than a parameter push.
    CHECK(ParticleFieldConfig::snow().structuralKey() !=
          ParticleFieldConfig::rain().structuralKey());
}

TEST_CASE("a particle field is a Group carrying the entry, and the factory writes snow") {

    Scene scene;
    auto field = ObjectFactory::createParticleField(scene);
    scene.add(field);

    CHECK(field->name == "Particles");
    CHECK(ParticleFieldConfig::isParticleField(*field));

    const auto read = ParticleFieldConfig::read(*field);
    REQUIRE(read.has_value());
    CHECK(*read == ParticleFieldConfig::snow());
    // The node IS the emitter frame, so a snow field has to stand above what it
    // snows on.
    CHECK(field->position.y > 0.f);

    // A second one names itself apart.
    auto second = ObjectFactory::createParticleField(scene);
    CHECK(second->name == "Particles 2");

    // A plain node is not a particle field, and erase puts it back.
    auto plain = Group::create();
    CHECK_FALSE(ParticleFieldConfig::isParticleField(*plain));
    CHECK_FALSE(ParticleFieldConfig::read(*plain).has_value());

    ParticleFieldConfig::erase(*field);
    CHECK_FALSE(ParticleFieldConfig::isParticleField(*field));
}

TEST_CASE("a ParticleField object never reaches a saved document") {

    // The safety net, not the mechanism: overlays and play sessions own their
    // fields and none of them is a document node. If one ever were, the mesh
    // branch would happily write its zero-area placeholder and the loader would
    // drop the node — bytes spent to lose the object.
    Scene scene;
    ParticleField::Config config;
    config.capacity = 128;
    auto field = ParticleField::create(config);
    field->name = "leaked";
    scene.add(field);

    ObjectExporter exporter;
    const auto json = exporter.toJson(scene);

    CHECK(json.find("ParticleField") == std::string::npos);
    REQUIRE_FALSE(exporter.warnings().empty());
    bool mentioned = false;
    for (const auto& warning : exporter.warnings()) {
        if (warning.find("leaked") != std::string::npos) mentioned = true;
    }
    CHECK(mentioned);
}


// ── GranularConfig ──────────────────────────────────────────────────────────

TEST_CASE("GranularConfig round-trips, defaults on empty, ignores unknown keys") {

    GranularConfig config;
    config.spacing = 0.035f;
    config.iterations = 12;
    config.capacity = 250000;
    config.maxVelocity = 6.f;
    config.friction = 0.55f;
    config.damping = 0.1f;
    config.adhesion = 0.02f;
    config.cohesion = 0.03f;
    config.viscosity = 0.04f;
    config.gravityScale = 0.8f;
    config.emitExtentX = 0.35f;
    config.emitExtentZ = 0.15f;
    config.rate = 12000.f;
    config.emitVelocity.set(0.2f, -2.f, 0.1f);
    config.mass = 0.004f;
    config.emitFor = 3.5f;
    config.jitter = 0.4f;
    config.visual = GranularConfig::Visual::Field;
    config.color = Color(0.46f, 0.42f, 0.33f);
    config.roughness = 0.95f;

    const auto text = config.encode();
    const auto decoded = GranularConfig::decode(text);
    REQUIRE(decoded.has_value());
    CHECK(*decoded == config);
    CHECK(decoded->encode() == text);

    const auto empty = GranularConfig::decode("");
    REQUIRE(empty.has_value());
    CHECK(*empty == GranularConfig{});

    const auto partial = GranularConfig::decode("spacing=0.1;futureKey=7;visual=instanced");
    REQUIRE(partial.has_value());
    CHECK(partial->spacing == Catch::Approx(0.1f));
    CHECK(partial->visual == GranularConfig::Visual::Instanced);
    CHECK(partial->friction == Catch::Approx(0.4f));// untouched default

    const auto bogus = GranularConfig::decode("visual=raytraced");
    REQUIRE(bogus.has_value());
    CHECK(bogus->visual == GranularConfig{}.visual);
}

TEST_CASE("GranularConfig clamps a hand-edited document") {

    const auto config = GranularConfig::decode(
            "spacing=0;iterations=0;capacity=-4;maxvel=-9;rate=-100;jitter=5;"
            "emitextx=-1;emitextz=-1;mass=-2;emitfor=-3");
    REQUIRE(config.has_value());

    // spacing is a divisor in the lattice emitter, not a taste knob.
    CHECK(config->spacing > 0.f);
    CHECK(config->iterations == 1);
    CHECK(config->capacity == 1);
    CHECK(config->maxVelocity == Catch::Approx(0.f));
    CHECK(config->rate == Catch::Approx(0.f));
    CHECK(config->jitter == Catch::Approx(1.f));
    CHECK(config->emitExtentX == Catch::Approx(0.f));
    CHECK(config->emitExtentZ == Catch::Approx(0.f));
    CHECK(config->mass == Catch::Approx(0.f));
    CHECK(config->emitFor == Catch::Approx(0.f));

    const auto many = GranularConfig::decode("iterations=999");
    REQUIRE(many.has_value());
    CHECK(many->iterations == 32);
}

TEST_CASE("a granular node is a Group carrying the entry") {

    Scene scene;
    auto granular = ObjectFactory::createGranular(scene);
    scene.add(granular);

    CHECK(granular->name == "Granular");
    CHECK(GranularConfig::isGranular(*granular));

    const auto read = GranularConfig::read(*granular);
    REQUIRE(read.has_value());
    CHECK(*read == GranularConfig{});
    // The chute pours down from where it stands.
    CHECK(granular->position.y > 0.f);
    CHECK(read->emitVelocity.y < 0.f);

    auto second = ObjectFactory::createGranular(scene);
    CHECK(second->name == "Granular 2");

    auto plain = Group::create();
    CHECK_FALSE(GranularConfig::isGranular(*plain));
    CHECK_FALSE(GranularConfig::read(*plain).has_value());

    GranularConfig::erase(*granular);
    CHECK_FALSE(GranularConfig::isGranular(*granular));
}
