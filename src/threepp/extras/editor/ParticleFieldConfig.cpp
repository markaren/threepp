
#include "threepp/extras/editor/ParticleFieldConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    ParticleFieldConfig::Proxy proxyFrom(std::string_view text,
                                         ParticleFieldConfig::Proxy fallback) {

        if (text == "none") return ParticleFieldConfig::Proxy::None;
        if (text == "sphere") return ParticleFieldConfig::Proxy::Sphere;
        if (text == "flake") return ParticleFieldConfig::Proxy::Flake;
        return fallback;
    }

    const char* proxyToken(ParticleFieldConfig::Proxy proxy) {

        switch (proxy) {
            case ParticleFieldConfig::Proxy::None: return "none";
            case ParticleFieldConfig::Proxy::Sphere: return "sphere";
            case ParticleFieldConfig::Proxy::Flake: return "flake";
        }
        return "flake";
    }

    float clamp(float value, float low, float high) {

        return std::max(low, std::min(high, value));
    }

    int clampi(int value, int low, int high) {

        return std::max(low, std::min(high, value));
    }

    // The smallest radius FireEffect is willing to hand a field. Under
    // WSemantic::Radius the proxy scales by w / uniformRadius, so a zero here
    // is a division by zero on every particle.
    constexpr float kMinRadius = 1e-4f;
    // ParticleFieldPass's height-bake bounds (kBakeResMin / kBakeResMax).
    constexpr int kMinSurfaceRes = 16;
    constexpr int kMaxSurfaceRes = 1024;

}// namespace


std::string ParticleFieldConfig::encode() const {

    std::string out;
    // Every key is emitted on every write, whatever the enable flags say, so
    // toggling a representation off and back on does not quietly reset the
    // parameters it hides — the rule VehicleConfig and JointConfig follow.
    out += "capacity=";
    out += std::to_string(capacity);
    out += ";radius=";
    out += number(radius);
    out += ";proxy=";
    out += proxyToken(proxy);
    out += ";densityres=";
    out += std::to_string(densityResolution);

    out += ";velx=";
    out += number(velocity.x);
    out += ";vely=";
    out += number(velocity.y);
    out += ";velz=";
    out += number(velocity.z);
    out += ";speedspread=";
    out += number(speedSpread);
    out += ";accx=";
    out += number(accel.x);
    out += ";accy=";
    out += number(accel.y);
    out += ";accz=";
    out += number(accel.z);
    out += ";windx=";
    out += number(wind.x);
    out += ";windy=";
    out += number(wind.y);
    out += ";windz=";
    out += number(wind.z);
    out += ";extx=";
    out += number(spawnHalfExtent.x);
    out += ";exty=";
    out += number(spawnHalfExtent.y);
    out += ";extz=";
    out += number(spawnHalfExtent.z);
    out += ";driftamp=";
    out += number(driftAmplitude);
    out += ";driftfreq=";
    out += number(driftFrequency);
    out += ";driftgrowth=";
    out += number(driftGrowth);
    out += ";driftscale=";
    out += number(driftScale);
    out += ";life=";
    out += number(lifetime);
    out += ";lifejitter=";
    out += number(lifetimeJitter);
    out += ";duty=";
    out += number(dutyCycle);
    out += ";size=";
    out += number(size);
    out += ";sizejitter=";
    out += number(sizeJitter);
    out += ";follow=";
    out += follow ? "1" : "0";
    out += ";followsnap=";
    out += number(followSnap);
    out += ";seed=";
    out += std::to_string(seed);

    out += ";surface=";
    out += surface ? "1" : "0";
    out += ";surfacerest=";
    out += number(surfaceRest);
    out += ";surfacerestjitter=";
    out += number(surfaceRestJitter);
    out += ";surfacefade=";
    out += number(surfaceFade);
    out += ";surfacesplash=";
    out += number(surfaceSplash);
    out += ";surfacesplashgrow=";
    out += number(surfaceSplashGrow);
    out += ";surfacebias=";
    out += number(surfaceBias);
    out += ";surfaceres=";
    out += std::to_string(surfaceResolution);

    out += ";billboard=";
    out += billboard ? "1" : "0";
    out += ";bbsize=";
    out += number(billboardSize);
    out += ";hotr=";
    out += number(colorHot.r);
    out += ";hotg=";
    out += number(colorHot.g);
    out += ";hotb=";
    out += number(colorHot.b);
    out += ";coolr=";
    out += number(colorCool.r);
    out += ";coolg=";
    out += number(colorCool.g);
    out += ";coolb=";
    out += number(colorCool.b);
    out += ";bbintensity=";
    out += number(billboardIntensity);
    out += ";bbsoftness=";
    out += number(billboardSoftness);
    out += ";bbfade=";
    out += number(billboardFade);
    out += ";bbjitter=";
    out += number(billboardJitter);
    out += ";bbtaper=";
    out += number(billboardTaper);
    out += ";bbstretch=";
    out += number(billboardStretch);
    out += ";bbstretchmax=";
    out += number(billboardStretchMax);
    out += ";bbnearfade=";
    out += number(billboardNearFade);
    out += ";bbglow=";
    out += number(billboardGlow);
    out += ";bbglowthr=";
    out += number(billboardGlowThreshold);

    out += ";mesh=";
    out += mesh ? "1" : "0";
    out += ";meshlodfar=";
    out += number(meshLodFar);
    out += ";meshlodfade=";
    out += number(meshLodFade);
    out += ";meshnearcull=";
    out += number(meshNearCull);

    out += ";density=";
    out += density ? "1" : "0";
    out += ";sigma=";
    out += number(sigma);
    out += ";albr=";
    out += number(albedo.r);
    out += ";albg=";
    out += number(albedo.g);
    out += ";albb=";
    out += number(albedo.b);
    out += ";aniso=";
    out += number(anisotropy);
    out += ";dextx=";
    out += number(densityHalfExtent.x);
    out += ";dexty=";
    out += number(densityHalfExtent.y);
    out += ";dextz=";
    out += number(densityHalfExtent.z);
    out += ";emissive=";
    out += number(emissiveIntensity);
    out += ";tempbottom=";
    out += number(tempBottom);
    out += ";temptop=";
    out += number(tempTop);
    out += ";tempfalloff=";
    out += number(tempFalloff);

    return out;
}

std::optional<ParticleFieldConfig> ParticleFieldConfig::decode(const std::string& text) {

    ParticleFieldConfig config;

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "capacity") {
            config.capacity = toInt(value, config.capacity);
        } else if (key == "radius") {
            config.radius = toFloat(value, config.radius);
        } else if (key == "proxy") {
            config.proxy = proxyFrom(value, config.proxy);
        } else if (key == "densityres") {
            config.densityResolution = toInt(value, config.densityResolution);
        } else if (key == "velx") {
            config.velocity.x = toFloat(value, config.velocity.x);
        } else if (key == "vely") {
            config.velocity.y = toFloat(value, config.velocity.y);
        } else if (key == "velz") {
            config.velocity.z = toFloat(value, config.velocity.z);
        } else if (key == "speedspread") {
            config.speedSpread = toFloat(value, config.speedSpread);
        } else if (key == "accx") {
            config.accel.x = toFloat(value, config.accel.x);
        } else if (key == "accy") {
            config.accel.y = toFloat(value, config.accel.y);
        } else if (key == "accz") {
            config.accel.z = toFloat(value, config.accel.z);
        } else if (key == "windx") {
            config.wind.x = toFloat(value, config.wind.x);
        } else if (key == "windy") {
            config.wind.y = toFloat(value, config.wind.y);
        } else if (key == "windz") {
            config.wind.z = toFloat(value, config.wind.z);
        } else if (key == "extx") {
            config.spawnHalfExtent.x = toFloat(value, config.spawnHalfExtent.x);
        } else if (key == "exty") {
            config.spawnHalfExtent.y = toFloat(value, config.spawnHalfExtent.y);
        } else if (key == "extz") {
            config.spawnHalfExtent.z = toFloat(value, config.spawnHalfExtent.z);
        } else if (key == "driftamp") {
            config.driftAmplitude = toFloat(value, config.driftAmplitude);
        } else if (key == "driftfreq") {
            config.driftFrequency = toFloat(value, config.driftFrequency);
        } else if (key == "driftgrowth") {
            config.driftGrowth = toFloat(value, config.driftGrowth);
        } else if (key == "driftscale") {
            config.driftScale = toFloat(value, config.driftScale);
        } else if (key == "life") {
            config.lifetime = toFloat(value, config.lifetime);
        } else if (key == "lifejitter") {
            config.lifetimeJitter = toFloat(value, config.lifetimeJitter);
        } else if (key == "duty") {
            config.dutyCycle = toFloat(value, config.dutyCycle);
        } else if (key == "size") {
            config.size = toFloat(value, config.size);
        } else if (key == "sizejitter") {
            config.sizeJitter = toFloat(value, config.sizeJitter);
        } else if (key == "follow") {
            config.follow = toBool(value, config.follow);
        } else if (key == "followsnap") {
            config.followSnap = toFloat(value, config.followSnap);
        } else if (key == "seed") {
            config.seed = toInt(value, config.seed);
        } else if (key == "surface") {
            config.surface = toBool(value, config.surface);
        } else if (key == "surfacerest") {
            config.surfaceRest = toFloat(value, config.surfaceRest);
        } else if (key == "surfacerestjitter") {
            config.surfaceRestJitter = toFloat(value, config.surfaceRestJitter);
        } else if (key == "surfacefade") {
            config.surfaceFade = toFloat(value, config.surfaceFade);
        } else if (key == "surfacesplash") {
            config.surfaceSplash = toFloat(value, config.surfaceSplash);
        } else if (key == "surfacesplashgrow") {
            config.surfaceSplashGrow = toFloat(value, config.surfaceSplashGrow);
        } else if (key == "surfacebias") {
            config.surfaceBias = toFloat(value, config.surfaceBias);
        } else if (key == "surfaceres") {
            config.surfaceResolution = toInt(value, config.surfaceResolution);
        } else if (key == "billboard") {
            config.billboard = toBool(value, config.billboard);
        } else if (key == "bbsize") {
            config.billboardSize = toFloat(value, config.billboardSize);
        } else if (key == "hotr") {
            config.colorHot.r = toFloat(value, config.colorHot.r);
        } else if (key == "hotg") {
            config.colorHot.g = toFloat(value, config.colorHot.g);
        } else if (key == "hotb") {
            config.colorHot.b = toFloat(value, config.colorHot.b);
        } else if (key == "coolr") {
            config.colorCool.r = toFloat(value, config.colorCool.r);
        } else if (key == "coolg") {
            config.colorCool.g = toFloat(value, config.colorCool.g);
        } else if (key == "coolb") {
            config.colorCool.b = toFloat(value, config.colorCool.b);
        } else if (key == "bbintensity") {
            config.billboardIntensity = toFloat(value, config.billboardIntensity);
        } else if (key == "bbsoftness") {
            config.billboardSoftness = toFloat(value, config.billboardSoftness);
        } else if (key == "bbfade") {
            config.billboardFade = toFloat(value, config.billboardFade);
        } else if (key == "bbjitter") {
            config.billboardJitter = toFloat(value, config.billboardJitter);
        } else if (key == "bbtaper") {
            config.billboardTaper = toFloat(value, config.billboardTaper);
        } else if (key == "bbstretch") {
            config.billboardStretch = toFloat(value, config.billboardStretch);
        } else if (key == "bbstretchmax") {
            config.billboardStretchMax = toFloat(value, config.billboardStretchMax);
        } else if (key == "bbnearfade") {
            config.billboardNearFade = toFloat(value, config.billboardNearFade);
        } else if (key == "bbglow") {
            config.billboardGlow = toFloat(value, config.billboardGlow);
        } else if (key == "bbglowthr") {
            config.billboardGlowThreshold = toFloat(value, config.billboardGlowThreshold);
        } else if (key == "mesh") {
            config.mesh = toBool(value, config.mesh);
        } else if (key == "meshlodfar") {
            config.meshLodFar = toFloat(value, config.meshLodFar);
        } else if (key == "meshlodfade") {
            config.meshLodFade = toFloat(value, config.meshLodFade);
        } else if (key == "meshnearcull") {
            config.meshNearCull = toFloat(value, config.meshNearCull);
        } else if (key == "density") {
            config.density = toBool(value, config.density);
        } else if (key == "sigma") {
            config.sigma = toFloat(value, config.sigma);
        } else if (key == "albr") {
            config.albedo.r = toFloat(value, config.albedo.r);
        } else if (key == "albg") {
            config.albedo.g = toFloat(value, config.albedo.g);
        } else if (key == "albb") {
            config.albedo.b = toFloat(value, config.albedo.b);
        } else if (key == "aniso") {
            config.anisotropy = toFloat(value, config.anisotropy);
        } else if (key == "dextx") {
            config.densityHalfExtent.x = toFloat(value, config.densityHalfExtent.x);
        } else if (key == "dexty") {
            config.densityHalfExtent.y = toFloat(value, config.densityHalfExtent.y);
        } else if (key == "dextz") {
            config.densityHalfExtent.z = toFloat(value, config.densityHalfExtent.z);
        } else if (key == "emissive") {
            config.emissiveIntensity = toFloat(value, config.emissiveIntensity);
        } else if (key == "tempbottom") {
            config.tempBottom = toFloat(value, config.tempBottom);
        } else if (key == "temptop") {
            config.tempTop = toFloat(value, config.tempTop);
        } else if (key == "tempfalloff") {
            config.tempFalloff = toFloat(value, config.tempFalloff);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    // Every bound below is one the ParticleField setters already enforce — by
    // clamp where they clamp and by throw where they throw. A hand-edited
    // document is the reason: the overlay builds a field from whatever this
    // returns, and an exception there is an editor that cannot open a scene.
    config.capacity = std::max(1, config.capacity);
    config.radius = std::max(config.radius, kMinRadius);
    config.densityResolution = clampi(config.densityResolution, 8, 256);
    config.surfaceResolution =
            clampi(config.surfaceResolution, kMinSurfaceRes, kMaxSurfaceRes);

    config.lifetime = std::max(config.lifetime, 1e-3f);
    config.lifetimeJitter = clamp(config.lifetimeJitter, 0.f, 1.f);
    config.dutyCycle = clamp(config.dutyCycle, 1e-3f, 1.f);
    config.driftGrowth = clamp(config.driftGrowth, 0.f, 1.f);
    config.size = std::max(config.size, 0.f);
    config.sizeJitter = clamp(config.sizeJitter, 0.f, 1.f);
    config.surfaceRestJitter = clamp(config.surfaceRestJitter, 0.f, 1.f);

    config.billboardIntensity = std::max(config.billboardIntensity, 0.f);
    config.billboardSize = std::max(config.billboardSize, kMinRadius);

    config.sigma = std::max(config.sigma, kMinRadius);
    config.densityHalfExtent.set(std::max(config.densityHalfExtent.x, kMinRadius),
                                 std::max(config.densityHalfExtent.y, kMinRadius),
                                 std::max(config.densityHalfExtent.z, kMinRadius));

    return config;
}

std::optional<ParticleFieldConfig> ParticleFieldConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;
    return decode(std::any_cast<const std::string&>(it->second));
}

void ParticleFieldConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void ParticleFieldConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool ParticleFieldConfig::isParticleField(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

std::string ParticleFieldConfig::structuralKey() const {

    std::string out;
    out += std::to_string(capacity);
    out += '/';
    out += number(radius);
    out += '/';
    out += proxyToken(proxy);
    out += '/';
    out += std::to_string(densityResolution);
    return out;
}

const char* ParticleFieldConfig::label(Proxy proxy) {

    switch (proxy) {
        case Proxy::None: return "None";
        case Proxy::Sphere: return "Sphere";
        case Proxy::Flake: return "Flake";
    }
    return "Flake";
}


// ── Presets ─────────────────────────────────────────────────────────────────
//
// Four points in one parameter space, not four shaders — the claim
// ParticleField.hpp makes about archetypes, stated here as four initialisers.
// The spawn CENTRE is missing from all of them on purpose: the authored node's
// transform is the emitter frame, so "where" is the gizmo's business.

ParticleFieldConfig ParticleFieldConfig::snow() {

    ParticleFieldConfig c;
    // vulkan_snow.cpp's tuned snowfall, in its --rest form: a flake that lands
    // and rests is the whole reason the surface block exists, and a preset that
    // shipped with it off would look like the feature is missing.
    c.capacity = 300000;
    c.radius = 0.016f;
    c.proxy = Proxy::Flake;
    c.densityResolution = 96;

    // Terminal velocity, so ZERO acceleration.
    c.velocity.set(0.f, -1.25f, 0.f);
    c.speedSpread = 0.10f;
    c.wind.set(0.52f, 0.f, 0.21f);
    // driftScale couples the phase to position, so neighbouring flakes lean
    // together and the field reads as gusts rather than 300k independent jitters.
    c.driftAmplitude = 0.34f;
    c.driftFrequency = 0.15f;
    c.driftScale = 11.f;
    c.spawnHalfExtent.set(24.f, 0.30f, 24.f);
    // The lifetime has to CONTAIN fall, rest and fade, or the ground carpet
    // blinks: ~10 s of fall + 2.2 x 1.6 s of rest + 1 s of fade.
    c.lifetime = 17.f;
    c.dutyCycle = 0.94f;
    // Bigger than the falling-only 1.6 cm: at 7 m a 1.6 cm flake is two pixels,
    // and two white pixels on a white surface cannot show that anything settled.
    c.size = 0.028f;
    c.sizeJitter = 0.50f;
    c.seed = 20260812;

    c.surface = true;
    c.surfaceResolution = 512;
    c.surfaceRest = 2.2f;
    c.surfaceRestJitter = 0.6f;
    c.surfaceFade = 1.0f;
    c.surfaceBias = 0.012f;

    c.mesh = true;
    c.meshLodFar = 22.f;
    c.meshLodFade = 6.f;
    // A cap, not a hole: inside the band the linear ramp and 1/d cancel, so
    // every near flake projects to the same apparent size.
    c.meshNearCull = 3.0f;

    // The far representation: at 3 px a flake is a highlight on the overcast
    // sky rather than a lit solid.
    c.billboard = true;
    c.colorHot = Color(0.88f, 0.91f, 0.97f);
    c.colorCool = Color(0.80f, 0.84f, 0.92f);
    c.billboardIntensity = 0.30f;
    c.billboardSoftness = 0.62f;
    c.billboardFade = 0.f;// a flake does not burn out
    c.billboardTaper = 0.f;
    c.billboardJitter = 0.40f;

    // Total optical mass is N * sigma, and resting flakes concentrate into a
    // one-voxel layer — hence a sigma two orders below FireEffect's smoke.
    c.density = true;
    c.sigma = 0.026f;
    c.albedo = Color(0.90f, 0.93f, 0.98f);
    c.anisotropy = 0.35f;
    c.densityHalfExtent.set(26.f, 7.5f, 26.f);

    return c;
}

ParticleFieldConfig ParticleFieldConfig::rain() {

    ParticleFieldConfig c;
    c.capacity = 90000;
    c.radius = 0.013f;
    // A drop is not a shape; no amount of shading a 1.3 cm solid makes a streak.
    c.proxy = Proxy::None;
    c.densityResolution = 96;

    c.velocity.set(0.f, -9.0f, 0.f);
    c.speedSpread = 0.35f;
    c.wind.set(1.30f, 0.f, 0.45f);
    c.driftAmplitude = 0.03f;// rain barely wanders
    c.driftFrequency = 0.9f;
    c.driftScale = 6.f;
    c.spawnHalfExtent.set(22.f, 0.30f, 22.f);
    c.lifetime = 1.9f;
    c.dutyCycle = 0.92f;
    c.size = 0.013f;
    c.sizeJitter = 0.30f;
    c.seed = 20260813;

    c.mesh = false;
    c.billboard = true;
    // Water does not glow; it catches a little of the sky and smears it, so the
    // honest values are an order of magnitude below an ember's.
    c.colorHot = Color(0.72f, 0.79f, 0.90f);
    c.colorCool = Color(0.60f, 0.67f, 0.78f);
    c.billboardIntensity = 0.070f;
    c.billboardSize = 0.30f;
    c.billboardSoftness = 0.95f;// soft along its whole length, no hot core
    c.billboardFade = 0.f;      // a drop does not fade; it lands
    c.billboardTaper = 0.f;
    c.billboardJitter = 0.55f;
    c.billboardStretch = 0.024f;// ~22 cm of travel at 9 m/s
    c.billboardStretchMax = 30.f;
    c.billboardNearFade = 1.20f;

    c.density = true;
    c.sigma = 0.014f;
    // A rain curtain is DARK; snow's haze albedo would make a downpour a whiteout.
    c.albedo = Color(0.42f, 0.46f, 0.52f);
    c.anisotropy = 0.35f;
    c.densityHalfExtent.set(26.f, 7.5f, 26.f);

    return c;
}

ParticleFieldConfig ParticleFieldConfig::embers() {

    ParticleFieldConfig c;
    // FireEffect's spark emitter at its default flame size (1.2 m tall, 0.25 m
    // base). Its own campfire runs 150 slots because the flame volume and the
    // point light carry most of that look; an authored ember field is the whole
    // effect, so it gets a source worth looking at on its own.
    c.capacity = 2000;
    c.radius = 0.011f;
    c.proxy = Proxy::None;

    c.velocity.set(0.f, 1.45f, 0.f);// buoyant
    c.speedSpread = 0.50f;
    c.accel.set(0.f, 0.45f, 0.f);
    c.spawnHalfExtent.set(0.1375f, 0.072f, 0.1375f);// thrown off the fuel bed
    // driftGrowth 1: the base of the column is steady and the tips wander,
    // which is what turns a cone of sparks into a plume.
    c.driftAmplitude = 0.13f;
    c.driftFrequency = 0.55f;
    c.driftGrowth = 1.f;
    c.driftScale = 0.7f;
    c.lifetime = 2.2f;
    // Wide, because sparks that all die at the same height give the plume a
    // flat top.
    c.lifetimeJitter = 0.65f;
    // Duty < 1: the spark count BREATHES instead of sitting at a constant N.
    c.dutyCycle = 0.80f;
    c.size = 0.011f;
    c.sizeJitter = 0.60f;
    c.seed = 20260814;

    c.billboard = true;
    // The struct defaults ARE the blackbody arc a cooling ember walks, which is
    // what FireEffect derives from its 2100 K / 1150 K pair.
    c.billboardIntensity = 3.4f;
    c.billboardSoftness = 0.34f;// a spark, not a glow
    c.billboardFade = 1.75f;    // holds bright, then goes out
    c.billboardJitter = 0.55f;
    c.billboardTaper = 0.65f;// a cooling ember gets SMALLER
    c.billboardStretch = 0.012f;
    c.billboardStretchMax = 10.f;
    c.billboardNearFade = 0.25f;
    // Field billboards composite past the scene's bloom pyramid, so a spark has
    // no halo but its own falloff without this.
    c.billboardGlow = 8.0f;

    return c;
}

ParticleFieldConfig ParticleFieldConfig::motes() {

    ParticleFieldConfig c;
    // ParticleField.hpp's fourth archetype: velocity ~0, a slow wide wobble,
    // and a spawn box that IS the room — the box never sweeps, so for once the
    // "thin slab" rule does not apply.
    c.capacity = 20000;
    c.radius = 0.004f;
    c.proxy = Proxy::None;

    c.velocity.set(0.f, -0.02f, 0.f);
    c.speedSpread = 0.01f;
    c.spawnHalfExtent.set(6.f, 3.f, 6.f);
    c.driftAmplitude = 0.15f;
    c.driftFrequency = 0.05f;
    c.driftScale = 1.5f;
    c.lifetime = 20.f;
    c.lifetimeJitter = 0.50f;
    c.size = 0.004f;
    c.sizeJitter = 0.70f;
    c.seed = 20260815;

    c.billboard = true;
    c.colorHot = Color(0.95f, 0.95f, 1.00f);
    c.colorCool = Color(0.80f, 0.84f, 0.92f);
    c.billboardIntensity = 0.35f;
    c.billboardSoftness = 0.80f;
    c.billboardFade = 0.f;// a mote drifts out of the light, it does not burn out
    c.billboardTaper = 0.f;
    c.billboardJitter = 0.60f;

    return c;
}
