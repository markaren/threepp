
#include "threepp/extras/editor/RenderConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/renderers/Renderer.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    const char* toneMapToken(ToneMapping value) {

        switch (value) {
            case ToneMapping::None: return "none";
            case ToneMapping::Linear: return "linear";
            case ToneMapping::Reinhard: return "reinhard";
            case ToneMapping::Cineon: return "cineon";
            case ToneMapping::ACESFilmic: return "aces";
            case ToneMapping::Custom: return "custom";
            case ToneMapping::Neutral: return "neutral";
            case ToneMapping::AgX: return "agx";
        }
        return "none";
    }

    ToneMapping toneMapFrom(std::string_view text, ToneMapping fallback) {

        if (text == "none") return ToneMapping::None;
        if (text == "linear") return ToneMapping::Linear;
        if (text == "reinhard") return ToneMapping::Reinhard;
        if (text == "cineon") return ToneMapping::Cineon;
        if (text == "aces") return ToneMapping::ACESFilmic;
        if (text == "custom") return ToneMapping::Custom;
        if (text == "neutral") return ToneMapping::Neutral;
        if (text == "agx") return ToneMapping::AgX;
        return fallback;
    }

    const char* shadowToken(ShadowMap value) {

        switch (value) {
            case ShadowMap::Basic: return "basic";
            case ShadowMap::PFC: return "pcf";
            case ShadowMap::PFCSoft: return "pcfsoft";
            case ShadowMap::VSM: return "vsm";
        }
        return "pcf";
    }

    ShadowMap shadowFrom(std::string_view text, ShadowMap fallback) {

        if (text == "basic") return ShadowMap::Basic;
        if (text == "pcf") return ShadowMap::PFC;
        if (text == "pcfsoft") return ShadowMap::PFCSoft;
        if (text == "vsm") return ShadowMap::VSM;
        return fallback;
    }

    const char* upscalerToken(RenderConfig::Upscaler value) {

        switch (value) {
            case RenderConfig::Upscaler::Taa: return "taa";
            case RenderConfig::Upscaler::Fsr: return "fsr";
            case RenderConfig::Upscaler::Dlss: return "dlss";
        }
        return "taa";
    }

    RenderConfig::Upscaler upscalerFrom(std::string_view text, RenderConfig::Upscaler fallback) {

        if (text == "taa") return RenderConfig::Upscaler::Taa;
        if (text == "fsr") return RenderConfig::Upscaler::Fsr;
        if (text == "dlss") return RenderConfig::Upscaler::Dlss;
        return fallback;
    }

    const char* envSunToken(RenderConfig::EnvSun value) {

        switch (value) {
            case RenderConfig::EnvSun::Auto: return "auto";
            case RenderConfig::EnvSun::Always: return "always";
            case RenderConfig::EnvSun::Off: return "off";
        }
        return "auto";
    }

    RenderConfig::EnvSun envSunFrom(std::string_view text, RenderConfig::EnvSun fallback) {

        if (text == "auto") return RenderConfig::EnvSun::Auto;
        if (text == "always") return RenderConfig::EnvSun::Always;
        if (text == "off") return RenderConfig::EnvSun::Off;
        return fallback;
    }

}// namespace


RenderConfig RenderConfig::capture(const Renderer& renderer) {

    RenderConfig config;
    config.toneMapping = renderer.toneMapping;
    config.exposure = renderer.toneMappingExposure;
    config.shadows = renderer.shadowMap().enabled;
    config.shadowType = renderer.shadowMap().type;

#ifdef THREEPP_WITH_VULKAN
    const auto* vk = dynamic_cast<const VulkanRenderer*>(&renderer);
    if (!vk) return config;

    config.renderScale = vk->renderScale();
    config.upscaler = vk->dlss() ? Upscaler::Dlss : (vk->fsr() ? Upscaler::Fsr : Upscaler::Taa);
    config.gbufferMsaa = static_cast<int>(vk->gbufferMsaa());
    config.textureAnisotropy = vk->textureAnisotropy();

    config.autoExposure = vk->autoExposure();
    config.autoExposureSpeed = vk->autoExposureSpeed();
    const auto [wbK, wbTint] = vk->whiteBalance();
    config.whiteBalanceK = wbK;
    config.whiteBalanceTint = wbTint;
    config.sharpen = vk->sharpenStrength();

    config.ao = vk->deferredAO();
    config.probeGI = vk->probeGI();
    config.denoise = vk->denoise();
    config.restirDI = vk->restirDIEnabled();
    config.sunAngularRadius = vk->sunAngularRadius();
    switch (vk->envSunPolicy()) {
        case VulkanRenderer::EnvSunPolicy::Auto: config.envSun = EnvSun::Auto; break;
        case VulkanRenderer::EnvSunPolicy::Always: config.envSun = EnvSun::Always; break;
        case VulkanRenderer::EnvSunPolicy::Off: config.envSun = EnvSun::Off; break;
    }
    config.fireflyClamp = vk->fireflyClamp();

    config.physicalCamera = vk->physicalCamera();
    config.physicalLightUnits = vk->physicalLightUnits();
    const auto exposure = vk->cameraExposure();
    config.aperture = exposure.aperture;
    config.shutterSeconds = exposure.shutterSeconds;
    config.iso = exposure.iso;
    config.exposureCompensation = vk->exposureCompensation();
    config.depthOfField = vk->depthOfField();
    config.focusDistance = vk->focusDistance();
    config.motionBlur = vk->motionBlur();

    config.bloom = vk->bloomIntensity();
    config.bloomThreshold = vk->bloomThreshold();
    if (const auto fog = vk->heightFog()) {
        config.heightFog = true;
        config.fogDensity = fog->density;
        config.fogBaseY = fog->baseY;
        config.fogFalloff = fog->falloff;
        config.fogNoise = fog->noiseAmount;
    }
    config.fogAnisotropy = vk->getFogAnisotropy();
    const auto [beams, beamAniso] = vk->deferredVolumetrics();
    config.beamDensity = beams;
    config.beamAnisotropy = beamAniso;
    if (const auto clouds = vk->clouds()) {
        config.clouds = true;
        config.cloudCoverage = clouds->coverage;
        config.cloudDensity = clouds->density;
        config.cloudBottomY = clouds->bottomY;
        config.cloudTopY = clouds->topY;
        config.cloudWindX = clouds->wind.x;
        config.cloudWindZ = clouds->wind.z;
        config.cloudEvolve = clouds->evolveSpeed;
    }

    config.occlusionCulling = vk->occlusionCulling();
    config.autoLod = vk->autoLod();
    config.autoLodError = vk->autoLodError();
    config.normalMapToksvig = vk->normalMapToksvig();
#endif

    return config;
}

void RenderConfig::apply(Renderer& renderer) const {

    renderer.toneMapping = toneMapping;
    renderer.toneMappingExposure = exposure;

    auto& shadowMap = renderer.shadowMap();
    if (shadowMap.enabled != shadows || shadowMap.type != shadowType) {
        shadowMap.enabled = shadows;
        shadowMap.type = shadowType;
        shadowMap.needsUpdate = true;
    }

#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(&renderer);
    if (!vk) return;

    // Every setter below is guarded against a no-op write. Some of them are
    // expensive (setRenderScale and setGbufferMsaa reallocate the whole render
    // extent behind a vkDeviceWaitIdle; an EnvSunPolicy Off toggle forces an
    // environment re-upload), and applying a document must not pay for a knob
    // that already holds the value the document asks for.
    const RenderConfig now = capture(renderer);

    if (renderScale != now.renderScale) vk->setRenderScale(renderScale);
    if (upscaler != now.upscaler) {
        // A document authored on a DLSS machine opened on one without it gets
        // the built-in TAA rather than nothing at all.
        vk->setDlss(upscaler == Upscaler::Dlss && vk->dlssAvailable());
        vk->setFsr(upscaler == Upscaler::Fsr && vk->fsrAvailable());
    }
    if (gbufferMsaa != now.gbufferMsaa) vk->setGbufferMsaa(static_cast<uint32_t>(gbufferMsaa));
    if (textureAnisotropy != now.textureAnisotropy) vk->setTextureAnisotropy(textureAnisotropy);

    if (autoExposure != now.autoExposure) vk->setAutoExposure(autoExposure);
    if (autoExposureSpeed != now.autoExposureSpeed) vk->setAutoExposureSpeed(autoExposureSpeed);
    // Rebakes a LUT on the CPU — one call for the pair, and only on a change.
    if (whiteBalanceK != now.whiteBalanceK || whiteBalanceTint != now.whiteBalanceTint) {
        vk->setWhiteBalance(whiteBalanceK, whiteBalanceTint);
    }
    if (sharpen != now.sharpen) vk->setSharpenStrength(sharpen);

    if (ao != now.ao) vk->setDeferredAO(ao);
    if (probeGI != now.probeGI) vk->setProbeGI(probeGI);
    if (denoise != now.denoise) vk->setDenoise(denoise);
    if (restirDI != now.restirDI) vk->setRestirDIEnabled(restirDI);
    if (sunAngularRadius != now.sunAngularRadius) vk->setSunAngularRadius(sunAngularRadius);
    if (envSun != now.envSun) {
        switch (envSun) {
            case EnvSun::Auto: vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Auto); break;
            case EnvSun::Always: vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Always); break;
            case EnvSun::Off: vk->setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Off); break;
        }
    }
    if (fireflyClamp != now.fireflyClamp) vk->setFireflyClamp(fireflyClamp);

    if (physicalCamera != now.physicalCamera) vk->setPhysicalCamera(physicalCamera);
    if (physicalLightUnits != now.physicalLightUnits) vk->setPhysicalLightUnits(physicalLightUnits);
    if (aperture != now.aperture || shutterSeconds != now.shutterSeconds || iso != now.iso) {
        vk->setCameraExposure(aperture, shutterSeconds, iso);
    }
    if (exposureCompensation != now.exposureCompensation) {
        vk->setExposureCompensation(exposureCompensation);
    }
    if (depthOfField != now.depthOfField) vk->setDepthOfField(depthOfField);
    if (focusDistance != now.focusDistance) vk->setFocusDistance(focusDistance);
    if (motionBlur != now.motionBlur) vk->setMotionBlur(motionBlur);

    if (bloom != now.bloom) vk->setBloomIntensity(bloom);
    if (bloomThreshold != now.bloomThreshold) vk->setBloomThreshold(bloomThreshold);
    if (heightFog != now.heightFog || fogDensity != now.fogDensity ||
        fogBaseY != now.fogBaseY || fogFalloff != now.fogFalloff ||
        fogNoise != now.fogNoise) {
        if (heightFog) {
            VulkanRenderer::HeightFogSettings fog;
            fog.density = fogDensity;
            fog.baseY = fogBaseY;
            fog.falloff = fogFalloff;
            fog.noiseAmount = fogNoise;
            vk->setHeightFog(fog);
        } else if (now.heightFog) {
            // Only clear a medium that is actually there: with no explicit fog
            // on either side the profile fields describe nothing, and calling
            // the setter would be a write for a difference nobody can see.
            vk->setHeightFog(std::nullopt);
        }
    }
    if (fogAnisotropy != now.fogAnisotropy) vk->setFogAnisotropy(fogAnisotropy);
    if (beamDensity != now.beamDensity || beamAnisotropy != now.beamAnisotropy) {
        vk->setDeferredVolumetrics(beamDensity, beamAnisotropy);
    }
    if (clouds != now.clouds || cloudCoverage != now.cloudCoverage ||
        cloudDensity != now.cloudDensity || cloudBottomY != now.cloudBottomY ||
        cloudTopY != now.cloudTopY || cloudWindX != now.cloudWindX ||
        cloudWindZ != now.cloudWindZ || cloudEvolve != now.cloudEvolve) {
        if (clouds) {
            VulkanRenderer::CloudSettings settings;
            settings.coverage = cloudCoverage;
            settings.density = cloudDensity;
            settings.bottomY = cloudBottomY;
            settings.topY = cloudTopY;
            settings.wind.set(cloudWindX, 0.f, cloudWindZ);
            settings.evolveSpeed = cloudEvolve;
            vk->setClouds(settings);
        } else if (now.clouds) {
            vk->setClouds(std::nullopt);
        }
    }

    if (occlusionCulling != now.occlusionCulling) vk->setOcclusionCulling(occlusionCulling);
    if (autoLod != now.autoLod) vk->setAutoLod(autoLod);
    if (autoLodError != now.autoLodError) vk->setAutoLodError(autoLodError);
    if (normalMapToksvig != now.normalMapToksvig) vk->setNormalMapToksvig(normalMapToksvig);
#endif
}

std::string RenderConfig::encode(const RenderConfig& base) const {

    std::string out;
    const auto put = [&out](const char* key, const std::string& value) {
        if (!out.empty()) out += ';';
        out += key;
        out += '=';
        out += value;
    };
    const auto putFloat = [&put](const char* key, float value, float other) {
        if (value != other) put(key, number(value));
    };
    const auto putBool = [&put](const char* key, bool value, bool other) {
        if (value != other) put(key, value ? "1" : "0");
    };

    if (toneMapping != base.toneMapping) put("tonemap", toneMapToken(toneMapping));
    putFloat("exposure", exposure, base.exposure);
    putBool("shadows", shadows, base.shadows);
    if (shadowType != base.shadowType) put("shadowtype", shadowToken(shadowType));

    putFloat("renderscale", renderScale, base.renderScale);
    if (upscaler != base.upscaler) put("upscaler", upscalerToken(upscaler));
    if (gbufferMsaa != base.gbufferMsaa) put("msaa", std::to_string(gbufferMsaa));
    putFloat("aniso", textureAnisotropy, base.textureAnisotropy);

    putBool("autoexposure", autoExposure, base.autoExposure);
    putFloat("autoexposurespeed", autoExposureSpeed, base.autoExposureSpeed);
    putFloat("wbk", whiteBalanceK, base.whiteBalanceK);
    putFloat("wbtint", whiteBalanceTint, base.whiteBalanceTint);
    putFloat("sharpen", sharpen, base.sharpen);

    putBool("ao", ao, base.ao);
    putBool("probegi", probeGI, base.probeGI);
    putBool("denoise", denoise, base.denoise);
    putBool("restir", restirDI, base.restirDI);
    putFloat("sunsoft", sunAngularRadius, base.sunAngularRadius);
    if (envSun != base.envSun) put("envsun", envSunToken(envSun));
    putFloat("fireflyclamp", fireflyClamp, base.fireflyClamp);

    putBool("physcam", physicalCamera, base.physicalCamera);
    putBool("physunits", physicalLightUnits, base.physicalLightUnits);
    putFloat("aperture", aperture, base.aperture);
    putFloat("shutter", shutterSeconds, base.shutterSeconds);
    putFloat("iso", iso, base.iso);
    putFloat("evcomp", exposureCompensation, base.exposureCompensation);
    putBool("dof", depthOfField, base.depthOfField);
    putFloat("focus", focusDistance, base.focusDistance);
    putFloat("motionblur", motionBlur, base.motionBlur);

    putFloat("bloom", bloom, base.bloom);
    putFloat("bloomthreshold", bloomThreshold, base.bloomThreshold);
    putBool("heightfog", heightFog, base.heightFog);
    putFloat("fogdensity", fogDensity, base.fogDensity);
    putFloat("fogbase", fogBaseY, base.fogBaseY);
    putFloat("fogfalloff", fogFalloff, base.fogFalloff);
    putFloat("fognoise", fogNoise, base.fogNoise);
    putFloat("foganiso", fogAnisotropy, base.fogAnisotropy);
    putFloat("beams", beamDensity, base.beamDensity);
    putFloat("beamaniso", beamAnisotropy, base.beamAnisotropy);
    putBool("clouds", clouds, base.clouds);
    putFloat("cloudcoverage", cloudCoverage, base.cloudCoverage);
    putFloat("clouddensity", cloudDensity, base.cloudDensity);
    putFloat("cloudbottom", cloudBottomY, base.cloudBottomY);
    putFloat("cloudtop", cloudTopY, base.cloudTopY);
    putFloat("cloudwindx", cloudWindX, base.cloudWindX);
    putFloat("cloudwindz", cloudWindZ, base.cloudWindZ);
    putFloat("cloudevolve", cloudEvolve, base.cloudEvolve);

    putBool("occlusion", occlusionCulling, base.occlusionCulling);
    putBool("autolod", autoLod, base.autoLod);
    putFloat("lodpx", autoLodError, base.autoLodError);
    putBool("toksvig", normalMapToksvig, base.normalMapToksvig);

    return out;
}

RenderConfig RenderConfig::decode(const std::string& text, const RenderConfig& base) {

    RenderConfig config = base;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {

        if (key == "tonemap") {
            config.toneMapping = toneMapFrom(value, config.toneMapping);
        } else if (key == "exposure") {
            config.exposure = toFloat(value, config.exposure);
        } else if (key == "shadows") {
            config.shadows = toBool(value, config.shadows);
        } else if (key == "shadowtype") {
            config.shadowType = shadowFrom(value, config.shadowType);

        } else if (key == "renderscale") {
            config.renderScale = toFloat(value, config.renderScale);
        } else if (key == "upscaler") {
            config.upscaler = upscalerFrom(value, config.upscaler);
        } else if (key == "msaa") {
            config.gbufferMsaa = toInt(value, config.gbufferMsaa);
        } else if (key == "aniso") {
            config.textureAnisotropy = toFloat(value, config.textureAnisotropy);

        } else if (key == "autoexposure") {
            config.autoExposure = toBool(value, config.autoExposure);
        } else if (key == "autoexposurespeed") {
            config.autoExposureSpeed = toFloat(value, config.autoExposureSpeed);
        } else if (key == "wbk") {
            config.whiteBalanceK = toFloat(value, config.whiteBalanceK);
        } else if (key == "wbtint") {
            config.whiteBalanceTint = toFloat(value, config.whiteBalanceTint);
        } else if (key == "sharpen") {
            config.sharpen = toFloat(value, config.sharpen);

        } else if (key == "ao") {
            config.ao = toBool(value, config.ao);
        } else if (key == "probegi") {
            config.probeGI = toBool(value, config.probeGI);
        } else if (key == "denoise") {
            config.denoise = toBool(value, config.denoise);
        } else if (key == "restir") {
            config.restirDI = toBool(value, config.restirDI);
        } else if (key == "sunsoft") {
            config.sunAngularRadius = toFloat(value, config.sunAngularRadius);
        } else if (key == "envsun") {
            config.envSun = envSunFrom(value, config.envSun);
        } else if (key == "fireflyclamp") {
            config.fireflyClamp = toFloat(value, config.fireflyClamp);

        } else if (key == "physcam") {
            config.physicalCamera = toBool(value, config.physicalCamera);
        } else if (key == "physunits") {
            config.physicalLightUnits = toBool(value, config.physicalLightUnits);
        } else if (key == "aperture") {
            config.aperture = toFloat(value, config.aperture);
        } else if (key == "shutter") {
            config.shutterSeconds = toFloat(value, config.shutterSeconds);
        } else if (key == "iso") {
            config.iso = toFloat(value, config.iso);
        } else if (key == "evcomp") {
            config.exposureCompensation = toFloat(value, config.exposureCompensation);
        } else if (key == "dof") {
            config.depthOfField = toBool(value, config.depthOfField);
        } else if (key == "focus") {
            config.focusDistance = toFloat(value, config.focusDistance);
        } else if (key == "motionblur") {
            config.motionBlur = toFloat(value, config.motionBlur);

        } else if (key == "bloom") {
            config.bloom = toFloat(value, config.bloom);
        } else if (key == "bloomthreshold") {
            config.bloomThreshold = toFloat(value, config.bloomThreshold);
        } else if (key == "heightfog") {
            config.heightFog = toBool(value, config.heightFog);
        } else if (key == "fogdensity") {
            config.fogDensity = toFloat(value, config.fogDensity);
        } else if (key == "fogbase") {
            config.fogBaseY = toFloat(value, config.fogBaseY);
        } else if (key == "fogfalloff") {
            config.fogFalloff = toFloat(value, config.fogFalloff);
        } else if (key == "fognoise") {
            config.fogNoise = toFloat(value, config.fogNoise);
        } else if (key == "foganiso") {
            config.fogAnisotropy = toFloat(value, config.fogAnisotropy);
        } else if (key == "beams") {
            config.beamDensity = toFloat(value, config.beamDensity);
        } else if (key == "beamaniso") {
            config.beamAnisotropy = toFloat(value, config.beamAnisotropy);
        } else if (key == "clouds") {
            config.clouds = toBool(value, config.clouds);
        } else if (key == "cloudcoverage") {
            config.cloudCoverage = toFloat(value, config.cloudCoverage);
        } else if (key == "clouddensity") {
            config.cloudDensity = toFloat(value, config.cloudDensity);
        } else if (key == "cloudbottom") {
            config.cloudBottomY = toFloat(value, config.cloudBottomY);
        } else if (key == "cloudtop") {
            config.cloudTopY = toFloat(value, config.cloudTopY);
        } else if (key == "cloudwindx") {
            config.cloudWindX = toFloat(value, config.cloudWindX);
        } else if (key == "cloudwindz") {
            config.cloudWindZ = toFloat(value, config.cloudWindZ);
        } else if (key == "cloudevolve") {
            config.cloudEvolve = toFloat(value, config.cloudEvolve);

        } else if (key == "occlusion") {
            config.occlusionCulling = toBool(value, config.occlusionCulling);
        } else if (key == "autolod") {
            config.autoLod = toBool(value, config.autoLod);
        } else if (key == "lodpx") {
            config.autoLodError = toFloat(value, config.autoLodError);
        } else if (key == "toksvig") {
            config.normalMapToksvig = toBool(value, config.normalMapToksvig);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<RenderConfig> RenderConfig::read(const Object3D& object, const RenderConfig& base) {

    // decode() needs the inherited base, so this one cannot go through
    // readEntry — the presence check is the shared half.
    if (!hasEntry(object, userDataKey)) return std::nullopt;
    return decode(readString(object, userDataKey), base);
}

void RenderConfig::write(Object3D& object, const RenderConfig& base) const {

    const auto encoded = encode(base);
    if (encoded.empty()) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encoded;
}

void RenderConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
