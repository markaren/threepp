
#include "threepp/lights/RectAreaLightUniformsLib.hpp"

#include "threepp/lights/ltc/ltc_data.hpp"
#include "threepp/textures/DataTexture.hpp"

using namespace threepp;


namespace {

    std::shared_ptr<Texture> makeLtcTexture(const std::array<float, ltc::LUT_ELEMENTS>& data) {

        std::vector<float> buffer(data.begin(), data.end());
        auto tex = DataTexture::create(
                ImageData(std::move(buffer)),
                ltc::LUT_SIZE, ltc::LUT_SIZE);

        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Nearest;
        tex->generateMipmaps = false;
        tex->unpackAlignment = 1;
        tex->wrapS = TextureWrapping::ClampToEdge;
        tex->wrapT = TextureWrapping::ClampToEdge;
        tex->needsUpdate();

        return tex;
    }

}// namespace


RectAreaLightUniformsLib& RectAreaLightUniformsLib::instance() {

    static RectAreaLightUniformsLib inst;
    return inst;
}

void RectAreaLightUniformsLib::init() {

    if (!ltc1_) ltc1_ = makeLtcTexture(ltc::LTC_MAT_1);
    if (!ltc2_) ltc2_ = makeLtcTexture(ltc::LTC_MAT_2);
}

std::shared_ptr<Texture> RectAreaLightUniformsLib::ltc_1() const {

    return ltc1_;
}

std::shared_ptr<Texture> RectAreaLightUniformsLib::ltc_2() const {

    return ltc2_;
}
