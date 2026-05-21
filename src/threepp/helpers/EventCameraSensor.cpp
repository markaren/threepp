
#include "threepp/helpers/EventCameraSensor.hpp"

#include "threepp/core/Clock.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace threepp;

namespace {
    // Process-local clock so timestamps survive across sensor instances
    // and behave consistently if the user constructs the sensor lazily.
    float epochSeconds() {
        using clock = std::chrono::steady_clock;
        static const auto t0 = clock::now();
        const auto now = clock::now();
        return std::chrono::duration<float>(now - t0).count();
    }
}// namespace


EventCameraSensor::EventCameraSensor(unsigned int width, unsigned int height)
    : width_(width), height_(height) {
    allocateImages();
    startTime_ = epochSeconds();
}

void EventCameraSensor::allocateImages() {
    const size_t pixels = static_cast<size_t>(width_) * height_;
    logRef_.assign(pixels, 0.f);
    vizAccum_.assign(pixels, 0.f);
    readbackBuf_.clear();

    // RGBA8 grey baseline (128, 128, 128, 255) so the texture renders as
    // mid-grey before any events arrive. Tagged sRGB so the renderer's
    // sprite path samples it back into linear space cleanly.
    std::vector<unsigned char> initial(pixels * 4);
    for (size_t i = 0; i < pixels; ++i) {
        initial[i * 4 + 0] = 128;
        initial[i * 4 + 1] = 128;
        initial[i * 4 + 2] = 128;
        initial[i * 4 + 3] = 255;
    }
    vizTex_ = DataTexture::create(ImageData{std::move(initial)}, width_, height_);
    vizTex_->colorSpace = ColorSpace::sRGB;
}

void EventCameraSensor::setSize(unsigned int width, unsigned int height) {
    if (width == width_ && height == height_) return;
    width_  = width;
    height_ = height;
    allocateImages();
    firstFrame_ = true;
}

void EventCameraSensor::captureEvents(Renderer& renderer,
                                       std::vector<EventCameraEvent>& events) {
    auto pixels = renderer.readRGBPixels();
    ingestPixels(pixels.data(), pixels.size(), events);
}

void EventCameraSensor::ingestPixels(const unsigned char* rgb,
                                      std::size_t bytes,
                                      std::vector<EventCameraEvent>& events) {
    events.clear();

    const size_t expected = static_cast<size_t>(width_) * height_ * 3;
    if (rgb == nullptr || bytes < expected) {
        // Renderer hasn't rendered yet, or size mismatches — bail out
        // without touching the accumulator. Caller will retry next frame.
        return;
    }

    const float t       = epochSeconds() - startTime_;
    const float thresh  = std::max(1e-4f, params.contrastThreshold);
    const float minLuma = std::max(1e-6f, params.minLuma);
    const float decay   = std::clamp(params.vizDecay, 0.f, 1.f);
    const int   maxEv   = std::max(1, params.maxEventsPerPixel);

    auto& vizBytes = vizTex_->image().data<unsigned char>();

    for (unsigned y = 0; y < height_; ++y) {
        for (unsigned x = 0; x < width_; ++x) {
            const size_t idx = static_cast<size_t>(y) * width_ + x;
            const size_t pix = idx * 3;

            // sRGB-domain luminance. We deliberately don't sRGB-decode:
            // real DVS photoreceptors are log-of-physical-intensity and
            // sRGB is already a perceptual log-ish encoding, so the
            // relative log-differences track the same events. Absolute
            // contrast thresholds end up in "sRGB log units" rather than
            // SI radiance log units — fine for visualisation + relative
            // perception research, not directly comparable to DVS128
            // datasheet C values.
            const float r = static_cast<float>(rgb[pix + 0]) * (1.f / 255.f);
            const float g = static_cast<float>(rgb[pix + 1]) * (1.f / 255.f);
            const float b = static_cast<float>(rgb[pix + 2]) * (1.f / 255.f);
            const float luma   = std::max(minLuma, 0.2126f * r + 0.7152f * g + 0.0722f * b);
            const float logLum = std::log(luma);

            if (firstFrame_) {
                logRef_[idx] = logLum;
                continue;
            }

            const float delta = logLum - logRef_[idx];
            const float absD  = std::abs(delta);

            if (absD >= thresh) {
                // Emit one event per integer multiple of threshold,
                // capped to keep the event stream bounded under sudden
                // lighting changes. Update the reference by the quantised
                // amount, leaving the residual for next frame.
                const int n = std::min(maxEv, static_cast<int>(absD / thresh));
                const int polarity = (delta > 0.f) ? 1 : -1;
                for (int k = 0; k < n; ++k) {
                    EventCameraEvent ev{};
                    ev.x         = static_cast<uint16_t>(x);
                    ev.y         = static_cast<uint16_t>(y);
                    ev.timestamp = t;
                    ev.polarity  = static_cast<int8_t>(polarity);
                    events.push_back(ev);
                }
                logRef_[idx] += static_cast<float>(polarity * n) * thresh;
                vizAccum_[idx] = static_cast<float>(polarity);
            } else {
                vizAccum_[idx] *= decay;
            }
        }
    }

    firstFrame_ = false;
    lastTime_   = t;

    // Project accumulator into the visualisation texture. Map [-1, +1]
    // into [0, 1] with 0.5 = "no recent event". sRGB byte encoding.
    //
    // We flip Y when writing: the renderer's readRGBPixels delivers pixels
    // in Vulkan top-down order (row 0 = top of image), but the sprite
    // sampler treats V = 0 as the bottom of the texture, so a direct copy
    // would render upside-down. Flipping during the projection costs the
    // same as iterating in order.
    for (unsigned y = 0; y < height_; ++y) {
        const unsigned dstY = height_ - 1u - y;
        for (unsigned x = 0; x < width_; ++x) {
            const size_t srcIdx = static_cast<size_t>(y) * width_ + x;
            const size_t dstIdx = static_cast<size_t>(dstY) * width_ + x;
            const float v   = std::clamp(vizAccum_[srcIdx], -1.f, 1.f);
            const float vis = 0.5f + 0.5f * v;
            const auto  byte = static_cast<unsigned char>(std::clamp(vis * 255.f, 0.f, 255.f));
            const size_t b = dstIdx * 4;
            vizBytes[b + 0] = byte;
            vizBytes[b + 1] = byte;
            vizBytes[b + 2] = byte;
            vizBytes[b + 3] = 255;
        }
    }
    vizTex_->needsUpdate();
}
