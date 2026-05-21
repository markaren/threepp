// Event-camera (DVS) sensor simulation. Detects per-pixel log-intensity
// crossings of the renderer's output to emit a sparse, asynchronous-style
// event stream — the standard model used by ESIM and most academic DVS
// simulators.
//
// Per-pixel model: the sensor stores the log-intensity at which the pixel
// most recently "fired". Each new frame, the renderer's pixel value is
// compared against that reference; if |Δlog I| exceeds `contrastThreshold`,
// one or more events are emitted (one per integer multiple of the
// threshold) with polarity = sign(Δ). The reference is then advanced by
// `polarity · threshold · N` so subsequent frames see the residual delta,
// matching what real DVS analog front-ends do.
//
// Output is two-pronged:
//   1. A `std::vector<EventCameraEvent>` filled per frame with discrete
//      (x, y, timestamp, polarity) tuples — what downstream perception
//      code consumes.
//   2. A `DataTexture` accumulator that paints recent events into a
//      monochrome image (positive → bright, negative → dark, no event →
//      mid-grey with exponential decay) — the standard DVS visualisation
//      pattern, ready to drop into a `Sprite` overlay.
//
// Limitations of this first version:
//   - Per-frame discretisation (timestamps = frame time), not sub-frame.
//     Real DVS sensors operate at μs-level temporal resolution; this is
//     at the renderer's frame rate.
//   - Operates on display-encoded sRGB pixels rather than scene-referred
//     linear radiance. The log-difference threshold still works correctly
//     because sRGB encoding is monotonic and the photoreceptor model is
//     itself ~logarithmic in physical units — but absolute threshold
//     values are not directly comparable to the C values in the DVS128
//     datasheet.
//   - CPU readback per frame stalls the GPU pipeline briefly; expect a
//     couple of ms overhead on the renderer's frame time.

#ifndef THREEPP_EVENTCAMERASENSOR_HPP
#define THREEPP_EVENTCAMERASENSOR_HPP

#include "threepp/textures/DataTexture.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace threepp {

    class Renderer;
    class Canvas;

    struct EventCameraEvent {
        uint16_t x;
        uint16_t y;
        float    timestamp;// seconds since sensor construction
        int8_t   polarity; // +1 = brighter, -1 = darker
    };

    struct EventCameraParams {
        // Log-intensity change required to trigger an event. Real DVS
        // sensors use ~0.10-0.30 (i.e. ~10-35% intensity change). Lower
        // values = noisier / more events; higher = sparser. 0.15 is a
        // reasonable starting point.
        float contrastThreshold = 0.15f;

        // Per-frame exponential decay applied to the visualisation
        // accumulator. 0 = events flash and vanish immediately; 1 =
        // accumulator never fades. 0.85 gives a ~6-frame trail at 60Hz.
        float vizDecay = 0.85f;

        // Soft pixel floor used before taking log() so dark areas don't
        // generate spurious events from quantisation noise. Real DVS
        // sensors have a similar floor set by analog dark current.
        float minLuma = 0.005f;

        // Cap on events emitted per pixel per frame. Real DVS sensors
        // can't fire infinitely fast; the analog front-end has a
        // refractory period. Capping at ~5 events/pixel/frame keeps the
        // event stream sensible during sudden lighting changes.
        int maxEventsPerPixel = 5;
    };

    /**
     * Event camera sensor. Resolution matches the renderer's swapchain.
     * Construct after creating the renderer; call `captureEvents()` each
     * frame after `renderer.render()` and before any UI overlay.
     */
    class EventCameraSensor {

    public:
        EventCameraParams params;

        /**
         * @param width  Sensor pixel width. Should match the swapchain
         *               width — readRGBPixels returns swapchain-sized data.
         * @param height Sensor pixel height.
         */
        EventCameraSensor(unsigned int width, unsigned int height);

        /**
         * Sample the renderer's current output, detect events relative
         * to the previous capture, and update the visualisation
         * accumulator. Reads pixels via `renderer.readRGBPixels()` —
         * note this captures the FINAL composed swapchain (including
         * any sprite or ImGui overlays). For sensor pipelines that
         * need a clean scene-only image, use the lower-level
         * `ingestPixels` overload with the renderer's
         * `readSceneRGBPixels()` output.
         */
        void captureEvents(Renderer& renderer, std::vector<EventCameraEvent>& events);

        /**
         * Run event detection against externally-supplied RGB byte data.
         * Length must be at least `width * height * 3`. Useful when the
         * caller has its own readback path (e.g. VulkanRenderer's
         * `readSceneRGBPixels` for pre-overlay sampling) or wants to
         * inject synthetic frames.
         */
        void ingestPixels(const unsigned char* rgb,
                          std::size_t bytes,
                          std::vector<EventCameraEvent>& events);

        /**
         * Resize the sensor (call after the canvas resizes). Clears
         * history so the first frame after resize doesn't emit a
         * burst of spurious events.
         */
        void setSize(unsigned int width, unsigned int height);

        /**
         * RGBA8 visualisation texture (drop into a SpriteMaterial::map).
         * Pixel value: +polarity events → white, -polarity → black,
         * no recent events → mid-grey. Decays toward grey between
         * events at `params.vizDecay`.
         */
        [[nodiscard]] std::shared_ptr<DataTexture> visualisation() const { return vizTex_; }

        [[nodiscard]] unsigned int width() const { return width_; }
        [[nodiscard]] unsigned int height() const { return height_; }

    private:
        unsigned int width_  = 0;
        unsigned int height_ = 0;

        std::vector<float>         logRef_;       // per-pixel reference log-luminance
        std::vector<float>         vizAccum_;     // per-pixel viz value in [-1, +1]
        std::vector<unsigned char> readbackBuf_;  // RGB readback scratch
        std::shared_ptr<DataTexture> vizTex_;

        float startTime_  = 0.f;
        float lastTime_   = 0.f;
        bool  firstFrame_ = true;

        void allocateImages();
    };

}// namespace threepp

#endif//THREEPP_EVENTCAMERASENSOR_HPP
