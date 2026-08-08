
#include "threepp/helpers/CameraSensor.hpp"

#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/utils/ImageUtils.hpp"

// stb_image_write - implementation is compiled in utils/StbImageWrite.cpp.
#include "stb_image_write.h"

#include <algorithm>
#include <system_error>

using namespace threepp;

CameraSensor::CameraSensor(float fovY, unsigned int width, unsigned int height,
                           float near, float far)
    // The cast is not decoration: Sensor's one-argument constructor and its
    // (deleted) copy constructor both accept a `CameraSensor&`, and overload
    // resolution picks the copy. Naming the Object3D base disambiguates. The
    // ranging sensors write `VisionSensor(*this, noise)` and never meet this,
    // because two arguments cannot match a copy.
    : Sensor(static_cast<Object3D&>(*this)),
      width_(std::max(width, 1u)),
      height_(std::max(height, 1u)),
      camera_(fovY, static_cast<float>(std::max(width, 1u)) / static_cast<float>(std::max(height, 1u)),
              near, far) {

    RenderTarget::Options options;
    options.format = Format::RGB;
    options.generateMipmaps = false;
    options.depthBuffer = true;
    options.stencilBuffer = false;
    // The frame is a PICTURE, so it is encoded for display. Rendering into a
    // target takes the output colour space from the target's own texture (a
    // render target is usually an intermediate, where an early sRGB encode
    // would be wrong), and leaving it linear here would hand every reader a
    // dark, washed-out image that does not match the viewport it was captured
    // beside.
    options.encoding = ColorSpace::sRGB;
    // A camera with stair-stepped edges reads as a rendering artefact rather
    // than as a sensor. The resolve happens on unbind and the readback path
    // already knows to read the resolved framebuffer, so this costs a little
    // memory and nothing in correctness.
    options.samples = 4;
    target_ = RenderTarget::create(width_, height_, options);

    addRef(camera_);
}

CameraSensor::~CameraSensor() = default;

bool CameraSensor::capture(Renderer& renderer, Scene& scene) {

    // Bind first and ASK: a backend without render targets (VulkanRenderer
    // ignores setRenderTarget and always reports nullptr) would otherwise
    // render this camera straight over the window and read the swapchain back
    // as if it were the sensor's frame. Probing the renderer's own answer keeps
    // that decision out of an #ifdef and out of a dynamic_cast.
    auto* restore = renderer.getRenderTarget();
    renderer.setRenderTarget(target_.get());
    if (renderer.getRenderTarget() != target_.get()) {
        renderer.setRenderTarget(restore);
        return false;
    }

    renderer.render(scene, camera_);

    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(restore);

    const auto expected = static_cast<std::size_t>(width_) * height_ * 3;
    if (pixels.size() != expected) return false;

    // A raster readback is bottom-up; image() promises top-left origin.
    flipImage(pixels, 3, static_cast<int>(width_), static_cast<int>(height_));
    image_ = std::move(pixels);

    ++frames_;
    lastCaptureTime_ = simTime();
    due_ = false;
    return true;
}

bool CameraSensor::writeImage(const std::filesystem::path& file) const {

    if (image_.empty()) return false;

    if (file.has_parent_path() && !exists(file.parent_path())) {
        std::error_code ec;
        create_directories(file.parent_path(), ec);
        if (ec) return false;
    }

    const auto w = static_cast<int>(width_);
    const auto h = static_cast<int>(height_);
    const auto ext = file.extension().string();
    const std::string path = file.string();// must outlive the c_str() below

    if (ext == ".png") {
        return stbi_write_png(path.c_str(), w, h, 3, image_.data(), w * 3) != 0;
    }
    if (ext == ".jpg" || ext == ".jpeg") {
        return stbi_write_jpg(path.c_str(), w, h, 3, image_.data(), 95) != 0;
    }
    if (ext == ".bmp") {
        return stbi_write_bmp(path.c_str(), w, h, 3, image_.data()) != 0;
    }
    return false;
}

void CameraSensor::reset() {

    image_.clear();
    frames_ = 0;
    lastCaptureTime_ = 0.0;
    due_ = false;
    resetTiming();
}
