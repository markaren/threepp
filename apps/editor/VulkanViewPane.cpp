
#include "VulkanViewPane.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/renderers/Renderer.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cmath>

using namespace threepp;
using namespace threepp::editor;

VulkanViewPane::~VulkanViewPane() {

    release();
}

void VulkanViewPane::attach(Renderer* renderer) {

    release();
#ifdef THREEPP_WITH_VULKAN
    vk_ = dynamic_cast<VulkanRenderer*>(renderer);
#else
    (void) renderer;
    vk_ = nullptr;
#endif
}

bool VulkanViewPane::active() const {

    return handle_ != 0 && renders_ >= 2;
}

void VulkanViewPane::release() {

#ifdef THREEPP_WITH_VULKAN
    if (vk_ && handle_ != 0) vk_->removeView(handle_);
#endif
    handle_ = 0;
    w_ = h_ = 0;
    wantW_ = wantH_ = 0;
    settled_ = 0;
    renders_ = 0;
    borrowed_ = nullptr;
}

void VulkanViewPane::sync(Camera* camera, int x, int y, int w, int h) {

#ifdef THREEPP_WITH_VULKAN
    if (!vk_) return;

    // Nothing to show: hand the memory back rather than keep a hidden view
    // alive for a dock that may stay collapsed for the rest of the session.
    if (!camera || w < 1 || h < 1) {
        release();
        return;
    }

    // Debounce the size. A panel drag walks the rect through a hundred
    // intermediate widths, and each one would be a device drain plus a full
    // deferred-chain reallocation. Only a size that has held still for
    // kResizeSettleFrames is acted on — and the FIRST size is acted on
    // immediately, because there is nothing to churn yet.
    if (w != wantW_ || h != wantH_) {
        wantW_ = w;
        wantH_ = h;
        settled_ = 0;
    } else if (settled_ < kResizeSettleFrames) {
        ++settled_;
    }

    const bool sizeChanged = (w != w_ || h != h_);
    const bool sizeReady = handle_ == 0 || settled_ >= kResizeSettleFrames;
    if (sizeChanged && sizeReady) {
        if (handle_ != 0) vk_->removeView(handle_);
        handle_ = vk_->addView(*camera, w, h);
        w_ = w;
        h_ = h;
        renders_ = 0;
        if (handle_ == 0) return;// pre-first-render; retried next frame
        // A pane is a VIEWPORT, and the primary viewport beside it draws
        // splats — as does this pane on the OpenGL backend, which has no such
        // switch. The cost (a second sort over the whole cloud) is the price of
        // a pane that shows the document.
        vk_->setViewSplats(handle_, true);
    }
    if (handle_ == 0) return;

    // Unconditionally, every frame. See the header: a scene replace rebuilds
    // every camera in the document behind the same uuid, so "the uuid did not
    // change" is exactly the case where the renderer's Camera* went stale.
    // setViewCamera compares uuids itself and only drops the temporal history
    // when the camera genuinely changed, so paying this every frame costs a
    // string compare and buys the absence of a dangling pointer.
    vk_->setViewCamera(handle_, *camera);

    // The pane is not the shape the camera is authored for. Borrow its
    // projection for this frame the way the OpenGL dock always has, and give
    // it back in endFrame() — the inspector, the frustum helper and the
    // document all read the authored value.
    const float aspect = static_cast<float>(w) / static_cast<float>(h);
    borrowed_ = camera;
    if (auto* persp = camera->as<PerspectiveCamera>()) {
        borrowedOrtho_ = false;
        savedAspect_ = persp->aspect;
        persp->aspect = aspect;
        persp->updateProjectionMatrix();
    } else if (auto* ortho = camera->as<OrthographicCamera>()) {
        // Height is what an orthographic camera is authored by; the width
        // follows the pane, centred on whatever the authored frustum was
        // centred on.
        borrowedOrtho_ = true;
        savedLeft_ = ortho->left;
        savedRight_ = ortho->right;
        const float halfH = std::abs(ortho->top - ortho->bottom) * 0.5f;
        const float mid = (ortho->left + ortho->right) * 0.5f;
        ortho->left = mid - halfH * aspect;
        ortho->right = mid + halfH * aspect;
        ortho->updateProjectionMatrix();
    } else {
        borrowed_ = nullptr;
    }

    vk_->setViewDisplayRect(handle_, x, y, w, h);
    ++renders_;
#else
    (void) camera;
    (void) x;
    (void) y;
    (void) w;
    (void) h;
#endif
}

void VulkanViewPane::endFrame() {

    if (!borrowed_) return;
    if (borrowedOrtho_) {
        if (auto* ortho = borrowed_->as<OrthographicCamera>()) {
            ortho->left = savedLeft_;
            ortho->right = savedRight_;
            ortho->updateProjectionMatrix();
        }
    } else {
        if (auto* persp = borrowed_->as<PerspectiveCamera>()) {
            persp->aspect = savedAspect_;
            persp->updateProjectionMatrix();
        }
    }
    borrowed_ = nullptr;
}
