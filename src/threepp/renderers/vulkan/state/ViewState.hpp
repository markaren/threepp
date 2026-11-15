// ViewState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the views (the primary and the secondaries), the current view, the primary's frame-prep output, pending
// view changes and the view composites.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_VIEWSTATE_HPP
#define THREEPP_VULKAN_STATE_VIEWSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct ViewState {
        // Definition moved to VulkanViewContext.hpp.
        using ViewContext = vulkan::impl::ViewContext;
        // Every view this renderer drives. views_[0] is the PRIMARY (the
        // swapchain-presenting one) and always exists — it is created in the
        // constructor and never removed, so `view()` is unconditionally valid.
        std::vector<std::unique_ptr<ViewContext>> views_;
        // The view currently being set up / recorded. Always views_[0] today;
        // the per-view render loop retargets it. A raw pointer (not an index)
        // so a view() call is one indirection in the hot record path.
        ViewContext* curView_ = nullptr;
        // Handles are dense and monotonic, never reused. 0 is reserved for the
        // primary (and doubles as "invalid" from addView), so a stale handle
        // from a removed view can never alias a later one.
        uint32_t nextViewId_ = 1;

        // What the frame prep left for the passes of the view being built or
        // recorded (ViewPrepOut, defined with ViewContext). A view's prep
        // running on a pool thread writes into its own ViewContext::prepStage
        // instead, which the frame thread takes in view order before that
        // view's passes are built; prepOut() is the one this thread's code
        // writes and reads.
        using ViewPrepOut     = vulkan::impl::ViewPrepOut;
        using ViewPrepScratch = vulkan::impl::ViewPrepScratch;
        ViewPrepOut prepOut_;

        bool pendingViewChanges_ = false;

        struct ViewCompositeJob {
            VkImage    src;
            VkOffset2D srcOff;
            VkOffset2D dstOff;
            VkExtent2D ext;
        };
        std::vector<ViewCompositeJob> viewCompositeJobs_;// this frame's, built by addViewCompositePass
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_VIEWSTATE_HPP
