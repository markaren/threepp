
#ifndef THREEPP_EDITOR_VULKANVIEWPANE_HPP
#define THREEPP_EDITOR_VULKANVIEWPANE_HPP

#include <cstdint>
#include <string>

namespace threepp {

    class Camera;
    class Renderer;
    class VulkanRenderer;

}// namespace threepp

namespace threepp::editor {

    // One persistent VulkanRenderer secondary view, shown inside the editor's
    // own frame.
    //
    // The point of the class is that the editor never touches a view handle
    // directly. A view is a real allocation — a whole second deferred chain,
    // ~46 MB at 320x200 — so the rules around it are not optional:
    //
    //   * ONE view, re-pointed. Changing which camera it shows is
    //     setViewCamera, never remove + add.
    //   * A size change is the only thing that recreates it, and only after
    //     the size has held still (kResizeSettleFrames), because a window drag
    //     would otherwise churn a full chain every frame.
    //   * The handle is released by release(), by attaching a different
    //     renderer, and by the destructor. There is no path that drops it.
    //
    // On a non-Vulkan renderer every call is a no-op and active() is false, so
    // the editor keeps one code path and asks `active()` which one ran.
    class VulkanViewPane {

    public:
        VulkanViewPane() = default;
        ~VulkanViewPane();

        VulkanViewPane(const VulkanViewPane&) = delete;
        VulkanViewPane& operator=(const VulkanViewPane&) = delete;

        // Renderer may be anything; only a VulkanRenderer arms the pane.
        void attach(Renderer* renderer);
        // True when this pane can render at all — i.e. the renderer is Vulkan.
        [[nodiscard]] bool supported() const { return vk_ != nullptr; }

        // Once per frame, BEFORE Renderer::render(): the view is recorded
        // inside that call, so everything it needs must be true by then.
        //
        // `camera` null, or an empty rect, releases the view — a collapsed dock
        // should not be holding 46 MB.
        //
        // The camera is taken by pointer EVERY frame and handed to the renderer
        // every frame rather than cached. Play and Stop replace the whole scene,
        // which destroys and rebuilds every camera in it: a cached Camera& (or a
        // "same uuid, skip the update" optimisation) would leave the renderer
        // holding a pointer into freed memory, and it would survive the uuid
        // check precisely because the uuid is what stayed the same.
        void sync(Camera* camera, int x, int y, int w, int h);

        // After Renderer::render(): puts back what sync() borrowed. The pane
        // renders the camera at the PANE's aspect, which is not the aspect the
        // camera is authored with, and the inspector shows the authored one.
        void endFrame();

        // True once the view has been composited into the frame at least once.
        // Until then the pane's pixels are whatever the primary drew there, and
        // the caller should paint over them.
        [[nodiscard]] bool active() const;

        [[nodiscard]] std::uint32_t handle() const { return handle_; }
        // Pixel size the view was actually created at (0 when there is none).
        [[nodiscard]] int width() const { return w_; }
        [[nodiscard]] int height() const { return h_; }

        void release();

    private:
        // How many consecutive frames a new size must hold before the view is
        // recreated at it. A panel drag changes the rect every frame; a view
        // costs a device drain and a full reallocation.
        static constexpr int kResizeSettleFrames = 10;

        // Never dereferenced outside the Vulkan build; held as a bare pointer
        // because the editor owns the renderer and outlives this.
        VulkanRenderer* vk_ = nullptr;
        std::uint32_t handle_ = 0;
        // The size the live view was created at.
        int w_ = 0, h_ = 0;
        // The size most recently ASKED for, and for how many frames running.
        int wantW_ = 0, wantH_ = 0;
        int settled_ = 0;
        // Renders since the view was created. The first render after addView is
        // where its resources are allocated and it first draws, and that render
        // can be deferred a frame (the shared render pass has to exist), so the
        // pane only claims to be live once it has certainly drawn.
        int renders_ = 0;

        // What sync() borrowed from the camera, for endFrame() to put back.
        // Kept as values plus a pointer used only for identity: the camera may
        // be destroyed between the two calls only if the scene is replaced
        // mid-frame, which the editor does not do, and the pointer comparison
        // is what makes that assumption checkable rather than assumed.
        Camera* borrowed_ = nullptr;
        float savedAspect_ = 0.f;
        float savedLeft_ = 0.f, savedRight_ = 0.f;
        bool borrowedOrtho_ = false;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_VULKANVIEWPANE_HPP
