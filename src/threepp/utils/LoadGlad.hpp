
#ifndef THREEPP_LOAD_GLAD_HPP
#define THREEPP_LOAD_GLAD_HPP

#include <glad/glad.h>


namespace threepp {

    void loadGlad();

    // For a context GLAD cannot reach on its own. gladLoadGL() dlopens
    // libGL.so.1 and resolves every entry point through glXGetProcAddressARB —
    // the GLX dispatch, not the current context — so it fails outright where
    // libGL carries no GLX (a driver installed for EGL only) and resolves
    // against the wrong stack for an EGL or OSMesa context even when it
    // succeeds. Hand it the getter belonging to the context instead:
    // glfwGetProcAddress for a GLFW window, eglGetProcAddress for a context
    // made directly against an EGL device.
    void loadGlad(GLADloadproc procAddress);

    // Resolve every entry point again through `procAddress`, whether or not
    // they were loaded before. For a GL module that has been unloaded and
    // loaded again: glfwTerminate() closes the libGL it opened, so once the
    // last Canvas is gone and a new one initialises GLFW afresh, the pointers
    // GLAD holds may point into an unmapped library.
    void reloadGlad(GLADloadproc procAddress);

    /// True once GLAD's pointers have been filled in, by either overload.
    /// Call it before loadGlad() when a context may already have loaded them
    /// through its own getter — GLRenderer's size-only constructor does, so
    /// that an EglContext followed by a renderer is not mistaken for two
    /// competing loaders.
    [[nodiscard]] bool gladLoaded();
}

#endif//THREEPP_LOAD_GLAD_HPP
