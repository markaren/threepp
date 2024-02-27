# threepp + WxWidgets

This is a simple example of how to use [threepp](https://github.com/markaren/threepp) with [WxWidgets](https://www.wxwidgets.org/).

As WxWidget is a complete GUI library, it is not necessary to use threepp's `Canvas` class. Instead, we can use WxWidgets' 
`wxGLCanvas` and `wxGLContext` to create a window and a GL context, and then use threepp's `GLRenderer` to render to it.

Alas, the option `THREEPP_WITH_GLFW` should be set to `OFF` when building threepp.
