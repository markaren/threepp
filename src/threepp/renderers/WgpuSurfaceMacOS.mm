// macOS Metal surface creation helper for WgpuRenderer.
// This .mm file is required for Objective-C++ CAMetalLayer creation.

#ifdef __APPLE__

#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>

extern "C" void* wgpu_create_metal_layer(void* nsWindow) {
    NSWindow* window = (__bridge NSWindow*)nsWindow;
    NSView* view = [window contentView];
    [view setWantsLayer:YES];
    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    [view setLayer:metalLayer];
    return (__bridge void*)metalLayer;
}

#endif // __APPLE__
