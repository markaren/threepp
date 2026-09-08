#!/usr/bin/env python3
"""Enumerate Vulkan devices and the extensions threepp's Vulkan backend needs.

Pure ctypes into the Vulkan loader: no threepp, no numpy, no display. Meant
for a GPU node where vulkaninfo is not installed:

    python vk_extensions.py

The threepp Vulkan backend shades through ray query and has NO raster
fallback (VulkanContext.cpp), so a device that lacks the KHR ray tracing set
cannot run it at all. Datacenter parts (A100, H100) have no RT cores; whether
the driver still exposes the extensions is exactly what this prints.

Exit status: 0 if at least one device carries the full required set, 1 if
none does, 2 if the loader or instance creation fails.
"""
import ctypes
import struct
import sys

WIN = sys.platform == "win32"

# What VulkanContext.cpp asks for on the device (kRayTracingExtensions + the
# base set) and on the instance (headless surface for display-less canvases).
REQUIRED_DEVICE = [
    "VK_KHR_swapchain",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_acceleration_structure",
    "VK_KHR_deferred_host_operations",
    "VK_KHR_buffer_device_address",
    # VulkanContext probes this one as optional, but every scene shading pass
    # (deferred_shade.comp and its GI / reflection / water stages, the probe
    # and froxel updates) traces through rayQueryEXT, so without it nothing
    # lights the frame. Measured 2026-09-08 on IDUN: the H100 driver 575.57
    # exposes the four KHR ray tracing extensions above but NOT ray query.
    "VK_KHR_ray_query",
]
INTERESTING_DEVICE = [
    "VK_KHR_external_memory_win32" if WIN else "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore_win32" if WIN else "VK_KHR_external_semaphore_fd",
    "VK_EXT_mesh_shader",
    "VK_NV_ray_tracing_motion_blur",
]
INTERESTING_INSTANCE = [
    "VK_KHR_surface",
    "VK_EXT_headless_surface",
    "VK_KHR_external_memory_capabilities",
    "VK_KHR_get_physical_device_properties2",
]

DEVICE_TYPES = {0: "other", 1: "integrated GPU", 2: "discrete GPU", 3: "virtual GPU", 4: "CPU"}


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def vk_version(v):
    return f"{v >> 22}.{(v >> 12) & 0x3ff}.{v & 0xfff}"


def nv_driver_version(v):
    # NVIDIA packs major.minor.patch as 10.8.8 bits; other vendors differ.
    return f"{v >> 22}.{(v >> 14) & 0xff}.{(v >> 6) & 0xff}"


def main():
    try:
        lib = ctypes.CDLL("vulkan-1.dll" if WIN else "libvulkan.so.1")
    except OSError as e:
        print(f"no Vulkan loader: {e}")
        return 2

    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.vkEnumerateInstanceExtensionProperties.argtypes = [ctypes.c_char_p, u32p, ctypes.c_void_p]
    lib.vkCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, u32p, ctypes.c_void_p]
    lib.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [ctypes.c_void_p, ctypes.c_char_p, u32p, ctypes.c_void_p]
    for fn in ("vkEnumerateInstanceExtensionProperties", "vkCreateInstance",
               "vkEnumeratePhysicalDevices", "vkEnumerateDeviceExtensionProperties"):
        getattr(lib, fn).restype = ctypes.c_int

    n = ctypes.c_uint32(0)
    lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(n), None)
    arr = (VkExtensionProperties * max(n.value, 1))()
    lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(n), arr)
    inst_exts = {arr[i].extensionName.decode() for i in range(n.value)}
    print(f"instance: {n.value} extensions")
    for e in INTERESTING_INSTANCE:
        print(f"  {'yes' if e in inst_exts else 'NO ':<3} {e}")

    ci = VkInstanceCreateInfo()
    ci.sType = 1  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    inst = ctypes.c_void_p()
    r = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    if r != 0:
        print(f"vkCreateInstance failed: VkResult {r}")
        return 2

    n = ctypes.c_uint32(0)
    r = lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
    devs = (ctypes.c_void_p * max(n.value, 1))()
    lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
    print(f"\nphysical devices: {n.value} (VkResult {r})")
    if n.value == 0:
        print("  none. The loader found no ICD that reports a device: check that the"
              " NVIDIA ICD json and libGLX_nvidia.so.0 are present and that the job"
              " has a GPU (CUDA_VISIBLE_DEVICES).")

    any_ok = False
    for i in range(n.value):
        d = devs[i]
        buf = ctypes.create_string_buffer(8192)  # VkPhysicalDeviceProperties is ~824 B
        lib.vkGetPhysicalDeviceProperties(d, buf)
        api, drv, vendor, _dev, dtype = struct.unpack_from("<IIIII", buf, 0)
        name = buf.raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
        drv_s = nv_driver_version(drv) if vendor == 0x10DE else f"0x{drv:x}"
        print(f"\n[{i}] {name}  ({DEVICE_TYPES.get(dtype, dtype)}, vendor 0x{vendor:04x}, "
              f"api {vk_version(api)}, driver {drv_s})")

        m = ctypes.c_uint32(0)
        lib.vkEnumerateDeviceExtensionProperties(d, None, ctypes.byref(m), None)
        ext = (VkExtensionProperties * max(m.value, 1))()
        lib.vkEnumerateDeviceExtensionProperties(d, None, ctypes.byref(m), ext)
        dev_exts = {ext[j].extensionName.decode() for j in range(m.value)}
        print(f"    {m.value} device extensions")
        ok = True
        print("    required by threepp's Vulkan backend:")
        for e in REQUIRED_DEVICE:
            have = e in dev_exts
            ok &= have
            print(f"      {'yes' if have else 'NO ':<3} {e}")
        print("    also of interest:")
        for e in INTERESTING_DEVICE:
            print(f"      {'yes' if e in dev_exts else 'NO ':<3} {e}")
        print(f"    verdict: {'CAN run the threepp Vulkan backend' if ok else 'CANNOT run it (missing required extensions)'}")
        any_ok |= ok

    lib.vkDestroyInstance(inst, None)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
