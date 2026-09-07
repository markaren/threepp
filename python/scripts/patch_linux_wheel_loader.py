"""Post-`auditwheel` surgery on the Linux wheels, plus the gate that keeps it honest.

Run from CIBW_REPAIR_WHEEL_COMMAND_LINUX, immediately after `auditwheel repair`.
Two jobs, both about the vendored Vulkan loader:

1. GIVE THE VENDORED LOADER ITS SONAME BACK. auditwheel renames every grafted
   library to a hash-mangled, collision-proof name — `libvulkan.so.1` becomes
   `threepp.libs/libvulkan-a60cf778.so.1.4.328` — and rewrites the extension
   module's DT_NEEDED to match. That is right for libraries nobody looks up by
   name, and WRONG here, because GLFW does not link the loader: it
   `dlopen("libvulkan.so.1")`s it (src/external/glfw/src/vulkan.c). Under a
   mangled name GLFW cannot find the wheel's copy, so it falls through to the
   system loader — and then glfwGetRequiredInstanceExtensions() /
   glfwCreateWindowSurface() run against a DIFFERENT loader instance from the one
   the module linked, holding a VkInstance the other loader has never heard of.
   Two loaders in one process is not a degraded mode, it is undefined behaviour:
   instance handles are loader-internal.

   So: rename the file back to `libvulkan.so.1`, restore its SONAME, and point the
   module's DT_NEEDED at it. auditwheel's own DT_RPATH ($ORIGIN/../threepp.libs)
   then serves BOTH the link-time resolution and GLFW's dlopen, and glibc dedupes
   the two by inode into one loader. It also, for free, gets the behaviour the
   plain soname implies: if something else in the process already loaded a system
   `libvulkan.so.1`, ld.so reuses that one rather than adding a second.

   The mangling exists to stop two wheels colliding in one process. Nothing
   collides here: the only path that reaches this file is our own DT_RPATH, which
   no other distribution points at.

2. FAIL THE BUILD IF THE LOADER HAS NO WSI. The loader's window-system support is
   compile-time (VK_USE_PLATFORM_XLIB_KHR and friends), and vcpkg's
   `vulkan-loader` port defaults every one of them OFF — they are opt-in port
   FEATURES, which is how threepp 2026.8.9 shipped a loader that could not create
   a window surface on any Linux machine. LOADER_INSTANCE_EXTENSIONS is a FILTER:
   a loader built without a platform strips that platform's surface extension out
   of vkEnumerateInstanceExtensionProperties even when the ICD offers it, so
   vkCreateInstance answers VK_ERROR_EXTENSION_NOT_PRESENT for the very extensions
   GLFW just asked for.

   String-probing the .so does not detect this — the extension NAMES survive in
   the loader's proc-address tables whether or not the platform is compiled in
   (verified against the broken 2026.8.9 wheel: `VK_KHR_xcb_surface` present,
   `vkCreateXcbSurfaceKHR` nowhere). The EXPORTED ENTRY POINTS are the honest
   signal, so that is what this checks.

   VK_EXT_headless_surface is deliberately not treated as a WSI build option: the
   loader compiles it unconditionally, and whether it is ADVERTISED depends on the
   ICD forwarding it (loader/wsi.c dispatches VK_ICD_WSI_PLATFORM_HEADLESS straight
   to the driver). Mesa/lavapipe has it, NVIDIA's Linux ICD does not. Nothing this
   script or the loader build can change — it is asserted as exported, not as
   working.

Usage:  python patch_linux_wheel_loader.py <dir-of-wheels-or-wheel> ...
Idempotent: a wheel whose loader already carries its soname is only verified.
"""

import base64
import csv
import hashlib
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Entry points that exist only when the loader was built with that platform's
# VK_USE_PLATFORM_* define. One missing name means the WSI feature list in
# vcpkg.json did not reach the port, or its headers were absent from the
# manylinux container.
REQUIRED_LOADER_EXPORTS = {
    "vkCreateXlibSurfaceKHR": "xlib (BUILD_WSI_XLIB_SUPPORT / libX11-devel)",
    "vkCreateXcbSurfaceKHR": "xcb (BUILD_WSI_XCB_SUPPORT / libxcb-devel)",
    "vkCreateWaylandSurfaceKHR": "wayland (BUILD_WSI_WAYLAND_SUPPORT / wayland-devel)",
    # Unconditional in the loader; asserted so a truncated or substituted loader
    # is caught here rather than at someone's first vkCreateInstance.
    "vkCreateHeadlessSurfaceEXT": "headless surface (always compiled in)",
    "vkGetInstanceProcAddr": "loader core",
}

SONAME = "libvulkan.so.1"
# auditwheel's mangling: <stem>-<hash>.so<rest>
MANGLED = re.compile(r"^libvulkan-[0-9a-f]{8,}\.so.*$")


# ── minimal ELF64 reader ──────────────────────────────────────────────────────
# pyelftools comes with auditwheel and would be available here, but the whole
# need is two tables, and depending on someone else's transitive dependency is a
# worse trade than sixty lines.
class Elf:

    def __init__(self, blob):
        if blob[:4] != b"\x7fELF" or blob[4] != 2:
            raise ValueError("not an ELF64 object")
        (_, _, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum,
         e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", blob, 16)
        self.b = blob
        self.sections = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            name, _typ, _flags, _addr, offset, size, _link, _info, _align, entsize = \
                struct.unpack_from("<IIQQQQIIQQ", blob, off)
            self.sections.append({"name": name, "offset": offset, "size": size, "entsize": entsize})
        shstr = self.sections[e_shstrndx]["offset"]
        for s in self.sections:
            s["sname"] = self._cstr(shstr + s["name"])

    def _cstr(self, off):
        return self.b[off:self.b.index(b"\0", off)].decode("utf-8", "replace")

    def _section(self, name):
        return next((s for s in self.sections if s["sname"] == name), None)

    def exports(self):
        """Defined symbols in .dynsym (st_shndx != SHN_UNDEF)."""
        sym, strt = self._section(".dynsym"), self._section(".dynstr")
        if not sym or not strt:
            return set()
        out = set()
        for i in range(sym["size"] // 24):
            st_name, _info, _other, st_shndx, _value, _size = \
                struct.unpack_from("<IBBHQQ", self.b, sym["offset"] + i * 24)
            if st_name and st_shndx != 0:
                out.add(self._cstr(strt["offset"] + st_name))
        return out

    def needed(self):
        dyn, strt = self._section(".dynamic"), self._section(".dynstr")
        if not dyn:
            return []
        out, off = [], dyn["offset"]
        while True:
            d_tag, d_val = struct.unpack_from("<qQ", self.b, off)
            off += 16
            if d_tag == 0:
                return out
            if d_tag == 1:  # DT_NEEDED
                out.append(self._cstr(strt["offset"] + d_val))


def patchelf(*args):
    subprocess.run(["patchelf", *args], check=True)


def record_row(arcname, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [arcname, "sha256=" + digest, str(len(data))]


def is_elf(data):
    return data[:4] == b"\x7fELF"


def patch_wheel(wheel):
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        loaders = [n for n in names if "/" in n
                   and (MANGLED.match(n.rsplit("/", 1)[1]) or n.endswith("/" + SONAME))]
        if not loaders:
            raise SystemExit(
                wheel.name + ": no vendored Vulkan loader found. The Linux wheels build with "
                "-DTHREEPP_WITH_VULKAN=ON and auditwheel is expected to graft libvulkan.so.1 "
                "into <pkg>.libs — if that stopped happening, the wheel has silently lost its "
                "Vulkan backend.")
        if len(loaders) > 1:
            raise SystemExit("%s: more than one vendored loader: %s" % (wheel.name, loaders))

        old = loaders[0]
        libs_dir, old_base = old.rsplit("/", 1)
        new = libs_dir + "/" + SONAME
        already = old_base == SONAME

        blobs = {n: z.read(n) for n in names}
        infos = {i.filename: i for i in z.infolist()}

    # Before touching anything: if the loader has no WSI the wheel is unusable and
    # renaming it would only make a broken wheel look tidier.
    verify_loader(wheel.name, blobs[old])

    tmp = Path(tempfile.mkdtemp(prefix="threepp-wheel-"))
    try:
        if already:
            print("  %s: already carries its soname, verifying only" % old)
        else:
            # The loader: soname back to the real thing.
            f = tmp / SONAME
            f.write_bytes(blobs[old])
            patchelf("--set-soname", SONAME, str(f))
            blobs[old] = f.read_bytes()

            # Everything that names the mangled loader: point it at the soname.
            for n, data in list(blobs.items()):
                if n == old or not is_elf(data) or old_base not in Elf(data).needed():
                    continue
                g = tmp / Path(n).name
                g.write_bytes(data)
                patchelf("--replace-needed", old_base, SONAME, str(g))
                blobs[n] = g.read_bytes()
                print("  %s: DT_NEEDED %s -> %s" % (n, old_base, SONAME))
            blobs[new] = blobs.pop(old)
            print("  %s -> %s (soname restored)" % (old, new))

        verify_single_loader(wheel.name, blobs)
        if already:
            return

        # RECORD is a manifest of paths + hashes; rewrite the rows we moved.
        record = next(n for n in blobs if n.endswith(".dist-info/RECORD"))
        rows = []
        for row in csv.reader(io.StringIO(blobs[record].decode())):
            if not row:
                continue
            path = new if row[0] == old else row[0]
            if path == record:
                rows.append([record, "", ""])
            elif path in blobs:
                rows.append(record_row(path, blobs[path]))
            else:
                rows.append(row)
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        blobs[record] = out.getvalue().encode()

        # Rebuild in the original entry order, preserving each entry's mode and
        # compression — the loader is 0755 and pip honours what the zip says.
        with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                arc = new if n == old else n
                src = infos[n]
                info = zipfile.ZipInfo(arc, date_time=src.date_time)
                info.external_attr = src.external_attr
                info.create_system = src.create_system
                info.compress_type = src.compress_type
                z.writestr(info, blobs[arc])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_loader(wheel_name, blob):
    exported = Elf(blob).exports()
    missing = {s: why for s, why in REQUIRED_LOADER_EXPORTS.items() if s not in exported}
    if missing:
        raise SystemExit(
            "%s: the vendored Vulkan loader is missing WSI entry points:\n%s\n"
            "  vcpkg's vulkan-loader port defaults BUILD_WSI_* to OFF; the xcb/xlib/wayland\n"
            "  features in vcpkg.json turn them on, and they need libxcb-devel / libX11-devel /\n"
            "  wayland-devel present in the manylinux container (CIBW_BEFORE_ALL_LINUX).\n"
            "  A loader without them strips VK_KHR_*_surface out of\n"
            "  vkEnumerateInstanceExtensionProperties, so every windowed Vulkan run fails with\n"
            "  VK_ERROR_EXTENSION_NOT_PRESENT."
            % (wheel_name, "\n".join("    %s  <- %s" % kv for kv in missing.items())))
    print("  loader WSI OK: " + ", ".join(sorted(REQUIRED_LOADER_EXPORTS)))


def verify_single_loader(wheel_name, blobs):
    """No ELF in the wheel may still reference a mangled loader name."""
    for n, data in blobs.items():
        if not is_elf(data) or n.endswith("/" + SONAME):
            continue
        stale = [d for d in Elf(data).needed() if MANGLED.match(d)]
        if stale:
            raise SystemExit(
                "%s: %s still needs %s — GLFW's dlopen(\"%s\") would load a second loader"
                % (wheel_name, n, stale, SONAME))


def main(argv):
    targets = []
    for a in argv:
        p = Path(a)
        targets.extend(sorted(p.glob("*.whl")) if p.is_dir() else [p])
    if not targets:
        raise SystemExit("no wheels given")
    for w in targets:
        print(w.name + ":")
        patch_wheel(w)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [os.getcwd()]))
