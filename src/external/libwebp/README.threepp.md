# libwebp (vendored, decoder subset)

WebP decoding for `ImageLoader`. stb_image — the decoder behind every other image format
threepp reads — has no WebP support and upstream has said it never will, so the reference
implementation is vendored instead.

| | |
|---|---|
| Upstream | https://github.com/webmproject/libwebp |
| Version | **v1.6.0** (June 2025) |
| Source archive | `https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz` |
| SHA-256 | `e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564` |
| License | BSD-3-Clause (`COPYING`) + WebM patent grant (`PATENTS`) |

Never downgrade below 1.3.2: CVE-2023-4863 is an out-of-bounds write in the lossless
Huffman decoder, reachable from an ordinary `.webp` file.

## What was copied

Exactly the file set of upstream's own decoder-only artifact, the `webpdecoder` target.
That target is the manifest — do not hand-pick. In the tarball it is assembled from three
object libraries, each of whose source list is parsed straight out of a `Makefile.am`:

| Object library | Source list |
|---|---|
| `webpdecode` | every `*_SOURCES` in `src/dec/Makefile.am` |
| `webpdspdecode` | `COMMON_SOURCES` + every `libwebpdspdecode_<simd>_la_SOURCES` in `src/dsp/Makefile.am` |
| `webputilsdecode` | `COMMON_SOURCES` in `src/utils/Makefile.am` |

Plus three headers the code includes but those lists do not name — upstream never notices
because it always builds from a complete source tree: `src/dsp/common_sse41.h`
(`yuv_sse41.c`), `src/webp/encode.h` (`palette.c`, `utils.c`) and `src/webp/mux_types.h`
(`webp_dec.c`, for `ALPHA_FLAG`). They are headers only; no encoder is compiled.

The directory layout is preserved because libwebp's internal includes are rooted at the
package directory (`#include "src/dec/vp8i_dec.h"`), which is why `src/CMakeLists.txt` puts
`external/libwebp` on the include path *before* anything else that could shadow a `src/`
subdirectory.

## What was stripped

`src/enc/`, `src/mux/`, `src/demux/`, `sharpyuv/`, `extras/`, `examples/`, `imageio/`,
`tests/`, `swig/`, the man pages and every upstream build file (autotools, Android.mk,
gradle, Xcode). Consequences worth knowing:

- **No encoder.** Nothing in threepp writes WebP; the exporter still emits PNG. Adding it
  later means `src/enc` **and** `sharpyuv`, which the encoder depends on.
- **No demux, so no animated WebP.** `ImageLoader` reports and rejects a file whose
  `has_animation` feature is set rather than decoding a wrong-looking first frame.
- **No `src/webp/config.h`.** It is an autotools artifact and every include of it is behind
  `#ifdef HAVE_CONFIG_H`, which the threepp build does not define. Feature detection falls
  back to libwebp's compile-time checks, which is what a non-autotools consumer gets.

## Build notes

Compiled directly into the `threepp` target (see `src/CMakeLists.txt`), like meshoptimizer
and pugixml — upstream's CMake is not used, since `add_subdirectory` would drag in install
rules, pkg-config generation and package config exports.

- **No `WEBP_USE_THREAD`**: decoding is single-threaded per image. Callers that decode many
  images at once (FBXLoader) already parallelize across images.
- **No per-file SIMD flags.** SSE2 is baseline on x64 and NEON on aarch64, so those kernels
  compile and are picked by runtime dispatch. The SSE4.1/AVX2 files compile everywhere but
  self-disable without `-msse4.1`/`-mavx2` on GCC/Clang, and dispatch then skips them —
  intended degradation, not breakage. The MIPS/MSA files are inert off those targets.
- The vendored C is compiled with warnings off (`-w` / `/w`) so that upstream code can never
  fail a `THREEPP_TREAT_WARNINGS_AS_ERRORS` build.

## Updating

Re-download the pinned tarball, re-derive the file set from the three `Makefile.am` lists
above, and diff. Keep every per-file license header intact and re-copy `COPYING` and
`PATENTS` verbatim; if the version changes, update the row in the repository's
[THIRD_PARTY.md](../../../THIRD_PARTY.md) in the same commit.
