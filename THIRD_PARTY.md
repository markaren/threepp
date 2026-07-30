# Third-party software and assets

threepp itself is MIT-licensed — see [LICENSE](LICENSE). This file indexes everything in or
around the project that is *someone else's*, what license it carries, and where its full license
text lives. Vendored files keep their original notices in place; this document is the map, not a
replacement for them.

Two notices are reproduced **in full** below (three.js and the embedded typeface), because the
derived or embedded material in this repository does not otherwise carry them.

---

## three.js

threepp is a C++ port of [three.js](https://github.com/mrdoob/three.js) (r129): the public API
mirrors it by design, the OpenGL backend is a port of its WebGL renderer, and the GLSL in
[`src/shaders/`](src/shaders) derives directly from its shader chunks. Substantial portions of
this repository are therefore a derivative work of three.js, and its license rides with them:

> The MIT License
>
> Copyright © 2010-2021 three.js authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Embedded in the threepp library

**helvetiker (MgOpen Moderna) typeface** — `src/resources/helvetiker_bold.typeface.json` is
compiled into the library binary (`cmake/embed.cmake` → `FontLoader::defaultFont()`). It is the
typeface three.js ships for the same purpose, and it carries the MAGENTA Ltd. font license,
reproduced here because the JSON conversion has no header to carry it:

> Copyright @ 2004 by MAGENTA Ltd. All Rights Reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of the fonts
> accompanying this license ("Fonts") and associated documentation files (the "Font Software"),
> to reproduce and distribute the Font Software, including without limitation the rights to use,
> copy, merge, publish, distribute, and/or sell copies of the Font Software, and to permit
> persons to whom the Font Software is furnished to do so, subject to the following conditions:
>
> The above copyright and this permission notice shall be included in all copies of one or more
> of the Font Software typefaces.
>
> The Font Software may be modified, altered, or added to, and in particular the designs of
> glyphs or characters in the Fonts may be modified and additional glyphs or characters may be
> added to the Fonts, only if the fonts are renamed to names not containing the word "MgOpen",
> or if the modifications are accepted for inclusion in the Font Software itself by the each
> appointed Administrator.
>
> This License becomes null and void to the extent applicable to Fonts or Font Software that has
> been modified and is distributed under the "MgOpen" name.
>
> The Font Software may be sold as part of a larger software package but no copy of one or more
> of the Font Software typefaces may be sold by itself.
>
> THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL
> MAGENTA OR PERSONS OR BODIES IN CHARGE OF ADMINISTRATION AND MAINTENANCE OF THE FONT SOFTWARE
> BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT,
> INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
> ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN
> THE FONT SOFTWARE.

---

## Vendored source

Each library's own license text is retained inside the vendored files; the paths below point at
where it sits. Everything under `src/external/` is compiled **into the threepp library itself**;
Dear ImGui, under `examples/external/`, builds only the examples and the editor — a project
consuming threepp as a library does not link it.

| Library | License | Copyright | Notice location |
|---|---|---|---|
| [earcut.hpp](src/external/earcut/earcut.hpp) | ISC | © 2015 Mapbox | file header |
| [glad](src/external/glad) (generated GL loader) | generated code, no restrictions; generator MIT © David Herberth | — | `glad.h` header states the generator |
| [khrplatform.h](src/external/glad/KHR/khrplatform.h) | MIT-style (Khronos) | © 2008-2018 The Khronos Group Inc. | file header |
| [GLFW](src/external/glfw) | zlib/libpng | © 2002-2006 Marcus Geelnard, © 2006-2019 Camilla Löwy | [LICENSE.md](src/external/glfw/LICENSE.md); its `deps/` files carry their own notices |
| [meshoptimizer](src/external/meshoptimizer/meshoptimizer.h) | MIT | © 2016-2026 Arseny Kapoulkine | end of file |
| [miniaudio](src/external/miniaudio/miniaudio.h) | your choice of public domain or MIT No Attribution | © 2023 David Reid | end of file |
| [nlohmann/json](src/external/nlohmann/nlohmann/json.hpp) | MIT | © 2013-2023 Niels Lohmann | file header (SPDX) |
| [pugixml](src/external/pugixml/pugixml.hpp) | MIT | © 2006-2023 Arseny Kapoulkine; based on work © 2003 Kristen Wegner | end of file |
| [quickhull](src/external/quickhull/quickhull.hpp) | BSD-style (single retain-notice condition) | © 2014-2015 Anatoliy V. Tomilov | file header |
| [stb](src/external/stb) (`stb_image`, `stb_image_write`, `stb_truetype`) | your choice of public domain (Unlicense) or MIT | © Sean Barrett and contributors | end of each file |
| [Dear ImGui](examples/external/imgui) v1.92.6 — examples + editor only | MIT | © 2014-2024 Omar Cornut | [LICENSE.txt](examples/external/imgui/LICENSE.txt) |

---

## Fetched at build time (FetchContent — not part of this repository)

These are downloaded by CMake when the corresponding option is on. They are listed so a
downstream distributor knows what a *built* threepp contains; consult each upstream for its full
license text.

| Dependency | When | License |
|---|---|---|
| [threepp_data](https://github.com/markaren/threepp_data) | examples or tests | **per-directory** — fonts under the MAGENTA license above, models under their own terms (including CC-BY, CC-BY-**ND** and CC0 entries), and `urdf/spot/` under the **Boston Dynamics SDK License** (see the `LICENSE` and `NOTICE.md` in that directory; simulation use only, not MIT) |
| [tinyusdz](https://github.com/lighttransport/tinyusdz) | `THREEPP_WITH_USD` | Apache-2.0, © Syoyo Fujita / Light Transport Entertainment Inc. Note: `install()` copies its headers and static lib alongside threepp, so an installed tree redistributes it |
| [OpenFBX](https://github.com/nem0/OpenFBX) | FBX loader | MIT, © 2017 Mikulas Florek |
| [glslang](https://github.com/KhronosGroup/glslang) | `THREEPP_WITH_VULKAN` (shader compilation) | mixed BSD-3/Apache-2.0/MIT — see its LICENSE |
| [pybind11](https://github.com/pybind/pybind11) | `THREEPP_WITH_PYTHON` / editor scripting | BSD-3, © Wenzel Jakob |
| [Catch2](https://github.com/catchorg/Catch2) | tests only | Boost Software License 1.0 |
| [RLtools](https://github.com/rl-tools/rl-tools) | `THREEPP_WITH_RLTOOLS` (opt-in example) | MIT, © 2023 Jonas Eschmann |
| AMD FidelityFX SDK (FSR) | `THREEPP_WITH_VULKAN` upscaler | MIT, © AMD — the fetch records the SDK's LICENSE.txt path |
| **NVIDIA DLSS SDK** | `THREEPP_WITH_DLSS` (opt-in) | **"NVIDIA RTX SDKs License" — proprietary, not open source.** The build fetches the SDK and copies `nvngx_dlss.dll` beside the executable; redistribution of that binary is governed by that license, not by anything in this repository |

## Resolved by vcpkg / the system (per feature)

Linked, not redistributed in this repository: NVIDIA **PhysX** (BSD-3) and **V-HACD** (BSD-3)
for the `physx` feature; **Assimp** (BSD-3) for `assimp`; **Vulkan-Headers** (Apache-2.0 OR
MIT), **Vulkan-Loader** (Apache-2.0) and **Vulkan Memory Allocator** (MIT) for `vulkan`;
optionally **GLFW** via vcpkg (zlib/libpng) instead of the vendored copy; **CPython** (PSF
license) when Python or editor scripting is enabled.

---

*When adding a dependency — vendored, fetched, or linked — add it here in the same pass.*
