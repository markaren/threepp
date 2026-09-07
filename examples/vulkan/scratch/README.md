# Local scratch harnesses

Drop throwaway capture / repro / profiling executables here as `*.cpp`. Each file
becomes its own example target automatically — the parent `CMakeLists.txt` globs
`scratch/*.cpp` — so you never edit a tracked build file to add or remove one.
The target name is the file stem (e.g. `scratch/vulkan_aaa_capture.cpp` →
`cmake --build <dir> --target vulkan_aaa_capture`).

Everything in this directory is gitignored **except this README**, so scratch
harnesses, their configs, and local build scripts stay out of the project's
history. Render outputs go to `<project>/aaa_caps/` (also gitignored).

The split is deliberate:

- **Capability → committed.** Durable capture machinery — config-driven camera,
  the `--profile` frame-timing dump, image diff (MSE/SSIM) — lives in the tracked
  `examples/vulkan/capture_util.hpp`, so both the real demos and these scratch
  tools share one implementation.
- **Repro → gitignored.** The specific scene, camera pose, and flags you are
  poking at this week belong here, never in `main`'s history.

```cpp
// scratch/my_repro.cpp  ->  builds as target `my_repro`
#include "threepp/threepp.hpp"
#include "../capture_util.hpp"   // shared, committed harness machinery
int main(int argc, char** argv) { /* ... */ }
```
