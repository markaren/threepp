# threepp-editor

The [threepp](https://github.com/markaren/threepp) scene editor as a pip
package: author physics-ready 3D scenes — rigid bodies, joints, vehicles,
robots, sensors, procedural trees, splines — press Play to simulate them, and
attach Python scripts that drive objects MonoBehaviour-style.

```sh
pip install threepp-editor
threepp-editor                 # or: python -m threepp_editor
```

Open an example straight from the binary:

```sh
threepp-editor --example=hover-arena --play
```

`threepp-editor --help` lists the full flag set (screenshot passes, benches,
the acceptance self-test).

## What's in the wheel

- The editor with the **GL viewport**. A source build with
  `-DTHREEPP_WITH_VULKAN=ON` adds the Vulkan view pane; the wheel deliberately
  omits it so installing never requires a Vulkan runtime.
- **CPU PhysX** play sessions: rigid bodies, joints, vehicles work out of the
  box. Soft bodies need GPU dynamics — drop `PhysXGpu_64.dll` +
  `PhysXDevice64.dll` into `threepp_editor/bin/` next to the executable.
- **Python scripting**: scripts attached to scene objects run in an embedded
  interpreter that shares one type registry with the play session. The
  launcher wires the embedded interpreter to *your* Python — the venv you
  installed into — so editor scripts can import your packages.
- The **union type stubs** (`threepp` + the editor-only physics handles) that
  the editor's "Edit in VS Code" flow hands to Pylance.

## Relationship to the `threepp` package

Independent — install either or both. The editor embeds its own bindings, so
editor scripts see `threepp` (and `threepp.editor` with the live-session
physics handles) without the wheel being installed. The `threepp` wheel is the
same API for *your own* programs: headless rendering, AOVs, physics, splats.
Scenes saved here load there (`threepp.editor.spline_from_object` and friends
read what the editor authors).

## Per-interpreter wheels

The editor embeds CPython, so each wheel is bound to one Python minor version
(the executable links `python3XX.dll`). Install with the interpreter you plan
to script with.
