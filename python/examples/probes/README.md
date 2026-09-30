# probes: instruments, not demos

Each script here answers one question and prints (or writes) the answer. None of them is a
showcase; each is a measurement that a claim somewhere else rests on.

| Script | What it measures |
| --- | --- |
| `smoke_test.py` | Regression smoke of the Python bindings: math types, mutate-in-place, scene graph, STL load, a headless render. Prints `ALL OK`. |
| `idun_egl_smoke.py` | Hardware OpenGL through EGL on a node with no display, and that the GPU (not a software rasteriser) drew it. Exit 0 = pass; `scripts/idun/render_warp_demo.slurm` runs it as a preflight gate. |
| `sensor_audit.py` | Whether every sensor stream of one scripted scene replays bit for bit across fresh processes and GPUs (the paper's replay audit, against the pip wheel). |
| `multiview_bench.py` | E2: the same-instant multi-view triptych and the frame-time curve for 1 to 8 views against sequential renders. |
| `tendon_probe.py` | What PhysX articulation tendons actually do on a two-link finger; the numbers behind `TendonCable`. |

## Run

    python python/examples/probes/smoke_test.py
    python python/examples/probes/idun_egl_smoke.py --save frame.png
    python python/examples/probes/sensor_audit.py --frames 120 --out a.json
    python python/examples/probes/sensor_audit.py --compare a.json b.json     # exit 0 = bit-identical
    python python/examples/probes/multiview_bench.py --out e2/
    python python/examples/probes/tendon_probe.py --only E4

`smoke_test.py` and `idun_egl_smoke.py` put `python/` on `sys.path` and so run the in-tree build;
the other three import whatever `threepp` is installed (the wheel, an editable install, or
`PYTHONPATH=python`). `sensor_audit.py` is also a module: `../netpen/warp_netpen.py` imports it
for its manifest format, and `../colab/make_sensor_audit_notebook.py` embeds it verbatim in the
Colab notebook, so a change here changes both.
