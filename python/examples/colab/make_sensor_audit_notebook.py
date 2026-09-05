"""Generate threepp_sensor_audit.ipynb: the RA-L paper's cross-GPU determinism row.

The notebook installs threepp on a Colab T4, runs examples/sensor_audit.py in
two fresh subprocesses, compares them (the "fresh process, T4" cells of the
reproducibility matrix), and compares against a manifest recorded on the
development machine (RTX 4070, Windows) for the cross-machine row.

    python make_sensor_audit_notebook.py [--baseline rtx4070.json]

Regenerate rather than hand-edit: the audit script is embedded verbatim from
../sensor_audit.py and the setup cell is shared with threepp_colab.ipynb.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(HERE, "sensor_audit_rtx4070.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "threepp_sensor_audit.ipynb"))
    a = ap.parse_args()

    base_nb = json.load(open(os.path.join(HERE, "threepp_colab.ipynb"), encoding="utf-8"))
    setup_cell = "".join(base_nb["cells"][2]["source"])
    # A release candidate sits on TestPyPI before it reaches PyPI, and the T4
    # column is measured from whichever the form flag names. Dependencies still
    # resolve from PyPI through the extra index; the manifest records the
    # installed version either way (sensor_audit.py writes threepp.__version__).
    pip_line = 'sh(f"{sys.executable} -m pip -q install threepp")'
    assert pip_line in setup_cell, "threepp_colab.ipynb setup cell changed; update the pip anchor"
    # Not an extra index: TestPyPI carries dev versions (0.1.0.devN) that sort
    # BELOW the calendar releases on PyPI under PEP 440, so a resolver that can
    # see both always picks PyPI, and dev releases are skipped without --pre
    # anyway. The one dependency comes from PyPI first; threepp itself then
    # comes from TestPyPI alone, pre-releases allowed, replacing whatever an
    # earlier cell run left installed.
    setup_cell = setup_cell.replace(
        pip_line,
        'if USE_TESTPYPI:\n'
        '    sh(f"{sys.executable} -m pip -q install numpy")\n'
        '    sh(f"{sys.executable} -m pip -q install --pre --no-deps --force-reinstall "\n'
        '       "--index-url https://test.pypi.org/simple/ threepp")\n'
        'else:\n'
        '    sh(f"{sys.executable} -m pip -q install threepp")\n'
        'print(sh(f"{sys.executable} -c \\"import threepp; print(\'threepp\', threepp.__version__, threepp.__file__)\\""))', 1)
    form_anchor = '{ display-mode: "form" }\n'
    assert form_anchor in setup_cell, "setup cell lost its form title line"
    setup_cell = setup_cell.replace(
        form_anchor,
        form_anchor + 'USE_TESTPYPI = False  # @param {type:"boolean"}  (a release candidate on TestPyPI)\n', 1)
    audit_src = open(os.path.join(HERE, "..", "sensor_audit.py"), encoding="utf-8").read()
    baseline = open(a.baseline, encoding="utf-8").read().strip() if os.path.exists(a.baseline) else None

    cells = [
        md("""# threepp sensor replay audit - is the sensor stack bit-reproducible on *this* GPU?

Companion to the paper *Measured Reproducibility of a Same-Instant Multi-Modal
Synthetic Sensor Stack*. The paper measures, per modality, whether a simulated
sensor stream replays bit for bit across fresh processes. This notebook runs the
same measurement on the GPU in front of you and fills in two cells of the
paper's reproducibility matrix:

* **fresh process, same GPU** - the audit script runs twice in two fresh Python
  processes; every stream hash must match.
* **different GPU** - the same run compared against a manifest recorded on the
  development machine (RTX 4070, Windows). Rasterized labels and the
  proprioceptive streams are the interesting rows: a match across GPUs and
  operating systems is a much stronger statement than a match across processes.

Streams: PhysX IMU (seeded noise on) and contact; raster depth / normals /
instance ids / motion / albedo; the rendered frame; path-traced VLP-16 lidar;
imaging-sonar echograms.

Before you start: **Runtime > Change runtime type > T4 GPU** (T4 / L4 carry RT
cores; A100 / V100 / H100 do not). Total runtime about 8 minutes.
"""),
        md("""## 1 - Setup (~2 min)

Same setup cell as the threepp Colab tour: matched driver, Xvfb, `pip install threepp`,
and a `run()` helper that executes every GPU script in a **fresh subprocess** - which
is exactly the replay condition the audit measures.
"""),
        code(setup_cell),
        md("""## 2 - The audit script

Written to disk verbatim from the repository (`python/examples/sensor_audit.py`).
It builds the scripted scene, drives every clock from the frame index, hashes
each stream over 120 frames, and writes a JSON manifest. Rows the installed wheel
cannot produce are reported as `absent`, never as a match.
"""),
        code("%%writefile sensor_audit.py\n" + audit_src),
        md("""## 3 - Fresh process x 2 on this GPU (~4 min)

Two runs, two processes, one compare. `OK` rows are bit-identical; a `DIFF` row
is a finding, not a failure of the notebook - the paper's method is to report it.
"""),
        code("""run("sensor_audit.py", "--frames", 120, "--out", "t4_a.json")
run("sensor_audit.py", "--frames", 120, "--out", "t4_b.json")
run("sensor_audit.py", "--compare", "t4_a.json", "t4_b.json")
"""),
        md("""## 4 - Against the development machine (RTX 4070, Windows)

The manifest below was recorded with the same script on the machine the paper's
C++ harnesses ran on. Rows that match across two GPUs, two operating systems and
two compilers are reproducible in the strongest sense this method can test; rows
that differ tell you where device-specific arithmetic enters the stream.
"""),
        code("%%writefile rtx4070.json\n" + (baseline if baseline else '{"meta": {"note": "baseline not recorded yet"}, "rows": {}}')),
        code("""run("sensor_audit.py", "--compare", "rtx4070.json", "t4_a.json")
"""),
        md("""## 5 - Reading the result

* `prop.*` bit-identical across machines means CPU PhysX plus the seeded noise
  streams are platform-independent for this scene: same bits on Linux/GCC and
  Windows/MSVC.
* `aov.*` rows across machines test whether rasterization of the same scene at
  the same resolution is bit-identical across GPU architectures. The paper's
  claim is per device; a cross-device match is a bonus, a mismatch is expected.
* `lidar` / `sonar` across machines test hardware ray tracing across
  architectures. Same caveat.
* Within one machine (section 3), every row is expected `OK`. If one is not,
  please open an issue with both manifests attached - that is a result the paper
  wants to know about.
"""),
    ]
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", a.out, "baseline embedded" if baseline else "NO baseline yet")


if __name__ == "__main__":
    main()
