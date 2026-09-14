"""Tile the sea-drone co-design pilot into ONE png.

Nothing is computed here. The inputs are what `optimize.py --selftest`,
`warp_prop_vortex.py --design ... --design-op sprint` and
`codesign.py --selftest --shots out/shots` already wrote; this script reads
their JSON for the captions and stacks the four images:

    top     the propeller pilot at the 7 m/s sprint, iteration 0 vs optimised
            (rendered by the prop-vortex demo under similitude)
    bottom  the co-designed hull, three views with the disc to scale,
            step 0 (the blob) and after 2000 coupled steps

    python python/examples/drone_codesign/figure.py            # -> doc/screenshots/drone_codesign.png
    python python/examples/drone_codesign/figure.py --out x.png

Regenerate the inputs first if they are missing:
    python python/examples/drone_codesign/optimize.py --selftest
    python python/examples/warp_prop_vortex.py --shot 4 --design python/examples/drone_codesign/out/design_iter0000.json --out python/examples/drone_codesign/out/drone_sprint_iter0000.png
    python python/examples/warp_prop_vortex.py --shot 4 --design python/examples/drone_codesign/out/design_final.json    --out python/examples/drone_codesign/out/drone_sprint_final.png
    python python/examples/drone_codesign/codesign.py --selftest --shots python/examples/drone_codesign/out/shots
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_PNG = os.path.join(ROOT, "doc", "screenshots", "drone_codesign.png")

W = 1800                       # figure width, px
PAD = 24
BG = (250, 250, 250)
INK = (30, 30, 30)
DIM = (105, 105, 105)


def font(px, mono=False):
    for name in (("consola.ttf",) if mono else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def load_json(*parts):
    with open(os.path.join(OUT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def fit(img, width):
    h = int(round(img.height * width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_PNG)
    args = ap.parse_args()

    p0, p1 = load_json("design_iter0000.json"), load_json("design_final.json")
    h0, h1 = load_json("shots", "design_iter0000.json"), load_json("shots", "design_final.json")
    prop_imgs = [Image.open(os.path.join(OUT, n)).convert("RGB")
                 for n in ("drone_sprint_iter0000.png", "drone_sprint_final.png")]
    hull_imgs = [Image.open(os.path.join(OUT, "shots", n)).convert("RGB")
                 for n in ("hull_iter0000.png", "hull_final.png")]

    f_title, f_head, f_cap, f_note = font(30), font(24), font(21, mono=True), font(19)

    def prop_cap(d, tag):
        return (f"{tag}   D {d['D']:.2f} m   P/D {d['PD']:.2f}   "
                f"{d['n_sprint'] * 60:.0f} rpm at the {d['V_sprint']:.0f} m/s sprint   "
                f"Burrill {d['burrill_ratio_sprint']:.2f}"
                f"{' (cavitating)' if d['burrill_ratio_sprint'] > 1.02 else ' (at inception)'}   "
                f"range {d['range_km']:.0f} km at {d['V_survey']:.0f} m/s")

    def hull_cap(d, tag):
        return (f"{tag}   D {d['D']:.2f} m   P/D {d['PD']:.2f}   "
                f"clearance {d['clearance']:+.2f} m (needs {d['clear_req']:.2f})   "
                f"R_T {d['R_T_survey']:.0f} N at {d['V_survey']:.0f} m/s   "
                f"S_wet {d['S_wet']:.1f} m2   Burrill(sprint) {d['burrill_ratio_sprint']:.2f}   "
                f"range {d['range_km']:.1f} km")

    title = ("Sea-drone co-design pilot: gradient descent through a differentiable "
             "propeller and hull model in Warp, rendered with threepp")
    head_top = ("Propeller alone, fixed hull: iteration 0 (left) and the optimum (right), "
                "rendered at the demo's 1.8 m scale under similitude (J, P/D and the "
                "cavitation number matched)")
    head_bot = ("Hull and screw on one tape: the blob at step 0 and the hull after 2000 "
                "coupled steps, three views, disc to scale, waterline dashed")
    scope = ("Scope. Propeller: Wageningen B5-75 regression, momentum disc, Burrill cavitation, "
             "self-propulsion solved by Newton on the tape. Hull: the warp_hull_sculpt model, "
             "Newtonian form drag + ITTC friction + Michell wave resistance, constant wake fraction "
             "and thrust deduction. Gates: central finite differences on every gradient, and Adam "
             "must reproduce a brute-force grid optimum. What it does not show: at this brief the "
             "hull-to-screw coupling ended slack, so the bottom row's range gain is the hull "
             "sculpt's own; the screw grew to its diameter limit on a strut below the keel. A worked "
             "example of differentiable design with gradient gates, not a validated design tool.")

    # -- layout ---------------------------------------------------------------
    col_w = (W - 3 * PAD) // 2
    prop_imgs = [fit(im, col_w) for im in prop_imgs]
    hull_imgs = [fit(im, W - 2 * PAD) for im in hull_imgs]
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    scope_lines = wrap(probe, scope, f_note, W - 2 * PAD)
    line_h = 26

    y = PAD
    y_title = y
    y += 44
    y_head_top = y
    y += 36 + 6
    y_prop = y
    y += prop_imgs[0].height + 8
    y_prop_cap = y
    y += 2 * line_h + 22
    y_head_bot = y
    y += 36 + 6
    y_hull = []
    y_hull_cap = []
    for im in hull_imgs:
        y_hull.append(y)
        y += im.height + 4
        y_hull_cap.append(y)
        y += line_h + 14
    y_scope = y + 6
    y += 6 + len(scope_lines) * line_h + PAD
    H = y

    fig = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(fig)
    d.text((PAD, y_title), title, font=f_title, fill=INK)
    d.text((PAD, y_head_top), head_top, font=f_head, fill=DIM)
    for i, im in enumerate(prop_imgs):
        x = PAD + i * (col_w + PAD)
        fig.paste(im, (x, y_prop))
    d.text((PAD, y_prop_cap), prop_cap(p0, "iteration 0   "), font=f_cap, fill=INK)
    d.text((PAD, y_prop_cap + line_h), prop_cap(p1, f"iteration {p1['iteration']}"),
           font=f_cap, fill=INK)
    d.text((PAD, y_head_bot), head_bot, font=f_head, fill=DIM)
    for im, yy, yc, dd, tag in zip(hull_imgs, y_hull, y_hull_cap, (h0, h1),
                                    ("step 0        ", "co-design final")):
        fig.paste(im, (PAD, yy))
        d.text((PAD, yc), hull_cap(dd, tag), font=f_cap, fill=INK)
    for k, ln in enumerate(scope_lines):
        d.text((PAD, y_scope + k * line_h), ln, font=f_note, fill=DIM)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.save(args.out)
    print(f"wrote {args.out}  ({W} x {H})")


if __name__ == "__main__":
    main()
