"""Collect experiment results into site/data.js and copy figures into site/img/.

Run after any new results:  python3 build.py
Only uses the Python standard library.
"""
import datetime
import glob
import json
import os
import shutil
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEREO = os.path.join(ROOT, "stereo")
CONDS = ["clean", "fast", "flicker", "texture", "distractor"]


def load(pattern):
    out = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def r1(x):
    return None if x is None else round(x, 2)


# ------------------------------------------------------------------ 2D experiment
def exp2d():
    groups = defaultdict(list)
    for r in load(os.path.join(ROOT, "results", "*.json")):
        m = r["method"] + ("+mixed" if r.get("train_cond") == "mixed" else "")
        groups[(m, r.get("k"))].append(r)

    def err(m, k, cond, h="100ms"):
        rs = groups.get((m, k))
        return r1(mean(x["test"][cond][h] for x in rs)) if rs else None

    def row(id_, label, m, k, readings):
        if (m, k) not in groups:
            return None
        d = {"id": id_, "label": label, "readings": readings,
             "seeds": len(groups[(m, k)])}
        for c in CONDS:
            d[c] = err(m, k, c)
        d["clean_10ms"] = err(m, k, "clean", "10ms")
        return d

    rows = [
        row("frames", "Full frames + CNN", "frames", None, 4096),
        row("frames_mixed", "Full frames + CNN, trained on varied scenes", "frames+mixed", None, 4096),
        row("diff", "Frame differences + CNN", "diff", None, 4096),
        row("events", "Event camera + CNN", "events", None, 8192),
        row("kalman", "Classical tracker (Kalman filter)", "kalman", None, 4096),
        row("mask16", "16 learned masks", "mask", 16, 16),
        row("random16", "16 random masks", "random_mask", 16, 16),
        row("norm16", "16 learned masks + total-light sensor", "mask_norm", 16, 17),
        row("norm64_mixed", "64 learned masks + total-light sensor, varied training",
            "mask_norm+mixed", 64, 65),
    ]
    curve = []
    for k in sorted({k for (m, k) in groups if m == "mask"}):
        curve.append({"k": k, "learned": err("mask", k, "clean"), "random": err("random_mask", k, "clean")})
    return {"rows": [r for r in rows if r], "curve": curve}


# ------------------------------------------------------------------ 3D stereo experiment
def exp3d():
    groups = defaultdict(list)
    for r in load(os.path.join(STEREO, "results", "*.json")):
        groups[(r["setup"], r.get("readings"), r["size"], r["baseline"])].append(r)

    def m(setup, readings, size, metric, b=0.1):
        rs = groups.get((setup, readings, size, b))
        if not rs:
            return None
        return r1(mean(x["test"][x["size"]][metric] for x in rs))

    labels = [("classical_mono", None, "Classical, 1 eye (distance from size)"),
              ("classical_stereo", None, "Classical, 2 eyes (triangulation)"),
              ("mono_frames", None, "CNN, 1 eye"), ("stereo_frames", None, "CNN, 2 eyes"),
              ("mono_mask", 32, "32 masks, 1 eye"),
              ("sep_mask", 32, "32 masks split over 2 eyes"),
              ("bino_mask", 32, "32 binocular masks (each sees both eyes)")]
    rows = []
    for s, k, lab in labels:
        f, rnd = m(s, k, "fixed", "depth_100ms"), m(s, k, "random", "depth_100ms")
        if f is None or rnd is None:
            continue
        rows.append({"id": s + (f"_{k}" if k else ""), "label": lab, "fixed": f, "random": rnd,
                     "random_lateral": m(s, k, "random", "lateral_100ms"),
                     "near": m(s, k, "random", "depth_100ms_z1.0-1.7"),
                     "far": m(s, k, "random", "depth_100ms_z2.3-3.0")})

    sep = [{"label": "1 eye", "value": m("mono_frames", None, "random", "depth_100ms")}]
    for b in (0.03, 0.1, 0.3):
        v = m("stereo_frames", None, "random", "depth_100ms", b)
        if v is not None:
            sep.append({"label": f"2 eyes, {b * 100:.0f} cm apart", "value": v})

    long_rows = []
    for r in load(os.path.join(STEREO, "results_long", "*.json")):
        long_rows.append({"setup": r["setup"], "readings": r["readings"],
                          "depth": r1(r["test"]["random"]["depth_100ms"])})

    probe = []
    for s, lab in (("ssl_mono", "1 eye"), ("ssl_stereo", "2 eyes")):
        rs = groups.get((s, None, "random", 0.1))
        if rs:
            p, u = rs[0]["probe"], rs[0]["probe_untrained"]
            probe.append({"label": lab, "trained": r1(p["depth_now"]["r2"]),
                          "untrained": r1(u["depth_now"]["r2"])})
    rs = groups.get(("stereo_frames", None, "random", 0.1))
    supervised_r2 = r1(rs[0]["probe"]["depth_now"]["r2"]) if rs else None
    return {"rows": rows, "separation": sep, "long": long_rows, "probe": probe,
            "supervised_stereo_r2": supervised_r2}


def copy_images():
    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    for src, name in [(os.path.join(ROOT, "learned_masks.png"), "learned_masks.png"),
                      (os.path.join(STEREO, "binocular_masks_16k.png"), "binocular_masks_16k.png"),
                      (os.path.join(STEREO, "binocular_masks.png"), "binocular_masks_4k.png")]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, "img", name))


def main():
    data = {"updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "exp2d": exp2d(), "exp3d": exp3d()}
    with open(os.path.join(HERE, "data.js"), "w") as f:
        f.write("// generated by build.py, do not edit\nwindow.DATA = ")
        json.dump(data, f, indent=1)
        f.write(";\n")
    copy_images()
    print(f"wrote data.js ({len(data['exp2d']['rows'])} 2D rows, "
          f"{len(data['exp3d']['rows'])} 3D rows), updated {data['updated']}")


if __name__ == "__main__":
    main()
