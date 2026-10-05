# -*- coding: utf-8 -*-
"""Shared settings for the manuscript figures (one script per figure, SVIS house style).

Journal overrides (Springer Cluster Computing artwork guidelines take precedence over SVIS
for these files only, as SVIS section 0 allows):
  * final width 174 mm (full text width of a large-format Springer journal);
  * lettering must print at 8-12 pt, so the design-to-print reduction is 0.80 instead of
    the SVIS default 0.70 (the 10 pt tier prints at 8.0 pt);
  * figure parts are labelled with lowercase letters (a, b);
  * file stems follow the journal's naming rule ("Fig1", "Fig2", ...).
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sciviz as sv  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTEFACT = HERE.parent
RESULTS = ARTEFACT / "results"
SUMMARY = RESULTS / "summary"
OUT = HERE / "out"

FINAL_WIDTH_MM = 174          # Springer large-format journal, full text width
PRINT_SCALE = 0.80            # 10 pt design text -> 8.0 pt printed (Springer minimum 8 pt)
MIN_PRINTED_PT = 8.0

# One colour index per condition, used identically in every figure (SVIS 4.2, rule 7).
# Indices 3 and 4 of Ocean Sunset (#94d2bd, #e9d8a6) are too light for lines on white,
# so no plotted condition is mapped to them.
GROUP_INDEX = {
    "FARDA (LSTM + MILP)": 0,                       # exact allocator, calibrated LSTM forecast
    "MILP, reactive": 1,                            # exact allocator, last observed value
    "Proportional spreading, no optimiser": 2,
    "PSO, reactive": 5,
    "GA, reactive": 7,
    "ACO, reactive": 9,
    # allocators of the weight sweep (all under the calibrated LSTM forecast)
    "milp": 0, "pso": 5, "ga": 7, "aco": 9,
    # forecasters
    "lstm": 0, "naive": 1, "lightgbm": 6, "seasonal_naive": 8,
}
DISPLAY = {
    "FARDA (LSTM + MILP)": "FARDA: exact allocator, LSTM + margin",
    "MILP, reactive": "Exact allocator, last observed value",
    "Proportional spreading, no optimiser": "Proportional spreading",
    "PSO, reactive": "PSO, last observed value",
    "GA, reactive": "GA, last observed value",
    "ACO, reactive": "ACO, last observed value",
    "milp": "Exact allocator (FARDA)", "pso": "PSO", "ga": "GA", "aco": "ACO",
    "lstm": "LSTM, 64 units", "naive": "Last observed value", "lightgbm": "LightGBM",
    "seasonal_naive": "Seasonal naive",
}


def setup():
    """SVIS setup with the journal print scale; the palette is read from RULES/Ocean Sunset.ase
    when the RULES folder exists above this file, otherwise the identical built-in list is used."""
    sv.MAX_REDUCTION = PRINT_SCALE
    rules = next((p / "RULES" for p in [HERE, *HERE.parents] if (p / "RULES").is_dir()), None)
    out = sv.setup(rules_dir=str(rules) if rules else None)
    pal = sv.active_palette()
    if pal.source.startswith("ASE file "):        # record the palette file name only, not a local path
        pal.source = "ASE file " + Path(pal.source[len("ASE file "):]).name
    return out


def style(pal, key, alpha=1.0):
    i = GROUP_INDEX[key]
    return pal.series(i, alpha)


def edge_ink(png_path, border_px=3):
    """Return the sides on which non-white pixels touch the image border (tight export
    keeps a 0.05 in white pad, so ink on the border means a clipped label or line)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("L"))
    sides = {"top": a[:border_px], "bottom": a[-border_px:], "left": a[:, :border_px], "right": a[:, -border_px:]}
    return [s for s, px in sides.items() if (px < 250).any()]


def finish(fig, stem, axes=None, encodes="", exceptions=()):
    """Panel letters, strict QC (SVIS + journal 8 pt floor), six exports, delivery note."""
    if axes is not None:
        sv.panel_labels(fig, axes, lowercase=True)
    sv.qc_figure(fig, final_width_mm=FINAL_WIDTH_MM, strict=True)
    small, scale = sv.audit_min_font(fig, FINAL_WIDTH_MM, min_pt=MIN_PRINTED_PT)
    if small:
        raise RuntimeError("; ".join(small))
    paths = sv.export_figure(fig, stem, OUT)
    clipped = edge_ink(OUT / f"{stem}_white.png")
    if clipped:
        raise RuntimeError(f"{stem}: ink touches the image border ({clipped}); something is clipped")
    fig.canvas.draw()
    fonts = [t.get_fontsize() for _, _, t in sv._visible_texts(fig)]
    note = {
        "figure": stem, "script": os.path.basename(__import__("inspect").stack()[1].filename),
        "palette": sv.active_palette().source, "final_width_mm": FINAL_WIDTH_MM,
        "design_width_in": round(fig.get_figwidth(), 2), "print_scale": round(scale, 3),
        "min_text_pt": round(min(fonts) * scale, 1), "max_text_pt": round(max(fonts) * scale, 1),
        "qc": "overlaps 0 | legends outside | palette OK | 8 pt floor OK | exports verified (6 files)",
        "encodes": encodes, "exceptions": list(exceptions), "files": [p.name for p in paths],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}_delivery.json").write_text(json.dumps(note, indent=1), encoding="utf-8")
    return note
