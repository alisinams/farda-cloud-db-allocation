# -*- coding: utf-8 -*-
"""Rebuild every manuscript figure (strict SVIS QC + journal checks) and write out/delivery_notes.md."""
import importlib
import json

import fig_config as fc

SCRIPTS = ["fig1_surge", "fig2_commit_delay", "fig3_forecast", "fig4_weight_sweep"]


def main():
    notes = []
    for name in SCRIPTS:
        mod = importlib.import_module(name)
        notes.append(mod.main())
        import matplotlib.pyplot as plt
        plt.close("all")
    lines = ["# Figure delivery notes", ""]
    for n in notes:
        lines += [f"## {n['figure']}",
                  f"- Script: `{n['script']}`; palette: {n['palette']}",
                  f"- Size: {n['final_width_mm']} mm final (design {n['design_width_in']} in), print scale "
                  f"{n['print_scale']}, printed text {n['min_text_pt']}–{n['max_text_pt']} pt",
                  f"- QC: {n['qc']}",
                  f"- Encodes: {n['encodes']}",
                  f"- Exceptions: {'; '.join(n['exceptions']) or 'none'}",
                  f"- Files: {', '.join(n['files'])}", ""]
    (fc.OUT / "delivery_notes.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([n["figure"] for n in notes]))


if __name__ == "__main__":
    main()
