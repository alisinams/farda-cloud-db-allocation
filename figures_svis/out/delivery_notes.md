# Figure delivery notes

## Fig1
- Script: `fig1_surge.py`; palette: ASE file Ocean Sunset.ase
- Size: 174 mm final (design 8.56 in), print scale 0.816, printed text 9.8–13.1 pt
- QC: overlaps 0 | legends outside | palette OK | 8 pt floor OK | exports verified (6 files)
- Encodes: (a) median over seeds with interquartile range; (b) one point per seed, median (vertical bar) and interquartile range (horizontal bar); n = 30 per cell; log axes
- Exceptions: Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel letters, file stem Fig1
- Files: Fig1_white.pdf, Fig1_white.tiff, Fig1_white.png, Fig1_transparent.pdf, Fig1_transparent.tiff, Fig1_transparent.png

## Fig2
- Script: `fig2_commit_delay.py`; palette: ASE file Ocean Sunset.ase
- Size: 174 mm final (design 8.56 in), print scale 0.816, printed text 9.8–13.1 pt
- QC: overlaps 0 | legends outside | palette OK | 8 pt floor OK | exports verified (6 files)
- Encodes: mean and 95% confidence interval; (a) steady regime, n = 30 seeds per OH value; (b) adversarial regime, n = 10 seeds per OH value; symmetric-log OH axis (linear below 0.01 s)
- Exceptions: Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel letters, file stem Fig2; no legend: single series per panel (SVIS 8, rule 1)
- Files: Fig2_white.pdf, Fig2_white.tiff, Fig2_white.png, Fig2_transparent.pdf, Fig2_transparent.tiff, Fig2_transparent.png

## Fig3
- Script: `fig3_forecast.py`; palette: ASE file Ocean Sunset.ase
- Size: 174 mm final (design 8.56 in), print scale 0.806, printed text 9.7–12.9 pt
- QC: overlaps 0 | legends outside | palette OK | 8 pt floor OK | exports verified (6 files)
- Encodes: MAE mean and SD over the 4 held-out days (rolling origin); shared MAE axis; LightGBM drawn as horizontal reference lines in (b)
- Exceptions: Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel letters, file stem Fig3; per-panel legends above the axes (encodings differ between panels, so no shared legend)
- Files: Fig3_white.pdf, Fig3_white.tiff, Fig3_white.png, Fig3_transparent.pdf, Fig3_transparent.tiff, Fig3_transparent.png

## Fig4
- Script: `fig4_weight_sweep.py`; palette: ASE file Ocean Sunset.ase
- Size: 174 mm final (design 8.56 in), print scale 0.816, printed text 9.8–13.1 pt
- QC: overlaps 0 | legends outside | palette OK | 8 pt floor OK | exports verified (6 files)
- Encodes: means over 3 seeds per triple; dominance on (exposure, latency); shared logarithmic latency axis
- Exceptions: Springer overrides: 174 mm, print scale 0.80 (8 pt floor), lowercase panel letters, file stem Fig4
- Files: Fig4_white.pdf, Fig4_white.tiff, Fig4_white.png, Fig4_transparent.pdf, Fig4_transparent.tiff, Fig4_transparent.png
