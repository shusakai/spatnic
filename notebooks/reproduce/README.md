# Paper reproduction — per-figure / per-table notebooks

Each notebook inlines the **actual generating code** for one paper figure/table (copied from the source
notebooks/scripts), and embeds the published result for reference. **Edit the path roots in the first
config cell** of any notebook before running. Cost tags: 🟢 light (reads caches) · 🟡 moderate · 🔴 heavy (GPU / multi-GB / long).

## Figures

- [`Fig1.ipynb`](Fig1.ipynb) — Fig 1 — cancer/normal classification across primary & metastatic CRC
- [`Fig2.ipynb`](Fig2.ipynb) — Fig 2 — accuracy & compute efficiency on Xenium
- [`Fig3.ipynb`](Fig3.ipynb) — Fig 3 — zero-shot generalization across platforms
- [`Fig4.ipynb`](Fig4.ipynb) — Fig 4 — POU2F3+ tuft-like cancer population & relapse
- [`Fig5.ipynb`](Fig5.ipynb) — Fig 5 — patient-specific ADC target screening (Visium HD)
- [`SuppFig1.ipynb`](SuppFig1.ipynb) — Supp Fig 1 — top discriminative genes
- [`SuppFig2.ipynb`](SuppFig2.ipynb) — Supp Fig 2 — strict-confidence mode
- [`SuppFig3.ipynb`](SuppFig3.ipynb) — Supp Fig 3 — architecture ablation (Student-t)
- [`SuppFig4.ipynb`](SuppFig4.ipynb) — Supp Fig 4 — representative spatial vs other tools
- [`SuppFig5.ipynb`](SuppFig5.ipynb) — Supp Fig 5 — matches/exceeds tools (Visium & scRNA-seq)
- [`SuppFig6.ipynb`](SuppFig6.ipynb) — Supp Fig 6 — zero-shot to Visium / Visium HD / pseudo-Xenium
- [`SuppFig7.ipynb`](SuppFig7.ipynb) — Supp Fig 7 — scRNA-seq labels agree + deconvolution-consistent
- [`SuppFig8.ipynb`](SuppFig8.ipynb) — Supp Fig 8 — FP/FN calls localize to distinct clusters
- [`SuppFig9.ipynb`](SuppFig9.ipynb) — Supp Fig 9 — FP-specific POU2F3+/ASCL2+ tuft-like (Hippo/Wnt/TGF-β)
- [`SuppFig10.ipynb`](SuppFig10.ipynb) — Supp Fig 10 — tuft-like signature stratifies relapse (TCGA)
- [`SuppFig11.ipynb`](SuppFig11.ipynb) — Supp Fig 11 — full ADC-target dot-plot (all antigens)

## Tables

- [`Table1.ipynb`](Table1.ipynb) — Table 1 — datasets (train + test)
- [`Table2.ipynb`](Table2.ipynb) — Table 2 — pooled performance (SPATNIC)
- [`SuppTable1.ipynb`](SuppTable1.ipynb) — Supp Table 1 — per-sample performance (SPATNIC)
- [`SuppTable2.ipynb`](SuppTable2.ipynb) — Supp Table 2 — pooled AUROC (model × test)
- [`SuppTable3.ipynb`](SuppTable3.ipynb) — Supp Table 3 — per-sample AUROC (model × test)
- [`SuppTable4.ipynb`](SuppTable4.ipynb) — Supp Table 4 — overall benchmark (SPATNIC vs tools)
- [`SuppTable5.ipynb`](SuppTable5.ipynb) — Supp Table 5 — per-sample benchmark
- [`SuppTable6.ipynb`](SuppTable6.ipynb) — Supp Table 6 — architecture ablation (pooled)
- [`SuppTable7.ipynb`](SuppTable7.ipynb) — Supp Table 7 — architecture ablation (per-sample)
- [`SuppTable8.ipynb`](SuppTable8.ipynb) — Supp Table 8 — curated ADC target panel

## Excluded (non-code) panels

Schematics and H&E/IHC/Xenium micrographs are not reproduced: Fig 1a/1d, Fig 4b, Fig 5a/5c/5d, Supp Fig 1b.
