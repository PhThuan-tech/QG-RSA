# ACL version of the paper

- `main.tex`: ACL long paper, review mode (`\usepackage[review]{acl}`); switch to `[final]` and fill the authors for camera-ready.
- `acl.sty`, `acl_natbib.bst`: official files from github.com/acl-org/acl-style-files; `acl_template_example.tex` is the unmodified official example.
- `sections/`: one file per section; `references.bib`.
- `figures/`: `pipeline.pdf`, `frontier.pdf`, `comparison.pdf`, `probe_auc.pdf` from `python paper_acl/make_figures.py`; `delta_per_stage.pdf`, `accuracy_curves.pdf` from `paper/make_figures.py`. `*_preview.png` are previews only.
- Published numbers of other methods (Table 2, Figure comparison, Table exemplar) are copied from Tables 1-2 of the RSIAT paper (CVPR 2026 open access); they use the authors' partitions and one run.
- Overleaf: upload `dist/paper_acl_overleaf.zip` (New Project > Upload Project) or this whole folder, set `main.tex` as the main document, compiler pdfLaTeX.
- Every number comes from `exps/*/results/`. Open items are marked `\todo{...}` (red in the PDF).
