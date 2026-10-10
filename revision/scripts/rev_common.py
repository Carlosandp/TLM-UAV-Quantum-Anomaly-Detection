"""Shared loader for the revision analyses (manuscript AESM-2026-0104).

Re-executes, unmodified, the data-construction, proxy-audit, helper, split and
per-seed preparation cells of notebooks/TLM_DRU_FINAL.ipynb, so that every
revision analysis starts from exactly the objects used for the stored results.

Paths (all overridable through environment variables):
  TLM_UAV_ROOT        repository root (default: two levels above this file)
  TLM_DATASET_DIR     folder with the raw TLM:UAV sensor folders (ATT/, BARO/, ...)
                      (default: <root>/data/dataset)
  TLM_STORED_RESULTS  stored per-run results of the main study (default: <root>/results)
  TLM_REV_DIR         output folder of the revision analyses (default: <root>/revision/results)
  TLM_REV_FIG_DIR     output folder of the revision figures (default: <root>/revision/figures)
The notebook's own side outputs (log, EDA files) go to <TLM_REV_DIR>/_nb_scratch
and never overwrite the stored results.
"""
import os, json
from pathlib import Path

ROOT = Path(os.environ.get('TLM_UAV_ROOT', Path(__file__).resolve().parents[2]))
STORED = Path(os.environ.get('TLM_STORED_RESULTS', ROOT / 'results'))
REV_DIR = Path(os.environ.get('TLM_REV_DIR', ROOT / 'revision' / 'results'))
FIG_DIR = Path(os.environ.get('TLM_REV_FIG_DIR', ROOT / 'revision' / 'figures'))
NB = Path(os.environ.get('TLM_NOTEBOOK', ROOT / 'notebooks' / 'TLM_DRU_FINAL.ipynb'))
SCRATCH = REV_DIR / '_nb_scratch'
for d in (REV_DIR, FIG_DIR, SCRATCH): d.mkdir(parents=True, exist_ok=True)
os.environ['TLM_UAV_ROOT'] = str(ROOT)
os.environ['TLM_RESULT_DIR'] = str(SCRATCH)   # keeps the stored results and log intact
os.environ.setdefault('MPLBACKEND', 'Agg')


def load_notebook_namespace(cells=(2, 4, 10, 11, 13, 15, 17)):
    nb = json.load(open(NB))
    ns = {'__name__': 'nb', 'display': lambda *a, **k: None}
    for i in cells:
        src = ''.join(nb['cells'][i]['source'])
        if i == 4:  # keep the dataset build, drop the EDA plots that follow it
            src = src.split('# ── EDA básico')[0]
        exec(compile(src, f'<nb cell {i}>', 'exec'), ns)
    return ns


def confusion_rates(y_true, y_pred):
    """Fault = positive class (1). Conventional rates.

    far  = nominal samples flagged as faults / all nominal samples   (FP/(FP+TN))
    miss = fault samples labeled nominal     / all fault samples     (FN/(FN+TP))
    """
    import numpy as np
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                far=fp / (fp + tn) if (fp + tn) else float('nan'),
                miss=fn / (fn + tp) if (fn + tp) else float('nan'))
