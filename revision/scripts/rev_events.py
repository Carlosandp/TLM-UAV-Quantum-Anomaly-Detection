"""A7  Fault-event analysis of the stored blocked partitions.

The table holds four contiguous anomaly intervals (one per fault type), each
spanning two adjacent blocks. For every partition we separate test anomaly rows
whose fault interval also contributes rows to training ("event partly seen")
from those whose interval is absent from training ("event unseen") and report
the detection rate of the classical baselines on each group.
"""
import warnings; warnings.filterwarnings('ignore')
import io, contextlib
import numpy as np, pandas as pd
from rev_common import load_notebook_namespace, REV_DIR
with contextlib.redirect_stdout(io.StringIO()):
    ns = load_notebook_namespace()
y_raw = ns['y_raw']; rows = []; comp = []
for s in ns['SEEDS_B2_BINARY']:
    S = ns['B2_SPLITS'][s]; lab_tr = set(y_raw[S['idx_train']].tolist()) - {0}; lab_te = y_raw[S['idx_test']]
    comp.append(dict(seed=s, fault_types_in_train=sorted(lab_tr), fault_types_in_test=sorted(set(lab_te.tolist()) - {0}),
                     test_anomaly_rows=int((lab_te > 0).sum()), test_anomaly_rows_event_unseen=int(((lab_te > 0) & ~np.isin(lab_te, list(lab_tr))).sum())))
    for mode in ('full', 'no_proxy_strict'):
        prep = ns['B2_PREPARED'][(s, mode)]
        for name, mdl in ns['make_classical_models'](s).items():
            mdl.fit(prep['X_train_bal'], prep['y_train_bal']); yp = mdl.predict(prep['X_test_sc'])
            for grp, mk in (('event partly seen in training', (lab_te > 0) & np.isin(lab_te, list(lab_tr))), ('event unseen in training', (lab_te > 0) & ~np.isin(lab_te, list(lab_tr)))):
                if mk.sum(): rows.append(dict(seed=s, mode=mode, model=name, group=grp, n=int(mk.sum()), detected=int((yp[mk] == 1).sum())))
comp = pd.DataFrame(comp); comp.to_csv(REV_DIR / 'A7_fault_events_per_split.csv', index=False); print(comp.to_string(index=False))
d = pd.DataFrame(rows); d.to_csv(REV_DIR / 'A7_detection_by_event_exposure_per_run.csv', index=False)
p = d.groupby(['mode', 'model', 'group'])[['n', 'detected']].sum(); p['detection_rate'] = (p.detected / p.n).round(3)
p.to_csv(REV_DIR / 'A7_detection_by_event_exposure_pooled.csv'); print(p.to_string())
q = d.groupby(['mode', 'group'])[['n', 'detected']].sum(); q['detection_rate'] = (q.detected / q.n).round(3); print(q.to_string())
print('partitions with at least one unseen fault event in test:', int((comp.test_anomaly_rows_event_unseen > 0).sum()), 'of', len(comp))

# Contiguous label runs along the time axis (used by the timeline figure).
t = ns['time_series_raw']; blk = ns['_block_id_b2']; o = np.argsort(t)
ys, ts, bs = y_raw[o], (t[o] - t.min()) / 1e6, blk[o]; runs = []; st0 = 0
for i in range(1, len(ys) + 1):
    if i == len(ys) or ys[i] != ys[st0]:
        runs.append(dict(label=int(ys[st0]), n=i - st0, t_start_s=round(float(ts[st0]), 1), t_end_s=round(float(ts[i - 1]), 1), blocks=sorted(set(bs[st0:i].tolist())))); st0 = i
pd.DataFrame(runs).to_csv(REV_DIR / 'A1_label_runs.csv', index=False)
print('label runs:', len(runs), '| fault intervals:', sum(r['label'] > 0 for r in runs))
