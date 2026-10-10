"""Revision analyses that need no quantum simulation (run time: minutes).

A0  reproduce the stored classical results (determinism check)
A1  episode / block / split structure and autocorrelation (label runs: rev_events.py)
A2  random-row split vs blocked split
A3  embargo, forward-chaining and episode-held-out sensitivity
A4  sensitivity to SMOTETomek + synthetic-sample diagnostics
A5  conventional false-alarm rate recovered from the stored per-run files
A6  fairness decomposition XGBoost -> XGB-raw, and label-informed stacking controls
"""
import warnings, time, sys, platform
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from rev_common import load_notebook_namespace, confusion_rates, ROOT, REV_DIR, STORED

ns = load_notebook_namespace()
g = ns.__getitem__
B2_SPLITS, B2_PREPARED = g('B2_SPLITS'), g('B2_PREPARED')
SEEDS, MODES = g('SEEDS_B2_BINARY'), g('FEATURE_SET_MODES')
compute_metrics, make_classical_models = g('compute_metrics'), g('make_classical_models')
get_active_features = g('get_active_features')
time_us, y_raw, df_raw = g('time_series_raw'), g('y_raw'), g('df_raw')
block_id = g('_block_id_b2'); X_full = g('_X_full_b2'); y_bin = g('_y_bin_all')
episode = g('episode_series_raw')
FEATURE_COLS_FULL = g('FEATURE_COLS_FULL'); FEATURE_COLS_STRICT = g('FEATURE_COLS_STRICT')
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import sklearn, imblearn, xgboost

OUT = REV_DIR
log = open(OUT / 'revision_classical_log.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

P('python', sys.version.split()[0], '| numpy', np.__version__, '| pandas', pd.__version__,
  '| sklearn', sklearn.__version__, '| imblearn', imblearn.__version__, '| xgboost', xgboost.__version__,
  '|', platform.platform())
P('table:', X_full.shape, 'anomaly fraction', round(y_bin.mean(), 4))


def evaluate(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    proba = model.predict_proba(Xte) if hasattr(model, 'predict_proba') else None
    m = compute_metrics(yte, yp, proba, n_classes=2)
    m.update(confusion_rates(yte, yp))
    return m

KEEP = ['f1_macro', 'balanced_acc', 'mcc', 'roc_auc_ovr', 'far', 'miss']

def run_classical(Xtr, ytr, Xte, yte, seed, resample=True, tag=None):
    rows = []
    if resample:
        Xtr, ytr = SMOTETomek(random_state=42 + seed).fit_resample(Xtr, ytr)
    for name, mdl in make_classical_models(seed).items():
        m = evaluate(mdl, Xtr, ytr, Xte, yte)
        rows.append(dict(model=name, seed=seed, n_train=len(ytr), **{k: m[k] for k in KEEP + ['tp','tn','fp','fn']}, **(tag or {})))
    return rows

def agg(df, by):
    a = df.groupby(by)[KEEP].agg(['mean', 'std']).round(4)
    a.columns = [f'{c}_{s}' for c, s in a.columns]
    return a.reset_index()

# ───────────────────────────── A0 reproduction ─────────────────────────────
P('\n=== A0 reproduction of stored classical per-run results ===')
rows = []
for s in SEEDS:
    for mode in MODES:
        prep = B2_PREPARED[(s, mode)]
        for name, mdl in make_classical_models(s).items():
            m = evaluate(mdl, prep['X_train_bal'], prep['y_train_bal'], prep['X_test_sc'], prep['y_test'])
            rows.append(dict(model=name, seed=s, mode=mode, far_class0=m['far_class0'], dr_anomaly=m['dr_anomaly'],
                             **{k: m[k] for k in KEEP + ['tp','tn','fp','fn']}))
base = pd.DataFrame(rows); base.to_csv(OUT / 'A0_classical_blocked_smotetomek_per_run.csv', index=False)
stored = pd.read_csv(STORED / 'classical/b2_classical_per_run.csv')
mg = base.merge(stored, on=['model', 'seed', 'mode'], suffixes=('', '_stored'))
for k in ['f1_macro', 'roc_auc_ovr', 'far_class0']:
    d = (mg[k] - mg[k + '_stored']).abs()
    P(f'  {k}: rows={len(mg)} max|diff|={d.max():.2e} exact-match rows={(d < 1e-9).sum()}')
P('  per-model max |F1 diff|:', mg.assign(d=(mg.f1_macro - mg.f1_macro_stored).abs()).groupby('model').d.max().round(4).to_dict())
P('  identity check: stored far_class0 == miss (FN/(FN+TP)) max|diff| =',
  f"{(base.far_class0 - base.miss).abs().max():.2e}", '| far_class0 + dr_anomaly - 1 max =',
  f"{(base.far_class0 + base.dr_anomaly - 1).abs().max():.2e}")
A0 = agg(base, ['model', 'mode']); A0.to_csv(OUT / 'A0_classical_blocked_smotetomek_agg.csv', index=False)
P(A0[['model','mode','f1_macro_mean','f1_macro_std','roc_auc_ovr_mean','far_mean','far_std','miss_mean','miss_std']].to_string(index=False))

# ───────────────────────────── A1 structure ─────────────────────────────
P('\n=== A1 episode / block / split structure ===')
order = np.argsort(time_us); rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
t_s = (time_us - time_us.min()) / 1e6
ep_rows = []
for e in np.unique(episode):
    mk = episode == e
    ep_rows.append(dict(episode=int(e), n=int(mk.sum()), t_start_s=round(float(t_s[mk].min()), 1), t_end_s=round(float(t_s[mk].max()), 1),
                        duration_s=round(float(t_s[mk].max() - t_s[mk].min()), 1),
                        **{f'label_{c}': int((y_raw[mk] == c).sum()) for c in range(5)},
                        blocks=sorted(set(block_id[mk].tolist()))))
ep_df = pd.DataFrame(ep_rows); ep_df.to_csv(OUT / 'A1_episodes.csv', index=False); P(ep_df.to_string(index=False))
dts = np.diff(np.sort(time_us)) / 1e6
P('  gaps > 1 s between consecutive rows (s):', np.round(dts[dts > 1.0], 2).tolist(), '| median dt (s):', round(float(np.median(dts)), 4))
bl = []
for b in range(10):
    mk = block_id == b
    bl.append(dict(block=b, n=int(mk.sum()), t_start_s=round(float(t_s[mk].min()), 1), t_end_s=round(float(t_s[mk].max()), 1),
                   anomaly_frac=round(float(y_bin[mk].mean()), 3), episodes=sorted(set(episode[mk].tolist())),
                   **{f'label_{c}': int((y_raw[mk] == c).sum()) for c in range(5)}))
bl_df = pd.DataFrame(bl); bl_df.to_csv(OUT / 'A1_blocks.csv', index=False); P(bl_df.to_string(index=False))
sp = []
for s in SEEDS:
    S = B2_SPLITS[s]
    btr, bva, bte = (sorted(set(S[f'block_id_{k}'].tolist())) for k in ('train', 'val', 'test'))
    adj = sum(1 for b in bte for nb_ in (b - 1, b + 1) if nb_ in btr)
    sp.append(dict(seed=s, blocks_train=btr, blocks_val=bva, blocks_test=bte,
                   train_rows_later_than_first_test_row=int((rank[S['idx_train']] > rank[S['idx_test']].min()).sum()),
                   frac_train_later_than_first_test=round(float((rank[S['idx_train']] > rank[S['idx_test']].min()).mean()), 3),
                   test_block_edges_touching_train=adj,
                   episodes_in_train=sorted(set(episode[S['idx_train']].tolist())),
                   episodes_in_test=sorted(set(episode[S['idx_test']].tolist())),
                   test_anomaly_frac=round(float(S['y_test'].mean()), 3)))
sp_df = pd.DataFrame(sp); sp_df.to_csv(OUT / 'A1_splits.csv', index=False); P(sp_df.to_string(index=False))
tb = pd.Series([b for r in sp for b in r['blocks_test']]).value_counts().sort_index()
P('  times each block is used as test across the 10 seeds:', tb.to_dict())
P('  distinct test-block pairs:', len({tuple(r['blocks_test']) for r in sp}), 'of', len(sp))
# autocorrelation of strict features inside the main episode
main = ep_df.sort_values('n').iloc[-1]['episode']; mk = (episode == main)
Xm = pd.DataFrame(X_full[mk][np.argsort(time_us[mk])], columns=FEATURE_COLS_FULL)[FEATURE_COLS_STRICT]
ac = {lag: Xm.apply(lambda c: c.autocorr(lag)).abs() for lag in (1, 10, 50, 100, 482)}
ac_df = pd.DataFrame(ac); ac_df.columns = [f'lag_{k}' for k in ac]; ac_df.to_csv(OUT / 'A1_autocorr_strict.csv')
P('  |autocorrelation| of the 22 strict features in the main episode (median [min,max]):')
for c in ac_df: P(f'    {c}: median={ac_df[c].median():.3f} min={ac_df[c].min():.3f} max={ac_df[c].max():.3f}  (>0.5 in {(ac_df[c] > 0.5).sum()} of 22)')

# ───────────────────────────── A2 random-row vs blocked ─────────────────────────────
P('\n=== A2 random-row stratified split vs blocked split (same pipeline, SMOTETomek on train) ===')
rows = []
for s in SEEDS:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=s)
    i_tv, i_te = next(sss.split(X_full, y_bin))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=s)
    i_tr_rel, _ = next(sss2.split(X_full[i_tv], y_bin[i_tv])); i_tr = i_tv[i_tr_rel]
    sc = RobustScaler().fit(X_full[i_tr])
    for mode in MODES:
        _, idx = get_active_features(mode)
        rows += run_classical(sc.transform(X_full[i_tr])[:, idx], y_bin[i_tr], sc.transform(X_full[i_te])[:, idx], y_bin[i_te],
                              s, tag=dict(mode=mode, split='random_row'))
rnd = pd.DataFrame(rows); rnd.to_csv(OUT / 'A2_random_row_per_run.csv', index=False)
A2 = agg(rnd, ['model', 'mode']); A2.to_csv(OUT / 'A2_random_row_agg.csv', index=False)
P(A2[['model','mode','f1_macro_mean','f1_macro_std','roc_auc_ovr_mean','far_mean','miss_mean']].to_string(index=False))

# ───────────────────────────── A3 embargo / forward / episodes ─────────────────────────────
P('\n=== A3a embargo: drop training rows within E samples (10 Hz) of any test row ===')
rows = []
for s in SEEDS:
    S = B2_SPLITS[s]; r_te = np.sort(rank[S['idx_test']]); r_tr = rank[S['idx_train']]
    pos = np.searchsorted(r_te, r_tr)
    dist = np.minimum(np.abs(r_tr - r_te[np.clip(pos - 1, 0, len(r_te) - 1)]), np.abs(r_te[np.clip(pos, 0, len(r_te) - 1)] - r_tr))
    for E in (0, 50, 150, 300, 482):
        keep = dist > E
        if len(np.unique(S['y_train'][keep])) < 2: continue
        sc = RobustScaler().fit(S['X_train_raw'][keep])
        for mode in ('full', 'no_proxy_strict'):
            _, idx = get_active_features(mode)
            rows += run_classical(sc.transform(S['X_train_raw'][keep])[:, idx], S['y_train'][keep],
                                  sc.transform(S['X_test_raw'])[:, idx], S['y_test'], s,
                                  tag=dict(mode=mode, embargo=E, n_train_real=int(keep.sum())))
emb = pd.DataFrame(rows); emb.to_csv(OUT / 'A3a_embargo_per_run.csv', index=False)
A3a = agg(emb, ['mode', 'embargo', 'model']); A3a.to_csv(OUT / 'A3a_embargo_agg.csv', index=False)
P(emb.groupby(['mode', 'embargo']).agg(n_seeds=('seed', 'nunique'), n_train_real=('n_train_real', 'mean'), f1=('f1_macro', 'mean'),
                                       auc=('roc_auc_ovr', 'mean'), far=('far', 'mean'), miss=('miss', 'mean')).round(3).to_string())
P(A3a.pivot_table(index=['mode', 'model'], columns='embargo', values='f1_macro_mean').round(3).to_string())

P('\n=== A3b forward-chaining: train on blocks 0..k-1, test on block k (no future data in training) ===')
rows = []
for k in range(1, 10):
    tr = block_id < k; te = block_id == k
    if len(np.unique(y_bin[tr])) < 2: continue
    sc = RobustScaler().fit(X_full[tr])
    for mode in ('full', 'no_proxy_strict'):
        _, idx = get_active_features(mode)
        Xtr, ytr = SMOTETomek(random_state=42).fit_resample(sc.transform(X_full[tr])[:, idx], y_bin[tr])
        for name, mdl in make_classical_models(0).items():
            mdl.fit(Xtr, ytr); yp = mdl.predict(sc.transform(X_full[te])[:, idx])
            pr = mdl.predict_proba(sc.transform(X_full[te])[:, idx])[:, 1]
            cr = confusion_rates(y_bin[te], yp)
            rows.append(dict(test_block=k, mode=mode, model=name, n_train=int(tr.sum()), test_anomaly_frac=round(float(y_bin[te].mean()), 3),
                             accuracy=float((yp == y_bin[te]).mean()), **cr))
fw = pd.DataFrame(rows); fw.to_csv(OUT / 'A3b_forward_chaining_per_fold.csv', index=False)
pooled = fw.groupby(['mode', 'model'])[['tp', 'tn', 'fp', 'fn']].sum()
pooled['far'] = pooled.fp / (pooled.fp + pooled.tn); pooled['miss'] = pooled.fn / (pooled.fn + pooled.tp)
pooled['balanced_acc'] = 1 - (pooled.far + pooled.miss) / 2
rec = pooled.tp / (pooled.tp + pooled.fn); prec = pooled.tp / (pooled.tp + pooled.fp).replace(0, np.nan)
f1p = 2 * prec * rec / (prec + rec); recn = pooled.tn / (pooled.tn + pooled.fp); precn = pooled.tn / (pooled.tn + pooled.fn).replace(0, np.nan)
f1n = 2 * precn * recn / (precn + recn); pooled['f1_macro'] = (f1p.fillna(0) + f1n.fillna(0)) / 2
pooled.round(4).to_csv(OUT / 'A3b_forward_chaining_pooled.csv')
P('  test blocks used:', sorted(fw.test_block.unique().tolist()), '| pooled over folds:')
P(pooled.round(3).to_string())
P(fw.pivot_table(index=['mode', 'test_block', 'test_anomaly_frac'], columns='model', values='accuracy').round(2).to_string())

P('\n=== A3c gap-defined episodes held out (train on the other two) ===')
rows = []
for e in np.unique(episode):
    te = episode == e; tr = ~te
    if len(np.unique(y_bin[tr])) < 2: continue
    sc = RobustScaler().fit(X_full[tr])
    for mode in ('full', 'no_proxy_strict'):
        _, idx = get_active_features(mode)
        Xtr, ytr = SMOTETomek(random_state=42).fit_resample(sc.transform(X_full[tr])[:, idx], y_bin[tr])
        for name, mdl in make_classical_models(0).items():
            mdl.fit(Xtr, ytr); yp = mdl.predict(sc.transform(X_full[te])[:, idx])
            rows.append(dict(held_out_episode=int(e), n_test=int(te.sum()), test_anomaly_frac=round(float(y_bin[te].mean()), 3), mode=mode, model=name,
                             accuracy=float((yp == y_bin[te]).mean()), **confusion_rates(y_bin[te], yp)))
epo = pd.DataFrame(rows); epo.to_csv(OUT / 'A3c_episode_holdout.csv', index=False)
P(epo[['held_out_episode','n_test','test_anomaly_frac','mode','model','accuracy','far','miss']].round(3).to_string(index=False))

# ───────────────────────────── A4 SMOTETomek sensitivity ─────────────────────────────
P('\n=== A4 blocked split WITHOUT SMOTETomek (everything else unchanged) ===')
rows = []
for s in SEEDS:
    for mode in MODES:
        prep = B2_PREPARED[(s, mode)]
        rows += run_classical(prep['X_train_sc'], prep['y_train'], prep['X_test_sc'], prep['y_test'], s, resample=False, tag=dict(mode=mode))
nos = pd.DataFrame(rows); nos.to_csv(OUT / 'A4_no_resampling_per_run.csv', index=False)
A4 = agg(nos, ['model', 'mode']); A4.to_csv(OUT / 'A4_no_resampling_agg.csv', index=False)
cmp_ = A0.merge(A4, on=['model', 'mode'], suffixes=('_smt', '_none'))
for k in ['f1_macro', 'roc_auc_ovr', 'far', 'miss']:
    cmp_[f'd_{k}'] = (cmp_[f'{k}_mean_none'] - cmp_[f'{k}_mean_smt']).round(4)
cmp_.to_csv(OUT / 'A4_smotetomek_vs_none.csv', index=False)
P(cmp_[['model','mode','f1_macro_mean_smt','f1_macro_mean_none','d_f1_macro','roc_auc_ovr_mean_smt','roc_auc_ovr_mean_none','d_roc_auc_ovr','d_far','d_miss']].to_string(index=False))
pr = base.merge(nos, on=['model', 'seed', 'mode'], suffixes=('_smt', '_none'))
P('  per-run |F1 diff| SMOTETomek vs none: mean=%.4f max=%.4f ; runs where none>=SMOTETomek: %d of %d' % (
    (pr.f1_macro_smt - pr.f1_macro_none).abs().mean(), (pr.f1_macro_smt - pr.f1_macro_none).abs().max(),
    (pr.f1_macro_none >= pr.f1_macro_smt).sum(), len(pr)))

P('\n=== A4b synthetic-sample diagnostics (strict mode, physical channels) ===')
rows = []
for s in SEEDS:
    S = B2_SPLITS[s]; prep = B2_PREPARED[(s, 'no_proxy_strict')]
    real = {r.tobytes() for r in np.ascontiguousarray(prep['X_train_sc'])}
    is_syn = np.array([r.tobytes() not in real for r in np.ascontiguousarray(prep['X_train_bal'])])
    _, idx = get_active_features('no_proxy_strict')
    sc = S['scaler']; ctr, scl = sc.center_[idx], sc.scale_[idx]
    raw_bal = pd.DataFrame(prep['X_train_bal'] * scl + ctr, columns=FEATURE_COLS_STRICT)
    raw_real = pd.DataFrame(S['X_train_raw'][:, idx], columns=FEATURE_COLS_STRICT)
    def norms(d): return (np.linalg.norm(d[['MagX','MagY','MagZ']].values, axis=1), np.linalg.norm(d[['abAccX','abAccY','abAccZ']].values, axis=1))
    mag_r, acc_r = norms(raw_real); syn = raw_bal[is_syn]
    row = dict(seed=s, n_train_real=len(raw_real), n_bal=len(raw_bal), n_synthetic=int(is_syn.sum()),
               synthetic_frac=round(float(is_syn.mean()), 3), n_real_removed_by_tomek=int(len(raw_real) - (~is_syn).sum()),
               minority_class=int(np.argmin(np.bincount(S['y_train']))))
    if is_syn.sum():
        mag_s, acc_s = norms(syn)
        lo, hi = np.percentile(mag_r, [1, 99]); row['syn_mag_norm_outside_real_1_99pct'] = round(float(((mag_s < lo) | (mag_s > hi)).mean()), 3)
        lo, hi = np.percentile(acc_r, [1, 99]); row['syn_acc_norm_outside_real_1_99pct'] = round(float(((acc_s < lo) | (acc_s > hi)).mean()), 3)
        # heading wrap-around: synthetic yaw values in a band that real minority rows never occupy
        yaw_min_real = raw_real['Yaw'][S['y_train'] == row['minority_class']].values
        hist, edges = np.histogram(yaw_min_real, bins=36, range=(0, 360)); empty = hist == 0
        b = np.clip(np.digitize(syn['Yaw'].values, edges) - 1, 0, 35)
        row['syn_yaw_in_heading_bins_empty_for_real_minority'] = round(float(empty[b].mean()), 3)
        row['real_minority_empty_yaw_bins_of_36'] = int(empty.sum())
    rows.append(row)
syn_df = pd.DataFrame(rows); syn_df.to_csv(OUT / 'A4b_synthetic_diagnostics.csv', index=False); P(syn_df.to_string(index=False))
P('  synthetic share of balanced training set: mean=%.3f min=%.3f max=%.3f' % (syn_df.synthetic_frac.mean(), syn_df.synthetic_frac.min(), syn_df.synthetic_frac.max()))

# ───────────────────────────── A5 conventional FAR from stored files ─────────────────────────────
P('\n=== A5 conventional FAR recovered from the STORED per-run files (far = 2*far_macro - far_class0) ===')
st = []
for f, col in (('classical/b2_classical_per_run.csv', 'model'), ('quantum/b2_dru_per_run.csv', 'model'), ('hybrid/b2_hybrid_per_run.csv', 'variant')):
    d = pd.read_csv(STORED / f); d['name'] = d[col]
    d['far'] = 2 * d.far_macro - d.far_class0; d['miss'] = d.far_class0
    st.append(d[['name', 'seed', 'mode', 'f1_macro', 'balanced_acc', 'mcc', 'roc_auc_ovr', 'pr_auc', 'far', 'miss', 'dr_anomaly']])
st = pd.concat(st); st.to_csv(OUT / 'A5_all_models_per_run_with_conventional_far.csv', index=False)
P('  check miss + dr_anomaly == 1: max|.| =', f"{(st.miss + st.dr_anomaly - 1).abs().max():.2e}")
chk = st.merge(base.rename(columns={'model': 'name'}), on=['name', 'seed', 'mode'], suffixes=('', '_re'))
P('  derived FAR vs recomputed confusion-matrix FAR (classical): max|diff| =', f"{(chk.far - chk.far_re).abs().max():.2e}")
A5 = st.groupby(['name', 'mode'])[['f1_macro', 'balanced_acc', 'mcc', 'roc_auc_ovr', 'far', 'miss']].agg(['mean', 'std', 'median', 'min', 'max']).round(4)
A5.columns = [f'{c}_{s}' for c, s in A5.columns]; A5 = A5.reset_index(); A5.to_csv(OUT / 'A5_all_models_agg_with_conventional_far.csv', index=False)
P(A5[['name','mode','f1_macro_mean','f1_macro_std','roc_auc_ovr_mean','roc_auc_ovr_std','far_mean','far_std','miss_mean','miss_std','balanced_acc_mean','mcc_mean']].to_string(index=False))
deg = st.assign(const=((st.far.isin([0.0, 1.0])) & (st.miss.isin([0.0, 1.0])) & ((st.far + st.miss) == 1.0)))
P('  runs that predict a single class for the whole test fold (of 10):')
P(deg.groupby(['name', 'mode']).const.sum().unstack().to_string())
P('  paired seed-wise comparison, strict mode, XGB-DRU-trained vs each other model (wins/ties/losses over the 10 partitions):')
piv = {k: st[st['mode'] == 'no_proxy_strict'].pivot(index='seed', columns='name', values=k) for k in ['f1_macro', 'roc_auc_ovr', 'far', 'miss']}
w = []
for other in piv['f1_macro'].columns:
    if other == 'XGB-DRU-trained': continue
    r = dict(vs=other)
    for k, hib in (('f1_macro', True), ('roc_auc_ovr', True), ('far', False), ('miss', False)):
        d = piv[k]['XGB-DRU-trained'] - piv[k][other]; d = d if hib else -d
        r[k] = f'{int((d > 1e-12).sum())}/{int((d.abs() <= 1e-12).sum())}/{int((d < -1e-12).sum())}'; r[k + '_mean_diff'] = round(float((d if hib else -d).mean()), 3)
    w.append(r)
w = pd.DataFrame(w); w.to_csv(OUT / 'A5_paired_wins_strict.csv', index=False); P(w.to_string(index=False))

# ───────────────────────────── A6 fairness decomposition ─────────────────────────────
P('\n=== A6 why does classical XGBoost differ from the hybrid control XGB-raw? step-by-step decomposition ===')
make_disjoint = g('make_disjoint_quantum_subsets'); CONFIG = g('CONFIG')
def xgb_default(seed): return XGBClassifier(n_estimators=100, random_state=seed, objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1)
def xgb_hybrid(seed):  return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=seed, n_jobs=-1, verbosity=0)
rows = []; sub_rows = []
for s in SEEDS:
    for mode in MODES:
        prep = B2_PREPARED[(s, mode)]; yte = prep['y_test']
        Xq, yb, Xq_te = prep['Xq_train_bal'], prep['y_train_bal'], prep['Xq_test']
        n_per = min(CONFIG['DRU_TRAIN_BUDGET'] // 2, int(min(np.bincount(yb))) // 2)
        (XA, yA), (XB, yB) = make_disjoint(Xq, yb, n_per, seed=s + 1)
        real = {r.tobytes() for r in np.ascontiguousarray(prep['X_train_sc'][:, prep['q_idx_local']])}
        Xq_unscaled_bal = prep['X_train_bal'][:, prep['q_idx_local']]
        key = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(Xq))}
        def syn_frac(Xs):
            ii = [key[r.tobytes()] for r in np.ascontiguousarray(Xs)]
            return float(np.mean([np.ascontiguousarray(Xq_unscaled_bal[i]).tobytes() not in real for i in ii]))
        sub_rows.append(dict(seed=s, mode=mode, n_A=len(yA), n_B=len(yB), n_bal=len(yb), n_features_mode=prep['n_features'],
                             synthetic_frac_A=round(syn_frac(XA), 3), synthetic_frac_B=round(syn_frac(XB), 3),
                             test_rows_outside_train_angle_range=round(float(((Xq_te < -np.pi - 1e-9) | (Xq_te > np.pi + 1e-9)).any(axis=1).mean()), 3)))
        steps = [
            ('1 XGBoost: all features, all balanced rows, default lr=0.3 (classical baseline)', xgb_default(s), prep['X_train_bal'], yb, prep['X_test_sc']),
            ('2 as 1 but only the 5 MI-selected features', xgb_default(s), Xq, yb, Xq_te),
            ('3 as 2 but only the 800-row subset B', xgb_default(s + 1), XB, yB, Xq_te),
            ('4 as 3 with hybrid hyper-parameters lr=0.1 (= XGB-raw)', xgb_hybrid(s + 1), XB, yB, Xq_te),
        ]
        for nm, mdl, Xtr, ytr, Xte in steps:
            m = evaluate(mdl, Xtr, ytr, Xte, yte); rows.append(dict(step=nm, seed=s, mode=mode, n_train=len(ytr), n_feat=Xtr.shape[1], **{k: m[k] for k in KEEP}))
        # label-informed classical stacking controls: a classical model fitted on A supplies one score to XGBoost on B
        for nm, clf in (('5 stack-LogReg(A) score + angles -> XGB on B', LogisticRegression(max_iter=1000, random_state=s + 1)),
                        ('6 stack-MLP[4](A, 29 weights) score + angles -> XGB on B', MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=s + 1))):
            clf.fit(XA, yA)
            m = evaluate(xgb_hybrid(s + 1), np.hstack([XB, clf.predict_proba(XB)[:, [1]]]), yB, np.hstack([Xq_te, clf.predict_proba(Xq_te)[:, [1]]]), yte)
            rows.append(dict(step=nm, seed=s, mode=mode, n_train=len(yB), n_feat=6, **{k: m[k] for k in KEEP}))
            yp = clf.predict(Xq_te); m2 = compute_metrics(yte, yp, clf.predict_proba(Xq_te), n_classes=2); m2.update(confusion_rates(yte, yp))
            rows.append(dict(step=nm.split(' score')[0].replace('stack-', 'standalone ') + ' (same 800 rows / 5 features as the DRU)', seed=s, mode=mode, n_train=len(yA), n_feat=5, **{k: m2[k] for k in KEEP}))
dec = pd.DataFrame(rows); dec.to_csv(OUT / 'A6_fairness_decomposition_per_run.csv', index=False)
subs = pd.DataFrame(sub_rows); subs.to_csv(OUT / 'A6_subset_sizes.csv', index=False)
P(subs.groupby('mode')[['n_A', 'n_B', 'n_bal', 'synthetic_frac_A', 'synthetic_frac_B', 'test_rows_outside_train_angle_range']].agg(['mean', 'min', 'max']).round(3).to_string())
A6 = agg(dec, ['mode', 'step']); A6.to_csv(OUT / 'A6_fairness_decomposition_agg.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
P(A6[['mode','step','f1_macro_mean','f1_macro_std','roc_auc_ovr_mean','far_mean','miss_mean']].to_string(index=False))
hy = pd.read_csv(STORED / 'hybrid/b2_hybrid_per_run.csv'); hy = hy[hy.variant == 'XGB-raw']
c4 = dec[dec.step.str.startswith('4')].merge(hy, on=['seed', 'mode'], suffixes=('', '_stored'))
P('  step 4 vs stored XGB-raw: max |F1 diff| =', f"{(c4.f1_macro - c4.f1_macro_stored).abs().max():.2e}", '(rows', len(c4), ')')
d4 = dec[dec.step.str.startswith('4')]
P('  XGB-raw runs predicting one class only:', d4.assign(c=(d4.balanced_acc == 0.5)).groupby('mode').c.sum().to_dict())
tr_h = pd.read_csv(STORED / 'hybrid/b2_hybrid_per_run.csv'); tr_h = tr_h[tr_h.variant == 'XGB-DRU-trained'][['seed', 'mode', 'f1_macro']]
for stp in ('5', '6'):
    c = dec[dec.step.str.startswith(stp) & dec.step.str.contains('XGB on B')].merge(tr_h, on=['seed', 'mode'], suffixes=('', '_dru'))
    P(f'  stacking control {stp} vs stored XGB-DRU-trained (F1): ' + ' | '.join(
        f"{m}: ctrl={x.f1_macro.mean():.3f} dru={x.f1_macro_dru.mean():.3f} ctrl>=dru in {(x.f1_macro >= x.f1_macro_dru).sum()}/10" for m, x in c.groupby('mode')))
P('\nDONE', time.strftime('%Y-%m-%d %H:%M:%S'))
