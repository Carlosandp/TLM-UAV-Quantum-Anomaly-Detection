"""Revision analyses that need the simulated DRU circuit.

D1  serial re-fit of one stored configuration (seed 0, strict, SMOTETomek):
    determinism check, circuit-evaluation count, wall-clock training time,
    loss trajectory and inference latency on the classical simulator host.
D2  strict mode WITHOUT synthetic oversampling, 10 partitions:
    standalone DRU, the six hybrid variants and two label-informed classical
    stacking controls, all trained on real rows only.
"""
import warnings, time, sys, platform, json, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from multiprocessing import Pool
from rev_common import load_notebook_namespace, confusion_rates, ROOT, REV_DIR, STORED

_ns = None
def NS():
    global _ns
    if _ns is None: _ns = load_notebook_namespace()
    return _ns

KEEP = ['f1_macro', 'balanced_acc', 'mcc', 'roc_auc_ovr', 'far', 'miss']

def make_dru(seed):
    ns = NS(); C = ns['CONFIG']
    return ns['DataReuploadingClassifier'](n_qubits=C['DRU_N_QUBITS'], n_layers=C['DRU_N_LAYERS'], encoding=C['DRU_ENCODING'],
                                           entanglement=C['DRU_ENTANGLEMENT'], optimizer=C['DRU_OPTIMIZER'], max_iter=C['DRU_MAX_ITER'], seed=seed)

def d1():
    ns = NS(); C = ns['CONFIG']; s, mode = 0, 'no_proxy_strict'
    prep = ns['B2_PREPARED'][(s, mode)]
    n_per = min(C['DRU_TRAIN_BUDGET'] // 2, int(min(np.bincount(prep['y_train_bal']))) // 2)
    (XA, yA), (XB, yB) = ns['make_disjoint_quantum_subsets'](prep['Xq_train_bal'], prep['y_train_bal'], n_per, seed=s + 1)
    dru = make_dru(s + 1)
    calls = {'batches': 0, 'circuits': 0}
    orig = type(dru)._evaluate_batch
    def counted(self, w, X, obs):
        calls['batches'] += 1; calls['circuits'] += len(X) * len(obs); return orig(self, w, X, obs)
    type(dru)._evaluate_batch = counted
    t0 = time.perf_counter(); dru.fit(XA, yA); t_fit = time.perf_counter() - t0
    fit_calls = dict(calls)
    Xte, yte = prep['Xq_test'], prep['y_test']
    yp = dru.predict(Xte); m = ns['compute_metrics'](yte, yp, dru.predict_proba(Xte), n_classes=2); m.update(confusion_rates(yte, yp))
    stored = pd.read_csv(STORED / 'quantum/b2_dru_per_run.csv'); st = stored[(stored.seed == s) & (stored['mode'] == mode)].iloc[0]
    batch_t = []
    for _ in range(5):
        t0 = time.perf_counter(); dru.predict_proba(Xte); batch_t.append(time.perf_counter() - t0)
    single_t = []
    for i in range(100):
        t0 = time.perf_counter(); dru.predict_proba(Xte[i:i + 1]); single_t.append(time.perf_counter() - t0)
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=s + 1, n_jobs=-1, verbosity=0)
    f = ns['extract_quantum_features']
    xgb.fit(np.hstack([XB, f(dru, XB)]), yB)
    hyb_single = []
    for i in range(100):
        t0 = time.perf_counter(); xgb.predict(np.hstack([Xte[i:i + 1], f(dru, Xte[i:i + 1])])); hyb_single.append(time.perf_counter() - t0)
    xgb_only = []
    xr = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', random_state=s + 1, n_jobs=-1, verbosity=0).fit(XB, yB)
    for i in range(100):
        t0 = time.perf_counter(); xr.predict(Xte[i:i + 1]); xgb_only.append(time.perf_counter() - t0)
    qc = dru._circuit_.circuit
    import qiskit, scipy
    lh = list(dru.loss_history_)
    out = dict(config=dict(seed=s, mode=mode, n_train_A=len(yA), n_test=len(yte), n_qubits=qc.num_qubits, n_trainable=int(dru._circuit_.n_weights),
                           depth=int(qc.depth()), gate_counts={k: int(v) for k, v in qc.count_ops().items()}, optimizer=C['DRU_OPTIMIZER'], max_iter=C['DRU_MAX_ITER'],
                           estimator=type(dru._estimator_).__name__, shots=None),
               host=dict(platform=platform.platform(), processor=os.environ.get('TLM_HOST_DESCRIPTION', platform.processor() or platform.machine()), python=sys.version.split()[0], qiskit=qiskit.__version__, scipy=scipy.__version__, numpy=np.__version__),
               fit=dict(wall_clock_s=round(t_fit, 1), loss_evaluations=fit_calls['batches'], circuit_evaluations=fit_calls['circuits'],
                        loss_first=round(lh[0], 4), loss_last=round(lh[-1], 4), loss_min=round(min(lh), 4), n_loss_history=len(lh),
                        stored_wall_clock_s=round(float(st.train_time_s), 1)),
               reproduction=dict(f1_macro=m['f1_macro'], f1_macro_stored=float(st.f1_macro), balanced_acc=m['balanced_acc'], balanced_acc_stored=float(st.balanced_acc),
                                 roc_auc=m['roc_auc_ovr'], roc_auc_stored=float(st.roc_auc_ovr)),
               inference=dict(batch_test_fold_s_median=round(float(np.median(batch_t)), 3), per_sample_in_batch_ms=round(1e3 * float(np.median(batch_t)) / len(yte), 3),
                              single_sample_dru_ms_median=round(1e3 * float(np.median(single_t)), 2), single_sample_dru_ms_p95=round(1e3 * float(np.percentile(single_t, 95)), 2),
                              single_sample_hybrid_ms_median=round(1e3 * float(np.median(hyb_single)), 2), single_sample_hybrid_ms_p95=round(1e3 * float(np.percentile(hyb_single, 95)), 2),
                              single_sample_xgb_only_ms_median=round(1e3 * float(np.median(xgb_only)), 2)))
    json.dump(out, open(REV_DIR / 'D1_dru_cost_and_reproduction.json', 'w'), indent=2)
    pd.Series(lh).to_csv(REV_DIR / 'D1_loss_history_seed0_strict.csv', index=False, header=['mse_loss'])
    print(json.dumps(out, indent=2), flush=True)

def d2_one(s):
    ns = NS(); C = ns['CONFIG']
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.decomposition import PCA
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    mode = 'no_proxy_strict'; prep = ns['B2_PREPARED'][(s, mode)]
    Xtr, ytr, Xte_sc, yte = prep['X_train_sc'], prep['y_train'], prep['X_test_sc'], prep['y_test']   # real rows only
    mi = mutual_info_classif(Xtr, ytr, random_state=42 + s, discrete_features=False)
    q = np.argsort(mi)[::-1][:5]; feats = [prep['feat_cols'][i] for i in q]
    ang = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(Xtr[:, q]); Xq, Xq_te = ang.transform(Xtr[:, q]), ang.transform(Xte_sc[:, q])
    n_per = min(C['DRU_TRAIN_BUDGET'] // 2, int(min(np.bincount(ytr))) // 2)
    (XA, yA), (XB, yB) = ns['make_disjoint_quantum_subsets'](Xq, ytr, n_per, seed=s + 1)
    rows = []
    def rec(name, yp, proba, extra=None):
        m = ns['compute_metrics'](yte, yp, proba, n_classes=2); m.update(confusion_rates(yte, yp))
        rows.append(dict(name=name, seed=s, mode=mode, n_per_class=n_per, top_mi=','.join(feats), **{k: m[k] for k in KEEP}, **(extra or {})))
    dru = make_dru(s + 1); t0 = time.perf_counter(); dru.fit(XA, yA); t_fit = time.perf_counter() - t0
    lh = list(dru.loss_history_)
    rec('DRU-Binary-B2', dru.predict(Xq_te), dru.predict_proba(Xq_te), dict(loss_first=lh[0], loss_last=lh[-1], fit_s_parallel=t_fit))
    def xgb(): return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=s + 1, n_jobs=1, verbosity=0)
    f = ns['extract_quantum_features']
    # untrained DRU exactly as in the notebook's hybrid cell
    unt = ns['DataReuploadingClassifier'](n_qubits=5, n_layers=2, encoding=C['DRU_ENCODING'], entanglement=C['DRU_ENTANGLEMENT'], optimizer=C['DRU_OPTIMIZER'], max_iter=1, seed=s + 1)
    try: unt.fit(np.zeros((4, 5)), np.array([0, 1, 0, 1]))
    except Exception: pass
    unt.weights_ = np.random.default_rng((s + 1) * 7919 + 13).uniform(-np.pi, np.pi, size=unt.weights_.shape)
    pca = PCA(n_components=5, random_state=s + 1).fit(XB); poly = PolynomialFeatures(degree=2, include_bias=False).fit(XB)
    rbf = RBFSampler(gamma=1.0, n_components=max(poly.transform(XB).shape[1] - 5, 16), random_state=s + 1).fit(XB)
    lr = LogisticRegression(max_iter=1000, random_state=s + 1).fit(XA, yA)
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=s + 1).fit(XA, yA)
    T = {'XGB-raw': lambda X: X, 'XGB-PCA': lambda X: np.hstack([X, pca.transform(X)]), 'XGB-Poly2': lambda X: np.hstack([X, poly.transform(X)]),
         'XGB-RandomRBF': lambda X: np.hstack([X, rbf.transform(X)]), 'XGB-DRU-untrained': lambda X: np.hstack([X, f(unt, X).reshape(len(X), -1)]),
         'XGB-DRU-trained': lambda X: np.hstack([X, f(dru, X).reshape(len(X), -1)]),
         'XGB-stack-LogReg(A)': lambda X: np.hstack([X, lr.predict_proba(X)[:, [1]]]), 'XGB-stack-MLP4(A)': lambda X: np.hstack([X, mlp.predict_proba(X)[:, [1]]])}
    for name, tf in T.items():
        mdl = xgb().fit(tf(XB), yB); rec(name, mdl.predict(tf(Xq_te)), mdl.predict_proba(tf(Xq_te)))
    rec('LogReg on A (5 features, 800 rows)', lr.predict(Xq_te), lr.predict_proba(Xq_te))
    rec('MLP4 on A (5 features, 800 rows)', mlp.predict(Xq_te), mlp.predict_proba(Xq_te))
    print(f'D2 seed {s} done in {t_fit:.0f}s  DRU F1={rows[0]["f1_macro"]:.3f}', flush=True)
    return rows

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'd1': d1()
    else:
        with Pool(5) as p: res = p.map(d2_one, list(range(10)))
        df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(REV_DIR / 'D2_strict_no_resampling_dru_hybrid_per_run.csv', index=False)
        a = df.groupby('name')[KEEP].agg(['mean', 'std']).round(4); a.columns = [f'{c}_{s}' for c, s in a.columns]
        a.reset_index().to_csv(REV_DIR / 'D2_strict_no_resampling_dru_hybrid_agg.csv', index=False)
        pd.set_option('display.width', 250); print(a.to_string()); print(df[df.name == 'DRU-Binary-B2'][['seed', 'n_per_class', 'top_mi', 'loss_first', 'loss_last', 'f1_macro']].to_string())
