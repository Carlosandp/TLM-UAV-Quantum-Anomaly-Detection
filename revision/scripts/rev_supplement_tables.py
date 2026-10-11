"""Writes the LaTeX tables of the Supplementary Material directly from the
stored results and the revision outputs, so no number is transcribed by hand."""
import os, json
import numpy as np, pandas as pd
from pathlib import Path
from rev_common import ROOT, REV_DIR as REV
OUT = Path(os.environ.get('TLM_SUPP_DIR', ROOT / 'revision' / 'supplement_tables')); OUT.mkdir(parents=True, exist_ok=True)
NAMES = {'LogReg': 'Logistic regression', 'SVM-RBF': 'SVM', 'MLP': 'MLP', 'RandomForest': 'Random forest', 'XGBoost': 'XGBoost',
         'DRU-Binary-B2': 'DRU circuit alone', 'XGB-raw': 'XGB only (5 variables)', 'XGB-PCA': 'XGB + principal components', 'XGB-Poly2': 'XGB + polynomial terms',
         'XGB-RandomRBF': 'XGB + random RBF features', 'XGB-DRU-untrained': 'XGB + untrained DRU output', 'XGB-DRU-trained': 'XGB + trained DRU output'}
MODES = {'full': 'all', 'no_proxy_loose': 'reduced', 'no_proxy_strict': 'physical'}
def w(name, body): (OUT / name).write_text(body + '\n')
def ms(m, s): return f'{m:.2f} ({s:.2f})'
def tab(cols, rows, align=None):
    align = align or ('l' + 'c' * (len(cols) - 1))
    return '\\begin{tabular}{' + align + '}\n\\toprule\n' + ' & '.join(cols) + ' \\\\\n\\midrule\n' + '\n'.join(' & '.join(str(c) for c in r) + ' \\\\' for r in rows) + '\n\\bottomrule\n\\end{tabular}'

st = pd.read_csv(REV / 'A5_all_models_per_run_with_conventional_far.csv')
# S1: all models x 3 input sets x 4 metrics
for k, lab in (('f1_macro', 'f1'), ('roc_auc_ovr', 'auc'), ('far', 'far'), ('miss', 'miss')):
    g = st.groupby(['name', 'mode'])[k].agg(['mean', 'std'])
    rows = [[NAMES[n]] + [ms(*g.loc[(n, m)]) for m in MODES] for n in NAMES]
    w(f'S1_{lab}.tex', tab(['Model'] + list(MODES.values()), rows))
# S2: partitions
sp = pd.read_csv(REV / 'A1_splits.csv'); ev = pd.read_csv(REV / 'A7_fault_events_per_split.csv')
rows = [[r.seed, r.blocks_train.strip('[]'), r.blocks_test.strip('[]'), f'{r.test_anomaly_frac:.2f}', f'{100 * r.frac_train_later_than_first_test:.0f}\\%',
         r.test_block_edges_touching_train, e.fault_types_in_test.strip('[]'), e.test_anomaly_rows_event_unseen] for (_, r), (_, e) in zip(sp.iterrows(), ev.iterrows())]
w('S2_partitions.tex', tab(['Seed', 'Training blocks', 'Test blocks', 'Fault share in test', 'Training rows later than first test row', 'Test-block edges touching training', 'Fault types in test', 'Test fault rows from intervals unseen in training'], rows, 'cllccccc'))
bl = pd.read_csv(REV / 'A1_blocks.csv')
w('S2_blocks.tex', tab(['Block', 'Rows', 'Start (s)', 'End (s)', 'Nominal', 'GPS', 'Accel.', 'Engine', 'RC'], [[r.block, r.n, r.t_start_s, r.t_end_s, r.label_0, r.label_1, r.label_2, r.label_3, r.label_4] for _, r in bl.iterrows()], 'ccccccccc'))
# S3: split design
rnd = pd.read_csv(REV / 'A2_random_row_per_run.csv'); emb = pd.read_csv(REV / 'A3a_embargo_per_run.csv')
order = ['LogReg', 'SVM-RBF', 'MLP', 'RandomForest', 'XGBoost']
rows = []
for mode in ('full', 'no_proxy_strict'):
    for m in order + ['mean']:
        def v(df):
            d = df[df['mode'] == mode]; g = d.groupby('model').f1_macro.mean()
            return f'{(g.mean() if m == "mean" else g[m]):.2f}'
        rows.append([MODES[mode], 'Mean of five' if m == 'mean' else NAMES[m], v(rnd)] + [v(emb[emb.embargo == E]) for E in (0, 50, 150, 300, 482)])
w('S3_split_design.tex', tab(['Inputs', 'Model', 'Random rows', 'Blocks', '5 s gap', '15 s gap', '30 s gap', '48 s gap'], rows, 'llcccccc'))
n_tr = emb.groupby('embargo').n_train_real.mean().round(0).astype(int).to_dict()
fw = pd.read_csv(REV / 'A3b_forward_chaining_per_fold.csv')
piv = fw[fw['mode'] == 'no_proxy_strict'].pivot_table(index=['test_block', 'test_anomaly_frac'], columns='model', values='accuracy')[order]
w('S3_forward.tex', tab(['Test block', 'Fault share'] + [NAMES[m] for m in order], [[b, f'{f:.2f}'] + [f'{x:.2f}' for x in r] for (b, f), r in zip(piv.index, piv.values)], 'cc' + 'c' * 5))
pooled = pd.read_csv(REV / 'A3b_forward_chaining_pooled.csv')
w('S3_forward_pooled.tex', tab(['Inputs', 'Model', 'F1', 'False-alarm rate', 'Miss rate'], [[MODES[r['mode']], NAMES[r.model], f'{r.f1_macro:.2f}', f'{r.far:.2f}', f'{r.miss:.2f}'] for _, r in pooled.iterrows()], 'llccc'))
# S4: event exposure
e = pd.read_csv(REV / 'A7_detection_by_event_exposure_pooled.csv')
rows = [[MODES[mode], NAMES[m]] + [f"{float(e[(e['mode'] == mode) & (e.model == m) & (e.group == g)].detection_rate.iloc[0]):.2f}" for g in ('event partly seen in training', 'event unseen in training')] for mode in ('full', 'no_proxy_strict') for m in order]
w('S4_events.tex', tab(['Inputs', 'Model', 'Interval partly in training', 'Interval absent from training'], rows, 'llcc'))
# S5: fairness
dec = pd.read_csv(REV / 'A6_fairness_decomposition_per_run.csv')
g = dec.groupby(['step', 'mode']).f1_macro.agg(['mean', 'std'])
labels = [('1 ', 'Classical XGBoost: all variables of the set, all training rows'), ('2 ', 'Same, restricted to the five selected variables'), ('3 ', 'Same, restricted to the 800-row subset B'),
          ('4 ', 'Same, with the hybrid learning rate (this is the ``XGB only\'\' control)'), ('5 stack', 'XGB + output of logistic regression fitted on subset A'), ('6 stack', 'XGB + output of a 29-weight MLP fitted on subset A')]
steps = sorted(dec.step.unique())
rows = [[txt] + [ms(*g.loc[([s for s in steps if s.startswith(pre)][0], m)]) for m in MODES] for pre, txt in labels]
hy = st[st.name == 'XGB-DRU-trained'].groupby('mode').f1_macro.agg(['mean', 'std'])
rows.append(['XGB + trained DRU output (stored result)'] + [ms(*hy.loc[m]) for m in MODES])
w('S5_fairness.tex', tab(['Configuration'] + list(MODES.values()), rows, 'p{0.5\\linewidth}ccc'))
wins = pd.read_csv(REV / 'A5_paired_wins_strict.csv')
w('S5_wins.tex', tab(['Compared with', 'F1', 'ROC AUC', 'False-alarm rate', 'Miss rate'], [[NAMES[r.vs], r.f1_macro, r.roc_auc_ovr, r.far, r.miss] for _, r in wins.iterrows()], 'lcccc'))
# S6: resampling
c = pd.read_csv(REV / 'A4_smotetomek_vs_none.csv')
w('S6_resampling.tex', tab(['Model', 'Inputs', 'F1 with', 'F1 without', 'AUC with', 'AUC without'], [[NAMES[r.model], MODES[r['mode']], f'{r.f1_macro_mean_smt:.3f}', f'{r.f1_macro_mean_none:.3f}', f'{r.roc_auc_ovr_mean_smt:.3f}', f'{r.roc_auc_ovr_mean_none:.3f}'] for _, r in c.iterrows()], 'llcccc'))
d2 = pd.read_csv(REV / 'D2_strict_no_resampling_dru_hybrid_per_run.csv'); g2 = d2.groupby('name')[['f1_macro', 'roc_auc_ovr', 'far', 'miss']].agg(['mean', 'std'])
g1 = st[st['mode'] == 'no_proxy_strict'].groupby('name')[['f1_macro', 'roc_auc_ovr', 'far', 'miss']].agg(['mean', 'std'])
extra = {'XGB-stack-LogReg(A)': 'XGB + logistic-regression output (control)', 'XGB-stack-MLP4(A)': 'XGB + 29-weight MLP output (control)'}
rows = [[NAMES[n], ms(*g1.loc[n, 'f1_macro']), ms(*g2.loc[n, 'f1_macro']), ms(*g2.loc[n, 'roc_auc_ovr']), ms(*g2.loc[n, 'far']), ms(*g2.loc[n, 'miss'])] for n in list(NAMES)[5:]]
rows += [[t, '--', ms(*g2.loc[n, 'f1_macro']), ms(*g2.loc[n, 'roc_auc_ovr']), ms(*g2.loc[n, 'far']), ms(*g2.loc[n, 'miss'])] for n, t in extra.items()]
w('S6_dru_no_resampling.tex', tab(['Model', 'F1 with resampling', 'F1 without', 'ROC AUC without', 'False-alarm rate without', 'Miss rate without'], rows, 'lccccc'))
syn = pd.read_csv(REV / 'A4b_synthetic_diagnostics.csv')
w('S6_synthetic.tex', tab(['Seed', 'Real training rows', 'Synthetic rows', 'Synthetic share', 'Magnetic-field norm outside real 1--99\\% range', 'Acceleration norm outside real 1--99\\% range'],
                         [[r.seed, r.n_train_real, r.n_synthetic, f'{r.synthetic_frac:.2f}', f'{r.syn_mag_norm_outside_real_1_99pct:.2f}', f'{r.syn_acc_norm_outside_real_1_99pct:.2f}'] for _, r in syn.iterrows()], 'cccccc'))
# S7: cost
j = json.load(open(REV / 'D1_dru_cost_and_reproduction.json'))
rows = [['Qubits / trainable weights / circuit depth', f"{j['config']['n_qubits']} / {j['config']['n_trainable']} / {j['config']['depth']}"],
        ['Gates', ', '.join(f'{v} {k}' for k, v in j['config']['gate_counts'].items())],
        ['Simulator', 'exact state vector (Qiskit StatevectorEstimator), no shots, no noise'],
        ['Host', j['host']['processor'] + '; ' + j['host']['platform']],
        ['Training rows per fit', j['config']['n_train_A']], ['Optimizer iterations (COBYLA)', j['config']['max_iter']],
        ['Loss evaluations per fit (measured)', j['fit']['loss_evaluations']], ['Simulated circuit runs per fit (measured)', f"{j['fit']['circuit_evaluations']:,}".replace(',', '\\,')],
        ['Wall-clock time per fit (measured; stored run)', f"{j['fit']['wall_clock_s']} s; {j['fit']['stored_wall_clock_s']} s"],
        ['Training loss, first / last evaluation', f"{j['fit']['loss_first']} / {j['fit']['loss_last']}"],
        ['Inference, one sample, circuit (median / 95th percentile)', f"{j['inference']['single_sample_dru_ms_median']} / {j['inference']['single_sample_dru_ms_p95']} ms"],
        ['Inference, one sample, hybrid (median / 95th percentile)', f"{j['inference']['single_sample_hybrid_ms_median']} / {j['inference']['single_sample_hybrid_ms_p95']} ms"],
        ['Inference, one sample, XGBoost alone (median)', f"{j['inference']['single_sample_xgb_only_ms_median']} ms"],
        ['Stored result reproduced (F1, seed 0, physical-only)', f"{j['reproduction']['f1_macro']:.4f} vs. {j['reproduction']['f1_macro_stored']:.4f}"]]
w('S7_cost.tex', tab(['Quantity', 'Value'], rows, 'p{0.45\\linewidth}p{0.48\\linewidth}'))
# numbers used in the supplement text
ac = pd.read_csv(REV / 'A1_autocorr_strict.csv', index_col=0).median().round(2).to_dict()
json.dump({'embargo_train_rows': n_tr, 'autocorr_median': ac, 'synthetic_share_mean': float(syn.synthetic_frac.mean()), 'mag_outside_mean': float(syn.syn_mag_norm_outside_real_1_99pct.mean())}, open(OUT / 'numbers.json', 'w'), indent=1)
print('supplement tables written to', OUT)
