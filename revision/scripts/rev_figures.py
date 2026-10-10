"""Figures for the revised manuscript (all text is 8 pt at printed size, as the reviewers required), drawn only from stored results
(results/*/b2_*_per_run.csv) and from the revision outputs in revision/results.
Reads the per-run files in revision/results and results/, and writes vector
PDFs to revision/figures/rev_*.pdf."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rev_common import REV_DIR as REV, FIG_DIR as FIG
BLUE, ORANGE, TEAL, VIOLET, INK, MUTED, GRID = '#2a78d6', '#d9622b', '#1f9d8a', '#8a56c9', '#0b0b0b', '#52514e', '#d9d8d3'
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})

# ── Fig. 1: timeline with the ten blocks ─────────────────────────────────
runs = pd.read_csv(REV / 'A1_label_runs.csv'); blocks = pd.read_csv(REV / 'A1_blocks.csv')
names = {0: 'Nominal', 1: 'GPS fault', 2: 'Accelerometer fault', 3: 'Engine fault', 4: 'RC fault'}
fig, ax = plt.subplots(figsize=(3.5, 1.6))
for _, b in blocks.iterrows():
    if b.block % 2 == 0: ax.axvspan(b.t_start_s, b.t_end_s, color='#f0efec', lw=0)
    ax.text((b.t_start_s + b.t_end_s) / 2, 1.95, str(int(b.block)), ha='center', va='center', fontsize=8, color=MUTED)
ax.text(-4, 1.95, 'Block', ha='right', va='center', fontsize=8, color=MUTED)
GAPS = [(14.3, 30.6), (491.1, 499.5)]   # logging gaps > 1 s (A1_episodes.csv)
for _, r in runs.iterrows():
    y = 0 if r.label == 0 else 1
    segs = [(r.t_start_s, r.t_end_s)]
    for g0, g1 in GAPS:
        segs = [p for a0, a1 in segs for p in (((a0, g0), (g1, a1)) if a0 < g0 and a1 > g1 else ((a0, a1),))]
    for a0, a1 in segs: ax.plot([a0, a1], [y, y], lw=4.5, solid_capstyle='butt', color=MUTED if r.label == 0 else ORANGE)
    if r.label: ax.text((r.t_start_s + r.t_end_s) / 2, 1.42, names[r.label].replace(' fault', '').replace('Accelerometer', 'Accel.').replace('Engine', 'Engine'), ha='center', va='center', fontsize=8)
ax.set_yticks([0, 1]); ax.set_yticklabels(['Nominal', 'Fault']); ax.set_ylim(-0.45, 2.2); ax.set_xlim(0, 508)
ax.set_xlabel('Time since start of log (s)'); ax.tick_params(length=2)
fig.tight_layout(pad=0.3); fig.savefig(FIG / 'rev_timeline_blocks.pdf'); plt.close(fig)

# ── Fig. 2: what the split design does ───────────────────────────────────
rnd = pd.read_csv(REV / 'A2_random_row_per_run.csv'); emb = pd.read_csv(REV / 'A3a_embargo_per_run.csv')
ev = pd.read_csv(REV / 'A7_detection_by_event_exposure_pooled.csv')
cats = ['Random\nrows', 'Time\nblocks', 'Blocks,\n5 s gap', 'Blocks,\n15 s gap', 'Blocks,\n30 s gap', 'Blocks,\n48 s gap']
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={'width_ratios': [1.2, 1.1]})
for mode, col, mk, lab, dx in (('full', BLUE, 'o', 'All 72 variables', -0.06), ('no_proxy_strict', ORANGE, 's', '22 physical-only variables', 0.06)):
    per = [rnd[rnd['mode'] == mode].groupby('model').f1_macro.mean()] + [emb[(emb['mode'] == mode) & (emb.embargo == E)].groupby('model').f1_macro.mean() for E in (0, 50, 150, 300, 482)]
    m = [p.mean() for p in per]; x = np.arange(len(cats)) + dx
    pass
    a.plot(x, m, marker=mk, ms=4.5, lw=1.4, color=col, label=lab, mec='white', mew=0.6)
    for xi, mi in list(zip(x, m))[:2] + list(zip(x, m))[4:5]: a.text(xi, mi + (0.085 if dx < 0 else -0.095) * (1 if abs(m[0]-mi) > 1e-9 or dx < 0 else 1), f'{mi:.2f}', ha='center', va='center', fontsize=8, color=MUTED)
a.set_xticks(range(len(cats))); a.set_xticklabels(cats, fontsize=8); a.set_ylim(0.15, 1.12); a.set_ylabel('F1 score (mean of 5 models)')
a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True); a.legend(loc='upper right', handlelength=1.6)
a.set_title('(a) Same data and models, different split', loc='left')
e = ev[ev['mode'] == 'no_proxy_strict']; models = ['LogReg', 'SVM-RBF', 'MLP', 'RandomForest', 'XGBoost']; lbl = ['Logistic\nregr.', 'SVM', 'MLP', 'Random\nforest', 'XGBoost']
w = 0.36
for g, col, off, lab, hatch in (('event partly seen in training', BLUE, -w / 2 - 0.02, 'Yes, partly', None), ('event unseen in training', ORANGE, w / 2 + 0.02, 'No', '////')):
    v = [float(e[(e.model == m) & (e.group == g)].detection_rate.iloc[0]) for m in models]
    bars = b.bar(np.arange(5) + off, v, w, color=col, label=lab, hatch=hatch, edgecolor='white', lw=0.4)
    for xi, vi in zip(np.arange(5) + off, v): b.text(xi, vi + 0.02, f'{vi:.2f}', ha='center', fontsize=8)
b.set_xticks(range(5)); b.set_xticklabels(lbl); b.set_ylim(0, 1.25); b.set_yticks([0, 0.25, 0.5, 0.75, 1]); b.set_ylabel('Fault samples detected')
b.yaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True); b.legend(loc='upper center', ncol=2, handlelength=1.4, borderaxespad=0.1)
b.set_title('(b) Was the fault interval seen in training?', loc='left')
fig.tight_layout(pad=0.4, w_pad=1.5); fig.savefig(FIG / 'rev_split_lessons.pdf'); plt.close(fig)

# ── Fig. 3: all models, physical-only features, per-partition spread ─────
st = pd.read_csv(REV / 'A5_all_models_per_run_with_conventional_far.csv'); st = st[st['mode'] == 'no_proxy_strict']
order = [('LogReg', 'Logistic regression', 'c'), ('SVM-RBF', 'SVM (RBF)', 'c'), ('MLP', 'MLP', 'c'), ('RandomForest', 'Random forest', 'c'), ('XGBoost', 'XGBoost', 'c'),
         ('DRU-Binary-B2', 'DRU circuit alone', 'q'), ('XGB-raw', 'XGB only', 'h'), ('XGB-PCA', 'XGB + PCA', 'h'), ('XGB-Poly2', 'XGB + polynomial', 'h'),
         ('XGB-RandomRBF', 'XGB + random RBF', 'h'), ('XGB-DRU-untrained', 'XGB + untrained DRU', 'h'), ('XGB-DRU-trained', 'XGB + trained DRU', 't')]
sty = {'c': (BLUE, 'o', 'Classical'), 'q': (VIOLET, 'D', 'Simulated circuit'), 'h': (TEAL, 's', 'Hybrid controls'), 't': (ORANGE, '^', 'Hybrid, trained circuit')}
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.65), sharey=True)
for ax, (k, ttl) in zip(axs, (('f1_macro', 'F1 score'), ('roc_auc_ovr', 'ROC AUC'), ('far', 'False-alarm rate'), ('miss', 'Miss rate'))):
    for i, (nm, lab, grp) in enumerate(order):
        v = st[st.name == nm][k].values; y = len(order) - 1 - i; col, mk, _ = sty[grp]
        ax.scatter(v, np.full(len(v), y), s=7, color='#9a9994', alpha=0.75, lw=0, zorder=2)
        ax.scatter([v.mean()], [y], s=34, color=col, marker=mk, edgecolor='white', lw=0.7, zorder=3)
    ax.set_xlim(-0.04, 1.04); ax.set_xticks([0, 0.5, 1]); ax.set_title(ttl, fontsize=8, loc='left'); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if k in ('f1_macro', 'roc_auc_ovr'): ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.tick_params(length=2)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([o[1] for o in order][::-1]); axs[0].set_ylim(-0.7, len(order) - 0.3)
h = [plt.Line2D([], [], marker=sty[g][1], color=sty[g][0], ls='', ms=5.5, mec='white', label='Mean: ' + sty[g][2].lower()) for g in 'cqht']
h.append(plt.Line2D([], [], marker='o', color='#9a9994', ls='', ms=3, label='One of the 10 partitions'))
fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, handletextpad=0.1, columnspacing=1.2, fontsize=8)
fig.tight_layout(pad=0.4, rect=(0, 0.17, 1, 1)); fig.savefig(FIG / 'rev_strict_models.pdf'); plt.close(fig)
print('figures written')
