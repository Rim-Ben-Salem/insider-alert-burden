"""Run from the repository root. Tables II-V and Fig. 2 of the narrowed paper from revisions 10 and 11 (seed means)."""
import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
R10 = sys.argv[1] if len(sys.argv) > 1 else 'results/11_unilateral_defenses_power/'
R11 = sys.argv[2] if len(sys.argv) > 2 else 'results/12_baselines_sensitivity_clearance/'
O = sys.argv[3] if len(sys.argv) > 3 else 'tables/'
FIG = sys.argv[4] if len(sys.argv) > 4 else 'figures/'
import fairirs_rev10 as n10
REL = [('r4.2', 'r42', 51), ('r5.2', 'r52', 76)]
DETS = ['IF', 'AE', 'LR', 'GBM', 'LSTM']
P = 'B11c deficit apportionment (outcome-aware)'
N1 = pd.concat([pd.read_csv(R10 + f'n1_seeds_{t}.csv') for _, t, _ in REL])
N5 = pd.concat([pd.read_csv(R10 + f'n5_power_{t}.csv') for _, t, _ in REL])
O1 = pd.concat([pd.read_csv(R11 + f'o1_baselines_{t}.csv') for _, t, _ in REL])
def fmtp(p): return f'{p:.3f}' if p >= 0.001 else '$<$0.001'
# ---------- Table II: burden, seed means
L = []
for rel, t, nins in REL:
    L.append(f'\\multicolumn{{8}}{{l}}{{\\emph{{{rel}}} ({nins} test insiders)}} \\\\')
    for d in DETS:
        x = N1[(N1.release == rel) & (N1.detector == d) & (N1.policy == 'B0 global')]
        o = N5[(N5.release == rel) & (N5.detector == d) & (N5.kind == 'observed')].iloc[0]
        L.append(f'{d} & {len(x)} & {x.insiders_detected.mean():.1f} & {x.gini_F.mean():.3f} & {x.top1_share.mean():.2f} & {x.max_F_benign.mean():.0f} & {x.delta_FAR.mean():.4f} & {fmtp(o.p_range)} / {fmtp(o.p_weighted)} \\\\')
tab = r"""\begin{table}[!t]
\caption{Alert Burden Under Global Top-$k$ Alerting ($b=1\%$, Functional Units; Means Over the Retrained Seeds). $p$: Permutation Tests With Group Labels Shuffled Across Employees, on the Range of Group Rates / a Size-Weighted Variance of Group Rates (First Seed)\label{tab:rq1}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
Det. & Seeds & Detected & Gini($F$) & Top-1\% & $\max F_i$ & $\Delta_{\mathrm{FAR}}$ & $p$ (range / weighted) \\
\midrule
""" + '\n'.join(L) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(O + 'tab_rq1_11.tex', 'w').write(tab)
# ---------- Table III: benefit, seed means
POLS = [('global', 'Global'), ('cooldown 30 d, override 0.999', 'Cooldown'), ('repeat cap: 1 alert per 30 days, override 0.999', 'Repeat cap'), ('deficit apportionment (outcome-aware)', 'Parity')]
L = []
for rel, t, _ in REL:
    L.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in DETS:
        cells = []
        for p, _ in POLS:
            x = O1[(O1.release == rel) & (O1.detector == d) & (O1.policy == p)]
            cells.append(f'{x.insiders_detected.mean():.1f} / {x.max_F_benign.mean():.0f}')
        L.append(d + ' & ' + ' & '.join(cells) + ' \\\\')
tab = r"""\begin{table}[!t]
\caption{Benefit of History-Dependent Rules: Insiders Detected / Alerts on the Most-Flagged Benign Employee ($\max F_i$), Means Over the Retrained Seeds. Cooldown: 30 Days After Clearance; Repeat Cap: No Alert After an Alert in the Last 30 Days, Whatever the Verdict; Both With Override $0.999$. Parity: Outcome-Aware Deficit Apportionment ($b=1\%$)\label{tab:benefit}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Det. & """ + ' & '.join(l for _, l in POLS) + r""" \\
\midrule
""" + '\n'.join(L) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(O + 'tab_benefit11.tex', 'w').write(tab)
# ---------- Table V: unilateral timing
S2 = pd.concat([pd.read_csv(R10 + f'n2_summary_{t}.csv') for _, t, _ in REL])
O4 = pd.concat([pd.read_csv(R11 + f'o4_online_keep_{t}_per_insider.csv') for _, t, _ in REL])
k = O4[O4.attack.str.startswith('online')].groupby(['release', 'detector', 'seed', 'policy', 'user'], sort=False)['d_keep'].mean().reset_index().rename(columns={'d_keep': 'd'})
k['attack'] = 'online, keeps timing'
SK = n10.n2_summary(k)
SK.to_csv(O + 'o4_keep_summary.csv', index=False)
S = pd.concat([S2, SK])
S = S[S.policy == P]
order = [('r4.2', 'GBM'), ('r5.2', 'GBM'), ('r4.2', 'LSTM'), ('r5.2', 'LSTM'), ('r4.2', 'AE'), ('r5.2', 'AE')]
A = [('random', 'Random (control)'), ('online deficit, q=0.9', 'Online: reports and colleagues$^\\ddagger$'),
     ('online, keeps timing', '\\quad keeps timing if no trigger'), ('hindsight deficit', 'Hindsight: same signals'),
     ('knows allocation: largest cut', 'Knows the allocation'), ('oracle: maximizes excess loss', 'Oracle: knows every score')]
L = []; TS = {}
for a, lab in A:
    cells, vals, ns = [], [], 0
    for rel, det in order:
        r = S[(S.release == rel) & (S.detector == det) & (S.attack == a)].iloc[0]
        st = '$^\\ast$' if r.lo > 1e-9 else ''
        ns += st != ''
        vals.append(r.excess); cells.append(f'{r.excess:.1f}{st}')
    TS[a] = (vals, [S[(S.release == rel) & (S.detector == det) & (S.attack == a)].iloc[0].lo > 1e-9 for rel, det in order])
    L.append(lab + ' & ' + ' & '.join(cells) + f' & {np.mean(vals):.1f} & {ns} \\\\')
seeds = [int(S[(S.release == rel) & (S.detector == det)].seeds.iloc[0]) for rel, det in order]
tab = r"""\begin{table}[!t]
\caption{Timing Attacks Against Outcome-Aware Deficit Apportionment, One Insider at a Time Against the Others' Behavior: Excess Loss Over the Global Policy on the Same Relocated Attack, Summed Over Insiders and Averaged Over Seeds (""" + ', '.join(f'{s}' for s in seeds) + r""" Seeds), the Mean Over the Six Settings and the Number of Settings Whose 95\% Interval, Resampling Seeds and Insiders, Excludes Zero ($^\ast$). $^\ddagger$Primary Attacker, Which Acts on Its Last Possible Day When Its Signal Never Triggers ($b=1\%$)\label{tab:timing}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{AE} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Attacker & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & Mean & Sig. \\
\midrule
""" + '\n'.join(L) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(O + 'tab_timing11.tex', 'w').write(tab)
# ---------- Table IV: farming variants (best configuration per run, pooled)
R12 = sys.argv[5] if len(sys.argv) > 5 else 'results/13_late_verdicts_permutation/'
N3 = pd.concat([pd.read_csv(R10 + f'n3_farming_{t}.csv') for _, t, _ in REL])
N3 = N3[~N3.variant.str.startswith('verdicts')]          # revision 10 late-verdict runs had a bug; replaced by revision 12
P2 = pd.concat([pd.read_csv(R12 + f'p2_farming_{t}.csv') for _, t, _ in REL])
P2 = P2[(P2.detector != 'LSTM') & P2.variant.isin(['verdicts 7 days late', 'verdicts 14 days late', 'no override, verdicts 7 days late'])]
N3 = pd.concat([N3, P2]).reset_index(drop=True)
best = N3.loc[N3.groupby(['release', 'detector', 'seed', 'variant'], sort=False)['evaded'].idxmax()]
V = [('cooldown 30 d, override 0.999', 'Cooldown (30 d, override $0.999$)'), ('no override', '\\quad without override'),
     ('verdicts 7 days late', '\\quad verdicts $7$ days late'), ('verdicts 14 days late', '\\quad verdicts $14$ days late'),
     ('no override, verdicts 7 days late', '\\quad no override, verdicts $7$ days late'), ('insider-level verdicts', '\\quad probes during the episode confirmed'), ('no backfill', '\\quad suppressed alerts not replaced'),
     ('defense: re-check 10% of suppressed alerts', '\\quad re-check $10\\%$ of suppressed alerts'),
     ('defense: farming-signature monitor (score >= 0.99 within 7 days of a clearance)', '\\quad escalate $s\\ge0.99$ within $7$ d of a clearance'),
     ('defense: 3-day cooldown after clearing a score >= 0.99', '\\quad $3$-d cooldown after clearing $s\\ge0.99$'),
     ('global (no cooldown)', 'Global policy')]
L = []
for v, lab in V:
    cells = []
    for rel, _, _ in REL:
        x = best[(best.release == rel) & (best.variant == v)]
        cells.append(f'{100 * x.evaded.sum() / x.detected_no_farming.sum():.0f}')
    x = best[best.variant == v]
    y = N3[N3.variant == v].groupby(['release', 'detector', 'seed']).first().reset_index()
    det = y.groupby(['release', 'detector']).insiders_detected_variant.mean().sum()
    mf = y.groupby(['release', 'detector']).max_F_variant.mean().mean()
    L.append(lab + ' & ' + ' & '.join(cells) + f' & {100 * x.evaded.sum() / x.detected_no_farming.sum():.1f} & {det:.0f} & {mf:.0f} \\\\')
tab = r"""\begin{table}[!t]
\caption{Discount Farming Against the Cooldown, With Probes Rescored, Capped at the Role's 99th Percentile and Always Cleared: Share of Insiders Detected Without Farming Who Evade With It (\%, Best of Four Configurations per Run, Pooled Over IF, AE, LR and GBM and Their Seeds), Insiders Detected Without Farming (Summed Over the Eight Settings, Seed Means) and Mean $\max F_i$ ($b=1\%$; Farmers: the 45 of 51 r4.2 and 73 of 76 r5.2 Insiders With a Benign Test Day Before Their First Malicious Day)\label{tab:farm}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{3}{c}{Evade (\%)} & & \\
\cmidrule(lr){2-4}
Variant & r4.2 & r5.2 & All & Detected & $\max F_i$ \\
\midrule
""" + '\n'.join(L) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(O + 'tab_farm11.tex', 'w').write(tab)
# ---------- Fig. 2
O2 = pd.concat([pd.read_csv(R11 + f'o2_sweep_{t}.csv') for _, t, _ in REL])
O3 = pd.concat([pd.read_csv(R11 + f'o3_clearance_{t}.csv') for _, t, _ in REL])
m = O2[O2.budget == 0.01].groupby(['release', 'detector', 'policy'], sort=False).mean(numeric_only=True).reset_index()
s = m.groupby('policy', sort=False).agg(det=('insiders_detected', 'sum'), mf=('max_F_benign', 'mean'), ev=('farm_evaded', 'sum'), de=('farm_detected_no_farming', 'sum'))
s['share'] = 100 * s.ev / s.de
s.to_csv(O + 'o2_tradeoff.csv')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED = '#2b2b29', '#8a897f'
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1.1, 1, 1.15]))
a = ax[0]
ov = [('cooldown 30 d, override 0.99', '0.99'), ('cooldown 30 d, override 0.995', '0.995'), ('cooldown 30 d, override 0.999', '0.999'), ('cooldown 30 d, override 0.9995', '0.9995'), ('cooldown 30 d, override none', 'none')]
ln = [('cooldown 3 d, override 0.999', '3 d'), ('cooldown 7 d, override 0.999', '7 d'), ('cooldown 14 d, override 0.999', '14 d'), ('cooldown 30 d, override 0.999', '30 d'), ('cooldown 60 d, override 0.999', '60 d')]
xs = [s.loc[p, 'mf'] for p, _ in ov]; ys = [s.loc[p, 'share'] for p, _ in ov]
a.plot(xs, ys, '-o', color=C[0], ms=4.5, label='override (30 d)')
OFF = {'0.99': (-4, 6), '0.995': (4, 4), '0.999': (-26, -3), '0.9995': (-27, -9), 'none': (5, -2)}
for (p, l), x, y in zip(ov, xs, ys):
    a.annotate(l, (x, y), xytext=OFF[l], textcoords='offset points', color=INK, fontsize=6.3)
xs = [s.loc[p, 'mf'] for p, _ in ln]; ys = [s.loc[p, 'share'] for p, _ in ln]
a.plot(xs, ys, '--s', color=C[1], ms=4, label='length (override 0.999)')
for (p, l), x, y in zip(ln, xs, ys):
    if l != '30 d':
        a.annotate(l, (x, y), xytext=(-6, 5) if l == '60 d' else (5, -6), textcoords='offset points', color=INK, fontsize=6.3)
p = 'repeat cap: 1 per 30 days, override 0.999'
a.plot([s.loc[p, 'mf']], [s.loc[p, 'share']], '^', color=C[2], ms=5.5, label='repeat cap, 1 per 30 d')
g = s.loc['global']
a.plot([g.mf], [g.share], 'D', color=MUTED, ms=4.5, label='global policy')
a.set_xlabel('Alerts on the most-flagged benign employee'); a.set_ylabel('Evasions by farming (% of detected)')
a.set_xlim(0, 130); a.set_ylim(0, 25)
a.legend(frameon=False, fontsize=6, loc='upper right', bbox_to_anchor=(1.02, 1.0))
a.set_title('(a) Burden against farming exposure', fontsize=7.5, loc='left', color=INK)
a = ax[1]
mm = O3.groupby(['release', 'detector', 'seed', 'variant', 'knowledge', 'p_clear'], sort=False).mean(numeric_only=True).reset_index()
mm = mm.groupby(['release', 'detector', 'variant', 'knowledge', 'p_clear'], sort=False).mean(numeric_only=True).reset_index()
mm = mm[mm.knowledge == 'generic']
VV = [('cooldown 30 d, no override', 'cooldown, no override'), ('cooldown 30 d, override 0.999', 'cooldown, override 0.999'), ('repeat cap: 1 per 30 days, override 0.999', 'repeat cap'), ('global', 'global policy')]
mk = ['o', 's', '^', 'D']
for i, (v, l) in enumerate(VV):
    q = mm[mm.variant == v].groupby('p_clear')[['net_attacker_gain', 'detected_no_farming']].sum()
    a.plot(q.index, 100 * q.net_attacker_gain / q.detected_no_farming, '-' + mk[i], color=C[i] if i < 3 else MUTED, ms=4, label=l)
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlabel('Probability that a probe alert is cleared'); a.set_ylabel('Net gain of farming (% of detected)')
a.set_xticks([0.25, 0.5, 0.75, 1.0]); a.legend(frameon=False, fontsize=6.2, loc='upper left')
a.set_title('(b) Farming with uncertain clearance', fontsize=7.5, loc='left', color=INK)
a = ax[2]
names = ['Random', 'Online', 'Online,\nkeeps', 'Hind-\nsight', 'Knows\nalloc.', 'Oracle']
keys = ['random', 'online deficit, q=0.9', 'online, keeps timing', 'hindsight deficit', 'knows allocation: largest cut', 'oracle: maximizes excess loss']
for j, kk in enumerate(keys):
    v, sg = TS[kk]
    xj = j + np.linspace(-0.18, 0.18, len(v))
    a.scatter(xj, v, s=14, c=[C[1] if z else C[0] for z in sg], zorder=3, edgecolors='white', linewidths=0.6)
    a.plot([j - 0.3, j + 0.3], [np.mean(v)] * 2, color=INK, lw=1.2)
a.axhline(0, color=MUTED, lw=0.8)
a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=6)
a.set_ylabel('Excess loss (insiders)')
a.scatter([], [], s=14, c=C[1], label='95% interval excludes 0'); a.scatter([], [], s=14, c=C[0], label='includes 0')
a.legend(frameon=False, fontsize=6.2, loc='upper left')
a.set_title('(c) Timing, one insider at a time', fontsize=7.5, loc='left', color=INK)
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG + 'security.pdf'); fig.savefig(O + 'security.png', dpi=170)
print(s.round(2).to_string())
