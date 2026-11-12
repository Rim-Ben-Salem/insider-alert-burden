"""Run from the repository root after tables_main_final_and_fig2.py. Supplementary tables for revisions 10-11 (seed means, per-employee baselines, power, sweep,
farming configurations and clearance, unilateral timing for every parity rule)."""
import pandas as pd, numpy as np, sys
R10 = sys.argv[1] if len(sys.argv) > 1 else 'results/11_unilateral_defenses_power/'
R11 = sys.argv[2] if len(sys.argv) > 2 else 'results/12_baselines_sensitivity_clearance/'
O = sys.argv[3] if len(sys.argv) > 3 else 'tables/'
TK = O + 'o4_keep_summary.csv'
REL = [('r4.2', 'r42'), ('r5.2', 'r52')]
DETS = ['IF', 'AE', 'LR', 'GBM', 'LSTM']
def rd(prefix, base):
    return pd.concat([pd.read_csv(base + f'{prefix}_{t}.csv') for _, t in REL])
def table(label, caption, cols, rows, wide=False, size='\\footnotesize'):
    env = 'table*' if wide else 'table'
    w = '\\textwidth' if wide else '\\columnwidth'
    return (f'\\begin{{{env}}}[!t]\n\\caption{{{caption}\\label{{{label}}}}}\n\\centering\n{size}\n\\setlength{{\\tabcolsep}}{{3pt}}\n'
            f'\\resizebox{{{w}}}{{!}}{{%\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}}\n' + f'\\end{{{env}}}\n')
out = []
# --- seeds
N1 = rd('n1_seeds', R10)
P = [('B0 global', 'Global'), ('B0 + cooldown 30 days', 'Cooldown'), ('B0 + cooldown 30 days, no override', 'No override'),
     ('B0 + cooldown 30 days, no backfill', 'No backfill'), ('B0 + scheduling', 'Scheduler'), ('B4c outcome-aware', 'Exact rule'),
     ('B11c deficit apportionment (outcome-aware)', 'Deficit app.'), ('randomized global, rho=0.2', 'Random., $\\rho=0.2$'), ('randomized global, rho=0.5', 'Random., $\\rho=0.5$')]
rows = ['Det. & Seeds & ' + ' & '.join(l for _, l in P) + ' \\\\', '\\midrule']
for rel, _ in REL:
    rows.append(f'\\multicolumn{{{len(P) + 2}}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in DETS:
        x = N1[(N1.release == rel) & (N1.detector == d)]
        cells = []
        for p, _ in P:
            y = x[x.policy == p]
            cells.append(f'{y.insiders_detected.mean():.1f} / {y.max_F_benign.mean():.0f} / {1000 * y.delta_FAR.mean():.1f}')
        rows.append(f'{d} & {x.seed.nunique()} & ' + ' & '.join(cells) + ' \\\\')
out.append(table('tab:s-seeds11', 'Burden and Detection Over Retrained Seeds: Insiders Detected / $\\max F_i$ / $1000\\,\\Delta_{\\mathrm{FAR}}$, Means Over Seeds ($b=1\\%$). Randomized Global: Each Alert Is Replaced, With Probability $\\rho$, by an Employee Drawn From the Next Five Ranks', 'l' + 'c' * (len(P) + 1), rows, wide=True))
# --- daily budget
# --- per-employee baselines
rows = ['Det. & Global & Cooldown & No override & Parity & Farming (cooldown) & Timing oracle & Online \\\\', '\\midrule']
for rel, t in REL:
    b = pd.read_csv(R10 + f'n4_user_burden_{t}.csv'); f = pd.read_csv(R10 + f'n4_user_farming_{t}.csv').reset_index(drop=True)
    s = pd.read_csv(R10 + f'n4_user_timing_{t}_summary.csv')
    rows.append(f'\\multicolumn{{8}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in ['IF-user', 'AE-user', 'GBM-user']:
        g = lambda p: b[(b.detector == d) & (b.policy == p)].iloc[0]
        cells = [f'{g(p).insiders_detected} / {g(p).max_F_benign}' for p in ('B0 global', 'B0 + cooldown 30 days', 'B0 + cooldown 30 days, no override', 'B11c deficit apportionment (outcome-aware)')]
        ff = f[(f.detector == d) & (f.variant == 'cooldown 30 d, override 0.999')]
        bb = ff.loc[ff.evaded.idxmax()]
        so = s[(s.detector == d) & (s.policy.str.contains('B11c')) & (s.attack == 'oracle: maximizes excess loss')].iloc[0]
        sn = s[(s.detector == d) & (s.policy.str.contains('B11c')) & (s.attack == 'online deficit, q=0.9')].iloc[0]
        rows.append(f'{d.replace("-user", "")} & ' + ' & '.join(cells) + f' & {bb.evaded} of {bb.detected_no_farming} & {so.excess:.0f} [{so.lo:.0f}, {so.hi:.0f}] & {sn.excess:.1f} [{sn.lo:.0f}, {sn.hi:.0f}] \\\\')
out.append(table('tab:s-user11', 'Detectors on Deviations From Each Employee\'s Own Training Baseline (One Run): Insiders Detected / $\\max F_i$ Under Each Policy; Farmers Who Evade the Cooldown (Best Configuration); Excess Loss of the Oracle and of the Primary Online Attacker Against Deficit Apportionment, One Insider at a Time, With 95\\% Intervals ($b=1\\%$)', 'lccccccc', rows, wide=True))
# --- power
N5 = rd('n5_power', R10)
rows = ['& & \\multicolumn{2}{c}{Observed $p$} & \\multicolumn{2}{c}{Smallest unit $\\times2$} & \\multicolumn{2}{c}{Largest unit $\\times2$} \\\\', '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}', 'Rel. & Det. & Range & Weighted & Range & Weighted & Range & Weighted \\\\', '\\midrule']
for rel, _ in REL:
    for d in DETS:
        x = N5[(N5.release == rel) & (N5.detector == d)]
        o = x[x.kind == 'observed'].iloc[0]
        sm = x[(x.kind == 'power') & x.group.str.startswith('smallest') & (x.effect == 1.0)].iloc[0]
        lg = x[(x.kind == 'power') & x.group.str.startswith('largest') & (x.effect == 1.0)].iloc[0]
        rows.append(f'{rel} & {d} & {o.p_range:.3f} & {o.p_weighted:.4f} & {sm.p_range:.2f} & {sm.p_weighted:.2f} & {lg.p_range:.2f} & {lg.p_weighted:.2f} \\\\')
sizes = N5[N5.kind == 'power'].groupby('release').group.unique()
out.append(table('tab:s-power', 'Permutation Tests of Group Disparity Under the Global Policy (First Seed, 2000 Permutations): Observed $p$-Values for the Range of Group False-Alert Rates and for Their Size-Weighted Variance, and Power at the 5\\% Level When the False Alerts of the Smallest or the Largest Functional Unit Are Raised by Their Own Rate (Poisson Additions, 50 Injections, 200 Permutations Each). Power Is Not Defined Relative to a Null Disparity: Where the Observed $p$ Is Already Below $0.05$, Power Is High at Any Effect', 'llcccccc', rows))
# --- sweep
O2 = rd('o2_sweep', R11)
m = O2.groupby(['release', 'detector', 'policy', 'budget'], sort=False).mean(numeric_only=True).reset_index()
s = m.groupby(['policy', 'budget'], sort=False).agg(det=('insiders_detected', 'sum'), mf=('max_F_benign', 'mean'), al=('alerted_employees', 'sum'), dl=('delay_median', 'mean'), ev=('farm_evaded', 'sum'), de=('farm_detected_no_farming', 'sum')).reset_index()
rows = ['Policy & $b$ & Detected & $\\max F_i$ & Alerted & Delay (d) & Farming (\\%) \\\\', '\\midrule']
for _, r in s.iterrows():
    fv = f'{100 * r.ev / r.de:.1f}' if r.de > 0 else '--'
    rows.append(f'{r.policy.replace("cooldown", "Cooldown").replace("repeat cap", "Repeat cap").replace("global", "Global")} & {100 * r.budget:.0f}\\% & {r.det:.0f} & {r.mf:.0f} & {r.al:.0f} & {r.dl:.1f} & {fv} \\\\')
rows.append('\\midrule')
rows.append('\\multicolumn{7}{l}{Insiders detected at $b=2\\%$, global / cooldown (override $0.999$) / cooldown (override $0.995$)} \\\\')
for rel, _ in REL:
    for d in DETS:
        g = lambda p: m[(m.release == rel) & (m.detector == d) & (m.policy == p) & (m.budget == 0.02)].insiders_detected.iloc[0]
        rows.append(f'\\multicolumn{{2}}{{l}}{{{rel} {d}}} & \\multicolumn{{5}}{{l}}{{{g("global"):.1f} / {g("cooldown 30 d, override 0.999"):.1f} / {g("cooldown 30 d, override 0.995"):.1f}}} \\\\')
out.append(table('tab:s-sweep', 'Cooldown Sensitivity: Insiders Detected and Employees Alerted (Summed Over the Ten Detector--Release Settings), Mean $\\max F_i$, Mean Median Days From an Insider\'s First Malicious Day to Their First Alert, and Share of Detected Insiders Who Evade by Farming (Generic Profile, Three Probes, Always Cleared; $b=1\\%$ Only); Means Over Up to Three Seeds per Detector', 'lcccccc', rows))
# --- farming configurations
R12 = sys.argv[4] if len(sys.argv) > 4 else 'results/13_late_verdicts_permutation/'
N3 = rd('n3_farming', R10)
N3 = N3[~N3.variant.str.startswith('verdicts')]
P2 = rd('p2_farming', R12)
N3 = pd.concat([N3, P2[(P2.detector != 'LSTM') & P2.variant.str.contains('verdicts')]])
c = N3.groupby(['variant', 'knowledge', 'n_probe', 'gap'], sort=False).agg(ev=('evaded', 'sum'), de=('detected_no_farming', 'sum')).reset_index()
rows = ['Variant & Own, 3 & Generic, 3 & Generic, 1 & Own, 3, 14 d earlier \\\\', '\\midrule']
for v in c.variant.unique():
    x = c[c.variant == v]
    rows.append(v.replace('>=', '$\\ge$').replace('%', '\\%') + ' & ' + ' & '.join(f'{100 * r.ev / r.de:.1f}' for _, r in x.iterrows()) + ' \\\\')
O3 = rd('o3_clearance', R11)
rows.append('\\midrule')
rows.append('\\multicolumn{5}{l}{Cooldown (override $0.999$), by detector, always cleared (all seeds, revision 11): own / generic} \\\\')
y = O3[(O3.p_clear == 1.0) & (O3.variant == 'cooldown 30 d, override 0.999')].groupby(['detector', 'knowledge'])[['evaded', 'detected_no_farming']].sum()
for d in DETS:
    a, b = y.loc[(d, 'own')], y.loc[(d, 'generic')]
    rows.append(f'{d} & \\multicolumn{{4}}{{l}}{{{100 * a.evaded / a.detected_no_farming:.1f} / {100 * b.evaded / b.detected_no_farming:.1f}}} \\\\')
out.append(table('tab:s-farm-config', 'Discount Farming by Configuration: Share of Insiders Detected Without Farming Who Evade With It (\\%), Pooled Over IF, AE, LR and GBM and Their Seeds (Revision 10); Last Rows: Every Detector Including the LSTM (Revision 11)', 'lcccc', rows))
# --- clearance
mm = O3.groupby(['release', 'detector', 'seed', 'variant', 'knowledge', 'p_clear'], sort=False).mean(numeric_only=True).reset_index()
mm = mm.groupby(['release', 'detector', 'variant', 'knowledge', 'p_clear'], sort=False).mean(numeric_only=True).reset_index()
q = mm.groupby(['variant', 'knowledge', 'p_clear'], sort=False)[['detected_no_farming', 'caught_with_farming', 'net_attacker_gain', 'evaded', 'exposed_by_probe']].sum().reset_index()
rows = ['Variant & Profile & $p$ & Caught without & Caught with & Net gain & Evaded & Exposed \\\\', '\\midrule']
for _, r in q.iterrows():
    rows.append(f'{r.variant} & {r.knowledge} & {r.p_clear:.2f} & {r.detected_no_farming:.0f} & {r.caught_with_farming:.0f} & {r.net_attacker_gain:.0f} ({100 * r.net_attacker_gain / r.detected_no_farming:.1f}\\%) & {r.evaded:.0f} & {r.exposed_by_probe:.0f} \\\\')
out.append(table('tab:s-clearance', 'Farming When Each Probe Alert Is Cleared With Probability $p$ and a Confirmed Probe Exposes the Insider: Farmers Caught Without and With Farming, Net Gain of the Attacker, Farmers Who Evade and Farmers Exposed by a Probe, Summed Over the Ten Detector--Release Settings (Seed Means, Up to Three Seeds; Three Draws for $p<1$)', 'llcccccc', rows, wide=True))
# --- unilateral timing, every rule
S2 = rd('n2_summary', R10)
SK = pd.read_csv(TK)
rows = ['Rule & Attacker & Mean & Range & Sig. \\\\', '\\midrule']
for pol in S2.policy.unique():
    for a in S2.attack.unique():
        x = S2[(S2.policy == pol) & (S2.attack == a)]
        rows.append(f'{pol} & {a} & {x.excess.mean():.1f} & [{x.excess.min():.1f}, {x.excess.max():.1f}] & {(x.lo > 1e-9).sum()} of {len(x)} \\\\')
    x = SK[SK.policy == pol]
    if len(x):
        rows.append(f'{pol} & online, keeps timing if no trigger & {x.excess.mean():.1f} & [{x.excess.min():.1f}, {x.excess.max():.1f}] & {(x.lo > 1e-9).sum()} of {len(x)} \\\\')
    rows.append('\\midrule')
rows = rows[:-1]
O4 = rd('o4_online_keep', R11.replace('rev11r/', 'rev11r/')) if False else pd.concat([pd.read_csv(R11 + f'o4_online_keep_{t}_per_insider.csv') for _, t in REL])
r = O4[O4.attack == 'relocated to its original start']
out.append(table('tab:s-timing-uni', 'Timing Attacks One Insider at a Time, Every Parity Rule: Excess Loss Over the Global Policy, Mean and Range Over the Six Detector--Release Settings (GBM, LSTM, AE), and the Number of Settings Whose 95\\% Interval, Resampling Seeds and Insiders, Excludes Zero. Relocating Each Insider to Its Own Original Start Changed Global Detection for ' + f'{100 * (r.G1 != r.G0).mean():.1f}' + '\\% of Insiders ($b=1\\%$)', 'llccc', rows, wide=True))
# --- baselines (O1)
O1 = rd('o1_baselines', R11)
P = [('global', 'Global'), ('global, confirmed insiders removed', 'Global, removed'), ('cooldown 30 d, override 0.999', 'Cooldown'), ('cooldown 30 d, confirmed insiders removed', 'Cooldown, removed'),
     ('repeat cap: 1 alert per 30 days, override 0.999', 'Cap 1/30 d'), ('repeat cap: 3 alerts per 30 days, override 0.999', 'Cap 3/30 d'), ('deficit apportionment (outcome-aware)', 'Deficit app.'), ('deficit apportionment, confirmed insiders removed', 'Deficit, removed')]
rows = ['Det. & ' + ' & '.join(l for _, l in P) + ' \\\\', '\\midrule']
for rel, _ in REL:
    rows.append(f'\\multicolumn{{{len(P) + 1}}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in DETS:
        x = O1[(O1.release == rel) & (O1.detector == d)]
        rows.append(d + ' & ' + ' & '.join(f'{x[x.policy == p].insiders_detected.mean():.1f} / {x[x.policy == p].max_F_benign.mean():.0f} / {x[x.policy == p].false_alerts.mean():.0f}' for p, _ in P) + ' \\\\')
out.append(table('tab:s-baselines11', 'Baselines Over Seeds: Insiders Detected / $\\max F_i$ / Total False Alerts. ``Removed\'\': a Confirmed Insider Is Not Alerted Again; Cap $n$/30 d: No Alert on an Employee Alerted $n$ Times in the Last 30 Days, Whatever the Verdict, Unless $s\\ge0.999$ ($b=1\\%$)', 'l' + 'c' * len(P), rows, wide=True))
# --- late verdicts (revision 12)
P1 = rd('p1_delay', R12)
V = [('global (no cooldown)', 'Global'), ('cooldown 30 d, override 0.999', 'Immediate'), ('verdicts 7 days late', '7 d late'), ('verdicts 14 days late', '14 d late'), ('no override, verdicts 7 days late', 'No override, 7 d late')]
rows = ['Det. & ' + ' & '.join(l for _, l in V) + ' \\\\', '\\midrule']
for rel, _ in REL:
    rows.append(f'\\multicolumn{{{len(V) + 1}}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in DETS:
        x = P1[(P1.release == rel) & (P1.detector == d)]
        rows.append(d + ' & ' + ' & '.join(f'{x[x.policy == p].insiders_detected.mean():.1f} / {x[x.policy == p].max_F_benign.mean():.0f} / {x[x.policy == p].alerted_employees.mean():.0f}' for p, _ in V) + ' \\\\')
out.append(table('tab:s-delay', 'Cooldown (30 Days) When Verdicts Arrive Late: Insiders Detected / $\\max F_i$ / Benign Employees Alerted, Means Over Seeds. The Cooldown Starts When the Verdict Arrives, and Alerts Whose Verdicts Are Pending Do Not Postpone an Arrived Clearance ($b=1\\%$)', 'l' + 'c' * len(V), rows, wide=True))
# --- permutation without most-flagged (revision 12)
P3 = rd('p3_permutation', R12)
rows = ['Rel. & Det. & All & 5 removed & 10 removed & 20 removed \\\\', '\\midrule']
for rel, _ in REL:
    for d in DETS:
        x = P3[(P3.release == rel) & (P3.detector == d)].set_index('removed')
        rows.append(f'{rel} & {d} & ' + ' & '.join(f'{x.loc[r].p_range:.3f} / {x.loc[r].p_weighted:.4f}' for r in (0, 5, 10, 20)) + ' \\\\')
out.append(table('tab:s-perm-removed', 'Permutation Tests of Group Disparity Without the Most-Flagged Benign Employees: $p$-Values of the Range / Size-Weighted Statistic (First Seed, 2000 Permutations, $b=1\\%$)', 'llcccc', rows))
open(O + 'sup11_tables.tex', 'w').write('\n'.join(out))
print(len(out), 'tables')
