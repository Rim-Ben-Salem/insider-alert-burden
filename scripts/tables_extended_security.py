import pandas as pd, numpy as np
R='results/09_security_rescored/'
def f1(x): return f'{x:.1f}'.replace('-0.0','0.0')
def esc(s): return s.replace('%','\\%').replace('&','\\&').replace('_','\\_')
parts=[]
# full timing summaries
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    d=pd.read_csv(R+f'l1_attacks_{rk}_summary.csv')
    rows=[]
    for _,r in d.iterrows():
        rows.append(f"{r.detector} & {esc(r.policy)} & {r.attack} & {r.none_P:.0f} & {r.none_G:.0f} & {r.attacked_P:.1f} & {r.attacked_G:.1f} & {f1(r.excess_loss)} & [{f1(r.excess_lo)}, {f1(r.excess_hi)}] & {f1(r.excess_loss_min)}--{f1(r.excess_loss_max)} \\\\")
    parts.append(r"""\begin{table*}[!t]
\caption{Timing Attacks With Behavioral Relocation and Rescoring, All Attackers ("""+rel+r""", $b=1\%$, $\delta=0.002$). Detected Under the Parity Rule P and the Global Policy G Without and With the Attack (Mean Over Draws); Excess Loss With the Mean of Per-Draw Paired Bootstrap Intervals and the Range Over Draws\label{tab:e-l1-"""+rk+r"""}}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllccccccc}
\toprule
Det. & Parity rule & Attacker & P & G & P, attacked & G, attacked & Excess & 95\% & Range \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
""")
# full farming summaries
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    d=pd.read_csv(R+f'l3_farming_{rk}_summary.csv')
    rows=[]
    for _,r in d.iterrows():
        rows.append(f"{r.detector} & {r.knowledge} & {r.n_probe} & {r.p_clear:.1f} & {esc(r.policy)} & {r.detected_no_farming:.0f} & {r.probes_alerted_insiders:.1f} & {r.attack_detected_farming:.1f} & {r.exposed_by_probe:.1f} & {r.caught_farming:.1f} & {r.evaded:.1f} & {r.gained:.1f} \\\\")
    parts.append(r"""\begin{longtable}{lllllccccccc}
\caption{Realistic Discount Farming, All Settings ("""+rel+r"""; Probes Capped at the Role's Benign 99th Percentile and Rescored; Mean of Three Draws When $p<1$). Det.: Detected Without Farming; Probed: Farmers With an Alerted Probe; Attack Det.: Attack Days Alerted; Exposed: a Probe Confirmed; Caught: Either\label{tab:e-l3-"""+rk+r"""}}\\
\toprule
Det. & Knowl. & $n$ & $p$ & Policy & Det. & Probed & Attack det. & Exposed & Caught & Evaded & Gained \\
\midrule
\endfirsthead
\toprule
Det. & Knowl. & $n$ & $p$ & Policy & Det. & Probed & Attack det. & Exposed & Caught & Evaded & Gained \\
\midrule
\endhead
"""+'\n'.join(rows)+r"""
\bottomrule
\end{longtable}
""")
# errors full
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    d=pd.read_csv(R+f'l4_errors_{rk}.csv')
    g=d.groupby(['detector','condition','policy'],sort=False)[['insiders_detected','delta_FAR','max_F','distinct_alerted']].mean().reset_index()
    rows=[f"{r.detector} & {esc(r.condition)} & {esc(r.policy)} & {r.insiders_detected:.1f} & {r.delta_FAR:.4f} & {r.max_F:.1f} & {r.distinct_alerted:.1f} \\\\" for _,r in g.iterrows()]
    parts.append(r"""\begin{longtable}{lp{5.2cm}p{4.4cm}cccc}
\caption{Verdict Errors and Inconclusive Verdicts, All Settings ("""+rel+r""", Mean of Two Draws)\label{tab:e-l4-"""+rk+r"""}}\\
\toprule
Det. & Verdict model & Policy & Detected & $\Delta_{\mathrm{FAR}}$ & $\max F_i$ & Alerted \\
\midrule
\endfirsthead
\toprule
Det. & Verdict model & Policy & Detected & $\Delta_{\mathrm{FAR}}$ & $\max F_i$ & Alerted \\
\midrule
\endhead
"""+'\n'.join(rows)+r"""
\bottomrule
\end{longtable}
""")
# social cost full
rows=[]
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    d=pd.read_csv(R+f'l5_social_cost_{rk}.csv')
    for _,r in d.iterrows():
        be='' if pd.isna(r.break_even_ratio) else f'{r.break_even_ratio:.3g}'
        rows.append(f"{rel} & {r.detector} & {esc(r.policy)} & {r.alerted_employees} & {r.repeat_alerts} & {r.false_alerts} & {r.max_F} & {r.insiders_detected} & {be} & {esc(str(r.cheaper_when))} \\\\")
parts.append(r"""\begin{longtable}{llp{4.6cm}ccccccl}
\caption{Social Cost Components and Break-Even Ratios $c_2/c_1$ Relative to the Global Policy ($b=1\%$)\label{tab:e-l5}}\\
\toprule
Rel. & Det. & Policy & Alerted & Repeats & False alerts & $\max F_i$ & Detected & $r^\ast$ & Cheaper when \\
\midrule
\endfirsthead
\toprule
Rel. & Det. & Policy & Alerted & Repeats & False alerts & $\max F_i$ & Detected & $r^\ast$ & Cheaper when \\
\midrule
\endhead
"""+'\n'.join(rows)+r"""
\bottomrule
\end{longtable}
""")
open('tables/ext8_tables.tex','w').write('\n'.join(parts))
print(len(parts))
