import pandas as pd
R='results/10_attack_realism/'
def esc(s): return str(s).replace('%','\\%').replace('&','\\&').replace('_','\\_')
C=pd.read_csv(R+'m1_constrained.csv')
rows=[f"{r.release} & {r.detector} & {esc(r.policy)} & {esc(r.attack)} & {esc(r.constraint)} & {r.admissible_share:.2f} & {r.excess:.1f} & [{r.lo:.1f}, {r.hi:.1f}] \\\\" for _,r in C.iterrows()]
p1=r"""\begin{longtable}{lllp{4.2cm}p{3.4cm}ccc}
\caption{Timing Attacks Under Plausibility Constraints: Share of Insiders Allowed to Move, Excess Loss and 95\% Interval, Every Attacker and Setting\label{tab:e-m1-constrained}}\\
\toprule
Rel. & Det. & Rule & Attacker & Constraint & Admissible & Excess & 95\% \\
\midrule
\endfirsthead
\toprule
Rel. & Det. & Rule & Attacker & Constraint & Admissible & Excess & 95\% \\
\midrule
\endhead
"""+'\n'.join(rows)+r"""
\bottomrule
\end{longtable}
"""
parts=[p1]
for t,rel in (('r42','r4.2'),('r52','r5.2')):
    S=pd.read_csv(R+f'm2_farming_{t}_summary.csv')
    rows=[f"{r.detector} & {r.knowledge} & {r.n_probe} & {esc(r.policy)} & {r.level} & {r.farmers} & {r.admissible} & {r.detected_no_farming} & {r.evaded} & {r.evaded_unconstrained} \\\\" for _,r in S.iterrows()]
    parts.append(r"""\begin{longtable}{lllp{4.4cm}cccccc}
\caption{Farming Under Plausibility Constraints ("""+rel+r"""): Level Is the Quantile of Innocent Alerted User-Days (``own'': the Insider's Own Most Unusual Benign Day); Farmers With Admissible Probes, Detected Without Farming, Evaded Under the Constraint and Without It\label{tab:e-m2-"""+t+r"""}}\\
\toprule
Det. & Knowl. & $n$ & Policy & Level & Farmers & Admissible & Detected & Evaded & Unconstr. \\
\midrule
\endfirsthead
\toprule
Det. & Knowl. & $n$ & Policy & Level & Farmers & Admissible & Detected & Evaded & Unconstr. \\
\midrule
\endhead
"""+'\n'.join(rows)+r"""
\bottomrule
\end{longtable}
""")
open('tables/ext9_tables.tex','w').write('\n'.join(parts))
