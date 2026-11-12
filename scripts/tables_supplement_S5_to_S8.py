import pandas as pd, numpy as np
R='results/09_security_rescored/'
def f1(x): return f'{x:.1f}'.replace('-0.0','0.0')
out={}
# L0
L0=pd.concat([pd.read_csv(R+f'l0_checks_{r}.csv') for r in ('r42','r52')])
g=L0.groupby(['release','detector'],sort=False).mean(numeric_only=True)
rows=[]
for (rel,d),r in g.iterrows():
    rows.append(f"{rel} & {d} & {r.max_abs_diff_no_change:.0e} & {r.max_abs_diff_rescore_unchanged:.0e} & {int(r.insiders_detected_original)} & {r.insiders_detected_score_preserving:.1f} & {r.insiders_detected_behavioral:.1f} & {r.mean_malicious_score_original:.3f} & {r.mean_malicious_score_behavioral:.3f} \\\\")
out['l0']=r"""\begin{table}[!t]
\caption{Rescoring Checks. Max. Diff.: Largest Absolute Difference Between Cached and Recomputed Scores Without Change and After Rescoring Unchanged Malicious Days (LSTM: Whole Sequences). Insiders Detected by the Global Policy Without Relocation and Under Random Relocation, Moving Scores or Behavior (Mean of Five Draws), and Mean Score of Malicious User-Days\label{tab:s-l0}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
 & & \multicolumn{2}{c}{Max. diff.} & \multicolumn{3}{c}{Detected (global)} & \multicolumn{2}{c}{Malicious score} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Rel. & Det. & none & rescored & original & scores moved & behavior moved & original & behavior moved \\
\midrule
"""+'\n'.join(rows).replace('0e+00','0')+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
# timing per attacker
M=pd.read_csv(R+'l2_multiplicity.csv')
ATK=['knows allocation','observes colleagues','reads reports','estimates deficit','random']
PN={'B4 label-free':'Exact, label-free','B4c outcome-aware':'Exact, outcome-aware','B4c outcome-aware, 7-day verdict delay':'Exact, outcome-aware, 7-day delay','B11c deficit apportionment (outcome-aware)':'Deficit apportionment'}
rows=[]
for rel in ('r4.2','r5.2'):
    rows.append(f'\\multicolumn{{9}}{{l}}{{\\emph{{{rel}}} ({int(M[M.release==rel].insiders.iloc[0])} insiders)}} \\\\')
    for d in ('GBM','LSTM','AE'):
        for p in PN:
            r=M[(M.release==rel)&(M.detector==d)&(M.policy==p)].iloc[0]
            cells=[]
            for a in ATK:
                v=f1(r[f'{a} | excess'])
                if r[f'{a} | bonferroni_lo']>1e-9: v+='$^\\dagger$'
                if a=='estimates deficit' and r.primary_lo>1e-9: v+='$^\\ast$'
                cells.append(v)
            rows.append(f"{d} & {PN[p]} & "+' & '.join(cells)+f" & [{f1(r.primary_lo)}, {f1(r.primary_hi)}] & [{f1(r.joint_max_lo)}, {f1(r.joint_max_hi)}] \\\\")
out['timing']=r"""\begin{table*}[!t]
\caption{Timing Attacks With Behavioral Relocation and Rescoring: Excess Loss Over the Global Policy of Each Attacker, in Insiders (Mean Over Five Draws, Ten for the Random Attacker). $^\ast$: 95\% Paired Bootstrap Interval of the Prespecified Attacker (Estimates Deficit) Excludes Zero; $^\dagger$: Bonferroni-Corrected Interval (Level $1-0.05/5$) Excludes Zero. Last Columns: 95\% Interval of the Prespecified Attacker, and Bootstrap Interval of the Maximum Over Attackers, Resampling Insiders Jointly ($b=1\%$, $\delta=0.002$, Functional Units)\label{tab:s-timing8}}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccccc}
\toprule
Det. & Parity rule & Knows alloc. & Obs. colleagues & Reads reports & Est. deficit & Random & Prespec., 95\% & Max., 95\% \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# farming grid
rows=[]
F={r:pd.read_csv(R+f'l3_farming_{r}_summary.csv') for r in ('r42','r52')}
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    for d in ('GBM','AE','LR','IF'):
        cells=[]
        for p in ('B0 + cooldown 30 days','B0 + cooldown 30 days, no override'):
            D=F[rk][(F[rk].detector==d)&(F[rk].policy==p)]
            base=int(D.detected_no_farming.iloc[0])
            for k in ('own','scenario','generic'):
                x=D[(D.knowledge==k)&(D.p_clear==1)]
                cells.append('--'.join(str(int(round(v))) for v in sorted(set([x.evaded.min(),x.evaded.max()]))))
            for pc in (0.8,0.5):
                x=D[(D.p_clear==pc)]
                cells.append(f1(x.evaded.max()))
            cells.append(f1(D[D.p_clear==0.5].exposed_by_probe.max()))
            cells.insert(len(cells)-6, str(base))
        rows.append(f'{rel} & {d} & '+' & '.join(cells)+' \\\\')
out['farm']=r"""\begin{table*}[!t]
\caption{Discount Farming With Rescored, Role-Capped Probes Against a 30-Day Cooldown With and Without the Override $0.999$: Insiders Detected Without Farming (Det.), Insiders Who Evade by Farming When Every Probe Alert Is Cleared (Range Over One, Three and Five Probes) for Each Level of Knowledge, Maximum Mean Evasions When Probe Alerts Are Cleared With Probability $0.8$ or $0.5$ (Three Probes, Three Draws), and Maximum Mean Number of Farmers Exposed by a Confirmed Probe at $0.5$ ($b=1\%$; 45 Farming Insiders in r4.2, 73 in r5.2)\label{tab:s-farm8}}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccccccccccc}
\toprule
 & & \multicolumn{7}{c}{Cooldown 30 days, override $0.999$} & \multicolumn{7}{c}{Cooldown 30 days, no override} \\
\cmidrule(lr){3-9}\cmidrule(lr){10-16}
Rel. & Det. & Det. & Own & Scen. & Gen. & $p=0.8$ & $p=0.5$ & Exposed & Det. & Own & Scen. & Gen. & $p=0.8$ & $p=0.5$ & Exposed \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# errors summary
E=pd.concat([pd.read_csv(R+f'l4_errors_{r}.csv') for r in ('r42','r52')])
g=E.groupby(['release','detector','condition','policy'],sort=False)[['insiders_detected','delta_FAR','max_F']].mean().reset_index()
b=g[g.condition=='no errors'].set_index(['release','detector','policy'])
g=g.join(b,on=['release','detector','policy'],rsuffix='0')
g['dd']=g.insiders_detected-g.insiders_detected0; g['dm']=g.max_F-g.max_F0
CN=[('no errors','None'),('score-dependent misses','Score-dependent misses'),('score-dependent false','Score-dependent false confirmations'),
    ('group-dependent: smallest','Smallest unit worse'),('group-dependent: largest','Largest unit worse'),('inconclusive 20%','Inconclusive 20\\%'),
    ('inconclusive 50%','Inconclusive 50\\%'),('score-dependent errors +','Score-dependent errors + inconclusive 50\\%')]
def rng(x): 
    lo,hi=x.min(),x.max(); return f'{f1(lo)} to {f1(hi)}' if lo!=hi else f1(lo)
rows=[]
for key,lab in CN:
    G=g[g.condition.str.startswith(key)]
    cells=[]
    for p in ('B0 + cooldown 30 days','B0 + cooldown 30 days, no override','B0 + scheduling'):
        x=G[G.policy==p]; cells+= [rng(x.dd), f1(x.dm.max())]
    x=G[G.policy.isin(['B4c outcome-aware','B11c deficit apportionment (outcome-aware)'])]
    cells+=[rng(x.dd), f'{x.delta_FAR.max():.4f}']
    rows.append(lab+' & '+' & '.join(cells)+' \\\\')
out['errors']=r"""\begin{table*}[!t]
\caption{Verdict Errors and Inconclusive Verdicts: Change in Insiders Detected (Range Over Five Detectors and Two Releases) and Largest Increase of $\max F_i$ Relative to Correct Verdicts, and, for the Outcome-Aware Exact Rule and Deficit Apportionment, Largest $\Delta_{\mathrm{FAR}}$ (Mean of Two Draws, $b=1\%$, $\delta=0.002$). Score-Dependent: Rates $0.30$/$0.05$ (Misses) or $0.01$/$0.10$ (False Confirmations) Below/Above $0.999$; Unit: $0.30$ Misses and $0.05$ False Confirmations in One Unit, $0.05$ and $0.01$ Elsewhere\label{tab:s-errors8}}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{Cooldown 30 d} & \multicolumn{2}{c}{No override} & \multicolumn{2}{c}{Scheduler} & \multicolumn{2}{c}{Outcome-aware parity} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Verdict model & $\Delta$ detected & $\Delta\max F$ & $\Delta$ detected & $\Delta\max F$ & $\Delta$ detected & $\Delta\max F$ & $\Delta$ detected & $\max\Delta_{\mathrm{FAR}}$ \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
import json; json.dump(out,open('tables/tables_sup8.json','w'))
for v in out.values(): print(v[:200].split('\n')[1][:80])
print(out['farm'][-1500:]); print(out['errors'][-1300:])
