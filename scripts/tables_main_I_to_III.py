import pandas as pd, numpy as np, json
R5='results/06_noninferiority_attacks/'; R6='results/07_r52_rules_permutation/'; R7='results/08_cooldown/'
i7=pd.read_csv(R5+'i7_attacks_summary.csv'); j4=pd.read_csv(R6+'j4_attacks_r42_summary.csv')
j8=pd.read_csv(R6+'j8_attacks_r52_summary.csv'); k3=pd.read_csv(R7+'k3_attacks_r42_summary.csv')
k1a=pd.read_csv(R7+'k1_farming_r42.csv'); k1b=pd.read_csv(R7+'k1_farming_r52.csv')
def worst(df,det,pol):
    D=df[(df.detector==det)&(df.policy==pol)]
    if D.empty: return '--'
    r=D.loc[D.excess_loss.idxmax()]
    star='$^\\ast$' if r.excess_lo>0 else ''
    return f'{r.excess_loss:.1f}{star}'
def farm(df,det,pol):
    D=df[(df.detector==det)&(df.policy==pol)&(df.alpha==1.0)]
    if D.empty: return '--'
    r=D.iloc[0]; return f'{int(r.lost)}/{int(r.detected_no_farming)}'
rows=[
 ('Cooldown 30 d, override $0.999$','farming', farm(k1a,'GBM','B0 + cooldown 30 days'),farm(k1b,'GBM','B0 + cooldown 30 days'),farm(k1a,'AE','B0 + cooldown 30 days'),farm(k1b,'AE','B0 + cooldown 30 days')),
 ('Cooldown 30 d, no override','farming', farm(k1a,'GBM','B0 + cooldown 30 days, no override'),farm(k1b,'GBM','B0 + cooldown 30 days, no override'),farm(k1a,'AE','B0 + cooldown 30 days, no override'),farm(k1b,'AE','B0 + cooldown 30 days, no override')),
 ('Similarity scheduler','farming', farm(k1a,'GBM','B0 + scheduling'),farm(k1b,'GBM','B0 + scheduling'),farm(k1a,'AE','B0 + scheduling'),farm(k1b,'AE','B0 + scheduling')),
 ('None (global top-$k$)','farming', farm(k1a,'GBM','B0 global'),farm(k1b,'GBM','B0 global'),farm(k1a,'AE','B0 global'),farm(k1b,'AE','B0 global')),
 ('Exact rule, label-free','timing', worst(i7,'GBM','B4 label-free'),worst(j8,'GBM','B4 exact rule, label-free'),worst(i7,'LSTM','B4 label-free'),worst(j8,'LSTM','B4 exact rule, label-free')),
 ('Exact rule, outcome-aware','timing', worst(i7,'GBM','B4c outcome-aware'),worst(j8,'GBM','B4c exact rule'),worst(i7,'LSTM','B4c outcome-aware'),worst(j8,'LSTM','B4c exact rule')),
 ('\\quad verdicts after 7 days','timing', worst(j4,'GBM','B4c, verdicts after 7 days'),worst(j8,'GBM','B4c, verdicts after 7 days'),worst(j4,'LSTM','B4c, verdicts after 7 days'),worst(j8,'LSTM','B4c, verdicts after 7 days')),
 ('\\quad verdicts after 30 days','timing', worst(j4,'GBM','B4c, verdicts after 30 days'),'--',worst(j4,'LSTM','B4c, verdicts after 30 days'),'--'),
 ('Deficit apportionment, outcome-aware','timing', worst(j4,'GBM','B11c deficit apportionment (outcome-aware)'),worst(j8,'GBM','B11c deficit apportionment (outcome-aware)'),worst(j4,'LSTM','B11c deficit apportionment (outcome-aware)'),worst(j8,'LSTM','B11c deficit apportionment (outcome-aware)')),
 ('\\quad with cooldown 30 d','timing', worst(k3,'GBM','B11c + cooldown 30 days'),'--',worst(k3,'LSTM','B11c + cooldown 30 days'),'--'),
]
lines=[' & '.join(r)+' \\\\' for r in rows]
lines.insert(4,'\\midrule')
tab=r"""\begin{table}[!t]
\caption{Exposure of History-Dependent Mechanisms. Farming: Farming Insiders Missed Because They Farmed, Out of Those Detected Without Farming ($\alpha=1$). Timing: Mean Excess Loss of the Worst Attacker Over the Global Policy on the Same Relocated Attacks, in Insiders; $^\ast$: 95\% Interval Excludes Zero ($b=1\%$, $\delta=0.002$)\label{tab:exposure}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{AE (farming) / LSTM (timing)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Mechanism & Channel & r4.2 & r5.2 & r4.2 & r5.2 \\
\midrule
"""+"\n".join(lines)+r"""
\bottomrule
\end{tabular}}
\end{table}"""
open('tables/tab_exposure.tex','w').write(tab); print(tab)

# ---------- Table II: burden under global top-k, both releases, permutation tests
j2=pd.read_csv(R6+'j2_baselines_r42.csv'); k2=pd.read_csv(R7+'k2_suppression_r52.csv')
p4=pd.read_csv(R6+'j1_permutation_r42.csv'); p5=pd.read_csv(R6+'j1_permutation_r52.csv')
expo4={'IF':0.484,'AE':0.379,'LR':-0.115,'GBM':0.030,'LSTM':0.451}
j8r=pd.read_csv(R6+'j8_rules_r52.csv')
L=[]
for rel,B,P in (('r4.2',j2,p4),('r5.2',k2,p5)):
    L.append(f'\\multicolumn{{9}}{{l}}{{\\emph{{{rel}}} ({"51" if rel=="r4.2" else "76"} test insiders)}} \\\\')
    for d in ['IF','AE','LR','GBM','LSTM']:
        r=B[(B.detector==d)&(B.policy=='B0 global')].iloc[0]
        pp=P[(P.detector==d)&(P.policy=='B0 global')]
        p0=pp[pp.dropped_top==0].p_value.iloc[0]; p10=pp[pp.dropped_top==10].p_value.iloc[0]
        ex=expo4[d] if rel=='r4.2' else j8r[(j8r.detector==d)&(j8r.policy=='B0 global')].exposure_bias.iloc[0]
        f=lambda p: f'{p:.3f}' if p>=0.001 else '$<$0.001'
        L.append(f'{d} & {int(r.insiders_detected)} & {r.gini_burden:.3f} & {r.top1pct_share:.2f} & {int(r.max_F)} & {r.delta_FAR:.4f} & {f(p0)} & {f(p10)} & {ex:.2f} \\\\')
tab2=r"""\begin{table}[!t]
\caption{Alert Burden Under Global Top-$k$ Alerting ($b=1\%$, Functional Units). $p$: Permutation Test of $\Delta_{\mathrm{FAR}}$ With Group Labels Shuffled Across Employees, Without and With the Ten Most-Flagged Benign Employees Removed\label{tab:rq1}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Det. & Detected & Gini($F$) & Top-1\% & $\max F_i$ & $\Delta_{\mathrm{FAR}}$ & $p$ & $p$, top 10 out & Exp. bias \\
\midrule
"""+"\n".join(L)+r"""
\bottomrule
\end{tabular}}
\end{table}"""
open('tables/tab_rq1.tex','w').write(tab2)

# ---------- Table III: benefit (detected / max F), both releases
pols=[('B0 global','Global'),('B0 + cooldown 30 days','Cooldown'),('B0 + scheduling','Scheduler'),('B11c deficit apportionment (outcome-aware)','Parity'),('B11c + cooldown 30 days','Parity + cooldown')]
L=[]
for rel,B in (('r4.2',j2),('r5.2',k2)):
    L.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in ['IF','AE','LR','GBM','LSTM']:
        cells=[]
        for p,_ in pols:
            r=B[(B.detector==d)&(B.policy==p)]
            r=r.iloc[0] if len(r) else None
            cells.append('--' if r is None else f'{int(r.insiders_detected)} / {int(r.max_F)}')
        L.append(d+' & '+' & '.join(cells)+' \\\\')
tab3=r"""\begin{table}[!t]
\caption{Benefit of History-Dependent Mechanisms: Insiders Detected / Alerts on the Most-Flagged Benign Employee ($\max F_i$). Cooldown: 30 Days After Clearance, Override $0.999$; Parity: Outcome-Aware Deficit Apportionment ($b=1\%$; Seeds in Tables~\EREF{tab:s-ni-r42} and~\EREF{tab:s-ni-r52})\label{tab:benefit}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Det. & """+' & '.join(n for _,n in pols)+r""" \\
\midrule
"""+"\n".join(L)+r"""
\bottomrule
\end{tabular}}
\end{table}"""
open('tables/tab_benefit.tex','w').write(tab3)
print(tab2); print(tab3)
