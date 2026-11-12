import pandas as pd, numpy as np, json
R9='results/10_attack_realism/'; W=''
def f1(x): return f'{x:.1f}'.replace('-0.0','0.0')
out={}
M=pd.read_csv(R9+'m1_multiplicity.csv')
ATK=['random','online deficit, q=0.8','online deficit, q=0.9','online deficit, q=0.95','online colleagues, q=0.9','online reports, q=0.9','hindsight deficit','oracle: knows allocation (hindsight)']
PN={'B4 label-free':'Exact, label-free','B4c outcome-aware':'Exact, outcome-aware','B4c outcome-aware, 7-day verdict delay':'Exact, OA, 7-day delay','B11c deficit apportionment (outcome-aware)':'Deficit apportionment'}
rows=[]
for rel in ('r4.2','r5.2'):
    rows.append(f'\\multicolumn{{11}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in ('GBM','LSTM','AE'):
        for p in PN:
            r=M[(M.release==rel)&(M.detector==d)&(M.policy==p)].iloc[0]
            cells=[]
            for a in ATK:
                v=f1(r[f'{a} | excess'])
                if r[f'{a} | lo']>1e-9: v+='$^\\ast$'
                if f'{a} | bonferroni_lo' in r and pd.notna(r.get(f'{a} | bonferroni_lo')) and r[f'{a} | bonferroni_lo']>1e-9: v+='$^\\dagger$'
                cells.append(v)
            a='online deficit, q=0.9'
            rows.append(f"{d} & {PN[p]} & "+' & '.join(cells)+f" & [{f1(r[a+' | lo'])}, {f1(r[a+' | hi'])}] \\\\")
out['timing']=r"""\begin{table*}[!t]
\caption{Timing Attacks With Behavioral Relocation and Rescoring, Every Attacker: Excess Loss Over the Global Policy, in Insiders. Online Attackers Start the First Day After a 30-Day Warm-Up When Their Signal Is Positive and Reaches the $q$-Quantile of Its Past Values, or at the Last Possible Day; Hindsight Attackers Choose the Best Day Over the Whole Period. $^\ast$: 95\% Paired Bootstrap Interval Excludes Zero; $^\dagger$: Bonferroni-Corrected Interval Over the Five Online Attackers Excludes Zero. Last Column: 95\% Interval of the Prespecified Attacker (Online Deficit, $q=0.9$)\label{tab:s-timing8}}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccccccc}
\toprule
 & & & \multicolumn{5}{c}{Online} & \multicolumn{2}{c}{Hindsight} & \\
\cmidrule(lr){4-8}\cmidrule(lr){9-10}
Det. & Parity rule & Random & Def., $0.8$ & Def., $0.9$ & Def., $0.95$ & Colleagues & Reports & Deficit & Allocation & Prespecified, 95\% \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# decomposition
D=pd.read_csv(R9+'m1_decomposition.csv')
rows=[]
for rel in ('r4.2','r5.2'):
    for d in ('GBM','LSTM','AE'):
        for p in PN:
            cells=[]
            for a in ('online deficit, q=0.9','oracle: knows allocation (hindsight)'):
                r=D[(D.release==rel)&(D.detector==d)&(D.policy==p)&(D.attack==a)].iloc[0]
                cells+= [f'{r.relocation_effect_global:.1f}', f'{r.relocation_effect_policy:.1f}', f'{r.lost_by_mechanism_only:.1f}', f'{r.gained_by_mechanism_only:.1f}']
            rows.append(f'{rel} & {d} & {PN[p]} & '+' & '.join(cells)+' \\\\')
out['decomp']=r"""\begin{table*}[!t]
\caption{Decomposition of the Excess Loss, $(P_1-P_0)-(G_1-G_0)$ With Signs Reversed: Change in Insiders Detected by the Global Policy ($G_1-G_0$) and by the Parity Rule ($P_1-P_0$) When the Attack Is Relocated, and Insiders Detected After the Attack by the Global Policy but Not the Parity Rule (Lost to the Mechanism) or the Reverse (Gained), Mean Over Draws\label{tab:s-decomp}}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllcccccccc}
\toprule
 & & & \multicolumn{4}{c}{Online, prespecified} & \multicolumn{4}{c}{Knows the allocation} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}
Rel. & Det. & Parity rule & $G_1-G_0$ & $P_1-P_0$ & Lost & Gained & $G_1-G_0$ & $P_1-P_0$ & Lost & Gained \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# NI margins
def ni_rows():
    i2=pd.read_csv('results/06_noninferiority_attacks/i2_pooled.csv'); j6=pd.read_csv('results/07_r52_rules_permutation/j6_pooled_r42.csv')
    j8=pd.read_csv('results/07_r52_rules_permutation/j8_ni_pooled_r52.csv'); k2=pd.read_csv('results/08_cooldown/k2_ni_pooled_r52.csv')
    src={('r4.2','Cooldown 30 d'):(j6,'B0 + cooldown 30 days'),('r4.2','Scheduler'):(j6,'B0 + scheduling'),
         ('r4.2','Exact rule, OA'):(i2,'B4c outcome-aware'),('r4.2','Deficit apportionment'):(j6,'B11c deficit apportionment (outcome-aware)'),
         ('r5.2','Cooldown 30 d'):(k2,'B0 + cooldown 30 days'),('r5.2','Scheduler'):(j8,'B0 + scheduling'),
         ('r5.2','Exact rule, OA'):(j8,'B4c exact rule'),('r5.2','Deficit apportionment'):(j8,'B11c deficit apportionment (outcome-aware)')}
    rows=[]
    for mech in ('Cooldown 30 d','Scheduler','Exact rule, OA','Deficit apportionment'):
        for d in ('IF','AE','LR','GBM','LSTM'):
            cells=[]
            for rel in ('r4.2','r5.2'):
                df,p=src[(rel,mech)]
                x=df[(df.detector==d)&(df.policy==p)&(np.isclose(df.budget,0.01))]
                if x.empty: cells+=['--']*4; continue
                x=x.iloc[0]; rel_=100*x['diff']/x.recall_ref if x.recall_ref>0 else np.nan
                marks=''.join('\\checkmark' if x.one_sided_lo>-m else '$\\times$' for m in (0.025,0.05,0.10))
                cells+=[f'{x.recall_ref:.2f}', f'{x["diff"]:+.3f} ({rel_:+.0f}\\%)', f'{x.one_sided_lo:+.3f}', marks]
            rows.append(f'{mech} & {d} & '+' & '.join(cells)+' \\\\')
    return rows
out['ni']=r"""\begin{table*}[!t]
\caption{Non-Inferiority to the Global Policy at Margins $0.025$, $0.05$ (Prespecified) and $0.10$: Recall of the Global Policy, Recall Difference (Relative Change), One-Sided 95\% Lower Bound, and Whether It Exceeds Minus Each Margin ($b=1\%$, $\delta=0.002$; Pooled Over Seeds, Insiders Resampled)\label{tab:s-ni-margins}}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{r4.2} & \multicolumn{4}{c}{r5.2} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Mechanism & Det. & Global & Difference & Bound & $.025/.05/.10$ & Global & Difference & Bound & $.025/.05/.10$ \\
\midrule
"""+'\n'.join(ni_rows())+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
# claims
out['claims']=r"""\begin{table*}[!t]
\caption{Status of the Paper's Claims: Prespecified (Confirmatory) Tests, With Their Multiplicity Handling, Against Exploratory or Descriptive Results\label{tab:s-claims}}
\centering
\footnotesize
\begin{tabular}{p{4.6cm}p{2.0cm}p{4.0cm}p{5.6cm}}
\toprule
Claim & Status & Test and multiplicity & Result \\
\midrule
Burden concentrates on few innocent employees & Descriptive & Gini, top-1\% share, $\max F_i$ with employee-clustered intervals & Gini $0.91$--$0.98$ for every detector and release \\
Group disparity under the global policy exceeds relabeling & Confirmatory & Permutation test, 10 detector--release tests, unadjusted and Bonferroni & $p=0.012$ (IF) and $0.048$ (GBM) on r4.2 unadjusted; none after Bonferroni \\
A 30-day cooldown is non-inferior in detection & Confirmatory & One-sided bound, margin $0.05$ fixed in advance; per detector, no adjustment & Non-inferior for 4 of 5 detectors on r4.2 and r5.2; robust to margins $0.025$ and $0.10$ for most (Table~\ref{tab:s-ni-margins}) \\
Parity rules reach the disparity target & Descriptive & Cumulative $\Delta_{\mathrm{FAR}}$ against $\delta$ & At or below target for every rule with memory \\
Parity costs detection & Confirmatory & As for the cooldown & Not non-inferior for GBM on both releases \\
Farming defeats suppression & Exploratory & Counts over a prespecified grid; plausibility levels & Up to 11 insiders with the override, 18 without; up to 9 and 10 at the median plausibility level \\
An online insider can time attacks against parity & Confirmatory (prespecified attacker) & Paired bootstrap per setting; Bonferroni over 5 online attackers & Significant in 1 of 24 settings, none after correction; mean $1.8$ insiders \\
Better-informed insiders can time attacks & Exploratory (upper bounds) & Paired bootstrap per setting & Hindsight: 11 of 24 significant; knows allocation: 7 of 24 \\
Verdict errors and inconclusive verdicts & Exploratory & Sensitivity over prespecified models & Parity robust; inconclusive verdicts weaken the cooldown \\
\bottomrule
\end{tabular}
\end{table*}
"""
json.dump(out,open('tables/tables_sup9.json','w'))
print(out['ni'][-3000:])
