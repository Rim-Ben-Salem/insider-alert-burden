import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='results/09_security_rescored/'
M=pd.read_csv(R+'l2_multiplicity.csv')
S={r:pd.read_csv(R+f'l1_attacks_{r}_summary.csv') for r in ('r42','r52')}
F={r:pd.read_csv(R+f'l3_farming_{r}_summary.csv') for r in ('r42','r52')}
C={r:pd.read_csv(R+f'l5_social_cost_{r}.csv') for r in ('r42','r52')}
ATK=['knows allocation','observes colleagues','reads reports','estimates deficit','random']
POL=[('B4 label-free','Exact rule, label-free'),('B4c outcome-aware','Exact rule, outcome-aware'),
     ('B4c outcome-aware, 7-day verdict delay','\\quad verdicts after 7 days'),('B11c deficit apportionment (outcome-aware)','Deficit apportionment, outcome-aware')]
def f1(x): return f'{x:.1f}'.replace('-0.0','0.0')
# ---------- timing table
rows=[]
for p,lab in POL:
    cells=[]
    for d in ('GBM','LSTM','AE'):
        for rel in ('r4.2','r5.2'):
            r=M[(M.release==rel)&(M.detector==d)&(M.policy==p)].iloc[0]
            a=f1(r.primary_excess)+('$^\\ast$' if r.primary_lo>1e-9 else '')
            nb=sum(r[f"{x} | bonferroni_lo"]>1e-9 for x in ATK)
            b=f1(r.worst_excess)+('$^\\dagger$' if nb>0 else '')
            cells.append(f'{a} / {b}')
    rows.append(lab+' & '+' & '.join(cells)+' \\\\')
tim=r"""\begin{table}[!t]
\caption{Timing Attacks With Behavioral Relocation: The Insider Performs the Malicious Actions on Other Days and the Saved Detector Rescores the Modified History. Excess Loss Over the Global Policy on the Same Relocated Attacks, in Insiders: Prespecified Attacker (Estimates Its Group's Excess From Noisy Reports and Colleagues' Investigations) / Worst of Five Attackers. $^\ast$: 95\% Interval of the Prespecified Attacker Excludes Zero; $^\dagger$: Some Attacker's Bonferroni-Corrected Interval Excludes Zero ($b=1\%$, $\delta=0.002$)\label{tab:timing}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{AE} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Parity rule & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open('tables/tab_timing8.tex','w').write(tim)
# ---------- farming table
FP=[('B0 global','None (global top-$k$)'),('B0 + scheduling','Similarity scheduler'),('B0 + cooldown 30 days','Cooldown 30 d, override $0.999$'),
    ('B0 + cooldown 30 days, no override','Cooldown 30 d, no override'),('B11c + cooldown 30 days','Parity + cooldown 30 d')]
rows=[]; farmjson={}
for p,lab in FP:
    cells=[]
    for d in ('GBM','AE','LR'):
        for rk,rel in (('r42','r4.2'),('r52','r5.2')):
            D=F[rk][(F[rk].detector==d)&(F[rk].policy==p)]
            one=D[D.p_clear==1.0]; half=D[(D.p_clear==0.5)]
            e1=one.evaded.max(); base=int(one.detected_no_farming.iloc[0])
            e5=half.evaded.max()
            cells.append(f'{int(round(e1))}/{base}; {f1(e5)}')
            farmjson[(p,d,rel)]=(e1,base,e5,half.exposed_by_probe.max())
    rows.append(lab+' & '+' & '.join(cells)+' \\\\')
farm=r"""\begin{table}[!t]
\caption{Discount Farming: Insiders Who Evade Because They Farmed, Out of Those Detected Without Farming, Maximum Over Attacker Knowledge (Own, Scenario or Generic Profile) and One, Three or Five Probes When Every Probe Alert Is Cleared; After the Semicolon, the Maximum Mean When Each Probe Alert Is Cleared With Probability $0.5$ (Three Probes). Probes Are Capped at the 99th Percentile of Colleagues in the Same Role and Rescored ($b=1\%$)\label{tab:farm}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{AE} & \multicolumn{2}{c}{LR} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Mechanism & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open('tables/tab_farm8.tex','w').write(farm)
# ---------- social cost table
CP=[('B0 + scheduling','Sched.'),('B0 + cooldown 30 days','Cooldown'),('B0 + cooldown 30 days, no override','No overr.'),
    ('B11c deficit apportionment (outcome-aware)','Parity'),('B11c + cooldown 30 days','Par.+cool.')]
def be(r):
    dN,dR=r.d_alerted_employees,r.d_repeat_alerts
    if dR<0 and dN<=0: return 'always'
    if dR>=0 and dN>=0: return 'never' if (dR>0 or dN>0) else 'equal'
    rs=-dN/dR
    if dR<0: return f'$>{rs:.2f}$' if rs<10 else f'$>{rs:.0f}$'
    return f'$<{rs:.2f}$'
rows=[]
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    rows.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{rel}}}}} \\\\')
    for d in ('IF','AE','LR','GBM','LSTM'):
        D=C[rk][C[rk].detector==d].set_index('policy')
        g=D.loc['B0 global']
        cells=[be(D.loc[p]) for p,_ in CP]
        rows.append(f'{d} ({int(g.alerted_employees)}, {int(g.repeat_alerts)}) & '+' & '.join(cells)+' \\\\')
cost=r"""\begin{table}[!t]
\caption{Social Cost of False Alerts, $C=\sum_i[c_1\ind(F_i>0)+c_2(F_i-1)_+]$ Over Benign Employees: Values of $c_2/c_1$ for Which Each Mechanism Costs Less Than the Global Policy. In Parentheses, Employees Alerted and Repeat Alerts Under the Global Policy ($b=1\%$)\label{tab:cost}}
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Det. (global) & Scheduler & Cooldown & No override & Parity & Parity + cooldown \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open('tables/tab_cost8.tex','w').write(cost)
print(tim); print(farm); print(cost)
# ---------- figure
pts=[]
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    Fr=F[rk]
    for d in ('GBM','AE','IF','LR'):
        Cd=C[rk][C[rk].detector==d].set_index('policy'); g=Cd.loc['B0 global','insiders_detected']
        for p,lab in (('B0 + cooldown 30 days','Cooldown'),('B0 + cooldown 30 days, no override','Cooldown, no override'),('B0 + scheduling','Scheduler'),('B11c + cooldown 30 days','Parity + cooldown')):
            D=Fr[(Fr.detector==d)&(Fr.policy==p)&(Fr.p_clear==1)]
            if D.empty: continue
            pts.append((rel,d,lab,Cd.loc[p,'insiders_detected']-g,D.evaded.max(),False))
    for p,lab in POL:
        lab={'B4 label-free':'Parity, label-free','B4c outcome-aware':'Parity, outcome-aware','B4c outcome-aware, 7-day verdict delay':'Parity, outcome-aware, late verdicts','B11c deficit apportionment (outcome-aware)':'Deficit apportionment'}[p]
        for d in ('GBM','LSTM','AE'):
            r=M[(M.release==rel)&(M.detector==d)&(M.policy==POL[[q for q,_ in POL].index(p)][0] if False else p)] if False else None
for rk,rel in (('r42','r4.2'),('r52','r5.2')):
    for p,_ in POL:
        lab={'B4 label-free':'Parity, label-free','B4c outcome-aware':'Parity, outcome-aware','B4c outcome-aware, 7-day verdict delay':'Parity, outcome-aware, late verdicts','B11c deficit apportionment (outcome-aware)':'Deficit apportionment'}[p]
        for d in ('GBM','LSTM','AE'):
            r=M[(M.release==rel)&(M.detector==d)&(M.policy==p)].iloc[0]
            s=S[rk]; s=s[(s.detector==d)&(s.policy==p)].iloc[0]
            pts.append((rel,d,lab,s.none_P-s.none_G,r.primary_excess,r.primary_lo>1e-9))
P=pd.DataFrame(pts,columns=['rel','det','mech','gain','exposure','sig']); P.to_csv('tables/fig_be8_points.csv',index=False)
plt.rcParams.update({'font.size':8,'font.family':'serif'})
fig,axes=plt.subplots(1,2,figsize=(7.0,2.8),sharey=True,dpi=150)
mk={'Cooldown':'o','Cooldown, no override':'s','Scheduler':'^','Parity + cooldown':'*','Parity, label-free':'v','Parity, outcome-aware':'D','Parity, outcome-aware, late verdicts':'P','Deficit apportionment':'X'}
col={'GBM':'#d62728','AE':'#1f77b4','IF':'#2ca02c','LR':'#9467bd','LSTM':'#ff7f0e'}
for ax,rel in zip(axes,('r4.2','r5.2')):
    for _,r in P[P.rel==rel].iterrows():
        ax.scatter(r.gain,r.exposure,marker=mk[r.mech],color=col[r.det],s=(60 if r.mech=='Parity + cooldown' else 26)*(1.3 if r.sig else 1),
                   edgecolors='k',linewidths=1.3 if r.sig else 0.3,zorder=3,alpha=0.9)
    ax.axvline(0,color='0.6',lw=0.6,zorder=1); ax.axhline(0,color='0.6',lw=0.6,zorder=1)
    ax.set_title(f'CERT {rel}',fontsize=8); ax.set_xlabel('Insiders gained without attack (vs. global policy)'); ax.grid(alpha=0.25,lw=0.4)
axes[0].set_ylabel('Insiders lost to the attack')
from matplotlib.lines import Line2D
h1=[Line2D([],[],marker=m,ls='',color='0.75',markeredgecolor='k',markeredgewidth=0.3,markersize=7 if l=='Parity + cooldown' else 5,label=l) for l,m in mk.items()]
h2=[Line2D([],[],marker='o',ls='',color=c,label=d) for d,c in col.items()]
h2.append(Line2D([],[],marker='o',ls='',color='white',markeredgecolor='k',markeredgewidth=1.3,label='95% interval excludes 0'))
fig.legend(handles=h1,loc='lower center',ncol=4,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,0.06))
fig.legend(handles=h2,loc='lower center',ncol=6,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=(0,0.2,1,1))
fig.savefig('figures/benefit_exposure.pdf',bbox_inches='tight')
fig.savefig('figures/be8.png',dpi=150,bbox_inches='tight')
