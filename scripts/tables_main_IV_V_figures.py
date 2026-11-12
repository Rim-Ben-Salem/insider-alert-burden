import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R9='results/10_attack_realism/'; R8='results/09_security_rescored/'
FIG='figures/'
def f1(x): return f'{x:.1f}'.replace('-0.0','0.0')
M=pd.read_csv(R9+'m1_multiplicity.csv')
RULES=[('B4 label-free','Exact, label-free'),('B4c outcome-aware','Exact, outcome-aware'),
       ('B4c outcome-aware, 7-day verdict delay','\\quad verdicts 7 days late'),('B11c deficit apportionment (outcome-aware)','Deficit apportionment')]
ATK=[('random','Random (control)'),('online deficit, q=0.9','Online, reports and colleagues$^\\ddagger$'),
     ('hindsight deficit','Hindsight, reports and colleagues'),('oracle: knows allocation (hindsight)','Knows the allocation')]
rows=[]
for a,al in [x for x in ATK if x[0]!='hindsight deficit']:
    rows.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{al}}}}} \\\\')
    for p,pl in RULES:
        cells=[]
        for d in ('GBM','LSTM','AE'):
            for rel in ('r4.2','r5.2'):
                r=M[(M.release==rel)&(M.detector==d)&(M.policy==p)].iloc[0]
                v=f1(r[f'{a} | excess'])
                if r[f'{a} | lo']>1e-9: v+='$^\\ast$'
                if f'{a} | bonferroni_lo' in r and r[f'{a} | bonferroni_lo']>1e-9: v+='$^\\dagger$'
                cells.append(v)
        rows.append('\\quad '+pl+' & '+' & '.join(cells)+' \\\\')
open('tables/tab_timing9.tex','w').write(r"""\begin{table}[!t]
\caption{Timing Attacks Against Parity by Attacker Information, With Behavior Relocated and Rescored: Excess Loss Over the Global Policy on the Same Relocated Attacks, in Insiders (Mean Over Draws). $^\ddagger$Prespecified: Starts the First Day Its Estimate of the Group's Excess, From Noisy Reports and Colleagues' Investigations Seen So Far, Reaches the 90th Percentile of Its Past Values. $^\ast$: 95\% Interval Excludes Zero; $^\dagger$: Also After Bonferroni Correction Over the Five Online Attackers ($b=1\%$, $\delta=0.002$)\label{tab:timing}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{AE} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Attacker and parity rule & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}}
\end{table}
""")
# farming table from M2
F={t:pd.read_csv(R9+f'm2_farming_{t}_summary.csv') for t in ('r42','r52')}
FP=[('B0 global','None (global top-$k$)'),('B0 + scheduling','Similarity scheduler'),('B0 + cooldown 30 days','Cooldown 30 d, override $0.999$'),('B0 + cooldown 30 days, no override','Cooldown 30 d, no override')]
rows=[]
for p,pl in FP:
    cells=[]
    for d in ('GBM','AE','LR','IF'):
        for t in ('r42','r52'):
            D=F[t][(F[t].detector==d)&(F[t].policy==p)]
            a=D.evaded_unconstrained.max(); b=D[D.level.astype(str)=='0.5'].evaded.max(); base=D.detected_no_farming.iloc[0]
            cells.append(f'{a}/{base}; {b}')
    rows.append(pl+' & '+' & '.join(cells)+' \\\\')
open('tables/tab_farm9.tex','w').write(r"""\begin{table}[!t]
\caption{Discount Farming With Rescored, Role-Capped and Consistent Probes: Insiders Who Evade Because They Farmed, Out of Those Detected Without Farming (Maximum Over Own or Generic Profile and One or Three Probes, Every Probe Alert Cleared); After the Semicolon, When Probes Must Be No More Unusual Than the Median Innocent User-Day the Detector Alerts ($b=1\%$; 45 Farming Insiders in r4.2, 73 in r5.2)\label{tab:farm}}
\centering
\setlength{\tabcolsep}{2pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{AE} & \multicolumn{2}{c}{LR} & \multicolumn{2}{c}{IF} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Mechanism & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}}
\end{table}
""")
# realism figure
plt.rcParams.update({'font.size':8,'font.family':'serif'})
C=pd.read_csv(R9+'m1_constrained.csv')
levs=['none','within original 100%','within original 90%','within original 75%','within original 50%']
labs=['none','100%','90%','75%','50%']
fig,ax=plt.subplots(1,3,figsize=(7.0,2.75),dpi=150)
# (a) strip by attacker
order=[a for a,_ in ATK]; names=['Random','Online\n(prespecified)','Hindsight','Knows\nallocation']
for i,a in enumerate(order):
    v=M[f'{a} | excess'].to_numpy(); sig=(M[f'{a} | lo']>1e-9).to_numpy()
    xs=i+np.linspace(-0.25,0.25,len(v))
    ax[0].scatter(xs[~sig],v[~sig],s=10,color='0.6',edgecolors='none')
    ax[0].scatter(xs[sig],v[sig],s=14,color='#d62728',edgecolors='k',linewidths=0.4)
    ax[0].hlines(v.mean(),i-0.3,i+0.3,color='k',lw=1.2)
ax[0].axhline(0,color='0.6',lw=0.6); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names,fontsize=6.5)
ax[0].set_ylabel('Excess loss (insiders)'); ax[0].set_title('(a) Timing: attacker information',fontsize=8)
# (b) timing vs plausibility
for a,nm,c in [('oracle: knows allocation (hindsight)','Knows allocation','#1f77b4'),('hindsight deficit','Hindsight','#9467bd'),(order[1],'Online (prespecified)','#d62728'),('random','Random','0.5')]:
    y=[C[(C.attack==a)&(C.constraint==l)].excess.mean() for l in levs]
    ax[1].plot(range(5),y,marker='o',ms=3,color=c,label=nm)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labs,fontsize=7); ax[1].axhline(0,color='0.6',lw=0.6)
ax[1].set_xlabel('Relocated days within this quantile\nof the original attacks',fontsize=7)
ax[1].set_ylabel('Mean excess loss (24 settings)'); ax[1].set_title('(b) Timing: plausibility',fontsize=8); ax[1].legend(fontsize=5.5,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.36))
# (c) farming vs plausibility
L=['1.0','0.9','0.75','0.5']; LL=['none','90%','75%','50%']
for t,ls in (('r42','--'),('r52','-')):
    for p,nm,c in [('B0 + cooldown 30 days','Cooldown','#2ca02c'),('B0 + cooldown 30 days, no override','No override','#ff7f0e'),('B0 + scheduling','Scheduler','#8c564b')]:
        D=F[t][F[t].policy==p]
        y=[(D.groupby('detector').evaded_unconstrained.max().sum() if l=='1.0' else D[D.level.astype(str)==l].groupby('detector').evaded.max().sum()) for l in L]
        ax[2].plot(range(4),y,ls=ls,marker='o',ms=3,color=c,label=f'{nm}, {"r4.2" if t=="r42" else "r5.2"}')
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(LL,fontsize=7)
ax[2].set_xlabel('Probes within this quantile of\ninnocent alerted user-days',fontsize=7)
ax[2].set_ylabel('Evasions, sum of 4 detectors'); ax[2].set_title('(c) Farming: plausibility',fontsize=8); ax[2].legend(fontsize=5.5,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.36))
for a in ax: a.grid(alpha=0.25,lw=0.4)
fig.tight_layout(); fig.savefig(FIG+'realism_all_rules.pdf',bbox_inches='tight'); fig.savefig('figures/realism_all_rules.png',dpi=130,bbox_inches='tight')
# frontier figure from L5
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5),dpi=150,sharey=True)
col={'B0 + scheduling':'#8c564b','B0 + cooldown 30 days':'#2ca02c','B0 + cooldown 30 days, no override':'#ff7f0e','B11c deficit apportionment (outcome-aware)':'#1f77b4','B11c + cooldown 30 days':'#9467bd'}
nm={'B0 + scheduling':'Scheduler','B0 + cooldown 30 days':'Cooldown','B0 + cooldown 30 days, no override':'Cooldown, no override','B11c deficit apportionment (outcome-aware)':'Parity (deficit apport.)','B11c + cooldown 30 days':'Parity + cooldown'}
mk={'IF':'o','AE':'s','LR':'^','GBM':'D','LSTM':'v'}
for a_,t,rel in ((ax[0],'r42','r4.2'),(ax[1],'r52','r5.2')):
    X=pd.read_csv(R8+f'l5_social_cost_{t}.csv')
    for d in mk:
        D=X[X.detector==d].set_index('policy'); g=D.loc['B0 global']
        for p in col:
            if p not in D.index: continue
            r=D.loc[p]; xs=r.insiders_detected-g.insiders_detected
            def cost(q,rr): return (rr.alerted_employees+q*rr.repeat_alerts)
            ys=[100*(cost(q,r)-cost(q,g))/cost(q,g) for q in (0.25,1,4)]
            a_.plot([xs,xs],[ys[0],ys[2]],color=col[p],lw=0.8,alpha=0.7)
            a_.scatter([xs],[ys[1]],marker=mk[d],color=col[p],s=18,edgecolors='k',linewidths=0.3,zorder=3)
    a_.axhline(0,color='0.5',lw=0.6); a_.axvline(0,color='0.5',lw=0.6); a_.grid(alpha=0.25,lw=0.4)
    a_.set_title(f'CERT {rel}',fontsize=8); a_.set_xlabel('Insiders gained over the global policy')
ax[0].set_ylabel('Change in social cost (%)')
from matplotlib.lines import Line2D
h=[Line2D([],[],color=c,lw=2,label=nm[p]) for p,c in col.items()]+[Line2D([],[],marker=m,ls='',color='0.7',markeredgecolor='k',label=d) for d,m in mk.items()]
fig.legend(handles=h,loc='lower center',ncol=5,fontsize=6,frameon=False,bbox_to_anchor=(0.5,-0.08))
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig(FIG+'frontier.pdf',bbox_inches='tight'); fig.savefig('figures/frontier.png',dpi=130,bbox_inches='tight')
print(open('tables/tab_timing9.tex').read()[-2500:]); print(open('tables/tab_farm9.tex').read()[-900:])
