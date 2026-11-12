"""Main-paper tables II-V and Fig. 2 for the narrowed paper (cooldown and outcome-aware deficit apportionment).
Run from the repository root after tables_main_I_to_III.py and tables_main_IV_V_figures.py."""
import pandas as pd, numpy as np, re
W=''; O='tables/'
P='B11c deficit apportionment (outcome-aware)'
m=pd.read_csv('results/10_attack_realism/m1_multiplicity.csv'); m=m[m.policy==P]
order=[('r4.2','GBM'),('r5.2','GBM'),('r4.2','LSTM'),('r5.2','LSTM'),('r4.2','AE'),('r5.2','AE')]
A=[('random','Random (control)'),('online deficit, q=0.9','Online: reports and colleagues$^\\ddagger$'),('online reports, q=0.9','Online: last report'),
   ('online colleagues, q=0.9','Online: colleagues investigated'),('hindsight deficit','Hindsight: reports and colleagues'),('oracle: knows allocation (hindsight)','Knows the allocation')]
L=[]
for a,lab in A:
    cells=[]; vals=[]; nsig=0
    for rel,det in order:
        r=m[(m.release==rel)&(m.detector==det)].iloc[0]
        v=r[a+' | excess']; lo=r[a+' | lo']; vals.append(v)
        st=''
        if lo>1e-9: st='$^\\ast$'; nsig+=1
        b=a+' | bonferroni_lo'
        if b in r and pd.notna(r[b]) and r[b]>1e-9: st+='$^\\dagger$'
        cells.append(f'{v:.1f}{st}')
    L.append(lab+' & '+' & '.join(cells)+f' & {np.mean(vals):.1f} & {nsig} \\\\')
    if a=='online deficit, q=0.9':
        iv=[]
        for rel,det in order:
            r=m[(m.release==rel)&(m.detector==det)].iloc[0]; iv.append(f'[{r[a+" | lo"]:.0f}, {r[a+" | hi"]:.0f}]')
        L.append('\\quad 95\\% interval & '+' & '.join(iv)+' & & \\\\')
tab=r"""\begin{table}[!t]
\caption{Timing Attacks Against Outcome-Aware Deficit Apportionment, With Behavior Relocated and Rescored: Excess Loss Over the Global Policy on the Same Relocated Attacks, in Insiders (Mean Over Draws), the Mean Over the Six Settings and the Number of Settings Whose 95\% Interval Excludes Zero ($^\ast$). $^\ddagger$Primary Attacker: Starts on the First Day Its Estimate of the Group's Excess, From Noisy Reports and Colleagues' Investigations Seen So Far, Reaches the 90th Percentile of Its Past Values; $^\dagger$: Also After Bonferroni Correction Over the Online Attackers ($b=1\%$)\label{tab:timing}}
\centering
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{2}{c}{GBM} & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{AE} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Attacker & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & Mean & Sig. \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(O+'tab_timing10.tex','w').write(tab); print(tab)
# farm table: drop scheduler row, add pooled share column
f=open('tables/tab_farm9.tex').read()
rows=[l for l in f.splitlines() if l.startswith(('None','Cooldown','Similarity'))]
out=[]
for l in rows:
    if l.startswith('Similarity'): continue
    cells=l.rstrip(' \\').split(' & ')
    ev=sum(int(c.split('/')[0]) for c in cells[1:]); de=sum(int(c.split('/')[1].split(';')[0]) for c in cells[1:])
    ev2=sum(int(c.split(';')[1]) for c in cells[1:])
    out.append(l.rstrip().rstrip('\\').rstrip()+f' & {100*ev/de:.0f}\\%; {100*ev2/de:.0f}\\% \\\\')
    print(cells[0],ev,de,ev2)
f2=f.split('\\midrule')[0]+'\\midrule\n'+'\n'.join(out)+'\n\\bottomrule'+f.split('\\bottomrule')[1]
f2=f2.replace('{lcccccccc}','{lccccccccc}').replace('Mechanism & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 \\\\','Mechanism & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & r4.2 & r5.2 & All \\\\')
f2=f2.replace('($b=1\\%$; 45 Farming Insiders in r4.2, 73 in r5.2)','All: Share of Detected Insiders Who Evade, Over the Eight Settings ($b=1\\%$; 45 of the 51 Test Insiders of r4.2 and 73 of the 76 of r5.2 Have a Benign Test Day Before Their First Malicious Day and Can Farm)')
f2=f2.replace('Alerts All:','Alerts. All:'); open(O+'tab_farm10.tex','w').write(f2); print(f2)
# benefit: drop scheduler column, label the run
b=open('tables/tab_benefit.tex').read()
lines=[]
for l in b.splitlines():
    if ' & ' in l and l.endswith('\\\\') and not l.startswith('\\multicolumn'):
        c=l[:-2].strip().split(' & '); del c[3]; l=' & '.join(c)+' \\\\'
    lines.append(l)
b='\n'.join(lines).replace('{lccccc}','{lcccc}').replace('\\multicolumn{6}','\\multicolumn{5}')
b=b.replace('Parity: Outcome-Aware Deficit Apportionment ($b=1\\%$; Seeds in Tables~\\EREF{tab:s-ni-r42} and~\\EREF{tab:s-ni-r52})',
 'Parity: Outcome-Aware Deficit Apportionment ($b=1\\%$; One Run, the First Seed of Each Detector; Detection Averaged Over Seeds in Table~\\SREF{tab:s-ni-margins})')
assert 'First Seed' in b
open(O+'tab_benefit10.tex','w').write(b)
r=open('tables/tab_rq1.tex').read().replace('($b=1\\%$, Functional Units).','($b=1\\%$, Functional Units; One Run, the First Seed of Each Detector).')
assert 'First Seed' in r
open(O+'tab_rq1_10.tex','w').write(r)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R9='results/10_attack_realism/'; R8='results/09_security_rescored/'
FIG='figures/'
P='B11c deficit apportionment (outcome-aware)'
M=pd.read_csv(R9+'m1_multiplicity.csv'); M=M[M.policy==P]
ATK=[('random','Random'),('online deficit, q=0.9','Online'),('hindsight deficit','Hindsight'),('oracle: knows allocation (hindsight)','Knows allocation')]
F={t:pd.read_csv(R9+f'm2_farming_{t}_summary.csv') for t in ('r42','r52')}
# realism figure
plt.rcParams.update({'font.size':8,'font.family':'serif'})
C=pd.read_csv(R9+'m1_constrained.csv'); C=C[C.policy==P]
levs=['none','within original 100%','within original 90%','within original 75%','within original 50%']
labs=['none','100%','90%','75%','50%']
fig,ax=plt.subplots(1,3,figsize=(7.0,2.75),dpi=150)
# (a) strip by attacker
order=[a for a,_ in ATK]; names=['Random','Online\n(primary)','Hindsight','Knows\nallocation']
for i,a in enumerate(order):
    v=M[f'{a} | excess'].to_numpy(); sig=(M[f'{a} | lo']>1e-9).to_numpy()
    xs=i+np.linspace(-0.25,0.25,len(v))
    ax[0].scatter(xs[~sig],v[~sig],s=10,color='0.6',edgecolors='none')
    ax[0].scatter(xs[sig],v[sig],s=14,color='#d62728',edgecolors='k',linewidths=0.4)
    ax[0].hlines(v.mean(),i-0.3,i+0.3,color='k',lw=1.2)
ax[0].axhline(0,color='0.6',lw=0.6); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names,fontsize=6.5)
ax[0].set_ylabel('Excess loss (insiders)'); ax[0].set_title('(a) Timing: attacker information',fontsize=8)
# (b) timing vs plausibility
for a,nm,c in [('oracle: knows allocation (hindsight)','Knows allocation','#1f77b4'),('hindsight deficit','Hindsight','#9467bd'),(order[1],'Online (primary)','#d62728'),('random','Random','0.5')]:
    y=[C[(C.attack==a)&(C.constraint==l)].excess.mean() for l in levs]
    ax[1].plot(range(5),y,marker='o',ms=3,color=c,label=nm)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labs,fontsize=7); ax[1].axhline(0,color='0.6',lw=0.6)
ax[1].set_xlabel('Relocated days within this quantile\nof the original attacks',fontsize=7)
ax[1].set_ylabel('Mean excess loss (6 settings)'); ax[1].set_title('(b) Timing: plausibility',fontsize=8); ax[1].legend(fontsize=5.5,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.36))
# (c) farming vs plausibility
L=['1.0','0.9','0.75','0.5']; LL=['none','90%','75%','50%']
for t,ls in (('r42','--'),('r52','-')):
    for p,nm,c in [('B0 + cooldown 30 days','Cooldown','#2ca02c'),('B0 + cooldown 30 days, no override','No override','#ff7f0e')]:
        D=F[t][F[t].policy==p]
        y=[(D.groupby('detector').evaded_unconstrained.max().sum() if l=='1.0' else D[D.level.astype(str)==l].groupby('detector').evaded.max().sum()) for l in L]
        ax[2].plot(range(4),y,ls=ls,marker='o',ms=3,color=c,label=f'{nm}, {"r4.2" if t=="r42" else "r5.2"}')
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(LL,fontsize=7)
ax[2].set_xlabel('Probes within this quantile of\ninnocent alerted user-days',fontsize=7)
ax[2].set_ylabel('Evasions, sum of 4 detectors'); ax[2].set_title('(c) Farming: plausibility',fontsize=8); ax[2].legend(fontsize=5.5,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.36))
for a in ax: a.grid(alpha=0.25,lw=0.4)
fig.tight_layout(); fig.savefig(FIG+'realism.pdf',bbox_inches='tight'); fig.savefig('figures/realism.png',dpi=130,bbox_inches='tight')
