import sys, json, re
sys.path.insert(0,'src')
import pandas as pd, numpy as np, fairirs as fi
R='results/05_additional_D/'; W='tables/'
T={}
def tex(df,cap,lab,fmt='{:.3f}',narrow=False):
    fi.to_latex(df,W+'t.tex',cap,lab,fmt); t=open(W+'t.tex').read().strip()
    t=t.replace('Gini($B$)','Gini($F$)').replace(r'$\max B_i$',r'$\max F_i$')
    spec=re.search(r'\\begin\{tabular\}\{([^}]*)\}',t).group(1)
    if len(spec)<=5 or narrow:
        t=t.replace('\\resizebox{\\columnwidth}{!}{%\n','').replace('\\end{tabular}}','\\end{tabular}')
        t=t.replace('\\centering\n','\\centering\n\\setlength{\\tabcolsep}{4pt}\n',1)
    return t
DO={'IF':0,'AE':1,'LR':2,'GBM':3,'LSTM':4}
# --- paired
h1=pd.read_csv(R+'h1_paired.csv'); m=h1[h1.run=='main'].copy()
m['Difference [95\\% CI]']=[f'{d:+.3f} [{lo:+.2f}, {hi:+.2f}]' for d,lo,hi in zip(m['diff'],m.diff_lo,m.diff_hi)]
m['p']=m.mcnemar_p.map(lambda v:f'{v:.2f}')
m=m.rename(columns={'recall_ref':'Recall B0','recall_policy':'Recall','lost':'Lost','gained':'Gained','p':'McNemar $p$'})
T['s-paired']=tex(m[['detector','policy','Recall B0','Recall','Difference [95\\% CI]','Lost','Gained','McNemar $p$']],
 'Paired Comparison of Insider Detection With the Global Policy (r4.2, Main Run): Insiders Lost and Gained, Recall Difference With Paired Bootstrap Interval, and Exact McNemar Test','tab:s-paired')
p=pd.read_csv(R+'h1_pooled.csv'); p['detector']=p.detector.replace({'all':'all detectors'})
p['o']=p.detector.map(lambda d:DO.get(d,9)); p=p.sort_values(['policy','o'])
p['pooled_sign_p']=p.pooled_sign_p.map(lambda v:f'{v:.3f}')
T['s-paired-pooled']=tex(p[['detector','policy','runs','mean_diff','min_diff','max_diff','lost_total','gained_total','pooled_sign_p']].rename(columns={'runs':'Runs','mean_diff':'Mean diff.','min_diff':'Min','max_diff':'Max','lost_total':'Lost','gained_total':'Gained','pooled_sign_p':'Sign test $p$'}),
 'Paired Comparison Pooled Over Five Seeds (r4.2; Descriptive, Since the Seeds Share the Same Insiders)','tab:s-paired-pooled')
# --- dual
h2=pd.read_csv(R+'h2_dual.csv')
sel=h2[(h2.policy=='B0 global')|(h2.policy.str.startswith('FAIR-IRS'))|((h2['mode']=='label-free')&(((h2.policy.str.contains('proportional'))&h2.eta.isin([0.1,1.0,10.0]))|((h2.policy.str.contains('ascent'))&h2.eta.isin([0.3,3.0]))))].copy()
sel['policy']=sel.policy.str.replace('B10 dual prices (proportional)','B10 proportional').str.replace('B10 dual prices (dual ascent)','B10 dual ascent').str.replace('FAIR-IRS label-free','B4 label-free').str.replace('FAIR-IRS outcome-aware','B4c outcome-aware')
sel['eta']=sel.eta.map(lambda v:'--' if v==0 else f'{v:g}')
sel['o']=sel.detector.map(DO); sel=sel.sort_values('o',kind='stable')
T['s-dual']=tex(sel[['detector','policy','eta','insiders_detected','precision','delta_AR','delta_FAR','ratio_FAR','max_F']].rename(columns={'eta':'$\\eta$','insiders_detected':'Detected','max_F':'$\\max F_i$'}),
 'Online Dual-Price Baseline (B10, Label-Free Rates) Versus FAIR-IRS (r4.2, Functional Units, $\\delta$ = 0.002). The Step Size $\\eta$ Is Chosen on the Test Period, Which Favors the Baseline','tab:s-dual',fmt='{:.4f}')
# --- frontier
h3=pd.read_csv(R+'h3_frontier_summary.csv')
h3['mode']=h3['mode'].map({'global':'global','alerts':'label-free','cleared':'outcome-aware'})
h3['delta']=h3.delta.map(lambda v:'--' if v>=1 else f'{v:g}'); h3['protect']=h3.protect.map(lambda v:f'{int(v*100)}%')
h3['random']=h3['random'].map(lambda v:f'{v:.1f}')
cols=['mode','delta','protect','delta_FAR','fallback_days','detected','knows_allocation','observes_colleagues','random','worst_case_loss','average_loss']
ren={'mode':'Mode','delta':'$\\delta$','protect':'Protected','fallback_days':'Fallback days','detected':'No attack','knows_allocation':'Knows alloc.','observes_colleagues':'Observes','random':'Random','worst_case_loss':'Worst loss','average_loss':'Mean loss'}
for c in ['knows_allocation','observes_colleagues']: h3[c]=h3[c].map(lambda v:f'{v:g}')
h3['worst_case_loss']=h3.worst_case_loss.map(lambda v:f'{v:.1f}')
h3['average_loss']=h3.average_loss.map(lambda v:f'{v:.1f}')
T['s-frontier']=tex(h3[['detector']+cols].rename(columns=ren),'Fairness--Security Frontier: Insiders Detected Under Each Attacker and Worst-Case and Mean Loss, by Mode, Tolerance and Share of the Budget Protected for the Global Top Scores (r4.2)','tab:s-frontier',fmt='{:.4f}')
G=h3[h3.detector=='GBM']
keep=[('global','--','0%'),('label-free','0.002','0%'),('outcome-aware','0.002','0%'),('label-free','0.002','50%'),('outcome-aware','0.005','0%'),('outcome-aware','0.01','0%'),('outcome-aware','0.01','25%'),('outcome-aware','0.01','50%')]
rows=[G[(G['mode']==a)&(G.delta==b)&(G.protect==c)].iloc[0] for a,b,c in keep]
F=pd.DataFrame(rows)
T['frontier']=tex(F[cols].rename(columns=ren).drop(columns=['Random']),'Fairness--Security Frontier for GBM: Insiders Detected (of 51) Under Timing Attacks (r4.2; Protected: Share of the Budget Always Given to the Global Top Scores)','tab:frontier',fmt='{:.4f}')
# --- false confirmations
bc=pd.read_csv(R+'h4_bound_check.csv'); bc=bc[bc.frac==0.1]
h4=pd.read_csv(R+'h4_false_confirmation.csv')
a=h4.groupby(['detector','where','rate','policy'])[['delta_FAR','delta_FAR_hat']].mean().unstack('policy')
b=bc.groupby(['detector','where','rate'])[['bound','holds']].mean()
rows=[]
for det in ['GBM','LSTM','IF']:
    for wh in ['all groups','smallest group','largest group']:
        for r in [0.0,0.05,0.1,0.2,0.3]:
            if r==0 and wh!='all groups': continue
            x=a.loc[(det,wh,r)]; y=b.loc[(det,wh,r)]
            rows.append({'detector':det,'Errors in':('--' if r==0 else wh),'$f$':f'{r:g}','$\\Delta_{\\mathrm{FAR}}$ B4':x[('delta_FAR','B4 label-free')],
                         '$\\Delta_{\\mathrm{FAR}}$ B4c':x[('delta_FAR','B4c outcome-aware')],'Audit B4c':x[('delta_FAR_hat','B4c outcome-aware')],
                         'Bound, re-check':y['bound'],'Holds':f"{y['holds']*100:.0f}%"})
T['s-fc']=tex(pd.DataFrame(rows),'Benign Alerts Wrongly Confirmed at Rate $f$: True Disparity, Outcome-Based Audit, and the Bound of Corollary~1 With an Independent Re-Check of 10\\% of Confirmed Alerts (r4.2, $\\delta$ = 0.002, Mean of Three Draws)','tab:s-fc',fmt='{:.4f}')
json.dump(T,open(W+'tables_v4.json','w'))
print(T['frontier']); print(T['s-fc'][:900])
