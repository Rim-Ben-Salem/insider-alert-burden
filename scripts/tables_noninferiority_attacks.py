import sys, json, re
sys.path.insert(0,'src')
import pandas as pd, numpy as np, fairirs as fi
R='results/06_noninferiority_attacks/'; W='tables/'
T={}
def tex(df,cap,lab,fmt='{:.3f}',narrow=False):
    fi.to_latex(df,W+'t.tex',cap,lab,fmt); t=open(W+'t.tex').read().strip()
    if narrow:
        t=t.replace('\\resizebox{\\columnwidth}{!}{%\n','').replace('\\end{tabular}}','\\end{tabular}')
        t=t.replace('\\centering\n','\\centering\n\\setlength{\\tabcolsep}{4pt}\n',1)
    return t
DO={'IF':0,'AE':1,'LR':2,'GBM':3,'LSTM':4}
def short(p):
    return (p.replace('outcome-aware','OA').replace('label-free','LF')
             .replace('discounted history (half-life 30 days)','discounted (30 d)')
             .replace('deficit apportionment','deficit apport.').replace('randomized displacement','random displ.')
             .replace(', no memory','').replace(' + scheduling',' + sched.'))
# ---------- I2 non-inferiority
d=pd.read_csv(R+'i2_pooled.csv')
d['o']=d.detector.map(DO); d=d.sort_values(['budget','o'],kind='stable')
d['Budget']=d.budget.map(lambda v:f'{v*100:g}%')
d['Policy']=d.policy.map(short)
d['Diff. [95\\% CI]']=[f'{a:+.3f} [{b:+.2f}, {c:+.2f}]' for a,b,c in zip(d['diff'],d.ci_lo,d.ci_hi)]
d['Per run']=d.diff_insiders_per_run.map(lambda v:f'{v:+.1f}')
d['One-sided']=d.one_sided_lo.map(lambda v:f'{v:+.3f}')
d['NI']=d.noninferior.map(lambda v:'yes' if v else 'no')
d['Recall B0']=d.recall_ref.map(lambda v:f'{v:.3f}'); d['Recall']=d.recall.map(lambda v:f'{v:.3f}')
T['s-noninf']=tex(d[['Budget','detector','Policy','runs','Recall B0','Recall','Diff. [95\\% CI]','Per run','One-sided','NI']].rename(columns={'runs':'Runs'}),
 'Non-Inferiority of Detection Against the Global Policy, Pooled Over Seeds (r4.2, $\\delta$ = 0.2 $\\times$ Budget). Per-Insider Detection Is Averaged Over Runs and Resampled Over Insiders; ``Per Run\'\' Is the Mean Change in Insiders Detected; Non-Inferior (NI) if the One-Sided 95\\% Lower Bound Exceeds $-0.05$','tab:s-noninf')
# ---------- I3 windows
d=pd.read_csv(R+'i3_windows.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o',kind='stable')
d['Policy']=d.policy.map(short)
cols=['detector','Policy','insiders_detected','delta_FAR','roll30_dFAR_median','roll30_dFAR_max','roll90_dFAR_median','roll90_dFAR_max','employee_month_dFAR_mean','max_F','fallback_days']
ren={'insiders_detected':'Det.','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','roll30_dFAR_median':'30 d med.','roll30_dFAR_max':'30 d max','roll90_dFAR_median':'90 d med.','roll90_dFAR_max':'90 d max','employee_month_dFAR_mean':'Empl.-month','max_F':'$\\max F_i$','fallback_days':'Fallb.'}
T['s-windows']=tex(d[cols].rename(columns=ren),
 'Memory-Based Parity Rules and Histories: Detection, Cumulative False-Alert Disparity, Disparity Over Rolling 30- and 90-Day Windows (Median and Maximum Over Windows), Mean Employee-Level Disparity Over Calendar Months, Maximum Individual Burden and Fallback Days (r4.2, Functional Units, $\\delta$ = 0.002)','tab:s-windows',fmt='{:.4f}')
# ---------- I4 clustered
d=pd.read_csv(R+'i4_uncertainty.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o',kind='stable')
d['Policy']=d.policy.map(short)
f4=lambda v:'$\\infty$' if not np.isfinite(v) else f'{v:.4f}'
f2=lambda v:'$\\infty$' if not np.isfinite(v) else f'{v:.2f}'
d['$\\Delta_{\\mathrm{FAR}}$']=d.delta_FAR.map(f4)
d['Pair [95\\% CI]']=[f'[{a:+.4f}, {b:+.4f}]' for a,b in zip(d.pair_lo,d.pair_hi)]
d['$\\rho_{\\mathrm{FAR}}$ [95\\% CI]']=[f'{f2(r)} [{f2(a)}, {f2(b)}]' for r,a,b in zip(d.ratio_FAR,d.rho_lo,d.rho_hi)]
d['w/o top 1']=d.dFAR_without_top1.map(f4); d['w/o top 10']=d.dFAR_without_top10.map(f4)
d['Gini']=d.gini_false_alerts.map(lambda v:f'{v:.3f}'); d['Gini random']=d.gini_random_allocation.map(lambda v:f'{v:.3f}')
T['s-cluster']=tex(d[['detector','Policy','$\\Delta_{\\mathrm{FAR}}$','Pair [95\\% CI]','$\\rho_{\\mathrm{FAR}}$ [95\\% CI]','w/o top 1','w/o top 10','Gini','Gini random']],
 'Employee-Clustered Uncertainty and Individual Concentration (r4.2, Functional Units). Pair: 95\\% Interval for the Difference Between the Most- and Least-Flagged Groups, Resampling Benign Employees Within Groups; w/o Top $m$: $\\Delta_{\\mathrm{FAR}}$ Without the $m$ Most-Flagged Benign Employees; Gini of False Alerts per Benign Employee, Actual and Under Random Allocation of the Same Budget','tab:s-cluster')
# ---------- I5 removal
d=pd.read_csv(R+'i5_removal.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o',kind='stable')
d['Policy']=d.policy.map(short); d['Removed']=d.insiders_removed.map(lambda v:'yes' if v else 'no')
T['s-removal']=tex(d[['detector','Policy','Removed','insiders_detected','precision','delta_FAR','ratio_FAR','gini_burden','max_F','distinct_alerted']].rename(columns={'insiders_detected':'Det.','precision':'Prec.','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','ratio_FAR':'$\\rho_{\\mathrm{FAR}}$','gini_burden':'Gini($F$)','max_F':'$\\max F_i$','distinct_alerted':'Alerted'}),
 'Insiders Removed From the Population After Their First Confirmed Alert (r4.2, Functional Units, $\\delta$ = 0.002). Alerted: Benign Employees Alerted at Least Once','tab:s-removal',fmt='{:.4f}')
# ---------- I6 joint errors
d=pd.read_csv(R+'i6_joint_errors.csv'); d=d[d.frac==0.1]
rows=[]
for (pm,wh,f),D in d.groupby(['missed','where','fc_rate'],sort=False):
    r=dict(Missed=f'{int(pm*100)}%',Where=wh,f=f'{f:g}',True_=D[(D.recheck_error==0)&(~D.bonferroni)].true_dFAR.mean(),
           Audit=D.audit.mean(),
           Bound=D[(D.recheck_error==0)&(D.bonferroni)].bound_mean.mean())
    for b in (False,True):
        for e in (0.0,0.2):
            E=D[(D.bonferroni==b)&(D.recheck_error==e)]
            r[('Bonf.' if b else 'Per group')+f' {int(e*100)}\\%']=f'{E.coverage.mean():.3f} ({E.coverage.min():.2f})'
    rows.append(r)
J=pd.DataFrame(rows).rename(columns={'True_':'True $\\Delta_{\\mathrm{FAR}}$','f':'$f$'})
T['s-joint']=tex(J,'Missed Insiders and False Confirmations Together, With a Re-Checker Who Misses 0 or 20\\% of the False Confirmations (r4.2, Outcome-Aware Mode, 10\\% of Confirmations Re-Checked, GBM, LSTM and IF, Three Draws, 20 Re-Check Samples Each). Bound: Corollary~1 With Bonferroni Correction. Coverage: Mean (Minimum) Share of Re-Check Samples in Which the Bound Holds, With Per-Group 95\\% Clopper--Pearson Bounds or With Bonferroni Correction Across Groups','tab:s-joint',fmt='{:.4f}')
# ---------- I7 attacks
d=pd.read_csv(R+'i7_attacks_summary.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o',kind='stable')
d['Policy']=d.policy.map(short)
d['Excess loss [95\\% CI]']=[f'{a:+.1f} [{b:+.1f}, {c:+.1f}]' for a,b,c in zip(d.excess_loss,d.excess_lo,d.excess_hi)]
d['Range']=[f'{a:+.0f} to {b:+.0f}' for a,b in zip(d.excess_loss_min,d.excess_loss_max)]
d['Bound / obs.']=[('--' if not np.isfinite(a) else f'{a:.1f} / {b:.1f}') for a,b in zip(d.bound_expected_misses,d.observed_misses_among_caught_by_global)]
for c in ['none_P','none_G','attacked_P','attacked_G']: d[c]=d[c].map(lambda v:f'{v:.1f}'.rstrip('0').rstrip('.'))
T['s-attacks2']=tex(d[['detector','Policy','attack','draws','none_P','none_G','attacked_P','attacked_G','Excess loss [95\\% CI]','Range','Bound / obs.']].rename(columns={'attack':'Attacker','draws':'Draws','none_P':'P','none_G':'B0','attacked_P':'P att.','attacked_G':'B0 att.'}),
 'Timing Attackers Over Several Draws (r4.2, Functional Units, $\\delta$ = 0.002). Insiders Detected Under Policy P and Under the Global Policy (B0), Without and With the Attack, on the Same Relocated Episodes; Excess Loss: Insiders Lost Under P Beyond Those Lost Under B0, Mean Over Draws With a Paired Bootstrap Interval Over Insiders and the Range Over Draws; Bound / Obs.: Expected Misses Under the Bound of Proposition~S1 and Observed Misses Among Insiders Caught by B0','tab:s-attacks2')
# ---------- I8 validation
d=pd.read_csv(R+'i8_validation.csv'); d['o']=d.detector.map(DO); d=d.sort_values(['o'],kind='stable')
d['Mode']=d['mode'].map({'alerts':'label-free','cleared':'outcome-aware'})
d['Chosen']=d.chosen.map(lambda v:'$\\ast$' if v else '')
d['$\\delta$']=d.delta.map(lambda v:f'{v:g}')
T['s-valid']=tex(d[['detector','Mode','$\\delta$','Chosen','val_lost','val_gained','val_lo','test_detected','test_detected_global','test_dFAR','test_dFAR_global']].rename(columns={'val_lost':'Val. lost','val_gained':'Val. gained','val_lo':'Val. lower bd.','test_detected':'Test det.','test_detected_global':'Test det. B0','test_dFAR':'Test $\\Delta_{\\mathrm{FAR}}$','test_dFAR_global':'Test $\\Delta_{\\mathrm{FAR}}$ B0'}),
 'Tolerance Chosen on a Validation Period (First Half of the r4.2 Test Period, 39 Insiders) and Evaluated on the Second Half (17 Insiders). $\\ast$: Smallest $\\delta$ Whose One-Sided 95\\% Lower Bound on the Validation Recall Difference Exceeds $-0.05$ (the Largest $\\delta$ if None Does)','tab:s-valid',fmt='{:.4f}')
# ---------- I9 ties and groups
d=pd.read_csv(R+'i9_ties.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o')
T['s-ties']=tex(d[['detector','days_with_tie_at_threshold','share_alerts_score_ge_0999','alerted_score_min','alerted_score_median']].rename(columns={'days_with_tie_at_threshold':'Days with tie','share_alerts_score_ge_0999':'Alerts $\\ge0.999$','alerted_score_min':'Min. alerted','alerted_score_median':'Median alerted'}),
 'Score Ties at the Alert Threshold and Scores of Alerted User-Days Under Global Top-$k$ (r4.2, 251 Test Days)','tab:s-ties')
g=pd.read_csv(R+'i9_groups.csv')
g['grouping']=g.grouping.map({'g_functional_unit':'Functional unit','g_department':'Department','g_role':'Role','g_afterhours':'After-hours','g_exposure':'Exposure tercile'})
T['s-groups']=tex(g[['grouping','groups','smallest','median','largest']].rename(columns={'grouping':'Grouping','groups':'Groups','smallest':'Smallest','median':'Median','largest':'Largest'}),
 'Number and Sizes (Employees Active in the Test Period) of Groups After Merging Groups Below 20 Employees Into ``Other\'\' (r4.2)','tab:s-groups',fmt='{:.0f}',narrow=True)
# ---------- I10 warm start
d=pd.read_csv(R+'i10_warm.csv'); d['o']=d.detector.map(DO); d=d.sort_values('o',kind='stable')
d['Mode']=d['mode'].map({'alerts':'label-free','cleared':'outcome-aware'})
d['Start']=d.warm.fillna('none').map({'none':'none','nan':'none','shadow':'shadow month','prior':'prior'})
T['s-warm']=tex(d[['detector','Mode','Start','insiders_detected','delta_FAR','fallback_days','max_constrained_first_30','max_constrained_after_burn_in','share_days_above_delta_after_burn_in']].rename(columns={'insiders_detected':'Det.','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','fallback_days':'Fallback days','max_constrained_first_30':'Max, days 1--30','max_constrained_after_burn_in':'Max, later','share_days_above_delta_after_burn_in':'Days $>\\delta$, later'}),
 'Shadow Month (Global Top-$k$ for 30 Days, Then Parity) and Prior Warm Start (Histories Begin With 30 Pseudo-Days per Group at the Budget Rate) (r4.2, $\\delta$ = 0.002). Max: Largest Disparity of the Constrained Cumulative Rate, Measured on Actual Alerts','tab:s-warm',fmt='{:.4f}')
# ---------- I11 power
d=pd.read_csv(R+'i11_power.csv')
d['grouping']=d.grouping.map({'g_functional_unit':'Functional unit','g_department':'Department','g_role':'Role'})
d['$\\varepsilon$']=d.eps.map(lambda v:'none' if str(v)=='none' else f'{float(v):g}')
T['s-power']=tex(d[['grouping','smallest','largest','$\\varepsilon$','power_gap_0.02','power_gap_0.05','power_gap_0.1','power_gap_0.2','min_detectable_gap_80']].rename(columns={'grouping':'Grouping','smallest':'$R_{\\min}$','largest':'$R_{\\max}$','power_gap_0.02':'Gap 0.02','power_gap_0.05':'Gap 0.05','power_gap_0.1':'Gap 0.1','power_gap_0.2':'Gap 0.2','min_detectable_gap_80':'MDG$_{80}$'}),
 'Power of a Two-Sided 5\\% Test Comparing the Smallest and Largest Group in One Monthly Private Report (Share of the Roster Alerted at Least Once; Base Rate 0.042), and Minimal Gap Detectable With 80\\% Power (MDG$_{80}$), by Privacy Budget per Report (``none\'\': Without Noise)','tab:s-power')
json.dump(T,open(W+'tables_v5.json','w'),indent=1)
for k,v in T.items(): print(k, v.count('\\\\'))
