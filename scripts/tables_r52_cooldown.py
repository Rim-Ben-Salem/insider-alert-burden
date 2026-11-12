import sys, json
sys.path.insert(0,'src')
import pandas as pd, numpy as np, fairirs as fi
R6='results/07_r52_rules_permutation/'; R7='results/08_cooldown/'; W='tables/'
T={}
def tex(df,cap,lab,fmt='{:.3f}'):
    fi.to_latex(df,W+'t.tex',cap,lab,fmt); return open(W+'t.tex').read().strip()
DO={'IF':0,'AE':1,'LR':2,'GBM':3,'LSTM':4}
def short(p):
    return (p.replace(' (outcome-aware)','').replace('deficit apportionment','deficit apport.').replace('discounted','disc.')
             .replace('randomized rounding','rand. rounding').replace('B0 + ','').replace('scheduling','scheduler'))
def srt(d,cols=('o',)):
    d=d.copy(); d['o']=d.detector.map(DO); return d.sort_values(list(cols),kind='stable')
# permutation
P=pd.concat([pd.read_csv(R6+'j1_permutation_r42.csv'),pd.read_csv(R6+'j1_permutation_r52.csv')])
P=srt(P,('release','o')); P['Policy']=P.policy.map(short)
P['Top 10 removed']=P.dropped_top.map(lambda v:'yes' if v else 'no')
T['s-perm']=tex(P[['release','detector','Policy','Top 10 removed','delta_FAR','null_median','null_p95','p_value']].rename(columns={'release':'Release','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','null_median':'Null median','null_p95':'Null 95th pct.','p_value':'$p$'}),
 'Permutation Test of Group False-Alert Disparity: Group Labels Shuffled Across Employees (5,000 Permutations, Group Sizes Fixed), With and Without the Ten Most-Flagged Benign Employees (Functional Units, $b=1\\%$)','tab:s-perm',fmt='{:.4f}')
# user baselines
U=pd.read_csv(R6+'j3_user_baselines.csv'); U['Policy']=U.policy.map(short)
T['s-user']=tex(U[['detector','Policy','insiders_detected','precision','delta_FAR','ratio_FAR','gini_burden','max_F','distinct_alerted','exposure_bias']].rename(columns={'detector':'Detector','insiders_detected':'Det.','precision':'Prec.','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','ratio_FAR':'$\\rho_{\\mathrm{FAR}}$','gini_burden':'Gini($F$)','max_F':'$\\max F_i$','distinct_alerted':'Alerted','exposure_bias':'Exp. bias'}),
 'Detectors Trained on Deviations From Each Employee\'s Own Training Baseline (r4.2, Functional Units, $b=1\\%$)','tab:s-user',fmt='{:.4f}')
cols=['detector','Policy','insiders_detected','precision','delta_FAR','roll30_dFAR_max','max_F','distinct_alerted','gini_burden','fallback_days']
ren={'insiders_detected':'Det.','precision':'Prec.','delta_FAR':'$\\Delta_{\\mathrm{FAR}}$','roll30_dFAR_max':'30 d max','max_F':'$\\max F_i$','distinct_alerted':'Alerted','gini_burden':'Gini($F$)','fallback_days':'Fallb.'}
J2=pd.read_csv(R6+'j2_baselines_r42.csv'); J2=J2.groupby(['detector','policy'],sort=False)[[c for c in cols if c not in ('detector','Policy')]].mean().reset_index()
J2=srt(J2); J2['Policy']=J2.policy.map(short)
T['s-baselines-r42']=tex(J2[cols].rename(columns=ren),'Suppression, Memoryless Quota and Deficit Apportionment (r4.2, Functional Units, $b=1\\%$, $\\delta=0.002$; Randomized Quota: Mean of Three Draws). Alerted: Benign Employees Alerted at Least Once; Cooldowns Use the Override $0.999$','tab:s-baselines-r42',fmt='{:.4f}')
J8=srt(pd.read_csv(R6+'j8_rules_r52.csv')); J8['Policy']=J8.policy.map(short)
T['s-rules-r52']=tex(J8[cols].rename(columns=ren),'Parity Rules and Scheduler on r5.2 (Functional Units, $b=1\\%$, $\\delta=0.002$)','tab:s-rules-r52',fmt='{:.4f}')
K2=srt(pd.read_csv(R7+'k2_suppression_r52.csv')); K2['Policy']=K2.policy.map(short)
T['s-suppress-r52']=tex(K2[cols].rename(columns=ren),'Suppression on r5.2: Cooldowns of 7 and 30 Days (Override $0.999$, $0.995$ or None), Scheduler, and Their Combination With Deficit Apportionment (Functional Units, $b=1\\%$)','tab:s-suppress-r52',fmt='{:.4f}')
def ni(d,cap,lab):
    d=srt(d,('budget','o')); d['Budget']=d.budget.map(lambda v:f'{v*100:g}%'); d['Policy']=d.policy.map(short)
    d['Per run']=d.diff_insiders_per_run.map(lambda v:f'{v:+.1f}')
    d['Diff. [95\\% CI]']=[f'{a:+.3f} [{b:+.2f}, {c:+.2f}]' for a,b,c in zip(d['diff'],d.ci_lo,d.ci_hi)]
    d['NI']=d.noninferior.map(lambda v:'yes' if v else 'no')
    return tex(d[['Budget','detector','Policy','runs','recall_ref','recall','Per run','Diff. [95\\% CI]','NI']].rename(columns={'runs':'Runs','recall_ref':'Recall B0','recall':'Recall'}),cap,lab)
T['s-ni-r42']=ni(pd.read_csv(R6+'j6_pooled_r42.csv'),'Non-Inferiority Against the Global Policy for Suppression, Deficit Apportionment and the Randomized Quota (r4.2, Pooled Over Seeds, Margin $-0.05$)','tab:s-ni-r42')
T['s-ni-r52']=ni(pd.concat([pd.read_csv(R6+'j8_ni_pooled_r52.csv'),pd.read_csv(R7+'k2_ni_pooled_r52.csv')]),'Non-Inferiority Against the Global Policy on r5.2 (76 Insiders; Three Seeds for IF, AE and GBM, One Run for LR and the LSTM; Margin $-0.05$)','tab:s-ni-r52')
def farm(f,rel):
    d=srt(pd.read_csv(f)); d['Policy']=d.policy.map(short)
    return tex(d[['detector','alpha','Policy','farmers','detected_no_farming','detected_farming','lost','gained','farmer_benign_alerts']].rename(columns={'alpha':'$\\alpha$','farmers':'Farmers','detected_no_farming':'Det. no farming','detected_farming':'Det. farming','lost':'Lost','gained':'Gained','farmer_benign_alerts':'Benign alerts'}),
      f'Discount Farming Against Suppression ({rel}, $b=1\\%$): Farming Insiders Detected Without and With Farming, Insiders Lost and Gained, and Alerts on the Benign Days of Farming Insiders','tab:s-farm-'+rel.replace('.',''),fmt='{:.0f}')
T['s-farm-r42']=farm(R7+'k1_farming_r42.csv','r4.2'); T['s-farm-r52']=farm(R7+'k1_farming_r52.csv','r5.2')
def att(d,cap,lab):
    d=srt(d); d['Policy']=d.policy.map(short)
    d['Excess [95\\% CI]']=[f'{a:+.1f} [{b:+.1f}, {c:+.1f}]' for a,b,c in zip(d.excess_loss,d.excess_lo,d.excess_hi)]
    d['Range']=[f'{a:+.0f} to {b:+.0f}' for a,b in zip(d.excess_loss_min,d.excess_loss_max)]
    for c in ['none_P','none_G','attacked_P','attacked_G']: d[c]=d[c].map(lambda v:f'{v:.1f}'.rstrip('0').rstrip('.'))
    return tex(d[['detector','Policy','attack','draws','none_P','none_G','attacked_P','attacked_G','Excess [95\\% CI]','Range']].rename(columns={'attack':'Attacker','draws':'Draws','none_P':'P','none_G':'B0','attacked_P':'P att.','attacked_G':'B0 att.'}),cap,lab)
T['s-attacks-delay']=att(pd.concat([pd.read_csv(R6+'j4_attacks_r42_summary.csv'),pd.read_csv(R7+'k3_attacks_r42_summary.csv')]),
 'Timing Attackers Against Outcome-Aware Parity With Delayed or Missing Verdicts, Against Deficit Apportionment, and Against Parity Combined With a 30-Day Cooldown (r4.2). Columns as in Table~\\ref{tab:s-attacks2}','tab:s-attacks-delay')
T['s-attacks-r52']=att(pd.read_csv(R6+'j8_attacks_r52_summary.csv'),'Timing Attackers on r5.2 (76 Insiders, Three Draws). Columns as in Table~\\ref{tab:s-attacks2}','tab:s-attacks-r52')
C=pd.concat([pd.read_csv(R6+'j7_calibration_r42.csv'),pd.read_csv(R6+'j7_calibration_r52.csv')]); C=srt(C,('release','o'))
C['band']=C.band.str.replace('daily score mass vs. true alerts (Spearman)','Spearman: daily score mass vs. true alerts',regex=False)
T['s-calib']=tex(C[['release','detector','band','user_days','malicious','share_malicious']].rename(columns={'release':'Release','band':'Score band','user_days':'User-days','malicious':'Malicious','share_malicious':'Share malicious'}),
 'Score Calibration: Share of Malicious User-Days by Score Band, and Rank Correlation Between the Daily Score Mass of Global Top-$k$ Alerts and the Number of True Alerts (Last Row per Detector)','tab:s-calib',fmt='{:.4f}')
G=pd.read_csv(R6+'j8_groups_r52.csv'); G['grouping']=G.grouping.map({'g_functional_unit':'Functional unit','g_department':'Department','g_role':'Role','g_afterhours':'After-hours','g_exposure':'Exposure tercile'})
T['s-groups-r52']=tex(G[['grouping','groups','smallest','median','largest']].rename(columns={'grouping':'Grouping','groups':'Groups','smallest':'Smallest','median':'Median','largest':'Largest'}),
 'Number and Sizes of Groups in r5.2 After Merging Groups Below 20 Employees Into ``Other\'\'','tab:s-groups-r52',fmt='{:.0f}')
json.dump(T,open(W+'tables_v7.json','w'),indent=1)
for k,v in T.items(): print(k, v.count('\\\\'))
