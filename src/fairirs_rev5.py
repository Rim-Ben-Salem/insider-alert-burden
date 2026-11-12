"""Experiments for the fifth review of the FAIR-IRS paper.

I1  LSTM retrained with several seeds (cached)
I2  Non-inferiority of detection, seed-pooled, for three budgets
I3  Disparity over rolling 30- and 90-day windows; windowed and discounted histories;
    deficit apportionment as a baseline
I4  Employee-clustered bootstrap intervals, removal of the most-flagged employees,
    reference Gini under random allocation
I5  Insiders removed after their first confirmed alert
I6  Missed insiders and false confirmations together, with a fallible re-checker;
    joint (Bonferroni) confidence level for the bound of Corollary 1
I7  Timing attackers over many draws, compared with the global policy on the SAME
    relocated episodes; deficit-estimating attacker; randomized displacement
I8  Tolerance chosen on a validation period (first half of the test period)
I9  Score ties and group counts
I10 Shadow month and prior warm start
I11 Power of a test on private monthly reports
"""
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import fairirs as fi
import fairirs_rev as fr
import fairirs_rev3 as g3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================
# Simulator (extends fairirs_rev3.simulate_g; defaults reproduce it)
# =============================================================================

def _deficit(sc, gids, k, A, C):
    """Deficit apportionment used alone: today's alerts go to groups in proportion to
    their deficit with respect to the common cumulative rate (largest remainder);
    within a group, the highest scores are alerted."""
    labels, inv = np.unique(gids, return_inverse=True)
    G = len(labels)
    order = [np.nonzero(inv == g)[0] for g in range(G)]
    order = [o[np.argsort(-sc[o], kind="stable")] for o in order]
    n = np.array([len(o) for o in order])
    a = np.array([A.get(l, 0) for l in labels], float)
    D = np.array([C.get(l, 0) for l in labels], float) + n
    k = min(k, int(n.sum()))
    target = (k + a.sum()) * D / D.sum() - a
    e = np.clip(target, 0, n)
    if e.sum() <= 0:
        e = n.astype(float)
    exact = k * e / e.sum()
    kg = np.minimum(np.floor(exact).astype(int), n)
    rem = k - kg.sum()
    for g in np.argsort(-(exact - kg)):
        if rem <= 0:
            break
        if kg[g] < n[g]:
            kg[g] += 1
            rem -= 1
    while rem > 0:
        g = int(np.argmax(n - kg))
        kg[g] += 1
        rem -= 1
    m = np.zeros(len(sc), bool)
    for g in range(G):
        m[order[g][:kg[g]]] = True
    return m


def _randround(sc, gids, k, rng):
    """Memoryless quota: each group's expected share k*n_g/N, rounded by systematic
    sampling so that the shares sum to k and each group receives its share in expectation."""
    labels, inv = np.unique(gids, return_inverse=True)
    n = np.bincount(inv, minlength=len(labels)).astype(float)
    exact = k * n / n.sum()
    cum = np.concatenate([[0.0], np.cumsum(exact)])
    u = rng.random()
    pts = u + np.arange(k)
    kg = np.array([int(((pts >= cum[g]) & (pts < cum[g + 1])).sum()) for g in range(len(labels))])
    kg = np.minimum(kg, n.astype(int))
    m = np.zeros(len(sc), bool)
    for g in range(len(labels)):
        o = np.nonzero(inv == g)[0]
        o = o[np.argsort(-sc[o], kind="stable")]
        m[o[:kg[g]]] = True
    return m


def simulate_h(sd, budget=0.01, delta=0.002, group_col="g_functional_unit", history="alerts",
               allocator="fair", floor=None, delay=0, missing=0.0,
               p_miss=0.0, p_fc=0.0, err_group=None, p_miss_g=0.0, p_fc_g=0.0,
               schedule=False, lam=0.01, gamma=0.8, tau_ov=0.999, m_cap=3, horizon=30,
               record=False, seed=0, eta=0.0,
               hist_window=None, hist_decay=None, warm=None, warm_days=30,
               remove_confirmed=False, displace="lowest", eval_from=None, cooldown=0,
               p_inconclusive=0.0, p_miss_low=None, p_fc_high=None, score_split=0.999, fc_col=None,
               cd_delay=0, cd_backfill=True, cd_recheck=0.0, cd_recent_days=0, cd_recent_tau=None, cd_high_days=None, cd_high_split=0.99, state_log=None,
               rc_cap=0, rc_window=30):
    """history: 'global', 'alerts' (label-free), 'cleared' (outcome-aware), 'none' (no memory).
    allocator: 'fair' (FAIR-IRS), 'deficit' (deficit apportionment alone), 'fairstar', 'dual'.
    hist_window  only the last `hist_window` days count in the group histories
    hist_decay   day weights hist_decay**age in the group histories
    warm         None; 'shadow' (global top-k for the first `warm_days` days, histories
                 accumulated, parity enforced afterwards); 'prior' (histories start with
                 `warm_days` pseudo-days per group at the budget rate)
    remove_confirmed  an insider leaves the population once an alert on them is confirmed
    displace     'lowest' (FAIR-IRS: a group whose alerts are cut keeps its highest scores) or
                 'random' (a cut group keeps k_g alerts drawn uniformly from its global top K_g)
    cooldown     an employee whose alert was cleared is not alerted again for `cooldown` days
                 unless their score reaches tau_ov (ordinary SOC practice)
    allocator 'randround': memoryless proportional quota with randomized (systematic) rounding
    p_inconclusive  share of benign alerts closed as inconclusive: they count as burden but start
                    no cooldown and are not stored by the scheduler
    p_miss_low      if set, malicious alerts scored below score_split are missed with this
                    probability (p_miss applies above it)
    p_fc_high       if set, benign alerts scored at or above score_split are wrongly confirmed
                    with this probability (p_fc applies below it)
    cd_delay        days before a clearance verdict is known (the cooldown starts then)
    cd_backfill     False: a suppressed alert is dropped and its slot is not given to anyone else
    cd_recheck      probability that a suppressed alert is investigated anyway (random re-check)
    cd_recent_days, cd_recent_tau
                    farming-signature monitor: within cd_recent_days of a clearance, a score of at
                    least cd_recent_tau is not suppressed
    cd_high_days    cooldown length after clearing an alert scored at least cd_high_split
                    (score-dependent cooldown); `cooldown` applies otherwise
    fc_col          optional column of per-row probabilities that a benign alert is confirmed
                    (NaN: the default applies); used for farming probes that may not be cleared
    rc_cap, rc_window
                    verdict-independent repeat cap: an employee with at least rc_cap alerts in the
                    last rc_window days is not alerted unless their score reaches tau_ov
    """
    rng = np.random.default_rng(seed)
    T = sd.test
    days = T["day"].to_numpy()
    users = T["user"].to_numpy()
    y = T["y"].to_numpy()
    s = T["score"].to_numpy().astype(np.float64)
    fcv = T[fc_col].to_numpy(np.float64) if fc_col is not None and fc_col in T else None
    g = T[group_col].to_numpy()
    alert = np.zeros(len(T), bool)
    cleared = np.zeros(len(T), bool)
    removed = np.zeros(len(T), bool)
    gone = {}                                            # user -> day index from which removed
    cumA, cumC, cumConf = {}, {}, {}
    ledger = hist_window is not None or hist_decay is not None
    dA, dC, dF = [], [], []                              # per-day group counts (ledger)
    pending = []                                         # (reveal day, group, alert day, user)
    fb_days, budget_total, disp, rec = [], 0, [], []
    store, n_disc = {}, 0
    mu = {}
    last_clear = {}
    rc_hist = {}
    if schedule:
        phin = sd.phi / (np.linalg.norm(sd.phi, axis=1, keepdims=True) + 1e-9)
    uniq_days, starts = np.unique(days, return_index=True)
    bounds = list(starts) + [len(T)]
    if warm == "prior":
        first = np.arange(bounds[0], bounds[1])
        for lab, cnt in zip(*np.unique(g[first], return_counts=True)):
            cumC[lab] = cnt * warm_days
            cumA[lab] = budget * cnt * warm_days
    for di, d in enumerate(uniq_days):
        idx = np.arange(bounds[di], bounds[di + 1])
        if gone:
            keep_mask = np.array([gone.get(users[i], 10 ** 9) > di for i in idx])
            removed[idx[~keep_mask]] = True
            idx = idx[keep_mask]
            if len(idx) == 0:
                continue
        if ledger:
            dA.append({}); dC.append({}); dF.append({})
        keep = []
        for (rd, lab, ad, u) in pending:
            if rd <= di:
                if history == "cleared":
                    cumA[lab] = cumA.get(lab, 0) - 1
                    cumConf[lab] = cumConf.get(lab, 0) + 1
                    if ledger:
                        dA[ad][lab] = dA[ad].get(lab, 0) - 1
                        dF[ad][lab] = dF[ad].get(lab, 0) + 1
                if remove_confirmed and u is not None:
                    gone.setdefault(u, di)
            else:
                keep.append((rd, lab, ad, u))
        pending = keep
        sc = s[idx].copy()
        if schedule:
            for j, i in enumerate(idx):
                hist = store.get(users[i])
                if not hist:
                    continue
                hist[:] = [(dd, v) for dd, v in hist if (d - dd) / np.timedelta64(1, "D") <= horizon]
                if not hist or sc[j] >= tau_ov:
                    continue
                if max(float(phin[i] @ v) for _, v in hist) >= gamma:
                    sc[j] -= lam * min(len(hist), m_cap)
                    n_disc += 1
        supp = []
        if cooldown > 0 and last_clear:
            for j, i in enumerate(idx):
                lc = last_clear.get(users[i])
                if not lc:
                    continue
                arrived = [e for e in lc if e[0] <= di]
                if not arrived:
                    continue
                lday, lsc = arrived[-1]
                length = cooldown if (cd_high_days is None or lsc < cd_high_split) else cd_high_days
                if di - lday <= length and sc[j] < tau_ov:
                    if cd_recent_tau is not None and di - lday <= cd_recent_days and sc[j] >= cd_recent_tau:
                        continue
                    if cd_recheck > 0 and rng.random() < cd_recheck:
                        continue
                    if cd_backfill:
                        sc[j] = -1.0
                    else:
                        supp.append(i)
        if rc_cap > 0 and rc_hist:
            for j, i in enumerate(idx):
                hh = rc_hist.get(users[i])
                if not hh or sc[j] >= tau_ov:
                    continue
                if sum(1 for x in hh if di - x <= rc_window) >= rc_cap:
                    sc[j] = -1.0
        k = max(1, int(round(budget * len(idx))))
        gi = g[idx]
        # histories seen by the allocator
        if ledger:
            A0, C0 = {}, {}
            now = len(dA) - 1
            for dd in range(now):
                age = now - dd
                if hist_window is not None and age > hist_window:
                    continue
                w = 1.0 if hist_decay is None else hist_decay ** age
                for lab, v in dA[dd].items():
                    A0[lab] = A0.get(lab, 0.0) + w * v
                for lab, v in dC[dd].items():
                    C0[lab] = C0.get(lab, 0.0) + w * v
                if history == "cleared":
                    for lab, v in dF[dd].items():
                        C0[lab] = C0.get(lab, 0.0) - w * v
        elif history == "cleared":
            A0 = cumA
            C0 = {lab: cumC.get(lab, 0) - cumConf.get(lab, 0) for lab in cumC}
        else:
            A0, C0 = cumA, cumC
        use_global = history == "global" or delta >= 1 or (warm == "shadow" and di < warm_days)
        if state_log is not None:
            state_log[d] = (dict(A0), dict(C0), k, use_global)
        fb = 0
        if use_global:
            m = np.zeros(len(idx), bool)
            m[np.argsort(-sc, kind="stable")[:k]] = True
        elif allocator == "dual":
            adj = sc - np.array([mu.get(lab, 0.0) for lab in gi])
            m = np.zeros(len(idx), bool)
            m[np.argsort(-adj, kind="stable")[:k]] = True
        elif allocator == "fairstar":
            m = g3._fairstar(sc, gi, k)
        elif allocator == "deficit":
            m = _deficit(sc, gi, k, A0, C0)
        elif allocator == "randround":
            m = _randround(sc, gi, k, rng)
        elif history == "none":
            m, fb = fi.fair_allocate_cum(sc, gi, k, delta, {}, {})
        else:
            m, fb = fi.fair_allocate_cum(sc, gi, k, delta, A0, C0)
        rank = np.empty(len(idx), int)
        rank[np.argsort(-sc, kind="stable")] = np.arange(1, len(idx) + 1)
        top = rank <= k
        if displace == "random" and not use_global:
            m2 = m.copy()
            for lab in np.unique(gi):
                sel = gi == lab
                Kg, kg = int((top & sel).sum()), int((m & sel).sum())
                if kg < Kg:
                    cand = np.flatnonzero(top & sel)
                    m2[sel] = False
                    if kg > 0:
                        m2[rng.choice(cand, size=kg, replace=False)] = True
            m = m2
        if fb:
            fb_days.append(di)
        if floor is not None:
            m = m & (rank <= int(np.ceil(floor * k)))
        disp.extend((rank[m & ~top] / k).tolist())
        budget_total += k
        if record:
            for lab in np.unique(gi):
                sel = gi == lab
                rec.append((d, lab, int(top[sel].sum()), int(m[sel].sum()), int(sel.sum())))
        a_idx = idx[m]
        if supp:
            a_idx = np.setdiff1d(a_idx, np.asarray(supp), assume_unique=False)
        alert[a_idx] = True
        if rc_cap > 0:
            for i in a_idx:
                rc_hist.setdefault(users[i], []).append(di)
        for lab, cnt in zip(*np.unique(gi, return_counts=True)):
            cumC[lab] = cumC.get(lab, 0) + cnt
            if ledger:
                dC[-1][lab] = dC[-1].get(lab, 0) + cnt
        for i in a_idx:
            lab = g[i]
            pm = p_miss_g if (err_group is not None and lab == err_group) else p_miss
            pf = p_fc_g if (err_group is not None and lab == err_group) else p_fc
            if p_miss_low is not None and y[i] == 1 and s[i] < score_split:
                pm = p_miss_low
            if p_fc_high is not None and y[i] == 0 and s[i] >= score_split:
                pf = p_fc_high
            if fcv is not None and y[i] == 0 and not np.isnan(fcv[i]):
                pf = fcv[i]
            if y[i] == 1:
                final_clear = rng.random() < pm
            else:
                final_clear = not (rng.random() < pf)
            cleared[i] = final_clear
            inconclusive = p_inconclusive > 0 and y[i] == 0 and final_clear and rng.random() < p_inconclusive
            if final_clear and not inconclusive:
                lst = last_clear.setdefault(users[i], [])
                lst.append((di + cd_delay, s[i]))
                if len(lst) > 1:
                    lst.sort(key=lambda e: e[0])
                    while len(lst) > 1 and lst[1][0] <= di:
                        lst.pop(0)                      # keep the latest arrived verdict and those pending
            cumA[lab] = cumA.get(lab, 0) + 1
            if ledger:
                dA[-1][lab] = dA[-1].get(lab, 0) + 1
            if schedule and final_clear and not inconclusive:
                store.setdefault(users[i], []).append((d, phin[i]))
            if final_clear or rng.random() < missing:
                continue
            if history == "cleared" and y[i] == 1:
                rng.random()                             # keeps the random stream of simulate_g (revise=0)
            u = users[i] if (remove_confirmed and y[i] == 1) else None
            if history == "cleared" or u is not None:
                pending.append((di + max(delay, 1 if history != "cleared" else 0), lab, len(dA) - 1 if ledger else 0, u))
        if allocator == "dual":
            den = ({lab: cumC[lab] - cumConf.get(lab, 0) for lab in cumC} if history == "cleared" else dict(cumC))
            rates = {lab: cumA.get(lab, 0) / max(den[lab], 1) for lab in den}
            rbar = sum(cumA.get(lab, 0) for lab in den) / max(sum(den.values()), 1)
            for lab, r in rates.items():
                mu[lab] = eta * (r - rbar)
    out = T[["user", "day", "y", "score", "exposure", "scenario", group_col]].copy()
    out["alert"], out["cleared"] = alert, cleared
    if remove_confirmed:
        out = out[~removed].copy()
    if eval_from is not None:
        out = out[out["day"] >= eval_from].copy()
    out.attrs.update(fallback_days=len(fb_days), fallback_list=fb_days, discounts=n_disc,
                     n_days=len(uniq_days), disp_ranks=np.array(disp), budget_total=budget_total,
                     removed_rows=int(removed.sum()))
    if record:
        return out, pd.DataFrame(rec, columns=["day", "group", "K", "k", "n"])
    return out


def metrics(out, group_col, **extra):
    m = fi.policy_metrics(out, group_col)
    m.update(g3.harm(out))
    m.update(extra)
    return m


def insider_hits(out):
    mal = out[out["y"] == 1]
    return mal.groupby("user")["alert"].any().astype(int)


DETS = ("IF", "AE", "LR", "GBM", "LSTM")


# =============================================================================
# I1  LSTM seeds
# =============================================================================

def lstm_seeds(ds, seeds, cache_dir):
    """Retrain the LSTM with each seed (cached on Drive). Returns {seed: ScoredData}."""
    import os
    import pickle
    res = {}
    for s in seeds:
        p = os.path.join(cache_dir, f"scored_{ds.name}_lstm_seed{s}_v2.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res[s] = pickle.load(f)
            log(f"LSTM seed {s}: reloaded")
            continue
        sd = fi.run_detector(ds, fi.LSTMDetector(s))
        with open(p, "wb") as f:
            pickle.dump(sd, f)
        res[s] = sd
        log(f"LSTM seed {s}: trained and cached")
    return res


# =============================================================================
# I2  Non-inferiority
# =============================================================================

def _boot_mean(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1)


def i2_noninferiority(runs, budgets=(0.005, 0.01, 0.02), delta_rel=0.2, margin=0.05,
                      group_col="g_functional_unit", pols=None, sched_budget=0.01):
    """runs: {detector: {run_name: ScoredData}}. For each detector, budget and policy:
    per run, insiders lost and gained against the global policy; seed-pooled, the
    per-insider detection averaged over runs, and a paired bootstrap over insiders of
    the recall difference. Non-inferiority: one-sided 95% lower bound > -margin."""
    pols = pols or [("B4 label-free", dict(history="alerts")),
                    ("B4c outcome-aware", dict(history="cleared")),
                    ("B5c outcome-aware + scheduling", dict(history="cleared", schedule=True))]
    per_run, pooled = [], []
    for det, rr in runs.items():
        for b in budgets:
            dl = delta_rel * b
            ref = {}
            for rn, sd in rr.items():
                ref[rn] = insider_hits(simulate_h(sd, b, 1.0, group_col, history="global"))
            for pol, kw in pols:
                if kw.get("schedule") and sched_budget is not None and abs(b - sched_budget) > 1e-12:
                    continue
                hits = {}
                for rn, sd in rr.items():
                    if kw.get("schedule") and sd.phi is None:
                        continue
                    h = insider_hits(simulate_h(sd, b, dl, group_col, **kw))
                    a, c = ref[rn].align(h, join="inner")
                    hits[rn] = (a, c)
                    per_run.append(dict(detector=det, budget=b, delta=dl, policy=pol, run=rn,
                                        insiders=len(a), detected_ref=int(a.sum()), detected=int(c.sum()),
                                        lost=int(((a == 1) & (c == 0)).sum()), gained=int(((a == 0) & (c == 1)).sum())))
                if not hits:
                    continue
                A = pd.concat([v[0] for v in hits.values()], axis=1).mean(axis=1)
                Cc = pd.concat([v[1] for v in hits.values()], axis=1).mean(axis=1)
                A, Cc = A.align(Cc, join="inner")
                diff = (Cc - A).to_numpy()
                bs = _boot_mean(diff)
                lo95_one = float(np.percentile(bs, 5))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pooled.append(dict(detector=det, budget=b, delta=dl, policy=pol, runs=len(hits), insiders=len(diff),
                                   recall_ref=float(A.mean()), recall=float(Cc.mean()), diff=float(diff.mean()),
                                   diff_insiders_per_run=float(diff.sum()), ci_lo=float(lo), ci_hi=float(hi),
                                   one_sided_lo=lo95_one, margin=margin, noninferior=bool(lo95_one > -margin)))
            log(f"I2 {det} budget {b} done")
    return pd.DataFrame(per_run), pd.DataFrame(pooled)


# =============================================================================
# I3  Rolling windows, windowed histories, deficit apportionment
# =============================================================================

def rolling_disparity(out, group_col, windows=(30, 90)):
    o = out.copy()
    o["ben"] = (o["y"] == 0).astype(int)
    o["fa"] = (o["alert"] & (o["y"] == 0)).astype(int)
    o["n"] = 1
    D = o.groupby(["day", group_col])[["n", "alert", "ben", "fa"]].sum().unstack(fill_value=0)
    res = {}
    for W in windows:
        R = D.rolling(W, min_periods=W).sum().dropna()
        if R.empty:
            res.update({f"roll{W}_{q}": float("nan") for q in ("dFAR_median", "dFAR_p90", "dFAR_max", "dAR_median", "dAR_max")})
            continue
        far = R["fa"] / R["ben"].where(R["ben"] > 0)
        ar = R["alert"] / R["n"].where(R["n"] > 0)
        dfar = far.max(axis=1) - far.min(axis=1)
        dar = ar.max(axis=1) - ar.min(axis=1)
        res.update({f"roll{W}_dFAR_median": float(dfar.median()), f"roll{W}_dFAR_p90": float(dfar.quantile(0.9)),
                    f"roll{W}_dFAR_max": float(dfar.max()), f"roll{W}_dAR_median": float(dar.median()),
                    f"roll{W}_dAR_max": float(dar.max())})
    # employee level, non-overlapping 30-day windows (the quantity of the private reports)
    d0 = o["day"].min()
    o["w"] = (o["day"] - d0).dt.days // 30
    ins = set(o.loc[o["y"] == 1, "user"])
    U = o[~o["user"].isin(ins)].groupby(["w", "user"]).agg(gr=(group_col, "first"), al=("alert", "any")).reset_index()
    roster = o[~o["user"].isin(ins)].groupby(group_col)["user"].nunique()
    vals = []
    for w, Dw in U.groupby("w"):
        cnt = Dw.groupby("gr")["al"].sum().reindex(roster.index).fillna(0)
        r = cnt / roster
        vals.append(r.max() - r.min())
    res["employee_month_dFAR_mean"] = float(np.mean(vals)) if vals else float("nan")
    res["employee_month_dFAR_max"] = float(np.max(vals)) if vals else float("nan")
    return res


I3_POLICIES = [
    ("B0 global", dict(history="global")),
    ("B8 daily parity, no memory", dict(history="none")),
    ("B9 FA*IR bands", dict(allocator="fairstar", history="alerts")),
    ("B10 dual prices (eta 10), label-free", dict(allocator="dual", history="alerts", eta=10.0)),
    ("B11 deficit apportionment, label-free", dict(allocator="deficit", history="alerts")),
    ("B11c deficit apportionment, outcome-aware", dict(allocator="deficit", history="cleared")),
    ("B4 label-free", dict(history="alerts")),
    ("B4 label-free, 30-day history", dict(history="alerts", hist_window=30)),
    ("B4 label-free, 90-day history", dict(history="alerts", hist_window=90)),
    ("B4c outcome-aware", dict(history="cleared")),
    ("B4c outcome-aware, 30-day history", dict(history="cleared", hist_window=30)),
    ("B4c outcome-aware, 90-day history", dict(history="cleared", hist_window=90)),
    ("B4c outcome-aware, discounted history (half-life 30 days)", dict(history="cleared", hist_decay=0.5 ** (1 / 30))),
]


def i3_windows(sc, budget=0.01, delta=0.002, group_col="g_functional_unit"):
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in I3_POLICIES:
            out = simulate_h(sc[det], budget, delta, group_col, **kw)
            m = metrics(out, group_col, detector=det, policy=pol)
            m.update(rolling_disparity(out, group_col))
            rows.append(m)
        log(f"I3 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I4  Uncertainty and individual concentration
# =============================================================================

def _far_by_group(ben, group_col):
    G = ben.groupby(group_col)
    return G["fa"].sum() / G["n"].sum()


def i4_uncertainty(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", n_boot=2000, n_random=20):
    pols = [("B0 global", dict(history="global")), ("B4 label-free", dict(history="alerts")),
            ("B4c outcome-aware", dict(history="cleared"))]
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in pols:
            out = simulate_h(sc[det], budget, delta, group_col, **kw)
            ben = out[out["y"] == 0]
            E = ben.groupby("user").agg(gr=(group_col, "first"), fa=("alert", "sum"), n=("alert", "size")).reset_index()
            far = _far_by_group(E, "gr")
            dfar, rho = far.max() - far.min(), (far.max() / far.min() if far.min() > 0 else np.inf)
            rng = np.random.default_rng(0)
            groups = {lab: D for lab, D in E.groupby("gr")}
            labs = list(groups)
            imax, imin = labs.index(far.idxmax()), labs.index(far.idxmin())
            bd, br, bp = [], [], []
            for _ in range(n_boot):
                fa, nn = [], []
                for lab, D in groups.items():
                    j = rng.integers(0, len(D), len(D))
                    fa.append(D["fa"].to_numpy()[j].sum()); nn.append(D["n"].to_numpy()[j].sum())
                f = np.array(fa) / np.array(nn)
                bd.append(f.max() - f.min()); br.append(f.max() / f.min() if f.min() > 0 else np.inf)
                bp.append(f[imax] - f[imin])
            bd, br, bp = np.array(bd), np.array(br), np.array(bp)
            # most-flagged benign employees removed (never-malicious employees only)
            ins = set(out.loc[out["y"] == 1, "user"])
            Eb = E[~E["user"].isin(ins)]
            res = dict(detector=det, policy=pol, delta_FAR=dfar, ratio_FAR=rho,
                       dFAR_lo=float(np.percentile(bd, 2.5)), dFAR_hi=float(np.percentile(bd, 97.5)),
                       pair_lo=float(np.percentile(bp, 2.5)), pair_hi=float(np.percentile(bp, 97.5)),
                       rho_lo=float(np.percentile(br, 2.5, method="lower")),
                       rho_hi=float(np.percentile(br, 97.5, method="higher")),
                       rho_inf_share=float(np.isinf(br).mean()))
            for top in (1, 10):
                drop = set(Eb.sort_values("fa", ascending=False)["user"].iloc[:top])
                f2 = _far_by_group(E[~E["user"].isin(drop)], "gr")
                res[f"dFAR_without_top{top}"] = float(f2.max() - f2.min())
                res[f"rho_without_top{top}"] = float(f2.max() / f2.min()) if f2.min() > 0 else np.inf
            # share of the max-min gap due to the single most-flagged employee of the max group
            gmax = far.idxmax()
            Dg = E[E["gr"] == gmax]
            res["max_group_top_employee_share_of_rate"] = float(Dg["fa"].max() / max(Dg["fa"].sum(), 1))
            # Gini of false alerts per benign employee, actual and under random allocation
            F = out[~out["user"].isin(ins)].groupby("user")["alert"].sum()
            res["gini_false_alerts"] = fi.gini(F.values)
            gr = []
            T = out[["day", "user"]].reset_index(drop=True)
            dayidx = T.groupby("day").indices
            for r in range(n_random):
                rg = np.random.default_rng(100 + r)
                a = np.zeros(len(T), bool)
                for dd, ii in dayidx.items():
                    kk = max(1, int(round(budget * len(ii))))
                    a[rg.choice(ii, kk, replace=False)] = True
                Fr = pd.Series(a, index=T["user"]).groupby(level=0).sum()
                Fr = Fr[~Fr.index.isin(ins)]
                gr.append(fi.gini(Fr.values))
            res["gini_random_allocation"] = float(np.mean(gr))
            rows.append(res)
        log(f"I4 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I5  Insiders removed after their first confirmed alert
# =============================================================================

def i5_removal(sc, budget=0.01, delta=0.002, group_col="g_functional_unit"):
    pols = [("B0 global", dict(history="global")), ("B4 label-free", dict(history="alerts")),
            ("B4c outcome-aware", dict(history="cleared"))]
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in pols:
            for rm in (False, True):
                out = simulate_h(sc[det], budget, delta, group_col, remove_confirmed=rm, **kw)
                rows.append(metrics(out, group_col, detector=det, policy=pol, insiders_removed=rm,
                                    removed_rows=out.attrs["removed_rows"]))
        log(f"I5 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I6  Joint verdict errors, fallible re-checker, joint confidence
# =============================================================================

def cp_upper(x, m, alpha):
    """One-sided Clopper-Pearson upper bound at level 1 - alpha."""
    if m == 0:
        return 1.0
    if x >= m:
        return 1.0
    return float(beta_dist.ppf(1 - alpha, x + 1, m - x))


def group_stats(out, group_col):
    """Per group: active user-days, cleared alerts, labels of confirmed alerts."""
    st = {}
    g = out[group_col].to_numpy()
    a = out["alert"].to_numpy(); c = out["cleared"].to_numpy(); yy = out["y"].to_numpy()
    for lab in np.unique(g):
        sel = g == lab
        st[lab] = (int(sel.sum()), int((sel & a & c).sum()), yy[sel & a & ~c])
    ben = out[out["y"] == 0]
    far = ben.groupby(group_col)["alert"].mean()
    return st, float(far.max() - far.min())


def corollary_bound(stats, frac=0.1, recheck_err=0.0, alpha=0.05, bonferroni=True, pi_bar=0.05, rng=None):
    """Bound of Corollary 1 from one re-check sample of each group's confirmed alerts.
    The re-checker misses a false confirmation with probability recheck_err.
    Returns (bound, plain outcome-based audit)."""
    rng = rng or np.random.default_rng(0)
    a_g = alpha / len(stats) if bonferroni else alpha
    lo, up = [], []
    for lab, (n, cl, conf_y) in stats.items():
        C = len(conf_y)
        msz = int(math.ceil(frac * C))
        if msz > 0:
            samp = conf_y[rng.choice(C, msz, replace=False)]
            found = (samp == 0) & (rng.random(msz) >= recheck_err)
            q = cp_upper(int(found.sum()), msz, a_g)
        else:
            q = 0.0
        fbar = q * C / (cl + q * C) if (cl + q * C) > 0 else 0.0
        base = cl / n
        lo.append(base)
        up.append(base / ((1 - pi_bar) * (1 - fbar)) if fbar < 1 else np.inf)
    return max(up) - min(lo), max(lo) - min(lo)


def i6_joint_errors(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
                    detectors=("GBM", "LSTM", "IF"), seeds=(0, 1, 2), n_samples=20):
    base = next(iter(sc.values())).test
    users_per = base.groupby(group_col)["user"].nunique().sort_values()
    small = users_per.index[0]
    settings = []
    for pm in (0.0, 0.3):
        settings.append((pm, "--", 0.0, {}))
        for where in ("all groups", "smallest group"):
            for f in (0.1, 0.2, 0.3):
                kw = dict(p_fc=f) if where == "all groups" else dict(err_group=small, p_fc_g=f)
                settings.append((pm, where, f, kw))
    rows = []
    for det in [d for d in detectors if d in sc]:
        for pm, where, f, kw in settings:
            for s in seeds:
                out = simulate_h(sc[det], budget, delta, group_col, history="cleared", p_miss=pm, seed=s, **kw)
                stats, tru = group_stats(out, group_col)
                for frac in (0.1, 0.2):
                    for rerr in (0.0, 0.1, 0.2):
                        for bonf in (False, True):
                            rng = np.random.default_rng(1000 * s + 7)
                            res = [corollary_bound(stats, frac, rerr, bonferroni=bonf, rng=rng)
                                   for _ in range(n_samples)]
                            b = np.array([r[0] for r in res])
                            rows.append(dict(detector=det, missed=pm, where=where, fc_rate=f, seed=s, frac=frac,
                                             recheck_error=rerr, bonferroni=bonf, true_dFAR=tru,
                                             audit=res[0][1], bound_mean=float(b.mean()),
                                             coverage=float((tru <= b + 1e-12).mean())))
        log(f"I6 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I7  Timing attackers: many draws, paired with the global policy, randomized displacement
# =============================================================================

def _monthly_noisy(rec, eps, rng):
    """Published monthly noisy cumulative alert counts per group (roster-free version:
    counts of alerts and active user-days, Laplace noise on the alert counts)."""
    r = rec.copy()
    r["month"] = pd.to_datetime(r["day"]).dt.strftime("%Y-%m")
    M = r.groupby(["month", "group"])[["k", "n"]].sum().reset_index()
    if eps is not None:
        M["k"] = M["k"] + rng.laplace(0, 1 / eps, len(M))
    return M


def i7_attacks(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=("GBM", "LSTM"),
               det_draws=5, rand_draws=10, window=30, eps=1.0, n_boot=5000, policies=None, attackers_only=None,
               shifter=None, collect=None):
    """For each FAIR-IRS policy P, attackers choose when to act from P's unattacked run
    (allocation, colleagues' investigations, monthly reports, deficit estimate).
    The relocated episodes are scored under P AND under the global policy, so the
    allocation's effect is isolated from the effect of relocation itself."""
    policies = policies or [("B4 label-free", dict(history="alerts")),
                            ("B4c outcome-aware", dict(history="cleared")),
                            ("B4 label-free, randomized displacement", dict(history="alerts", displace="random")),
                            ("B4c outcome-aware, randomized displacement", dict(history="cleared", displace="random"))]
    rows, per_ins = [], []
    for det in [d for d in detectors if d in sc]:
        sd0 = sc[det]
        out_g0 = simulate_h(sd0, budget, 1.0, group_col, history="global")
        hit_g0 = insider_hits(out_g0)
        for pol, kw in policies:
            out0, rec = simulate_h(sd0, budget, delta, group_col, record=True, seed=0, **kw)
            hit0 = insider_hits(out0)
            r = rec.copy()
            r["cut"] = (r["K"] - r["k"]).clip(lower=0)
            r["cfrac"] = np.where(r["K"] > 0, r["cut"] / r["K"].clip(lower=1), 0.0)
            cut = r.set_index(["day", "group"])["cut"].to_dict()
            obs = {}
            for lab, D in r.sort_values("day").groupby("group"):
                a = D["k"].rolling(window, min_periods=1).sum().shift(1).fillna(0)
                n = D["n"].rolling(window, min_periods=1).sum().shift(1).fillna(1)
                for dd, v in zip(D["day"], (a / n).to_numpy()):
                    obs[(dd, lab)] = v
            rep = fr.report_targets(out0, group_col)

            def make_deficit(seed):
                """Estimated cumulative excess of the group's alert rate over the overall rate:
                noisy published monthly counts up to the last report, plus the colleagues'
                investigations observed since then (own group only)."""
                M = _monthly_noisy(rec, eps, np.random.default_rng(500 + seed))
                months = sorted(M["month"].unique())
                cumk = M.pivot(index="month", columns="group", values="k").fillna(0).cumsum()
                cumn = M.pivot(index="month", columns="group", values="n").fillna(0).cumsum()
                overall = cumk.sum(axis=1) / cumn.sum(axis=1)
                R = r.copy()
                R["month"] = pd.to_datetime(R["day"]).dt.strftime("%Y-%m")
                est = {}
                for lab, D in R.sort_values("day").groupby("group"):
                    D = D.copy()
                    D["prev"] = D["month"].map(lambda mth: months[months.index(mth) - 1] if months.index(mth) > 0 else None)
                    D["k_since"] = D.groupby("month")["k"].cumsum().shift(1).fillna(0)
                    D.loc[D["month"] != D["month"].shift(1), "k_since"] = 0
                    D["n_since"] = D.groupby("month")["n"].cumsum().shift(1).fillna(0)
                    D.loc[D["month"] != D["month"].shift(1), "n_since"] = 0
                    for dd, pm, ks, ns in zip(D["day"], D["prev"], D["k_since"], D["n_since"]):
                        if pm is None or pd.isna(pm) or lab not in cumk.columns:
                            est[(dd, lab)] = 0.0
                            continue
                        ck, cn = cumk.loc[pm, lab] + ks, cumn.loc[pm, lab] + ns
                        est[(dd, lab)] = ck / max(cn, 1) - overall.loc[pm]
                return est

            def oracle(u, rows_, T, L, rng):
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([cut.get((dd[j], gu[j]), 0) for j in range(len(rows_))], float)
                win = np.convolve(c, np.ones(L), "valid")
                return int(rng.choice(np.flatnonzero(win == win.max())))

            def observing(u, rows_, T, L, rng):
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([obs.get((dd[j], gu[j]), 0) for j in range(len(rows_) - L + 1)], float)
                return int(rng.choice(np.flatnonzero(c == c.max())))

            def reads_reports(u, rows_, T, L, rng):
                gu = T.loc[rows_[0], group_col]
                months = T.loc[rows_, "day"].dt.strftime("%Y-%m").to_numpy()
                if gu not in rep.columns:
                    return int(rng.integers(0, len(rows_) - L + 1))
                ex = rep[gu].dropna()
                allm = sorted(rep.index)
                for mth in list(ex.sort_values(ascending=False).index):
                    nxt = allm.index(mth) + 1
                    if nxt >= len(allm):
                        continue
                    pos = np.flatnonzero(months == allm[nxt])
                    pos = pos[pos <= len(rows_) - L]
                    if len(pos):
                        return int(rng.choice(pos[:5]))
                return int(rng.integers(0, len(rows_) - L + 1))

            def rand(u, rows_, T, L, rng):
                return int(rng.integers(0, len(rows_) - L + 1))

            attackers = [("knows allocation", oracle, det_draws), ("observes colleagues", observing, det_draws),
                         ("reads reports", reads_reports, det_draws), ("estimates deficit", None, det_draws),
                         ("random", rand, rand_draws)]
            if attackers_only:
                attackers = [a for a in attackers if a[0] in attackers_only]
            for name, fn, nd in attackers:
                for dr in range(nd):
                    if name == "estimates deficit":
                        est = make_deficit(dr)

                        def fn(u, rows_, T, L, rng, est=est):
                            gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                            c = np.array([est.get((dd[j], gu[j]), 0) for j in range(len(rows_) - L + 1)], float)
                            return int(rng.choice(np.flatnonzero(c == c.max())))
                    sd = fr.shift_to(sd0, group_col, fn, dr) if shifter is None else shifter(det, sd0, group_col, fn, dr)
                    outP, recP = simulate_h(sd, budget, delta, group_col, record=True, seed=dr + 1, **kw)
                    outG = simulate_h(sd, budget, 1.0, group_col, history="global")
                    hP, hG = insider_hits(outP), insider_hits(outG)
                    # per insider: allocation effect under attack minus allocation effect without attack
                    D = pd.concat([hit0.rename("P0"), hit_g0.rename("G0"), hP.rename("PA"), hG.rename("GA")], axis=1).fillna(0)
                    dd_ = ((D["PA"] - D["GA"]) - (D["P0"] - D["G0"])).to_numpy()
                    if collect is not None:
                        collect.extend(dict(detector=det, policy=pol, attack=name, draw=dr, user=u, d=float(v))
                                       for u, v in zip(D.index, dd_))
                    bs = _boot_mean(dd_, n_boot, seed=dr)
                    row = dict(detector=det, policy=pol, attack=name, draw=dr, insiders=len(D),
                               none_P=int(D["P0"].sum()), none_G=int(D["G0"].sum()),
                               attacked_P=int(D["PA"].sum()), attacked_G=int(D["GA"].sum()),
                               excess_loss=float(-dd_.sum()), excess_lo=float(-np.percentile(bs, 97.5) * len(D)),
                               excess_hi=float(-np.percentile(bs, 2.5) * len(D)),
                               delta_FAR=fi.policy_metrics(outP, group_col)["delta_FAR"])
                    if kw.get("displace") == "random":
                        # bound of the randomized-displacement proposition, per insider:
                        # P(miss | caught by global) <= prod over the insider's global-top-k days of c_{g,t}
                        Rp = recP.copy()
                        Rp["cfrac"] = np.where(Rp["K"] > 0, (Rp["K"] - Rp["k"]).clip(lower=0) / Rp["K"].clip(lower=1), 0.0)
                        cf = Rp.set_index(["day", "group"])["cfrac"].to_dict()
                        topG = outG[(outG["y"] == 1) & outG["alert"]]
                        bnd = topG.groupby("user").apply(lambda Q: float(np.prod([cf.get((a, b), 0.0) for a, b in zip(Q["day"], Q[group_col])])))
                        caughtG = bnd.index
                        row["bound_expected_misses"] = float(bnd.sum())
                        row["observed_misses_among_caught_by_global"] = int((hP.reindex(caughtG).fillna(0) == 0).sum())
                        row["max_cut_fraction"] = float(Rp["cfrac"].max())
                        row["mean_cut_fraction_on_cut_days"] = float(Rp.loc[Rp["cfrac"] > 0, "cfrac"].mean()) if (Rp["cfrac"] > 0).any() else 0.0
                    rows.append(row)
            log(f"I7 {det} {pol} done")
    return pd.DataFrame(rows)


def i7_summary(df):
    g = df.groupby(["detector", "policy", "attack"], sort=False)
    s = g[["none_P", "none_G", "attacked_P", "attacked_G", "excess_loss", "excess_lo", "excess_hi"]].mean()
    s["draws"] = g.size()
    s["excess_loss_min"] = g["excess_loss"].min()
    s["excess_loss_max"] = g["excess_loss"].max()
    for c in ("bound_expected_misses", "observed_misses_among_caught_by_global", "max_cut_fraction"):
        if c in df.columns:
            s[c] = g[c].mean()
    return s.reset_index()


# =============================================================================
# I8  Tolerance chosen on a validation period
# =============================================================================

def i8_validation(sc, budget=0.01, group_col="g_functional_unit", deltas=(0.001, 0.002, 0.005, 0.01),
                  margin=0.05):
    """The first half of the test period is the validation period. For each detector
    and mode, delta* = smallest tolerance whose validation-period recall difference
    with the global policy has a one-sided 95% lower bound above -margin. Reported on
    the second half (alerts and histories run over the whole period)."""
    rows = []
    for det in [d for d in DETS if d in sc]:
        sd = sc[det]
        dd = np.sort(sd.test["day"].unique())
        mid = pd.Timestamp(dd[len(dd) // 2])
        outG = simulate_h(sd, budget, 1.0, group_col, history="global")
        for mode in ("alerts", "cleared"):
            res = {}
            for dl in deltas:
                out = simulate_h(sd, budget, dl, group_col, history=mode)
                vG, vP = insider_hits(outG[outG["day"] < mid]), insider_hits(out[out["day"] < mid])
                a, c = vG.align(vP, join="inner")
                lo = float(np.percentile(_boot_mean((c - a).to_numpy().astype(float)), 5)) if len(a) else 0.0
                tG, tP = outG[outG["day"] >= mid], out[out["day"] >= mid]
                mT = fi.policy_metrics(tP, group_col)
                mG = fi.policy_metrics(tG, group_col)
                res[dl] = dict(val_insiders=len(a), val_lost=int(((a == 1) & (c == 0)).sum()),
                               val_gained=int(((a == 0) & (c == 1)).sum()), val_lo=lo,
                               test_detected=mT["insiders_detected"], test_detected_global=mG["insiders_detected"],
                               test_insiders=mT["insiders_test"], test_dFAR=mT["delta_FAR"], test_dFAR_global=mG["delta_FAR"])
            ok = [dl for dl in deltas if res[dl]["val_lo"] > -margin]
            chosen = min(ok) if ok else max(deltas)
            for dl in deltas:
                rows.append(dict(detector=det, mode=mode, delta=dl, chosen=dl == chosen, **res[dl]))
        log(f"I8 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I9  Score ties and group counts
# =============================================================================

def i9_ties(sc, budget=0.01):
    rows = []
    for det in [d for d in DETS if d in sc]:
        T = sc[det].test
        tied_top, n_top, sat = 0, 0, 0
        sc_top = []
        for dd, D in T.groupby("day"):
            k = max(1, int(round(budget * len(D))))
            s = np.sort(D["score"].to_numpy())[::-1]
            thr = s[k - 1]
            tied_top += int((s[:k] == thr).sum() > 1 or (k < len(s) and s[k] == thr))
            n_top += 1
            sat += int((s[:k] >= 0.999).sum())
            sc_top.extend(s[:k].tolist())
        sc_top = np.array(sc_top)
        rows.append(dict(detector=det, days=n_top, days_with_tie_at_threshold=tied_top,
                         share_alerts_score_ge_0999=sat / max(len(sc_top), 1),
                         alerted_score_min=float(sc_top.min()), alerted_score_median=float(np.median(sc_top)),
                         distinct_scores_share=float(T["score"].nunique() / len(T))))
    return pd.DataFrame(rows)


def i9_groups(ds):
    rows = []
    te = ds.df[~ds.df["train"]]
    for gc in ds.group_cols:
        u = te.groupby(gc)["user"].nunique()
        rows.append(dict(grouping=gc, groups=len(u), smallest=int(u.min()), largest=int(u.max()),
                         median=float(u.median()), sizes=";".join(f"{k}:{v}" for k, v in u.sort_values().items())))
    return pd.DataFrame(rows)


# =============================================================================
# I10 Shadow month and prior warm start
# =============================================================================

def i10_warm(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", burn_in=30):
    rows = []
    for det in [d for d in DETS if d in sc]:
        for mode in ("alerts", "cleared"):
            for warm in (None, "shadow", "prior"):
                out = simulate_h(sc[det], budget, delta, group_col, history=mode, warm=warm)
                t = g3.trajectory(out, group_col)
                target = "dCR" if mode == "cleared" else "dAR"
                after = t.iloc[burn_in:]
                m = metrics(out, group_col, detector=det, mode=mode, warm=str(warm))
                m.update(fallback_list=";".join(map(str, out.attrs["fallback_list"])),
                         max_constrained_after_burn_in=float(after[target].max()),
                         share_days_above_delta_after_burn_in=float((after[target] > delta + 1e-12).mean()),
                         max_constrained_first_30=float(t[target].iloc[:burn_in].max()))
                rows.append(m)
        log(f"I10 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# I11 Power of a test on private monthly reports
# =============================================================================

def i11_power(ds, sc_det, budget=0.01, delta=0.002, group_col="g_functional_unit",
              eps_list=(0.5, 1.0, 2.0, 4.0, None), gaps=(0.01, 0.02, 0.05, 0.1, 0.2), n_sim=4000,
              groupings=("g_functional_unit", "g_department", "g_role")):
    """Monthly roster-based reports (share of each group's roster alerted at least once).
    For the smallest and the largest group of each grouping, the oversight body tests
    H0: equal rates against a gap, with the Laplace noise and the binomial sampling of
    employees. Base rate: the median monthly employee-level alert rate under global top-k.
    Returns the power at alpha = 0.05 (two-sided) and the minimal detectable gap at 80% power."""
    rows = []
    te = ds.df[~ds.df["train"]]
    for gc in groupings:
        out = simulate_h(sc_det, budget, 1.0, gc, history="global")
        o = out.copy()
        o["w"] = (o["day"] - o["day"].min()).dt.days // 30
        U = o.groupby(["w", "user"])["alert"].any().reset_index()
        p0 = float(U.groupby("w")["alert"].mean().median())
        u = te.groupby(gc)["user"].nunique()
        n1, n2 = int(u.min()), int(u.max())
        rng = np.random.default_rng(0)
        for eps in eps_list:
            def stat(gap, size):
                x1 = rng.binomial(n1, min(1, p0 + gap), size)
                x2 = rng.binomial(n2, p0, size)
                if eps is not None:
                    x1 = x1 + rng.laplace(0, 1 / eps, size)
                    x2 = x2 + rng.laplace(0, 1 / eps, size)
                return x1 / n1 - x2 / n2
            null = np.abs(stat(0.0, n_sim))
            crit = np.quantile(null, 0.95)
            pw = {f"power_gap_{gp}": float((np.abs(stat(gp, n_sim)) > crit).mean()) for gp in gaps}
            grid = np.linspace(0.0, 0.6, 121)
            mdg = next((float(gp) for gp in grid if (np.abs(stat(gp, 2000)) > crit).mean() >= 0.8), float("nan"))
            rows.append(dict(grouping=gc, smallest=n1, largest=n2, base_rate=p0,
                             eps=("none" if eps is None else eps), min_detectable_gap_80=mdg, **pw))
    return pd.DataFrame(rows)
