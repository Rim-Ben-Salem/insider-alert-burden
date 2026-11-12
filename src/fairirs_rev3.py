"""Additional experiments (part C) for the FAIR-IRS paper.

G1  Realistic investigation feedback: delays, missing, revised and group-dependent verdicts (M2)
G2  Total harm: distinct employees investigated and convex repeat costs (M4)
G3  Sequential attackers and interactions with floors and randomized tolerance (M5)
G4  Stronger fairness-aware baselines: daily parity without memory, FA*IR-style bands (M7)
G5  Disparity over time and during fallback (M3)
G6  Synthetic stress tests independent of CERT (M1)
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import binom
import fairirs as fi
import fairirs_rev as fr


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================
# General day-by-day simulator
# =============================================================================

def _fairstar(sc, gids, k, alpha=0.1):
    """Multi-group adaptation of FA*IR to alerting: each group receives between
    the alpha/2 and 1-alpha/2 binomial quantiles of its proportional share of the
    k alerts; within the bands, alerts follow the global score order."""
    labs, inv = np.unique(gids, return_inverse=True)
    n = np.bincount(inv, minlength=len(labs))
    p = n / n.sum()
    lo = np.minimum(binom.ppf(alpha / 2, k, p).astype(int), n)
    hi = np.minimum(np.maximum(binom.ppf(1 - alpha / 2, k, p).astype(int), lo), n)
    if lo.sum() > k:                                    # keep the minima feasible
        lo = np.floor(lo * k / lo.sum()).astype(int)
    order = np.argsort(-sc, kind="stable")
    m = np.zeros(len(sc), bool)
    cnt = np.zeros(len(labs), int)
    for g in range(len(labs)):                          # minimum representation
        if lo[g] > 0:
            idx = order[inv[order] == g][:lo[g]]
            m[idx] = True
            cnt[g] += len(idx)
    total = int(m.sum())
    for i in order:                                     # fill within the caps
        if total >= k:
            break
        if not m[i] and cnt[inv[i]] < hi[inv[i]]:
            m[i] = True
            cnt[inv[i]] += 1
            total += 1
    if total < k:                                       # caps too tight: global order
        for i in order:
            if total >= k:
                break
            if not m[i]:
                m[i] = True
                total += 1
    return m


def simulate_g(sd, budget=0.01, delta=0.002, group_col="g_functional_unit", history="alerts",
               allocator="fair", floor=None, delta_jitter=False,
               delay=0, missing=0.0, revise=0.0, revise_days=30,
               p_miss=0.0, p_fc=0.0, err_group=None, p_miss_g=0.0, p_fc_g=0.0,
               schedule=False, lam=0.01, gamma=0.8, tau_ov=0.999, m_cap=3, horizon=30,
               record=False, seed=0, protect=0.0, eta=0.0, dual_mode="prop"):
    """history: 'global' (top-k), 'alerts' (label-free), 'cleared' (outcome-aware),
    'none' (parity enforced on each day alone, no memory).
    Investigation feedback (affects the outcome-aware history and the final verdicts):
      delay        verdicts become known `delay` days after the alert
      missing      share of verdicts that never arrive (alert stays counted as cleared)
      revise       share of malicious alerts first cleared, then confirmed after revise_days
      p_miss       share of malicious alerts wrongly cleared for good
      p_fc         share of benign alerts wrongly confirmed as malicious
      err_group    group with its own error rates p_miss_g, p_fc_g
    Until its verdict is known, an alert counts as cleared (provisional clearance).
    protect      the global top ceil(protect*k) are always alerted; parity applies to the rest
    allocator='dual': online dual prices; each day the top k of score - mu_g are alerted,
                 with mu_g = eta*(r_g - r) ('prop') or mu_g += eta*(r_g - r) ('ascent'),
                 r_g the group's cumulative (alert or cleared-alert) rate and r the overall rate."""
    rng = np.random.default_rng(seed)
    T = sd.test
    days = T["day"].to_numpy()
    users = T["user"].to_numpy()
    y = T["y"].to_numpy()
    s = T["score"].to_numpy().astype(np.float64)
    g = T[group_col].to_numpy()
    alert = np.zeros(len(T), bool)
    cleared = np.zeros(len(T), bool)
    cumA, cumC, cumConf = {}, {}, {}
    pending = []                                        # (reveal day index, group)
    fb_days, budget_total, disp, rec = [], 0, [], []
    store, n_disc = {}, 0
    mu = {}
    if schedule:
        phin = sd.phi / (np.linalg.norm(sd.phi, axis=1, keepdims=True) + 1e-9)
    uniq_days, starts = np.unique(days, return_index=True)
    bounds = list(starts) + [len(T)]
    for di, d in enumerate(uniq_days):
        idx = np.arange(bounds[di], bounds[di + 1])
        # verdicts revealed by today
        keep = []
        for (rd, lab) in pending:
            if rd <= di:
                cumA[lab] = cumA.get(lab, 0) - 1
                cumConf[lab] = cumConf.get(lab, 0) + 1
            else:
                keep.append((rd, lab))
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
        k = max(1, int(round(budget * len(idx))))
        dl = delta * rng.uniform(0, 2) if delta_jitter else delta
        if allocator == "dual" and history in ("alerts", "cleared") and dl < 1:
            adj = sc - np.array([mu.get(lab, 0.0) for lab in g[idx]])
            m = np.zeros(len(idx), bool)
            m[np.argsort(-adj, kind="stable")[:k]] = True
            fb = 0
        elif protect > 0 and allocator == "fair" and history in ("alerts", "cleared", "none") and dl < 1:
            order = np.argsort(-sc, kind="stable")
            P = order[:min(k, int(np.ceil(protect * k)))]
            rest = np.setdiff1d(np.arange(len(idx)), P)
            if history == "none":
                A0, C0 = {}, {}
            elif history == "cleared":
                A0 = dict(cumA); C0 = {lab: cumC.get(lab, 0) - cumConf.get(lab, 0) for lab in cumC}
            else:
                A0, C0 = dict(cumA), dict(cumC)
            for j in P:
                lab = g[idx][j]
                A0[lab] = A0.get(lab, 0) + 1
                C0[lab] = C0.get(lab, 0) + 1
            m = np.zeros(len(idx), bool)
            m[P] = True
            fb = 0
            if k - len(P) > 0 and len(rest):
                mr, fb = fi.fair_allocate_cum(sc[rest], g[idx][rest], k - len(P), dl, A0, C0)
                m[rest[mr]] = True
        elif history == "global" or dl >= 1:
            m = np.zeros(len(idx), bool)
            m[np.argsort(-sc, kind="stable")[:k]] = True
            fb = 0
        elif allocator == "fairstar":
            m, fb = _fairstar(sc, g[idx], k), 0
        elif history == "none":
            m, fb = fi.fair_allocate_cum(sc, g[idx], k, dl, {}, {})
        elif history == "cleared":
            Ceff = {lab: cumC.get(lab, 0) - cumConf.get(lab, 0) for lab in cumC}
            m, fb = fi.fair_allocate_cum(sc, g[idx], k, dl, cumA, Ceff)
        else:
            m, fb = fi.fair_allocate_cum(sc, g[idx], k, dl, cumA, cumC)
        if fb:
            fb_days.append(di)
        rank = np.empty(len(idx), int)
        rank[np.argsort(-sc, kind="stable")] = np.arange(1, len(idx) + 1)
        if floor is not None:
            m = m & (rank <= int(np.ceil(floor * k)))
        disp.extend((rank[m & (rank > k)] / k).tolist())
        budget_total += k
        if record:
            top = rank <= k
            for lab in np.unique(g[idx]):
                sel = g[idx] == lab
                rec.append((d, lab, int(top[sel].sum()), int(m[sel].sum()), int(sel.sum())))
        a_idx = idx[m]
        alert[a_idx] = True
        for lab, cnt in zip(*np.unique(g[idx], return_counts=True)):
            cumC[lab] = cumC.get(lab, 0) + cnt
        for i in a_idx:
            lab = g[i]
            pm = p_miss_g if (err_group is not None and lab == err_group) else p_miss
            pf = p_fc_g if (err_group is not None and lab == err_group) else p_fc
            if y[i] == 1:
                final_clear = rng.random() < pm
            else:
                final_clear = not (rng.random() < pf)
            cleared[i] = final_clear
            if history == "alerts" or history == "none" or history == "global":
                cumA[lab] = cumA.get(lab, 0) + 1
                if schedule and final_clear:
                    store.setdefault(users[i], []).append((d, phin[i]))
                continue
            cumA[lab] = cumA.get(lab, 0) + 1            # provisional clearance
            if schedule and final_clear:
                store.setdefault(users[i], []).append((d, phin[i]))
            if final_clear or rng.random() < missing:
                continue
            if y[i] == 1 and rng.random() < revise:
                pending.append((di + max(revise_days, delay), lab))
            else:
                pending.append((di + delay, lab))
        if allocator == "dual":
            if history == "cleared":
                den = {lab: cumC[lab] - cumConf.get(lab, 0) for lab in cumC}
            else:
                den = dict(cumC)
            rates = {lab: cumA.get(lab, 0) / max(den[lab], 1) for lab in den}
            rbar = sum(cumA.get(lab, 0) for lab in den) / max(sum(den.values()), 1)
            for lab, r in rates.items():
                if dual_mode == "ascent":
                    mu[lab] = mu.get(lab, 0.0) + eta * (r - rbar)
                else:
                    mu[lab] = eta * (r - rbar)
    out = T[["user", "day", "y", "score", "exposure", "scenario", group_col]].copy()
    out["alert"], out["cleared"] = alert, cleared
    out.attrs.update(fallback_days=len(fb_days), fallback_list=fb_days, discounts=n_disc,
                     n_days=len(uniq_days), disp_ranks=np.array(disp), budget_total=budget_total)
    if record:
        return out, pd.DataFrame(rec, columns=["day", "group", "K", "k", "n"])
    return out


def harm(out):
    """Burden on benign employees beyond disparity."""
    ins = set(out.loc[out["y"] == 1, "user"])
    ben = out[~out["user"].isin(ins)]
    F = ben.groupby("user")["alert"].sum()
    Fc = ben.groupby("user")["cleared"].sum()
    return dict(distinct_alerted=int((F > 0).sum()), employees_F_ge2=int((F >= 2).sum()),
                employees_F_ge5=int((F >= 5).sum()), employees_F_ge10=int((F >= 10).sum()),
                false_alerts=int(F.sum()), cost_squared=float((F ** 2).sum()),
                cost_repeat=float((F * (F + 1) / 2).sum()), max_F=int(F.max()) if len(F) else 0,
                benign_employees=int(len(F)), wrongly_confirmed=int((ben["alert"] & ~ben["cleared"]).sum()))


def metrics(out, group_col, **extra):
    m = fi.policy_metrics(out, group_col)
    m.update(harm(out))
    m.update(extra)
    return m


# =============================================================================
# G1  Investigation feedback (M2)
# =============================================================================

def g1_feedback(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
                detectors=("GBM", "LSTM", "IF"), seeds=(0, 1, 2)):
    rows = []
    base = next(iter(sc.values())).test
    users_per = base.groupby(group_col)["user"].nunique().sort_values()
    small, large = users_per.index[0], users_per.index[-1]
    variants = [("correct, immediate", {})]
    variants += [(f"delay {d} days", dict(delay=d)) for d in (3, 7, 14, 30)]
    variants += [(f"missing {int(100*m)}%", dict(missing=m)) for m in (0.2, 0.5)]
    variants += [("revised after 30 days (30%)", dict(revise=0.3, revise_days=30)),
                 ("missed 30%, all groups", dict(p_miss=0.3)),
                 ("false confirmation 5%, all groups", dict(p_fc=0.05))]
    for tg, lab in ((small, "smallest"), (large, "largest")):
        variants += [(f"missed 30%, {lab} group only", dict(err_group=tg, p_miss_g=0.3)),
                     (f"false confirmation 5%, {lab} group only", dict(err_group=tg, p_fc_g=0.05)),
                     (f"false confirmation 20%, {lab} group only", dict(err_group=tg, p_fc_g=0.2))]
    variants += [("delay 7 days + missing 20% + missed 30%", dict(delay=7, missing=0.2, p_miss=0.3))]
    for det in detectors:
        if det not in sc:
            continue
        sd = sc[det]
        for pol, h in (("B0 global", "global"), ("B4 label-free", "alerts")):
            out = simulate_g(sd, budget, delta, group_col, history=h)
            rows.append(metrics(out, group_col, detector=det, policy=pol, feedback="(not used)", seed=0,
                                far_small=float(out[(out.y == 0) & (out[group_col] == small)]["alert"].mean()),
                                far_large=float(out[(out.y == 0) & (out[group_col] == large)]["alert"].mean())))
        for name, kw in variants:
            for s in seeds:
                out = simulate_g(sd, budget, delta, group_col, history="cleared", seed=s, **kw)
                rows.append(metrics(out, group_col, detector=det, policy="B4c outcome-aware", feedback=name, seed=s,
                                    far_small=float(out[(out.y == 0) & (out[group_col] == small)]["alert"].mean()),
                                    far_large=float(out[(out.y == 0) & (out[group_col] == large)]["alert"].mean())))
                if not kw:
                    break
        log(f"G1 {det} done")
    df = pd.DataFrame(rows)
    df.attrs["groups"] = (str(small), str(large))
    return df


# =============================================================================
# G2  Total harm (M4)
# =============================================================================

def g2_harm(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", lams=(0.0, 0.01, 0.02, 0.05)):
    rows = []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        sd = sc[det]
        out = simulate_g(sd, budget, delta, group_col, history="global")
        rows.append(metrics(out, group_col, detector=det, policy="B0 global", lam=0.0))
        for h, name in (("alerts", "label-free"), ("cleared", "outcome-aware")):
            for lam in lams:
                out = simulate_g(sd, budget, delta, group_col, history=h, schedule=lam > 0, lam=max(lam, 1e-9))
                rows.append(metrics(out, group_col, detector=det, policy=name, lam=lam))
        log(f"G2 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# G3  Sequential attackers (M5)
# =============================================================================

def g3_attacks(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
               detectors=("GBM", "LSTM"), seeds=(0, 1, 2, 3, 4), window=30):
    policies = [("B0 global", dict(history="global")),
                ("B4 label-free", dict(history="alerts")),
                ("B4c outcome-aware", dict(history="cleared")),
                ("B4 label-free, floor 2", dict(history="alerts", floor=2.0)),
                ("B4c outcome-aware, floor 2", dict(history="cleared", floor=2.0)),
                ("B4c outcome-aware, delta 0.005", dict(history="cleared", delta=0.005)),
                ("B4c outcome-aware, randomized delta", dict(history="cleared", delta_jitter=True))]
    rows = []
    for det in detectors:
        if det not in sc:
            continue
        sd0 = sc[det]
        for pol, kw in policies:
            kw = dict(kw)
            dl = kw.pop("delta", delta)
            out0, rec = simulate_g(sd0, budget, dl, group_col, record=True, **kw)
            r = rec.copy()
            r["cut"] = (r["K"] - r["k"]).clip(lower=0)
            cut = r.set_index(["day", "group"])["cut"].to_dict()
            # what an employee can observe: colleagues investigated in the past `window` days
            r = r.sort_values("day")
            obs = {}
            for lab, D in r.groupby("group"):
                D = D.sort_values("day")
                a = D["k"].rolling(window, min_periods=1).sum().shift(1).fillna(0)
                n = D["n"].rolling(window, min_periods=1).sum().shift(1).fillna(1)
                for dd, v in zip(D["day"], (a / n).to_numpy()):
                    obs[(dd, lab)] = v

            def oracle(u, rows_, T, L, rng):
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([cut.get((dd[j], gu[j]), 0) for j in range(len(rows_))], float)
                win = np.convolve(c, np.ones(L), "valid")
                return int(rng.choice(np.flatnonzero(win == win.max())))

            def observing(u, rows_, T, L, rng):
                # acts when colleagues were investigated most often in the past window
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([obs.get((dd[j], gu[j]), 0) for j in range(len(rows_) - L + 1)], float)
                return int(rng.choice(np.flatnonzero(c == c.max())))

            def rand(u, rows_, T, L, rng):
                return int(rng.integers(0, len(rows_) - L + 1))

            scen = [("none", 0, sd0), ("knows allocation", 0, fr.shift_to(sd0, group_col, oracle, 0)),
                    ("observes colleagues", 0, fr.shift_to(sd0, group_col, observing, 0))]
            scen += [("random", s, fr.shift_to(sd0, group_col, rand, s)) for s in seeds]
            for attack, s, sd in scen:
                out = out0 if attack == "none" else simulate_g(sd, budget, dl, group_col, **kw)
                ins = set(sd.test.loc[sd.test["y"] == 1, "user"])
                rr = fr.recall_of(out, ins)
                m = fi.policy_metrics(out, group_col)
                rows.append(dict(detector=det, policy=pol, attack=attack, seed=s, detected=rr[1],
                                 insiders=rr[2], delta_FAR=m["delta_FAR"], ratio_FAR=m["ratio_FAR"],
                                 alerts_used=m["alerts_used"], fallback_days=m["fallback_days"]))
        log(f"G3 {det} done")
    return pd.DataFrame(rows)


def g3_summary(df):
    g = df.groupby(["detector", "policy", "attack"], sort=False)
    s = g[["detected", "delta_FAR", "alerts_used", "fallback_days"]].mean()
    s["detected_sd"] = g["detected"].std().fillna(0)
    return s.reset_index()


# =============================================================================
# G4  Stronger baselines (M7)
# =============================================================================

def g4_baselines(sc, budget=0.01, delta=0.002, group_col="g_functional_unit"):
    pols = [("B0 global", dict(history="global")),
            ("daily parity, no memory (label-free)", dict(history="none")),
            ("FA*IR-style bands, alpha 0.1", dict(allocator="fairstar", history="alerts")),
            ("B4 label-free (cumulative)", dict(history="alerts")),
            ("B4c outcome-aware (cumulative)", dict(history="cleared"))]
    rows = []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        for pol, kw in pols:
            out = simulate_g(sc[det], budget, delta, group_col, **kw)
            rows.append(metrics(out, group_col, detector=det, policy=pol))
        log(f"G4 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# G5  Disparity over time (M3)
# =============================================================================

def trajectory(out, group_col):
    o = out.copy()
    o["ben"] = (o["y"] == 0).astype(int)
    o["fa"] = (o["alert"] & (o["y"] == 0)).astype(int)
    o["cl"] = (o["alert"] & o["cleared"]).astype(int)
    o["conf"] = (o["alert"] & ~o["cleared"]).astype(int)
    o["n"] = 1
    D = o.groupby(["day", group_col])[["n", "alert", "ben", "fa", "cl", "conf"]].sum().groupby(level=1).cumsum()
    D = D.reset_index()
    D["AR"] = D["alert"] / D["n"]
    D["FAR"] = D["fa"] / D["ben"].clip(lower=1)
    D["CR"] = D["cl"] / (D["n"] - D["conf"]).clip(lower=1)
    t = D.groupby("day").agg(dAR=("AR", lambda v: v.max() - v.min()), dFAR=("FAR", lambda v: v.max() - v.min()),
                             dCR=("CR", lambda v: v.max() - v.min()))
    return t.reset_index()


def g5_over_time(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", burn_in=30):
    pols = [("B0 global", dict(history="global")), ("B4 label-free", dict(history="alerts")),
            ("B4c outcome-aware", dict(history="cleared")),
            ("B4 label-free, floor 2", dict(history="alerts", floor=2.0)),
            ("B4 label-free, floor 1", dict(history="alerts", floor=1.0)),
            ("B4c outcome-aware, floor 2", dict(history="cleared", floor=2.0)),
            ("B4c outcome-aware, floor 1", dict(history="cleared", floor=1.0))]
    rows, traj = [], []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        for pol, kw in pols:
            out = simulate_g(sc[det], budget, delta, group_col, **kw)
            t = trajectory(out, group_col)
            t["detector"], t["policy"] = det, pol
            traj.append(t)
            after = t.iloc[burn_in:]
            fbl = out.attrs["fallback_list"]
            fb_after = [i for i in fbl if i >= burn_in]
            target = "dCR" if kw.get("history") == "cleared" else "dAR"
            rows.append(dict(detector=det, policy=pol, constrained=target,
                             final=float(t[target].iloc[-1]), max_after_burn_in=float(after[target].max()),
                             share_days_above_delta=float((after[target] > delta + 1e-12).mean()),
                             max_excess=float(max(0.0, after[target].max() - delta)),
                             fallback_days=len(fbl), fallback_days_after_burn_in=len(fb_after),
                             final_true_dFAR=float(t["dFAR"].iloc[-1]),
                             max_true_dFAR_after_burn_in=float(after["dFAR"].max()),
                             alerts_used=out["alert"].sum() / out.attrs["budget_total"]))
        log(f"G5 {det} done")
    return pd.DataFrame(rows), pd.concat(traj, ignore_index=True)


# =============================================================================
# G6  Synthetic stress tests (M1)
# =============================================================================

def synth_data(sizes=(100, 100, 100, 100, 100), act_gap=0.0, risk_ratio=1.0, persistence=0.5,
               base_rate=0.004, n_days=250, episode=10, boost=2.5, seed=0):
    """Fully synthetic scores, independent of CERT.
    Benign score  = mu_g + u_i + e_it, with mu_g spread over [0, act_gap] across groups,
                    user effect u_i ~ N(0, persistence^2) and daily noise e_it ~ N(0, 1).
    Insiders      : prevalence proportional to risk_ratio^(g/(G-1)) (risk increases with
                    group activity when risk_ratio > 1), mean base_rate of user-days; each
                    insider has one episode of `episode` days with a score boost `boost`."""
    rng = np.random.default_rng(seed)
    G = len(sizes)
    mu = np.linspace(0, act_gap, G)
    w = risk_ratio ** (np.arange(G) / max(G - 1, 1))
    users, groups, ueff = [], [], []
    for gi, n in enumerate(sizes):
        for j in range(n):
            users.append(f"g{gi}_u{j}"); groups.append(f"G{gi}")
            ueff.append(mu[gi] + rng.normal(0, persistence))
    U = len(users)
    n_ins_total = max(1, int(round(base_rate * U * n_days / episode)))
    probs = np.repeat(w, sizes); probs = probs / probs.sum()
    ins_idx = rng.choice(U, size=min(n_ins_total, U), replace=False, p=probs)
    start = {i: rng.integers(n_days // 2, n_days - episode) for i in ins_idx}
    days = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows_u = np.repeat(np.arange(U), n_days)
    rows_d = np.tile(np.arange(n_days), U)
    score = np.array(ueff)[rows_u] + rng.normal(0, 1, U * n_days)
    y = np.zeros(U * n_days, int)
    for i, st in start.items():
        sel = (rows_u == i) & (rows_d >= st) & (rows_d < st + episode)
        y[sel] = 1
        score[sel] += boost
    T = pd.DataFrame({"user": np.array(users)[rows_u], "day": days[rows_d], "y": y, "score": score,
                      "exposure": np.array(ueff)[rows_u], "scenario": np.where(y == 1, "synthetic", ""),
                      "g_group": np.array(groups)[rows_u]})
    T = T[T["day"] >= days[n_days // 2]].sort_values(["day", "user"]).reset_index(drop=True)
    return fi.ScoredData("synthetic", T, None, None, None, None, None)


def g6_stress(budget=0.01, delta=0.002, seeds=(0, 1, 2)):
    settings = []
    for a in (0.0, 0.25, 0.5, 1.0):
        settings.append(("activity gap", a, dict(act_gap=a)))
    for name, sz in (("balanced", (100,) * 5), ("skewed", (20, 40, 80, 160, 200)), ("one tiny group", (10, 120, 120, 125, 125))):
        settings.append(("group sizes", name, dict(sizes=sz, act_gap=0.5)))
    for rr in (1.0, 3.0, 10.0):
        settings.append(("risk ratio", rr, dict(risk_ratio=rr, act_gap=0.5)))
    for p in (0.0, 0.5, 1.0):
        settings.append(("persistence", p, dict(persistence=p, act_gap=0.5)))
    pols = [("B0 global", dict(history="global")), ("daily parity, no memory", dict(history="none")),
            ("FA*IR-style bands", dict(allocator="fairstar", history="alerts")),
            ("B4 label-free", dict(history="alerts")), ("B4c outcome-aware", dict(history="cleared"))]
    rows = []
    for factor, level, kw in settings:
        for s in seeds:
            sd = synth_data(seed=s, **kw)
            for pol, pk in pols:
                out = simulate_g(sd, budget, delta, "g_group", **pk)
                m = metrics(out, "g_group")
                m.update(factor=factor, level=str(level), seed=s, policy=pol)
                rows.append(m)
        log(f"G6 {factor}={level} done")
    return pd.DataFrame(rows)


def summarize(df, keys, cols):
    return df.groupby(keys, sort=False)[cols].mean().reset_index()
