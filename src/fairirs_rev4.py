"""Additional experiments (part D) for the FAIR-IRS paper.

H1  Paired statistical tests and effect sizes for detection differences
H2  Online dual-price baseline (an established longitudinal allocation strategy)
H3  Fairness-security frontier: tolerance x protected top x attacker knowledge
H4  Sweep of non-random false-confirmation rates, with per-group counts for the audit bound
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import fairirs as fi
import fairirs_rev as fr
import fairirs_rev3 as g3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def insider_hits(out):
    """Per insider: 1 if at least one malicious test day was alerted."""
    mal = out[out["y"] == 1]
    return mal.groupby("user")["alert"].any().astype(int)


# =============================================================================
# H1  Paired tests
# =============================================================================

def paired(a, b, n_boot=10000, seed=0):
    """a, b: per-insider detection (aligned Series). McNemar exact test and paired
    bootstrap CI of the recall difference b - a."""
    a, b = a.align(b, join="inner")
    a, b = a.to_numpy(), b.to_numpy()
    n = len(a)
    lost = int(((a == 1) & (b == 0)).sum())
    gained = int(((a == 0) & (b == 1)).sum())
    p = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained > 0 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(insiders=n, recall_ref=a.mean(), recall_policy=b.mean(), diff=b.mean() - a.mean(),
                diff_lo=lo, diff_hi=hi, lost=lost, gained=gained, mcnemar_p=p,
                diff_insiders=int(b.sum() - a.sum()))


H1_POLICIES = [("B4 label-free", dict(history="alerts")),
               ("B4c outcome-aware", dict(history="cleared")),
               ("B5 label-free + scheduling", dict(history="alerts", schedule=True, lam=0.01)),
               ("B5c outcome-aware + scheduling", dict(history="cleared", schedule=True, lam=0.01)),
               ("B4 label-free, floor 2", dict(history="alerts", floor=2.0))]


def h1_paired(sc, seeds_sc=None, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    rows = []
    runs = [("main", sc)] + [(f"seed {s}", v) for s, v in (seeds_sc or {}).items()]
    for run, scs in runs:
        for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in scs]:
            sd = scs[det]
            ref = insider_hits(g3.simulate_g(sd, budget, delta, group_col, history="global"))
            pols = H1_POLICIES if run == "main" else H1_POLICIES[:2]
            for pol, kw in pols:
                if kw.get("schedule") and sd.phi is None:
                    continue
                hit = insider_hits(g3.simulate_g(sd, budget, delta, group_col, **kw))
                r = paired(ref, hit)
                r.update(release=release, run=run, detector=det, policy=pol)
                rows.append(r)
        log(f"H1 {run} done")
    return pd.DataFrame(rows)


def h1_pooled(df):
    """Pooled over the seed runs of each detector and policy (descriptive: the seeds
    share the same insiders)."""
    d = df[df["run"].str.startswith("seed")]
    rows = []
    for (det, pol), D in d.groupby(["detector", "policy"]):
        lost, gained = int(D["lost"].sum()), int(D["gained"].sum())
        p = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
        rows.append(dict(detector=det, policy=pol, runs=len(D), mean_diff=D["diff"].mean(),
                         min_diff=D["diff"].min(), max_diff=D["diff"].max(),
                         lost_total=lost, gained_total=gained, pooled_sign_p=p))
    if not d.empty:
        for pol, D in d.groupby("policy"):
            lost, gained = int(D["lost"].sum()), int(D["gained"].sum())
            p = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
            rows.append(dict(detector="all", policy=pol, runs=len(D), mean_diff=D["diff"].mean(),
                             min_diff=D["diff"].min(), max_diff=D["diff"].max(),
                             lost_total=lost, gained_total=gained, pooled_sign_p=p))
    return pd.DataFrame(rows)


# =============================================================================
# H2  Online dual-price baseline
# =============================================================================

def h2_dual(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
            etas=(0.1, 0.3, 1.0, 3.0, 10.0)):
    rows = []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        sd = sc[det]
        out = g3.simulate_g(sd, budget, delta, group_col, history="global")
        rows.append(g3.metrics(out, group_col, detector=det, policy="B0 global", eta=0.0, mode="--"))
        for h, name in (("alerts", "label-free"), ("cleared", "outcome-aware")):
            out = g3.simulate_g(sd, budget, delta, group_col, history=h)
            rows.append(g3.metrics(out, group_col, detector=det, policy=f"FAIR-IRS {name}", eta=0.0, mode=name))
            for dm in ("prop", "ascent"):
                for eta in etas:
                    out = g3.simulate_g(sd, budget, delta, group_col, history=h, allocator="dual",
                                        eta=eta, dual_mode=dm)
                    rows.append(g3.metrics(out, group_col, detector=det,
                                           policy=f"B10 dual prices ({'proportional' if dm == 'prop' else 'dual ascent'})",
                                           eta=eta, mode=name))
        log(f"H2 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# H3  Fairness-security frontier
# =============================================================================

def h3_frontier(sc, budget=0.01, group_col="g_functional_unit", detectors=("GBM", "LSTM"),
                deltas=(0.002, 0.005, 0.01), protects=(0.0, 0.25, 0.5, 0.75), seeds=(0, 1, 2), window=30):
    rows = []
    base = next(iter(sc.values())).test
    gsize = base.groupby(group_col)["user"].nunique()
    small_groups = set(gsize[gsize < gsize.median()].index)
    for det in detectors:
        if det not in sc:
            continue
        sd0 = sc[det]
        configs = [("global", 1.0, 0.0)] + [(h, dl, pr) for h in ("alerts", "cleared") for dl in deltas for pr in protects]
        for h, dl, pr in configs:
            kw = dict(history="global" if h == "global" else h, protect=pr)
            out0, rec = g3.simulate_g(sd0, budget, dl, group_col, record=True, **kw)
            r = rec.copy()
            r["cut"] = (r["K"] - r["k"]).clip(lower=0)
            cut = r.set_index(["day", "group"])["cut"].to_dict()
            obs = {}
            for lab, D in r.sort_values("day").groupby("group"):
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
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([obs.get((dd[j], gu[j]), 0) for j in range(len(rows_) - L + 1)], float)
                return int(rng.choice(np.flatnonzero(c == c.max())))

            def rand(u, rows_, T, L, rng):
                return int(rng.integers(0, len(rows_) - L + 1))

            scen = [("none", 0, sd0), ("knows allocation", 0, fr.shift_to(sd0, group_col, oracle, 0)),
                    ("observes colleagues", 0, fr.shift_to(sd0, group_col, observing, 0))]
            scen += [("random", s, fr.shift_to(sd0, group_col, rand, s)) for s in seeds]
            for attack, s, sd in scen:
                out = out0 if attack == "none" else g3.simulate_g(sd, budget, dl, group_col, **kw)
                hits = insider_hits(out)
                ins_group = out[out["y"] == 1].groupby("user")[group_col].first()
                small = hits[ins_group.reindex(hits.index).isin(small_groups)]
                m = fi.policy_metrics(out, group_col)
                rows.append(dict(detector=det, mode=h, delta=dl, protect=pr, attack=attack, seed=s,
                                 detected=int(hits.sum()), insiders=len(hits),
                                 detected_small_groups=int(small.sum()), insiders_small_groups=len(small),
                                 delta_FAR=m["delta_FAR"], ratio_FAR=m["ratio_FAR"],
                                 fallback_days=m["fallback_days"], alerts_used=m["alerts_used"]))
        log(f"H3 {det} done")
    return pd.DataFrame(rows)


def h3_summary(df):
    keys = ["detector", "mode", "delta", "protect"]
    rows = []
    for k, D in df.groupby(keys, sort=False):
        base = D[D["attack"] == "none"].iloc[0]
        att = D[D["attack"] != "none"]
        per = att.groupby("attack")["detected"].mean()
        rows.append(dict(zip(keys, k), detected=int(base["detected"]), delta_FAR=base["delta_FAR"],
                         fallback_days=int(base["fallback_days"]),
                         knows_allocation=per.get("knows allocation", np.nan),
                         observes_colleagues=per.get("observes colleagues", np.nan),
                         random=per.get("random", np.nan),
                         worst_case_loss=base["detected"] - per.min(),
                         average_loss=base["detected"] - per.mean(),
                         worst_case_loss_small_groups=base["detected_small_groups"]
                         - att.groupby("attack")["detected_small_groups"].mean().min()))
    return pd.DataFrame(rows)


# =============================================================================
# H4  False-confirmation sweep
# =============================================================================

def group_counts(out, group_col):
    o = out.assign(fa=out["alert"] & (out["y"] == 0), ta=out["alert"] & (out["y"] == 1),
                   conf=out["alert"] & ~out["cleared"], wrong=out["alert"] & ~out["cleared"] & (out["y"] == 0),
                   ben=out["y"] == 0)
    G = o.groupby(group_col).agg(n=("y", "size"), benign=("ben", "sum"), alerts=("alert", "sum"),
                                 false_alerts=("fa", "sum"), true_alerts=("ta", "sum"),
                                 confirmed=("conf", "sum"), wrongly_confirmed=("wrong", "sum"))
    return G.reset_index().rename(columns={group_col: "group"})


def h4_false_confirmation(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
                          detectors=("GBM", "LSTM", "IF"), rates=(0.0, 0.05, 0.1, 0.2, 0.3), seeds=(0, 1, 2)):
    rows, counts = [], []
    base = next(iter(sc.values())).test
    users_per = base.groupby(group_col)["user"].nunique().sort_values()
    small, large = users_per.index[0], users_per.index[-1]
    for det in detectors:
        if det not in sc:
            continue
        sd = sc[det]
        for where, kwf in (("all groups", lambda f: dict(p_fc=f)),
                           ("smallest group", lambda f: dict(err_group=small, p_fc_g=f)),
                           ("largest group", lambda f: dict(err_group=large, p_fc_g=f))):
            for f in rates:
                for s in seeds:
                    for pol, h in (("B4 label-free", "alerts"), ("B4c outcome-aware", "cleared")):
                        out = g3.simulate_g(sd, budget, delta, group_col, history=h, seed=s, **kwf(f))
                        m = fi.policy_metrics(out, group_col)
                        rows.append(dict(detector=det, where=where, rate=f, seed=s, policy=pol,
                                         delta_FAR=m["delta_FAR"], delta_FAR_hat=m["delta_FAR_hat"],
                                         ratio_FAR=m["ratio_FAR"], detected=m["insiders_detected"]))
                        c = group_counts(out, group_col)
                        c["detector"], c["where"], c["rate"], c["seed"], c["policy"] = det, where, f, s, pol
                        counts.append(c)
                    if f == 0:
                        break
        log(f"H4 {det} done")
    return pd.DataFrame(rows), pd.concat(counts, ignore_index=True)
