"""CERT r5.2, permutation tests and parity rules for the FAIR-IRS paper.

J1  Permutation test of group disparity (group labels shuffled across employees)
J2  Scheduling and cooldown on top of the global policy; memoryless quota with randomized
    rounding; deficit apportionment with histories (the new default)
J3  Detectors on per-employee baselines
J4  Timing attacks with delayed and missing verdicts; deficit apportionment under attack
J5  Discount farming against the outcome-aware mode and deficit apportionment
J6  Non-inferiority of the new default and of the new baselines
J7  Score calibration: share of malicious user-days by score band
J8  CERT r5.2: burden, parity rules, non-inferiority, attacks, permutation test
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev3 as g3
import fairirs_rev5 as h5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


DETS = ("IF", "AE", "LR", "GBM", "LSTM")
HALF30 = 0.5 ** (1 / 30)

DEFAULT = ("B11c deficit apportionment (outcome-aware)", dict(allocator="deficit", history="cleared"))
DEFAULT_DISC = ("B11c deficit apportionment, discounted", dict(allocator="deficit", history="cleared", hist_decay=HALF30))


# =============================================================================
# J1  Permutation test
# =============================================================================

def _user_table(out, group_col):
    ben = out[out["y"] == 0]
    E = ben.groupby("user").agg(gr=(group_col, lambda s: s.mode().iloc[0]), fa=("alert", "sum"),
                                n=("alert", "size")).reset_index()
    return E


def _dfar(fa, n, codes, G):
    FA = np.bincount(codes, weights=fa, minlength=G)
    N = np.bincount(codes, weights=n, minlength=G)
    r = FA / np.maximum(N, 1)
    return r.max() - r.min()


def j1_permutation(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", n_perm=5000, release="r4.2"):
    """Null: group labels exchangeable across employees (group sizes kept). Statistic: the
    max-min difference of group false-alert rates, which is biased upward under the null too;
    the permutation distribution accounts for that bias."""
    pols = [("B0 global", dict(history="global")), ("B4c exact rule", dict(history="cleared")), DEFAULT]
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in pols:
            out = h5.simulate_h(sc[det], budget, delta, group_col, **kw)
            E = _user_table(out, group_col)
            ins = set(out.loc[out["y"] == 1, "user"])
            for drop in (0, 10):
                Ed = E
                if drop:
                    top = set(E[~E["user"].isin(ins)].sort_values("fa", ascending=False)["user"].iloc[:drop])
                    Ed = E[~E["user"].isin(top)]
                codes, labs = pd.factorize(Ed["gr"])
                G = len(labs)
                fa, n = Ed["fa"].to_numpy(float), Ed["n"].to_numpy(float)
                obs = _dfar(fa, n, codes, G)
                rng = np.random.default_rng(0)
                perm = np.array([_dfar(fa, n, rng.permutation(codes), G) for _ in range(n_perm)])
                rows.append(dict(release=release, detector=det, policy=pol, dropped_top=drop, delta_FAR=obs,
                                 null_median=float(np.median(perm)), null_p95=float(np.quantile(perm, 0.95)),
                                 p_value=float((1 + (perm >= obs - 1e-15).sum()) / (1 + n_perm))))
        log(f"J1 {release} {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# J2  Baselines: scheduling and cooldown on global, randomized rounding, default
# =============================================================================

J2_POLICIES = [
    ("B0 global", dict(history="global")),
    ("B0 + scheduling", dict(history="global", schedule=True)),
    ("B0 + cooldown 7 days", dict(history="global", cooldown=7)),
    ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
    ("B1r quota, randomized rounding", dict(allocator="randround", history="alerts")),
    ("B4c exact rule", dict(history="cleared")),
    ("B4c exact rule, discounted", dict(history="cleared", hist_decay=HALF30)),
    ("B4c + scheduling", dict(history="cleared", schedule=True)),
    ("B4c + cooldown 30 days", dict(history="cleared", cooldown=30)),
    DEFAULT,
    DEFAULT_DISC,
    ("B11c + scheduling", dict(allocator="deficit", history="cleared", schedule=True)),
    ("B11c + cooldown 30 days", dict(allocator="deficit", history="cleared", cooldown=30)),
]


def j2_baselines(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2", seeds=(0, 1, 2)):
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in J2_POLICIES:
            if kw.get("schedule") and sc[det].phi is None:
                continue
            ss = seeds if kw.get("allocator") == "randround" else (0,)
            for s in ss:
                out = h5.simulate_h(sc[det], budget, delta, group_col, seed=s, **kw)
                m = h5.metrics(out, group_col, release=release, detector=det, policy=pol, seed=s)
                m.update(h5.rolling_disparity(out, group_col))
                rows.append(m)
        log(f"J2 {release} {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# J3  Detectors on per-employee baselines
# =============================================================================

def run_detector_user(ds, det):
    """Same pipeline as fairirs.run_detector, but every user-day is expressed as its
    deviation from the employee's own mean over the training period."""
    F = ds.df
    tr_mask = F["train"].to_numpy()
    y = F["y"].to_numpy()
    tr_fit = tr_mask & (y == 0) if not det.supervised else tr_mask
    ft = fi.FeatureTransform("raw").fit(F[tr_fit])
    X0 = ft.transform(F)
    base = fi.user_baseline(F, X0, tr_mask & (y == 0))
    X = fi.explanations(F, X0, base).astype(np.float32)
    det.fit(X[tr_fit], y[tr_fit])
    raw = det.score(X[~tr_mask])
    raw_tr = det.score(X[tr_mask & (y == 0)])
    ref = np.sort(raw_tr)
    test = F[~tr_mask].copy().reset_index(drop=True)
    test["raw_score"] = raw
    test["score"] = fi.to_quantile_ref(raw, ref)
    return fi.ScoredData(det.name + "-user", test, X[~tr_mask].copy(), det, ft, ref, None)


def j3_user_baselines(ds, budget=0.01, delta=0.002, group_col="g_functional_unit", seed=0):
    rows, scs = [], {}
    for det in (fi.IFDetector(seed), fi.AEDetector(seed), fi.GBMDetector(seed)):
        try:
            sd = run_detector_user(ds, det)
        except Exception as ex:
            log(f"[skip] {det.name}-user: {type(ex).__name__}: {ex}")
            continue
        scs[sd.detector] = sd
        for pol, kw in [("B0 global", dict(history="global")), ("B0 + scheduling", dict(history="global", schedule=True)),
                        ("B4c exact rule", dict(history="cleared")), DEFAULT]:
            out = h5.simulate_h(sd, budget, delta, group_col, **kw)
            m = h5.metrics(out, group_col, detector=sd.detector, policy=pol)
            m["exposure_bias"] = fi.exposure_bias(sd.test, group_col=group_col)
            rows.append(m)
        log(f"J3 {sd.detector} done")
    return pd.DataFrame(rows), scs


# =============================================================================
# J4  Timing attacks with delayed verdicts; deficit apportionment under attack
# =============================================================================

J4_POLICIES = [
    ("B4c, verdicts after 7 days", dict(history="cleared", delay=7)),
    ("B4c, verdicts after 30 days", dict(history="cleared", delay=30)),
    ("B4c, 20% verdicts missing + 7-day delay", dict(history="cleared", delay=7, missing=0.2)),
    DEFAULT,
    DEFAULT_DISC,
    ("B11c, verdicts after 7 days", dict(allocator="deficit", history="cleared", delay=7)),
]


def j4_attacks(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=("GBM", "LSTM"),
               det_draws=5, rand_draws=5, release="r4.2"):
    df = h5.i7_attacks(sc, budget, delta, group_col, detectors=detectors, det_draws=det_draws,
                       rand_draws=rand_draws, policies=J4_POLICIES)
    df["release"] = release
    return df


# =============================================================================
# J5  Discount farming against outcome-aware accounting
# =============================================================================

def j5_farming(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=("GBM", "IF"),
               alphas=(0.5, 1.0)):
    pols = [("B0 global", dict(history="global")), ("B4 label-free", dict(history="alerts")),
            ("B4c exact rule", dict(history="cleared")), DEFAULT]
    rows = []
    for det in [d for d in detectors if d in sc]:
        sd = sc[det]
        for alpha in alphas:
            sdf, farmers = fi.discount_farming(sd, n_farm=3, alpha=alpha)
            if not farmers:
                continue
            for pol, kw in pols:
                for sched in (False, True):
                    res = {}
                    for lab, s_ in (("no farming", sd), ("farming", sdf)):
                        out = h5.simulate_h(s_, budget, delta, group_col, schedule=sched, **kw)
                        sub = out[out["user"].isin(farmers) & (out["y"] == 1)]
                        res[lab] = sub.groupby("user")["alert"].any()
                        if lab == "farming":
                            fa = out[out["user"].isin(farmers) & (out["y"] == 0)]
                            farm_alerts = int(fa["alert"].sum())
                    a, b = res["no farming"].align(res["farming"], join="inner")
                    rows.append(dict(detector=det, alpha=alpha, policy=pol, schedule=sched, farmers=len(farmers),
                                     detected_no_farming=int(a.sum()), detected_farming=int(b.sum()),
                                     lost=int((a & ~b).sum()), gained=int((~a & b).sum()),
                                     benign_alerts_on_farmers=farm_alerts))
        log(f"J5 {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# J7  Score calibration
# =============================================================================

def j7_calibration(sc, budget=0.01, release="r4.2"):
    bands = [(0.0, 0.95), (0.95, 0.99), (0.99, 0.995), (0.995, 0.999), (0.999, 1.01)]
    rows = []
    for det in [d for d in DETS if d in sc]:
        T = sc[det].test
        s, y = T["score"].to_numpy(), T["y"].to_numpy()
        for lo, hi in bands:
            m = (s >= lo) & (s < hi)
            rows.append(dict(release=release, detector=det, band=f"[{lo:g}, {min(hi, 1):g}{']' if hi > 1 else ')'}",
                             user_days=int(m.sum()), malicious=int(y[m].sum()),
                             share_malicious=float(y[m].mean()) if m.any() else float("nan")))
        # does the score mass of the day's alerts track the number of true alerts?
        out = h5.simulate_h(sc[det], budget, 1.0, "g_functional_unit", history="global")
        D = out[out["alert"]].groupby("day").agg(mass=("score", "sum"), hits=("y", "sum"))
        from scipy.stats import spearmanr
        rho = spearmanr(D["mass"], D["hits"]).correlation if D["hits"].std() > 0 else float("nan")
        rows.append(dict(release=release, detector=det, band="daily score mass vs. true alerts (Spearman)",
                         user_days=len(D), malicious=int(D["hits"].sum()), share_malicious=float(rho)))
    return pd.DataFrame(rows)


# =============================================================================
# J8  r5.2 summary of burden and parity rules
# =============================================================================

def j8_rules(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r5.2"):
    pols = [("B0 global", dict(history="global")), ("B8 daily parity, no memory", dict(history="none")),
            ("B1r quota, randomized rounding", dict(allocator="randround", history="alerts")),
            ("B4 exact rule, label-free", dict(history="alerts")), ("B4c exact rule", dict(history="cleared")),
            ("B4c exact rule, discounted", dict(history="cleared", hist_decay=HALF30)), DEFAULT, DEFAULT_DISC,
            ("B0 + scheduling", dict(history="global", schedule=True)),
            ("B11c + scheduling", dict(allocator="deficit", history="cleared", schedule=True))]
    rows = []
    for det in [d for d in DETS if d in sc]:
        for pol, kw in pols:
            if kw.get("schedule") and sc[det].phi is None:
                continue
            out = h5.simulate_h(sc[det], budget, delta, group_col, **kw)
            m = h5.metrics(out, group_col, release=release, detector=det, policy=pol)
            m.update(h5.rolling_disparity(out, group_col))
            if pol == "B0 global":
                m["exposure_bias"] = fi.exposure_bias(sc[det].test, group_col=group_col)
            rows.append(m)
        log(f"J8 {release} {det} done")
    return pd.DataFrame(rows)
