"""Experiments for revision 12.

P1  Cooldown with late verdicts, corrected: the cooldown starts when the verdict arrives and is
    not postponed by later alerts whose verdicts are still pending (the revision 10 runs let
    each new alert overwrite the pending clearance, which disabled the cooldown); burden and
    detection over seeds
P2  Farming against the cooldown with late verdicts, over seeds, every detector
P3  Permutation tests (range and size-weighted statistics) after removing the most-flagged
    benign employees
"""
import numpy as np
import pandas as pd
import fairirs_rev5 as h5
import fairirs_rev10 as n10
import fairirs_rev11 as o11

log = o11.log

P_VARIANTS = [("cooldown 30 d, override 0.999", dict(history="global", cooldown=30)),
              ("verdicts 7 days late", dict(history="global", cooldown=30, cd_delay=7)),
              ("verdicts 14 days late", dict(history="global", cooldown=30, cd_delay=14)),
              ("no override, verdicts 7 days late", dict(history="global", cooldown=30, cd_delay=7, tau_ov=1.01)),
              ("global (no cooldown)", dict(history="global"))]


def p1_delay(runs, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    rows = []
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            for pol, kw in P_VARIANTS:
                out = h5.simulate_h(sd, budget, delta, group_col, **kw)
                rows.append(o11.row(out, group_col, release=release, detector=det, seed=seed, policy=pol, budget=budget))
            log(f"P1 {release} {det} seed {seed} done")
    return pd.DataFrame(rows)


def p2_farming(runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    saved = n10.N3_VARIANTS
    n10.N3_VARIANTS = P_VARIANTS
    try:
        return n10.n3_farming(runs, rescorers, budget, delta, group_col, release=release)
    finally:
        n10.N3_VARIANTS = saved


def p3_permutation(runs, budget=0.01, group_col="g_functional_unit", removes=(0, 5, 10, 20), release="r4.2", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for det, by_seed in runs.items():
        sd = by_seed.get(0) or next(iter(by_seed.values()))
        out = h5.simulate_h(sd, budget, 1.0, group_col, history="global")
        ins = set(out.loc[out["y"] == 1, "user"])
        b = out[(out["y"] == 0) & ~out["user"].isin(ins)]
        E = b.groupby("user").agg(F=("alert", "sum"), n=("alert", "size"), g=(group_col, lambda s: s.mode().iloc[0]))
        for r in removes:
            Er = E.drop(E["F"].sort_values(ascending=False).index[:r]) if r else E
            labels = sorted(Er["g"].unique())
            gi = Er["g"].map({l: i for i, l in enumerate(labels)}).to_numpy()
            p, obs = n10.permutation_p(Er["F"].to_numpy(float), Er["n"].to_numpy(float), gi, labels, 2000, rng)
            rows.append(dict(release=release, detector=det, removed=r, p_range=float(p[0]), p_weighted=float(p[1]),
                             stat_range=obs[0], stat_weighted=obs[1]))
        log(f"P3 {release} {det} done")
    return pd.DataFrame(rows)
