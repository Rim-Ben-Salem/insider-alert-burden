"""Experiments for the repositioned paper (history-dependent alerting as fairness tool and attack surface).

K1  Discount farming against a cooldown after clearance (with and without the override), on r4.2 and r5.2
K2  Cooldown and scheduling on r5.2 (burden, detection, non-inferiority over seeds)
K3  Timing attacks against parity combined with a cooldown
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev5 as h5
import fairirs_rev6 as j6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


SUPPRESSION = [
    ("B0 global", dict(history="global")),
    ("B0 + scheduling", dict(history="global", schedule=True)),
    ("B0 + cooldown 7 days", dict(history="global", cooldown=7)),
    ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
    ("B0 + cooldown 30 days, no override", dict(history="global", cooldown=30, tau_ov=1.01)),
    ("B0 + cooldown 30 days, override 0.995", dict(history="global", cooldown=30, tau_ov=0.995)),
    ("B11c + cooldown 30 days", dict(allocator="deficit", history="cleared", cooldown=30)),
]


def k1_farming(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=("GBM", "AE", "IF", "LR"),
               alphas=(0.5, 1.0), n_farm=(3,), release="r4.2"):
    """Farming insiders make the benign days before their attack resemble it, hoping that these
    days are alerted and cleared, so that the attack falls under the cooldown or the discount."""
    rows = []
    for det in [d for d in detectors if d in sc]:
        sd = sc[det]
        for nf in n_farm:
            for alpha in alphas:
                sdf, farmers = fi.discount_farming(sd, n_farm=nf, alpha=alpha)
                if not farmers:
                    continue
                for pol, kw in SUPPRESSION:
                    res, inv = {}, 0
                    for lab, s_ in (("no farming", sd), ("farming", sdf)):
                        out = h5.simulate_h(s_, budget, delta, group_col, **kw)
                        sub = out[out["user"].isin(farmers) & (out["y"] == 1)]
                        res[lab] = sub.groupby("user")["alert"].any()
                        if lab == "farming":
                            inv = int(out[out["user"].isin(farmers) & (out["y"] == 0)]["alert"].sum())
                    a, b = res["no farming"].align(res["farming"], join="inner")
                    rows.append(dict(release=release, detector=det, n_farm=nf, alpha=alpha, policy=pol,
                                     farmers=len(farmers), detected_no_farming=int(a.sum()),
                                     detected_farming=int(b.sum()), lost=int((a & ~b).sum()),
                                     gained=int((~a & b).sum()), farmer_benign_alerts=inv))
            log(f"K1 {release} {det} done")
    return pd.DataFrame(rows)


def k3_attacks(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=("GBM", "LSTM"), release="r4.2"):
    pols = [("B11c + cooldown 30 days", dict(allocator="deficit", history="cleared", cooldown=30)),
            ("B4c + cooldown 30 days", dict(history="cleared", cooldown=30))]
    df = h5.i7_attacks(sc, budget, delta, group_col, detectors=detectors, det_draws=3, rand_draws=3, policies=pols)
    df["release"] = release
    return df
