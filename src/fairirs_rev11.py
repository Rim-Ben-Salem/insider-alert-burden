"""Experiments for the second referee report (revision 11).

O1  Baselines over seeds: confirmed insiders removed, a verdict-independent repeat cap, and the
    time from an insider's first malicious day to their first alert
O2  Cooldown sensitivity over seeds: override threshold, cooldown length and budget, with burden,
    detection, detection delay and farming exposure
O3  Farming as a function of the probability that a probe alert is cleared: the attacker's net
    outcome (insiders caught without farming minus insiders caught with it, a confirmed probe
    exposing the insider), every detector including the LSTM
O4  Online timing attackers whose signal never triggers keep their original timing; insiders
    relocated to their own original start (effect of relocation alone); unilateral deviations
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev5 as h5
import fairirs_rev6 as j6
import fairirs_rev8 as l8
import fairirs_rev9 as m9
import fairirs_rev10 as n10


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def delay_stats(out):
    """Days from each detected insider's first malicious day to their first alerted malicious day."""
    mal = out[out["y"] == 1]
    if mal.empty:
        return dict(delay_median=np.nan, delay_mean=np.nan)
    first = mal.groupby("user")["day"].min()
    hit = mal[mal["alert"]].groupby("user")["day"].min()
    dl = ((hit - first.reindex(hit.index)) / pd.Timedelta(days=1)).astype(float)
    return dict(delay_median=float(dl.median()) if len(dl) else np.nan, delay_mean=float(dl.mean()) if len(dl) else np.nan)


def row(out, group_col, **extra):
    r = n10.burden_row(out, group_col, **extra)
    r.update(delay_stats(out))
    r["removed_rows"] = out.attrs.get("removed_rows", 0)
    return r


CD = dict(history="global", cooldown=30)
O1_POLICIES = [("global", dict(history="global")),
               ("global, confirmed insiders removed", dict(history="global", remove_confirmed=True)),
               ("cooldown 30 d, override 0.999", CD),
               ("cooldown 30 d, confirmed insiders removed", dict(CD, remove_confirmed=True)),
               ("repeat cap: 1 alert per 30 days, override 0.999", dict(history="global", rc_cap=1, rc_window=30)),
               ("repeat cap: 3 alerts per 30 days, override 0.999", dict(history="global", rc_cap=3, rc_window=30)),
               ("repeat cap: 1 alert per 30 days, confirmed insiders removed", dict(history="global", rc_cap=1, rc_window=30, remove_confirmed=True)),
               ("deficit apportionment (outcome-aware)", j6.DEFAULT[1]),
               ("deficit apportionment, confirmed insiders removed", dict(j6.DEFAULT[1], remove_confirmed=True))]


def o1_baselines(runs, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    rows = []
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            for pol, kw in O1_POLICIES:
                out = h5.simulate_h(sd, budget, delta, group_col, **kw)
                rows.append(row(out, group_col, release=release, detector=det, seed=seed, policy=pol, budget=budget))
            log(f"O1 {release} {det} seed {seed} done")
    return pd.DataFrame(rows)


def o2_variants(budget=0.01):
    b1, b2 = budget, 2 * budget
    V = [("global", b1, dict(history="global")), ("global", b2, dict(history="global"))]
    for tau, lab in ((0.99, "0.99"), (0.995, "0.995"), (0.999, "0.999"), (0.9995, "0.9995"), (1.01, "none")):
        for b in (b1, b2):
            V.append((f"cooldown 30 d, override {lab}", b, dict(history="global", cooldown=30, tau_ov=tau)))
    for C in (3, 7, 14, 60):
        V.append((f"cooldown {C} d, override 0.999", b1, dict(history="global", cooldown=C)))
    for cap in (1, 3):
        V.append((f"repeat cap: {cap} per 30 days, override 0.999", b1, dict(history="global", rc_cap=cap, rc_window=30)))
    return V


O3_VARIANTS = [("global", dict(history="global")),
               ("cooldown 30 d, override 0.999", CD),
               ("cooldown 30 d, no override", dict(CD, tau_ov=1.01)),
               ("repeat cap: 1 per 30 days, override 0.999", dict(history="global", rc_cap=1, rc_window=30))]
O3_CONFIGS = (("own", 3), ("generic", 3))
P_CLEAR = (1.0, 0.75, 0.5, 0.25)


def _farm_eval(sd, sdf, farmers, probes, budget, delta, group_col, kw, p_clear, a0, seed=0):
    Tf = sdf.test.copy()
    Tf.loc[probes, "fc"] = 1.0 - p_clear
    sdx = fi.ScoredData(sdf.detector, Tf, sdf.phi, sdf.model, sdf.transform, sdf.raw_ref, sdf.train_ref)
    of = h5.simulate_h(sdx, budget, delta, group_col, fc_col="fc", seed=seed, **kw)
    det0 = a0.reindex(farmers).fillna(False).astype(bool)
    a1 = of[(of["y"] == 1) & of["user"].isin(farmers)].groupby("user")["alert"].any()
    det1 = a1.reindex(farmers).fillna(False).astype(bool)
    pr = of.loc[probes]
    exp_ = pr[pr["alert"] & ~pr["cleared"]].groupby("user").size()
    exposed = pd.Series(False, index=farmers)
    exposed[exposed.index.isin(exp_.index)] = True
    caught1 = det1 | exposed
    return dict(farmers=len(farmers), detected_no_farming=int(det0.sum()), caught_with_farming=int(caught1.sum()),
                net_attacker_gain=int(det0.sum() - caught1.sum()), evaded=int((det0 & ~caught1).sum()),
                exposed_by_probe=int(exposed.sum()), newly_caught=int((~det0 & caught1).sum()),
                probes_alerted=int(pr["alert"].sum()))


def o23_farming(runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2",
                draws=3, max_seeds=3):
    """Returns (O2 sweep rows, O3 clearance rows)."""
    r2, r3 = [], []
    for det, by_seed in runs.items():
        for seed, sd in list(by_seed.items())[:max_seeds]:
            if seed not in rescorers.get(det, {}):
                continue
            R = rescorers[det][seed]
            try:
                farms = {cfg: n10.farm_gap(sd, R, cfg[0], cfg[1], 0) for cfg in O3_CONFIGS}
            except Exception as e:
                log(f"O2/O3 {release} {det} seed {seed}: farming failed ({e})")
                farms = {}
            for name, b, kw in o2_variants(budget):
                out = h5.simulate_h(sd, b, delta, group_col, **kw)
                rr = row(out, group_col, release=release, detector=det, seed=seed, policy=name, budget=b)
                if b == budget and ("generic", 3) in farms:
                    sdf, farmers, probes = farms[("generic", 3)]
                    a0 = out[out["y"] == 1].groupby("user")["alert"].any()
                    f = _farm_eval(sd, sdf, farmers, probes, b, delta, group_col, kw, 1.0, a0)
                    rr.update({"farm_" + k: v for k, v in f.items()})
                r2.append(rr)
            for vname, kw in O3_VARIANTS:
                out = h5.simulate_h(sd, budget, delta, group_col, **kw)
                a0 = out[out["y"] == 1].groupby("user")["alert"].any()
                for cfg, (sdf, farmers, probes) in farms.items():
                    for p in P_CLEAR:
                        for dr in range(1 if p == 1.0 else draws):
                            f = _farm_eval(sd, sdf, farmers, probes, budget, delta, group_col, kw, p, a0, seed=dr)
                            r3.append(dict(release=release, detector=det, seed=seed, variant=vname, knowledge=cfg[0],
                                           n_probe=cfg[1], p_clear=p, draw=dr, **f))
            log(f"O2/O3 {release} {det} seed {seed} done")
    return pd.DataFrame(r2), pd.DataFrame(r3)


O4_POLICIES = [j6.DEFAULT, ("B4c outcome-aware", dict(history="cleared"))]


def o4_online_keep(runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit", policies=O4_POLICIES,
                  burn=30, q=0.9, noisy_draws=5, eps=1.0, seq_ctx=None, release="r4.2", seq_stride=1):
    """runs: {detector: {seed: ScoredData}}; rescorers: same shape. Returns per-insider rows:
    for each attacker, the insider's excess contribution (G1 - P1) - (G0 - P0), its predicted
    P1 and G1, and the chosen start."""
    out_rows, agree_rows = [], []
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            T = sd.test
            outG = h5.simulate_h(sd, budget, 1.0, group_col, history="global")
            Tg = T.assign(_all="all")
            outG2 = outG.assign(_all="all")
            FG = n10.FixedState(Tg, outG2, "_all")
            hitG0 = h5.insider_hits(outG)
            ctx = seq_ctx if getattr(sd.model, "sequence", False) else None
            cand = {}
            for u in T.loc[T["y"] == 1, "user"].unique():
                c = n10._insider_candidates(sd, None, u, group_col, ctx, stride=seq_stride if ctx is not None else 1)
                if c is not None:
                    cand[u] = c
            for pol, kw in policies:
                state = {}
                out0, rec = h5.simulate_h(sd, budget, delta, group_col, record=True, seed=0, state_log=state, **kw)
                FP = n10.FixedState(T, out0, group_col)
                DA = n10.DayAllocator(sd, kw, delta, group_col, state)
                agree_rows.append(dict(release=release, detector=det, seed=seed, policy=pol, agree_parity=FP.agree, agree_global=FG.agree))
                hitP0 = h5.insider_hits(out0)
                cut, obs, rr = n10.timing_signals(out0, rec, group_col)
                defs = [n10.deficit_map(rr, rec, eps, dr) for dr in range(noisy_draws)]
                for u, (rows, mal_idx, L, starts, raw, days) in cand.items():
                    g = T.loc[rows[0], group_col]
                    own = T["raw_score"].to_numpy()
                    own_by_day = dict(zip(T.loc[rows, "day"], own[rows]))
                    row_by_day = dict(zip(T.loc[rows, "day"], rows))
                    n_exact = 0
                    dvec = T.loc[rows, "day"].to_numpy()
                    tP = np.array([FP.threshold(x, g, own_by_day.get(x)) for x in dvec])
                    tG = np.array([FG.threshold(x, "all", own_by_day.get(x)) for x in dvec])
                    tN = np.array([FP.near(x, g, own_by_day.get(x)) for x in dvec])
                    pos = starts[:, None] + np.arange(L)[None, :]
                    aP = raw > tP[pos]
                    aG = raw > tG[pos]
                    P1 = aP.any(1).astype(int)
                    G1 = aG.any(1).astype(int)
                    cand_exact = (~aP) & ((raw > tG[pos]) | (raw > tN[pos]))
                    for j in np.flatnonzero((P1 == 0) & cand_exact.any(1)):
                        for l in np.flatnonzero(cand_exact[j]):
                            d = days[j, l]
                            n_exact += 1
                            if DA.alerted(d, row_by_day[d], raw[j, l]):
                                P1[j] = 1
                                break
                    p0, g0 = int(hitP0.get(u, 0)), int(hitG0.get(u, 0))
                    val = (G1 - P1) - (g0 - p0)
                    dd = T.loc[rows, "day"].to_numpy()
                    gcut = np.array([cut.get((x, g), 0) for x in dd], float)
                    sig_cut = np.convolve(gcut, np.ones(L), "valid")[starts]
                    scen = T.loc[rows[0], "scenario"] if "scenario" in T else None
                    base_ = dict(release=release, detector=det, seed=seed, policy=pol, user=u, scenario=scen, P0=p0, G0=g0)
                    s0 = int(np.flatnonzero(np.isin(rows, mal_idx))[0])
                    j0 = int(np.argmin(np.abs(starts - s0)))
                    out_rows.append(dict(base_, attack="relocated to its original start", draw=-1, triggered=np.nan,
                                         d=float(val[j0]), d_keep=float(val[j0]), P1=float(P1[j0]), G1=float(G1[j0]), start=int(starts[j0]),
                                         original_start=s0, last_start=int(starts[-1])))
                    for dr, est in enumerate(defs):
                        s_ = np.array([est.get((x, g), 0.0) for x in dd])
                        jpos, trig = m9._online_start(s_, L, burn, q)
                        jj = int(np.argmin(np.abs(starts - jpos)))
                        out_rows.append(dict(base_, attack=f"online deficit, q={q}", draw=dr, triggered=bool(trig),
                                             d=float(val[jj]), d_keep=float(val[jj]) if trig else 0.0,
                                             P1=float(P1[jj]) if trig else float(p0), G1=float(G1[jj]) if trig else float(g0),
                                             P1_forced=float(P1[jj]), G1_forced=float(G1[jj]), start=int(starts[jj]),
                                             original_start=s0, last_start=int(starts[-1])))
                    out_rows.append(dict(base_, attack="random", draw=-1, triggered=np.nan, d=float(val.mean()), d_keep=float(val.mean()),
                                         P1=float(P1.mean()), G1=float(G1.mean()), start=-1, original_start=s0, last_start=int(starts[-1])))
                    out_rows.append(dict(base_, attack="oracle: maximizes excess loss", draw=-1, triggered=np.nan, d=float(val.max()),
                                         d_keep=float(max(val.max(), 0.0)), P1=float(P1[int(np.argmax(val))]), G1=float(G1[int(np.argmax(val))]),
                                         start=int(starts[int(np.argmax(val))]), original_start=s0, last_start=int(starts[-1])))
                log(f"O4 {release} {det} seed {seed} {pol} done ({len(cand)} insiders)")
    return pd.DataFrame(out_rows), pd.DataFrame(agree_rows)

