"""Experiments for the referee reports (revision 10).

N1  Burden and detection for every retrained seed (seed means for the main tables), with a
    randomized global policy, convex costs of repeated suspicion and the daily budget k
N2  Timing attacks as unilateral deviations: one insider relocates while every other employee
    keeps the original behavior; every feasible start day is evaluated against the allocation
    the others produce, which gives an oracle that maximizes the excess loss; over seeds; a
    sample is checked with full simulations
N3  Farming over seeds with delayed and insider-level verdicts, earlier probes, a cooldown
    without backfill, and three defenses
N4  Per-employee-baseline detectors: benefit, farming and unilateral timing
N5  Power of the permutation test and a size-weighted statistic
N6  Insiders whose episodes span the training/test split
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev as fr
import fairirs_rev5 as h5
import fairirs_rev6 as j6
import fairirs_rev8 as l8
import fairirs_rev9 as m9

CF = fi.COUNT_FEATURES


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# =============================================================================
# N1  Burden and detection over seeds, randomized global policy, convex costs
# =============================================================================

def randomized_global(sd, budget=0.01, rho=0.2, pool=5, seed=0):
    """Memoryless randomized policy: each day the top (1 - rho) k scores are alerted, and rho k
    more are drawn uniformly from the next (pool - 1) k ranks."""
    rng = np.random.default_rng(seed)
    T = sd.test
    days = T["day"].to_numpy()
    s = T["score"].to_numpy()
    alert = np.zeros(len(T), bool)
    for d in np.unique(days):
        idx = np.flatnonzero(days == d)
        k = max(1, int(round(budget * len(idx))))
        order = idx[np.argsort(-s[idx], kind="stable")]
        k1 = int(round((1 - rho) * k))
        alert[order[:k1]] = True
        cand = order[k1:min(len(order), pool * k)]
        if len(cand) and k - k1 > 0:
            alert[rng.choice(cand, size=min(k - k1, len(cand)), replace=False)] = True
    out = T[["user", "day", "y", "score", "exposure", "scenario"] + [c for c in T.columns if c.startswith("g_")]].copy()
    out["alert"] = alert
    out["cleared"] = alert & (T["y"].to_numpy() == 0)
    return out


N1_POLICIES = [("B0 global", dict(history="global")),
               ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
               ("B0 + cooldown 30 days, no override", dict(history="global", cooldown=30, tau_ov=1.01)),
               ("B0 + cooldown 30 days, no backfill", dict(history="global", cooldown=30, cd_backfill=False)),
               ("B0 + scheduling", dict(history="global", schedule=True)),
               ("B4c outcome-aware", dict(history="cleared")),
               j6.DEFAULT]


def burden_row(out, group_col, **extra):
    m = h5.metrics(out, group_col, **extra)
    ins = set(out.loc[out["y"] == 1, "user"])
    b = out[(out["y"] == 0) & ~out["user"].isin(ins)]
    F = b.groupby("user")["alert"].sum()
    Fp = F[F > 0].to_numpy(float)
    m.update(alerted_employees=int((F > 0).sum()), repeat_alerts=int((F - 1).clip(lower=0).sum()),
             false_alerts=int(F.sum()), max_F_benign=int(F.max()) if len(F) else 0,
             gini_F=gini(F.to_numpy()), top1_share=float(np.sort(F.to_numpy())[::-1][:max(1, len(F) // 100)].sum() / max(F.sum(), 1)),
             cost_g1_5=float((Fp ** 1.5).sum()), cost_g2=float((Fp ** 2).sum()))
    return m


def n1_seeds(runs, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    """runs: {detector: {seed: ScoredData}}."""
    rows = []
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            for pol, kw in N1_POLICIES:
                if kw.get("schedule") and sd.phi is None:
                    continue
                out = h5.simulate_h(sd, budget, delta, group_col, **kw)
                rows.append(burden_row(out, group_col, release=release, detector=det, seed=seed, policy=pol))
            for rho in (0.2, 0.5):
                out = randomized_global(sd, budget, rho=rho, seed=seed)
                rows.append(burden_row(out, group_col, release=release, detector=det, seed=seed, policy=f"randomized global, rho={rho}"))
            log(f"N1 {release} {det} seed {seed} done")
    return pd.DataFrame(rows)


def daily_budget(sd, budget=0.01, release="r4.2"):
    n = sd.test.groupby("day").size()
    k = np.maximum(1, np.round(budget * n)).astype(int)
    return pd.DataFrame(dict(release=release, day=n.index, active=n.to_numpy(), k=k.to_numpy()))


# =============================================================================
# Signals of the timing attackers (as in revision 9)
# =============================================================================

def timing_signals(out0, rec, group_col, window=30):
    r = rec.copy()
    r["cut"] = (r["K"] - r["k"]).clip(lower=0)
    cut = r.set_index(["day", "group"])["cut"].to_dict()
    obs = {}
    for lab, D in r.sort_values("day").groupby("group"):
        a = D["k"].rolling(window, min_periods=1).sum().shift(1).fillna(0)
        n = D["n"].rolling(window, min_periods=1).sum().shift(1).fillna(1)
        for dd, v in zip(D["day"], (a / n).to_numpy()):
            obs[(dd, lab)] = v
    return cut, obs, r


def deficit_map(rec_r, rec, eps, seed):
    M = h5._monthly_noisy(rec, eps, np.random.default_rng(500 + seed))
    months = sorted(M["month"].unique())
    cumk = M.pivot(index="month", columns="group", values="k").fillna(0).cumsum()
    cumn = M.pivot(index="month", columns="group", values="n").fillna(0).cumsum()
    overall = cumk.sum(axis=1) / cumn.sum(axis=1)
    Rr = rec_r.copy()
    Rr["month"] = pd.to_datetime(Rr["day"]).dt.strftime("%Y-%m")
    est = {}
    for lab, D in Rr.sort_values("day").groupby("group"):
        D = D.copy()
        D["prev"] = D["month"].map(lambda mth: months[months.index(mth) - 1] if mth in months and months.index(mth) > 0 else None)
        first = D["month"] != D["month"].shift(1)
        D["k_since"] = D.groupby("month")["k"].cumsum().shift(1).fillna(0)
        D.loc[first, "k_since"] = 0
        D["n_since"] = D.groupby("month")["n"].cumsum().shift(1).fillna(0)
        D.loc[first, "n_since"] = 0
        for dd, pm, ks, ns in zip(D["day"], D["prev"], D["k_since"], D["n_since"]):
            if pm is None or (isinstance(pm, float) and np.isnan(pm)) or lab not in cumk.columns:
                est[(dd, lab)] = 0.0
                continue
            ck, cn = cumk.loc[pm, lab] + ks, cumn.loc[pm, lab] + ns
            est[(dd, lab)] = ck / max(cn, 1) - overall.loc[pm]
    return est


# =============================================================================
# N2  Unilateral timing attacks
# =============================================================================

class DayAllocator:
    """Recomputes one day's parity allocation with one row's score changed, from the history
    the allocator saw that day in the unattacked run (unilateral deviation)."""

    def __init__(self, sd, kw, delta, group_col, state):
        T = sd.test
        self.s = T["score"].to_numpy(np.float64)
        self.g = T[group_col].to_numpy()
        self.state, self.kw, self.delta = state, kw, delta
        self.rows_by_day = T.groupby("day").indices
        self.raw_sorted = np.sort(T["raw_score"].to_numpy(np.float64))
        self.ref, self.N = sd.raw_ref, len(T)

    def score_of(self, raw):
        q = np.searchsorted(self.ref, raw, side="right") / len(self.ref)
        return q + 1e-6 * (np.searchsorted(self.raw_sorted, raw, side="right") + 1) / (self.N + 1)

    def alerted(self, d, own_row, raw):
        st = self.state.get(d)
        if st is None:
            return None
        A0, C0, k, use_global = st
        idx = self.rows_by_day[d]
        sc = self.s[idx].copy()
        pos = np.flatnonzero(idx == own_row)
        if len(pos) == 0:
            return None
        sc[pos[0]] = self.score_of(raw)
        gi = self.g[idx]
        if self.kw.get("allocator") == "deficit":
            m = h5._deficit(sc, gi, k, A0, C0)
        else:
            m = fi.fair_allocate_cum(sc, gi, k, self.delta, A0, C0)[0]
        return bool(m[pos[0]])


class FixedState:
    """Alert thresholds of a completed run: on each day, the k-th largest raw score of each
    group (parity) or overall (global), with the insider's own row removed. An insider's
    relocated user-day is alerted if its raw score exceeds the threshold of its day."""

    def __init__(self, T, out, group_col):
        self.raw = T["raw_score"].to_numpy(np.float64)
        self.day = T["day"].to_numpy()
        self.grp = T[group_col].to_numpy()
        a = out["alert"].to_numpy()
        self.tab = {}
        self.agree = 0.0
        recomputed = np.zeros(len(T), bool)
        key = pd.Series(list(zip(self.day, self.grp)))
        for (d, g), idx in key.groupby(key).groups.items():
            idx = np.asarray(idx)
            r = np.sort(self.raw[idx])[::-1]
            k = int(a[idx].sum())
            self.tab[(d, g)] = (r, k)
            if k > 0:
                recomputed[idx[np.argsort(-self.raw[idx], kind="stable")[:k]]] = True
        self.agree = float((recomputed == a).mean())

    def near(self, d, g, own_raw, extra=5):
        """Score the insider would need to be within `extra` ranks of the group's cut."""
        r, k = self.tab.get((d, g), (np.array([]), 0))
        j = k + extra - 1
        return r[j] if len(r) > j else -np.inf

    def threshold(self, d, g, own_raw):
        r, k = self.tab.get((d, g), (np.array([]), 0))
        if k <= 0:
            return np.inf
        if own_raw is not None and len(r) >= k and own_raw >= r[k - 1]:
            return r[k] if len(r) > k else -np.inf
        return r[k - 1] if len(r) >= k else -np.inf


def _insider_candidates(sd, R, u, group_col, seq_ctx=None, stride=1):
    """Raw scores of the insider's relocated malicious days for every feasible start.
    Returns (rows, mal_idx, L, starts, raw[starts, L], days[starts, L])."""
    T = sd.test
    rows = T.index[T["user"] == u].to_numpy()
    mal_idx = T.index[(T["user"] == u) & (T["y"] == 1)].to_numpy()
    L, M = len(mal_idx), len(rows)
    if L == 0 or M <= L:
        return None
    ben = T.loc[rows][T.loc[rows, "y"] == 0]
    base = np.round(ben[CF].median().to_numpy(np.float64)) if len(ben) else np.zeros(len(CF))
    inc = np.clip(T.loc[mal_idx, CF].to_numpy(np.float64) - base, 0, None)
    X = T.loc[rows, CF].to_numpy(np.float64).copy()
    is_mal = np.isin(rows, mal_idx)
    X[is_mal] = base
    starts = np.arange(0, M - L + 1, stride)
    tf, model = sd.transform, sd.model
    seq = bool(getattr(model, "sequence", False))
    dayv = T.loc[rows, "day"].to_numpy()
    days = np.stack([dayv[s:s + L] for s in starts])
    if not seq:
        blocks = []
        for s in starts:
            Z = X[s:s + L] + inc
            blocks.append(Z)
        Z = np.concatenate(blocks)
        D = pd.DataFrame(Z, columns=CF)
        D["weekend"] = T.loc[rows, "weekend"].to_numpy()[np.concatenate([np.arange(s, s + L) for s in starts])]
        for c in T.columns:
            if c not in D and c in ("user", "role") or c.startswith("g_"):
                D[c] = T.loc[rows[0], c]
        D = m9.repair(D, D.index)
        raw = model.score(tf.transform(D)).reshape(len(starts), L)
        return rows, mal_idx, L, starts, raw, days
    # sequence model: score each candidate history as its own pseudo-employee
    C = seq_ctx[seq_ctx["user"] == u]
    cols = ["user", "day"] + CF + ["weekend"]
    Tu = T.loc[rows, cols].copy()
    for c in CF:
        Tu[c] = Tu[c].astype(np.float64)
    frames, tpos = [], []
    for s in starts:
        Z = X.copy()
        Z[s:s + L] = Z[s:s + L] + inc
        Tz = Tu.copy()
        Tz[CF] = Z
        Tz = m9.repair(Tz, Tz.index[s:s + L])
        uid = f"{u}#{s}"
        frames.append(C[cols].assign(user=uid, _p=-1))
        frames.append(Tz.assign(user=uid, _p=np.arange(len(rows))))
    Fu = pd.concat(frames, ignore_index=True)
    r = model.score_seq(Fu, tf.transform(Fu))
    raw = np.empty((len(starts), L))
    Fu["_r"] = r
    for j, s in enumerate(starts):
        sub = Fu[(Fu["user"] == f"{u}#{s}") & (Fu["_p"] >= 0)].sort_values("_p")
        raw[j] = sub["_r"].to_numpy()[s:s + L]
    return rows, mal_idx, L, starts, raw, days


N2_POLICIES = [("B4 label-free", dict(history="alerts")),
               ("B4c outcome-aware", dict(history="cleared")),
               ("B4c outcome-aware, 7-day verdict delay", dict(history="cleared", delay=7)),
               j6.DEFAULT]


def n2_unilateral(runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit", policies=N2_POLICIES,
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
            FG = FixedState(Tg, outG2, "_all")
            hitG0 = h5.insider_hits(outG)
            ctx = seq_ctx if getattr(sd.model, "sequence", False) else None
            cand = {}
            for u in T.loc[T["y"] == 1, "user"].unique():
                c = _insider_candidates(sd, None, u, group_col, ctx, stride=seq_stride if ctx is not None else 1)
                if c is not None:
                    cand[u] = c
            for pol, kw in policies:
                state = {}
                out0, rec = h5.simulate_h(sd, budget, delta, group_col, record=True, seed=0, state_log=state, **kw)
                FP = FixedState(T, out0, group_col)
                DA = DayAllocator(sd, kw, delta, group_col, state)
                agree_rows.append(dict(release=release, detector=det, seed=seed, policy=pol, agree_parity=FP.agree, agree_global=FG.agree))
                hitP0 = h5.insider_hits(out0)
                cut, obs, rr = timing_signals(out0, rec, group_col)
                defs = [deficit_map(rr, rec, eps, dr) for dr in range(noisy_draws)]
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
                    choice = {"oracle: maximizes excess loss": [int(np.argmax(val))],
                              "oracle: evades the parity rule if possible": [int(np.argmax(1 - P1 + 1e-6 * val))],
                              "knows allocation: largest cut": [int(np.argmax(sig_cut))]}
                    hs, on = [], []
                    for est in defs:
                        s_ = np.array([est.get((x, g), 0.0) for x in dd])
                        hs.append(int(np.argmax(s_[starts])))
                        j_, _ = m9._online_start(s_, L, burn, q)
                        on.append(int(np.argmin(np.abs(starts - j_))))
                    choice["hindsight deficit"] = hs
                    choice[f"online deficit, q={q}"] = on
                    for name, js in choice.items():
                        out_rows.append(dict(release=release, detector=det, seed=seed, policy=pol, user=u, attack=name,
                                             d=float(np.mean(val[js])), P1=float(np.mean(P1[js])), G1=float(np.mean(G1[js])),
                                             P0=p0, G0=g0, start=int(starts[js[0]])))
                    out_rows.append(dict(release=release, detector=det, seed=seed, policy=pol, user=u, attack="random",
                                         d=float(val.mean()), P1=float(P1.mean()), G1=float(G1.mean()), P0=p0, G0=g0, start=-1))
                log(f"N2 {release} {det} seed {seed} {pol} done ({len(cand)} insiders)")
    return pd.DataFrame(out_rows), pd.DataFrame(agree_rows)


def relocate_one(sd, u, start, rescorer):
    """Relocate only insider u to the given start; every other row is unchanged."""
    T = l8._float_features(sd.test)
    rows = T.index[T["user"] == u].to_numpy()
    mal_idx = T.index[(T["user"] == u) & (T["y"] == 1)].to_numpy()
    L = len(mal_idx)
    ben = T.loc[rows][T.loc[rows, "y"] == 0]
    base = np.round(ben[CF].median().to_numpy(np.float64))
    inc = np.clip(T.loc[mal_idx, CF].to_numpy(np.float64) - base, 0, None)
    scen = T.loc[mal_idx[0], "scenario"]
    new = rows[start:start + L]
    T.loc[mal_idx, CF] = base
    T.loc[mal_idx, "y"] = 0
    T.loc[mal_idx, "scenario"] = ""
    T.loc[new, CF] = T.loc[new, CF].to_numpy(np.float64) + inc
    T.loc[new, "y"] = 1
    T.loc[new, "scenario"] = scen
    T["y"] = T["y"].astype(np.int8)
    return m9.RepairingRescorer(rescorer)(T, list(mal_idx) + list(new))


def n2_validate(per, runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit", policies=N2_POLICIES,
                n_insiders=12, attack="oracle: maximizes excess loss", seed_only=0, release="r4.2"):
    """Full simulations for a sample of insiders: relocate one insider at the chosen start and
    rerun the parity rule and the global policy."""
    rows = []
    rng = np.random.default_rng(0)
    for det, by_seed in runs.items():
        if seed_only not in by_seed:
            continue
        sd = by_seed[seed_only]
        for pol, kw in policies:
            D = per[(per["detector"] == det) & (per["seed"] == seed_only) & (per["policy"] == pol) & (per["attack"] == attack)]
            if D.empty:
                continue
            pick = D.sample(min(n_insiders, len(D)), random_state=int(rng.integers(1 << 30)))
            for _, r in pick.iterrows():
                sdx = relocate_one(sd, r["user"], int(r["start"]), rescorers[det][seed_only])
                oP = h5.simulate_h(sdx, budget, delta, group_col, seed=0, **kw)
                oG = h5.simulate_h(sdx, budget, 1.0, group_col, history="global")
                hp, hg = int(h5.insider_hits(oP).get(r["user"], 0)), int(h5.insider_hits(oG).get(r["user"], 0))
                rows.append(dict(release=release, detector=det, policy=pol, user=r["user"], start=int(r["start"]),
                                 P1_pred=r["P1"], G1_pred=r["G1"], P1_full=hp, G1_full=hg,
                                 others_P_changed=int((h5.insider_hits(oP).drop(r["user"], errors="ignore") !=
                                                       h5.insider_hits(h5.simulate_h(sd, budget, delta, group_col, seed=0, **kw)).drop(r["user"], errors="ignore")).sum())))
            log(f"N2 check {release} {det} {pol} done")
    return pd.DataFrame(rows)


def n2_summary(per, n_boot=5000, seed=0):
    """Excess loss per setting (seed mean), with an interval that resamples seeds and insiders."""
    rng = np.random.default_rng(seed)
    out = []
    for (rel, det, pol, atk), D in per.groupby(["release", "detector", "policy", "attack"], sort=False):
        M = D.pivot_table(index="user", columns="seed", values="d", aggfunc="mean").fillna(0.0)
        X = M.to_numpy()
        n, S = X.shape
        est = X.sum(0).mean()
        bi = rng.integers(0, n, size=(n_boot, n))
        bs = rng.integers(0, S, size=(n_boot, S))
        B = np.array([X[bi[b]][:, bs[b]].sum(0).mean() for b in range(n_boot)])
        out.append(dict(release=rel, detector=det, policy=pol, attack=atk, seeds=S, insiders=n,
                        excess=float(est), lo=float(np.percentile(B, 2.5)), hi=float(np.percentile(B, 97.5)),
                        per_seed_min=float(X.sum(0).min()), per_seed_max=float(X.sum(0).max()),
                        positive_insiders=float((X.mean(1) > 0).sum()), negative_insiders=float((X.mean(1) < 0).sum())))
    return pd.DataFrame(out)


# =============================================================================
# N3  Farming over seeds, verdict models and defenses
# =============================================================================

N3_VARIANTS = [("cooldown 30 d, override 0.999", dict(history="global", cooldown=30)),
               ("no override", dict(history="global", cooldown=30, tau_ov=1.01)),
               ("verdicts 7 days late", dict(history="global", cooldown=30, cd_delay=7)),
               ("verdicts 14 days late", dict(history="global", cooldown=30, cd_delay=14)),
               ("no backfill", dict(history="global", cooldown=30, cd_backfill=False)),
               ("defense: re-check 10% of suppressed alerts", dict(history="global", cooldown=30, cd_recheck=0.1)),
               ("defense: farming-signature monitor (score >= 0.99 within 7 days of a clearance)",
                dict(history="global", cooldown=30, cd_recent_days=7, cd_recent_tau=0.99)),
               ("defense: 3-day cooldown after clearing a score >= 0.99", dict(history="global", cooldown=30, cd_high_days=3)),
               ("global (no cooldown)", dict(history="global"))]


def farm_gap(sd, rescorer, knowledge="own", n_probe=3, gap=0, role_col="role", cap_q=0.99, min_rows=50):
    """As fairirs_rev8.farm with every probe cleared, but the probes are the n benign days ending
    `gap` days before the first malicious day, and probes are made consistent."""
    T = l8._float_features(sd.test)
    mal = T[T["y"] == 1]
    ins = list(mal["user"].unique())
    inc, scen = {}, {}
    for u, D in mal.groupby("user"):
        ub = T[(T["user"] == u) & (T["y"] == 0)]
        b = ub[CF].median() if len(ub) else pd.Series(0.0, index=CF)
        inc[u] = (D[CF].mean() - b).clip(lower=0).to_numpy(np.float64)
        scen[u] = D["scenario"].iloc[0]
    ben = T[(T["y"] == 0) & ~T["user"].isin(ins)]
    gcap = ben[CF].quantile(cap_q).to_numpy(np.float64)
    caps = {r_: (D[CF].quantile(cap_q).to_numpy(np.float64) if len(D) >= min_rows else gcap)
            for r_, D in ben.groupby(role_col)} if role_col in T else {}
    T["fc"] = np.nan
    changed, farmers = [], []
    for u in ins:
        others = [v for v in ins if v != u]
        d = inc[u] if knowledge == "own" else (np.mean([inc[v] for v in others], axis=0) if others else inc[u])
        d0 = mal.loc[mal["user"] == u, "day"].min()
        prior = T[(T["user"] == u) & (T["y"] == 0) & (T["day"] < d0 - pd.Timedelta(days=gap))].sort_values("day").tail(n_probe)
        if prior.empty:
            continue
        idx = prior.index
        cap = caps.get(T.loc[idx[0], role_col], gcap) if role_col in T else gcap
        x = T.loc[idx, CF].to_numpy(np.float64)
        T.loc[idx, CF] = np.maximum(x, np.minimum(np.round(x + d), cap))
        T.loc[idx, "fc"] = 0.0
        changed.extend(idx)
        farmers.append(u)
    sdf = m9.RepairingRescorer(rescorer)(T, changed, name=sd.detector + "-farm")
    return sdf, farmers, list(changed)


def insider_level_fc(T):
    """Benign user-days of an insider between the first and last malicious day are confirmed
    (an investigation of the insider during the episode finds it), not cleared."""
    fc = np.full(len(T), np.nan)
    mal = T[T["y"] == 1].groupby("user")["day"].agg(["min", "max"])
    for u, r in mal.iterrows():
        m = (T["user"] == u).to_numpy() & (T["y"] == 0).to_numpy() & (T["day"] >= r["min"]).to_numpy() & (T["day"] <= r["max"]).to_numpy()
        fc[m] = 1.0
    return fc


def n3_farming(runs, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit",
               grid=(("own", 3, 0), ("generic", 3, 0), ("generic", 1, 0), ("own", 3, 14)), release="r4.2"):
    rows = []
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            R = rescorers[det][seed]
            base_T = sd.test
            fc_ins = insider_level_fc(base_T)
            farms = {}
            for knowledge, n, gap in grid:
                farms[(knowledge, n, gap)] = farm_gap(sd, R, knowledge, n, gap)
            for vname, kw in N3_VARIANTS + [("insider-level verdicts", dict(history="global", cooldown=30))]:
                ins_level = vname == "insider-level verdicts"
                sd0 = sd
                if ins_level:
                    sd0 = fi.ScoredData(sd.detector, base_T.assign(fc=fc_ins), sd.phi, sd.model, sd.transform, sd.raw_ref, sd.train_ref)
                o = h5.simulate_h(sd0, budget, delta, group_col, fc_col="fc" if ins_level else None, **kw)
                b = burden_row(o, group_col)
                a0 = o[o["y"] == 1].groupby("user")["alert"].any()
                for (knowledge, n, gap), (sdf, farmers, probes) in farms.items():
                    Tf = sdf.test
                    if ins_level:
                        f2 = Tf["fc"].to_numpy().copy()
                        m = np.isnan(f2)
                        f2[m] = fc_ins[m]
                        Tf = Tf.assign(fc=f2)
                    sdx = fi.ScoredData(sdf.detector, Tf, sdf.phi, sdf.model, sdf.transform, sdf.raw_ref, sdf.train_ref)
                    of = h5.simulate_h(sdx, budget, delta, group_col, fc_col="fc", **kw)
                    a1 = of[(of["y"] == 1) & of["user"].isin(farmers)].groupby("user")["alert"].any()
                    det0 = a0.reindex(farmers).fillna(False).astype(bool)
                    det1 = a1.reindex(farmers).fillna(False).astype(bool)
                    probed = of.loc[probes]
                    rows.append(dict(release=release, detector=det, seed=seed, variant=vname, knowledge=knowledge, n_probe=n, gap=gap,
                                     farmers=len(farmers), detected_no_farming=int(det0.sum()), evaded=int((det0 & ~det1).sum()),
                                     gained=int((~det0 & det1).sum()), probes_alerted=int(probed["alert"].sum()),
                                     insiders_detected_variant=b["insiders_detected"], max_F_variant=b["max_F_benign"],
                                     alerted_employees_variant=b["alerted_employees"]))
            log(f"N3 {release} {det} seed {seed} done")
    return pd.DataFrame(rows)


# =============================================================================
# N4  Per-employee-baseline detectors
# =============================================================================

class UserTransform:
    """Standardized log counts minus the employee's own training-period mean."""

    def __init__(self, ft, base):
        self.ft, self.base = ft, base

    def transform(self, F):
        X = self.ft.transform(F)
        B = self.base.reindex(F["user"].astype(str).str.split("#").str[0].values).fillna(0.0).to_numpy(np.float32)
        return (X - B).astype(np.float32)


def run_detector_user2(ds, det):
    F = ds.df
    tr = F["train"].to_numpy()
    y = F["y"].to_numpy()
    fit = tr & (y == 0) if not det.supervised else tr
    ft = fi.FeatureTransform("raw").fit(F[fit])
    X0 = ft.transform(F)
    base = fi.user_baseline(F, X0, tr & (y == 0))
    utf = UserTransform(ft, base)
    X = utf.transform(F)
    det.fit(X[fit], y[fit])
    raw = det.score(X[~tr])
    ref = np.sort(det.score(X[tr & (y == 0)]))
    test = F[~tr].copy().reset_index(drop=True)
    test["raw_score"] = raw
    test["score"] = fi.to_quantile_ref(raw, ref)
    return fi.ScoredData(det.name + "-user", test, X[~tr].copy(), det, utf, ref, None)


# =============================================================================
# N5  Power of the permutation test
# =============================================================================

def _group_stats(F, n, g, labels):
    sF = np.bincount(g, weights=F, minlength=len(labels))
    sn = np.bincount(g, weights=n, minlength=len(labels))
    rate = sF / np.maximum(sn, 1)
    overall = sF.sum() / sn.sum()
    return rate.max() - rate.min(), float((sn * (rate - overall) ** 2).sum() / sn.sum())


def permutation_p(F, n, g, labels, n_perm, rng):
    obs = _group_stats(F, n, g, labels)
    cnt = np.zeros(2)
    for _ in range(n_perm):
        gp = rng.permutation(g)
        st = _group_stats(F, n, gp, labels)
        cnt += np.array([st[0] >= obs[0] - 1e-15, st[1] >= obs[1] - 1e-15])
    return (1 + cnt) / (1 + n_perm), obs


def n5_power(runs, budget=0.01, group_col="g_functional_unit", effects=(0.25, 0.5, 1.0, 2.0, 4.0),
             n_inject=50, n_perm=200, release="r4.2", seed=0):
    """Observed p-values for both statistics, and power: the false alerts of one group (the
    smallest or the largest) are raised by a relative effect r through Poisson additions."""
    rng = np.random.default_rng(seed)
    rows = []
    for det, by_seed in runs.items():
        sd = by_seed.get(0) or next(iter(by_seed.values()))
        out = h5.simulate_h(sd, budget, 1.0, group_col, history="global")
        ins = set(out.loc[out["y"] == 1, "user"])
        b = out[(out["y"] == 0) & ~out["user"].isin(ins)]
        E = b.groupby("user").agg(F=("alert", "sum"), n=("alert", "size"), g=(group_col, lambda s: s.mode().iloc[0]))
        labels = sorted(E["g"].unique())
        gi = E["g"].map({l: i for i, l in enumerate(labels)}).to_numpy()
        F, n = E["F"].to_numpy(float), E["n"].to_numpy(float)
        p, obs = permutation_p(F, n, gi, labels, 2000, rng)
        rows.append(dict(release=release, detector=det, kind="observed", group="", effect=0.0,
                         p_range=float(p[0]), p_weighted=float(p[1]), stat_range=obs[0], stat_weighted=obs[1]))
        sizes = pd.Series(gi).value_counts()
        for which, gsel in (("smallest", sizes.idxmin()), ("largest", sizes.idxmax())):
            mem = gi == gsel
            rate_g = F[mem].sum() / n[mem].sum()
            for r in effects:
                hits = np.zeros(2)
                for _ in range(n_inject):
                    F2 = F.copy()
                    F2[mem] = F2[mem] + rng.poisson(r * rate_g * n[mem])
                    pp, _ = permutation_p(F2, n, gi, labels, n_perm, rng)
                    hits += pp < 0.05
                rows.append(dict(release=release, detector=det, kind="power", group=f"{which} ({labels[gsel]}, {int(mem.sum())} employees)",
                                 effect=r, p_range=float(hits[0] / n_inject), p_weighted=float(hits[1] / n_inject)))
        log(f"N5 {release} {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# N6  Insiders whose episodes span the split
# =============================================================================

def n6_spanning(ds, release="r4.2"):
    F = ds.df
    mal = F[F["y"] == 1]
    tr_ins = set(mal.loc[mal["train"], "user"])
    te_ins = set(mal.loc[~mal["train"], "user"])
    span = sorted(tr_ins & te_ins)
    return pd.DataFrame(dict(release=release, user=span)), dict(release=release, train_insiders=len(tr_ins),
                                                                test_insiders=len(te_ins), spanning=len(span))


def n6_detection(runs, spanning, budget=0.01, delta=0.002, group_col="g_functional_unit", release="r4.2"):
    """Insiders detected with and without the insiders whose episodes cross the split."""
    rows = []
    span = set(spanning)
    for det, by_seed in runs.items():
        for seed, sd in by_seed.items():
            for pol, kw in N1_POLICIES:
                if kw.get("schedule") and sd.phi is None:
                    continue
                h = h5.insider_hits(h5.simulate_h(sd, budget, delta, group_col, **kw))
                keep = h[~h.index.isin(span)]
                rows.append(dict(release=release, detector=det, seed=seed, policy=pol, insiders=len(h), detected=int(h.sum()),
                                 insiders_excluding_spanning=len(keep), detected_excluding_spanning=int(keep.sum())))
    return pd.DataFrame(rows)
