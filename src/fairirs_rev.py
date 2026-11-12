"""Additional experiments for the FAIR-IRS paper.

E1  Heterogeneous prevalence across groups (M2)
E2  Adaptive attacks on cumulative group allocation (M5)
E3  Sensitivity to role-level behavioural differences (M1)
E4  Several persistent benign behaviour patterns (M7)

All experiments reuse the cached detector scores of the main notebook.
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================
# Allocation policy with configurable history (no scheduling)
# =============================================================================

def simulate_rev(sd, budget=0.01, delta=0.002, group_col="g_functional_unit", history="alerts",
                 p_miss=0.0, cap=None, record=False, seed=0, floor=None, schedule=False,
                 lam=0.01, gamma=0.8, tau_ov=0.999, m_cap=3, horizon=30):
    """Day-by-day alerting without scheduling.

    history='alerts'  : parity on cumulative alert rates (FAIR-IRS as in the paper)
    history='cleared' : parity on cumulative cleared-alert rates, i.e., alerts that
                        investigations found benign, over user-days not confirmed
                        malicious (both observable by the security team)
    cap               : each employee contributes at most `cap` alerts per calendar
                        month to its group's cumulative count (limits budget exhaustion)
    record            : also return per-day, per-group allocation (K_g, k_g)
    floor             : score floor theta (only the global top ceil(theta*k) may be alerted)
    schedule          : burden-aware scheduling, exactly as in fairirs.simulate_policy
    """
    rng = np.random.default_rng(seed)
    T = sd.test
    days = T["day"].to_numpy()
    users = T["user"].to_numpy()
    y = T["y"].to_numpy()
    s = T["score"].to_numpy().astype(np.float64)
    g = T[group_col].to_numpy()
    alert = np.zeros(len(T), bool)
    cleared = np.zeros(len(T), bool)
    cumA, cumC, cumConf, per_user = {}, {}, {}, {}
    fallbacks, budget_total, disp = 0, 0, []
    rec = []
    store, n_disc = {}, 0
    if schedule:
        phin = sd.phi / (np.linalg.norm(sd.phi, axis=1, keepdims=True) + 1e-9)
    uniq_days, starts = np.unique(days, return_index=True)
    bounds = list(starts) + [len(T)]
    for di, d in enumerate(uniq_days):
        idx = np.arange(bounds[di], bounds[di + 1])
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
        if history == "cleared":
            Ceff = {lab: cumC.get(lab, 0) - cumConf.get(lab, 0) for lab in cumC}
            m, fb = fi.fair_allocate_cum(sc, g[idx], k, delta, cumA, Ceff)
        else:
            m, fb = fi.fair_allocate_cum(sc, g[idx], k, delta, cumA, cumC)
        fallbacks += fb
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
                rec.append((d, lab, int(top[sel].sum()), int(m[sel].sum())))
        a_idx = idx[m]
        alert[a_idx] = True
        u = rng.random(len(a_idx))
        clr = np.where(y[a_idx] == 1, u < p_miss, True)
        cleared[a_idx[clr]] = True
        if schedule:
            for i in a_idx[clr]:
                store.setdefault(users[i], []).append((d, phin[i]))
        for lab, cnt in zip(*np.unique(g[idx], return_counts=True)):
            cumC[lab] = cumC.get(lab, 0) + cnt
        counted = a_idx[clr] if history == "cleared" else a_idx
        for i in counted:
            if cap is not None:
                key = (users[i], str(d)[:7])
                if per_user.get(key, 0) >= cap:
                    continue
                per_user[key] = per_user.get(key, 0) + 1
            cumA[g[i]] = cumA.get(g[i], 0) + 1
        conf = a_idx[~clr]
        for i in conf:
            cumConf[g[i]] = cumConf.get(g[i], 0) + 1
    out = T[["user", "day", "y", "score", "exposure", "scenario", group_col]].copy()
    out["alert"], out["cleared"] = alert, cleared
    out.attrs.update(fallback_days=fallbacks, discounts=n_disc, n_days=len(uniq_days),
                     disp_ranks=np.array(disp), budget_total=budget_total)
    if record:
        return out, pd.DataFrame(rec, columns=["day", "group", "K", "k"])
    return out


def _with_test(sd, T, name=None):
    return fi.ScoredData(name or sd.detector, T.reset_index(drop=True), sd.phi, sd.model,
                         sd.transform, sd.raw_ref, sd.train_ref)


def recall_of(out, users):
    users = set(users)
    mal = out[(out["y"] == 1) & out["user"].isin(users)]
    if mal.empty:
        return float("nan"), 0, 0
    det = mal.loc[mal["alert"], "user"].nunique()
    n = mal["user"].nunique()
    return det / n, det, n


# =============================================================================
# E1  Heterogeneous prevalence
# =============================================================================

def inject_prevalence(sd, group_col, target, pi_target, seed=0):
    """Raise the share of malicious user-days of group `target` to pi_target by
    turning episodes of benign members into synthetic insiders whose scores are
    drawn from the detector's scores on real malicious user-days."""
    rng = np.random.default_rng(seed)
    T = sd.test.copy()
    mal_scores = T.loc[T["y"] == 1, "score"].to_numpy()
    ep_len = T[T["y"] == 1].groupby("user").size().to_numpy()
    real_ins = set(T.loc[T["y"] == 1, "user"])
    in_g = T[group_col] == target
    n_g = int(in_g.sum())
    need = int(round(pi_target * n_g)) - int(T.loc[in_g, "y"].sum())
    synth = []
    if need <= 0:
        return _with_test(sd, T), synth
    cand = [u for u in T.loc[in_g, "user"].unique() if u not in real_ins]
    rng.shuffle(cand)
    for u in cand:
        if need <= 0:
            break
        rows = T.index[(T["user"] == u) & in_g].to_numpy()
        if len(rows) < 5:
            continue
        L = int(min(len(rows), max(1, rng.choice(ep_len)), need))
        start = rng.integers(0, len(rows) - L + 1)
        sel = rows[start:start + L]
        T.loc[sel, "y"] = 1
        T.loc[sel, "score"] = rng.choice(mal_scores, size=L) + rng.uniform(-1e-6, 1e-6, L)
        T.loc[sel, "scenario"] = "synthetic"
        synth.append(u)
        need -= L
    return _with_test(sd, T), synth


def e1_prevalence(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
                  detectors=("GBM", "LSTM", "IF"), pis=(0.01, 0.02, 0.05, 0.10), seeds=(0, 1, 2)):
    rows = []
    base = next(iter(sc.values())).test
    users_per = base.groupby(group_col)["user"].nunique().sort_values()
    targets = [users_per.index[0], users_per.index[-1]]
    log("E1 target groups:", targets, "users:", [int(users_per[t]) for t in targets])
    policies = [("B0 global", dict(delta=1.0)),
                ("B4 alert parity", dict(delta=delta)),
                ("B4c cleared-alert parity", dict(delta=delta, history="cleared")),
                ("B0 global, p_miss 0.3", dict(delta=1.0, p_miss=0.3)),
                ("B4 alert parity, p_miss 0.3", dict(delta=delta, p_miss=0.3))]
    for det in detectors:
        if det not in sc:
            log(f"[skip] E1 {det}: not in the scored cache")
            continue
        sd0 = sc[det]
        real = set(sd0.test.loc[sd0.test["y"] == 1, "user"])
        settings = [(None, 0.0, 0)] + [(t, p, s) for t in targets for p in pis for s in seeds]
        for target, pi_t, seed in settings:
            if target is None:
                sd, synth = sd0, []
            else:
                sd, synth = inject_prevalence(sd0, group_col, target, pi_t, seed)
            pi_g = sd.test.groupby(group_col)["y"].mean()
            for pol, kw in policies:
                out = simulate_rev(sd, budget=budget, group_col=group_col, seed=seed, **kw)
                m = fi.policy_metrics(out, group_col, pi_assumed=0.05)
                m15 = fi.policy_metrics(out, group_col, pi_assumed=0.15)
                ben = out[out["y"] == 0]
                far = ben.groupby(group_col)["alert"].mean()
                r_real = recall_of(out, real)
                r_syn = recall_of(out, synth)
                rows.append(dict(detector=det, target=target or "none", target_users=int(users_per.get(target, 0)) if target else 0,
                                 pi_target=pi_t, pi_actual=float(pi_g.get(target, np.nan)) if target else float(pi_g.max()),
                                 seed=seed, policy=pol,
                                 delta_AR=m["delta_AR"], delta_FAR=m["delta_FAR"], ratio_FAR=m["ratio_FAR"],
                                 delta_FAR_hat=m["delta_FAR_hat"], audit_abs_err=m["audit_abs_err"],
                                 audit_bound_5=m["audit_bound"], audit_holds_5=m["audit_holds"],
                                 audit_bound_15=m15["audit_bound"], audit_holds_15=m15["audit_holds"],
                                 lemma1_bound=m["lemma1_bound"],
                                 far_target=float(far.get(target, np.nan)) if target else np.nan,
                                 far_others_mean=float(far.drop(target).mean()) if target else float(far.mean()),
                                 recall_real=r_real[0], detected_real=r_real[1], n_real=r_real[2],
                                 recall_synth=r_syn[0], detected_synth=r_syn[1], n_synth=r_syn[2],
                                 precision=m["precision"], fallback_days=m["fallback_days"]))
        log(f"E1 {det} done")
    return pd.DataFrame(rows)


def e1_summary(df):
    keys = ["detector", "target", "target_users", "pi_target", "policy"]
    num = ["pi_actual", "delta_AR", "delta_FAR", "ratio_FAR", "delta_FAR_hat", "audit_abs_err",
           "audit_bound_5", "audit_bound_15", "lemma1_bound", "far_target", "far_others_mean",
           "recall_real", "recall_synth", "precision"]
    s = df.groupby(keys, sort=False)[num].mean().reset_index()
    h = df.groupby(keys, sort=False)[["audit_holds_5", "audit_holds_15"]].mean().reset_index()
    return s.merge(h, on=keys)


# =============================================================================
# E2  Adaptive attacks on cumulative allocation
# =============================================================================

def _episodes(T):
    ep = {}
    for u, D in T[T["y"] == 1].groupby("user"):
        ep[u] = D.index.to_numpy()
    return ep


def shift_episodes(sd, group_col, rec=None, mode="adaptive", seed=0):
    """Move each insider's malicious episode to other active days of the same
    insider (scores and labels are swapped with the benign days they replace).
    mode='adaptive': start where the insider's group loses the most alerts under
    FAIR-IRS (oracle knowledge of the unattacked allocation, an upper bound on
    attacker knowledge); mode='random': uniformly random start."""
    rng = np.random.default_rng(seed)
    T = sd.test.copy()
    cut = None
    if rec is not None:
        r = rec.copy()
        r["cut"] = (r["K"] - r["k"]).clip(lower=0)
        cut = r.set_index(["day", "group"])["cut"].to_dict()
    for u, mal_idx in _episodes(T).items():
        rows = T.index[T["user"] == u].to_numpy()          # sorted by day
        L = len(mal_idx)
        if len(rows) <= L:
            continue
        starts = np.arange(0, len(rows) - L + 1)
        if mode == "adaptive":
            gu = T.loc[rows, group_col].to_numpy()
            dd = T.loc[rows, "day"].to_numpy()
            c = np.array([cut.get((dd[j], gu[j]), 0) for j in range(len(rows))], float)
            win = np.convolve(c, np.ones(L), "valid")
            best = np.flatnonzero(win == win.max())
            st = int(rng.choice(best))
        else:
            st = int(rng.choice(starts))
        new = rows[st:st + L]
        old_scores = T.loc[mal_idx, "score"].to_numpy().copy()
        moved_out = [i for i in new if i not in set(mal_idx)]
        freed = [i for i in mal_idx if i not in set(new)]
        ben_scores = T.loc[moved_out, "score"].to_numpy().copy()
        T.loc[mal_idx, "y"] = 0
        T.loc[freed, "score"] = ben_scores
        T.loc[new, "score"] = old_scores
        T.loc[new, "y"] = 1
    return _with_test(sd, T)


def flood_group(sd, group_col, n_acc=3, window=10, seed=0):
    """Accomplices: n_acc benign colleagues of each insider (same group) make
    themselves maximally suspicious on their active days in the `window` days
    before the insider's first malicious day, so that the group's cumulative
    alert count rises before the attack."""
    rng = np.random.default_rng(seed)
    T = sd.test.copy()
    ins = set(T.loc[T["y"] == 1, "user"])
    acc_rows = set()
    for u, D in T[T["y"] == 1].groupby("user"):
        d0 = D["day"].min()
        gu = D[group_col].iloc[0]
        lo = d0 - pd.Timedelta(days=window)
        pool = T[(T[group_col] == gu) & (T["day"] >= lo) & (T["day"] < d0) & ~T["user"].isin(ins)]
        cands = pool["user"].unique()
        if len(cands) == 0:
            continue
        acc = rng.choice(cands, size=min(n_acc, len(cands)), replace=False)
        acc_rows |= set(pool.index[pool["user"].isin(acc)])
    acc_rows = np.array(sorted(acc_rows), int)
    T.loc[acc_rows, "score"] = 1.0 + rng.uniform(0, 1e-3, len(acc_rows))
    return _with_test(sd, T), len(acc_rows)


def e2_attacks(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
               detectors=("GBM", "LSTM", "IF"), seeds=(0, 1, 2, 3, 4)):
    rows = []
    pols = [("B0 global", dict(delta=1.0)), ("B4 FAIR-IRS", dict(delta=delta)),
            ("B4 FAIR-IRS, cap 3", dict(delta=delta, cap=3))]
    for det in detectors:
        if det not in sc:
            log(f"[skip] E2 {det}: not in the scored cache")
            continue
        sd0 = sc[det]
        _, rec = simulate_rev(sd0, budget, delta, group_col, record=True)
        scen = [("none", 0, sd0, 0)]
        scen.append(("timing, adaptive", 0, shift_episodes(sd0, group_col, rec, "adaptive", 0), 0))
        for s in seeds:
            scen.append(("timing, random", s, shift_episodes(sd0, group_col, None, "random", s), 0))
        for s in seeds:
            sdf, nacc = flood_group(sd0, group_col, seed=s)
            scen.append(("flooding (3 accomplices, 10 days)", s, sdf, nacc))
        for attack, s, sd, nacc in scen:
            ins = set(sd.test.loc[sd.test["y"] == 1, "user"])
            for pol, kw in pols:
                out = simulate_rev(sd, budget=budget, group_col=group_col, **kw)
                r = recall_of(out, ins)
                m = fi.policy_metrics(out, group_col)
                rows.append(dict(detector=det, attack=attack, seed=s, policy=pol, recall=r[0],
                                 detected=r[1], insiders=r[2], delta_FAR=m["delta_FAR"],
                                 ratio_FAR=m["ratio_FAR"], accomplice_days=nacc))
        log(f"E2 {det} done")
    return pd.DataFrame(rows)


def e2_summary(df):
    g = df.groupby(["detector", "attack", "policy"], sort=False)
    s = g[["recall", "detected", "delta_FAR", "ratio_FAR", "accomplice_days"]].mean()
    s["recall_sd"] = g["recall"].std().fillna(0)
    s["runs"] = g.size()
    return s.reset_index()


# =============================================================================
# E3  Role-contrast sensitivity
# =============================================================================

def make_role_transform(alpha):
    """Feature transform that keeps a fraction alpha of the between-role
    differences in mean (log) activity: alpha=1 reproduces the paper's
    features, alpha=0 removes role-level mean differences entirely."""
    class RoleTransform(fi.FeatureTransform):
        def __init__(self, mode="raw"):
            super().__init__("raw")

        def fit(self, df):
            Z = self._base(df)
            mu = Z.mean(0)
            roles = df["role"].astype(str).to_numpy()
            rm = pd.DataFrame(Z).groupby(roles).mean()
            sh = (1 - alpha) * (rm.to_numpy() - mu)
            sh[:, -1] = 0.0                                   # keep the weekend indicator
            self.shift = dict(zip(rm.index, sh))
            Zs = Z - self._shift(df)
            self.mu = Zs.mean(0)
            self.sd = Zs.std(0) + 1e-6
            return self

        def _shift(self, df):
            zero = np.zeros(len(COLS))
            return np.vstack([self.shift.get(r, zero) for r in df["role"].astype(str).to_numpy()])

        def transform(self, df):
            return (((self._base(df) - self._shift(df)) - self.mu) / self.sd).astype(np.float32)
    COLS = fi.COUNT_FEATURES + ["weekend"]
    return RoleTransform


def e3_role_contrast(ds, budget=0.01, delta=0.002, alphas=(1.0, 0.5, 0.0),
                     groupings=("g_functional_unit", "g_department", "g_role"), seed=0):
    rows = []
    orig = fi.FeatureTransform
    try:
        for a in alphas:
            fi.FeatureTransform = make_role_transform(a)
            for det in [fi.IFDetector(seed), fi.AEDetector(seed), fi.LRDetector(), fi.GBMDetector(seed)]:
                try:
                    sd = fi.run_detector(ds, det)
                except Exception as ex:
                    log(f"[skip] E3 {det.name} alpha={a}: {type(ex).__name__}: {ex}")
                    continue
                if sd is None:
                    continue
                eb = fi.exposure_bias(sd.test)
                for gc in groupings:
                    for pol, dl in [("B0 global", 1.0), ("B4 FAIR-IRS", delta)]:
                        out = fi.simulate_policy(sd, fi.PolicyConfig(budget=budget, delta=dl, group_col=gc))
                        m = fi.policy_metrics(out, gc)
                        rows.append(dict(alpha=a, detector=sd.detector, grouping=gc, policy=pol,
                                         recall=m["recall_insider"], detected=m["insiders_detected"],
                                         delta_FAR=m["delta_FAR"], ratio_FAR=m["ratio_FAR"],
                                         gini=m["gini_burden"], max_F=m["max_alerts_one_benign_user"],
                                         top1=m["top1pct_share"], exposure_bias=eb))
                log(f"E3 alpha={a} {sd.detector} done")
    finally:
        fi.FeatureTransform = orig
    return pd.DataFrame(rows)


# =============================================================================
# E4  Several persistent benign patterns
# =============================================================================

PATTERNS = {
    "after-hours work": ["logon_ah", "file_ah", "http_ah", "email_ah"],
    "removable media": ["dev_n", "dev_ah", "file_n"],
    "external e-mail": ["email_ext", "email_att", "email_kb"],
}


def inject_patterns(sd, per_pattern=10, day_share=0.7, strength=1.0, seed=0):
    """Give `per_pattern` benign employees per pattern a persistent, legitimate
    habit: on a random `day_share` of their test days, each feature of the
    pattern is increased by `strength` times its 99th percentile among benign
    test user-days. The detector re-scores the modified days."""
    if sd.model is None or getattr(sd.model, "sequence", False) or isinstance(sd.model, fi.PerGroupDetector):
        raise ValueError("pattern injection needs a single-model, non-sequence detector")
    rng = np.random.default_rng(seed)
    T = sd.test.copy()
    ins = set(T.loc[T["y"] == 1, "user"])
    ben = [u for u in T["user"].unique() if u not in ins]
    chosen = rng.choice(ben, size=min(len(ben), per_pattern * len(PATTERNS)), replace=False)
    who = {}
    for j, (pname, feats) in enumerate(PATTERNS.items()):
        feats = [f for f in feats if f in T.columns]
        q = T.loc[T["y"] == 0, feats].quantile(0.99).clip(lower=1).to_numpy()
        for u in chosen[j * per_pattern:(j + 1) * per_pattern]:
            rows = T.index[T["user"] == u].to_numpy()
            sel = rows[rng.random(len(rows)) < day_share]
            T.loc[sel, feats] = (T.loc[sel, feats].to_numpy() + strength * q).round()
            who[u] = pname
    T["exposure"] = T[fi.VOLUME_FEATURES].sum(axis=1)
    X = sd.transform.transform(T)
    raw = sd.model.score(X)
    T["raw_score"] = raw
    T["score"] = fi.to_quantile_ref(raw, sd.raw_ref)
    return fi.ScoredData(sd.detector + "-patterns", T, X.copy(), sd.model, sd.transform,
                         sd.raw_ref, sd.train_ref), who


def e4_patterns(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
                detectors=("GBM", "IF", "AE"), seeds=(0, 1, 2)):
    rows = []
    pols = [("B0 global", dict(delta=1.0)), ("B4 FAIR-IRS", dict(delta=delta)),
            ("B5 scheduling, lambda 0.01", dict(delta=delta, schedule=True, lam=0.01)),
            ("B5 scheduling, lambda 0.02", dict(delta=delta, schedule=True, lam=0.02))]
    for det in detectors:
        if det not in sc:
            log(f"[skip] E4 {det}: not in the scored cache")
            continue
        for s in seeds:
            try:
                sd, who = inject_patterns(sc[det], seed=s)
            except Exception as ex:
                log(f"[skip] E4 {det}: {type(ex).__name__}: {ex}")
                break
            for pol, kw in pols:
                out = fi.simulate_policy(sd, fi.PolicyConfig(budget=budget, group_col=group_col, **kw))
                m = fi.policy_metrics(out, group_col)
                ins = set(out.loc[out["y"] == 1, "user"])
                F = out[~out["user"].isin(ins)].groupby("user")["cleared"].sum()
                inj = F[F.index.isin(who.keys())]
                oth = F[~F.index.isin(who.keys())]
                row = dict(detector=det, seed=s, policy=pol, recall=m["recall_insider"],
                           detected=m["insiders_detected"], tpr_day=m["tpr_day"],
                           delta_FAR=m["delta_FAR"], gini=m["gini_burden"], top1=m["top1pct_share"],
                           max_F_all=int(F.max()), mean_F_injected=float(inj.mean()), max_F_injected=int(inj.max()),
                           max_F_others=int(oth.max()), n_F_ge_10=int((F >= 10).sum()),
                           n_alerted_benign=int((F > 0).sum()), false_alerts=int(F.sum()))
                for pname in PATTERNS:
                    us = [u for u, p in who.items() if p == pname]
                    row[f"mean_F_{pname}"] = float(F.reindex(us).fillna(0).mean())
                rows.append(row)
        log(f"E4 {det} done")
    return pd.DataFrame(rows)


def e4_summary(df):
    num = [c for c in df.columns if c not in ("detector", "seed", "policy")]
    return df.groupby(["detector", "policy"], sort=False)[num].mean().reset_index()


# =============================================================================
# F1  Parity on cleared alerts: main results for every detector
# =============================================================================

def f1_cleared(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
               deltas=(1.0, 0.01, 0.005, 0.002, 0.001, 0.0)):
    """Policies with parity on alert rates (B4, as in the paper) and on cleared-alert
    rates (B4c), with delta sweep, score floors and scheduling."""
    rows = []
    dets = [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]
    for det in dets:
        sd = sc[det]
        settings = []
        for dl in deltas:
            for h in ("alerts", "cleared"):
                if dl >= 1 and h == "cleared":
                    continue
                settings.append((h, dl, None, False, 0.0))
        for h in ("alerts", "cleared"):
            for th in (2.0, 1.5, 1.0):
                settings.append((h, delta, th, False, 0.0))
            for lam in (0.01, 0.02):
                settings.append((h, delta, None, True, lam))
        for h, dl, th, sch, lam in settings:
            out = simulate_rev(sd, budget=budget, delta=dl, group_col=group_col, history=h,
                               floor=th, schedule=sch, lam=lam if sch else 0.01)
            m = fi.policy_metrics(out, group_col)
            m.update(detector=det, history=h, delta=dl, floor=th if th else "none",
                     schedule=sch, lam=lam)
            rows.append(m)
        log(f"F1 {det} done")
    return pd.DataFrame(rows)


def f1_groupings(sc, budget=0.01, delta=0.002, groupings=("g_functional_unit", "g_department", "g_role",
                                                          "g_afterhours", "g_exposure")):
    rows = []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        for gc in groupings:
            if gc not in sc[det].test.columns:
                continue
            for h, dl in (("global", 1.0), ("alerts", delta), ("cleared", delta)):
                out = simulate_rev(sc[det], budget=budget, delta=dl, group_col=gc,
                                   history="alerts" if h == "global" else h)
                m = fi.policy_metrics(out, gc)
                m.update(detector=det, grouping=gc, policy=h)
                rows.append(m)
        log(f"F1 groupings {det} done")
    return pd.DataFrame(rows)


def f1_release(sc, release, budget=0.01, delta=0.002, group_col="g_functional_unit"):
    rows = []
    for det in [d for d in ("IF", "AE", "LR", "GBM", "LSTM") if d in sc]:
        for h, dl in (("global", 1.0), ("alerts", delta), ("cleared", delta)):
            out = simulate_rev(sc[det], budget=budget, delta=dl, group_col=group_col,
                               history="alerts" if h == "global" else h)
            m = fi.policy_metrics(out, group_col)
            m.update(release=release, detector=det, policy=h)
            rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# F2  Timing attacks against several policies and attacker knowledge levels
# =============================================================================

def report_targets(out, group_col):
    """Monthly alert rate of each group relative to the mean over groups, as a
    burden report would publish it (without noise: an upper bound on what the
    attacker learns)."""
    o = out.copy()
    o["month"] = o["day"].dt.strftime("%Y-%m")
    ar = o.groupby(["month", group_col])["alert"].mean().unstack()
    return ar.sub(ar.mean(axis=1), axis=0)


def shift_to(sd, group_col, chooser, seed=0):
    """Move each insider's malicious episode; chooser(u, rows, T) returns the
    start position among the insider's active test days."""
    rng = np.random.default_rng(seed)
    T = sd.test.copy()
    for u, mal_idx in _episodes(T).items():
        rows = T.index[T["user"] == u].to_numpy()
        L = len(mal_idx)
        if len(rows) <= L:
            continue
        st = chooser(u, rows, T, L, rng)
        st = int(min(max(st, 0), len(rows) - L))
        new = rows[st:st + L]
        mal_set, new_set = set(mal_idx), set(new)
        old_scores = T.loc[mal_idx, "score"].to_numpy().copy()
        moved_out = [i for i in new if i not in mal_set]
        freed = [i for i in mal_idx if i not in new_set]
        ben_scores = T.loc[moved_out, "score"].to_numpy().copy()
        T.loc[mal_idx, "y"] = 0
        T.loc[freed, "score"] = ben_scores
        T.loc[new, "score"] = old_scores
        T.loc[new, "y"] = 1
    return _with_test(sd, T)


def f2_timing(sc, budget=0.01, delta=0.002, group_col="g_functional_unit",
              detectors=("GBM", "LSTM", "IF"), seeds=(0, 1, 2, 3, 4)):
    policies = [("B0 global", dict(delta=1.0)),
                ("B4 alert parity, delta 0.002", dict(delta=delta)),
                ("B4 alert parity, delta 0.005", dict(delta=0.005)),
                ("B4 alert parity, delta 0.01", dict(delta=0.01)),
                ("B4 alert parity, floor 2", dict(delta=delta, floor=2.0)),
                ("B4c cleared parity, delta 0.002", dict(delta=delta, history="cleared"))]
    rows = []
    for det in detectors:
        if det not in sc:
            log(f"[skip] F2 {det}: not in the scored cache")
            continue
        sd0 = sc[det]
        for pol, kw in policies:
            out0, rec = simulate_rev(sd0, budget=budget, group_col=group_col, record=True, **kw)
            r = rec.copy()
            r["cut"] = (r["K"] - r["k"]).clip(lower=0)
            cut = r.set_index(["day", "group"])["cut"].to_dict()
            rep = report_targets(out0, group_col)

            def oracle(u, rows_, T, L, rng):
                gu = T.loc[rows_, group_col].to_numpy(); dd = T.loc[rows_, "day"].to_numpy()
                c = np.array([cut.get((dd[j], gu[j]), 0) for j in range(len(rows_))], float)
                win = np.convolve(c, np.ones(L), "valid")
                return int(rng.choice(np.flatnonzero(win == win.max())))

            def informed(u, rows_, T, L, rng):
                # start in the month after the published report in which the
                # insider's group was most over-alerted
                gu = T.loc[rows_[0], group_col]
                months = T.loc[rows_, "day"].dt.strftime("%Y-%m").to_numpy()
                if gu not in rep.columns:
                    return int(rng.integers(0, len(rows_) - L + 1))
                ex = rep[gu].dropna()
                order = list(ex.sort_values(ascending=False).index)
                allm = sorted(rep.index)
                for mth in order:
                    nxt = allm.index(mth) + 1
                    if nxt >= len(allm):
                        continue
                    pos = np.flatnonzero(months == allm[nxt])
                    if len(pos) and pos[0] <= len(rows_) - L:
                        return int(pos[0])
                return int(rng.integers(0, len(rows_) - L + 1))

            def rand(u, rows_, T, L, rng):
                return int(rng.integers(0, len(rows_) - L + 1))

            scen = [("none", 0, sd0),
                    ("oracle (knows the allocation)", 0, shift_to(sd0, group_col, oracle, 0)),
                    ("report-informed (monthly group rates)", 0, shift_to(sd0, group_col, informed, 0))]
            scen += [("random", s, shift_to(sd0, group_col, rand, s)) for s in seeds]
            for attack, s, sd in scen:
                out = out0 if attack == "none" else simulate_rev(sd, budget=budget, group_col=group_col, **kw)
                ins = set(sd.test.loc[sd.test["y"] == 1, "user"])
                rr = recall_of(out, ins)
                m = fi.policy_metrics(out, group_col)
                rows.append(dict(detector=det, policy=pol, attack=attack, seed=s, recall=rr[0],
                                 detected=rr[1], insiders=rr[2], delta_FAR=m["delta_FAR"],
                                 ratio_FAR=m["ratio_FAR"], cut_alerts=int(r["cut"].sum())))
        log(f"F2 {det} done")
    return pd.DataFrame(rows)


def f2_summary(df):
    g = df.groupby(["detector", "policy", "attack"], sort=False)
    s = g[["recall", "detected", "delta_FAR", "ratio_FAR", "cut_alerts"]].mean()
    s["detected_sd"] = g["detected"].std().fillna(0)
    s["runs"] = g.size()
    return s.reset_index()
