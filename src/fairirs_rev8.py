"""Security experiments with behavioral relocation and rescoring.

L0  Check that rescoring reproduces the cached scores, and compare score-preserving with
    behavioral relocation under the global policy (random timing)
L1  Timing attacks against parity with behavioral relocation and rescoring
L2  Multiplicity: prespecified primary attacker, descriptive worst case, joint bootstrap of the maximum
L3  Realistic farming: attacker knowledge, number of probes, clearance probability, feasibility cap
L4  Score- and group-dependent verdict errors, and inconclusive verdicts
L5  Social cost of burden: break-even ratio of repeat-alert cost to first-alert cost
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev as fr
import fairirs_rev5 as h5
import fairirs_rev6 as j6
import fairirs_rev7 as k7

CF = fi.COUNT_FEATURES
VF = fi.VOLUME_FEATURES


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================
# Rescoring modified behavior with the saved models
# =============================================================================

def training_context(features_path, split_day):
    """Training-period feature rows (all users), needed so that the LSTM sees the same
    history before the test period as when the cached scores were computed."""
    F = pd.read_pickle(features_path)
    F = F[(F["exposure"] > 0) & (F["day"] < split_day)]
    return F[["user", "day"] + CF + ["weekend"]].copy()


class Rescorer:
    """Rescores test rows whose features were changed, with the detector that produced the
    cached scores. Unchanged rows keep their raw score; all rows are then put back on the
    causal quantile scale of the benign training-period scores."""

    def __init__(self, sd, context=None):
        self.sd = sd
        self.seq = bool(getattr(sd.model, "sequence", False))
        if isinstance(sd.model, fi.PerGroupDetector) or sd.model is None:
            raise ValueError(f"{sd.detector}: rescoring needs a single saved model")
        if self.seq and context is None:
            raise ValueError(f"{sd.detector}: the LSTM needs the training-period features")
        self.context = context

    def raw(self, T, rows):
        rows = np.asarray(sorted(set(int(r) for r in rows)), int)
        out = T["raw_score"].to_numpy(np.float64).copy()
        if len(rows) == 0:
            return out
        tf, model = self.sd.transform, self.sd.model
        if not self.seq:
            out[rows] = model.score(tf.transform(T.iloc[rows]))
            return out
        users = T["user"].iloc[rows].unique()
        Tu = T[T["user"].isin(users)]
        C = self.context[self.context["user"].isin(users)]
        cols = ["user", "day"] + CF + ["weekend"]
        Fu = pd.concat([C[cols].assign(_t=-1), Tu[cols].assign(_t=Tu.index.to_numpy())], ignore_index=True)
        r = model.score_seq(Fu, tf.transform(Fu))
        t = Fu["_t"].to_numpy()
        m = t >= 0
        out[t[m]] = r[m]
        return out

    def __call__(self, T, rows, name=None):
        T = T.copy()
        rows = sorted(set(int(r) for r in rows))
        if rows:
            T.loc[rows, "exposure"] = T.loc[rows, VF].sum(axis=1)
        raw = self.raw(T, rows)
        T["raw_score"] = raw
        T["score"] = fi.to_quantile_ref(raw, self.sd.raw_ref)
        phi = self.sd.phi
        if phi is not None and rows:
            phi = phi.copy()
            tf = self.sd.transform
            phi[rows] = phi[rows] + (tf.transform(T.loc[rows]) - tf.transform(self.sd.test.loc[rows]))
        return fi.ScoredData(name or self.sd.detector, T, phi, self.sd.model, self.sd.transform,
                             self.sd.raw_ref, self.sd.train_ref)


def _float_features(T):
    T = T.copy()
    for c in CF + ["exposure", "raw_score"]:
        if c in T:
            T[c] = T[c].astype(np.float64)
    return T


def relocate_behavior(sd, group_col, chooser, seed, rescorer):
    """Behavioral timing attack. Each insider keeps the malicious actions of the episode but
    performs them on other days: the malicious increment over the insider's typical benign
    day, (x_malicious - median benign)+, is added to the benign activity of the destination
    days, and the vacated days become typical benign days. Exposure is recomputed and the
    saved detector rescores the modified history."""
    rng = np.random.default_rng(seed)
    T = _float_features(sd.test)
    changed = []
    for u, mal_idx in fr._episodes(T).items():
        rows = T.index[T["user"] == u].to_numpy()
        L = len(mal_idx)
        if len(rows) <= L:
            continue
        st = chooser(u, rows, T, L, rng)
        st = int(min(max(st, 0), len(rows) - L))
        new = rows[st:st + L]
        ben = T.loc[rows][T.loc[rows, "y"] == 0]
        base = np.round(ben[CF].median().to_numpy(np.float64)) if len(ben) else np.zeros(len(CF))
        inc = np.clip(T.loc[mal_idx, CF].to_numpy(np.float64) - base, 0, None)
        scen = T.loc[mal_idx[0], "scenario"]
        T.loc[mal_idx, CF] = base
        T.loc[mal_idx, "y"] = 0
        T.loc[mal_idx, "scenario"] = ""
        T.loc[new, CF] = T.loc[new, CF].to_numpy(np.float64) + inc
        T.loc[new, "y"] = 1
        T.loc[new, "scenario"] = scen
        changed.extend(list(mal_idx) + list(new))
    T["y"] = T["y"].astype(np.int8)
    return rescorer(T, changed)


def make_shifter(rescorers):
    def shifter(det, sd0, group_col, fn, dr):
        return relocate_behavior(sd0, group_col, fn, dr, rescorers[det])
    return shifter


# =============================================================================
# L0  Checks
# =============================================================================

def l0_checks(sc, rescorers, budget=0.01, group_col="g_functional_unit", draws=5, release="r4.2"):
    rows = []
    for det, R in rescorers.items():
        sd = sc[det]
        same = R(_float_features(sd.test), [])
        moved_rows = sorted(set(int(i) for i in sd.test.index[sd.test["y"] == 1]))[:200]
        again = R(_float_features(sd.test), moved_rows)
        out0 = h5.simulate_h(sd, budget, 1.0, group_col, history="global")
        for dr in range(draws):
            rand = lambda u, rows_, T, L, rng: int(rng.integers(0, len(rows_) - L + 1))
            sp = fr.shift_to(sd, group_col, rand, dr)
            bh = relocate_behavior(sd, group_col, rand, dr, R)
            o_sp = h5.simulate_h(sp, budget, 1.0, group_col, history="global")
            o_bh = h5.simulate_h(bh, budget, 1.0, group_col, history="global")
            rows.append(dict(release=release, detector=det, draw=dr,
                             max_abs_diff_no_change=float(np.abs(same.test["score"] - sd.test["score"]).max()),
                             max_abs_diff_rescore_unchanged=float(np.abs(again.test["raw_score"] - sd.test["raw_score"]).max()),
                             insiders_detected_original=int(h5.insider_hits(out0).sum()),
                             insiders_detected_score_preserving=int(h5.insider_hits(o_sp).sum()),
                             insiders_detected_behavioral=int(h5.insider_hits(o_bh).sum()),
                             mean_malicious_score_original=float(sd.test.loc[sd.test["y"] == 1, "score"].mean()),
                             mean_malicious_score_behavioral=float(bh.test.loc[bh.test["y"] == 1, "score"].mean()),
                             share_malicious_above_0999_original=float((sd.test.loc[sd.test["y"] == 1, "score"] >= 0.999).mean()),
                             share_malicious_above_0999_behavioral=float((bh.test.loc[bh.test["y"] == 1, "score"] >= 0.999).mean())))
        log(f"L0 {release} {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# L1/L2  Timing attacks with rescoring, and multiplicity
# =============================================================================

L1_POLICIES = [("B4 label-free", dict(history="alerts")),
               ("B4c outcome-aware", dict(history="cleared")),
               ("B4c outcome-aware, 7-day verdict delay", dict(history="cleared", delay=7)),
               j6.DEFAULT]
PRIMARY = "estimates deficit"


def l1_attacks(sc, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit",
               detectors=("GBM", "LSTM", "AE"), det_draws=5, rand_draws=10, n_boot=5000, release="r4.2"):
    collect = []
    dets = tuple(d for d in detectors if d in sc and d in rescorers)
    df = h5.i7_attacks(sc, budget, delta, group_col, detectors=dets, det_draws=det_draws, rand_draws=rand_draws,
                       n_boot=n_boot, policies=L1_POLICIES, shifter=make_shifter(rescorers), collect=collect)
    df["release"] = release
    per = pd.DataFrame(collect)
    per["release"] = release
    return df, per


def l2_multiplicity(per, primary=PRIMARY, n_boot=5000, alpha=0.05, seed=0):
    """per: one row per (detector, policy, attack, draw, insider) with d = allocation effect under
    attack minus without. Excess loss of an attacker = -sum over insiders of d averaged over draws."""
    rows = []
    rng = np.random.default_rng(seed)
    for (rel, det, pol), D in per.groupby(["release", "detector", "policy"], sort=False):
        M = D.groupby(["user", "attack"])["d"].mean().unstack("attack").fillna(0.0)
        atk = list(M.columns)
        X = -M.to_numpy()
        n, A = X.shape
        est = X.sum(0)
        idx = rng.integers(0, n, size=(n_boot, n))
        B = X[idx].sum(1)                                   # n_boot x A, same resample for every attacker
        mx = B.max(1)
        a_lo, a_hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        bon = alpha / max(A, 1)
        w = int(np.argmax(est))
        r = dict(release=rel, detector=det, policy=pol, insiders=n, attackers=A,
                 worst_attacker=atk[w], worst_excess=float(est[w]),
                 joint_max_lo=float(np.percentile(mx, a_lo)), joint_max_hi=float(np.percentile(mx, a_hi)),
                 joint_max_one_sided_lo=float(np.percentile(mx, 100 * alpha)))
        if primary in atk:
            p = atk.index(primary)
            r.update(primary_attacker=primary, primary_excess=float(est[p]),
                     primary_lo=float(np.percentile(B[:, p], a_lo)), primary_hi=float(np.percentile(B[:, p], a_hi)))
        for j, a in enumerate(atk):
            r[f"{a} | excess"] = float(est[j])
            r[f"{a} | bonferroni_lo"] = float(np.percentile(B[:, j], 100 * bon / 2))
            r[f"{a} | bonferroni_hi"] = float(np.percentile(B[:, j], 100 * (1 - bon / 2)))
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# L3  Realistic farming
# =============================================================================

FARM_POLICIES = [("B0 global", dict(history="global")),
                 ("B0 + scheduling", dict(history="global", schedule=True)),
                 ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
                 ("B0 + cooldown 30 days, no override", dict(history="global", cooldown=30, tau_ov=1.01)),
                 ("B11c + cooldown 30 days", dict(allocator="deficit", history="cleared", cooldown=30))]
FARM_GRID = ([(k, n, 1.0) for k in ("own", "scenario", "generic") for n in (1, 3, 5)] +
             [(k, 3, p) for k in ("own", "scenario", "generic") for p in (0.5, 0.8)])


def farm(sd, rescorer, knowledge="own", n_probe=3, p_clear=1.0, alpha=1.0, cap_q=0.99, role_col="role", min_rows=50):
    """Each insider makes the n_probe benign days before the attack resemble it, so that they
    are alerted and cleared first. knowledge: 'own' (the insider's own planned increment),
    'scenario' (mean increment of the other insiders of the same scenario), 'generic' (mean
    increment of all other insiders). Each probe feature is capped at the 99th percentile of
    benign days of non-insiders in the same role (a probe beyond what colleagues do would be
    implausible); a probe alert is confirmed (not cleared) with probability 1 - p_clear."""
    T = _float_features(sd.test)
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
    caps = {}
    if role_col in T:
        for r_, D in ben.groupby(role_col):
            caps[r_] = D[CF].quantile(cap_q).to_numpy(np.float64) if len(D) >= min_rows else gcap
    T["fc"] = np.nan
    changed, farmers, capped = [], [], 0.0
    for u in ins:
        others = [v for v in ins if v != u]
        if knowledge == "own":
            d = inc[u]
        elif knowledge == "scenario":
            same = [v for v in others if scen[v] == scen[u]]
            d = np.mean([inc[v] for v in (same or others)], axis=0) if others else inc[u]
        else:
            d = np.mean([inc[v] for v in others], axis=0) if others else inc[u]
        d0 = mal.loc[mal["user"] == u, "day"].min()
        prior = T[(T["user"] == u) & (T["y"] == 0) & (T["day"] < d0)].sort_values("day").tail(n_probe)
        if prior.empty:
            continue
        idx = prior.index
        cap = caps.get(T.loc[idx[0], role_col], gcap) if role_col in T else gcap
        x = T.loc[idx, CF].to_numpy(np.float64)
        want = np.round(x + alpha * d)
        got = np.maximum(x, np.minimum(want, cap))
        capped += float((got < want).mean())
        T.loc[idx, CF] = got
        T.loc[idx, "fc"] = 1.0 - p_clear
        changed.extend(idx)
        farmers.append(u)
    sdf = rescorer(T, changed, name=sd.detector + "-farm")
    return sdf, farmers, list(changed), capped / max(len(farmers), 1)


def l3_farming(sc, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit",
               detectors=("GBM", "AE", "LR", "IF"), grid=FARM_GRID, policies=FARM_POLICIES, seeds=3, release="r4.2"):
    rows = []
    for det in [d for d in detectors if d in sc and d in rescorers]:
        sd = sc[det]
        base = {}
        for pol, kw in policies:
            if kw.get("schedule") and sd.phi is None:
                continue
            o = h5.simulate_h(sd, budget, delta, group_col, **kw)
            base[pol] = o[o["y"] == 1].groupby("user")["alert"].any()
        for knowledge, n_probe, p_clear in grid:
            sdf, farmers, probes, capped = farm(sd, rescorers[det], knowledge, n_probe, p_clear)
            if not farmers:
                continue
            probe_set = np.zeros(len(sdf.test), bool)
            probe_set[probes] = True
            for pol, kw in policies:
                if pol not in base:
                    continue
                for s in range(seeds if p_clear < 1 else 1):
                    o = h5.simulate_h(sdf, budget, delta, group_col, fc_col="fc", seed=s, **kw)
                    att = o[(o["y"] == 1) & o["user"].isin(farmers)].groupby("user")["alert"].any()
                    pr = o[probe_set]
                    probed = pr[pr["alert"]].groupby("user").size()
                    exposed = pr[pr["alert"] & ~pr["cleared"]].groupby("user").size()
                    a0 = base[pol].reindex(farmers).fillna(False).astype(bool)
                    a1 = att.reindex(farmers).fillna(False).astype(bool)
                    ex = pd.Series(False, index=farmers)
                    ex[ex.index.isin(exposed.index)] = True
                    caught = a1 | ex
                    rows.append(dict(release=release, detector=det, knowledge=knowledge, n_probe=n_probe,
                                     p_clear=p_clear, policy=pol, seed=s, farmers=len(farmers),
                                     probe_features_capped=capped,
                                     probes_alerted_insiders=int(len(probed)),
                                     detected_no_farming=int(a0.sum()), attack_detected_farming=int(a1.sum()),
                                     exposed_by_probe=int(ex.sum()), caught_farming=int(caught.sum()),
                                     evaded=int((a0 & ~caught).sum()), gained=int((~a0 & caught).sum())))
        log(f"L3 {release} {det} done")
    return pd.DataFrame(rows)


def l3_summary(df):
    keys = ["release", "detector", "knowledge", "n_probe", "p_clear", "policy"]
    return df.groupby(keys, sort=False)[["farmers", "probe_features_capped", "probes_alerted_insiders",
                                          "detected_no_farming", "attack_detected_farming", "exposed_by_probe",
                                          "caught_farming", "evaded", "gained"]].mean().reset_index()


# =============================================================================
# L4  Verdict errors and inconclusive verdicts
# =============================================================================

ERR_POLICIES = [("B0 global", dict(history="global")),
                ("B0 + scheduling", dict(history="global", schedule=True)),
                ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
                ("B0 + cooldown 30 days, no override", dict(history="global", cooldown=30, tau_ov=1.01)),
                ("B4c outcome-aware", dict(history="cleared")),
                j6.DEFAULT,
                ("B11c + cooldown 30 days", dict(allocator="deficit", history="cleared", cooldown=30))]


def error_conditions(sd, group_col):
    sizes = sd.test.groupby(group_col)["user"].nunique().sort_values()
    small, large = sizes.index[0], sizes.index[-1]
    return [("no errors", {}),
            ("score-dependent misses (0.30 below 0.999, 0.05 above)", dict(p_miss=0.05, p_miss_low=0.30)),
            ("score-dependent false confirmations (0.01 below 0.999, 0.10 above)", dict(p_fc=0.01, p_fc_high=0.10)),
            (f"group-dependent: smallest unit ({small}) worse", dict(p_miss=0.05, p_fc=0.01, err_group=small, p_miss_g=0.30, p_fc_g=0.05)),
            (f"group-dependent: largest unit ({large}) worse", dict(p_miss=0.05, p_fc=0.01, err_group=large, p_miss_g=0.30, p_fc_g=0.05)),
            ("inconclusive 20%", dict(p_inconclusive=0.2)),
            ("inconclusive 50%", dict(p_inconclusive=0.5)),
            ("score-dependent errors + inconclusive 50%", dict(p_miss=0.05, p_miss_low=0.30, p_fc=0.01, p_fc_high=0.10, p_inconclusive=0.5))]


def l4_errors(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=j6.DETS, seeds=2, release="r4.2"):
    rows = []
    for det in [d for d in detectors if d in sc]:
        sd = sc[det]
        for cond, ekw in error_conditions(sd, group_col):
            for pol, kw in ERR_POLICIES:
                if kw.get("schedule") and sd.phi is None:
                    continue
                for s in range(seeds if ekw else 1):
                    out = h5.simulate_h(sd, budget, delta, group_col, seed=s, **kw, **ekw)
                    m = h5.metrics(out, group_col, release=release, detector=det, policy=pol, condition=cond, seed=s)
                    m["confirmed_alerts_on_insiders"] = int((out["alert"] & ~out["cleared"] & (out["y"] == 1)).sum())
                    rows.append(m)
        log(f"L4 {release} {det} done")
    return pd.DataFrame(rows)


# =============================================================================
# L5  Social cost
# =============================================================================

COST_POLICIES = list(k7.SUPPRESSION) + [j6.DEFAULT, ("B4c outcome-aware", dict(history="cleared"))]


def l5_social_cost(sc, budget=0.01, delta=0.002, group_col="g_functional_unit", detectors=j6.DETS,
                   ratios=(0.1, 0.25, 0.5, 1.0, 2.0), release="r4.2"):
    """C = sum_i [c1 * 1(F_i > 0) + c2 * (F_i - 1)+] over benign employees (c1 = 1).
    Relative to the global policy, C_P - C_G = dN + r dR with r = c2/c1, so the policy is
    cheaper for every r above the break-even r* = -dN/dR when dR < 0 (and so on)."""
    rows = []
    for det in [d for d in detectors if d in sc]:
        sd = sc[det]
        ins = set(sd.test.loc[sd.test["y"] == 1, "user"])
        base = None
        for pol, kw in COST_POLICIES:
            if kw.get("schedule") and sd.phi is None:
                continue
            out = h5.simulate_h(sd, budget, delta, group_col, **kw)
            b = out[(out["y"] == 0) & ~out["user"].isin(ins)]
            F = b.groupby("user")["alert"].sum()
            N, R = int((F > 0).sum()), int((F - 1).clip(lower=0).sum())
            hits = int(h5.insider_hits(out).sum())
            r = dict(release=release, detector=det, policy=pol, alerted_employees=N, repeat_alerts=R,
                     false_alerts=int(F.sum()), max_F=int(F.max()) if len(F) else 0, insiders_detected=hits)
            if pol == "B0 global":
                base = r
            if base is not None:
                dN, dR = N - base["alerted_employees"], R - base["repeat_alerts"]
                r["d_alerted_employees"], r["d_repeat_alerts"] = dN, dR
                r["d_insiders_detected"] = hits - base["insiders_detected"]
                if dR != 0:
                    rs = -dN / dR
                    r["break_even_ratio"] = rs
                    r["cheaper_when"] = "never" if (dR > 0 and dN >= 0) else ("always" if (dR < 0 and dN <= 0) else
                                        (f"c2/c1 > {rs:.3g}" if dR < 0 else f"c2/c1 < {rs:.3g}"))
                else:
                    r["break_even_ratio"] = np.nan
                    r["cheaper_when"] = "always" if dN < 0 else ("never" if dN > 0 else "equal")
                for q in ratios:
                    r[f"cost_vs_global_r{q}"] = dN + q * dR
            rows.append(r)
        log(f"L5 {release} {det} done")
    return pd.DataFrame(rows)
