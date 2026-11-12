"""Experiments on attack realism (revision 9).

M1  Online timing attackers: the insider moves forward day by day, sees only the past, and
    starts the episode the first time its signal crosses a threshold set from the values it
    has seen so far. Hindsight attackers are kept as upper bounds. Behavior is relocated and
    rescored, and each relocated attack is scored under the parity rule and the global policy.
M2  Plausibility of the modified behavior: distance to benign days of the same role and
    structural consistency of the counts, for relocated attack days and farming probes;
    excess loss and evasions as plausibility requirements become stricter.
"""
import time
import numpy as np
import pandas as pd
import fairirs as fi
import fairirs_rev as fr
import fairirs_rev5 as h5
import fairirs_rev6 as j6
import fairirs_rev8 as l8

CF = fi.COUNT_FEATURES


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================
# Plausibility
# =============================================================================

SUBTOTALS = [("logon_ah", ["logon_n"]), ("dev_ah", ["dev_n"]), ("file_ah", ["file_n"]), ("file_usb", ["file_n"]),
             ("email_ah", ["email_n"]), ("email_ext", ["email_n"]), ("email_bcc", ["email_n"]),
             ("http_ah", ["http_n"]), ("http_job", ["http_n"]), ("http_leak", ["http_n"]),
             ("http_cloud", ["http_n"]), ("http_upload", ["http_n"])]
EXT = ["file_exe", "file_zip", "file_doc", "file_pdf", "file_txt", "file_jpg"]


def violations(F):
    """Number of structural violations per row: a subtotal above its total, or file
    extension classes summing to more than the file events."""
    v = np.zeros(len(F), int)
    for sub, tot in SUBTOTALS:
        if sub in F and tot[0] in F:
            v += (F[sub].to_numpy(np.float64) > F[tot[0]].to_numpy(np.float64) + 1e-9).astype(int)
    if all(c in F for c in EXT + ["file_n"]):
        v += (F[EXT].sum(axis=1).to_numpy(np.float64) > F["file_n"].to_numpy(np.float64) + 1e-9).astype(int)
    return v


def repair(T, rows):
    """Make modified rows structurally consistent: each total is raised to at least its
    largest subtotal (and file events to at least the sum of extension classes), as the
    actions that produce a subtotal also produce the total."""
    rows = list(rows)
    if not rows:
        return T
    for sub, tot in SUBTOTALS:
        if sub in T and tot[0] in T:
            T.loc[rows, tot[0]] = np.maximum(T.loc[rows, tot[0]].to_numpy(np.float64), T.loc[rows, sub].to_numpy(np.float64))
    if all(c in T for c in EXT + ["file_n"]):
        T.loc[rows, "file_n"] = np.maximum(T.loc[rows, "file_n"].to_numpy(np.float64), T.loc[rows, EXT].sum(axis=1).to_numpy(np.float64))
    return T


class RepairingRescorer:
    """Repairs structural consistency of the changed rows, then rescores them."""

    def __init__(self, rescorer):
        self.r = rescorer
        self.sd = rescorer.sd

    def __call__(self, T, rows, name=None):
        return self.r(repair(T, rows), rows, name=name)


class Plausibility:
    """Distance of a user-day to its k-th nearest benign day of a non-insider in the same
    role, in the detector's standardized feature space, expressed as the percentile of that
    distance among the role's benign days themselves (leave-one-out)."""

    def __init__(self, sd, k=5, max_ref=4000, role_col="role", min_rows=50, seed=0):
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(seed)
        T = sd.test
        ins = set(T.loc[T["y"] == 1, "user"])
        ben = T[(T["y"] == 0) & ~T["user"].isin(ins)]
        self.tf, self.k, self.role_col = sd.transform, k, role_col
        self.models, self.ref = {}, {}
        groups = list(ben.groupby(role_col)) if role_col in T else []
        allb = ben.sample(min(len(ben), max_ref), random_state=seed)
        for key, D in groups + [("__all__", allb)]:
            if key != "__all__" and len(D) < min_rows:
                continue
            D = D.sample(min(len(D), max_ref), random_state=int(rng.integers(1 << 30)))
            X = self.tf.transform(D)
            nn = NearestNeighbors(n_neighbors=min(k + 1, len(D))).fit(X)
            d, _ = nn.kneighbors(X)
            self.models[key] = nn
            self.ref[key] = np.sort(d[:, -1])               # leave-one-out: first neighbour is the point itself

    def percentile(self, F, raw=False):
        out = np.full(len(F), np.nan)
        keys = F[self.role_col].to_numpy() if self.role_col in F else np.array(["__all__"] * len(F))
        X = self.tf.transform(F)
        for key in np.unique(keys):
            m = keys == key
            kk = key if key in self.models else "__all__"
            nn, ref = self.models[kk], self.ref[kk]
            d, _ = nn.kneighbors(X[m], n_neighbors=min(self.k, len(ref)))
            out[m] = d[:, -1] / np.median(ref) if raw else np.searchsorted(ref, d[:, -1], side="right") / len(ref)
        return out

    def distance(self, F):
        """k-NN distance relative to the median k-NN distance of the role's benign days."""
        return self.percentile(F, raw=True)


# =============================================================================
# M1  Online timing attackers
# =============================================================================

def _online_start(sig, L, burn, q, require_positive=True):
    """First position j >= burn whose signal is at least the q-quantile of the signals seen
    at positions before j (and positive). If the signal never triggers, the insider acts at
    the last possible position. Returns (position, triggered)."""
    last = len(sig) - L
    for j in range(min(burn, last), last + 1):
        h = sig[:j]
        if len(h) == 0:
            continue
        if (not require_positive or sig[j] > 0) and sig[j] >= np.quantile(h, q):
            return j, True
    return last, False


M1_POLICIES = l8.L1_POLICIES
PRIMARY_ONLINE = "online deficit, q=0.9"


def m1_online(sc, rescorers, budget=0.01, delta=0.002, group_col="g_functional_unit",
              detectors=("GBM", "LSTM", "AE"), policies=M1_POLICIES, qs=(0.8, 0.9, 0.95), burn=30,
              noisy_draws=5, rand_draws=10, window=30, eps=1.0, n_boot=5000, plaus=None, release="r4.2"):
    """Returns (summary rows per draw, per-insider rows). Per insider: P0, G0 (detected under
    the parity rule and the global policy without attack), PA, GA (with the attack), the
    relocated days' plausibility percentile and structural violations, and whether the online
    rule triggered."""
    rows, per = [], []
    for det in [d for d in detectors if d in sc and d in rescorers]:
        sd0, R = sc[det], RepairingRescorer(rescorers[det])
        out_g0 = h5.simulate_h(sd0, budget, 1.0, group_col, history="global")
        hit_g0 = h5.insider_hits(out_g0)
        P = plaus.get(det) if plaus else None
        for pol, kw in policies:
            out0, rec = h5.simulate_h(sd0, budget, delta, group_col, record=True, seed=0, **kw)
            hit0 = h5.insider_hits(out0)
            r = rec.copy()
            r["cut"] = (r["K"] - r["k"]).clip(lower=0)
            cut = r.set_index(["day", "group"])["cut"].to_dict()
            obs = {}
            for lab, D in r.sort_values("day").groupby("group"):
                a = D["k"].rolling(window, min_periods=1).sum().shift(1).fillna(0)
                n = D["n"].rolling(window, min_periods=1).sum().shift(1).fillna(1)
                for dd, v in zip(D["day"], (a / n).to_numpy()):
                    obs[(dd, lab)] = v
            rep = fr.report_targets(out0, group_col)
            allm = sorted(rep.index)
            prev_month = {m: (allm[i - 1] if i > 0 else None) for i, m in enumerate(allm)}

            def deficit_map(seed):
                M = h5._monthly_noisy(rec, eps, np.random.default_rng(500 + seed))
                months = sorted(M["month"].unique())
                cumk = M.pivot(index="month", columns="group", values="k").fillna(0).cumsum()
                cumn = M.pivot(index="month", columns="group", values="n").fillna(0).cumsum()
                overall = cumk.sum(axis=1) / cumn.sum(axis=1)
                Rr = r.copy()
                Rr["month"] = pd.to_datetime(Rr["day"]).dt.strftime("%Y-%m")
                est = {}
                for lab, D in Rr.sort_values("day").groupby("group"):
                    D = D.copy()
                    D["prev"] = D["month"].map(lambda mth: months[months.index(mth) - 1] if months.index(mth) > 0 else None)
                    first = D["month"] != D["month"].shift(1)
                    D["k_since"] = D.groupby("month")["k"].cumsum().shift(1).fillna(0)
                    D.loc[first, "k_since"] = 0
                    D["n_since"] = D.groupby("month")["n"].cumsum().shift(1).fillna(0)
                    D.loc[first, "n_since"] = 0
                    for dd, pm, ks, ns in zip(D["day"], D["prev"], D["k_since"], D["n_since"]):
                        if pm is None or pd.isna(pm) or lab not in cumk.columns:
                            est[(dd, lab)] = 0.0
                            continue
                        ck, cn = cumk.loc[pm, lab] + ks, cumn.loc[pm, lab] + ns
                        est[(dd, lab)] = ck / max(cn, 1) - overall.loc[pm]
                return est

            def signal(kind, rows_, T, est=None):
                gu = T.loc[rows_, group_col].to_numpy()
                dd = T.loc[rows_, "day"].to_numpy()
                if kind == "deficit":
                    return np.array([est.get((dd[j], gu[j]), 0.0) for j in range(len(rows_))], float)
                if kind == "colleagues":
                    return np.array([obs.get((dd[j], gu[j]), 0.0) for j in range(len(rows_))], float)
                if kind == "reports":
                    mon = pd.to_datetime(T.loc[rows_, "day"]).dt.strftime("%Y-%m").to_numpy()
                    g = gu[0]
                    out = np.zeros(len(rows_))
                    if g in rep.columns:
                        for j, m in enumerate(mon):
                            pm = prev_month.get(m)
                            if pm is not None and not pd.isna(rep.loc[pm, g]):
                                out[j] = rep.loc[pm, g]
                    return out
                if kind == "cut":
                    return np.array([cut.get((dd[j], gu[j]), 0) for j in range(len(rows_))], float)
                raise ValueError(kind)

            trig = {}

            def make(kind, mode, q=None, est=None, name=None):
                def fn(u, rows_, T, L, rng):
                    s = signal(kind, rows_, T, est)
                    if mode == "hindsight":
                        if kind == "cut":
                            s = np.convolve(s, np.ones(L), "valid")
                        else:
                            s = s[:len(rows_) - L + 1]
                        trig[u] = True
                        return int(rng.choice(np.flatnonzero(s == s.max())))
                    j, ok = _online_start(s, L, burn, q)
                    trig[u] = ok
                    return j
                return fn

            def rand(u, rows_, T, L, rng):
                trig[u] = True
                return int(rng.integers(0, len(rows_) - L + 1))

            attackers = [("oracle: knows allocation (hindsight)", lambda dr: make("cut", "hindsight"), 3),
                         ("hindsight deficit", lambda dr: make("deficit", "hindsight", est=deficit_map(dr)), noisy_draws)]
            for q in qs:
                attackers.append((f"online deficit, q={q}", lambda dr, q=q: make("deficit", "online", q, est=deficit_map(dr)), noisy_draws))
            attackers += [("online colleagues, q=0.9", lambda dr: make("colleagues", "online", 0.9), 1),
                          ("online reports, q=0.9", lambda dr: make("reports", "online", 0.9), 1),
                          ("random", lambda dr: rand, rand_draws)]
            for name, factory, nd in attackers:
                for dr in range(nd):
                    trig.clear()
                    fn = factory(dr)
                    sd = l8.relocate_behavior(sd0, group_col, fn, dr, R)
                    outP = h5.simulate_h(sd, budget, delta, group_col, seed=dr + 1, **kw)
                    outG = h5.simulate_h(sd, budget, 1.0, group_col, history="global")
                    hP, hG = h5.insider_hits(outP), h5.insider_hits(outG)
                    D = pd.concat([hit0.rename("P0"), hit_g0.rename("G0"), hP.rename("PA"), hG.rename("GA")], axis=1).fillna(0)
                    d_ = ((D["PA"] - D["GA"]) - (D["P0"] - D["G0"])).to_numpy()
                    mal = sd.test[sd.test["y"] == 1]
                    pl = pd.Series(P.distance(mal), index=mal["user"].to_numpy()).groupby(level=0).max() if P else pd.Series(dtype=float)
                    vi = pd.Series(violations(mal), index=mal["user"].to_numpy()).groupby(level=0).sum()
                    for u, row in D.iterrows():
                        per.append(dict(release=release, detector=det, policy=pol, attack=name, draw=dr, user=u,
                                        P0=int(row.P0), G0=int(row.G0), PA=int(row.PA), GA=int(row.GA),
                                        d=float((row.PA - row.GA) - (row.P0 - row.G0)),
                                        triggered=bool(trig.get(u, True)),
                                        plaus_max=float(pl.get(u, np.nan)), violations=int(vi.get(u, 0))))
                    bs = h5._boot_mean(d_, n_boot, seed=dr)
                    rows.append(dict(release=release, detector=det, policy=pol, attack=name, draw=dr, insiders=len(D),
                                     none_P=int(D["P0"].sum()), none_G=int(D["G0"].sum()),
                                     attacked_P=int(D["PA"].sum()), attacked_G=int(D["GA"].sum()),
                                     excess_loss=float(-d_.sum()), excess_lo=float(-np.percentile(bs, 97.5) * len(D)),
                                     excess_hi=float(-np.percentile(bs, 2.5) * len(D)),
                                     triggered_share=float(np.mean(list(trig.values()))) if trig else np.nan))
            log(f"M1 {release} {det} {pol} done")
    return pd.DataFrame(rows), pd.DataFrame(per)


def original_plausibility(sc, plaus, release="r4.2"):
    """Plausibility percentile and violations of the original (unmoved) malicious days, per
    insider, the reference for relocated attacks."""
    rows = []
    for det, P in plaus.items():
        T = sc[det].test
        mal = T[T["y"] == 1]
        pl = pd.Series(P.distance(mal), index=mal["user"].to_numpy()).groupby(level=0).max()
        vi = pd.Series(violations(mal), index=mal["user"].to_numpy()).groupby(level=0).sum()
        ben = T[T["y"] == 0]
        for u in pl.index:
            rows.append(dict(release=release, detector=det, user=u, plaus_max=float(pl[u]), violations=int(vi[u])))
        rows.append(dict(release=release, detector=det, user="__benign_violation_rate__",
                         plaus_max=np.nan, violations=float((violations(ben) > 0).mean())))
    return pd.DataFrame(rows)


def m1_constrained(per, orig, levels=("none", "no violations beyond original", "within original 100%",
                                      "within original 90%", "within original 75%", "within original 50%"),
                   n_boot=5000, seed=0):
    """Excess loss when only insiders whose relocated attack is plausible may move: an
    insider whose relocation fails the requirement keeps the original timing (contribution 0).
    'within original x%': the relocated days' largest plausibility percentile does not exceed
    the x-th percentile of the same quantity over the original attacks (same release and
    detector); violations: no more structural violations than the insider's original days."""
    rng = np.random.default_rng(seed)
    out = []
    O = orig[orig["user"] != "__benign_violation_rate__"]
    for (rel, det, pol, atk), D in per.groupby(["release", "detector", "policy", "attack"], sort=False):
        o = O[(O["release"] == rel) & (O["detector"] == det)].set_index("user")
        if o.empty:
            continue
        for lev in levels:
            ok = np.ones(len(D), bool)
            if lev == "no violations beyond original":
                ok = D["violations"].to_numpy() <= o["violations"].reindex(D["user"]).fillna(0).to_numpy()
            elif lev.startswith("within original"):
                x = float(lev.split()[-1].rstrip("%")) / 100
                thr = np.nanquantile(o["plaus_max"].to_numpy(), x)
                ok = (D["plaus_max"].to_numpy() <= thr) & (D["violations"].to_numpy() <= o["violations"].reindex(D["user"]).fillna(0).to_numpy())
            dd = np.where(ok, D["d"].to_numpy(), 0.0)
            M = pd.DataFrame(dict(user=D["user"].to_numpy(), d=dd)).groupby("user")["d"].mean()
            X = -M.to_numpy()
            idx = rng.integers(0, len(X), size=(n_boot, len(X)))
            B = X[idx].sum(1)
            out.append(dict(release=rel, detector=det, policy=pol, attack=atk, constraint=lev,
                            admissible_share=float(ok.mean()), excess=float(X.sum()),
                            lo=float(np.percentile(B, 2.5)), hi=float(np.percentile(B, 97.5))))
    return pd.DataFrame(out)


def m1_decomposition(per):
    """2 x 2 components and per-insider outcome classes, averaged over draws."""
    g = per.groupby(["release", "detector", "policy", "attack", "draw"], sort=False)
    S = g[["P0", "G0", "PA", "GA"]].sum()
    lost_mech = ((per["GA"] == 1) & (per["PA"] == 0)).astype(int)
    lost_both = ((per["GA"] == 0) & (per["PA"] == 0) & ((per["G0"] == 1) | (per["P0"] == 1))).astype(int)
    tmp = per.assign(lost_by_mechanism_only=lost_mech, lost_by_both=lost_both,
                     gained_by_mechanism_only=((per["GA"] == 0) & (per["PA"] == 1)).astype(int))
    S2 = tmp.groupby(["release", "detector", "policy", "attack", "draw"], sort=False)[
        ["lost_by_mechanism_only", "lost_by_both", "gained_by_mechanism_only"]].sum()
    S = S.join(S2)
    S["relocation_effect_global"] = S["GA"] - S["G0"]
    S["relocation_effect_policy"] = S["PA"] - S["P0"]
    S["excess_loss"] = S["relocation_effect_global"] - S["relocation_effect_policy"]
    return S.groupby(level=[0, 1, 2, 3], sort=False).mean().reset_index()


def m1_multiplicity(per, primary=PRIMARY_ONLINE, family_prefix="online", n_boot=5000, alpha=0.05, seed=0):
    """Primary online attacker with its own interval; Bonferroni over the online family."""
    rng = np.random.default_rng(seed)
    out = []
    for (rel, det, pol), D in per.groupby(["release", "detector", "policy"], sort=False):
        M = D.groupby(["user", "attack"])["d"].mean().unstack("attack").fillna(0.0)
        atk = list(M.columns)
        X = -M.to_numpy()
        idx = rng.integers(0, X.shape[0], size=(n_boot, X.shape[0]))
        B = X[idx].sum(1)
        fam = [j for j, a in enumerate(atk) if a.startswith(family_prefix)]
        bon = alpha / max(len(fam), 1)
        r = dict(release=rel, detector=det, policy=pol, insiders=X.shape[0])
        for j, a in enumerate(atk):
            r[f"{a} | excess"] = float(X[:, j].sum())
            r[f"{a} | lo"] = float(np.percentile(B[:, j], 100 * alpha / 2))
            r[f"{a} | hi"] = float(np.percentile(B[:, j], 100 * (1 - alpha / 2)))
            if j in fam:
                r[f"{a} | bonferroni_lo"] = float(np.percentile(B[:, j], 100 * bon / 2))
        out.append(r)
    return pd.DataFrame(out)


# =============================================================================
# M2  Plausibility of farming probes
# =============================================================================

M2_POLICIES = [("B0 global", dict(history="global")),
               ("B0 + scheduling", dict(history="global", schedule=True)),
               ("B0 + cooldown 30 days", dict(history="global", cooldown=30)),
               ("B0 + cooldown 30 days, no override", dict(history="global", cooldown=30, tau_ov=1.01))]


def m2_farming(sc, rescorers, plaus, budget=0.01, delta=0.002, group_col="g_functional_unit",
               detectors=("GBM", "AE", "LR", "IF"), grid=(("own", 1), ("own", 3), ("generic", 1), ("generic", 3)),
               levels=(1.0, 0.9, 0.75, 0.5), release="r4.2"):
    """Per farming insider: evaded or not, the largest distance of its probes to benign days of
    its role, and structural violations. A probe must be alerted to work, so it is compared with
    the benign days that the detector itself ranks in its top 1% (the unusual but innocent days
    analysts already see and clear): at level x, a farmer may farm only if its probes are no
    farther from ordinary benign days than the x-quantile of those innocent alerted days and
    violate no count constraint; otherwise it does not farm (evasion counted as 0). Level 'own':
    probes no farther than the insider's own most unusual benign day."""
    per, summ, alerted_ref_by = [], [], {}
    for det in [d for d in detectors if d in sc and d in rescorers and d in plaus]:
        sd, R, P = sc[det], RepairingRescorer(rescorers[det]), plaus[det]
        T0 = sd.test
        ins0 = set(T0.loc[T0["y"] == 1, "user"])
        Bn = T0[(T0["y"] == 0) & ~T0["user"].isin(ins0)]
        Bn = Bn[Bn["score"] >= Bn["score"].quantile(0.99)]
        Bn = Bn.sample(min(len(Bn), 3000), random_state=0)
        alerted_ref = np.sort(P.distance(Bn))
        alerted_ref_by[det] = alerted_ref
        base = {}
        for pol, kw in M2_POLICIES:
            if kw.get("schedule") and sd.phi is None:
                continue
            o = h5.simulate_h(sd, budget, delta, group_col, **kw)
            base[pol] = o[o["y"] == 1].groupby("user")["alert"].any()
        for knowledge, n in grid:
            sdf, farmers, probes, _ = l8.farm(sd, R, knowledge, n, 1.0)
            if not farmers:
                continue
            Tp = sdf.test.loc[probes]
            pp = pd.Series(P.percentile(Tp), index=Tp["user"].to_numpy()).groupby(level=0).max()
            pd_ = pd.Series(P.distance(Tp), index=Tp["user"].to_numpy()).groupby(level=0).max()
            own = sdf.test[sdf.test["user"].isin(farmers) & (sdf.test["y"] == 0)].drop(index=probes)
            od = pd.Series(P.distance(own), index=own["user"].to_numpy()).groupby(level=0).max()
            vv = pd.Series(violations(Tp), index=Tp["user"].to_numpy()).groupby(level=0).sum()
            for pol, kw in M2_POLICIES:
                if pol not in base:
                    continue
                o = h5.simulate_h(sdf, budget, delta, group_col, fc_col="fc", **kw)
                att = o[(o["y"] == 1) & o["user"].isin(farmers)].groupby("user")["alert"].any()
                pr = o.loc[probes]
                exposed = set(pr.loc[pr["alert"] & ~pr["cleared"], "user"])
                for u in farmers:
                    a0 = bool(base[pol].get(u, False))
                    a1 = bool(att.get(u, False)) or (u in exposed)
                    per.append(dict(release=release, detector=det, knowledge=knowledge, n_probe=n, policy=pol, user=u,
                                    detected_no_farming=a0, evaded=a0 and not a1,
                                    probe_plaus_max=float(pp.get(u, np.nan)), probe_distance=float(pd_.get(u, np.nan)),
                                    own_benign_max_distance=float(od.get(u, np.nan)), probe_violations=int(vv.get(u, 0))))
        log(f"M2 {release} {det} done")
    per = pd.DataFrame(per)
    for (rel, det, k, n, pol), D in per.groupby(["release", "detector", "knowledge", "n_probe", "policy"], sort=False):
        for lev in list(levels) + ["own"]:
            if lev == "own":
                ok = (D["probe_distance"] <= D["own_benign_max_distance"]) & (D["probe_violations"] == 0)
            else:
                thr = np.quantile(alerted_ref_by[det], lev)
                ok = (D["probe_distance"] <= thr) & (D["probe_violations"] == 0)
            summ.append(dict(release=rel, detector=det, knowledge=k, n_probe=n, policy=pol, level=str(lev),
                             farmers=len(D), admissible=int(np.sum(ok)), detected_no_farming=int(D["detected_no_farming"].sum()),
                             evaded=int((D["evaded"] & ok).sum()), evaded_unconstrained=int(D["evaded"].sum())))
    return per, pd.DataFrame(summ)
