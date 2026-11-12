"""
fairirs.py -- Experiments for
the FAIR-IRS project: burden and evasion in
history-dependent insider-risk alerting

Pipeline
  1. load CERT release (r4.2 / r6.2 formats) + answers  -> user-day features, labels
  2. groups (LDAP structural groups, behavioural groups, semi-synthetic attribute)
  3. detectors (IF, AE, LR, DP-LR, GBM, per-group IF, plug-in scores)
  4. alert policies (global top-k, quota, FAIR-IRS allocation, + scheduling)
  5. metrics (detection, burden, exposure bias, Lemma 1 check, DP certificates)
  6. simulations (analyst noise, discount farming, exposure injection, kappa sweep)

Only numpy / pandas / scipy / scikit-learn / matplotlib are required.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import spearmanr

# =============================================================================
# 0. Utilities
# =============================================================================

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_files(root, name):
    """Case-insensitive recursive search for a file name."""
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == name.lower():
                out.append(os.path.join(dp, fn))
    return sorted(out)


def find_dir(root, name):
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if d.lower() == name.lower():
                return os.path.join(dp, d)
    return None


DATE_FMT = "%m/%d/%Y %H:%M:%S"
DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}")

# Expected columns when a CERT file has no header (keyed by number of fields)
NOHEADER_SCHEMAS = {
    "logon": {5: ["id", "date", "user", "pc", "activity"]},
    "device": {5: ["id", "date", "user", "pc", "activity"],
               6: ["id", "date", "user", "pc", "file_tree", "activity"]},
    "file": {6: ["id", "date", "user", "pc", "filename", "content"],
             9: ["id", "date", "user", "pc", "filename", "activity",
                 "to_removable_media", "from_removable_media", "content"]},
    "email": {11: ["id", "date", "user", "pc", "to", "cc", "bcc", "from",
                   "size", "attachments", "content"],
              12: ["id", "date", "user", "pc", "to", "cc", "bcc", "from",
                   "activity", "size", "attachments", "content"]},
    "http": {6: ["id", "date", "user", "pc", "url", "content"],
             7: ["id", "date", "user", "pc", "url", "activity", "content"]},
}

# columns we actually need per source (others are skipped while parsing)
NEEDED = {
    "logon": ["date", "user", "pc", "activity"],
    "device": ["date", "user", "activity"],
    "file": ["date", "user", "filename", "activity", "to_removable_media"],
    "email": ["date", "user", "to", "cc", "bcc", "activity", "size", "attachments"],
    "http": ["date", "user", "url", "activity"],
}


def _sniff_columns(path, kind):
    with open(path, "r", errors="ignore", newline="") as f:
        first = next(csv.reader([f.readline()]))
    if first and first[0].strip().lower() == "id":
        return [c.strip().lower() for c in first], 0
    names = NOHEADER_SCHEMAS[kind].get(len(first))
    if names is None:
        raise ValueError(f"Cannot infer schema of {path} ({len(first)} fields)")
    return names, None


def read_chunks(path, kind, chunksize):
    names, header = _sniff_columns(path, kind)
    usecols = [c for c in NEEDED[kind] if c in names]
    kw = dict(usecols=usecols, dtype=str, chunksize=chunksize,
              on_bad_lines="skip", encoding_errors="ignore")
    if header is None:
        kw.update(header=None, names=names)
    else:
        kw.update(header=0)
        # normalise header case
        kw["usecols"] = lambda c: c.strip().lower() in usecols
    for ch in pd.read_csv(path, **kw):
        ch.columns = [c.strip().lower() for c in ch.columns]
        ch["ts"] = pd.to_datetime(ch["date"], format=DATE_FMT, errors="coerce")
        ch = ch.dropna(subset=["ts", "user"])
        ch["day"] = ch["ts"].dt.normalize()
        h = ch["ts"].dt.hour
        ch["ah"] = ((h < 8) | (h >= 18) | (ch["ts"].dt.dayofweek >= 5)).astype(np.int32)
        yield ch


# =============================================================================
# 1. Feature extraction (user-day)
# =============================================================================

JOB_RE = r"job|career|monster|indeed|simplyhired|hire|recruit|linkedin|resume"
LEAK_RE = r"wikileaks|leak"
CLOUD_RE = r"dropbox|drive\.google|box\.com|onedrive|mega\.|wetransfer|4shared|mediafire"

COUNT_FEATURES = [
    "logon_n", "logoff_n", "logon_ah", "pc_n",
    "dev_n", "dev_ah",
    "file_n", "file_ah", "file_exe", "file_zip", "file_doc", "file_pdf",
    "file_txt", "file_jpg", "file_usb",
    "email_n", "email_ah", "email_ext", "email_att", "email_bcc", "email_kb",
    "http_n", "http_ah", "http_job", "http_leak", "http_cloud", "http_upload",
]
VOLUME_FEATURES = ["logon_n", "logoff_n", "dev_n", "file_n", "email_n", "http_n"]


def _agg(df, cols):
    return df.groupby(["user", "day"])[cols].sum()


def feats_logon(ch, pcs_acc):
    a = ch["activity"].str.lower().fillna("")
    d = pd.DataFrame({"user": ch["user"], "day": ch["day"],
                      "logon_n": (a == "logon").astype(np.int32),
                      "logoff_n": (a == "logoff").astype(np.int32),
                      "logon_ah": ((a == "logon") & (ch["ah"] == 1)).astype(np.int32)})
    pcs_acc.append(ch[["user", "day", "pc"]].drop_duplicates())
    return _agg(d, ["logon_n", "logoff_n", "logon_ah"])


def feats_device(ch):
    a = ch["activity"].str.lower().fillna("") if "activity" in ch else pd.Series("connect", index=ch.index)
    c = (a == "connect").astype(np.int32)
    d = pd.DataFrame({"user": ch["user"], "day": ch["day"], "dev_n": c, "dev_ah": c * ch["ah"]})
    return _agg(d, ["dev_n", "dev_ah"])


def feats_file(ch):
    fn = ch["filename"].fillna("").str.lower()
    ext = fn.str.extract(r"\.([a-z0-9]{1,4})$", expand=False).fillna("")
    d = pd.DataFrame({"user": ch["user"], "day": ch["day"],
                      "file_n": 1, "file_ah": ch["ah"],
                      "file_exe": (ext == "exe").astype(np.int32),
                      "file_zip": ext.isin(["zip", "rar", "7z"]).astype(np.int32),
                      "file_doc": ext.isin(["doc", "docx"]).astype(np.int32),
                      "file_pdf": (ext == "pdf").astype(np.int32),
                      "file_txt": (ext == "txt").astype(np.int32),
                      "file_jpg": ext.isin(["jpg", "jpeg", "png"]).astype(np.int32)})
    if "to_removable_media" in ch:
        d["file_usb"] = (ch["to_removable_media"].fillna("").str.lower() == "true").astype(np.int32)
    else:
        d["file_usb"] = 0
    return _agg(d, ["file_n", "file_ah", "file_exe", "file_zip", "file_doc",
                    "file_pdf", "file_txt", "file_jpg", "file_usb"])


def feats_email(ch):
    if "activity" in ch:
        ch = ch[ch["activity"].fillna("").str.lower().str.contains("send")]
    col = lambda c: ch[c].fillna("") if c in ch else pd.Series("", index=ch.index)
    rec = col("to") + ";" + col("cc") + ";" + col("bcc")
    ext = rec.str.contains(r"@(?!dtaa\.com)", regex=True).astype(np.int32)
    att_raw = ch["attachments"].fillna("") if "attachments" in ch else pd.Series("", index=ch.index)
    att_num = pd.to_numeric(att_raw, errors="coerce")
    att_cnt = np.where(att_num.notna(), att_num.fillna(0),
                       np.where(att_raw.str.len() > 0, att_raw.str.count(";") + 1, 0))
    size = pd.to_numeric(ch["size"], errors="coerce").fillna(0) if "size" in ch else 0
    d = pd.DataFrame({"user": ch["user"], "day": ch["day"], "email_n": 1,
                      "email_ah": ch["ah"], "email_ext": ext,
                      "email_att": att_cnt.astype(np.float64),
                      "email_bcc": (col("bcc").str.len() > 0).astype(np.int32),
                      "email_kb": np.asarray(size) / 1000.0})
    return _agg(d, ["email_n", "email_ah", "email_ext", "email_att", "email_bcc", "email_kb"])


def feats_http(ch):
    url = ch["url"].fillna("").str.lower()
    d = pd.DataFrame({"user": ch["user"], "day": ch["day"], "http_n": 1,
                      "http_ah": ch["ah"],
                      "http_job": url.str.contains(JOB_RE, regex=True).astype(np.int32),
                      "http_leak": url.str.contains(LEAK_RE, regex=True).astype(np.int32),
                      "http_cloud": url.str.contains(CLOUD_RE, regex=True).astype(np.int32)})
    if "activity" in ch:
        d["http_upload"] = ch["activity"].fillna("").str.lower().str.contains("upload").astype(np.int32)
    else:
        d["http_upload"] = 0
    return _agg(d, ["http_n", "http_ah", "http_job", "http_leak", "http_cloud", "http_upload"])


def extract_features(release_dir, use_http=True, chunksize=2_000_000):
    """Stream every CERT log file of one release and aggregate per user-day."""
    parts = []
    pcs_acc = []
    sources = ["logon", "device", "file", "email"] + (["http"] if use_http else [])
    for kind in sources:
        paths = find_files(release_dir, f"{kind}.csv")
        if not paths:
            log(f"[warn] {kind}.csv not found under {release_dir}; skipping")
            continue
        path = paths[0]
        log(f"reading {kind} from {path} ({os.path.getsize(path)/1e9:.2f} GB)")
        acc = []
        for i, ch in enumerate(read_chunks(path, kind, chunksize)):
            if kind == "logon":
                acc.append(feats_logon(ch, pcs_acc))
            elif kind == "device":
                acc.append(feats_device(ch))
            elif kind == "file":
                acc.append(feats_file(ch))
            elif kind == "email":
                acc.append(feats_email(ch))
            else:
                acc.append(feats_http(ch))
            if i % 5 == 4:
                log(f"  {kind}: {(i+1)*chunksize:,} rows")
        if acc:
            parts.append(pd.concat(acc).groupby(level=[0, 1]).sum())
    if not parts:
        raise FileNotFoundError(f"No CERT log files (logon.csv, ...) found under {release_dir}")
    F = pd.concat(parts, axis=1).fillna(0)
    if pcs_acc:
        pcs = pd.concat(pcs_acc).drop_duplicates().groupby(["user", "day"]).size().rename("pc_n")
        F = F.join(pcs, how="left")
    for c in COUNT_FEATURES:
        if c not in F:
            F[c] = 0.0
    F = F[COUNT_FEATURES].fillna(0).astype(np.float32).reset_index()
    F["exposure"] = F[VOLUME_FEATURES].sum(axis=1)
    F["weekend"] = (F["day"].dt.dayofweek >= 5).astype(np.int8)
    return F


# =============================================================================
# 2. LDAP, psychometrics, labels
# =============================================================================

def load_ldap(release_dir):
    d = find_dir(release_dir, "LDAP")
    if d is None:
        raise FileNotFoundError(f"No LDAP folder under {release_dir}")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        m = re.search(r"(\d{4}-\d{2})", os.path.basename(p))
        if not m:
            continue
        df = pd.read_csv(p, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        df["month"] = m.group(1)
        rows.append(df)
    L = pd.concat(rows, ignore_index=True).rename(columns={"user_id": "user"})
    for c in ["role", "business_unit", "functional_unit", "department", "team"]:
        if c not in L:
            L[c] = "NA"
        L[c] = L[c].fillna("NA").astype(str)
    return L[["user", "month", "role", "business_unit", "functional_unit", "department", "team"]]


def load_answers(answers_root, version):
    """Return (insiders table, set of malicious (user, day) pairs).

    Uses answers/insiders.csv. Malicious days are taken from the per-insider
    detail files when they can be found; otherwise every day between start
    and end is labelled malicious.
    """
    ins_paths = find_files(answers_root, "insiders.csv")
    if not ins_paths:
        raise FileNotFoundError(f"insiders.csv not found under {answers_root}")
    ins = pd.read_csv(ins_paths[0], dtype=str)
    ins.columns = [c.strip().lower() for c in ins.columns]
    v = float(str(version).lower().replace("r", ""))
    ins = ins[pd.to_numeric(ins["dataset"], errors="coerce").round(1) == round(v, 1)].copy()
    if ins.empty:
        raise ValueError(f"No insiders for dataset {version} in {ins_paths[0]}")
    ins["start"] = pd.to_datetime(ins["start"], format=DATE_FMT, errors="coerce")
    ins["end"] = pd.to_datetime(ins["end"], format=DATE_FMT, errors="coerce")
    ans_dir = os.path.dirname(ins_paths[0])
    # index all detail files once
    detail_index = {}
    for dp, _, fns in os.walk(ans_dir):
        for fn in fns:
            detail_index[fn.lower()] = os.path.join(dp, fn)
    mal = set()
    n_from_details = 0
    for _, r in ins.iterrows():
        p = detail_index.get(str(r.get("details", "")).lower())
        days = set()
        if p:
            with open(p, "r", errors="ignore", newline="") as f:
                for row in csv.reader(f):
                    for fld in row[:4]:
                        if DATE_RE.fullmatch(fld.strip()):
                            days.add(pd.Timestamp(pd.to_datetime(fld.strip(), format=DATE_FMT).normalize()))
                            break
        if days:
            n_from_details += 1
        elif pd.notna(r["start"]) and pd.notna(r["end"]):
            days = set(pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D"))
        for d in days:
            mal.add((r["user"], d))
    log(f"answers: {len(ins)} insiders for r{version}; {n_from_details} with activity-level detail files; "
        f"{len(mal)} malicious user-days")
    return ins, mal


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame            # one row per active user-day
    insiders: pd.DataFrame
    split_day: pd.Timestamp
    group_cols: list = field(default_factory=list)


def build_dataset(name, release_dir, answers_root, version, cache_dir,
                  use_http=True, train_frac=0.5, min_group=20, chunksize=2_000_000,
                  seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"features_{name}.pkl")
    if os.path.exists(fpath):
        log(f"loading cached features {fpath}")
        F = pd.read_pickle(fpath)
    else:
        F = extract_features(release_dir, use_http=use_http, chunksize=chunksize)
        F.to_pickle(fpath)
        log(f"cached features -> {fpath}")
    F = F[F["exposure"] > 0].copy()

    # labels
    ins, mal = load_answers(answers_root, version)
    key = pd.MultiIndex.from_arrays([F["user"], F["day"]])
    F["y"] = key.isin(list(mal)).astype(np.int8)

    # LDAP attributes per month (cached, so raw files can be deleted), fallback to user's modal value
    lpath = os.path.join(cache_dir, f"ldap_{name}.pkl")
    if os.path.exists(lpath):
        L = pd.read_pickle(lpath)
    else:
        L = load_ldap(release_dir)
        L.to_pickle(lpath)
    F["month"] = F["day"].dt.strftime("%Y-%m")
    F = F.merge(L, on=["user", "month"], how="left")
    mode = L.groupby("user")[["role", "business_unit", "functional_unit", "department", "team"]] \
            .agg(lambda s: s.value_counts().index[0])
    for c in mode.columns:
        F[c] = F[c].fillna(F["user"].map(mode[c])).fillna("NA")

    # temporal split
    days = np.sort(F["day"].unique())
    split_day = pd.Timestamp(days[int(len(days) * train_frac)])
    F["train"] = F["day"] < split_day

    # behavioural / derived groups, computed on the training period only
    tr = F[F["train"]]
    ah_share = (tr.groupby("user")["logon_ah"].sum() /
                tr.groupby("user")["logon_n"].sum().clip(lower=1))
    F["g_afterhours"] = np.where(F["user"].map(ah_share).fillna(0) > 0.2, "after-hours", "day")
    expo = tr.groupby("user")["exposure"].mean()
    terc = pd.qcut(expo.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    F["g_exposure"] = F["user"].map(terc).fillna("mid")
    # structural groups, rare values collapsed
    for c in ["functional_unit", "department", "role"]:
        users_per = F.groupby(c)["user"].nunique()
        rare = users_per[users_per < min_group].index
        F["g_" + c] = np.where(F[c].isin(rare), "Other", F[c])
    scen = ins.drop_duplicates("user").set_index("user")["scenario"].astype(str)
    F["scenario"] = F["user"].map(scen).fillna("none")
    F = F.sort_values(["day", "user"]).reset_index(drop=True)
    group_cols = ["g_functional_unit", "g_department", "g_role", "g_afterhours", "g_exposure"]
    return Dataset(name, F, ins, split_day, group_cols)


def add_synthetic_attribute(ds: Dataset, kappa: float, seed=0, col="g_synthA"):
    """Semi-synthetic binary attribute A whose dependence on role is set by kappa.

    Roles are randomly ordered and the first roles covering ~50% of users get
    propensity 1, the rest 0. Then P(A=1 | role) = kappa * p_role + (1-kappa) * 0.5.
    kappa = 0: A independent of role; kappa = 1: A determined by role.
    """
    rng = np.random.default_rng(seed)
    users = ds.df.groupby("user")["role"].agg(lambda s: s.value_counts().index[0])
    roles = users.value_counts()
    order = rng.permutation(roles.index.values)
    cum, p_role, total = 0, {}, len(users)
    for r in order:
        p_role[r] = 1.0 if cum < total / 2 else 0.0
        cum += roles[r]
    p = users.map(p_role).values * kappa + (1 - kappa) * 0.5
    a = (rng.random(len(users)) < p).astype(int)
    amap = dict(zip(users.index, np.where(a == 1, "A=1", "A=0")))
    ds.df[col] = ds.df["user"].map(amap)
    return col


def dataset_summary(ds: Dataset):
    F = ds.df
    te = F[~F["train"]]
    tr = F[F["train"]]
    ins_users = set(F.loc[F["y"] == 1, "user"])
    return {
        "release": ds.name,
        "users": F["user"].nunique(),
        "insiders": len(ins_users),
        "insiders_train": tr.loc[tr["y"] == 1, "user"].nunique(),
        "insiders_test": te.loc[te["y"] == 1, "user"].nunique(),
        "days": F["day"].nunique(),
        "user_days": len(F),
        "malicious_user_days": int(F["y"].sum()),
        "base_rate": F["y"].mean(),
        "roles": F["role"].nunique(),
        "functional_units": F["functional_unit"].nunique(),
        "departments": F["department"].nunique(),
        "split_day": str(ds.split_day.date()),
    }


# =============================================================================
# 3. Feature transforms and detectors
# =============================================================================

class FeatureTransform:
    """'raw': log1p(counts) standardised.  'share': counts divided by daily
    exposure (removes volume -> exposure-normalisation baseline B2)."""

    def __init__(self, mode="raw"):
        self.mode = mode

    def _base(self, df):
        X = df[COUNT_FEATURES].to_numpy(np.float64)
        if self.mode == "share":
            e = df["exposure"].to_numpy(np.float64)[:, None]
            X = X / np.maximum(e, 1.0)
        else:
            X = np.log1p(X)
        return np.hstack([X, df[["weekend"]].to_numpy(np.float64)])

    def fit(self, df):
        X = self._base(df)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, df):
        return ((self._base(df) - self.mu) / self.sd).astype(np.float32)


class IFDetector:
    name = "IF"
    supervised = False

    def __init__(self, seed=0, n_estimators=200, max_train=300_000):
        self.seed, self.n, self.max_train = seed, n_estimators, max_train

    def fit(self, X, y=None):
        from sklearn.ensemble import IsolationForest
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        self.m = IsolationForest(n_estimators=self.n, random_state=self.seed, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class AEDetector:
    """Under-complete MLP autoencoder; score = reconstruction MSE."""
    name = "AE"
    supervised = False

    def __init__(self, seed=0, hidden=(16, 6, 16), max_train=200_000, max_iter=60):
        self.seed, self.hidden, self.max_train, self.max_iter = seed, hidden, max_train, max_iter

    def fit(self, X, y=None):
        from sklearn.neural_network import MLPRegressor
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        self.m = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.max_iter,
                              early_stopping=True, random_state=self.seed).fit(X, X)
        return self

    def score(self, X):
        return ((self.m.predict(X) - X) ** 2).mean(1)


class LRDetector:
    name = "LR"
    supervised = True

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
        return self

    def score(self, X):
        return self.m.decision_function(X)


class GBMDetector:
    name = "GBM"
    supervised = True

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                                learning_rate=0.05, random_state=self.seed).fit(X, y)
        return self

    def score(self, X):
        return self.m.decision_function(X)


# ---------- DP-SGD logistic regression with an RDP accountant ----------

def _rdp_sampled_gaussian(q, sigma, alpha):
    """RDP of the Poisson-subsampled Gaussian mechanism at integer order alpha
    (upper bound of Mironov, Talwar & Zhang 2019, as used by Abadi et al. 2016)."""
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    k = np.arange(alpha + 1)
    logc = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = logc + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(terms) / (alpha - 1))


def dp_epsilon(q, sigma, steps, delta, orders=range(2, 257)):
    return min(steps * _rdp_sampled_gaussian(q, sigma, a) + math.log(1 / delta) / (a - 1)
               for a in orders)


def calibrate_sigma(q, steps, eps, delta):
    lo, hi = 0.3, 200.0
    if dp_epsilon(q, hi, steps, delta) > eps:
        raise ValueError("epsilon too small for these settings")
    for _ in range(60):
        mid = (lo + hi) / 2
        if dp_epsilon(q, mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


class DPLRDetector:
    """Logistic regression trained with DP-SGD (per-example clipping + Gaussian
    noise, Poisson sampling). Guarantee is record-level (one user-day)."""
    supervised = True

    def __init__(self, eps=1.0, delta=None, epochs=10, q=0.01, clip=1.0, lr=0.5, seed=0):
        self.eps, self.delta, self.epochs, self.q = eps, delta, epochs, q
        self.clip, self.lr, self.seed = clip, lr, seed
        self.name = "SGD-LR (no noise)" if eps == np.inf else f"DP-LR(eps={eps:g})"

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        delta = self.delta or 1.0 / (10 * n)
        steps = int(self.epochs / self.q)
        self.sigma = 0.0 if self.eps == np.inf else calibrate_sigma(self.q, steps, self.eps, delta)
        Xb = np.hstack([X, np.ones((n, 1), np.float32)])
        pos = y.mean()
        w = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
        w = w / w.max()                          # weights in (0,1]: clipping still bounds sensitivity
        theta = np.zeros(d + 1)
        # normalise by the expected *weighted* batch size; class counts are treated as public
        exp_batch = self.q * w.sum()
        for _ in range(steps):
            idx = np.nonzero(rng.random(n) < self.q)[0]
            g = np.zeros(d + 1)
            if len(idx):
                xb = Xb[idx]
                p = 1 / (1 + np.exp(-np.clip(xb @ theta, -30, 30)))
                G = (w[idx] * (p - y[idx]))[:, None] * xb
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G / np.maximum(1.0, norms / self.clip)
                g = G.sum(0)
            g = (g + rng.normal(0, self.sigma * self.clip, d + 1)) / exp_batch
            theta -= self.lr * g
        self.theta = theta
        self.eps_spent = np.inf if self.sigma == 0 else dp_epsilon(self.q, self.sigma, steps, delta)
        return self

    def score(self, X):
        return X @ self.theta[:-1] + self.theta[-1]


class LSTMDetector:
    """Sequence detector in the style of Tuor et al. (2017): an LSTM reads each
    employee's sequence of daily feature vectors and predicts the next day;
    the anomaly score of a user-day is the prediction error. Trained on
    benign training-period days only. Requires PyTorch (available on Colab)."""
    name = "LSTM"
    supervised = False
    sequence = True

    def __init__(self, seed=0, hidden=64, epochs=8, batch=64, lr=1e-3):
        self.seed, self.hidden, self.epochs, self.batch, self.lr = seed, hidden, epochs, batch, lr

    @staticmethod
    def _user_index(F):
        codes, uniq = pd.factorize(F["user"])
        order = np.lexsort((F["day"].to_numpy(), codes))
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        return np.split(order, bounds)

    def _pad(self, seqs, X, masks=None):
        L = max(len(q) for q in seqs)
        d = X.shape[1]
        xb = np.zeros((len(seqs), L, d), np.float32)
        mb = np.zeros((len(seqs), L), np.float32)
        for j, q in enumerate(seqs):
            xb[j, :len(q)] = X[q]
            mb[j, :len(q)] = 1.0 if masks is None else masks[q]
        return xb, mb

    def fit_seq(self, F, X, fit_mask, train_mask):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.net = torch.nn.LSTM(d, self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, d)
        params = list(self.net.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        seqs = [q[train_mask[q]] for q in self._user_index(F)]
        seqs = [q for q in seqs if len(q) >= 2]
        w = fit_mask.astype(np.float32)
        for ep in range(self.epochs):
            perm = rng.permutation(len(seqs))
            tot = 0.0
            for b0 in range(0, len(perm), self.batch):
                xb, mb = self._pad([seqs[j] for j in perm[b0:b0 + self.batch]], X, w)
                xb, mb = torch.from_numpy(xb), torch.from_numpy(mb)
                out, _ = self.net(xb[:, :-1])
                err = ((self.head(out) - xb[:, 1:]) ** 2).mean(-1)
                m = mb[:, 1:]
                loss = (err * m).sum() / m.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += float(loss)
            log(f"  LSTM epoch {ep + 1}/{self.epochs}: loss {tot:.3f}")
        return self

    def score_seq(self, F, X):
        import torch
        raw = np.full(len(F), np.nan)
        seqs = self._user_index(F)
        with torch.no_grad():
            for b0 in range(0, len(seqs), self.batch):
                part = seqs[b0:b0 + self.batch]
                xb, _ = self._pad(part, X)
                xt = torch.from_numpy(xb)
                out, _ = self.net(xt[:, :-1])
                err = ((self.head(out) - xt[:, 1:]) ** 2).mean(-1).numpy()
                for j, q in enumerate(part):
                    if len(q) > 1:
                        raw[q[1:]] = err[j, :len(q) - 1]
        first = np.isnan(raw)
        raw[first] = np.nanmedian(raw)          # first active day of a user: no context
        return raw


class PerGroupDetector:
    """Peer-group modelling baseline (B3): one detector per group."""
    supervised = False

    def __init__(self, base_factory, groups_train, groups_score_fn, name="IF-peer"):
        self.base_factory, self.gtr, self.gfn, self.name = base_factory, groups_train, groups_score_fn, name

    def fit(self, X, y=None):
        self.models = {g: self.base_factory().fit(X[self.gtr == g]) for g in np.unique(self.gtr)}
        self.default = self.base_factory().fit(X)
        return self

    def score_groups(self, X, groups):
        s = np.empty(len(X))
        for g in np.unique(groups):
            m = self.models.get(g, self.default)
            idx = groups == g
            s[idx] = m.score(X[idx])
        return s


def to_quantile(scores, reference=None):
    """Map scores to their empirical CDF in [0,1] (detector-agnostic scale)."""
    ref = np.sort(scores if reference is None else reference)
    return np.searchsorted(ref, scores, side="right") / len(ref)


def to_quantile_ref(scores, reference_sorted):
    """Causal score scale: empirical CDF of the BENIGN TRAINING-PERIOD scores.
    Test scores above every training score would all map to 1.0; a tiny
    order-preserving term (< 1e-6) keeps them ranked by raw score."""
    scores = np.asarray(scores, float)
    q = np.searchsorted(reference_sorted, scores, side="right") / len(reference_sorted)
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    return q + 1e-6 * (order + 1) / (len(scores) + 1)


def user_baseline(df, X, train_mask):
    """Mean transformed feature vector per user over the training period."""
    tmp = pd.DataFrame(X[train_mask])
    tmp["user"] = df.loc[train_mask, "user"].values
    return tmp.groupby("user").mean()


def explanations(df, X, base):
    """phi = deviation of a user-day from the user's own training baseline."""
    B = base.reindex(df["user"].values).fillna(0.0).to_numpy(np.float32)
    return X - B


@dataclass
class ScoredData:
    detector: str
    test: pd.DataFrame        # test rows with score columns
    phi: np.ndarray           # explanation vectors (aligned with test rows)
    model: object = None
    transform: object = None
    raw_ref: np.ndarray = None        # sorted benign training-period raw scores
    train_ref: pd.DataFrame = None    # benign training rows: raw score + group columns


def run_detector(ds: Dataset, det, feature_mode="raw", clean_train=True, peer_col=None,
                 phi_mode="population"):
    """phi_mode: 'population' -> explanation = standardised deviation from the
    population (training) baseline; 'self' -> deviation from the user's own
    training baseline."""
    F = ds.df
    tr_mask = F["train"].to_numpy()
    if clean_train and not det.supervised:
        tr_fit = tr_mask & (F["y"].to_numpy() == 0)
    else:
        tr_fit = tr_mask
    ft = FeatureTransform(feature_mode).fit(F[tr_fit])
    X = ft.transform(F)
    y = F["y"].to_numpy()
    if det.supervised and y[tr_fit].sum() == 0:
        log(f"[skip] {det.name}: no malicious user-days in the training period")
        return None
    t0 = time.time()
    tr_ben = tr_mask & (y == 0)
    if getattr(det, "sequence", False):
        det.fit_seq(F, X, tr_fit, tr_mask)
        raw_all = det.score_seq(F, X)
        raw, raw_tr = raw_all[~tr_mask], raw_all[tr_ben]
    elif isinstance(det, PerGroupDetector):
        det.gtr = F.loc[tr_fit, peer_col].to_numpy()
        det.fit(X[tr_fit])
        raw = det.score_groups(X[~tr_mask], F.loc[~tr_mask, peer_col].to_numpy())
        raw_tr = det.score_groups(X[tr_ben], F.loc[tr_ben, peer_col].to_numpy())
    else:
        det.fit(X[tr_fit], y[tr_fit])
        raw = det.score(X[~tr_mask])
        raw_tr = det.score(X[tr_ben])
    ref = np.sort(raw_tr)
    test = F[~tr_mask].copy().reset_index(drop=True)
    test["raw_score"] = raw
    test["score"] = to_quantile_ref(raw, ref)
    gcols = [c for c in F.columns if c.startswith("g_")]
    train_ref = F.loc[tr_ben, ["user"] + gcols].reset_index(drop=True)
    train_ref["raw_score"] = raw_tr
    if phi_mode == "self":
        phi = explanations(test, X[~tr_mask], user_baseline(F, X, tr_mask))
    else:
        phi = X[~tr_mask].copy()
    log(f"{det.name} [{feature_mode}] trained+scored in {time.time()-t0:.0f}s")
    return ScoredData(det.name if feature_mode == "raw" else f"{det.name}-share",
                      test, phi, det, ft, ref, train_ref)


def load_plugin_scores(ds: Dataset, csv_path, name="Plug-in"):
    """Use scores from your own model (e.g., the dynamic risk-indicator model).
    CSV columns: user, day (YYYY-MM-DD), score (higher = riskier)."""
    S = pd.read_csv(csv_path, dtype={"user": str})
    S["day"] = pd.to_datetime(S["day"])
    test = ds.df[~ds.df["train"]].merge(S[["user", "day", "score"]].rename(columns={"score": "raw_score"}),
                                        on=["user", "day"], how="left").reset_index(drop=True)
    test["raw_score"] = test["raw_score"].fillna(test["raw_score"].min())
    test["score"] = to_quantile(test["raw_score"].to_numpy())   # no training scores supplied
    phi = np.zeros((len(test), 1), np.float32)
    return ScoredData(name, test, phi)


# =============================================================================
# 4. Alert policies
# =============================================================================

def largest_remainder(n, k):
    exact = k * n / n.sum()
    kg = np.floor(exact).astype(int)
    rem = k - kg.sum()
    if rem > 0:
        kg[np.argsort(-(exact - kg))[:rem]] += 1
    return np.minimum(kg, n)


def fair_allocate(scores, gids, k, delta):
    """Exact FAIR-IRS allocation (Prop. 2): returns boolean alert mask.

    delta >= 1 -> global top-k;  delta == 'quota' -> proportional quota.
    Returns (mask, used_fallback).
    """
    N = len(scores)
    mask = np.zeros(N, bool)
    if k >= N:
        mask[:] = True
        return mask, False
    if delta != "quota" and delta >= 1:
        mask[np.argpartition(-scores, k - 1)[:k]] = True
        return mask, False
    labels, inv = np.unique(gids, return_inverse=True)
    G = len(labels)
    order = [np.nonzero(inv == g)[0] for g in range(G)]
    order = [o[np.argsort(-scores[o], kind="stable")] for o in order]
    sorted_s = [scores[o] for o in order]
    n = np.array([len(o) for o in order])

    def take(kg):
        for g in range(G):
            mask[order[g][:kg[g]]] = True
        return mask

    if delta == "quota":
        return take(largest_remainder(n, k)), False

    prefix = [np.concatenate([[0.0], np.cumsum(s)]) for s in sorted_s]
    rmax = k / N + 1e-12
    cands = {0.0, rmax}
    for g in range(G):
        for j in range(0, int(math.floor(rmax * n[g])) + 2):
            cands.add(j / n[g])
        for j in range(int(math.ceil(delta * n[g])), int(math.floor((rmax + delta) * n[g])) + 2):
            cands.add(j / n[g] - delta)
    cands = sorted(c for c in cands if 0 <= c <= rmax)
    mids = [(a + b) / 2 for a, b in zip(cands[:-1], cands[1:])]
    best_val, best_kg = -np.inf, None
    for r in cands + mids:
        L = np.ceil(r * n - 1e-9).astype(int)
        U = np.minimum(n, np.floor((r + delta) * n + 1e-9).astype(int))
        if L.sum() > k or U.sum() < k or np.any(L > U):
            continue
        val = sum(prefix[g][L[g]] for g in range(G))
        m = k - L.sum()
        kg = L.copy()
        if m > 0:
            pool = np.concatenate([sorted_s[g][L[g]:U[g]] for g in range(G)])
            pid = np.concatenate([np.full(U[g] - L[g], g) for g in range(G)])
            top = np.argpartition(-pool, m - 1)[:m]
            val += pool[top].sum()
            kg += np.bincount(pid[top], minlength=G)
        if val > best_val:
            best_val, best_kg = val, kg
    if best_kg is None:          # integrality makes this delta infeasible today
        return take(largest_remainder(n, k)), True
    return take(best_kg), False


def fair_allocate_cum(scores, gids, k, delta, A, C):
    """FAIR-IRS allocation with parity on CUMULATIVE alert rates.

    A[g], C[g]: alerts and active user-days of group g before today (dicts).
    Constraint: |(A_g + k_g)/(C_g + n_g) - (A_h + k_h)/(C_h + n_h)| <= delta.
    Same structure as fair_allocate (separable concave objective, box
    constraints for fixed r), hence exact. Returns (mask, used_fallback, labels, kg).
    """
    N = len(scores)
    mask = np.zeros(N, bool)
    labels, inv = np.unique(gids, return_inverse=True)
    G = len(labels)
    order = [np.nonzero(inv == g)[0] for g in range(G)]
    order = [o[np.argsort(-scores[o], kind="stable")] for o in order]
    sorted_s = [scores[o] for o in order]
    n = np.array([len(o) for o in order])
    a = np.array([A.get(l, 0) for l in labels], float)
    D = np.array([C.get(l, 0) for l in labels], float) + n
    k = min(k, N)

    def take(kg):
        for g in range(G):
            mask[order[g][:kg[g]]] = True
        return mask

    if delta != "quota" and delta >= 1:
        mask[np.argpartition(-scores, k - 1)[:k]] = True
        return mask, False
    prefix = [np.concatenate([[0.0], np.cumsum(s)]) for s in sorted_s]
    rmax = (k + a.sum()) / D.sum() + 1e-12
    cands = [np.array([0.0, rmax])]
    for g in range(G):
        jl = np.arange(0, n[g] + 1)
        cands.append((a[g] + jl) / D[g])
        cands.append((a[g] + jl) / D[g] - delta)
    c = np.unique(np.concatenate(cands))
    c = c[(c >= 0) & (c <= rmax)]
    r = np.unique(np.concatenate([c, (c[:-1] + c[1:]) / 2]))
    Lraw = np.ceil(r[:, None] * D - a - 1e-9)
    Uraw = np.floor((r[:, None] + delta) * D - a + 1e-9)
    ok = (Lraw <= n).all(1) & (Uraw >= 0).all(1)
    L = np.clip(Lraw, 0, n).astype(int)
    U = np.clip(Uraw, 0, n).astype(int)
    ok &= (L.sum(1) <= k) & (U.sum(1) >= k) & (L <= U).all(1)
    best_val, best_kg = -np.inf, None
    if ok.any():
        boxes = np.unique(np.hstack([L[ok], U[ok]]), axis=0)
        for row in boxes:
            Lb, Ub = row[:G], row[G:]
            val = sum(prefix[g][Lb[g]] for g in range(G))
            m = k - Lb.sum()
            kg = Lb.copy()
            if m > 0:
                pool = np.concatenate([sorted_s[g][Lb[g]:Ub[g]] for g in range(G)])
                pid = np.concatenate([np.full(Ub[g] - Lb[g], g) for g in range(G)])
                top = np.argpartition(-pool, m - 1)[:m]
                val += pool[top].sum()
                kg += np.bincount(pid[top], minlength=G)
            if val > best_val:
                best_val, best_kg = val, kg
    if best_kg is None:
        # deficit apportionment: give today's alerts to groups furthest below the
        # common cumulative rate (keeps cumulative rates within ~1/C_g)
        target = (k + a.sum()) * D / D.sum() - a
        e = np.clip(target, 0, n)
        if e.sum() <= 0:
            e = n.astype(float)
        exact = k * e / e.sum()
        kg = np.minimum(np.floor(exact).astype(int), n)
        rem = k - kg.sum()
        for g in np.argsort(-(exact - kg)):
            if rem <= 0:
                break
            if kg[g] < n[g]:
                kg[g] += 1
                rem -= 1
        while rem > 0:                          # capacity left elsewhere
            g = int(np.argmax(n - kg))
            kg[g] += 1
            rem -= 1
        return take(kg), True
    return take(best_kg), False


@dataclass
class PolicyConfig:
    budget: float = 0.01          # alerts per day as a fraction of active users
    delta: object = 1.0           # 1.0 = global top-k, 'quota', or a float in [0,1)
    parity: str = "cumulative"    # 'cumulative' (default) or 'daily' rates
    group_col: str = "g_functional_unit"
    schedule: bool = False
    lam: float = 0.01             # score discount per cleared similar alert
    gamma: float = 0.8            # explanation-similarity gate
    tau_ov: float = 0.999         # override threshold (score quantile)
    m_cap: int = 3                # max number of discounts
    horizon: int = 30             # days a cleared alert is remembered
    p_miss: float = 0.0           # analyst clears a truly malicious alert
    p_flag: float = 0.0           # analyst fails to clear a benign alert
    floor_mult: float = None      # only employees within the global top (floor_mult * k) may be
                                  # alerted; unfilled slots stay empty. 1.0 = leveling down.
    seed: int = 0


def simulate_policy(sd: ScoredData, cfg: PolicyConfig):
    """Run the alert policy day by day. Returns test df with columns
    alert, cleared, s_adj."""
    rng = np.random.default_rng(cfg.seed)
    T = sd.test
    days = T["day"].to_numpy()
    users = T["user"].to_numpy()
    y = T["y"].to_numpy()
    s = T["score"].to_numpy().astype(np.float64)
    g = T[cfg.group_col].to_numpy()
    phi = sd.phi
    phin = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-9)
    alert = np.zeros(len(T), bool)
    cleared = np.zeros(len(T), bool)
    s_adj = s.copy()
    store = {}                                   # user -> list of (day, unit phi)
    fallbacks = 0
    n_discounts = 0
    cumA, cumC = {}, {}
    disp_ranks = []          # global rank / k of alerts outside the global top-k
    budget_total = 0
    uniq_days, starts = np.unique(days, return_index=True)
    bounds = list(starts) + [len(T)]
    for di, d in enumerate(uniq_days):
        idx = np.arange(bounds[di], bounds[di + 1])
        sd_day = s[idx].copy()
        if cfg.schedule:
            for j, i in enumerate(idx):
                hist = store.get(users[i])
                if not hist:
                    continue
                hist[:] = [(dd, v) for dd, v in hist if (d - dd) / np.timedelta64(1, "D") <= cfg.horizon]
                if not hist or sd_day[j] >= cfg.tau_ov:
                    continue
                sims = [float(phin[i] @ v) for _, v in hist]
                if max(sims) >= cfg.gamma:
                    sd_day[j] -= cfg.lam * min(len(hist), cfg.m_cap)
                    n_discounts += 1
        s_adj[idx] = sd_day
        k = max(1, int(round(cfg.budget * len(idx))))
        if cfg.parity == "cumulative" and cfg.delta != "quota":
            m, fb = fair_allocate_cum(sd_day, g[idx], k, cfg.delta, cumA, cumC)
        else:
            m, fb = fair_allocate(sd_day, g[idx], k, cfg.delta)
        fallbacks += fb
        rank = np.empty(len(idx), int)
        rank[np.argsort(-sd_day, kind="stable")] = np.arange(1, len(idx) + 1)
        if cfg.floor_mult is not None:
            m = m & (rank <= int(np.ceil(cfg.floor_mult * k)))
        disp_ranks.extend((rank[m & (rank > k)] / k).tolist())
        budget_total += k
        for lab, cnt in zip(*np.unique(g[idx], return_counts=True)):
            cumC[lab] = cumC.get(lab, 0) + cnt
        for lab, cnt in zip(*np.unique(g[idx][m], return_counts=True)):
            cumA[lab] = cumA.get(lab, 0) + cnt
        a_idx = idx[m]
        alert[a_idx] = True
        u = rng.random(len(a_idx))
        clr = np.where(y[a_idx] == 1, u < cfg.p_miss, u >= cfg.p_flag)
        cleared[a_idx[clr]] = True
        if cfg.schedule:
            for i in a_idx[clr]:
                store.setdefault(users[i], []).append((d, phin[i]))
    out = T[["user", "day", "y", "score", "exposure", "scenario", cfg.group_col]].copy()
    out["alert"], out["cleared"], out["s_adj"] = alert, cleared, s_adj
    out.attrs["fallback_days"] = fallbacks
    out.attrs["discounts"] = n_discounts
    out.attrs["n_days"] = len(uniq_days)
    out.attrs["disp_ranks"] = np.array(disp_ranks)
    out.attrs["budget_total"] = budget_total
    return out


# =============================================================================
# 5. Metrics
# =============================================================================

def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def exposure_bias(test, score_col="score", group_col="g_functional_unit", min_rows=30):
    """Within-group Spearman correlation between score and exposure (benign rows)."""
    b = test[test["y"] == 0]
    rs, ws = [], []
    for _, d in b.groupby(group_col):
        if len(d) >= min_rows and d["exposure"].nunique() > 1:
            r = spearmanr(d[score_col], d["exposure"]).statistic
            if np.isfinite(r):
                rs.append(r)
                ws.append(len(d))
    return float(np.average(rs, weights=ws)) if rs else float("nan")


def wilson(x, n, z=1.96):
    """95% Wilson score interval for a binomial proportion x/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def policy_metrics(out, group_col, insiders=None, pi_assumed=0.05):
    res = {}
    mal = out[out["y"] == 1]
    ins_users = mal["user"].unique()
    det_users = mal.loc[mal["alert"], "user"].unique()
    res["insiders_test"] = len(ins_users)
    res["insiders_detected"] = len(det_users)
    res["recall_insider"] = len(det_users) / max(len(ins_users), 1)
    res["recall_lo"], res["recall_hi"] = wilson(len(det_users), len(ins_users))
    res["tpr_day"] = mal["alert"].mean() if len(mal) else float("nan")
    res["precision"] = out.loc[out["alert"], "y"].mean()
    # detection delay (days from first malicious test day to first alert on a malicious day)
    delays = []
    for u, d in mal.groupby("user"):
        hits = d.loc[d["alert"], "day"]
        if len(hits):
            delays.append((hits.min() - d["day"].min()).days)
    res["median_delay_days"] = float(np.median(delays)) if delays else float("nan")
    # group burden
    G = out.groupby(group_col)
    ar = G["alert"].mean()
    ben = out[out["y"] == 0]
    far = ben.groupby(group_col)["alert"].mean()
    pi = G["y"].mean()
    res["delta_AR"] = ar.max() - ar.min()
    res["delta_FAR"] = far.max() - far.min()
    res["ratio_FAR"] = far.max() / far.min() if far.min() > 0 else float("inf")
    res["pi_max"] = pi.max()
    res["lemma1_bound"] = res["delta_AR"] + 2 * res["pi_max"]
    res["lemma1_holds"] = bool(res["delta_FAR"] <= res["lemma1_bound"] + 1e-12)
    # audit from investigation outcomes: alerts confirmed as malicious are observed,
    # so P(a=1, y=0 | g) = AR_g - confirmed_g / n_g is known exactly; only the factor
    # 1/(1 - pi_g) is unknown. Interval for FAR_g with pi_g in [0, pi_assumed].
    conf = (out["alert"] & ~out["cleared"]).groupby(out[group_col]).sum() / G.size()
    lo = (ar - conf).clip(lower=0)
    hi = lo / (1 - pi_assumed)
    res["delta_FAR_hat"] = lo.max() - lo.min()
    res["audit_abs_err"] = abs(res["delta_FAR_hat"] - res["delta_FAR"])
    res["audit_bound"] = hi.max() - lo.min()
    res["audit_holds"] = bool(res["delta_FAR"] <= res["audit_bound"] + 1e-12)
    # longitudinal burden on never-malicious users
    benign_users = set(out["user"]) - set(ins_users)
    B = out[out["user"].isin(benign_users)].groupby("user")["cleared"].sum()
    res["gini_burden"] = gini(B.values)
    top = max(1, int(round(0.01 * len(B))))
    res["top1pct_share"] = B.sort_values(ascending=False).iloc[:top].sum() / max(B.sum(), 1)
    res["max_alerts_one_benign_user"] = int(B.max()) if len(B) else 0
    res["fallback_days"] = out.attrs.get("fallback_days", 0)
    res["discounts"] = out.attrs.get("discounts", 0)
    bt = out.attrs.get("budget_total", 0)
    res["alerts_used"] = out["alert"].sum() / bt if bt else float("nan")
    dr = out.attrs.get("disp_ranks", np.array([]))
    res["displaced_share"] = len(dr) / max(out["alert"].sum(), 1)
    res["displaced_rank_median"] = float(np.median(dr)) if len(dr) else float("nan")
    res["displaced_rank_p90"] = float(np.quantile(dr, 0.9)) if len(dr) else float("nan")
    return res


def lorenz(out):
    """Lorenz curve of cleared alerts over never-malicious employees."""
    ins = set(out.loc[out["y"] == 1, "user"])
    B = np.sort(out[~out["user"].isin(ins)].groupby("user")["cleared"].sum().to_numpy())
    x = np.concatenate([[0], np.arange(1, len(B) + 1) / len(B)])
    yv = np.concatenate([[0], np.cumsum(B) / max(B.sum(), 1)])
    return x, yv


def group_table(out, group_col):
    G = out.groupby(group_col)
    ben = out[out["y"] == 0].groupby(group_col)
    return pd.DataFrame({"users": G["user"].nunique(), "user_days": G.size(),
                         "pi": G["y"].mean(), "AR": G["alert"].mean(),
                         "FAR": ben["alert"].mean(), "TPR": out[out["y"] == 1].groupby(group_col)["alert"].mean()})


def dp_certificates(out, group_col, eps, beta=0.05, seed=0):
    """Per-day DP release of group alert counts and certificate soundness (Prop. 3)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, D in out.groupby("day"):
        G = D.groupby(group_col)
        n = G.size().to_numpy()
        if len(n) < 2:
            continue
        cnt = G["alert"].sum().to_numpy()
        noisy = cnt + rng.laplace(0, 1 / eps, len(cnt))
        ar, nar = cnt / n, noisy / n
        ben = D[D["y"] == 0].groupby(group_col)["alert"].mean()
        pi = G["y"].mean().max()
        slack = 2 * math.log(len(n) / beta) / (eps * n.min())
        bound = (nar.max() - nar.min()) + slack + 2 * pi
        dfar = ben.max() - ben.min()
        rows.append({"day": d, "true_dAR": ar.max() - ar.min(), "noisy_dAR": nar.max() - nar.min(),
                     "slack": slack, "bound": bound, "true_dFAR": dfar, "holds": dfar <= bound + 1e-12})
    R = pd.DataFrame(rows)
    return {"eps_per_day": eps, "eps_total_basic": eps * len(R), "days": len(R),
            "coverage": R["holds"].mean(), "mean_slack": R["slack"].mean(),
            "mean_abs_err_dAR": (R["noisy_dAR"] - R["true_dAR"]).abs().mean(),
            "mean_bound": R["bound"].mean(), "mean_true_dFAR": R["true_dFAR"].mean()}


def dp_certificates_window(out, group_col, eps, window=30, beta=0.05, seed=0, sizes="roster"):
    """Window-level DP burden report under ALERT-LOG semantics: the protected
    dataset is the alert log; neighbouring logs differ in one employee's rows.

    sizes='roster' (default): denominators are group rosters (every employee
      assigned to the group during the test period), which are public
      organisational information. Rates are the fraction of the roster alerted
      at least once in the window. Each employee adds at most 1 to one alert
      count, so the release of noisy counts has sensitivity 1 and is eps-DP.
      Slack (Prop. 3): 2 ln(|G|/beta) / (eps n_min).
    sizes='noisy': denominators are the employees ACTIVE in the window, which
      are not public; they are released with noise too, eps split evenly.
      Slack: 2 (b + b) / (n_min - b), b = ln(2|G|/beta) / (eps/2)."""
    rng = np.random.default_rng(seed)
    o = out.copy()
    d0 = o["day"].min()
    o["w"] = (o["day"] - d0).dt.days // window
    roster = o.groupby(group_col)["user"].nunique()
    U = o.groupby(["w", "user"]).agg(g=(group_col, "first"), alerted=("alert", "any"), mal=("y", "max")).reset_index()
    rows = []
    for w, D in U.groupby("w"):
        G = D.groupby("g")
        cnt = G["alerted"].sum()
        if sizes == "roster":
            n = roster.reindex(cnt.index).to_numpy().astype(float)
            mal = G["mal"].sum().to_numpy()
            ben_n = n - mal                               # benign roster members (inactive ones are benign)
            ben_alerted = (G["alerted"].sum() - D[D["mal"] == 1].groupby("g")["alerted"].sum().reindex(cnt.index).fillna(0)).to_numpy()
            far = ben_alerted / np.maximum(ben_n, 1)
            pi = (mal / n).max()
        else:
            n = G.size().to_numpy().astype(float)
            ben = D[D["mal"] == 0].groupby("g")["alerted"].mean().reindex(cnt.index)
            far = ben.to_numpy()
            pi = G["mal"].mean().max()
        if len(n) < 2:
            continue
        c = cnt.to_numpy().astype(float)
        ar = c / n
        if sizes == "roster":
            nar = (c + rng.laplace(0, 1 / eps, len(c))) / n
            slack = 2 * math.log(len(n) / beta) / (eps * n.min())
        else:
            e_half = eps / 2
            nn = n + rng.laplace(0, 1 / e_half, len(n))
            nar = (c + rng.laplace(0, 1 / e_half, len(c))) / np.maximum(nn, 1.0)
            bq = math.log(2 * len(n) / beta) / e_half
            slack = 2 * (bq + bq) / (n.min() - bq) if n.min() > bq else float("inf")
        bound = (nar.max() - nar.min()) + slack + 2 * pi
        dfar = np.nanmax(far) - np.nanmin(far)
        rows.append({"true_dAR": ar.max() - ar.min(), "noisy_dAR": nar.max() - nar.min(), "slack": slack,
                     "bound": bound, "true_dFAR": dfar, "holds": dfar <= bound + 1e-12, "n_min": n.min()})
    R = pd.DataFrame(rows)
    fin = R["slack"].replace(np.inf, np.nan)
    return {"window_days": window, "sizes": sizes, "eps_per_window": eps, "windows": len(R),
            "eps_total_basic": eps * len(R), "coverage": R["holds"].mean(),
            "median_slack": R["slack"].median(), "mean_slack": fin.mean(),
            "vacuous_share": float(np.isinf(R["slack"]).mean()),
            "median_abs_err_dAR": (R["noisy_dAR"] - R["true_dAR"]).abs().median(),
            "mean_true_dFAR": R["true_dFAR"].mean(), "median_min_group": R["n_min"].median()}


# =============================================================================
# 6. Simulations
# =============================================================================

def exposure_injection(ds: Dataset, factor, frac_roles=0.3, seed=0):
    """Return a copy of ds where benign activity counts of users in a random
    subset of roles are multiplied by `factor` (legitimately higher telemetry)."""
    rng = np.random.default_rng(seed)
    roles = ds.df["role"].unique()
    chosen = set(rng.choice(roles, max(1, int(len(roles) * frac_roles)), replace=False))
    D = ds.df.copy()
    sel = D["role"].isin(chosen).to_numpy()
    for c in COUNT_FEATURES:
        D.loc[sel, c] = np.round(D.loc[sel, c] * factor)
    D["exposure"] = D[VOLUME_FEATURES].sum(axis=1)
    D["g_injected"] = np.where(sel, "inflated", "normal")
    return Dataset(ds.name + f"-inj{factor}", D, ds.insiders, ds.split_day, ds.group_cols + ["g_injected"])


def discount_farming(sd: ScoredData, n_farm=3, alpha=1.0):
    """Insert farming days for each test insider: benign days just before the
    first malicious day are replaced with x_benign + alpha * (mean malicious -
    mean benign) so that they resemble the later malicious behaviour.
    Returns a new ScoredData and the list of farming insiders."""
    T = sd.test.copy()
    phi = sd.phi.copy()
    farmers = []
    for u, D in T[T["y"] == 1].groupby("user"):
        d0 = D["day"].min()
        ub = T[(T["user"] == u) & (T["y"] == 0)]
        prior = ub[ub["day"] < d0].sort_values("day").tail(n_farm)
        if prior.empty:
            continue
        delta = (D[COUNT_FEATURES].mean() - ub[COUNT_FEATURES].mean()).clip(lower=0)
        idx = prior.index
        T.loc[idx, COUNT_FEATURES] = (T.loc[idx, COUNT_FEATURES] + alpha * delta.values).round()
        T.loc[idx, "exposure"] = T.loc[idx, VOLUME_FEATURES].sum(axis=1)
        farmers.append(u)
    if not farmers:
        return None, []
    if isinstance(sd.model, PerGroupDetector) or sd.model is None:
        raise ValueError("discount_farming needs a single-model detector")
    X = sd.transform.transform(T)
    raw = sd.model.score(X)
    T["raw_score"] = raw
    T["score"] = to_quantile_ref(raw, sd.raw_ref)
    # recompute explanations for modified rows (baseline unchanged)
    Xo = sd.transform.transform(sd.test)
    base_shift = Xo - sd.phi
    phi = X - base_shift
    return ScoredData(sd.detector + "-farm", T, phi, sd.model, sd.transform, sd.raw_ref), farmers


# =============================================================================
# 7. Export helpers
# =============================================================================

PRETTY = {
    "recall_insider": "Recall", "precision": "Prec.", "median_delay_days": "Delay (d)",
    "delta_AR": r"$\Delta_{\mathrm{AR}}$", "delta_FAR": r"$\Delta_{\mathrm{FAR}}$",
    "ratio_FAR": r"$\rho_{\mathrm{FAR}}$", "gini_burden": r"Gini($B$)", "top1pct_share": "Top-1\\%",
    "exposure_bias": "Exp. bias", "max_alerts_one_benign_user": r"$\max B_i$", "pi_max": r"$\pi_{\max}$",
    "lemma1_bound": "Lemma 1 bound", "detector": "Detector", "grouping": "Grouping", "method": "Method",
    "delta": r"$\delta$", "lam": r"$\lambda$", "p_miss": r"$p_{\mathrm{miss}}$", "eps": r"$\varepsilon$",
    "eps_per_day": r"$\varepsilon$/day", "mean_slack": "Slack", "coverage": "Coverage",
    "mean_true_dFAR": r"True $\Delta_{\mathrm{FAR}}$", "n_groups": r"$|\mathcal{G}|$",
    "avg_min_group_size": r"$\bar{n}_{\min}$", "model": "Model", "kappa": r"$\kappa$",
    "factor": "Factor", "fallback_pct": "Fallback (\\%)", "release": "Release", "users": "Users",
    "insiders": "Insiders", "insiders_test": "Test ins.", "days": "Days", "user_days": "User-days",
    "base_rate": r"$\pi$", "roles": "Roles", "functional_units": "Func. units", "split_day": "Split",
    "farmer_recall": "Recall (farmers)", "alpha": r"$\alpha$", "gamma": r"$\gamma$",
    "tau_ov": r"$\tau_{\mathrm{ov}}$", "schedule": "Sched.",
    "window_days": "Window (d)", "avg_min_group_users": r"$\bar{n}_{\min}$", "eps_per_window": r"$\varepsilon$/window",
    "eps_total_basic": r"Total $\varepsilon$", "recall_no_farming": "Recall (no farming)", "recall_farming": "Recall (farming)",
    "tpr_day": "Day TPR", "recall_global": "Recall (global)", "recall_fair": "Recall (FAIR-IRS)",
    "fallback_days": "Fallback (days)", "lost_by_allocation": "Lost",
 "gained_by_allocation": "Gained", "mean_bound": "Bound", "setting": "Setting",
    "recall_ci": "Recall [95\\% CI]", "alerts_used": "Alerts used", "displaced_share": "Displaced",
    "displaced_rank_median": "Disp. rank (med.)", "displaced_rank_p90": "Disp. rank (p90)",
    "delta_FAR_hat": r"$\widehat{\Delta}_{\mathrm{FAR}}$", "audit_abs_err": "Audit error", "audit_bound": "Outcome bound",
    "floor": "Floor", "sizes": "Sizes", "median_slack": "Slack (median)", "vacuous_share": "Vacuous", "median_abs_err_dAR": "Error (median)",
    "median_min_group": r"$n_{\min}$ (median)", "weekdays_only": "Weekdays", "insiders_detected": "Detected",
    "policy": "Policy", "runs": "Runs", "budget": "Budget", "training": "Training", "baseline": "Expl. baseline",
    "recall_no_sched": "Recall (no sched.)", "recall_sched": "Recall (sched.)",
}
GROUP_PRETTY = {"g_functional_unit": "Func. unit", "g_department": "Department", "g_role": "Role",
                "g_afterhours": "After-hours", "g_exposure": "Exposure", "g_injected": "Injected",
                "g_synthA": "Synthetic $A$"}


def to_latex(df, path, caption, label, float_fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            df[c] = df[c].map(lambda v: GROUP_PRETTY.get(v, v))
    ONE_DECIMAL = {PRETTY.get(c, c) for c in ["median_delay_days", "fallback_pct", "avg_min_group_size",
                                               "max_alerts_one_benign_user"]}
    esc = lambda c: c if "$" in c or "\\" in c else c.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    df.columns = [PRETTY.get(c, esc(str(c))) for c in df.columns]
    cols = list(df.columns)
    lines = [r"\begin{table}[!t]", rf"\caption{{{caption}\label{{{label}}}}}", r"\centering",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{" + "l" * 1 + "c" * (len(cols) - 1) + "}",
             r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                fmt = "{:.1f}" if c in ONE_DECIMAL else float_fmt
                cells.append(r"$\infty$" if v == np.inf else ("--" if not np.isfinite(v) else fmt.format(v)))
            else:
                v = str(v)
                if "$" not in v:
                    v = v.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
                cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# 8. Toy CERT-format generator (for testing the pipeline before real data)
# =============================================================================

def make_toy_cert(root, version="4.2", n_users=150, n_days=120, n_insiders=8, seed=0):
    """Write a small CERT-format release + answers folder for smoke tests."""
    rng = np.random.default_rng(seed)
    rel = os.path.join(root, f"r{version}")
    os.makedirs(os.path.join(rel, "LDAP"), exist_ok=True)
    ans = os.path.join(root, "answers")
    os.makedirs(ans, exist_ok=True)
    users = [f"{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 3))}{1000+i}" for i in range(n_users)]
    roles = rng.choice(["ITAdmin", "Salesman", "Engineer", "Manager", "Clerk", "Scientist"], n_users,
                       p=[.12, .3, .2, .1, .18, .1])
    units = {"ITAdmin": "IT", "Salesman": "Sales", "Engineer": "R&D", "Manager": "Admin",
             "Clerk": "Admin", "Scientist": "R&D"}
    act = {"ITAdmin": 3.0, "Salesman": 1.0, "Engineer": 1.5, "Manager": 0.8, "Clerk": 0.7, "Scientist": 1.2}
    start = pd.Timestamp("2010-01-04")
    days = pd.date_range(start, periods=n_days, freq="D")
    months = sorted({d.strftime("%Y-%m") for d in days})
    for m in months:
        pd.DataFrame({"employee_name": users, "user_id": users,
                      "email": [u.lower() + "@dtaa.com" for u in users], "role": roles,
                      "business_unit": "1", "functional_unit": [units[r] for r in roles],
                      "department": [units[r] + "-" + str(i % 2) for i, r in enumerate(roles)],
                      "team": "t", "supervisor": "x"}).to_csv(os.path.join(rel, "LDAP", f"{m}.csv"), index=False)
    night = set(rng.choice(users, n_users // 8, replace=False))
    insiders = rng.choice(users, n_insiders, replace=False)
    ins_rows = []
    mal_days = {}
    for j, u in enumerate(insiders):
        s = int(rng.integers(n_days * 0.3, n_days * 0.85))
        mal_days[u] = [days[s + t] for t in range(int(rng.integers(2, 6))) if s + t < n_days]
        ins_rows.append({"dataset": version, "scenario": 1 + j % 3,
                         "details": f"r{version}-{1 + j % 3}-{u}.csv", "user": u,
                         "start": mal_days[u][0].strftime(DATE_FMT),
                         "end": (mal_days[u][-1] + pd.Timedelta(hours=23)).strftime(DATE_FMT)})
    r6 = str(version).startswith("6")
    logon, device, file, email, http = [], [], [], [], []
    detail = {u: [] for u in insiders}
    cnt = 0
    for d in days:
        wk = d.dayofweek >= 5
        for i, u in enumerate(users):
            if wk and rng.random() > 0.1:
                continue
            a = act[roles[i]]
            h0 = 20 if u in night else 8
            is_mal = u in mal_days and d in mal_days[u]
            ts = lambda h: (d + pd.Timedelta(hours=h % 24, minutes=int(rng.integers(0, 59)))).strftime(DATE_FMT)
            cnt += 1
            logon.append([f"{{L{cnt}}}", ts(h0), u, f"PC-{i}", "Logon"])
            logon.append([f"{{F{cnt}}}", ts(h0 + 9), u, f"PC-{i}", "Logoff"])
            for _ in range(rng.poisson(0.3 * a + (4 if is_mal else 0))):
                h = 22 if is_mal else h0 + 2
                row = [f"{{D{cnt}}}", ts(h), u, f"PC-{i}"] + (["C:\\"] if r6 else []) + ["Connect"]
                device.append(row)
                if is_mal:
                    detail[u].append(["device"] + row)
            for _ in range(rng.poisson(5 * a + (10 if is_mal else 0))):
                ext = rng.choice(["doc", "pdf", "txt", "jpg", "zip", "exe"])
                row = [f"{{Fi{cnt}}}", ts(h0 + 3), u, f"PC-{i}", f"X{cnt}.{ext}"]
                row += (["File Write", str(bool(is_mal)), "False", "kw"] if r6 else ["kw"])
                file.append(row)
            for _ in range(rng.poisson(4 * a)):
                to = "a@dtaa.com" if rng.random() > 0.1 else "b@gmail.com"
                row = [f"{{E{cnt}}}", ts(h0 + 4), u, f"PC-{i}", to, "", "", u.lower() + "@dtaa.com"]
                row += (["Send"] if r6 else []) + [str(int(rng.integers(1000, 90000))),
                                                    ("f.doc(1)" if r6 else "1") if rng.random() < .2 else ("" if r6 else "0"),
                                                    "kw"]
                email.append(row)
            for _ in range(rng.poisson(12 * a)):
                url = rng.choice(["http://news.com/x", "http://shop.com/y", "http://monster.com/job"],
                                 p=[.6, .38, .02])
                if is_mal and rng.random() < 0.3:
                    url = "http://wikileaks.org/up"
                row = [f"{{H{cnt}}}", ts(h0 + 5), u, f"PC-{i}", url] + (["WWW Visit"] if r6 else []) + ["kw"]
                http.append(row)
    hdr = lambda names: names
    def write(name, rows, cols, header=True):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(os.path.join(rel, name), index=False, header=header)
    write("logon.csv", logon, ["id", "date", "user", "pc", "activity"])
    write("device.csv", device, NOHEADER_SCHEMAS["device"][6 if r6 else 5])
    write("file.csv", file, NOHEADER_SCHEMAS["file"][9 if r6 else 6])
    write("email.csv", email, NOHEADER_SCHEMAS["email"][12 if r6 else 11])
    # http.csv without header, as in some CERT distributions
    write("http.csv", http, NOHEADER_SCHEMAS["http"][7 if r6 else 6], header=False)
    pd.DataFrame({"employee_name": users, "user_id": users, "O": 30, "C": 30, "E": 30, "A": 30, "N": 30}) \
        .to_csv(os.path.join(rel, "psychometric.csv"), index=False)
    ins_path = os.path.join(ans, "insiders.csv")
    old = pd.read_csv(ins_path, dtype=str) if os.path.exists(ins_path) else None
    new = pd.DataFrame(ins_rows)
    (pd.concat([old, new]) if old is not None else new).to_csv(ins_path, index=False)
    for j, u in enumerate(insiders):
        sub = os.path.join(ans, f"r{version}-{1 + j % 3}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"r{version}-{1 + j % 3}-{u}.csv"), "w", newline="") as f:
            csv.writer(f).writerows(detail[u] or [["logon", "{x}", mal_days[u][0].strftime(DATE_FMT), u, "PC", "Logon"]])
    return rel, ans


# =============================================================================
# 9. Experiments (one function per research question / simulation)
# =============================================================================

def default_detectors(seed=0):
    return [IFDetector(seed), AEDetector(seed), LRDetector(), GBMDetector(seed)]


def score_all(ds: Dataset, detectors=None, extra_modes=True, peer_col="g_role", seed=0, with_lstm=True):
    """Train every detector once and return {name: ScoredData}."""
    detectors = detectors or default_detectors(seed)
    if with_lstm:
        try:
            import torch  # noqa: F401
            detectors = list(detectors) + [LSTMDetector(seed)]
        except ImportError:
            log("[skip] LSTM: PyTorch is not installed")
    scored = {}
    for det in detectors:
        try:
            sd = run_detector(ds, det)
        except Exception as ex:                      # one detector must not stop the run
            log(f"[skip] {det.name}: {type(ex).__name__}: {ex}")
            continue
        if sd is not None:
            scored[sd.detector] = sd
    if extra_modes:
        sd = run_detector(ds, IFDetector(seed), feature_mode="share")       # B2
        scored[sd.detector] = sd
        pg = PerGroupDetector(partial(IFDetector, seed), None, None, name="IF-peer")  # B3
        sd = run_detector(ds, pg, peer_col=peer_col)
        sd.detector = "IF-peer"
        scored["IF-peer"] = sd
    return scored


def rq1_audit(ds, scored, budget=0.01, detectors=None):
    """Burden of unmitigated detectors (global top-k) for every grouping."""
    rows, gtabs = [], {}
    for name in detectors or ["IF", "AE", "LR", "GBM", "LSTM"]:
        if name not in scored:
            continue
        sd = scored[name]
        for gc in ds.group_cols:
            out = simulate_policy(sd, PolicyConfig(budget=budget, delta=1.0, group_col=gc))
            m = policy_metrics(out, gc)
            m.update(detector=name, grouping=gc, exposure_bias=exposure_bias(sd.test, group_col=gc))
            rows.append(m)
            gtabs[(name, gc)] = group_table(out, gc)
    return pd.DataFrame(rows), gtabs


def rq2_price_of_fairness(ds, scored, group_col, budget=0.01,
                          deltas=(1.0, 0.01, 0.005, 0.002, 0.001, 0.0, "quota"),
                          methods=("IF", "AE", "LR", "GBM", "LSTM", "IF-share", "IF-peer")):
    rows = []
    for name in methods:
        if name not in scored:
            continue
        for d in deltas:
            if name in ("IF-share", "IF-peer") and d != 1.0:
                continue                      # baselines are evaluated without allocation
            out = simulate_policy(scored[name], PolicyConfig(budget=budget, delta=d, group_col=group_col))
            m = policy_metrics(out, group_col)
            m.update(method=name, delta=str(d), grouping=group_col)
            rows.append(m)
    return pd.DataFrame(rows)


def rq3_scheduling(scored, group_col, budget=0.01, delta=0.002, detectors=("IF", "GBM", "LSTM"),
                   lams=(0.0, 0.005, 0.01, 0.02), p_miss=(0.0, 0.1, 0.3)):
    rows = []
    for name in detectors:
        if name not in scored:
            continue
        for pm in p_miss:
            for lam in lams:
                cfg = PolicyConfig(budget=budget, delta=delta, group_col=group_col,
                                   schedule=lam > 0, lam=lam, p_miss=pm)
                out = simulate_policy(scored[name], cfg)
                m = policy_metrics(out, group_col)
                m.update(detector=name, lam=lam, p_miss=pm)
                rows.append(m)
    return pd.DataFrame(rows)


def rq4a_dp_training(ds, group_col, budget=0.01, eps_list=(0.5, 1, 2, 4, 8), seeds=(0, 1, 2, 3, 4)):
    """LR, clipped SGD without noise, and DP-SGD, each over several seeds.
    Returns (raw rows, summary with mean and 95% CI)."""
    rows = []
    sd = run_detector(ds, LRDetector())
    if sd is None:
        return pd.DataFrame(), pd.DataFrame()
    m = policy_metrics(simulate_policy(sd, PolicyConfig(budget=budget, group_col=group_col)), group_col)
    m.update(model="LR (non-private)", eps=float("inf"), seed=0)
    rows.append(m)
    for e in [np.inf] + list(eps_list):
        for sdseed in seeds:
            det = DPLRDetector(eps=e, seed=sdseed)
            sd = run_detector(ds, det)
            m = policy_metrics(simulate_policy(sd, PolicyConfig(budget=budget, group_col=group_col)), group_col)
            m.update(model="SGD-LR (clipped, no noise)" if e == np.inf else "DP-LR", eps=e, seed=sdseed)
            rows.append(m)
    raw = pd.DataFrame(rows)
    summ = []
    for (mod, e), d in raw.groupby(["model", "eps"], sort=False):
        r = {"model": mod, "eps": e, "runs": len(d)}
        for c in ["recall_insider", "precision", "delta_FAR", "ratio_FAR", "gini_burden"]:
            mu, h = _ci(d[c].replace(np.inf, np.nan))
            r[c] = f"{mu:.3f}" + ("" if not np.isfinite(h) else f" $\\pm$ {h:.3f}")
        summ.append(r)
    return raw, pd.DataFrame(summ)


def rq4b_certificates(ds, scored, detector="IF", budget=0.01, delta=0.002,
                      eps_list=(0.5, 1, 2, 4), windows=(1, 7, 30), beta=0.05,
                      groupings=("g_functional_unit", "g_department", "g_role")):
    """Roster denominators for every window; noisy active sizes reported as a variant."""
    rows = []
    for gc in groupings:
        out = simulate_policy(scored[detector], PolicyConfig(budget=budget, delta=delta, group_col=gc))
        for w in windows:
            for sz in ("roster", "noisy"):
                for e in eps_list:
                    c = dp_certificates_window(out, gc, e, window=w, beta=beta, sizes=sz)
                    c.update(grouping=gc, n_groups=out[gc].nunique())
                    rows.append(c)
    return pd.DataFrame(rows)


def rq5_evasion(scored, group_col, budget=0.01, delta=0.002, detectors=("IF", "GBM", "LSTM"),
                alphas=(0.5, 1.0), n_farm=3, gammas=(0.7, 0.8, 0.9), tau_ovs=(0.995, 0.999, 1.01)):
    """(a) threshold exploitation: insiders caught by global top-k but missed by
    FAIR-IRS allocation (and vice versa); (b) discount farming against the
    scheduler, compared with the same insiders when they do not farm."""
    thr, rows = [], []
    det = lambda o: set(o.loc[(o["y"] == 1) & o["alert"], "user"])
    for name in detectors:
        if name not in scored:
            continue
        sd = scored[name]
        base = simulate_policy(sd, PolicyConfig(budget=budget, delta=1.0, group_col=group_col))
        fair = simulate_policy(sd, PolicyConfig(budget=budget, delta=delta, group_col=group_col))
        thr.append({"detector": name, "recall_global": policy_metrics(base, group_col)["recall_insider"],
                    "recall_fair": policy_metrics(fair, group_col)["recall_insider"],
                    "lost_by_allocation": len(det(base) - det(fair)),
                    "gained_by_allocation": len(det(fair) - det(base)),
                    "lost_users": " ".join(sorted(det(base) - det(fair))),
                    "gained_users": " ".join(sorted(det(fair) - det(base)))})
        if getattr(sd.model, "sequence", False):
            log(f"[skip] discount farming for {name}: needs the full sequence model")
            continue
        for alpha in alphas:
            sdf, farmers = discount_farming(sd, n_farm=n_farm, alpha=alpha)
            if not farmers:
                continue
            def farmer_recall(s_, cfg):
                out = simulate_policy(s_, cfg)
                sub = out[out["user"].isin(farmers) & (out["y"] == 1)]
                return sub.groupby("user")["alert"].any().mean()
            for g in gammas:
                for t in tau_ovs:
                    for sched in (False, True):
                        cfg = PolicyConfig(budget=budget, delta=delta, group_col=group_col,
                                           schedule=sched, gamma=g, tau_ov=t)
                        rows.append({"detector": name, "alpha": alpha, "gamma": g, "tau_ov": t, "schedule": sched,
                                     "farmers": len(farmers),
                                     "recall_no_farming": farmer_recall(sd, cfg),
                                     "recall_farming": farmer_recall(sdf, cfg)})
    return pd.DataFrame(thr), pd.DataFrame(rows)


def sim_kappa(ds, kappas=(0.0, 0.25, 0.5, 0.75, 1.0), budget=0.01, delta=0.002,
              detectors=("IF", "GBM"), seed=0, scored=None):
    rows = []
    scored = scored or score_all(ds, extra_modes=False)
    for kappa in kappas:
        col = add_synthetic_attribute(ds, kappa, seed=seed)
        for name in detectors:
            if name not in scored:
                continue
            sd = scored[name]
            sd.test[col] = sd.test["user"].map(ds.df.drop_duplicates("user").set_index("user")[col])
            for d in (1.0, delta):
                out = simulate_policy(sd, PolicyConfig(budget=budget, delta=d, group_col=col))
                m = policy_metrics(out, col)
                m.update(kappa=kappa, detector=name, delta=str(d))
                rows.append(m)
    return pd.DataFrame(rows)


def sim_exposure_injection(ds, factors=(1.0, 1.5, 2.0, 3.0), budget=0.01, delta=0.002, seed=0):
    rows = []
    for f in factors:
        dsi = exposure_injection(ds, f, seed=seed)
        sd = run_detector(dsi, IFDetector(seed))
        sds = run_detector(dsi, IFDetector(seed), feature_mode="share")
        for label, s, d in [("IF global", sd, 1.0), ("IF + FAIR-IRS", sd, delta), ("IF-share", sds, 1.0)]:
            out = simulate_policy(s, PolicyConfig(budget=budget, delta=d, group_col="g_injected"))
            m = policy_metrics(out, "g_injected")
            m.update(factor=f, method=label, exposure_bias=exposure_bias(s.test, group_col="g_functional_unit"))
            rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# 10. Figures
# =============================================================================

METHOD_LABEL = {"IF": "IF (D1)", "AE": "AE (D2)", "LR": "LR (D3)", "GBM": "GBM (D4)", "LSTM": "LSTM (D5)"}


def plot_pareto(df, path, title=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    marks = iter(["o", "s", "^", "D", "v", "P"])
    for name, d in df.groupby("method", sort=False):
        d = d.sort_values("delta_FAR")
        ax.plot(d["delta_FAR"], d["recall_insider"], marker=next(marks), ms=3.5, lw=1,
                label=METHOD_LABEL.get(name, name))
    ax.set_xlabel(r"False-alert disparity $\Delta_{\mathrm{FAR}}$")
    ax.set_ylabel("Insider recall")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.25, lw=.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lorenz(curves, path):
    """curves: list of (label, out) pairs; one panel per detector is drawn by the caller."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for (label, out), ls in zip(curves, ["-", "--", ":", "-."]):
        x, yv = lorenz(out)
        ax.plot(x, yv, ls, lw=1.2, label=label)
    ax.plot([0, 1], [0, 1], color="0.6", lw=.6)
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Share of benign employees (sorted by false alerts)")
    ax.set_ylabel("Cumulative share of false alerts")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(alpha=.25, lw=.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lines(df, x, y, hue, path, xlabel=None, ylabel=None, logx=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, d in df.groupby(hue):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], "o-", label=str(name), ms=4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# 11. Additional experiments for the final paper
# =============================================================================

def add_intersection(df, a, b, col, min_users=20):
    """Intersectional group a x b; cells with fewer than min_users users -> 'Other'."""
    v = df[a].astype(str) + " | " + df[b].astype(str)
    size = df.assign(_v=v).groupby("_v")["user"].nunique()
    small = set(size[size < min_users].index)
    df[col] = np.where(v.isin(small), "Other", v)
    return col


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    from scipy.stats import t
    return x.mean(), t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def run_seeds(ds, seeds=(0, 1, 2, 3, 4), budget=0.01, delta=0.002, group_col="g_functional_unit",
              cache_dir=None, metrics=("recall_insider", "precision", "delta_FAR", "ratio_FAR", "gini_burden")):
    """Retrain every detector with several seeds; report mean and 95% CI for
    global top-k and FAIR-IRS. LR is deterministic and is run once."""
    import pickle
    rows = []
    for sd_seed in seeds:
        p = os.path.join(cache_dir, f"scored_{ds.name}_seed{sd_seed}_v2.pkl") if cache_dir else None
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                sc = pickle.load(f)
        else:
            dets = [IFDetector(sd_seed), AEDetector(sd_seed), GBMDetector(sd_seed)] + ([LRDetector()] if sd_seed == seeds[0] else [])
            sc = {}
            for det in dets:
                s = run_detector(ds, det)
                if s is not None:
                    sc[s.detector] = s
            if p:
                with open(p, "wb") as f:
                    pickle.dump(sc, f)
        for name, s in sc.items():
            for d in (1.0, delta):
                out = simulate_policy(s, PolicyConfig(budget=budget, delta=d, group_col=group_col, seed=sd_seed))
                m = policy_metrics(out, group_col)
                m.update(detector=name, policy="global" if d == 1.0 else "FAIR-IRS", seed=sd_seed)
                rows.append(m)
    raw = pd.DataFrame(rows)
    summ = []
    for (det, pol), d in raw.groupby(["detector", "policy"], sort=False):
        r = {"detector": det, "policy": pol, "runs": len(d)}
        for m in metrics:
            mu, h = _ci(d[m])
            r[m] = f"{mu:.3f}" + ("" if not np.isfinite(h) else f" $\\pm$ {h:.3f}")
        summ.append(r)
    return raw, pd.DataFrame(summ)


def budget_sweep(sc, budgets=(0.005, 0.01, 0.02), delta_rel=0.2, group_col="g_functional_unit",
                 detectors=("IF", "AE", "LR", "GBM", "LSTM")):
    """delta is set relative to the budget: delta = delta_rel * budget."""
    rows = []
    for name in detectors:
        if name not in sc:
            continue
        for b in budgets:
            for d in (1.0, delta_rel * b):
                out = simulate_policy(sc[name], PolicyConfig(budget=b, delta=d, group_col=group_col))
                m = policy_metrics(out, group_col)
                m.update(detector=name, budget=b, policy="global" if d == 1.0 else "FAIR-IRS")
                rows.append(m)
    return pd.DataFrame(rows)


def contaminated_training(ds, budget=0.01, group_col="g_functional_unit", seed=0):
    """Unsupervised detectors trained on all training days (labels unused)."""
    rows = []
    for clean in (True, False):
        for det in (IFDetector(seed), AEDetector(seed)):
            s = run_detector(ds, det, clean_train=clean)
            out = simulate_policy(s, PolicyConfig(budget=budget, group_col=group_col))
            m = policy_metrics(out, group_col)
            m.update(detector=det.name, training="clean" if clean else "contaminated",
                     exposure_bias=exposure_bias(s.test, group_col=group_col))
            rows.append(m)
    return pd.DataFrame(rows)


def phi_comparison(ds, sc, budget=0.01, delta=0.002, group_col="g_functional_unit", lam=0.01, seed=0):
    """Scheduling with population- vs self-baseline explanations (IF)."""
    rows = []
    s_self = run_detector(ds, IFDetector(seed), phi_mode="self")
    for label, s in (("population", sc["IF"]), ("self", s_self)):
        for l in (0.0, lam):
            out = simulate_policy(s, PolicyConfig(budget=budget, delta=delta, group_col=group_col,
                                                  schedule=l > 0, lam=l))
            m = policy_metrics(out, group_col)
            m.update(baseline=label, lam=l)
            rows.append(m)
    return pd.DataFrame(rows)


def intersectional(sc, budget=0.01, delta=0.002, main_col="g_functional_unit", inter_col="g_inter",
                   detectors=("IF", "GBM")):
    """Parity enforced on the main grouping vs on the intersection; burden always
    measured on the intersection (fairness gerrymandering check)."""
    rows = []
    for name in detectors:
        if name not in sc:
            continue
        s = sc[name]
        for label, gcol, d in (("global", inter_col, 1.0), ("FAIR-IRS on main grouping", main_col, delta),
                               ("FAIR-IRS on intersection", inter_col, delta)):
            out = simulate_policy(s, PolicyConfig(budget=budget, delta=d, group_col=gcol))
            out[inter_col] = s.test[inter_col].values
            m = policy_metrics(out, inter_col)
            m.update(detector=name, policy=label)
            rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# 12. Additional experiments
# =============================================================================

def with_scores(sd: ScoredData, new_scores, name):
    """Copy of a ScoredData with a different score column (same explanations)."""
    T = sd.test.copy()
    T["score"] = new_scores
    return ScoredData(name, T, sd.phi, sd.model, sd.transform, sd.raw_ref, sd.train_ref)


def group_rank_scores(sd: ScoredData, group_col):
    """Baseline B6: rank scores within each group on each day (percentile), then
    take the global top-k. No memory across days."""
    T = sd.test
    return T.groupby(["day", T[group_col]])["score"].rank(pct=True, method="first").to_numpy() + 1e-9 * T["score"].to_numpy()


def simulate_group_thresholds(sd: ScoredData, group_col, budget=0.01):
    """Baseline B7 (Hardt-style): one threshold per group, tuned on the benign
    training period so that each group's training false-alert rate equals the
    budget rate; at test time everyone above their group's threshold is alerted,
    so the number of alerts per day is not fixed."""
    tr = sd.train_ref
    glob = np.quantile(tr["raw_score"], 1 - budget)
    thr = tr.groupby(group_col)["raw_score"].quantile(1 - budget)
    T = sd.test
    t = T[group_col].map(thr).fillna(glob).to_numpy()
    alert = T["raw_score"].to_numpy() > t
    out = T[["user", "day", "y", "score", "exposure", "scenario", group_col]].copy()
    out["alert"] = alert
    out["cleared"] = alert & (T["y"].to_numpy() == 0)
    out["s_adj"] = out["score"]
    k_total = sum(max(1, int(round(budget * n))) for n in T.groupby("day").size())
    out.attrs.update(fallback_days=0, discounts=0, n_days=T["day"].nunique(), disp_ranks=np.array([]),
                     budget_total=k_total)
    return out


def rq2_baselines(sc, group_col, budget=0.01, delta=0.002, detectors=("IF", "AE", "LR", "GBM", "LSTM")):
    """Global top-k, FAIR-IRS, B6 group-rank and B7 group thresholds for each detector."""
    rows = []
    for name in detectors:
        if name not in sc:
            continue
        sd = sc[name]
        runs = [("global (B0)", simulate_policy(sd, PolicyConfig(budget=budget, delta=1.0, group_col=group_col))),
                ("FAIR-IRS (B4)", simulate_policy(sd, PolicyConfig(budget=budget, delta=delta, group_col=group_col))),
                ("group rank (B6)", simulate_policy(with_scores(sd, group_rank_scores(sd, group_col), name),
                                                   PolicyConfig(budget=budget, delta=1.0, group_col=group_col))),
                ("group thresholds (B7)", simulate_group_thresholds(sd, group_col, budget))]
        for label, out in runs:
            m = policy_metrics(out, group_col)
            m.update(detector=name, policy=label)
            rows.append(m)
    return pd.DataFrame(rows)


def rq2_floor(sc, group_col, budget=0.01, delta=0.002, floors=(None, 2.0, 1.5, 1.0), detectors=("IF", "GBM", "LSTM")):
    """Weaker-evidence check: FAIR-IRS with a score floor. Only employees within the
    global top (floor * k) of the day may be alerted; unfilled slots stay empty.
    floor = 1 alerts only people the global policy would also have alerted (leveling down)."""
    rows = []
    for name in detectors:
        if name not in sc:
            continue
        base = simulate_policy(sc[name], PolicyConfig(budget=budget, delta=1.0, group_col=group_col))
        mb = policy_metrics(base, group_col)
        mb.update(detector=name, floor="global (B0)")
        rows.append(mb)
        for f in floors:
            out = simulate_policy(sc[name], PolicyConfig(budget=budget, delta=delta, group_col=group_col, floor_mult=f))
            m = policy_metrics(out, group_col)
            m.update(detector=name, floor="none" if f is None else f"{f:g}k")
            rows.append(m)
    return pd.DataFrame(rows)


def rq1_audit_outcomes(ds, sc, budget=0.01, delta=0.002, detectors=("IF", "AE", "LR", "GBM", "LSTM"), pi_assumed=0.05):
    """Label-free (Lemma 1) versus outcome-based audit, for global and FAIR-IRS policies."""
    rows = []
    for name in detectors:
        if name not in sc:
            continue
        for gc in ds.group_cols:
            for label, d in (("global", 1.0), ("FAIR-IRS", delta)):
                out = simulate_policy(sc[name], PolicyConfig(budget=budget, delta=d, group_col=gc))
                m = policy_metrics(out, gc, pi_assumed=pi_assumed)
                m.update(detector=name, grouping=gc, policy=label)
                rows.append(m)
    return pd.DataFrame(rows)


def recall_ci_str(df):
    """'0.961 [0.868, 0.989]' strings for tables."""
    return [f"{r:.3f} [{lo:.2f}, {hi:.2f}]" for r, lo, hi in zip(df["recall_insider"], df["recall_lo"], df["recall_hi"])]
