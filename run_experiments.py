#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments.py
==================
Problem 21 -- Typology of States by Crimes against Women (Clustering Case Study)
B.Sc. (H) Computer Science, Sem VI, University of Delhi -- Machine Learning
Student : Roopesh Kumar (Roll No. 24020570030)

ONE command does everything:
    1. loads the NCRB "Crime against Women" file downloaded from data.gov.in
    2. cleans it, converts counts to rates per lakh women, log-transforms, standardises
    3. screens outliers, runs PCA
    4. clusters the States/UTs with k-means and Ward hierarchical clustering
    5. validates the clusters (silhouette, Davies-Bouldin, Calinski-Harabasz,
       cophenetic correlation, bootstrap stability, ARI between methods)
    6. saves figures, tables (CSV/JSON) and a journal-format PDF paper
       (same layout as the sample: two columns, numbered sections, tables, figures)

USAGE
    pip install numpy pandas scikit-learn scipy matplotlib reportlab openpyxl
    python run_experiments.py --data crime_against_women.csv
    python run_experiments.py --data file.xlsx --years 2018-2022 --college "Your College" \
           --email you@college.du.ac.in --supervisor "Prof. Name"
    python run_experiments.py --demo          # SIMULATED data, only to test that the code runs

Useful options:  --k 4 (force number of clusters)   --pop my_female_population.csv
                 --exclude "Lakshadweep,Daman & Diu"   --out outputs
Every number and figure in the paper is produced by this script (random seed 42).
"""
import argparse, difflib, json, os, re, sys, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore")
SEED = 42
STUDENT = dict(name="Roopesh Kumar", roll="24020570030", problem="21")
TITLE = ("Typology of Indian States and Union Territories by Crimes against Women: "
         "A Clustering Case Study using NCRB Data")
DATA_URL = "https://www.data.gov.in/catalog/crime-against-women"

# ----------------------------------------------------------------------------
# 1. Census-2011 population (total, sex ratio) -> female population per State/UT
#    Values are Census 2011 totals / sex ratios (females per 1000 males).
#    Please verify against the Census tables or override with --pop file.csv
#    (columns: state, female_population).
# ----------------------------------------------------------------------------
CENSUS = {
    "uttar pradesh": (199812341, 912), "maharashtra": (112374333, 929),
    "bihar": (104099452, 918), "west bengal": (91276115, 950),
    "madhya pradesh": (72626809, 931), "tamil nadu": (72147030, 996),
    "rajasthan": (68548437, 928), "karnataka": (61095297, 973),
    "gujarat": (60439692, 919), "andhra pradesh": (84580777, 993),   # undivided
    "andhra pradesh (residual)": (49386799, 997), "telangana": (35193978, 988),
    "odisha": (41974218, 979), "kerala": (33406061, 1084),
    "jharkhand": (32988134, 948), "assam": (31205576, 958),
    "punjab": (27743338, 895), "chhattisgarh": (25545198, 991),
    "haryana": (25351462, 879), "delhi": (16787941, 868),
    "jammu and kashmir": (12541302, 889), "uttarakhand": (10086292, 963),
    "himachal pradesh": (6864602, 972), "tripura": (3673917, 960),
    "meghalaya": (2966889, 989), "manipur": (2855794, 992),
    "nagaland": (1978502, 931), "goa": (1458545, 973),
    "arunachal pradesh": (1383727, 938), "puducherry": (1247953, 1037),
    "mizoram": (1097206, 976), "chandigarh": (1055450, 818),
    "sikkim": (610577, 890), "andaman and nicobar islands": (380581, 876),
    "dadra and nagar haveli": (343709, 774), "daman and diu": (243247, 618),
    "lakshadweep": (64473, 946),
    "ladakh": (274289, 792),            # approximate (Leh + Kargil)
}
ALIASES = {"a and n islands": "andaman and nicobar islands",
           "andaman and nicobar": "andaman and nicobar islands",
           "d and n haveli": "dadra and nagar haveli", "nct of delhi": "delhi",
           "delhi ut": "delhi", "orissa": "odisha", "uttaranchal": "uttarakhand",
           "pondicherry": "puducherry", "jammu kashmir": "jammu and kashmir",
           "chhatisgarh": "chhattisgarh", "dadra and nagar haveli and daman and diu": "DNH_DD"}


def female_pop(key):
    t, sr = CENSUS[key]
    return t * sr / (1000.0 + sr)


def norm_state(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def pretty(key):
    k = key.title().replace(" And ", " & ").replace(" Of ", " of ")
    return k


# ----------------------------------------------------------------------------
# 2. Loading and tidying the NCRB table (wide or long layout, CSV or Excel)
# ----------------------------------------------------------------------------
HEAD_RULES = [
    (r"total|all crimes|^sl|^s\.?no|^sr", None),
    (r"murder.*rape|rape.*murder", "Murder with rape/gang rape"),
    (r"attempt.*rape", "Attempt to commit rape"),
    (r"dowry\s*death", "Dowry deaths"),
    (r"dowry\s*prohib|\bd\.?p\.? act", "Dowry Prohibition Act"),
    (r"cruelty|husband|498", "Cruelty by husband/relatives"),
    (r"kidnap|abduct", "Kidnapping & abduction"),
    (r"assault|outrage", "Assault on women (modesty)"),
    (r"insult", "Insult to modesty"),
    (r"immoral|traffick", "Immoral traffic & trafficking"),
    (r"acid", "Acid attack"),
    (r"abet", "Abetment to suicide"),
    (r"stalk", "Stalking"),
    (r"voyeur", "Voyeurism"),
    (r"cyber", "Cyber crimes against women"),
    (r"domestic|pwdva", "Domestic violence (PWDVA)"),
    (r"indecent", "Indecent representation"),
    (r"\brape\b|^rape", "Rape"),
]


def canon_head(name):
    n = str(name).lower().strip()
    for pat, canon in HEAD_RULES:
        if re.search(pat, n):
            return canon
    return None


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip()
                         .replace({"-": np.nan, "--": np.nan, "NA": np.nan, "nan": np.nan, "": np.nan}),
                         errors="coerce")


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        raw = pd.read_excel(path, header=None)
    else:
        raw = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                raw = pd.read_csv(path, header=None, encoding=enc, dtype=str, skip_blank_lines=True,
                                  on_bad_lines="skip", engine="python")
                break
            except Exception:
                continue
        if raw is None:
            sys.exit("Could not read the data file.")
    hdr = 0
    for i in range(min(15, len(raw))):                       # find the real header row
        vals = [str(v).lower() for v in raw.iloc[i].tolist()]
        if any(re.search(r"state|\bu\.?t\b|area", v) for v in vals) and sum(v != "nan" for v in vals) >= 3:
            hdr = i
            break
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[hdr].tolist()]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df.reset_index(drop=True)


def build_state_table(df, args, log):
    """Return DataFrame: state x crime head (RATE PER LAKH WOMEN) + meta dict."""
    meta = {}
    cols = list(df.columns)
    scand = [c for c in cols if re.search(r"state|\bu\.?t\b|area|region", c, re.I)]
    if not scand:
        scand = [c for c in cols if df[c].dtype == object][:1]
    scol = scand[0]
    df = df[df[scol].notna()].copy()
    df = df[~df[scol].astype(str).str.lower().str.contains(r"\btotal\b|all india|all-india|grand")]
    df["_state"] = df[scol].map(norm_state)
    ycol = next((c for c in cols if re.fullmatch(r"(?i)\s*year\s*", c) or re.search(r"(?i)^year", c)), None)
    if ycol is not None:
        df["_year"] = pd.to_numeric(df[ycol].astype(str).str.extract(r"((?:19|20)\d\d)")[0], errors="coerce")
        if args.years:
            a, b = [int(x) for x in args.years.split("-")] if "-" in args.years else (int(args.years),) * 2
            df = df[(df["_year"] >= a) & (df["_year"] <= b)]
        meta["years"] = f"{int(df['_year'].min())}-{int(df['_year'].max())}" if df["_year"].notna().any() else "n/a"
    else:
        df["_year"] = np.nan
        ym = re.findall(r"(?:19|20)\d\d", os.path.basename(args.data or ""))
        meta["years"] = ym[0] if ym else "as in file"

    # ---- long layout? (one text column holding crime heads)
    head_col = None
    for c in cols:
        if c in (scol, ycol):
            continue
        vals = df[c].dropna().astype(str).unique()
        if 3 <= len(vals) <= 60 and np.mean([canon_head(v) is not None for v in vals]) > 0.5 \
                and to_num(df[c]).notna().mean() < 0.3:
            head_col = c
            break
    yearcols = [c for c in cols if re.fullmatch(r"(?:19|20)\d\d", str(c).strip())]
    if head_col is not None and yearcols:                      # years spread over columns -> melt
        lo, hi = -1, 9999
        if args.years:
            lo, hi = [int(x) for x in args.years.split("-")] if "-" in args.years else (int(args.years),) * 2
        yc = [c for c in yearcols if lo <= int(c) <= hi]
        m_ = df.melt(id_vars=[scol, head_col, "_state"], value_vars=yc, var_name="_year", value_name="_val")
        m_["_year"] = m_["_year"].astype(int)
        m_["_val"] = to_num(m_["_val"])
        m_["_head"] = m_[head_col].map(canon_head)
        m_ = m_[m_["_head"].notna()]
        wide = m_.pivot_table(index=["_state", "_year"], columns="_head", values="_val", aggfunc="sum",
                              dropna=False).reset_index()
        is_rate = False
        meta["years"] = f"{min(yc)}-{max(yc)}"
        meta["layout"] = "long by crime head, years as columns"
    elif head_col is not None:
        vcols = [c for c in cols if c not in (scol, ycol, head_col) and to_num(df[c]).notna().mean() > 0.5]
        pref = [c for c in vcols if re.search(r"(?i)value|cases|number|count|rate|incid", c)]
        vcol = (pref or vcols)[-1]
        df["_head"] = df[head_col].map(canon_head)
        df = df[df["_head"].notna()]
        df["_val"] = to_num(df[vcol])
        wide = df.pivot_table(index=["_state", "_year"], columns="_head", values="_val", aggfunc="sum",
                              dropna=False).reset_index()
        is_rate = bool(re.search(r"(?i)rate", vcol))
        meta["layout"] = f"long (head column '{head_col}', value column '{vcol}')"
    else:
        groups = {}
        for c in cols:
            if c in (scol, ycol) or c.startswith("_"):
                continue
            ch = canon_head(c)
            if ch is None or to_num(df[c]).notna().mean() < 0.3:
                continue
            groups.setdefault(ch, []).append(c)
        if not groups:
            sys.exit("No crime-head columns recognised. Columns found: " + ", ".join(cols))
        used = {c for g in groups.values() for c in g}
        meta["ignored_columns"] = [c for c in cols if c not in used and c not in (scol, ycol) and not c.startswith("_")]
        meta["head_mapping"] = {ch: cs for ch, cs in groups.items()}
        has_rate = any(re.search(r"(?i)rate", c) for g in groups.values() for c in g)
        has_cnt = any(not re.search(r"(?i)rate", c) for g in groups.values() for c in g)
        use_rate = has_rate and not has_cnt
        wide = df[["_state", "_year"]].copy()
        for ch, cs in groups.items():
            cs = [c for c in cs if bool(re.search(r"(?i)rate", c)) == use_rate] or cs
            arr = pd.concat([to_num(df[c]) for c in cs], axis=1)
            yrs = {tuple(re.findall(r"(?:19|20)\d\d", c)) for c in cs}
            wide[ch] = arr.mean(axis=1) if len(yrs) > 1 else arr.sum(axis=1, min_count=1)
        is_rate = use_rate
        meta["layout"] = "wide"
    heads = [c for c in wide.columns if c not in ("_state", "_year")]
    wide = wide.rename(columns={"_state": "state"})
    wide[heads] = wide[heads].fillna(0.0)

    # ---- counts -> rate per lakh women
    if is_rate:
        rate = wide.copy()
        meta["input"] = "rates per lakh women (as supplied)"
    else:
        poptab = {}
        if args.pop:
            pdf = pd.read_csv(args.pop)
            pdf.columns = [c.lower() for c in pdf.columns]
            for _, r in pdf.iterrows():
                poptab[norm_state(r["state"])] = float(r["female_population"])
        else:
            keys = list(CENSUS.keys())
            have_tg = "telangana" in set(wide["state"])
            for k in keys:
                if k == "andhra pradesh (residual)":
                    continue
                poptab[k] = female_pop("andhra pradesh (residual)" if (k == "andhra pradesh" and have_tg) else k)
            poptab["DNH_DD"] = female_pop("dadra and nagar haveli") + female_pop("daman and diu")
        miss = []
        pops = []
        for s in wide["state"]:
            key = s if s in poptab else (difflib.get_close_matches(s, list(poptab), n=1, cutoff=0.82) or [None])[0]
            if key is None:
                miss.append(s)
            pops.append(poptab.get(key, np.nan))
        wide["_pop"] = pops
        log["dropped_states_no_pop"] = sorted(set(miss))
        wide = wide[wide["_pop"].notna()]
        rate = wide.copy()
        rate[heads] = wide[heads].div(wide["_pop"], axis=0) * 1e5
        rate = rate.drop(columns="_pop")
        meta["input"] = "counts, converted to rate per lakh women using Census-2011 female population"
    agg = rate.groupby("state")[heads].mean()          # mean over the selected years
    agg.index = [pretty(s) for s in agg.index]
    return agg, meta


def make_demo(path):
    """SIMULATED data (long layout) -- only to test the pipeline."""
    rng = np.random.default_rng(1)
    states = [k for k in CENSUS if k not in ("andhra pradesh (residual)", "ladakh", "andhra pradesh")]
    heads = ["Rape", "Kidnapping and Abduction of Women", "Dowry Deaths", "Cruelty by Husband or his Relatives",
             "Assault on Women with intent to outrage her Modesty", "Insult to the Modesty of Women",
             "Dowry Prohibition Act", "Immoral Traffic (Prevention) Act", "Total Crimes against Women"]
    base = {"Rape": 4, "Kidnapping and Abduction of Women": 9, "Dowry Deaths": 0.8,
            "Cruelty by Husband or his Relatives": 12, "Assault on Women with intent to outrage her Modesty": 9,
            "Insult to the Modesty of Women": 1.5, "Dowry Prohibition Act": 0.5,
            "Immoral Traffic (Prevention) Act": 0.3}
    prof = rng.integers(0, 4, len(states))
    mult = np.array([[0.5] * 8, [1.0, 2.2, 0.6, 1.0, 1.4, 1.0, 0.4, 1.0],
                     [1.2, 0.8, 1.6, 2.3, 0.9, 0.8, 2.0, 0.8], [2.2, 1.4, 0.6, 1.3, 2.2, 3.0, 0.5, 1.5]])
    rows = []
    for s, p in zip(states, prof):
        fp = female_pop(s)
        for y in (2020, 2021, 2022):
            tot = 0
            for j, h in enumerate(heads[:-1]):
                lam = base[h] * mult[p][j] * rng.lognormal(0, 0.25) * fp / 1e5
                v = rng.poisson(lam)
                tot += v
                rows.append([pretty(s), y, h, v])
            rows.append([pretty(s), y, heads[-1], tot])
    rows.append(["Total (All India)", 2020, "Rape", 99999])
    pd.DataFrame(rows, columns=["State/UT", "Year", "Crime Head", "Cases Reported"]).to_csv(path, index=False)


# ----------------------------------------------------------------------------
# 3. Analysis
# ----------------------------------------------------------------------------
def modified_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / mad if mad > 0 else np.zeros_like(x)


def run_analysis(rate, args, meta, log):
    R = {"meta": meta, "log": log, "run_date": datetime.date.today().isoformat()}
    # drop heads that are (almost) always zero, keep the rest
    nz = (rate > 0).mean()
    keep = [h for h in rate.columns if nz[h] >= args.min_nonzero and rate[h].std() > 0]
    R["dropped_heads"] = [h for h in rate.columns if h not in keep]
    rate = rate[keep]
    if args.exclude:
        ex = {norm_state(x) for x in args.exclude.split(",")}
        drop = [s for s in rate.index if norm_state(s) in ex]
        rate = rate.drop(index=drop)
        R["excluded_states"] = drop
    n, m = rate.shape
    R.update(n_states=n, n_heads=m, heads=list(rate.columns))
    R["desc"] = [dict(head=h, mean=float(rate[h].mean()), sd=float(rate[h].std()), median=float(rate[h].median()),
                      max=float(rate[h].max()), argmax=rate[h].idxmax(), skew=float(stats.skew(rate[h])))
                 for h in rate.columns]
    # ---- outliers
    L = np.log1p(rate)
    out = {}
    for h in rate.columns:
        mz = modified_z(L[h].values)
        for s, z in zip(rate.index, mz):
            if abs(z) > 3.5:
                out.setdefault(s, []).append(h)
    q1, q3 = rate.quantile(.25), rate.quantile(.75)
    iqr = q3 - q1
    R["iqr_outlier_pct"] = {h: float(((rate[h] < q1[h] - 1.5 * iqr[h]) | (rate[h] > q3[h] + 1.5 * iqr[h])).mean() * 100)
                            for h in rate.columns}
    R["outliers"] = {s: v for s, v in out.items()}
    R["skew_before"] = float(np.mean([stats.skew(rate[h]) for h in rate.columns]))
    R["skew_after"] = float(np.mean([stats.skew(L[h]) for h in rate.columns]))
    # ---- scaling
    X = StandardScaler().fit_transform(L.values)
    R["corr"] = pd.DataFrame(L.corr()).round(3).to_dict()
    cm = L.corr().values
    iu = np.triu_indices(m, 1)
    R["corr_mean"] = float(cm[iu].mean()) if m > 1 else 0.0
    if m > 1:
        j = np.argmax(cm[iu])
        R["corr_max"] = (rate.columns[iu[0][j]], rate.columns[iu[1][j]], float(cm[iu][j]))
        j = np.argmin(cm[iu])
        R["corr_min"] = (rate.columns[iu[0][j]], rate.columns[iu[1][j]], float(cm[iu][j]))
    # ---- PCA
    pca = PCA(random_state=SEED).fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    R["pca"] = dict(evr=evr.tolist(), n90=int(np.searchsorted(cum, .90) + 1), n95=int(np.searchsorted(cum, .95) + 1),
                    pc1=float(evr[0]), pc2=float(evr[1]) if len(evr) > 1 else 0.0)
    comp = pd.DataFrame(pca.components_[:2].T, index=rate.columns, columns=["PC1", "PC2"])
    R["pca"]["load"] = {pc: comp[pc].abs().sort_values(ascending=False).head(3).index.tolist() for pc in comp}
    R["pca"]["loadvals"] = comp.round(3).to_dict()
    P = pca.transform(X)
    # ---- choose k
    kmax = min(8, n - 1)
    Zw = linkage(X, "ward")
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=50, random_state=SEED).fit(X)
        wl = fcluster(Zw, k, "maxclust") - 1
        rows.append(dict(k=k, min_size=int(np.bincount(km.labels_).min()), inertia=float(km.inertia_),
                         sil=float(silhouette_score(X, km.labels_)),
                         dbi=float(davies_bouldin_score(X, km.labels_)),
                         ch=float(calinski_harabasz_score(X, km.labels_)),
                         ward_sil=float(silhouette_score(X, wl)) if len(set(wl)) > 1 else float("nan")))
    R["ksel"] = rows
    cand = [r for r in rows if 3 <= r["k"] <= min(6, kmax)] or rows
    ok = [r for r in cand if r["min_size"] >= args.min_size]      # avoid singleton "clusters"
    R["k_rule_fallback"] = not ok
    pool = ok or cand
    best_sil = max(r["sil"] for r in pool)
    # parsimony: smallest k whose silhouette is within --tol of the best one
    k = args.k or min((r for r in pool if r["sil"] >= best_sil - args.tol), key=lambda r: r["k"])["k"]
    R["k"] = int(k)
    R["k_forced"] = bool(args.k)
    # ---- final k-means (relabelled by overall level so that C1 = lowest)
    km = KMeans(k, n_init=100, random_state=SEED).fit(X)
    tot = rate.sum(axis=1)
    order = pd.Series({c: tot[km.labels_ == c].mean() for c in range(k)}).sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[l] for l in km.labels_])
    cent = km.cluster_centers_[order]
    wl = fcluster(Zw, k, "maxclust") - 1
    # match Ward labels to k-means labels via the same ordering rule
    word = pd.Series({c: tot[wl == c].mean() for c in range(k)}).sort_values().index.tolist()
    wlab = np.array([word.index(l) for l in wl])
    R["compare"] = dict(
        km_sil=float(silhouette_score(X, lab)), km_dbi=float(davies_bouldin_score(X, lab)),
        km_ch=float(calinski_harabasz_score(X, lab)),
        ward_sil=float(silhouette_score(X, wlab)), ward_dbi=float(davies_bouldin_score(X, wlab)),
        ward_ch=float(calinski_harabasz_score(X, wlab)),
        ari=float(adjusted_rand_score(lab, wlab)), nmi=float(normalized_mutual_info_score(lab, wlab)),
        agree_pct=float((lab == wlab).mean() * 100))
    coph = {}
    for meth in ("ward", "average", "complete", "single"):
        Zm = linkage(X, meth)
        coph[meth] = dict(coph=float(cophenet(Zm, pdist(X))[0]),
                          sil=float(silhouette_score(X, fcluster(Zm, k, "maxclust"))) if len(set(fcluster(Zm, k, "maxclust"))) > 1 else float("nan"))
    R["linkage"] = coph
    # ---- stability (80% subsamples)
    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(args.boot):
        idx = rng.choice(n, size=max(k + 1, int(.8 * n)), replace=False)
        kb = KMeans(k, n_init=10, random_state=int(rng.integers(1_000_000))).fit(X[idx])
        aris.append(adjusted_rand_score(lab[idx], kb.labels_))
    R["stability"] = dict(mean=float(np.mean(aris)), sd=float(np.std(aris)), B=args.boot,
                          p05=float(np.percentile(aris, 5)))
    # ---- PCA-space clustering & outlier-removed sensitivity
    npc = R["pca"]["n90"]
    kp = KMeans(k, n_init=50, random_state=SEED).fit(P[:, :npc])
    R["pca_ari"] = float(adjusted_rand_score(lab, kp.labels_))
    R["pca_npc"] = int(npc)
    out_states = [s for s in rate.index if s in out]
    R["ari_no_outliers"] = None
    if out_states and n - len(out_states) > 2 * k:
        mask = ~rate.index.isin(out_states)
        X2 = StandardScaler().fit_transform(L.values[mask])
        k2 = KMeans(k, n_init=50, random_state=SEED).fit(X2)
        R["ari_no_outliers"] = float(adjusted_rand_score(lab[mask], k2.labels_))
        R["n_no_outliers"] = int(mask.sum())
    # raw (no log) sensitivity
    Xr = StandardScaler().fit_transform(rate.values)
    kr = KMeans(k, n_init=50, random_state=SEED).fit(Xr)
    R["ari_raw"] = float(adjusted_rand_score(lab, kr.labels_))
    # ---- profiles
    sil_s = silhouette_samples(X, lab)
    R["weak_states"] = [rate.index[i] for i in np.argsort(sil_s) if sil_s[i] < 0.1][:8]
    F, p = f_classif(X, lab)
    R["anova"] = {h: dict(F=float(f), p=float(pp)) for h, f, pp in zip(rate.columns, F, p)}
    zc = pd.DataFrame(cent, columns=rate.columns)
    clusters = []
    for c in range(k):
        mem = list(rate.index[lab == c])
        z = zc.loc[c]
        top = [h for h in z.sort_values(ascending=False).index if z[h] > 0.5][:2]
        low = [h for h in z.sort_values().index if z[h] < -0.5][:2]
        clusters.append(dict(id=c + 1, n=len(mem), states=sorted(mem),
                             mean_total=float(tot[lab == c].mean()),
                             sil=float(sil_s[lab == c].mean()),
                             profile={h: float(rate.loc[lab == c, h].mean()) for h in rate.columns},
                             z={h: float(z[h]) for h in rate.columns}, top=top, low=low))
    R["clusters"] = clusters
    R["national_mean_total"] = float(tot.mean())
    R["extremes"] = dict(top_total=tot.sort_values(ascending=False).head(3).round(1).to_dict(),
                         bottom_total=tot.sort_values().head(3).round(1).to_dict())
    arrays = dict(rate=rate, L=L, X=X, P=P, lab=lab, wlab=wlab, Zw=Zw, evr=evr, sil=sil_s, comp=comp)
    return R, arrays


# ----------------------------------------------------------------------------
# 4. Figures (column-width, 3.45 inch)
# ----------------------------------------------------------------------------
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
W = 3.45


def make_figures(R, A, figdir):
    os.makedirs(figdir, exist_ok=True)
    rate, X, P, lab, k = A["rate"], A["X"], A["P"], A["lab"], R["k"]
    F = {}

    # Fig 1 -- mean rate per head
    d = pd.DataFrame(R["desc"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(W, 0.28 * len(d) + 0.8))
    ax.barh(d["head"], d["mean"], color="#1f4e79", xerr=d["sd"], error_kw=dict(lw=.6, capsize=1.5, ecolor="#777"))
    ax.set_xlabel("Mean rate per lakh women (bars: SD across States/UTs)")
    ax.margins(y=.02)
    fig.tight_layout()
    F["fig_rates"] = _save(fig, figdir, "fig1_mean_rates.png")

    # Fig 2 -- correlation
    C = A["L"].corr()
    fig, ax = plt.subplots(figsize=(W, W * .95))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)))
    ax.set_yticks(range(len(C)))
    ax.set_xticklabels([_short(c) for c in C.columns], rotation=60, ha="right", fontsize=5.5)
    ax.set_yticklabels([_short(c) for c in C.index], fontsize=5.5)
    if len(C) <= 12:
        for i in range(len(C)):
            for j in range(len(C)):
                ax.text(j, i, f"{C.values[i, j]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=4.8,
                        color="white" if abs(C.values[i, j]) > .6 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, fraction=.046, pad=.03)
    fig.tight_layout()
    F["fig_corr"] = _save(fig, figdir, "fig2_correlation.png")

    # Fig 3 -- scree
    evr = A["evr"]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr * 100, color="#1f4e79", label="Individual")
    ax.plot(x, np.cumsum(evr) * 100, "o-", color="#d95f02", ms=3, lw=1, label="Cumulative")
    ax.axhline(90, ls="--", lw=.6, color="grey")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(x)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    F["fig_scree"] = _save(fig, figdir, "fig3_scree.png")

    # Fig 4 -- choice of k
    ks = pd.DataFrame(R["ksel"])
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.9))
    axs[0].plot(ks.k, ks.inertia, "o-", ms=3, color="#1f4e79")
    axs[0].set_title("Inertia (elbow)")
    axs[1].plot(ks.k, ks.sil, "o-", ms=3, color="#d95f02", label="k-means")
    axs[1].plot(ks.k, ks.ward_sil, "s--", ms=3, color="#2ca02c", label="Ward")
    axs[1].set_title("Silhouette")
    axs[1].legend(frameon=False, fontsize=5)
    axs[2].plot(ks.k, ks.dbi, "o-", ms=3, color="#7f3b8f")
    axs[2].set_title("Davies-Bouldin")
    for a in axs:
        a.set_xlabel("k")
        a.axvline(k, ls=":", lw=.8, color="red")
        a.set_xticks(ks.k)
    fig.tight_layout()
    F["fig_k"] = _save(fig, figdir, "fig4_choose_k.png")

    # Fig 5 -- PCA scatter
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for c in range(k):
        m = lab == c
        ax.scatter(P[m, 0], P[m, 1], s=16, color=PAL[c % 8], label=f"C{c + 1}", edgecolor="white", lw=.3)
    for i, s in enumerate(rate.index):
        ax.annotate(_abbr(s), (P[i, 0], P[i, 1]), fontsize=4.6, xytext=(2, 2), textcoords="offset points", color="#333")
    ax.set_xlabel(f"PC1 ({R['pca']['pc1'] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({R['pca']['pc2'] * 100:.1f}% variance)")
    ax.axhline(0, lw=.4, color="grey")
    ax.axvline(0, lw=.4, color="grey")
    ax.legend(frameon=False, ncol=min(k, 6), loc="best")
    fig.tight_layout()
    F["fig_pca"] = _save(fig, figdir, "fig5_pca_scatter.png")

    # Fig 6 -- dendrogram
    n = len(rate)
    fig, ax = plt.subplots(figsize=(W, max(3.0, 0.115 * n + .6)))
    dendrogram(A["Zw"], labels=list(rate.index), orientation="right", leaf_font_size=5.3, ax=ax,
               color_threshold=A["Zw"][-(k - 1), 2] - 1e-9 if k > 1 else 0, above_threshold_color="#888")
    ax.set_xlabel("Ward linkage distance")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    F["fig_dendro"] = _save(fig, figdir, "fig6_dendrogram.png")

    # Fig 7 -- cluster z-profile heatmap
    Z = np.array([[c["z"][h] for c in R["clusters"]] for h in R["heads"]])
    fig, ax = plt.subplots(figsize=(W, 0.27 * len(R["heads"]) + 0.9))
    vmax = max(1.5, np.abs(Z).max())
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{c['id']}\n(n={c['n']})" for c in R["clusters"]])
    ax.set_yticks(range(len(R["heads"])))
    ax.set_yticklabels(R["heads"], fontsize=6)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(Z[i, j]) > .9 * vmax else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, fraction=.05, pad=.03, label="Cluster mean (z-score)")
    fig.tight_layout()
    F["fig_profile"] = _save(fig, figdir, "fig7_cluster_profiles.png")

    # Fig 8 -- silhouette plot
    fig, ax = plt.subplots(figsize=(W, max(3.0, .12 * n + .6)))
    y = 0
    sil = A["sil"]
    yt, yl = [], []
    for c in range(k):
        idx = np.where(lab == c)[0]
        idx = idx[np.argsort(sil[idx])]
        ax.barh(range(y, y + len(idx)), sil[idx], color=PAL[c % 8], height=.85)
        for j, i in enumerate(idx):
            yt.append(y + j)
            yl.append(_abbr(rate.index[i]))
        y += len(idx) + 1
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=4.8)
    ax.axvline(R["compare"]["km_sil"], ls="--", color="red", lw=.7)
    ax.set_xlabel("Silhouette coefficient (dashed: mean)")
    ax.invert_yaxis()
    fig.tight_layout()
    F["fig_sil"] = _save(fig, figdir, "fig8_silhouette.png")

    # Fig 9 -- pipeline
    fig, ax = plt.subplots(figsize=(W, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    steps = [("1  Raw data", "NCRB crimes against women\n(data.gov.in), States/UTs"),
             ("2  Tidy + normalise", "map crime heads, drop totals,\ncount -> rate per lakh women"),
             ("3  Transform", "log(1+x), outlier screen\n(modified z > 3.5), z-score"),
             ("4  Explore", "correlation, PCA (scree,\nbiplot)"),
             ("5  Cluster", "k-means (k = 2..8) and Ward\nhierarchical clustering"),
             ("6  Validate + profile", "silhouette, DBI, CH, cophenetic,\nbootstrap ARI, cluster profiles")]
    h = 1.35
    for i, (t, s) in enumerate(steps):
        y = 9.6 - (i + 1) * 1.62
        ax.add_patch(FancyBboxPatch((.4, y), 9.2, h, boxstyle="round,pad=.02,rounding_size=.15", fc="#eef3f8", ec="#1f4e79", lw=.8))
        ax.text(.7, y + h - .3, t, fontsize=7, fontweight="bold", va="top", color="#1f4e79")
        ax.text(.7, y + .08, s, fontsize=6, va="bottom", color="#222")
        if i < len(steps) - 1:
            ax.annotate("", xy=(5, y - .02), xytext=(5, y - .26), arrowprops=dict(arrowstyle="-|>", lw=.8, color="#1f4e79"))
    F["fig_pipe"] = _save(fig, figdir, "fig0_pipeline.png")
    return F


def _save(fig, d, name):
    p = os.path.join(d, name)
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def _short(s):
    return s if len(s) <= 18 else s[:16] + "."


ABBR = {"Andaman & Nicobar Islands": "A&N", "Andhra Pradesh": "AP", "Arunachal Pradesh": "Arunachal",
        "Dadra & Nagar Haveli": "D&NH", "Daman & Diu": "D&D", "Himachal Pradesh": "HP", "Jammu & Kashmir": "J&K",
        "Madhya Pradesh": "MP", "Uttar Pradesh": "UP", "West Bengal": "WB", "Tamil Nadu": "TN"}


def _abbr(s):
    return ABBR.get(s, s)


# ----------------------------------------------------------------------------
# 5. Journal-format PDF (reportlab) -- mimics the sample layout
# ----------------------------------------------------------------------------
def build_pdf(R, A, F, outpath, args):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table,
                                    TableStyle, FrameBreak, NextPageTemplate, KeepTogether)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader

    fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for nm, fn in (("DS", "DejaVuSerif.ttf"), ("DS-B", "DejaVuSerif-Bold.ttf"), ("DS-I", "DejaVuSerif-Italic.ttf"),
                   ("DS-BI", "DejaVuSerif-BoldItalic.ttf")):
        pdfmetrics.registerFont(TTFont(nm, os.path.join(fd, fn)))
    pdfmetrics.registerFontFamily("DS", normal="DS", bold="DS-B", italic="DS-I", boldItalic="DS-BI")

    demo = args.demo
    NAVY = colors.HexColor("#1f4e79")
    body = ParagraphStyle("body", fontName="DS", fontSize=7.9, leading=10.0, alignment=TA_JUSTIFY, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=body, fontName="DS-B", fontSize=9.6, leading=12, textColor=NAVY, spaceBefore=6,
                        spaceAfter=3, alignment=TA_LEFT, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=body, fontName="DS-BI", fontSize=8.2, leading=10.5, spaceBefore=3, spaceAfter=2,
                        alignment=TA_LEFT, keepWithNext=1)
    cap = ParagraphStyle("cap", parent=body, fontSize=6.9, leading=8.6, spaceBefore=1, spaceAfter=5)
    cell = ParagraphStyle("cell", parent=body, fontSize=6.5, leading=7.8, alignment=TA_LEFT, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="DS-B")
    ttl = ParagraphStyle("ttl", parent=body, fontName="DS-B", fontSize=14.5, leading=18, alignment=TA_CENTER,
                         spaceAfter=5, textColor=NAVY)
    auth = ParagraphStyle("auth", parent=body, fontName="DS-B", fontSize=9.5, alignment=TA_CENTER, spaceAfter=1)
    aff = ParagraphStyle("aff", parent=body, fontSize=7, leading=9, alignment=TA_CENTER, spaceAfter=1)
    absS = ParagraphStyle("abs", parent=body, fontSize=7.8, leading=9.9, spaceAfter=2)
    refS = ParagraphStyle("ref", parent=body, fontSize=6.7, leading=8.3, leftIndent=11, firstLineIndent=-11,
                          alignment=TA_LEFT, spaceAfter=1.5)

    PW, PH = A4
    lm = rm = 14 * mm
    gap = 6 * mm
    colw = (PW - lm - rm - gap) / 2.0
    top = PH - 16 * mm
    bot = 15 * mm

    def header_footer(c, d):
        c.saveState()
        c.setFont("DS-B", 6.8)
        c.setFillColor(NAVY)
        c.drawString(lm, PH - 10 * mm, "ML Application Assignment, B.Sc. (H) Computer Science, Sem VI (University of Delhi)")
        c.drawRightString(PW - rm, PH - 10 * mm, "Journal-format submission")
        c.setStrokeColor(NAVY)
        c.setLineWidth(.6)
        c.line(lm, PH - 11.5 * mm, PW - rm, PH - 11.5 * mm)
        c.setFillColor(colors.HexColor("#555555"))
        c.setFont("DS", 6.3)
        c.drawString(lm, 9 * mm, f"{STUDENT['name']} | Roll No. {STUDENT['roll']} | Problem No. {STUDENT['problem']} | Case Study")
        c.drawRightString(PW - rm, 9 * mm, f"Page {d.page}")
        c.line(lm, 11.5 * mm, PW - rm, 11.5 * mm)
        if demo:
            c.setFillColor(colors.red)
            c.setFont("DS-B", 8)
            c.drawCentredString(PW / 2, PH - 6.5 * mm, "DEMO RUN - SIMULATED DATA - NOT REAL RESULTS - DO NOT SUBMIT")
        c.restoreState()

    doc = BaseDocTemplate(outpath, pagesize=A4, leftMargin=lm, rightMargin=rm, topMargin=16 * mm, bottomMargin=15 * mm,
                          title=TITLE, author=STUDENT["name"])
    def P(t, s=body):
        return Paragraph(t, s)

    def E(t):
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def fig(path, caption, width=colw, maxh=None):
        w, h = ImageReader(path).getSize()
        wd = width
        ht = wd * h / w
        if maxh and ht > maxh:
            wd, ht = wd * maxh / ht, maxh
        return KeepTogether([Image(path, width=wd, height=ht), P(caption, cap)])

    def table(rows, widths, caption, hdr=1, align_right_from=1):
        data = []
        for i, r in enumerate(rows):
            data.append([Paragraph(str(x), cellb if i < hdr else cell) for x in r])
        t = Table(data, colWidths=widths, repeatRows=hdr)
        t.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), .9, colors.black), ("LINEBELOW", (0, hdr - 1), (-1, hdr - 1), .5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), .9, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("ROWBACKGROUNDS", (0, hdr), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")])]))
        return [P(caption, cap), t, Spacer(1, 5)]

    def eq(text, num):
        t = Table([[Paragraph(text, ParagraphStyle("eq", parent=body, alignment=TA_CENTER, spaceAfter=0, fontName="DS-I")),
                    Paragraph(f"({num})", ParagraphStyle("en", parent=body, alignment=2, spaceAfter=0))]],
                  colWidths=[colw - 30, 30])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        return t

    k, n, m = R["k"], R["n_states"], R["n_heads"]
    cl = R["clusters"]
    cmp_ = R["compare"]
    st = R["stability"]
    yrs = R["meta"].get("years", "as in file")
    heads = R["heads"]
    top_c, low_c = cl[-1], cl[0]

    def silq(s):
        return "strong" if s > .70 else "reasonable" if s > .50 else "weak-to-moderate" if s > .25 else "weak"

    # ---------------- title block
    S = []
    if demo:
        S.append(P("<font color='red'><b>DEMO RUN: this PDF was produced from SIMULATED data to test the code. "
                   "Run the script on the real NCRB file to obtain your submission.</b></font>", aff))
    S += [P(E(TITLE), ttl), P(STUDENT["name"], auth),
          P(f"Department of Computer Science, {E(args.college)}, University of Delhi, New Delhi, India | "
            f"Roll No.: {STUDENT['roll']} | E-mail: {E(args.email)}", aff),
          P(f"Course: Machine Learning (B.Sc. (H) Computer Science, Semester VI) | Supervisor: {E(args.supervisor)} | "
            f"Problem No. {STUDENT['problem']} from ML_Application_Problems.xlsx | Article type: Case Study", aff),
          Spacer(1, 4)]
    best_ex = ", ".join(top_c["states"][:4])
    abstract = (
        f"<b>Abstract.</b> Crimes against women in India can differ not only in volume but also in kind, for example "
        f"domestic-violence-related offences (cruelty by husband or relatives, dowry deaths) versus abduction or assault. "
        f"This case study asks whether the State-wise National Crime Records Bureau (NCRB) data can be "
        f"summarised into a small number of interpretable <i>typologies</i> using the unsupervised methods of an "
        f"undergraduate machine-learning syllabus. We use the open NCRB table <i>Crime against Women</i> from the Open "
        f"Government Data Platform (period: {E(yrs)}), giving {n} States/Union Territories described by {m} crime heads "
        f"expressed as rates per lakh women. After log transformation, standardisation and outlier screening, k-means and "
        f"Ward hierarchical clustering were compared and PCA was used for visualisation. The silhouette criterion "
        f"selected k = {k} clusters (mean silhouette {cmp_['km_sil']:.3f}, a {silq(cmp_['km_sil'])} structure); the two "
        f"algorithms agreed with an adjusted Rand index (ARI) of {cmp_['ari']:.2f}, and bootstrap sub-sampling gave a "
        f"mean stability ARI of {st['mean']:.2f}. The highest-rate cluster (C{top_c['id']}, {top_c['n']} States/UTs, "
        f"e.g. {E(best_ex)}) has a mean combined rate of {top_c['mean_total']:.1f} per lakh women against "
        f"{low_c['mean_total']:.1f} in the lowest-rate cluster. Because the figures are <i>registered</i> crimes, "
        f"differences reflect reporting and registration practice as well as incidence; the clusters are therefore "
        f"descriptive typologies for targeting awareness and policing resources, not a ranking of State safety.")
    S.append(P(abstract, absS))
    S.append(P("<b>Keywords:</b> crimes against women; NCRB; India; k-means; hierarchical clustering; PCA; silhouette score; "
               "SDG 5; SDG 16", absS))
    th = 5 * mm
    for fl in S:
        w_, h_ = fl.wrap(PW - lm - rm, 1000)
        th += h_ + getattr(getattr(fl, "style", None), "spaceBefore", 0) + getattr(getattr(fl, "style", None), "spaceAfter", 0)
    title_h = th
    f_title = Frame(lm, top - title_h, PW - lm - rm, title_h, id="t", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f1a = Frame(lm, bot, colw, top - title_h - bot, id="c1a", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f2a = Frame(lm + colw + gap, bot, colw, top - title_h - bot, id="c2a", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f1 = Frame(lm, bot, colw, top - bot, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f2 = Frame(lm + colw + gap, bot, colw, top - bot, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate("first", [f_title, f1a, f2a], onPage=header_footer),
                          PageTemplate("later", [f1, f2], onPage=header_footer)])

    S.append(FrameBreak())
    S.append(NextPageTemplate("later"))

    # ---------------- 1 Introduction
    S += [P("1 Introduction", h1),
          P("Violence against women is a persistent public-safety and human-rights concern in India. The NCRB publishes "
            "State-wise counts of registered crimes under heads such as rape, kidnapping and abduction, dowry deaths, "
            "cruelty by husband or relatives, assault with intent to outrage modesty and insult to modesty [1]. Headline "
            "totals can hide differences in the <i>mix</i> of crimes between States, yet a policy "
            "response suited to a State dominated by domestic-violence offences differs from one where abduction or public "
            "harassment dominate. This study is linked to SDG 5 (Gender Equality) and SDG 16 (Peace, Justice and Strong "
            "Institutions)."),
          P("The problem is framed as unsupervised learning. Instead of predicting a label, we ask which States have "
            "similar crime-rate <i>profiles</i> and how many natural groups exist. Our contributions are: (i) a "
            "population-normalised representation (rate per lakh women) with explicit skew and outlier treatment; "
            "(ii) a comparison of k-means and Ward hierarchical clustering with PCA visualisation; (iii) validation by "
            "internal indices, cophenetic correlation and bootstrap stability rather than a single run; and (iv) a "
            "profile of each cluster with an explicit discussion of reporting bias. Section 2 reviews related work, "
            "Section 3 describes the data, Sections 4-5 the methods and set-up, Section 6 the results and Section 7 the "
            "discussion, followed by conclusions."),
          P("2 Related Work", h1),
          P("Clustering has often been applied to Indian crime data. Agarwal et al. [5] applied k-means to crime data, and Goel and Yadav [6] built crime-susceptibility indexes "
            "for crimes against Indian women with principal component analysis. More recently, Dutta and Pramanick [7] "
            "applied k-means to NCRB panel data (2001-2023) on five major crime heads and reported four groups ranging "
            "from low to high crime, with Uttar Pradesh, Madhya Pradesh, Rajasthan, West Bengal and Maharashtra among the "
            "high-crime States. Those studies largely work with counts or total crime levels; we instead normalise by "
            "female population, cluster on the <i>profile</i> of crime heads and test whether the grouping is stable "
            "across algorithms and resamples. The Justice Verma Committee [3] and the National Family Health Survey [4] "
            "both stress that registered crime captures only part of the violence women experience, which motivates our "
            "cautious interpretation. The algorithms follow [8]-[10]; PCA follows [11]; agreement between partitions is measured "
            "with the adjusted Rand index [12]; and all models were implemented with scikit-learn [13]."),
          ]
    # ---------------- 3 Data
    d0 = R["desc"]
    S += [P("3 Dataset and Preprocessing", h1), P("3.1 Source and description", h2),
          P(f"The data are the NCRB table <i>Crime against Women</i> published on the Open Government Data Platform India "
            f"(OGD) [2] (file: {E(os.path.basename(args.data) if args.data else 'demo')}; period used: {E(yrs)}; "
            f"accessed {R['run_date']}). After tidying, the table has {n} States/UTs and {m} crime heads: "
            f"{E(', '.join(heads))}. Rows for national or sub-total entries were removed. Input type: "
            f"{E(R['meta'].get('input', ''))}. Table 1 summarises the variables.")]
    rows = [["Crime head", "Mean", "SD", "Median", "Max (State)"]]
    for r in d0:
        rows.append([E(r["head"]), f"{r['mean']:.2f}", f"{r['sd']:.2f}", f"{r['median']:.2f}",
                     f"{r['max']:.1f} ({E(_abbr(r['argmax']))})"])
    S += table(rows, [colw * .36, colw * .12, colw * .12, colw * .14, colw * .26],
               "Table 1. Crime heads used as features: rate per lakh women across States/UTs (averaged over the selected years).")
    if R["dropped_heads"]:
        S.append(P(f"Heads excluded because they were zero in most States: {E(', '.join(R['dropped_heads']))}."))
    if R["log"].get("dropped_states_no_pop"):
        S.append(P(f"Entries without a matching Census-2011 population were dropped: "
                   f"{E(', '.join(R['log']['dropped_states_no_pop']))}."))
    S.append(fig(F["fig_rates"], "Fig. 1. Mean rate per lakh women for each crime head (error bars: standard deviation across States/UTs)."))
    S += [P("3.2 Rate normalisation, skew and outliers", h2),
          P("Raw counts mostly measure population size, so counts were converted to rates per lakh women, "
            "r = 10<super>5</super> x count / female population (Census 2011)"
            + (" (rates supplied in the file were used directly)" if "rates" in R['meta'].get("input", "") else "") + ":"),
          eq("r<sub>ij</sub> = 10<super>5</super> · c<sub>ij</sub> / N<sub>i</sub><super>F</super>", 1)]
    nout = len(R["outliers"])
    ol = "; ".join(f"{E(s)} ({E(', '.join(_short(h) for h in v[:2]))})" for s, v in list(R["outliers"].items())[:6])
    S += [P(f"Crime rates are right-skewed (mean skewness {R['skew_before']:.2f}). Because high-rate States are real "
            f"observations rather than errors, they were not deleted; a log transform x' = ln(1 + x) reduced the mean "
            f"skewness to {R['skew_after']:.2f}. Outliers were screened with the modified z-score "
            f"M = 0.6745 (x - median) / MAD, flagging |M| > 3.5 [14]. "
            + (f"{nout} States/UTs were flagged on at least one head: {ol}. "
               if nout else "No State/UT was flagged on the log scale. ")
            + "Features were then standardised, z = (x - mean) / sd, so that no crime head dominates by scale."),
          eq("M<sub>i</sub> = 0.6745 (x<sub>i</sub> − median) / MAD", 2)]
    cm = R.get("corr_max")
    S += [P("3.3 Correlation and PCA", h2),
          P((f"The most strongly correlated pair of heads (log scale) is {E(cm[0])} and {E(cm[1])} (r = {cm[2]:.2f}); the mean "
             f"pairwise correlation is {R['corr_mean']:.2f} (Fig. 2). " if cm else "")
            + f"PCA on the standardised matrix needed {R['pca']['n90']} components for 90% and {R['pca']['n95']} for 95% "
              f"of the variance; PC1 alone explains {R['pca']['pc1'] * 100:.1f}% and PC2 {R['pca']['pc2'] * 100:.1f}% "
              f"(Fig. 3). PC1 is driven mainly by {E(', '.join(R['pca']['load']['PC1']))}, PC2 by "
              f"{E(', '.join(R['pca']['load']['PC2']))}."),
          fig(F["fig_corr"], "Fig. 2. Pearson correlation of log-transformed crime rates across States/UTs."),
          fig(F["fig_scree"], "Fig. 3. Individual and cumulative explained variance of the principal components.")]
    # ---------------- 4 Methods
    S += [P("4 Methodology", h1),
          P("Fig. 4 summarises the workflow. All scaling parameters are estimated on the full State table because "
            "there is no train/test split in unsupervised profiling; robustness is instead assessed by resampling."),
          fig(F["fig_pipe"], "Fig. 4. Methodology pipeline used in this study.", maxh=62 * mm),
          P("4.1 k-means", h2),
          P("k-means [8] partitions the standardised vectors x<sub>i</sub> into k groups by minimising the within-cluster "
            "sum of squares using Lloyd's iterations. 100 random initialisations (k-means++) were used and the best "
            "solution kept (seed 42)."),
          eq("J = Σ<sub>c=1..k</sub> Σ<sub>x∈C<sub>c</sub></sub> ‖x − μ<sub>c</sub>‖<super>2</super>", 3),
          P("4.2 Hierarchical clustering", h2),
          P("Agglomerative clustering starts with each State as its own cluster and repeatedly merges the pair whose merger "
            "gives the smallest increase in within-cluster variance (Ward's criterion [9]); the result is a dendrogram "
            "that can be cut at any k. Average, complete and single linkage were also compared using the cophenetic "
            "correlation [17], i.e. how faithfully the dendrogram preserves the original pairwise distances."),
          eq("Δ(A,B) = [n<sub>A</sub>n<sub>B</sub> / (n<sub>A</sub>+n<sub>B</sub>)] ‖μ<sub>A</sub> − μ<sub>B</sub>‖<super>2</super>", 4),
          P("4.3 Choosing and validating k", h2),
          P("For k = 2..8 we computed the inertia (elbow), the mean silhouette coefficient [10], the Davies-Bouldin index "
            "[16] (lower is better) and the Calinski-Harabasz index [15] (higher is better). For a State i, a(i) is the mean "
            "distance to its own cluster and b(i) the mean distance to the nearest other cluster:"),
          eq("s(i) = (b(i) − a(i)) / max{a(i), b(i)}", 5),
          P(f"A typology needs at least three groups, so k was chosen as the value in 3..6 with the highest mean "
            f"silhouette among solutions whose smallest cluster has at least {args.min_size} States/UTs (to avoid single-territory "
            f"'clusters'); if several values were within {args.tol:.2f} of the best silhouette, the smallest such k was "
            f"preferred for parsimony" + (f" (overridden manually: k = {k})" if R['k_forced'] else "") + ". Agreement between k-means and Ward was "
            "measured with ARI and NMI. Stability was assessed with "
            f"{st['B']} random 80% sub-samples: k-means was re-fitted on each and its partition compared with the "
            "reference partition on the sampled States using ARI.")]
    # ---------------- 5 setup
    S += [P("5 Experimental Setup", h1),
          P(f"Environment: Python, scikit-learn, SciPy, pandas and Matplotlib; random seed 42 for all stochastic steps. "
            f"The single script run_experiments.py regenerates every number, table and figure of this paper from the "
            f"downloaded NCRB file. Final clusters are labelled C1..C{k} in increasing order of the sum of the "
            f"{m} crime-head rates, so that C1 is the lowest-rate and C{k} the highest-rate group.")]
    # ---------------- 6 results
    S += [P("6 Results", h1), P("6.1 Choosing the number of clusters", h2)]
    rows = [["k", "Inertia", "Sil. (k-means)", "Sil. (Ward)", "DBI", "CH", "Min. size"]]
    for r in R["ksel"]:
        b = "<b>%s</b>" if r["k"] == k else "%s"
        rows.append([b % r["k"], b % f"{r['inertia']:.1f}", b % f"{r['sil']:.3f}", b % f"{r['ward_sil']:.3f}",
                     b % f"{r['dbi']:.2f}", b % f"{r['ch']:.1f}", b % r["min_size"]])
    S += table(rows, [colw * .07, colw * .16, colw * .2, colw * .17, colw * .11, colw * .15, colw * .14],
               "Table 2. Cluster-validity indices for k-means and Ward clustering (selected k in bold).")
    bs = max(R["ksel"], key=lambda r: r["sil"])
    S += [P(f"The silhouette is highest at k = {bs['k']} ({bs['sil']:.3f}) over the full range"
            + (f", but that solution contains a cluster of fewer than {args.min_size} States/UTs (see the last column), "
               "so it was not used" if bs['k'] != k and bs['min_size'] < args.min_size else "")
            + f"; the selected k = {k} has "
            f"silhouette {cmp_['km_sil']:.3f}, DBI {cmp_['km_dbi']:.2f} and CH {cmp_['km_ch']:.1f}. The elbow curve "
            f"(Fig. 5) was read together with the silhouette and the DBI; with only {n} observations the elbow is rarely "
            f"sharp, so the silhouette and the interpretability of the groups were given priority."),
          fig(F["fig_k"], "Fig. 5. Inertia, silhouette and Davies-Bouldin index versus k (dotted line: selected k)."),
          P("6.2 k-means versus hierarchical clustering", h2)]
    rows = [["Method", "Silhouette", "DBI", "CH", "Cophenetic"],
            ["k-means", f"{cmp_['km_sil']:.3f}", f"{cmp_['km_dbi']:.2f}", f"{cmp_['km_ch']:.1f}", "-"]]
    for meth, nm in (("ward", "Ward"), ("average", "Average"), ("complete", "Complete"), ("single", "Single")):
        L_ = R["linkage"][meth]
        if meth == "ward":
            rows.append([nm, f"{cmp_['ward_sil']:.3f}", f"{cmp_['ward_dbi']:.2f}", f"{cmp_['ward_ch']:.1f}", f"{L_['coph']:.3f}"])
        else:
            rows.append([nm, f"{L_['sil']:.3f}", "-", "-", f"{L_['coph']:.3f}"])
    S += table(rows, [colw * .24, colw * .2, colw * .16, colw * .18, colw * .22],
               f"Table 3. Comparison of clustering algorithms at k = {k}.")
    S += [P(f"The k-means and Ward partitions agree on {cmp_['agree_pct']:.0f}% of States (ARI = {cmp_['ari']:.2f}, "
            f"NMI = {cmp_['nmi']:.2f}). The linkage with the highest cophenetic correlation is "
            f"{max(R['linkage'], key=lambda x: R['linkage'][x]['coph'])} "
            f"({max(v['coph'] for v in R['linkage'].values()):.3f}). Fig. 6 shows the Ward dendrogram."),
          fig(F["fig_dendro"], "Fig. 6. Ward dendrogram of the States/UTs; colours mark the k clusters.", maxh=170 * mm),
          P("6.3 Cluster typology", h2),
          P(f"Fig. 7 projects the States on the first two principal components, which together explain "
            f"{(R['pca']['pc1'] + R['pca']['pc2']) * 100:.1f}% of the variance. Table 4 gives the mean rate of each "
            f"crime head per cluster and Fig. 8 the standardised profile (red: above the all-State average, blue: below)."),
          fig(F["fig_pca"], "Fig. 7. States/UTs on the first two principal components, coloured by k-means cluster.")]
    rows = [["Crime head"] + [f"C{c['id']}" for c in cl] + ["F"]]
    for h in heads:
        rows.append([E(h)] + [f"{c['profile'][h]:.1f}" for c in cl] + [f"{R['anova'][h]['F']:.1f}"])
    rows.append(["<b>Sum of heads</b>"] + [f"<b>{c['mean_total']:.1f}</b>" for c in cl] + [""])
    rows.append(["<b>n (States/UTs)</b>"] + [f"<b>{c['n']}</b>" for c in cl] + [""])
    cw = [colw * .38] + [colw * (.50 / k)] * k + [colw * .12]
    S += table(rows, cw, "Table 4. Mean rate per lakh women by cluster and crime head (F: one-way ANOVA statistic of the head across clusters).")
    S.append(fig(F["fig_profile"], "Fig. 8. Cluster profiles: mean standardised (z-score) value of each crime head."))
    S.append(P("<b>Cluster membership and profile.</b>"))
    for c in cl:
        tops = ", ".join(c["top"]) if c["top"] else "no head clearly above average"
        lows = ", ".join(c["low"]) if c["low"] else "none clearly below average"
        ex = ", ".join(c["states"])
        S.append(P(f"<b>C{c['id']}</b> (n = {c['n']}; mean combined rate {c['mean_total']:.1f}; mean silhouette {c['sil']:.2f}). "
                   f"Distinctively high: {E(tops)}. Distinctively low: {E(lows)}. Members: {E(ex)}."))
    S += [P("6.4 Robustness checks", h2)]
    rows = [["Check", "Result"],
            [f"Bootstrap stability ({st['B']} x 80% samples), ARI", f"{st['mean']:.2f} ± {st['sd']:.2f} (5th pct. {st['p05']:.2f})"],
            ["k-means vs Ward, ARI", f"{cmp_['ari']:.2f}"],
            [f"k-means in PCA space ({R['pca_npc']} PCs, 90% var.) vs original, ARI", f"{R['pca_ari']:.2f}"],
            ["k-means without log transform vs with, ARI", f"{R['ari_raw']:.2f}"]]
    if R.get("ari_no_outliers") is not None:
        rows.append([f"Excluding {len(R['outliers'])} flagged outlier States (n = {R['n_no_outliers']}), ARI",
                     f"{R['ari_no_outliers']:.2f}"])
    S += table(rows, [colw * .66, colw * .34], "Table 5. Robustness of the k-means partition (ARI = adjusted Rand index; 1 = identical).")
    weak = R["weak_states"]
    S += [P(f"The mean bootstrap ARI of {st['mean']:.2f} indicates that the partition is "
            f"{'highly' if st['mean'] > .8 else 'moderately' if st['mean'] > .6 else 'only weakly'} reproducible. "
            + (f"States with silhouette below 0.1 lie between clusters and should be read as borderline: "
               f"{E(', '.join(weak))} (Fig. 9)." if weak else "No State has a silhouette below 0.1.")),
          fig(F["fig_sil"], "Fig. 9. Silhouette coefficient of each State/UT, grouped by cluster.", maxh=165 * mm)]
    # ---------------- 7 discussion
    gaps = [c for c in cl if c["top"]]
    S += [P("7 Discussion", h1),
          P(f"<b>What the typology shows.</b> The States/UTs fall into {k} groups that differ in both level and mix of "
            f"registered crime. The lowest-rate group (C1) has a mean combined rate of {low_c['mean_total']:.1f} per "
            f"lakh women, whereas the highest-rate group (C{k}) has {top_c['mean_total']:.1f}. "
            + (f"The groups are not only ordered by level: for example, C{gaps[0]['id']} is characterised by "
               f"{E(', '.join(gaps[0]['top']))}" + (f", while C{gaps[-1]['id']} is characterised by "
                                                  f"{E(', '.join(gaps[-1]['top']))}" if len(gaps) > 1 and gaps[-1] is not gaps[0] else "")
               + ". This is the practical value of a profile-based typology: awareness campaigns, helplines and "
                 "police training can be targeted at the dominant crime types of each group. " if gaps else "")
            + f"The first two principal components capture {(R['pca']['pc1'] + R['pca']['pc2']) * 100:.0f}% of the variance, "
              + ("so the two-dimensional picture in Fig. 7 is a fair summary of the structure." if R['pca']['pc1'] + R['pca']['pc2'] >= .6
                 else "so the two-dimensional picture in Fig. 7 shows only part of the structure and should be read with Table 4.")),
          P("<b>Limitations.</b> (i) The data are <i>registered</i> crimes. A higher rate may mean better "
            "registration, greater awareness, or a more responsive police system, not necessarily more violence "
            "[3], [4]; a cluster with low rates may therefore hide under-reporting. (ii) State-level "
            "aggregates conceal large district differences. (iii) Small Union Territories have small populations, so their "
            "rates are volatile. (iv) Rates use Census-2011 female population rather than projected mid-year "
            "population, which slightly overstates rates in fast-growing States. (v) Clustering depends on the chosen "
            "crime heads and on scaling; the robustness checks in Table 5 quantify part of this dependence. (vi) With "
            f"only {n} objects, cluster boundaries are not sharp."),
          P("<b>Ethics and SDG impact.</b> The data are aggregate public statistics without personal information. The "
            "main risk is misreading the clusters as a league table of 'unsafe' States. Results should be "
            "reported together with the reporting caveat above and used to direct support services and prevention "
            "programmes (SDG targets 5.2 and 16.1), not to stigmatise regions."),
          P("8 Conclusion and Future Work", h1),
          P(f"Using NCRB State-wise data normalised per lakh women, k-means and Ward clustering both identified "
            f"{k} State typologies (mean silhouette {cmp_['km_sil']:.2f}, inter-method ARI {cmp_['ari']:.2f}, bootstrap "
            f"ARI {st['mean']:.2f}). The profile differences between clusters (Table 4, Fig. 8) show how the mix, and not only the level, of "
            f"registered crimes against women varies across States. Future work will use district-level data, add socio-economic covariates "
            "(literacy, sex ratio, urbanisation, police strength), analyse change over years and test "
            "density-based and Gaussian-mixture clustering."),
          P("Declarations", h1),
          P(f"<b>Data availability.</b> NCRB, <i>Crime against Women</i>, Open Government Data Platform India: "
            f"{E(DATA_URL)} (accessed {R['run_date']}). Female population: Census of India 2011. Code and figures: "
            f"run_experiments.py (a student places the code on GitHub)."),
          P("<b>Conflict of interest.</b> The author declares no conflict of interest. <b>Funding.</b> None."),
          P("<b>Use of AI tools.</b> The analysis code and the draft text of this report were prepared with the help of an AI "
            "assistant (Claude, Anthropic). The author ran the code on the downloaded dataset, checked the outputs "
            "and takes full responsibility for the content. [Edit this statement so that it states exactly which AI tools "
            "you used and for what, as your course requires.]"),
          P("References", h1)]
    refs = [
        "National Crime Records Bureau, <i>Crime in India</i> (annual reports), Ministry of Home Affairs, New Delhi. https://ncrb.gov.in",
        f"NCRB via Open Government Data (OGD) Platform India, \"Crime against Women,\" data catalogue. {DATA_URL}",
        "Justice J. S. Verma Committee, <i>Report of the Committee on Amendments to Criminal Law</i>, Government of India, New Delhi, Jan. 2013.",
        "International Institute for Population Sciences (IIPS) and ICF, <i>National Family Health Survey (NFHS-5), 2019-21: India</i>, Mumbai: IIPS, 2021.",
        "J. Agarwal, R. Nagpal and R. Sehgal, \"Crime analysis using K-means clustering,\" <i>International Journal of Computer Applications</i>, vol. 83, no. 4, 2013.",
        "P. Goel and V. Yadav, \"Crime against Indian women - women crime susceptibility indexes (WCSI): a principal component analysis,\" <i>OIDA International Journal of Sustainable Development</i>, vol. 6, no. 7, pp. 93-102, 2013.",
        "C. Dutta and A. Pramanick, \"Crime against women in India (2001-2023): a spatial analysis using K-means clustering approach,\" MPRA Paper No. 128208, University Library of Munich, 2026.",
        "J. MacQueen, \"Some methods for classification and analysis of multivariate observations,\" in <i>Proc. 5th Berkeley Symp. on Mathematical Statistics and Probability</i>, vol. 1, 1967, pp. 281-297.",
        "J. H. Ward, \"Hierarchical grouping to optimize an objective function,\" <i>Journal of the American Statistical Association</i>, vol. 58, no. 301, pp. 236-244, 1963.",
        "P. J. Rousseeuw, \"Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,\" <i>Journal of Computational and Applied Mathematics</i>, vol. 20, pp. 53-65, 1987.",
        "I. T. Jolliffe and J. Cadima, \"Principal component analysis: a review and recent developments,\" <i>Philosophical Transactions of the Royal Society A</i>, vol. 374, no. 2065, 20150202, 2016.",
        "L. Hubert and P. Arabie, \"Comparing partitions,\" <i>Journal of Classification</i>, vol. 2, pp. 193-218, 1985.",
        "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.",
        "B. Iglewicz and D. C. Hoaglin, <i>How to Detect and Handle Outliers</i>, ASQC Quality Press, Milwaukee, 1993.",
        "T. Calinski and J. Harabasz, \"A dendrite method for cluster analysis,\" <i>Communications in Statistics</i>, vol. 3, no. 1, pp. 1-27, 1974.",
        "D. L. Davies and D. W. Bouldin, \"A cluster separation measure,\" <i>IEEE Trans. Pattern Analysis and Machine Intelligence</i>, vol. PAMI-1, no. 2, pp. 224-227, 1979.",
        "R. R. Sokal and F. J. Rohlf, \"The comparison of dendrograms by objective methods,\" <i>Taxon</i>, vol. 11, no. 2, pp. 33-40, 1962.",
        "T. Hastie, R. Tibshirani and J. Friedman, <i>The Elements of Statistical Learning</i>, 2nd ed., Springer, New York, 2009.",
    ]
    for i, r in enumerate(refs, 1):
        S.append(P(f"[{i}] {r}", refS))
    doc.build(S)


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="NCRB 'Crime against Women' file (.csv / .xlsx)")
    ap.add_argument("--demo", action="store_true", help="use SIMULATED data to test the pipeline")
    ap.add_argument("--years", help="e.g. 2018-2022 (only if the file has a Year column)")
    ap.add_argument("--pop", help="CSV with columns state,female_population (overrides built-in Census 2011 table)")
    ap.add_argument("--k", type=int, help="force the number of clusters")
    ap.add_argument("--exclude", help="comma-separated States/UTs to exclude (e.g. tiny UTs)")
    ap.add_argument("--min-nonzero", type=float, default=0.6, help="keep heads non-zero in >= this share of States")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.02, help="silhouette tolerance for choosing the smallest good k")
    ap.add_argument("--min-size", type=int, default=3, help="smallest allowed cluster when choosing k")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--college", default="[College Name]")
    ap.add_argument("--email", default="[name@college.du.ac.in]")
    ap.add_argument("--supervisor", default="[Name]")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        args.data = os.path.join(args.out, "DEMO_simulated_crime_against_women.csv")
        make_demo(args.data)
        print("!! DEMO MODE: using SIMULATED data (not real).")
    if not args.data or not os.path.exists(args.data):
        sys.exit("Please give --data <file> (download it from " + DATA_URL + ") or use --demo.")
    log = {}
    raw = read_any(args.data)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
    rate, meta = build_state_table(raw, args, log)
    print(f"State table: {rate.shape[0]} States/UTs x {rate.shape[1]} crime heads ({meta.get('input')}; years: {meta.get('years')})")
    print("Crime heads used :", ", ".join(rate.columns))
    if meta.get("ignored_columns"):
        print("Columns ignored  :", ", ".join(meta["ignored_columns"]))
    if log.get("dropped_states_no_pop"):
        print("No population for:", ", ".join(log["dropped_states_no_pop"]), "(dropped; use --pop to supply)")
    if rate.shape[0] < 8:
        sys.exit("Too few States/UTs recognised (<8). Check the file layout / column names.")
    R, A = run_analysis(rate, args, meta, log)
    figdir = os.path.join(args.out, "figures")
    F = make_figures(R, A, figdir)
    # tables
    out = A["rate"].copy()
    out["cluster_kmeans"] = ["C%d" % (c + 1) for c in A["lab"]]
    out["cluster_ward"] = ["W%d" % (c + 1) for c in A["wlab"]]
    out["silhouette"] = A["sil"]
    out.round(3).to_csv(os.path.join(args.out, "state_clusters.csv"))
    pd.DataFrame({"C%d" % c["id"]: c["profile"] for c in R["clusters"]}).round(3).to_csv(
        os.path.join(args.out, "cluster_profiles.csv"))
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pdf = os.path.join(args.out, "ML_Project_Roopesh_Kumar_Problem21.pdf" if not args.demo else "DEMO_paper.pdf")
    build_pdf(R, A, F, pdf, args)
    print(f"\nk = {R['k']} | silhouette = {R['compare']['km_sil']:.3f} | ARI(k-means, Ward) = {R['compare']['ari']:.2f} "
          f"| stability ARI = {R['stability']['mean']:.2f}")
    for c in R["clusters"]:
        print(f"  C{c['id']} (n={c['n']}, total rate {c['mean_total']:.1f}): {', '.join(c['states'])}")
    print(f"\nDone. Paper: {pdf}\nTables/figures in: {args.out}/")


if __name__ == "__main__":
    main()
