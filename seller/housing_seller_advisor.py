#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
housing_seller_advisor.py  (INF-safe + pass-only metrics + rewrite preview + pretty)
-----------------------------------------------------------------------------------
Agrega CSVs de ejecuciones de compradores (ranked_results_with_llm.csv) para
producir métricas orientadas al vendedor por listing y, si detecta señales,
proponer textos de anuncio ("rewrites") más atractivos.

Novedades:
- make_pretty_metrics(): rellena NaN de rates con 0, y *_pass sin pasadores
  con sus pares *_all (fallback). std_rank->0 si solo 1 buyer, overbudget y
  buyers_pass->0. Se guarda un CSV "pretty" además del crudo.
- Clasificador audiencia vs amenities ampliado (dishwasher, hardwood, keyless, etc.).
"""

from __future__ import annotations
import argparse
import glob
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Logging
# ----------------------------
import logging
logger = logging.getLogger("seller")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

def set_logging(verbosity: str):
    level = logging.INFO
    if verbosity == "debug": level = logging.DEBUG
    elif verbosity == "quiet": level = logging.WARNING
    logger.setLevel(level)

# ----------------------------
# IO
# ----------------------------
REQUIRED_FOR_RANK = ["U_total"]

def _infer_buyer_id_from_path(path: str, i: int) -> str:
    base = os.path.splitext(os.path.basename(path))[0] or f"run_{i:04d}"
    dirn = os.path.basename(os.path.dirname(path))
    return f"{dirn}::{base}" if dirn else base

def read_runs(glob_pattern: str, max_files: Optional[int] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern, recursive=True))
    if not paths:
        raise ValueError(f"No files match --runs_glob: {glob_pattern}")
    if max_files:
        paths = paths[:max_files]

    frames: List[pd.DataFrame] = []
    for idx, p in enumerate(paths):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")
            continue

        buyer_id = _infer_buyer_id_from_path(p, idx)
        df["buyer_id"] = buyer_id
        df["_source_path"] = p

        for col in REQUIRED_FOR_RANK:
            if col not in df.columns:
                raise ValueError(f"{p} is missing required column '{col}'")

        if "gate_ok" not in df.columns:
            df["gate_ok"] = True

        # Safe numeric casting
        num_cols = [
            "U_total","U_base","U_text","U_eco",
            "sim_desc","sim_pers","RENT_PRICE",
            "N_i","A_i","S_i","Q_i","L_i",
            "budget","rank","hit_at_k",
            "BEDS","BATHS","SQFT"
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)

    if not frames:
        raise ValueError("All files failed to read.")
    return pd.concat(frames, ignore_index=True)

# ----------------------------
# Core helpers
# ----------------------------
def ensure_rank_and_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby("buyer_id")["U_total"].rank(ascending=False, method="first")
    df["hit_at_k"] = (df["rank"] <= float(k)).astype(int)
    return df

def add_overbudget_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RENT_PRICE" in df.columns and "budget" in df.columns:
        df["__overbudget_$__"] = np.maximum(
            0.0, df["RENT_PRICE"].astype(float) - df["budget"].astype(float)
        )
    else:
        df["__overbudget_$__"] = np.nan
    return df

def compute_similarity_variance(df: pd.DataFrame) -> None:
    id_cols = ["ADDRESS"] if "LISTING_ID" not in df.columns else ["LISTING_ID"]
    unique_pairs = df[["buyer_id"] + id_cols].drop_duplicates().shape[0]

    s_desc = pd.to_numeric(df.get("sim_desc", pd.Series(dtype=float)), errors="coerce")
    s_pers = pd.to_numeric(df.get("sim_pers", pd.Series(dtype=float)), errors="coerce")
    v_desc = s_desc.var(ddof=1)
    v_pers = s_pers.var(ddof=1)
    var_desc = 0.0 if pd.isna(v_desc) else float(v_desc)
    var_pers = 0.0 if pd.isna(v_pers) else float(v_pers)

    logger.info(f"Similarity variance: unique_pairs={unique_pairs}, var_desc={var_desc:.6f}, var_pers={var_pers:.6f}")

# ----------------------------
# Metrics
# ----------------------------
def compute_market_metrics(df: pd.DataFrame, k: int, min_buyers: int) -> pd.DataFrame:
    df = ensure_rank_and_topk(df, k)
    df = add_overbudget_column(df)

    # INF-safe y pass-only
    df = df.copy()
    df["U_total_all_noinf"] = df["U_total"].replace([np.inf, -np.inf], np.nan)

    gate_mask = df["gate_ok"].astype(bool)
    df["U_total_pass"]  = df["U_total_all_noinf"].where(gate_mask, np.nan)
    df["U_base_pass"]   = df["U_base"].where(gate_mask, np.nan)
    df["U_text_pass"]   = df["U_text"].where(gate_mask, np.nan)
    df["U_eco_pass"]    = df["U_eco"].where(gate_mask, np.nan)
    df["sim_desc_pass"] = df["sim_desc"].where(gate_mask, np.nan)
    df["sim_pers_pass"] = df["sim_pers"].where(gate_mask, np.nan)
    df["hit_at_k_pass"] = df["hit_at_k"].where(gate_mask, np.nan)

    key_cols = [c for c in ["LISTING_ID","ADDRESS","CITY","NEIGHBORHOOD"] if c in df.columns]
    if not key_cols:
        raise ValueError("Input CSV must contain at least one property identifier column (e.g., ADDRESS).")

    g_all = df.groupby(key_cols, dropna=False)
    pieces: List[pd.Series] = []

    buyers_seen = g_all["buyer_id"].nunique().rename("buyers_seen")
    buyers_pass = df[gate_mask].groupby(key_cols)["buyer_id"].nunique().rename("buyers_pass")
    pieces += [buyers_seen, buyers_pass]

    gate_pass_rate = (buyers_pass / buyers_seen).rename("gate_pass_rate")
    pieces.append(gate_pass_rate)

    hit_at_k_rate_all = g_all["hit_at_k"].mean().rename("hit_at_k_rate")
    pieces.append(hit_at_k_rate_all)

    hit_at_k_rate_pass = df.groupby(key_cols)["hit_at_k_pass"].mean().rename("hit_at_k_rate_pass")
    pieces.append(hit_at_k_rate_pass)

    # Means (all)
    for src, out in [
        ("U_total_all_noinf","mean_U_total_all"),
        ("U_base","mean_U_base_all"),
        ("U_text","mean_U_text_all"),
        ("U_eco","mean_U_eco_all"),
        ("sim_desc","mean_sim_desc_all"),
        ("sim_pers","mean_sim_pers_all"),
        ("RENT_PRICE","mean_price"),
        ("N_i","mean_recency_N"),
        ("rank","mean_rank"),
        ("A_i","mean_A_i_all"),
        ("S_i","mean_S_i_all"),
        ("L_i","mean_L_i_all"),
    ]:
        if src in df.columns:
            pieces.append(g_all[src].mean().rename(out))

    # Means (pass-only)
    for src, out in [
        ("U_total_pass","mean_U_total_pass"),
        ("U_base_pass","mean_U_base_pass"),
        ("U_text_pass","mean_U_text_pass"),
        ("U_eco_pass","mean_U_eco_pass"),
        ("sim_desc_pass","mean_sim_desc_pass"),
        ("sim_pers_pass","mean_sim_pers_pass"),
    ]:
        if src in df.columns:
            pieces.append(df.groupby(key_cols)[src].mean().rename(out))

    if "rank" in df.columns:
        pieces.append(g_all["rank"].std().rename("std_rank"))

    if "__overbudget_$__" in df.columns:
        pieces.append(g_all["__overbudget_$__"].median().rename("overbudget_$median"))
        pieces.append(g_all["__overbudget_$__"].quantile(0.75).rename("overbudget_$p75"))
        pieces.append(g_all["__overbudget_$__"].apply(lambda x: (pd.to_numeric(x, errors="coerce") > 0).mean())
                      .rename("overbudget_share"))

    metrics = pd.concat(pieces, axis=1).reset_index()

    if "buyers_seen" in metrics.columns:
        metrics = metrics[metrics["buyers_seen"] >= int(min_buyers)]

    sort_cols, asc = [], []
    if "mean_U_total_pass" in metrics.columns:
        sort_cols.append("mean_U_total_pass"); asc.append(False)
    elif "hit_at_k_rate_pass" in metrics.columns:
        sort_cols.append("hit_at_k_rate_pass"); asc.append(False)
    elif "mean_U_total_all" in metrics.columns:
        sort_cols.append("mean_U_total_all"); asc.append(False)
    if sort_cols:
        metrics = metrics.sort_values(sort_cols, ascending=asc)
    return metrics

# ----------------------------
# Pretty (reduce NaNs)
# ----------------------------
def make_pretty_metrics(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()

    # Rates -> 0.0 si NaN
    for col in ["gate_pass_rate", "hit_at_k_rate", "hit_at_k_rate_pass"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0.0)

    # buyers_pass -> 0 si NaN
    if "buyers_pass" in m.columns:
        m["buyers_pass"] = pd.to_numeric(m["buyers_pass"], errors="coerce").fillna(0).astype(float)

    # std_rank -> 0.0 cuando NaN (un solo buyer)
    if "std_rank" in m.columns:
        m["std_rank"] = pd.to_numeric(m["std_rank"], errors="coerce").fillna(0.0)

    # Overbudget -> 0 si NaN
    for col in ["overbudget_$median", "overbudget_$p75", "overbudget_share"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0.0)

    # Fallback: *_pass <- *_all cuando *_pass es NaN
    pairs = [
        ("mean_U_total_pass", "mean_U_total_all"),
        ("mean_U_base_pass",  "mean_U_base_all"),
        ("mean_U_text_pass",  "mean_U_text_all"),
        ("mean_U_eco_pass",   "mean_U_eco_all"),
        ("mean_sim_desc_pass","mean_sim_desc_all"),
        ("mean_sim_pers_pass","mean_sim_pers_all"),
    ]
    for pcol, acol in pairs:
        if pcol in m.columns and acol in m.columns:
            m[pcol] = pd.to_numeric(m[pcol], errors="coerce")
            m[acol] = pd.to_numeric(m[acol], errors="coerce")
            m[pcol] = m[pcol].fillna(m[acol])

    return m

# ----------------------------
# Rewrite heuristics & copy
# ----------------------------
def build_rewrite_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()

    # Thresholds heurísticos
    t_low_desc = 0.62
    t_hi_pers  = 0.58
    t_price_share = 0.50

    buyers_pass = m.get("buyers_pass")
    sim_desc    = m.get("mean_sim_desc_all")
    sim_pers    = m.get("mean_sim_pers_all")
    over_med    = m.get("overbudget_$median")
    over_share  = m.get("overbudget_share")
    hit_k_pass  = m.get("hit_at_k_rate_pass")

    reasons = []
    for i in range(len(m)):
        reason = None
        bp = buyers_pass.iloc[i] if buyers_pass is not None else np.nan
        sd = sim_desc.iloc[i] if sim_desc is not None else np.nan
        sp = sim_pers.iloc[i] if sim_pers is not None else np.nan
        om = over_med.iloc[i] if over_med is not None else np.nan
        os = over_share.iloc[i] if over_share is not None else np.nan
        hk = hit_k_pass.iloc[i] if hit_k_pass is not None else np.nan

        if pd.isna(bp) or (bp == 0):
            reason = "no_passers"
        elif (pd.notna(om) and om > 0) and (pd.notna(os) and os >= t_price_share):
            reason = "price_pressure"
        elif (pd.notna(sd) and sd < t_low_desc) and (pd.notna(sp) and sp >= t_hi_pers):
            reason = "low_desc_high_pers"
        elif pd.notna(sd) and sd < t_low_desc:
            reason = "low_desc"
        elif pd.notna(hk) and hk < 0.5:
            reason = "weak_hit@k"
        else:
            reason = None

        reasons.append(reason)

    m["rewrite_reason"] = reasons
    m["needs_rewrite"]  = m["rewrite_reason"].notna()
    m["rewrite_mode"]   = np.where(m["needs_rewrite"], "heuristic", "")
    return m

def _mode_or_first(s: pd.Series) -> str:
    if s is None or len(s) == 0:
        return ""
    try:
        val = s.mode(dropna=True)
        return str(val.iloc[0]) if len(val) else str(s.iloc[0])
    except Exception:
        return str(s.iloc[0])

def _median_or_nan(s: pd.Series) -> float:
    try:
        v = pd.to_numeric(s, errors="coerce")
        return float(v.median())
    except Exception:
        return float("nan")

def _present_features(group_rows: pd.DataFrame) -> List[str]:
    col_map = {
        "POOL": "pool",
        "GYM": "gym",
        "LAUNDRY": "laundry",
        "GRANITE": "granite",
        "STAINLESS": "stainless",
        "DOORMAN": "doorman",
        "FURNISHED": "furnished",
        "CLUBHOUSE": "clubhouse",
        "ELEVATOR": "elevator",
        "GARAGE": "garage",
        "PARKING": "parking",
        "BALCONY": "balcony",
        "TERRACE": "terrace",
    }
    feats = []
    for c, label in col_map.items():
        if c in group_rows.columns:
            v = group_rows[c]
            if pd.api.types.is_numeric_dtype(v):
                on = (pd.to_numeric(v, errors="coerce") > 0).mean() > 0.5
            else:
                on = v.astype(str).str.lower().isin(["1","true","yes","y"]).mean() > 0.5
            if on:
                feats.append(label)
    return feats

def _top_keypoints(group_rows: pd.DataFrame, col: str, topn: int = 4) -> List[str]:
    if col not in group_rows.columns:
        return []
    freq = {}
    for x in group_rows[col].dropna():
        s = str(x).strip().strip("[]").replace("'", "").replace('"', "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            freq[p] = freq.get(p, 0) + 1
    if not freq:
        return []
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k,_ in items[:topn]]

# --- audiencia vs amenity ----
_AUDIENCE_MAP = {
    "commuter": "easy commuters",
    "professionals": "urban professionals",
    "professional": "urban professionals",
    "student": "students",
    "students": "students",
    "family": "families",
    "families": "families",
    "pet": "pet owners",
    "pets": "pet owners",
    "remote": "remote workers",
    "work-from-home": "remote workers",
    "wfh": "remote workers",
    "quiet": "quiet-seekers",
    "luxury": "luxury-minded renters",
    "budget": "value-seekers",
    "value": "value-seekers",
    "nature": "park & river lovers",
    "outdoors": "park & river lovers",
}

_AMENITY_HINT_WORDS = {
    "concierge", "doorman", "terrace", "balcony", "patio", "rooftop",
    "roof", "grill", "bbq", "lounge", "clubhouse", "pool", "gym",
    "laundry", "granite", "stainless", "elevator", "garage", "parking",
    "bike", "storage", "yard", "garden", "spa", "sauna",
    "dishwasher", "stove", "oven", "microwave", "disposal",
    "washer", "dryer", "in-unit", "hardwood", "flooring",
    "keyless", "entry", "smart lock",
}

def _split_audience_vs_amenities(terms: List[str]) -> Tuple[List[str], List[str]]:
    aud, ame = [], []
    seen_aud, seen_ame = set(), set()
    for raw in terms or []:
        t = (raw or "").strip().lower()
        if not t or t == "(unspecified)":
            continue
        if any(h in t for h in _AMENITY_HINT_WORDS):
            if t not in seen_ame:
                ame.append(raw.strip())
                seen_ame.add(t)
            continue
        mapped = None
        for key, label in _AUDIENCE_MAP.items():
            if key in t:
                mapped = label
                break
        if mapped:
            if mapped not in seen_aud:
                aud.append(mapped)
                seen_aud.add(mapped)
        else:
            if len(t.split()) <= 2 and t.replace(" ", "").isalpha():
                if t not in seen_aud:
                    aud.append(t)
                    seen_aud.add(t)
            else:
                if t not in seen_ame:
                    ame.append(raw.strip())
                    seen_ame.add(t)
    return aud, ame

def _compose_heuristic_copy(group_rows: pd.DataFrame,
                            reason: str,
                            agg: Dict[str, float]) -> str:
    addr = _mode_or_first(group_rows.get("ADDRESS", pd.Series(["Listing"])))
    city = _mode_or_first(group_rows.get("CITY", pd.Series([""])))
    hood = _mode_or_first(group_rows.get("NEIGHBORHOOD", pd.Series([""])))
    beds = _median_or_nan(group_rows.get("BEDS", pd.Series([np.nan])))
    baths = _median_or_nan(group_rows.get("BATHS", pd.Series([np.nan])))
    sqft = _median_or_nan(group_rows.get("SQFT", pd.Series([np.nan])))

    raw_terms = _top_keypoints(group_rows, "tone_keypoints", topn=4)
    aud_terms, amenity_terms_from_kp = _split_audience_vs_amenities(raw_terms)

    feats = _present_features(group_rows)

    Ai = agg.get("A_i", np.nan); Si = agg.get("S_i", np.nan); Li = agg.get("L_i", np.nan)

    audience = ", ".join(aud_terms) if aud_terms else None
    if not audience:
        buckets = []
        if np.isfinite(Ai) and Ai >= 0.65: buckets.append("easy commuters")
        if np.isfinite(Si) and Si >= 0.55: buckets.append("amenity-seekers")
        if np.isfinite(Li) and Li >= 0.55: buckets.append("urban professionals")
        audience = ", ".join(buckets[:2]) if buckets else "everyday city living"

    lead_bits = []
    if np.isfinite(Ai) and Ai >= 0.65: lead_bits.append("quick transit & everyday access")
    if np.isfinite(Si) and Si >= 0.55: lead_bits.append("on-site comforts and services")
    if np.isfinite(Li) and Li >= 0.55: lead_bits.append("practical, urban lifestyle")
    lead_tail = "; ".join(lead_bits) if lead_bits else "comfort, access and convenience"

    amenity_merge = []
    if feats: amenity_merge.extend(feats)
    if amenity_terms_from_kp: amenity_merge.extend(amenity_terms_from_kp)
    seen = set(); amenities = []
    for t in amenity_merge:
        s = (t or "").strip()
        if not s: continue
        key = s.lower()
        if key in seen: continue
        amenities.append(s)
        seen.add(key)

    bullets = []
    # Headline helpers
    br = f"{int(beds)}BR" if beds == beds else "—BR"
    if baths == baths:
        ba_num = baths if float(baths).is_integer() is False else int(baths)
        ba = f"{ba_num}BA"
    else:
        ba = "—BA"
    sq = f"~{int(sqft)} sqft" if sqft == sqft else "— sqft"
    loc = hood or city
    headline = f"{br} / {ba} · {sq} — {loc}".strip(" —")

    bullets.append("Designed for: " + audience)
    if np.isfinite(Ai) and Ai >= 0.6:
        bullets.append("Easy access to transit, groceries and city hubs")
    if np.isfinite(Si) and Si >= 0.5:
        bullets.append("On-site amenities for smooth day-to-day living")
    if np.isfinite(Li) and Li >= 0.5:
        bullets.append("Community vibe that fits an urban, practical routine")
    if amenities:
        bullets.append("Notable features: " + ", ".join(amenities))
    if reason and str(reason).startswith("price"):
        bullets.append("Ask about move-in credits or flexible lease options")

    defaults = [
        "Bright layout with efficient use of space",
        "Calm block with everyday services nearby",
        "Ready for comfortable, low-friction living",
    ]
    bullets = [b for b in bullets if b]
    while len(bullets) < 4 and defaults:
        bullets.append(defaults.pop(0))
    bullets = bullets[:6]

    lead = (f"{addr}{', ' + city if city else ''}. "
            f"A fit for {audience}—prioritizing {lead_tail}.")

    block = [lead, "", "Highlights:"] + [f"• {b}" for b in bullets]
    return headline + "\n" + "\n".join(block)

def _key_columns_present(df: pd.DataFrame) -> List[str]:
    return [c for c in ["LISTING_ID","ADDRESS","CITY","NEIGHBORHOOD"] if c in df.columns]

def _mask_for_listing(df: pd.DataFrame, row: pd.Series, key_cols: List[str]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in key_cols:
        val = row[c]
        m &= (df[c] == val)
    return m

def propose_rewrites(metrics: pd.DataFrame, df_all: pd.DataFrame, preview_n: int = 5) -> List[Tuple[str, str, str]]:
    key_cols = _key_columns_present(df_all)
    out_blocks: List[Tuple[str,str,str]] = []
    if not key_cols:
        return out_blocks

    to_show = metrics[metrics.get("needs_rewrite", False) == True].copy()
    if to_show.empty:
        return out_blocks

    to_show = to_show.head(preview_n)

    print("\n=== Proposed Rewrites (preview) ===\n")
    for _, r in to_show.iterrows():
        m = _mask_for_listing(df_all, r, key_cols)
        group_rows = df_all[m]
        agg = {
            "A_i": float(pd.to_numeric(group_rows.get("A_i", pd.Series([np.nan])), errors="coerce").mean()),
            "S_i": float(pd.to_numeric(group_rows.get("S_i", pd.Series([np.nan])), errors="coerce").mean()),
            "L_i": float(pd.to_numeric(group_rows.get("L_i", pd.Series([np.nan])), errors="coerce").mean()),
        }
        reason = r.get("rewrite_reason", "")
        block = _compose_heuristic_copy(group_rows, reason, agg)
        header_id = r.get("ADDRESS", _mode_or_first(group_rows.get("ADDRESS", pd.Series(["Listing"]))))
        head = f"-- {header_id} [{r.get('rewrite_mode','heuristic')}; reason={reason}] --"
        print(head)
        print(block)
        print("\n")
        first_nl = block.find("\n")
        headline = block[:first_nl] if first_nl >= 0 else block
        body = block[first_nl+1:] if first_nl >= 0 else ""
        out_blocks.append((str(header_id), headline, body))
    return out_blocks

# ----------------------------
# Pretty print
# ----------------------------
def print_table(df: pd.DataFrame, cols: Optional[List[str]] = None, max_rows: int = 20, title: str = "Seller Market Metrics (top rows)"):
    if df.empty:
        print("\n(no listings pass the filters)")
        return
    if cols:
        cols = [c for c in cols if c in df.columns]
        view = df[cols].head(max_rows)
    else:
        view = df.head(max_rows)
    print(f"\n=== {title} ===")
    print(view.to_string(index=False))

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate buyer runs into seller market metrics and propose rewrites.")
    ap.add_argument("--runs_glob", required=True, help="Glob for ranked_results_with_llm.csv files")
    ap.add_argument("--k", type=int, default=10, help="Top-K used on buyer runs (for hit@k)")
    ap.add_argument("--min_buyers", type=int, default=5, help="Min distinct buyers to include a listing")
    ap.add_argument("--max_files", type=int, default=None, help="Optional cap on number of files")
    ap.add_argument("--verbosity", choices=["quiet","info","debug"], default="info")
    args = ap.parse_args()

    set_logging(args.verbosity)

    df_all = read_runs(args.runs_glob, max_files=args.max_files)
    compute_similarity_variance(df_all)

    metrics_raw = compute_market_metrics(df_all, k=args.k, min_buyers=args.min_buyers)
    metrics = build_rewrite_flags(metrics_raw)

    cols_show = [
        "ADDRESS","CITY","NEIGHBORHOOD",
        "buyers_seen","buyers_pass","gate_pass_rate",
        "hit_at_k_rate","hit_at_k_rate_pass",
        "mean_U_total_all","mean_U_total_pass",
        "mean_U_base_all","mean_U_text_all","mean_U_eco_all",
        "mean_U_base_pass","mean_U_text_pass","mean_U_eco_pass",
        "mean_sim_desc_all","mean_sim_pers_all",
        "mean_sim_desc_pass","mean_sim_pers_pass",
        "mean_price","mean_recency_N",
        "mean_rank","std_rank",
        "overbudget_$median","overbudget_$p75","overbudget_share",
        "needs_rewrite","rewrite_mode","rewrite_reason"
    ]
    print_table(metrics, cols=cols_show, max_rows=30, title="Seller Market Metrics (top rows)")

    # Pretty y guardado
    metrics_pretty = make_pretty_metrics(metrics)
    print_table(metrics_pretty, cols=cols_show, max_rows=30, title="Seller Market Metrics (pretty preview)")

    _ = propose_rewrites(metrics_pretty, df_all, preview_n=10)

    out_raw = "seller_market_metrics.csv"
    out_pretty = "seller_market_metrics_pretty.csv"
    metrics.to_csv(out_raw, index=False)
    metrics_pretty.to_csv(out_pretty, index=False)
    print(f"\nSaved seller metrics to: {out_raw}")
    print(f"Saved pretty seller metrics to: {out_pretty}")

if __name__ == "__main__":
    main()
