#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
housing_utility_llm_demo.py
---------------------------------------------------
Personalized housing utility model with LLM induction (Ollama llama3.1),
with live progress, logging, JSON-repair retries, coercion validators,
FULL export of all computed variables to CSV, mood shaping from text,
and CSV with Top-K sorted rows (console also prints Top-K).
"""
from __future__ import annotations
import argparse
import json
import os
import time
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ValidationError, field_validator

# tqdm (fallback si no está instalado)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# ----------------------------
# Helpers
# ----------------------------
def pyd_model_dump(model_obj):
    return model_obj.model_dump() if hasattr(model_obj, "model_dump") else model_obj.dict()

# Logging
logger = logging.getLogger("housing")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)

def set_logging(verbosity: str, logfile: Optional[str] = None):
    level = logging.INFO
    if verbosity == "debug": level = logging.DEBUG
    elif verbosity == "quiet": level = logging.WARNING
    logger.setLevel(level)
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

# ----------------------------
# Pydantic schemas (LLM I/O)
# ----------------------------
class BaseWeights(BaseModel):
    w_A: float; w_S: float; w_Q: float; w_N: float; w_L: float

class MoodVector(BaseModel):
    rush: float = 0.0; risk: float = 0.0; comfort: float = 0.0; urban_vibe: float = 0.0

class PersonalityVector(BaseModel):
    social: float; calm: float; luxury: float; practical: float; urban: float

class InducedProfile(BaseModel):
    hard_requirements: Dict[str, Any]
    budget: float
    personality_vector: PersonalityVector
    mood_vector: MoodVector
    base_weights: BaseWeights
    eta1: float = 0.6; eta2: float = 0.4
    lambda_up: float = 0.003; lambda_down: float = 1.0; c_cap: float = 0.15
    rationale: str

class InducedListing(BaseModel):
    tone_vector: PersonalityVector
    keypoints: List[str]
    risks: List[str] = []
    policy_flags: Dict[str, Any] = {}

    # Coerción y límites
    @field_validator("keypoints", mode="before")
    @classmethod
    def _coerce_keypoints(cls, v):
        if v is None: return ["(unspecified)"]
        if not isinstance(v, list): v = [v]
        v = [str(x).strip() for x in v if str(x).strip()]
        return (v or ["(unspecified)"])[:8]

    @field_validator("risks", mode="before")
    @classmethod
    def _coerce_risks(cls, v):
        if v is None: return []
        if not isinstance(v, list): v = [v]
        v = [str(x).strip() for x in v if str(x).strip()]
        return v[:8]

class InducedSimilarities(BaseModel):
    sim_desc: float
    sim_pers: float
    justification: str

# ----------------------------
# JSON repair utilities (por esquema)
# ----------------------------
PROFILE_SKELETON: Dict[str, Any] = {
    "hard_requirements": {"beds_min": 0, "baths_min": 0, "zones": [], "pets": False, "parking": "any"},
    "budget": 0.0,
    "personality_vector": {"social":0.5,"calm":0.5,"luxury":0.5,"practical":0.5,"urban":0.5},
    "mood_vector": {"rush":0.5,"risk":0.5,"comfort":0.5,"urban_vibe":0.5},
    "base_weights": {"w_A":0.2,"w_S":0.2,"w_Q":0.2,"w_N":0.2,"w_L":0.2},
    "eta1": 0.6, "eta2": 0.4,
    "lambda_up": 0.003, "lambda_down": 1.0,
    "c_cap": 0.15,
    "rationale": "Auto-repaired JSON."
}

LISTING_SKELETON: Dict[str, Any] = {
    "tone_vector": {"social":0.5,"calm":0.5,"luxury":0.5,"practical":0.5,"urban":0.5},
    "keypoints": ["(unspecified)"],
    "risks": [],
    "policy_flags": {}
}

SIMILARITY_SKELETON: Dict[str, Any] = {
    "sim_desc": 0.5,
    "sim_pers": 0.5,
    "justification": "Auto-repaired."
}

def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE|re.DOTALL)

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _balance_braces(s: str) -> str:
    opens = s.count("{"); closes = s.count("}")
    if closes < opens: s += "}" * (opens - closes)
    return s

def _extract_json_slice(text: str) -> str:
    start = text.find("{"); end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end+1].strip()
    return text.strip()

def _clip01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(w.get(k, 0.0)) for k in ["w_A","w_S","w_Q","w_N","w_L"])
    if s <= 0: return {k: 0.2 for k in ["w_A","w_S","w_Q","w_N","w_L"]}
    return {k: max(0.0, float(w.get(k,0.0))/s) for k in ["w_A","w_S","w_Q","w_N","w_L"]}

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

def parse_or_repair_with_skeleton(json_text: str, skeleton: Dict[str, Any]) -> Dict[str, Any]:
    raw = _balance_braces(_remove_trailing_commas(_strip_fences(json_text)))
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return json.loads(json.dumps(skeleton))
        try:
            data = json.loads(_remove_trailing_commas(_balance_braces(m.group(0))))
        except Exception:
            return json.loads(json.dumps(skeleton))

    out = json.loads(json.dumps(skeleton))  # deep copy
    if isinstance(data, dict):
        _deep_merge(out, data)

    # post-procesado específico
    if skeleton is PROFILE_SKELETON:
        for vec in ["personality_vector", "mood_vector"]:
            for k in list(out[vec].keys()):
                out[vec][k] = _clip01(out[vec][k])
        out["base_weights"] = _normalize_weights(out.get("base_weights", {}))
        hr = out["hard_requirements"]
        hr["beds_min"] = int(hr.get("beds_min", 0) or 0)
        hr["baths_min"] = int(hr.get("baths_min", 0) or 0)
        z = hr.get("zones") or []
        if not isinstance(z, list): z = [str(z)]
        hr["zones"] = [str(s) for s in z]
        hr["pets"] = bool(hr.get("pets", False))
        pk = str(hr.get("parking", "any")).lower()
        if pk not in {"any","prefer","must"}: pk = "any"
        hr["parking"] = pk
        for k in ["budget","eta1","eta2","lambda_up","lambda_down","c_cap"]:
            try: out[k] = float(out.get(k, PROFILE_SKELETON[k]))
            except Exception: out[k] = PROFILE_SKELETON[k]
        s_eta = out["eta1"] + out["eta2"]
        if s_eta > 1e-9 and abs(s_eta - 1.0) > 1e-6:
            out["eta1"] /= s_eta; out["eta2"] /= s_eta

    elif skeleton is LISTING_SKELETON:
        tv = out.get("tone_vector", {})
        for k in ["social","calm","luxury","practical","urban"]:
            tv[k] = _clip01(tv.get(k, LISTING_SKELETON["tone_vector"][k]))
        out["tone_vector"] = tv
        kp = out.get("keypoints", LISTING_SKELETON["keypoints"])
        if not isinstance(kp, list): kp = [kp]
        kp = [str(x).strip() for x in kp if str(x).strip()]
        out["keypoints"] = kp or ["(unspecified)"]
        rk = out.get("risks", [])
        if not isinstance(rk, list): rk = [rk]
        out["risks"] = [str(x).strip() for x in rk if str(x).strip()]
        pf = out.get("policy_flags", {})
        if not isinstance(pf, dict): pf = {}
        out["policy_flags"] = pf

    elif skeleton is SIMILARITY_SKELETON:
        try: out["sim_desc"] = float(out.get("sim_desc", 0.5))
        except Exception: out["sim_desc"] = 0.5
        try: out["sim_pers"] = float(out.get("sim_pers", 0.5))
        except Exception: out["sim_pers"] = 0.5
        out["sim_desc"] = _clip01(out["sim_desc"])
        out["sim_pers"] = _clip01(out["sim_pers"])
        out["justification"] = str(out.get("justification", "Auto-repaired.")) or "Auto-repaired."

    return out

# ----------------------------
# Mood shaping from text
# ----------------------------
def shape_mood_from_text(buyer_text: str, mv_in: Dict[str, float]) -> Dict[str, float]:
    bt = (buyer_text or "").lower()
    mv = {**{"rush":0.5,"risk":0.5,"comfort":0.5,"urban_vibe":0.5}, **(mv_in or {})}

    # Señales simples
    if any(k in bt for k in ["quiet","tranquil","silenc","calm","low noise"]):
        mv["comfort"] = max(mv["comfort"], 0.75)
        mv["rush"] = min(mv["rush"], 0.35)
    if any(k in bt for k in ["urgent","asap","move soon","prisa"]):
        mv["rush"] = max(mv["rush"], 0.75)
    if any(k in bt for k in ["risk tolerant","startup vibe","aventura","arriesg"]):
        mv["risk"] = max(mv["risk"], 0.7)
    if any(k in bt for k in ["transport","transit","mbta","subway","t line","commuter rail","bus","metro","céntrico","downtown","urban"]):
        mv["urban_vibe"] = max(mv["urban_vibe"], 0.7)

    # Evitar mood plano (spread mínimo)
    vals = list(mv.values())
    spread = max(vals) - min(vals)
    if spread < 0.15:
        mv["comfort"] = min(1.0, mv["comfort"] + 0.10)
        mv["rush"]    = max(0.0, mv["rush"]    - 0.10)

    # Clip a [0,1]
    for k in mv: mv[k] = max(0.0, min(1.0, float(mv[k])))
    return mv

# ----------------------------
# LLM helpers (tiempos + JSON repair + retries)
# ----------------------------
def call_ollama(prompt: str,
                temperature: float,
                num_predict: int,
                timeout: int) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None
    dt = time.perf_counter() - t0
    if resp.status_code == 200:
        data = resp.json()
        text = data.get("response", "")
        logger.debug(f"LLM ok in {dt:.2f}s, {len(text)} chars")
        return text
    logger.warning(f"Ollama HTTP {resp.status_code} in {dt:.2f}s")
    return None

def llm_json(prompt: str,
             model_cls,
             temperature: float,
             num_predict: int,
             timeout: int,
             retries: int = 1,
             skeleton: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    guard = ("Return ONLY valid minified JSON. "
             "No prose. No markdown. No backticks. No comments.")
    attempt = 0
    last_text = None

    while attempt <= retries:
        text = call_ollama(prompt + "\n\n" + guard,
                           temperature=temperature,
                           num_predict=num_predict,
                           timeout=timeout)
        last_text = text
        if text is None:
            logger.debug("LLM returned None")
            attempt += 1
            prompt = f"Fix to strict minified JSON (no prose, no markdown) this content:\n{last_text or ''}\n"
            temperature = 0.0
            num_predict = min(num_predict, 256)
            continue

        try:
            raw = _extract_json_slice(text)
            raw = _remove_trailing_commas(_strip_fences(raw))
            payload = parse_or_repair_with_skeleton(raw, skeleton or {})
            try:
                _ = model_cls.model_validate(payload)  # pydantic v2
            except Exception:
                _ = model_cls(**payload)               # pydantic v1
            return payload
        except Exception as e:
            logger.debug(f"LLM JSON parse/validate fail (attempt {attempt}): {e}\nRAW(first 400): {(_extract_json_slice(last_text)[:400] if last_text else 'None')}")
            prompt = f"Fix to strict minified JSON (no prose, no markdown) this content:\n{_extract_json_slice(last_text) if last_text else ''}\n"
            temperature = 0.0
            num_predict = min(num_predict, 256)
            attempt += 1

    return None

# ----------------------------
# Heurística para base_weights si el LLM se queda plano
# ----------------------------
def _hash_to_rng(buyer_text: str):
    h = hashlib.md5((buyer_text or "").encode("utf-8")).hexdigest()[:8]
    seed = int(h, 16)
    return np.random.default_rng(seed)

def prior_base_weights_from_text(buyer_text: str) -> Dict[str, float]:
    """
    Construye un prior razonable a partir de buyer_text:
    - 'quiet' -> sube L, baja ligeramente A
    - 'transport'/'transit'/'mbta'/'subway' -> sube A
    - '2br'/'two bedroom' -> sube Q
    """
    bt = (buyer_text or "").lower()
    w = {"w_A":0.22, "w_S":0.20, "w_Q":0.30, "w_N":0.08, "w_L":0.20}

    if any(k in bt for k in ["quiet","tranquil","low noise","calm","silenc"]):
        w["w_L"] += 0.05; w["w_A"] -= 0.02
    if any(k in bt for k in ["transport","transit","mbta","t line","subway","commuter rail","bus"]):
        w["w_A"] += 0.05
    if any(k in bt for k in ["2br","2 br","two bedroom","2 bedroom","2-bed","al menos 2"]):
        w["w_Q"] += 0.05

    for k in w: w[k] = max(0.0, w[k])
    w = _normalize_weights(w)

    rng = _hash_to_rng(bt)
    jitter = rng.normal(0.0, 0.005, 5)
    for (k, j) in zip(["w_A","w_S","w_Q","w_N","w_L"], jitter):
        w[k] = max(0.0, w[k] + float(j))
    w = _normalize_weights(w)
    return w

# ----------------------------
# Prompts LLM (hardened)
# ----------------------------
def _profile_schema_block() -> str:
    return (
        "{\n"
        '  "hard_requirements": {"beds_min": int, "baths_min": int, "zones": [string], "pets": boolean, "parking": "any|prefer|must"},\n'
        '  "budget": number,\n'
        '  "personality_vector": {"social": number, "calm": number, "luxury": number, "practical": number, "urban": number},\n'
        '  "mood_vector": {"rush": number, "risk": number, "comfort": number, "urban_vibe": number},\n'
        '  "base_weights": {"w_A": number, "w_S": number, "w_Q": number, "w_N": number, "w_L": number},\n'
        '  "eta1": number, "eta2": number,\n'
        '  "lambda_up": number, "lambda_down": number, "c_cap": number,\n'
        '  "rationale": string\n'
        "}"
    )

def prompt_induce_profile(buyer_text: str, salary: float) -> str:
    return f"""
Return ONLY strict JSON for a housing utility profile. No extra keys.

User constraints:
- buyer_text: "{buyer_text}"
- monthly_salary: {salary:.2f}

Rules:
- If budget not explicit, set ≈ 0.33 * monthly_salary.
- All vector elements in [0,1].
- base_weights (w_A,w_S,w_Q,w_N,w_L) must sum ≈ 1 AND must NOT be uniform.
- Enforce diversity: max(base_weights) - min(base_weights) >= 0.08.
- parking ∈ "any|prefer|must". zones is an array (may be empty).
- Provide a short rationale.

JSON Schema (informal):
{_profile_schema_block()}
"""

def prompt_induce_listing(description: str) -> str:
    return f"""
Return ONLY strict JSON with this structure (no extra keys):
{{
 "tone_vector": {{"social": number, "calm": number, "luxury": number, "practical": number, "urban": number}},
 "keypoints": [string], "risks": [string], "policy_flags": {{}}
}}
Description:
\"\"\"{description}\"\"\"
"""

def prompt_induce_similarity(buyer_text: str, listing_desc: str, vb: Dict[str,float], ti: Dict[str,float], seed: int) -> str:
    return f"""
Return ONLY strict JSON with fields:
{{"sim_desc": number in [0,1], "sim_pers": number in [0,1], "justification": string}}
Guidance:
- Compute fresh scores from the texts; DO NOT reuse constants across items.
- Use two decimals of precision.
- Deterministic tie-breaker: imagine random seed {seed} (do not include it in the JSON).

Buyer text:
\"\"\"{buyer_text}\"\"\"
Listing:
\"\"\"{listing_desc}\"\"\"
Buyer personality: {json.dumps(vb)}
Listing tone: {json.dumps(ti)}
"""

# ----------------------------
# Heurísticos (solo si LLM no devuelve nada usable)
# ----------------------------
def heuristic_profile(salary: float, buyer_text: str) -> InducedProfile:
    text = (buyer_text or "").lower()
    budget = round(0.33 * float(salary), 2)

    zones = []
    if "quiet" in text: zones.append("quiet street")
    if "transport" in text or "transit" in text or "subway" in text: zones.append("good transport access")
    pets = "pet" in text
    parking = "must" if "parking must" in text else ("prefer" if "parking" in text else "any")

    base_prior = prior_base_weights_from_text(buyer_text)
    base_weights = BaseWeights(**base_prior)
    personality = PersonalityVector(social=0.5, calm=0.9, luxury=0.3, practical=0.8, urban=0.4)
    mood = MoodVector(rush=0.3, risk=0.2, comfort=0.7, urban_vibe=0.4)
    return InducedProfile(
        hard_requirements={"beds_min": 2, "baths_min": 1, "zones": zones, "pets": pets, "parking": parking},
        budget=float(budget),
        personality_vector=personality,
        mood_vector=mood,
        base_weights=base_weights,
        rationale="Heuristic fallback aligned to CLI inputs."
    )

def heuristic_listing(desc: str) -> InducedListing:
    tone = PersonalityVector(social=0.4, calm=0.4, luxury=0.3, practical=0.5, urban=0.5)
    return InducedListing(tone_vector=tone, keypoints=["(unspecified)"], risks=[], policy_flags={})

def heuristic_similarity(buyer_text: str, listing_desc: str, vb: Dict[str,float], ti: Dict[str,float]) -> InducedSimilarities:
    bt = (buyer_text or "").lower()
    ld = (listing_desc or "").lower()
    want = {
        "quiet": any(w in ld for w in ["quiet", "tranquil", "calm", "low noise", "silenc"]),
        "transport": any(w in ld for w in ["transport", "transit", "mbta", "t line", "subway", "commuter rail", "bus"]),
        "2br": any(w in ld for w in ["2br","2 br","two bedroom","2 bedroom","2-bed"])
    }
    score_txt = (want["quiet"] + want["transport"] + want["2br"]) / 3.0
    vbv = np.array([vb[k] for k in ["social","calm","luxury","practical","urban"]], dtype=float)
    tiv = np.array([ti[k] for k in ["social","calm","luxury","practical","urban"]], dtype=float)
    if np.linalg.norm(vbv) == 0 or np.linalg.norm(tiv) == 0:
        sim_p = 0.5
    else:
        sim_p = float(np.dot(vbv, tiv) / (np.linalg.norm(vbv)*np.linalg.norm(tiv)))
    sim_p = 0.5*(sim_p+1.0)
    return InducedSimilarities(sim_desc=float(score_txt), sim_pers=float(sim_p), justification="Heuristic similarity (keyword match + cosine).")

# ----------------------------
# Core math
# ----------------------------
def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = s.min(), s.max()
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return (s - lo) / (hi - lo)
    return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

def build_latent_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    city_freq = df["CITY"].fillna("").map(df["CITY"].value_counts(normalize=True)).fillna(0.0)
    neigh_freq = df["NEIGHBORHOOD"].fillna("").map(df["NEIGHBORHOOD"].value_counts(normalize=True)).fillna(0.0)
    out["A_i"] = 0.6*city_freq + 0.4*neigh_freq

    svc_cols = [c for c in ["POOL","GYM","LAUNDRY","GRANITE","STAINLESS"] if c in df.columns]
    svc = (df[svc_cols].fillna(0).astype(float).clip(0,1)) if svc_cols else pd.DataFrame(0.0, index=df.index, columns=["_"])
    out["S_i"] = svc.mean(axis=1)

    out["Q_i"] = (
        0.3*normalize(df.get("BEDS", 0)) +
        0.3*normalize(df.get("BATHS", 0)) +
        0.3*normalize(df.get("SQFT", 0)) +
        0.1*normalize(df.get("YEAR_BUILT", 0))
    )

    if "DATE_POSTED" in df.columns:
        dt = pd.to_datetime(df["DATE_POSTED"], errors="coerce")
        ts = pd.to_numeric(dt, errors="coerce")
        out["N_i"] = normalize(ts.fillna(ts.mean()))
    else:
        out["N_i"] = 0.5

    flags_cols = [c for c in ["DOORMAN","FURNISHED","CLUBHOUSE"] if c in df.columns]
    flags = (df[flags_cols].fillna(0).astype(float).clip(0,1)) if flags_cols else pd.DataFrame(0.0, index=df.index, columns=["_"])
    btype = df.get("BUILDING_TYPE", pd.Series("", index=df.index)).fillna("")
    is_highrise = btype.str.contains("high", case=False, na=False).astype(float)
    out["L_i"] = 0.75*(flags.mean(axis=1) if not flags.empty else 0.0) + 0.25*is_highrise

    for col in ["A_i","S_i","Q_i","N_i","L_i"]:
        out[col] = out[col].clip(0,1).fillna(0.0)
    return out

def apply_mood_weights(base: BaseWeights, mood: MoodVector) -> Dict[str, float]:
    w = pyd_model_dump(base)
    w["w_L"] += 0.15 * mood.rush - 0.02 * mood.comfort + 0.08 * mood.urban_vibe
    w["w_S"] += -0.10 * mood.rush + 0.12 * mood.comfort + 0.05 * mood.risk
    w["w_Q"] += 0.05 * mood.comfort
    w["w_N"] += 0.02 * mood.risk
    w["w_A"] += 0.08 * mood.urban_vibe - 0.02 * mood.risk
    total = sum(w.values())
    if total > 0:
        for k in w:
            w[k] = max(0.0, w[k] / total)
    return w

def utility_components(row: pd.Series,
                       weights: Dict[str,float],
                       sim_desc: float,
                       sim_pers: float,
                       eta1: float,
                       eta2: float,
                       rent_price: float,
                       budget: float,
                       lambda_up: float,
                       lambda_down: float,
                       c_cap: float) -> Tuple[float,float,float]:
    U_base = (
        weights["w_A"] * row["A_i"] +
        weights["w_S"] * row["S_i"] +
        weights["w_Q"] * row["Q_i"] +
        weights["w_N"] * row["N_i"] +
        weights["w_L"] * row["L_i"]
    )
    U_text = eta1 * sim_desc + eta2 * sim_pers

    if budget > 1e-9:
        gap_rel = (rent_price - budget) / budget
    else:
        gap_rel = 0.0
    bonus_rel = max(0.0, (budget - rent_price) / max(budget, 1.0))
    U_eco = -lambda_up * max(0.0, gap_rel) + lambda_down * min(c_cap, bonus_rel)

    return float(U_base), float(U_text), float(U_eco)

def total_utility(U_base: float, U_text: float, U_eco: float, gate_ok: bool, gamma: float = 0.0) -> float:
    if not gate_ok: return float("-inf")
    return U_base + U_text + U_eco + gamma

# ----------------------------
# Pipeline (LLM-first con logs)
# ----------------------------
def induce_profile(buyer_text: str, salary: float,
                   temperature: float, num_predict: int, timeout: int, retries: int) -> InducedProfile:
    logger.info("Inducing buyer profile with LLM…")
    payload = llm_json(
        prompt_induce_profile(buyer_text, salary), InducedProfile,
        temperature, num_predict, timeout, retries=retries, skeleton=PROFILE_SKELETON
    )
    if payload is None:
        logger.warning("Profile induction fell back to heuristic.")
        return heuristic_profile(salary, buyer_text)

    # Budget repair si viene 0/negativo
    try:
        if float(payload.get("budget", 0.0)) <= 0.0:
            payload["budget"] = round(0.33 * float(salary), 2)
            payload["rationale"] = (payload.get("rationale") or "Auto-repaired JSON.") + " Budget derived from salary."
    except Exception:
        payload["budget"] = round(0.33 * float(salary), 2)

    # Anti-flat guard para base_weights
    try:
        bw = payload.get("base_weights", {})
        vec = np.array([float(bw.get(k, 0.0)) for k in ["w_A","w_S","w_Q","w_N","w_L"]], dtype=float)
        if not np.isfinite(vec).all(): raise ValueError
        spread = float(vec.max() - vec.min())
        if spread < 0.08:
            prior = prior_base_weights_from_text(buyer_text)
            payload["base_weights"] = prior
            payload["rationale"] = (payload.get("rationale") or "") + " Base weights auto-shaped from buyer_text (anti-flat)."
        else:
            payload["base_weights"] = _normalize_weights(bw)
    except Exception:
        payload["base_weights"] = prior_base_weights_from_text(buyer_text)
        payload["rationale"] = (payload.get("rationale") or "") + " Base weights repaired from buyer_text."

    # Mood shaping (anti-plano ligado al texto)
    mv = payload.get("mood_vector", {})
    try:
        vals = [float(mv.get(k, 0.5)) for k in ["rush","risk","comfort","urban_vibe"]]
    except Exception:
        vals = [0.5,0.5,0.5,0.5]
    if (max(vals) - min(vals)) < 0.15:
        payload["mood_vector"] = shape_mood_from_text(buyer_text, mv)
        payload["rationale"] = (payload.get("rationale") or "") + " Mood shaped from text."

    try:
        prof = InducedProfile(**payload)
        logger.debug(f"Profile rationale: {prof.rationale}")
        return prof
    except ValidationError as e:
        logger.warning(f"Profile validation failed ({e}); using heuristic.")
        return heuristic_profile(salary, buyer_text)

def induce_listing_llm(description: str,
                       temperature: float, num_predict: int, timeout: int, retries: int) -> InducedListing:
    payload = llm_json(
        prompt_induce_listing(description), InducedListing,
        temperature, num_predict, timeout, retries=retries, skeleton=LISTING_SKELETON
    )
    if payload is None:
        return heuristic_listing(description)
    try:
        return InducedListing(**payload)
    except ValidationError:
        return heuristic_listing(description)

def induce_similarity_llm(buyer_text: str, listing_desc: str,
                          vb: PersonalityVector, ti: PersonalityVector,
                          temperature: float, num_predict: int, timeout: int, retries: int,
                          seed: int) -> InducedSimilarities:
    payload = llm_json(
        prompt_induce_similarity(buyer_text, listing_desc, pyd_model_dump(vb), pyd_model_dump(ti), seed),
        InducedSimilarities, temperature, num_predict, timeout, retries=retries, skeleton=SIMILARITY_SKELETON
    )
    if payload is None:
        return heuristic_similarity(buyer_text, listing_desc, pyd_model_dump(vb), pyd_model_dump(ti))
    try:
        return InducedSimilarities(**payload)
    except ValidationError:
        return heuristic_similarity(buyer_text, listing_desc, pyd_model_dump(vb), pyd_model_dump(ti))

def gate_requirements(row: pd.Series, hard_req: Dict[str,Any]) -> bool:
    beds_min = int(hard_req.get("beds_min", 0) or 0)
    baths_min = int(hard_req.get("baths_min", 0) or 0)
    ok_beds = (row.get("BEDS", 0) or 0) >= beds_min
    ok_baths = (row.get("BATHS", 0) or 0) >= baths_min
    return bool(ok_beds and ok_baths)

def build_latent_and_profile(work: pd.DataFrame,
                             buyer_text: str,
                             salary: float,
                             temperature: float, num_predict: int, timeout: int, retries: int):
    logger.info("Building latent dimensions…")
    lat = build_latent_dimensions(work)
    work = pd.concat([work.reset_index(drop=True), lat.reset_index(drop=True)], axis=1)
    prof = induce_profile(buyer_text, salary, temperature, num_predict, timeout, retries)
    weights = apply_mood_weights(prof.base_weights, prof.mood_vector)
    return work, prof, weights

def rank_listings(df: pd.DataFrame,
                  buyer_text: str,
                  salary: float,
                  topk: int,
                  city_filter: Optional[str],
                  max_rows: Optional[int],
                  temperature: float,
                  num_predict: int,
                  timeout: int,
                  retries: int) -> Tuple[pd.DataFrame, InducedProfile, pd.DataFrame]:
    t0 = time.perf_counter()
    work = df.copy()

    if city_filter and "CITY" in work.columns:
        work = work[work["CITY"].astype(str).str.contains(city_filter, case=False, na=False)].copy()
        logger.info(f"City filter '{city_filter}': {len(work)} rows")
        if work.empty:
            raise ValueError(f"No listings match city filter: {city_filter}")

    if max_rows and len(work) > max_rows:
        work = work.head(max_rows).copy()
        logger.info(f"Limiting to first {max_rows} rows for faster run")

    work, prof, weights = build_latent_and_profile(work, buyer_text, salary, temperature, num_predict, timeout, retries)
    work = work.reset_index(drop=True)

    # --- REPETIR EN COLUMNAS TODO EL PERFIL PARA EXPORT A CSV ---
    # Hard requirements
    hr = prof.hard_requirements
    zones = hr.get("zones") or []
    if not isinstance(zones, list): zones = [zones]
    zones_str = "; ".join(str(z) for z in zones)
    work["hr_beds_min"] = int(hr.get("beds_min", 0) or 0)
    work["hr_baths_min"] = int(hr.get("baths_min", 0) or 0)
    work["hr_zones"] = zones_str
    work["hr_pets"] = bool(hr.get("pets", False))
    work["hr_parking"] = str(hr.get("parking", "any"))

    # Personality & mood
    vb = pyd_model_dump(prof.personality_vector)
    work["vb_social"] = float(vb["social"]); work["vb_calm"] = float(vb["calm"])
    work["vb_luxury"] = float(vb["luxury"]); work["vb_practical"] = float(vb["practical"]); work["vb_urban"] = float(vb["urban"])

    mv = pyd_model_dump(prof.mood_vector)
    work["mv_rush"] = float(mv["rush"]); work["mv_risk"] = float(mv["risk"])
    work["mv_comfort"] = float(mv["comfort"]); work["mv_urban_vibe"] = float(mv["urban_vibe"])

    # Pesos base y efectivos
    bw = pyd_model_dump(prof.base_weights)
    for k in ["A","S","Q","N","L"]:
        work[f"w_{k}_base"] = float(bw[f"w_{k}"])
        work[f"w_{k}_eff"]  = float(weights[f"w_{k}"])

    # Metaparámetros y budget
    work["eta1"] = float(prof.eta1); work["eta2"] = float(prof.eta2)
    work["lambda_up"] = float(prof.lambda_up); work["lambda_down"] = float(prof.lambda_down)
    work["c_cap"] = float(prof.c_cap); work["budget"] = float(prof.budget)

    logger.info(f"Scoring {len(work)} listings (LLM tone + similarity per row)…")
    sim_desc_list, sim_pers_list, sim_just_list = [], [], []
    gate_list, U_base_list, U_text_list, U_eco_list, U_total_list = [], [], [], [], []
    tone_rows, tone_policy_rows, keypoints_rows, risks_rows = [], [], [], []
    tone_dicts, desc_rows = [], []

    for idx, row in tqdm(work.iterrows(), total=len(work), desc="Scoring", unit="listing"):
        desc = str(row.get("DESCRIPTION", "") or "")
        listing = induce_listing_llm(desc, temperature, num_predict, timeout, retries)
        sims = induce_similarity_llm(
            buyer_text, desc, prof.personality_vector, listing.tone_vector,
            temperature, num_predict, timeout, retries, seed=int(idx)
        )

        gate_ok = gate_requirements(row, prof.hard_requirements)
        U_base, U_text, U_eco = utility_components(
            row=row, weights=weights,
            sim_desc=sims.sim_desc, sim_pers=sims.sim_pers,
            eta1=prof.eta1, eta2=prof.eta2,
            rent_price=float(row.get("RENT_PRICE", 0) or 0.0),
            budget=prof.budget, lambda_up=prof.lambda_up,
            lambda_down=prof.lambda_down, c_cap=prof.c_cap
        )
        U_total = total_utility(U_base, U_text, U_eco, gate_ok)

        # Acumular para CSV
        desc_rows.append(desc)
        sim_desc_list.append(sims.sim_desc); sim_pers_list.append(sims.sim_pers); sim_just_list.append(sims.justification)
        gate_list.append(gate_ok); U_base_list.append(U_base); U_text_list.append(U_text); U_eco_list.append(U_eco); U_total_list.append(U_total)
        tv_dict = pyd_model_dump(listing.tone_vector)
        tone_dicts.append(tv_dict)
        tone_rows.append(json.dumps(tv_dict, ensure_ascii=False))
        tone_policy_rows.append(json.dumps(listing.policy_flags or {}, ensure_ascii=False))
        keypoints_rows.append(", ".join(listing.keypoints))
        risks_rows.append(", ".join(listing.risks))

    # Variance guard de similitudes
    pair_uniques = len(set((round(d, 3), round(p, 3)) for d, p in zip(sim_desc_list, sim_pers_list)))
    var_desc = float(np.var(np.array(sim_desc_list, dtype=float))) if sim_desc_list else 0.0
    var_pers = float(np.var(np.array(sim_pers_list, dtype=float))) if sim_pers_list else 0.0
    low_variance = (pair_uniques <= max(1, len(work)//6)) or (var_desc < 1e-4 and var_pers < 1e-4)
    logger.info(
        f"Similarity variance: unique_pairs={pair_uniques}, var_desc={var_desc:.6f}, var_pers={var_pers:.6f}, "
        f"fallback={'yes' if low_variance else 'no'}"
    )

    if low_variance:
        sim_desc_list_h, sim_pers_list_h, sim_just_list_h = [], [], []
        U_text_list_h, U_total_list_h = [], []
        vb_dict = pyd_model_dump(prof.personality_vector)

        for i in range(len(work)):
            sims_h = heuristic_similarity(buyer_text, desc_rows[i], vb_dict, tone_dicts[i])
            sim_desc_list_h.append(sims_h.sim_desc); sim_pers_list_h.append(sims_h.sim_pers); sim_just_list_h.append("Heuristic (low-variance LLM fallback).")
            U_text_h = prof.eta1 * sims_h.sim_desc + prof.eta2 * sims_h.sim_pers
            U_text_list_h.append(U_text_h)
            U_total_list_h.append(U_base_list[i] + U_text_h + U_eco_list[i])

        sim_desc_list = sim_desc_list_h
        sim_pers_list = sim_pers_list_h
        sim_just_list = sim_just_list_h
        U_text_list = U_text_list_h
        U_total_list = U_total_list_h

    # Ensamblado final de columnas para CSV (TODAS)
    work["LLM_profile_rationale"] = prof.rationale
    work["sim_desc"] = sim_desc_list; work["sim_pers"] = sim_pers_list; work["sim_rationale"] = sim_just_list
    work["gate_ok"] = gate_list
    work["U_base"] = U_base_list; work["U_text"] = U_text_list; work["U_eco"] = U_eco_list; work["U_total"] = U_total_list
    work["tone_vector"] = tone_rows
    work["tone_policy_flags"] = tone_policy_rows
    work["tone_keypoints"] = keypoints_rows; work["tone_risks"] = risks_rows

    # Tone vector en columnas separadas
    tv_cols = ["tone_social","tone_calm","tone_luxury","tone_practical","tone_urban"]
    for i, tv in enumerate(tone_dicts):
        work.loc[i, tv_cols[0]] = float(tv["social"])
        work.loc[i, tv_cols[1]] = float(tv["calm"])
        work.loc[i, tv_cols[2]] = float(tv["luxury"])
        work.loc[i, tv_cols[3]] = float(tv["practical"])
        work.loc[i, tv_cols[4]] = float(tv["urban"])

    # Orden/Ranking final: guardamos TODO ordenado y mostramos Top-K
    sort_cols, ascend = ["U_total"], [False]
    if "RENT_PRICE" in work.columns: sort_cols.append("RENT_PRICE"); ascend.append(True)
    sorted_all = work.sort_values(sort_cols, ascending=ascend).reset_index(drop=True)
    ranked = sorted_all.head(topk).copy() if topk else sorted_all.copy()

    logger.info(f"Done in {time.perf_counter()-t0:.2f}s")
    return ranked, prof, sorted_all

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Personalized housing utility with LLM induction (progress + logging + JSON repair).")
    ap.add_argument("--csv", required=True, help="Path to listings CSV")
    ap.add_argument("--salary", type=float, default=6000.0, help="Buyer's net monthly salary (USD)")
    ap.add_argument("--buyer_text", type=str, default="Looking for at least 2 bedrooms, quiet street, good transport access")
    ap.add_argument("--city", type=str, default=None, help="Optional city filter (substring match)")
    ap.add_argument("--topk", type=int, default=10, help="Top-K listings to return")
    ap.add_argument("--max_rows", type=int, default=None, help="Limit number of rows to process")
    ap.add_argument("--verbosity", choices=["quiet","info","debug"], default="info", help="Logging verbosity")
    ap.add_argument("--logfile", type=str, default=None, help="Optional log file path")
    # LLM params
    ap.add_argument("--llm_temperature", type=float, default=0.2)
    ap.add_argument("--llm_num_predict", type=int, default=200, help="Max tokens in response")
    ap.add_argument("--llm_timeout", type=int, default=30, help="HTTP timeout per call (seconds)")
    ap.add_argument("--llm_retries", type=int, default=2, help="JSON-fix retries if invalid JSON")
    args = ap.parse_args()

    set_logging(args.verbosity, args.logfile)

    df = pd.read_csv(args.csv)
    # Normalizar booleanos/flags típicos
    for col in ["POOL","GYM","GARAGE","LAUNDRY","GRANITE","STAINLESS","DOORMAN","FURNISHED","CLUBHOUSE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["1","true","yes","y","t"]).astype(int)

    ranked, prof, sorted_all = rank_listings(
        df, buyer_text=args.buyer_text, salary=args.salary, topk=args.topk,
        city_filter=args.city, max_rows=args.max_rows,
        temperature=args.llm_temperature, num_predict=args.llm_num_predict, timeout=args.llm_timeout,
        retries=args.llm_retries
    )

    # Impresión amigable (subset de columnas clave) del Top-K
    cols_show = [
        "ADDRESS","CITY","NEIGHBORHOOD","RENT_PRICE","BEDS","BATHS","SQFT",
        "A_i","S_i","Q_i","N_i","L_i","sim_desc","sim_pers","U_base","U_text","U_eco","U_total","gate_ok",
        "w_A_eff","w_S_eff","w_Q_eff","w_N_eff","w_L_eff"
    ]
    cols_show = [c for c in cols_show if c in ranked.columns]
    print("\n=== Ranked Listings (Top-K) ===")
    print(ranked[cols_show].to_string(index=False) if cols_show else ranked.head(10).to_string(index=False))

    # CSV con TODAS las filas ordenadas
    #out_path = "ranked_results_with_llm.csv"
    #sorted_all.to_csv(out_path, index=False)
    #print(f"\nSaved full results (all sorted rows, with LLM rationales) to: {out_path}")

    # CSV con SOLO el Top-K ordenado
    out_path = f"ranked_top{args.topk}_with_llm.csv"
    ranked.to_csv(out_path, index=False)
    print(f"\nSaved Top-{args.topk} results (ranked, with LLM rationales) to: {out_path}")


    print("\n=== Induced Buyer Profile (LLM) ===")
    print(json.dumps(pyd_model_dump(prof), indent=2))

    # Pesos efectivos post-mood
    eff = apply_mood_weights(prof.base_weights, prof.mood_vector)
    print("\n=== Effective weights (after mood) ===")
    print(json.dumps(eff, indent=2))

if __name__ == "__main__":
    main()
