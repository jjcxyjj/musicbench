"""Weighted ranking: combine multiple metrics into a single "best algorithm".

Each metric contributes a normalized 0..1 sub-score. Metrics where lower is
better are inverted. Missing/NaN metrics (e.g. a deep-model metric that was
unavailable) are dropped so they do not skew the ranking.

The final score is a weighted sum; weights are renormalized to 1 over the
metrics that actually produced a number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# metric key (flattened, e.g. "audio_quality.mae_duration") -> (weight, higher_is_better)
DEFAULT_WEIGHTS: Dict[str, Tuple[float, bool]] = {
    # L0 objective features: lower is better (closer to reference)
    "audio_quality.mae_duration": (1.0, False),
    "audio_quality.mae_rms_db": (1.0, False),
    "audio_quality.mae_spectral_centroid": (1.0, False),
    "audio_quality.mae_zero_crossing_rate": (1.0, False),
    "audio_quality.mae_crest_factor": (1.0, False),
    "audio_quality.mae_silence_ratio": (1.0, False),
    # L1 professional metrics
    "fad.fad": (1.5, False),            # lower is better
    "kl_div.kl": (1.0, False),          # lower is better
    "clap_score.clap": (1.5, True),     # higher is better
    # SongEval aesthetics (higher is better)
    "songeval.coherence": (1.0, True),
    "songeval.musicality": (1.0, True),
    "songeval.memorability": (1.0, True),
    "songeval.clarity": (1.0, True),
    "songeval.naturalness": (1.0, True),
    # beat tracking
    "beat_fmeasure.f_measure": (1.0, True),
    "beat_fmeasure.cemgil": (0.5, True),
    "beat_fmeasure.p_score": (0.5, True),
    "tempo_mae.tempo_mae": (1.0, False),
    # tagging
    "tag_map.map": (1.0, True),
    "tag_macro_f1.macro_f1": (1.0, True),
    "tag_macro_auc.macro_auc": (1.0, True),
}


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def rank(
    algorithm_scores: Dict[str, Dict[str, Any]],
    weights: Optional[Dict[str, Tuple[float, bool]]] = None,
) -> List[Dict[str, Any]]:
    """Rank algorithms by a weighted, normalized mix of their metrics.

    Args:
        algorithm_scores: {algorithm_name: {metric_group: {sub: value, ...}}}.
        weights: optional {flat_metric_key: (weight, higher_is_better)}.

    Returns:
        A list of dicts (sorted best-first):
        ``{algorithm, score, normalized_breakdown, used_metrics, skipped_metrics}``
    """
    w = weights or DEFAULT_WEIGHTS

    # Flatten each algorithm's metrics into scalar key -> value.
    flat: Dict[str, Dict[str, float]] = {}
    for alg, scores in algorithm_scores.items():
        fd = _flatten(scores)
        flat[alg] = {k: float(v) for k, v in fd.items() if isinstance(v, (int, float))}

    # Decide which metrics to use: must be present and finite in EVERY algorithm,
    # and have a defined weight.
    keys = sorted({k for alg in flat for k in flat[alg] if k in w})
    used_keys: List[str] = []
    for k in keys:
        vals = [flat[alg].get(k, float("nan")) for alg in flat]
        if all(_finite(v) for v in vals):
            used_keys.append(k)

    # Per-metric min-max normalization -> 0..1, then invert lower-is-better.
    norm: Dict[str, Dict[str, float]] = {alg: {} for alg in flat}
    for k in used_keys:
        vals = [flat[alg][k] for alg in flat]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1e-12
        higher_better = w[k][1]
        for alg in flat:
            v = flat[alg][k]
            if higher_better:
                norm[alg][k] = (v - lo) / rng
            else:
                norm[alg][k] = (hi - v) / rng  # lower value -> higher sub-score

    # Weighted sum with renormalized weights.
    total_weight = sum(w[k][0] for k in used_keys)
    ranking = []
    for alg in flat:
        if not used_keys:
            score = 0.0
        else:
            score = sum(norm[alg][k] * w[k][0] for k in used_keys) / total_weight
        ranking.append(
            {
                "algorithm": alg,
                "score": round(score, 6),
                "breakdown": {k: round(norm[alg][k], 6) for k in used_keys},
                "used_metrics": used_keys,
                "skipped_metrics": [k for k in sorted(flat[alg]) if k not in used_keys],
            }
        )
    ranking.sort(key=lambda r: r["score"], reverse=True)
    return ranking


def _finite(v: float) -> bool:
    import math

    return isinstance(v, (int, float)) and math.isfinite(v)
