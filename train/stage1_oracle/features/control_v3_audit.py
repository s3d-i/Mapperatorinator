from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .control_v3 import (
    CONFIDENCE_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
)


FEATURE_CONFIDENCE_MAP = {
    "density_level": "density_confidence",
    "density_burst": "density_confidence",
    "ln_change_rate_gated": "ln_change_confidence",
    "chord_ratio": "chord_confidence",
    "jack_excess": "jack_confidence",
    "jack_streak_exposure": "jack_streak_confidence",
    "hand_imbalance_abs": "hand_confidence",
    "repeat_exact": "repeat_confidence",
    "repeat_shift": "repeat_confidence",
    "repeat_motion": "repeat_confidence",
}

MODEL_FEATURE_CONTRACT = {
    "ln_change_rate_gated": "confidence_gated_value",
    "density_level": "stable_value",
    "density_burst": "stable_value",
    "hold_occupancy": "stable_value",
    "chord_ratio": "confidence_gated_or_high_support",
    "jack_excess": "sparse_tail_with_confidence",
    "jack_streak_exposure": "support_sensitive",
    "hand_balance_signed": "confidence_gated_value",
    "hand_imbalance_abs": "confidence_gated_value",
    "repeat_exact": "support_sensitive",
    "repeat_shift": "support_sensitive",
    "repeat_motion": "support_sensitive",
}

DIAGNOSTIC_FEATURE_CONTRACT = {
    "ln_change_rate_raw": "raw_value_with_side_confidence",
}

FEATURE_CONTRACT = {
    **DIAGNOSTIC_FEATURE_CONTRACT,
    **MODEL_FEATURE_CONTRACT,
}

PEAK_DEBUG_COLUMNS = {
    "density_level": {"raw": "density_raw_med", "n_eff": "density_n_eff_med"},
    "density_burst": {"raw": "density_raw_short", "n_eff": "density_n_eff_short"},
    "ln_change_rate_gated": {"raw": "ln_change_rate_raw", "n_eff": "ln_change_n_eff"},
    "chord_ratio": {"numerator": "chord_num", "denominator": "chord_den", "n_eff": "chord_n_eff", "raw": "chord_ratio_raw"},
    "jack_excess": {"numerator": "jack_observed", "denominator": "jack_expected_null", "n_eff": "jack_n_eff", "raw": "jack_excess_raw"},
    "jack_streak_exposure": {"raw": "jack_streak_raw", "n_eff": "jack_streak_n_eff", "max_streak": "jack_streak_max"},
    "hand_balance_signed": {"left_load": "hand_left_load", "right_load": "hand_right_load", "n_eff": "hand_n_eff", "raw": "hand_balance_raw"},
    "hand_imbalance_abs": {"left_load": "hand_left_load", "right_load": "hand_right_load", "n_eff": "hand_n_eff", "raw": "hand_balance_raw"},
    "repeat_exact": {"n_eff": "repeat_exact_n_eff", "top1_freq": "repeat_exact_top1_freq", "pattern_variety": "repeat_exact_pattern_variety"},
    "repeat_shift": {"n_eff": "repeat_shift_n_eff", "top1_freq": "repeat_shift_top1_freq", "pattern_variety": "repeat_shift_pattern_variety"},
    "repeat_motion": {"n_eff": "repeat_motion_n_eff", "top1_freq": "repeat_motion_top1_freq", "pattern_variety": "repeat_motion_pattern_variety"},
}

FEATURE_SUPPORT_COLUMNS = {
    feature: columns["n_eff"]
    for feature, columns in PEAK_DEBUG_COLUMNS.items()
    if "n_eff" in columns
}

HIGH_VALUE_THRESHOLDS = {
    "density_level": 2.50,
    "density_burst": 0.50,
    "hold_occupancy": 0.50,
    "ln_change_rate_gated": 0.50,
    "chord_ratio": 0.50,
    "jack_excess": 0.50,
    "jack_streak_exposure": 0.50,
    "hand_imbalance_abs": 0.50,
    "repeat_exact": 0.50,
    "repeat_shift": 0.50,
    "repeat_motion": 0.50,
}

LOW_SUPPORT_THRESHOLDS = {
    "ln_change_rate_gated": 3.0,
    "chord_ratio": 3.0,
    "jack_excess": 3.0,
    "jack_streak_exposure": 2.0,
    "repeat_exact": 3.0,
    "repeat_shift": 3.0,
    "repeat_motion": 3.0,
}

SATURATION_THRESHOLDS = {
    "chord_ratio": (0.01, 0.95),
    "hand_imbalance_abs": (0.01, 0.95),
    "repeat_exact": (0.01, 0.95),
    "repeat_shift": (0.01, 0.95),
    "repeat_motion": (0.01, 0.95),
}

SECTION_SUMMARY_STATS = (
    "min",
    "mean",
    "std",
    "p20",
    "p50",
    "p80",
    "p90",
    "p95",
    "max",
    "duration_above_threshold",
    "slope",
    "local_variance",
)


def _numeric_summary_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column == "time_s":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            columns.append(column)
    return columns


def _section_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered = times - float(np.mean(times))
    denom = float(np.sum(centered * centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(centered * (values - float(np.mean(values)))) / denom)


def summarize_numeric_column(
    times: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
    grid_step: float,
) -> dict[str, float]:
    finite = np.isfinite(values)
    values = values[finite]
    times = times[finite]
    if len(values) == 0:
        return {stat: 0.0 for stat in SECTION_SUMMARY_STATS}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p20": float(np.percentile(values, 20)),
        "p50": float(np.percentile(values, 50)),
        "p80": float(np.percentile(values, 80)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "duration_above_threshold": float(np.sum(values >= threshold) * grid_step),
        "slope": _section_slope(times, values),
        "local_variance": float(np.var(values)),
    }


def append_feature_aligned_diagnostics(
    record: dict[str, Any],
    section_for_values: pd.DataFrame,
    *,
    top_quantile: float = 0.95,
) -> None:
    for feature, confidence in FEATURE_CONFIDENCE_MAP.items():
        if feature not in section_for_values.columns:
            continue
        values = pd.to_numeric(section_for_values[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue

        finite_positions = np.flatnonzero(finite)
        finite_values = values[finite]
        peak_pos = int(finite_positions[np.argmax(finite_values)])
        top_threshold = float(np.quantile(finite_values, top_quantile))
        top_mask = finite & (values >= top_threshold)
        record[f"{feature}_peak_time_s"] = float(section_for_values["time_s"].iloc[peak_pos])
        record[f"{feature}_peak_value"] = float(values[peak_pos])

        if confidence in section_for_values.columns:
            confidence_values = pd.to_numeric(section_for_values[confidence], errors="coerce").to_numpy(dtype=float)
            peak_confidence = confidence_values[peak_pos]
            top_confidence = confidence_values[top_mask]
            record[f"{feature}_confidence_at_peak"] = float(peak_confidence) if np.isfinite(peak_confidence) else math.nan
            finite_top_conf = top_confidence[np.isfinite(top_confidence)]
            if len(finite_top_conf):
                record[f"{feature}_confidence_top_value_mean"] = float(np.mean(finite_top_conf))
                record[f"{feature}_confidence_top_value_min"] = float(np.min(finite_top_conf))

        for label, column in PEAK_DEBUG_COLUMNS.get(feature, {}).items():
            if column not in section_for_values.columns:
                continue
            debug_values = pd.to_numeric(section_for_values[column], errors="coerce").to_numpy(dtype=float)
            peak_debug = debug_values[peak_pos]
            top_debug = debug_values[top_mask]
            record[f"{feature}_{label}_at_peak"] = float(peak_debug) if np.isfinite(peak_debug) else math.nan
            finite_top_debug = top_debug[np.isfinite(top_debug)]
            if len(finite_top_debug):
                record[f"{feature}_{label}_top_value_mean"] = float(np.mean(finite_top_debug))
                record[f"{feature}_{label}_top_value_min"] = float(np.min(finite_top_debug))


def section_summaries_for_frame(
    row: pd.Series | None,
    frame: pd.DataFrame,
    *,
    section_s: float = 8.0,
    stride_s: float = 4.0,
    bpm_median: float = math.nan,
    map_duration_s: float | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    duration = float(map_duration_s if map_duration_s is not None else frame["time_s"].max())
    starts = np.arange(0.0, max(0.0, duration - section_s) + 0.5 * stride_s, stride_s)
    numeric_columns = _numeric_summary_columns(frame)
    records: list[dict[str, Any]] = []
    time_values = frame["time_s"].to_numpy(dtype=float)
    grid_step = float(np.median(np.diff(time_values))) if len(time_values) > 1 else 0.10

    for start_s in starts:
        end_s = start_s + section_s
        section = frame.loc[(frame["time_s"] >= start_s) & (frame["time_s"] < end_s)].copy()
        if section.empty:
            continue
        if "valid_control_mask" in section.columns:
            valid = section["valid_control_mask"].to_numpy(dtype=bool)
            valid_fraction = float(np.mean(valid)) if len(valid) else 0.0
            section_for_values = section.loc[valid].copy()
            if section_for_values.empty:
                continue
        else:
            valid_fraction = 1.0
            section_for_values = section

        record: dict[str, Any] = {
            "section_start_s": float(start_s),
            "section_end_s": float(end_s),
            "section_center_s": float(start_s + 0.5 * section_s),
            "section_s": float(section_s),
            "stride_s": float(stride_s),
            "rows": int(len(section)),
            "valid_rows": int(len(section_for_values)),
            "valid_fraction": valid_fraction,
            "bpm_median": float(bpm_median),
            "map_duration_s": float(duration),
        }
        if row is not None:
            for column in ["filtered_index", "beatmap_id", "difficulty", "artist", "title", "version", "creator"]:
                if column in row:
                    value = row[column]
                    if isinstance(value, np.generic):
                        value = value.item()
                    record[column] = value

        times = section_for_values["time_s"].to_numpy(dtype=float)
        for column in numeric_columns:
            if column == "valid_control_mask":
                record["valid_control_mask_mean"] = float(np.mean(section[column].to_numpy(dtype=bool)))
                continue
            values = section_for_values[column].to_numpy(dtype=float)
            threshold = HIGH_VALUE_THRESHOLDS.get(column, 0.50)
            stats = summarize_numeric_column(times, values, threshold=threshold, grid_step=grid_step)
            for stat, value in stats.items():
                record[f"{column}_{stat}"] = value
        append_feature_aligned_diagnostics(record, section_for_values)
        records.append(record)

    return pd.DataFrame(records)


def _numeric_column_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(math.nan, index=df.index, dtype=float)


def _section_identity_columns(df: pd.DataFrame) -> list[str]:
    keep = [
        "filtered_index",
        "beatmap_id",
        "difficulty",
        "artist",
        "title",
        "version",
        "section_start_s",
        "section_end_s",
        "valid_fraction",
        "control_confidence_mean",
    ]
    return [column for column in keep if column in df.columns]


def high_value_confidence_audit(
    section_df: pd.DataFrame,
    *,
    value_threshold: float = 0.50,
    confidence_threshold: float = 0.20,
    low_support_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    support_thresholds = LOW_SUPPORT_THRESHOLDS if low_support_thresholds is None else low_support_thresholds
    rows: list[dict[str, Any]] = []
    identity_columns = _section_identity_columns(section_df)

    for feature, confidence in FEATURE_CONFIDENCE_MAP.items():
        value_col = f"{feature}_p95"
        if value_col not in section_df:
            continue
        threshold = HIGH_VALUE_THRESHOLDS.get(feature, value_threshold)
        value = pd.to_numeric(section_df[value_col], errors="coerce")
        high_value = value >= threshold
        if not high_value.any():
            continue

        confidence_at_peak = _numeric_column_or_nan(section_df, f"{feature}_confidence_at_peak")
        confidence_top_min = _numeric_column_or_nan(section_df, f"{feature}_confidence_top_value_min")
        confidence_top_mean = _numeric_column_or_nan(section_df, f"{feature}_confidence_top_value_mean")
        confidence_section_p20 = _numeric_column_or_nan(section_df, f"{confidence}_p20")
        confidence_section_min = _numeric_column_or_nan(section_df, f"{confidence}_min")
        pointwise = high_value & (
            (confidence_at_peak <= confidence_threshold)
            | (confidence_top_min <= confidence_threshold)
        )
        window_elsewhere = (
            high_value
            & ~pointwise
            & ((confidence_section_p20 <= confidence_threshold) | (confidence_section_min <= confidence_threshold))
        )

        support_col = FEATURE_SUPPORT_COLUMNS.get(feature)
        support_at_peak = _numeric_column_or_nan(section_df, f"{feature}_n_eff_at_peak")
        support_top_mean = _numeric_column_or_nan(section_df, f"{feature}_n_eff_top_value_mean")
        support_threshold = support_thresholds.get(feature, math.nan)
        low_support = high_value & pd.Series(False, index=section_df.index)
        if np.isfinite(support_threshold) and support_col is not None:
            low_support = high_value & (
                (support_at_peak < support_threshold)
                | (support_top_mean < support_threshold)
            )

        class_masks = [
            ("pointwise_high_value_low_confidence", pointwise),
            ("section_high_value_low_confidence_elsewhere", window_elsewhere),
            ("low_support_high_value", low_support),
        ]
        for audit_class, mask in class_masks:
            selected = section_df.loc[mask].copy()
            if selected.empty:
                continue
            for index, selected_row in selected.iterrows():
                row = {column: selected_row[column] for column in identity_columns}
                row.update(
                    {
                        "feature": feature,
                        "contract": FEATURE_CONTRACT.get(feature, ""),
                        "audit_class": audit_class,
                        "value": float(value.loc[index]) if pd.notna(value.loc[index]) else math.nan,
                        "value_threshold": float(threshold),
                        "confidence_at_peak": float(confidence_at_peak.loc[index]) if pd.notna(confidence_at_peak.loc[index]) else math.nan,
                        "confidence_top_value_mean": float(confidence_top_mean.loc[index]) if pd.notna(confidence_top_mean.loc[index]) else math.nan,
                        "confidence_top_value_min": float(confidence_top_min.loc[index]) if pd.notna(confidence_top_min.loc[index]) else math.nan,
                        "confidence_section_p20": float(confidence_section_p20.loc[index]) if pd.notna(confidence_section_p20.loc[index]) else math.nan,
                        "confidence_section_min": float(confidence_section_min.loc[index]) if pd.notna(confidence_section_min.loc[index]) else math.nan,
                        "confidence_threshold": float(confidence_threshold),
                        "n_eff_at_peak": float(support_at_peak.loc[index]) if pd.notna(support_at_peak.loc[index]) else math.nan,
                        "n_eff_top_value_mean": float(support_top_mean.loc[index]) if pd.notna(support_top_mean.loc[index]) else math.nan,
                        "n_eff_threshold": float(support_threshold) if np.isfinite(support_threshold) else math.nan,
                    }
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["audit_class", "feature", "value"], ascending=[True, True, False]).reset_index(drop=True)


def _unique_map_rate(section_df: pd.DataFrame, mask: pd.Series) -> float:
    denom_col = "filtered_index" if "filtered_index" in section_df else "beatmap_id" if "beatmap_id" in section_df else ""
    if not denom_col:
        return float(mask.mean()) if len(mask) else math.nan
    total = int(section_df[denom_col].nunique())
    if total <= 0:
        return math.nan
    return float(section_df.loc[mask, denom_col].nunique() / total)


def _audit_section_key_columns(audit_df: pd.DataFrame) -> list[str]:
    keep = [
        "filtered_index",
        "beatmap_id",
        "section_start_s",
        "section_end_s",
    ]
    return [column for column in keep if column in audit_df.columns]


def _audit_unique_section_count(audit_df: pd.DataFrame) -> int:
    if audit_df.empty:
        return 0
    key_columns = _audit_section_key_columns(audit_df)
    if not key_columns:
        return int(len(audit_df))
    return int(audit_df[key_columns].drop_duplicates().shape[0])


def _audit_unique_map_rate(section_df: pd.DataFrame, audit_df: pd.DataFrame) -> float:
    if audit_df.empty:
        return 0.0
    denom_col = "filtered_index" if "filtered_index" in section_df else "beatmap_id" if "beatmap_id" in section_df else ""
    if not denom_col or denom_col not in audit_df:
        return math.nan
    total = int(section_df[denom_col].nunique())
    if total <= 0:
        return math.nan
    return float(audit_df[denom_col].nunique() / total)


def required_feature_aligned_diagnostic_columns(section_df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for feature in FEATURE_CONFIDENCE_MAP:
        required = [
            f"{feature}_confidence_at_peak",
            f"{feature}_confidence_top_value_mean",
            f"{feature}_confidence_top_value_min",
        ]
        if feature in LOW_SUPPORT_THRESHOLDS:
            required.extend(
                [
                    f"{feature}_n_eff_at_peak",
                    f"{feature}_n_eff_top_value_mean",
                ]
            )
        missing.extend(column for column in required if column not in section_df)
    return missing


def feature_audit_summary(
    section_df: pd.DataFrame,
    *,
    value_threshold: float = 0.50,
    confidence_threshold: float = 0.20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_sections = max(1, len(section_df))
    audit = high_value_confidence_audit(
        section_df,
        value_threshold=value_threshold,
        confidence_threshold=confidence_threshold,
    )
    for feature, confidence in FEATURE_CONFIDENCE_MAP.items():
        value_col = f"{feature}_p95"
        if value_col not in section_df:
            continue
        threshold = HIGH_VALUE_THRESHOLDS.get(feature, value_threshold)
        value = pd.to_numeric(section_df[value_col], errors="coerce")
        high_value = value >= threshold
        feature_audit = audit.loc[audit["feature"].eq(feature)] if not audit.empty else pd.DataFrame()
        pointwise_count = int(feature_audit["audit_class"].eq("pointwise_high_value_low_confidence").sum()) if not feature_audit.empty else 0
        window_count = int(feature_audit["audit_class"].eq("section_high_value_low_confidence_elsewhere").sum()) if not feature_audit.empty else 0
        support_count = int(feature_audit["audit_class"].eq("low_support_high_value").sum()) if not feature_audit.empty else 0
        confidence_at_peak = _numeric_column_or_nan(section_df, f"{feature}_confidence_at_peak")
        section_confidence_p20 = _numeric_column_or_nan(section_df, f"{confidence}_p20")
        pointwise_mask = feature_audit["audit_class"].eq("pointwise_high_value_low_confidence") if not feature_audit.empty else pd.Series(False, index=[])
        window_mask = feature_audit["audit_class"].eq("section_high_value_low_confidence_elsewhere") if not feature_audit.empty else pd.Series(False, index=[])
        support_mask = feature_audit["audit_class"].eq("low_support_high_value") if not feature_audit.empty else pd.Series(False, index=[])
        rows.append(
            {
                "feature": feature,
                "contract": FEATURE_CONTRACT.get(feature, ""),
                "high_value_section_count": int(high_value.sum()),
                "high_value_section_rate": float(high_value.mean()) if len(high_value) else math.nan,
                "pointwise_high_value_low_conf_rate": pointwise_count / total_sections,
                "window_only_low_conf_rate": window_count / total_sections,
                "low_support_high_value_rate": support_count / total_sections,
                "confidence_at_peak_p10": float(confidence_at_peak.quantile(0.10)) if confidence_at_peak.notna().any() else math.nan,
                "confidence_at_peak_median": float(confidence_at_peak.median()) if confidence_at_peak.notna().any() else math.nan,
                "section_confidence_p20_median": float(section_confidence_p20.median()) if section_confidence_p20.notna().any() else math.nan,
                "high_value_unique_map_rate": _unique_map_rate(section_df, high_value),
                "high_value_difficulty_mean": float(pd.to_numeric(section_df.loc[high_value, "difficulty"], errors="coerce").mean()) if "difficulty" in section_df and high_value.any() else math.nan,
                "pointwise_unique_map_rate": _audit_unique_map_rate(section_df, feature_audit.loc[pointwise_mask]) if not feature_audit.empty else 0.0,
                "window_only_unique_map_rate": _audit_unique_map_rate(section_df, feature_audit.loc[window_mask]) if not feature_audit.empty else 0.0,
                "low_support_unique_map_rate": _audit_unique_map_rate(section_df, feature_audit.loc[support_mask]) if not feature_audit.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def feature_contract_report() -> pd.DataFrame:
    rows = []
    for feature, contract in MODEL_FEATURE_CONTRACT.items():
        rows.append(
            {
                "feature": feature,
                "contract": contract,
                "scope": "model",
                "audit_class": "",
            }
        )
    for feature, contract in DIAGNOSTIC_FEATURE_CONTRACT.items():
        rows.append(
            {
                "feature": feature,
                "contract": contract,
                "scope": "diagnostic",
                "audit_class": "",
            }
        )
    return pd.DataFrame(rows)


def saturation_report(
    section_df: pd.DataFrame,
    *,
    features: Sequence[str] = VALUE_FEATURE_NAMES,
    stat: str = "p95",
    thresholds: dict[str, tuple[float, float]] = SATURATION_THRESHOLDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        column = f"{feature}_{stat}"
        if column not in section_df:
            continue
        values = pd.to_numeric(section_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        p95 = float(np.percentile(values, 95))
        p99 = float(np.percentile(values, 99))
        max_value = float(np.max(values))
        row = {
            "feature": feature,
            "stat": stat,
            "count": int(len(values)),
            "p95": p95,
            "p99": p99,
            "max": max_value,
            "near_zero_rate": math.nan,
            "near_high_rate": math.nan,
            "tail_heaviness": math.nan,
            "max_to_p99": math.nan,
        }
        if feature in thresholds:
            low_threshold, high_threshold = thresholds[feature]
            row.update(
                {
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "near_zero_rate": float(np.mean(values <= low_threshold)),
                    "near_high_rate": float(np.mean(values >= high_threshold)),
                }
            )
        else:
            row.update(
                {
                    "low_threshold": math.nan,
                    "high_threshold": math.nan,
                    "tail_heaviness": p99 / max(abs(p95), 1e-9) if np.isfinite(p99) and np.isfinite(p95) else math.nan,
                    "max_to_p99": max_value / max(abs(p99), 1e-9) if np.isfinite(max_value) and np.isfinite(p99) else math.nan,
                }
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["sort_score"] = out["near_high_rate"].fillna(0.0) + out["tail_heaviness"].fillna(0.0)
    return out.sort_values("sort_score", ascending=False).drop(columns=["sort_score"]).reset_index(drop=True)


def _finite_rate(section_df: pd.DataFrame, columns: Sequence[str]) -> float:
    available = [column for column in columns if column in section_df]
    if not available or section_df.empty:
        return math.nan
    values = section_df[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(values).mean())


def _audit_gate_row(
    gate_name: str,
    metric: str,
    value: float,
    threshold: str,
    passed: bool,
    reason: str,
    *,
    severity: str = "hard_gate",
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "gate_name": gate_name,
        "metric": metric,
        "value": float(value) if np.isfinite(value) else math.nan,
        "threshold": threshold,
        "pass": bool(passed),
        "severity": severity,
        "reason": reason,
    }
    row.update(extra)
    return row


def evaluate_control_v3_audit(section_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if section_df.empty:
        return pd.DataFrame(
            [
                _audit_gate_row(
                    "section_count",
                    "rows",
                    0.0,
                    "> 0",
                    False,
                    "dataset audit produced no section rows",
                )
            ]
        )

    model_stat_columns = [f"{feature}_mean" for feature in MODEL_FEATURE_NAMES]
    finite_rate = _finite_rate(section_df, model_stat_columns)
    rows.append(
        _audit_gate_row(
            "finite_model_stats",
            "finite_rate",
            finite_rate,
            ">= 0.999",
            finite_rate >= 0.999,
            "model section statistics should be finite",
        )
    )

    if "valid_fraction" in section_df:
        valid_mean = float(pd.to_numeric(section_df["valid_fraction"], errors="coerce").mean())
        rows.append(
            _audit_gate_row(
                "valid_fraction",
                "mean",
                valid_mean,
                ">= 0.80",
                valid_mean >= 0.80,
                "section windows should mostly cover valid map span",
            )
        )

    missing_diagnostics = required_feature_aligned_diagnostic_columns(section_df)
    missing_preview = ", ".join(missing_diagnostics[:12])
    if len(missing_diagnostics) > 12:
        missing_preview += f", ... (+{len(missing_diagnostics) - 12} more)"
    rows.append(
        _audit_gate_row(
            "required_feature_aligned_diagnostics_present",
            "missing_column_count",
            float(len(missing_diagnostics)),
            "== 0",
            not missing_diagnostics,
            "all high-value audited features must carry peak-aligned confidence/support diagnostics"
            if not missing_diagnostics
            else f"missing peak-aligned diagnostics: {missing_preview}",
        )
    )

    audit = high_value_confidence_audit(section_df)
    total_feature_sections = max(1, len(section_df) * len(FEATURE_CONFIDENCE_MAP))
    for audit_class, threshold in [
        ("pointwise_high_value_low_confidence", 0.02),
        ("low_support_high_value", 0.02),
    ]:
        count = int(audit["audit_class"].eq(audit_class).sum()) if not audit.empty else 0
        rate = count / total_feature_sections
        rows.append(
            _audit_gate_row(
                audit_class,
                "feature_section_rate",
                rate,
                f"<= {threshold:.2f}",
                rate <= threshold,
                "hard failures require high value to align with low confidence or low support at the feature peak",
            )
        )

    per_feature_section_thresholds = {
        "pointwise_high_value_low_confidence": 0.01,
        "low_support_high_value": 0.02,
    }
    per_feature_given_high_thresholds = {
        "pointwise_high_value_low_confidence": 0.10,
        "low_support_high_value": 0.20,
    }
    total_sections = max(1, len(section_df))
    for audit_class in [
        "pointwise_high_value_low_confidence",
        "low_support_high_value",
    ]:
        section_threshold = per_feature_section_thresholds[audit_class]
        given_threshold = per_feature_given_high_thresholds[audit_class]
        for feature in FEATURE_CONFIDENCE_MAP:
            value_col = f"{feature}_p95"
            if value_col not in section_df:
                continue
            threshold = HIGH_VALUE_THRESHOLDS.get(feature, 0.50)
            values = pd.to_numeric(section_df[value_col], errors="coerce")
            high_value = values >= threshold
            high_value_count = int(high_value.sum())
            feature_audit = (
                audit.loc[audit["feature"].eq(feature) & audit["audit_class"].eq(audit_class)].copy()
                if not audit.empty
                else pd.DataFrame()
            )
            failure_count = int(len(feature_audit))
            section_rate = failure_count / total_sections
            unique_section_rate = _audit_unique_section_count(feature_audit) / total_sections
            given_high_value_rate = failure_count / max(1, high_value_count)
            unique_map_rate = _audit_unique_map_rate(section_df, feature_audit)
            reason = (
                "single-feature hard gate prevents global feature-section denominator "
                "from hiding a concentrated feature contract failure"
            )
            rows.append(
                _audit_gate_row(
                    f"{audit_class}_by_feature",
                    "section_rate",
                    section_rate,
                    f"<= {section_threshold:.2f}",
                    section_rate <= section_threshold,
                    reason,
                    feature=feature,
                    high_value_section_count=high_value_count,
                    failure_count=failure_count,
                )
            )
            rows.append(
                _audit_gate_row(
                    f"{audit_class}_by_feature",
                    "given_high_value_rate",
                    given_high_value_rate,
                    f"<= {given_threshold:.2f}",
                    given_high_value_rate <= given_threshold,
                    reason,
                    feature=feature,
                    high_value_section_count=high_value_count,
                    failure_count=failure_count,
                )
            )
            rows.append(
                _audit_gate_row(
                    f"{audit_class}_by_feature",
                    "unique_section_rate",
                    unique_section_rate,
                    f"<= {section_threshold:.2f}",
                    unique_section_rate <= section_threshold,
                    reason,
                    feature=feature,
                    high_value_section_count=high_value_count,
                    failure_count=failure_count,
                )
            )
            rows.append(
                _audit_gate_row(
                    f"{audit_class}_by_feature",
                    "unique_map_rate",
                    unique_map_rate,
                    "informational",
                    True,
                    "map spread for per-feature failures",
                    severity="warning",
                    feature=feature,
                    high_value_section_count=high_value_count,
                    failure_count=failure_count,
                )
            )

    window_only_count = int(audit["audit_class"].eq("section_high_value_low_confidence_elsewhere").sum()) if not audit.empty else 0
    rows.append(
        _audit_gate_row(
            "section_high_value_low_confidence_elsewhere",
            "feature_section_rate",
            window_only_count / total_feature_sections,
            "informational",
            True,
            "high value peak is supported, but the section has lower confidence elsewhere",
            severity="warning",
        )
    )

    coverage_low_count = 0
    for confidence in CONFIDENCE_FEATURE_NAMES:
        p20_col = f"{confidence}_p20"
        min_col = f"{confidence}_min"
        confidence_count = 0
        if p20_col in section_df or min_col in section_df:
            p20 = _numeric_column_or_nan(section_df, p20_col)
            min_value = _numeric_column_or_nan(section_df, min_col)
            confidence_count = int(((p20 <= 0.20) | (min_value <= 0.20)).sum())
            coverage_low_count += confidence_count
        rows.append(
            _audit_gate_row(
                "window_coverage_low_confidence_by_confidence",
                "section_rate",
                confidence_count / max(1, len(section_df)),
                "informational",
                True,
                "section contains local low-confidence regions for this confidence channel",
                severity="warning",
                confidence=confidence,
                feature_section_count=confidence_count,
            )
        )
    rows.append(
        _audit_gate_row(
            "window_coverage_low_confidence",
            "coverage_feature_section_rate",
            coverage_low_count / max(1, len(section_df) * len(CONFIDENCE_FEATURE_NAMES)),
            "informational",
            True,
            "section contains local low-confidence regions independent of feature value",
            severity="warning",
            feature_section_count=coverage_low_count,
        )
    )

    sat = saturation_report(section_df)
    for feature in SATURATION_THRESHOLDS:
        matched = sat.loc[sat["feature"].eq(feature)] if not sat.empty else pd.DataFrame()
        if matched.empty:
            continue
        near_high = float(matched.iloc[0].get("near_high_rate", math.nan))
        near_zero = float(matched.iloc[0].get("near_zero_rate", math.nan))
        rows.append(
            _audit_gate_row(
                f"{feature}_near_high",
                "near_high_rate",
                near_high,
                "<= 0.05",
                near_high <= 0.05,
                "bounded channels should not saturate high across the corpus",
            )
        )
        if feature.startswith("repeat_"):
            rows.append(
                _audit_gate_row(
                    f"{feature}_near_zero",
                    "near_zero_rate",
                    near_zero,
                    "<= 0.98",
                    near_zero <= 0.98,
                    "repeat channels should not be structurally zero",
                )
            )

    return pd.DataFrame(rows)


def stratified_review_queue(
    section_df: pd.DataFrame,
    *,
    n_per_bucket: int = 25,
) -> pd.DataFrame:
    buckets: list[pd.DataFrame] = []
    audit = high_value_confidence_audit(section_df)
    if not audit.empty:
        for audit_class in [
            "pointwise_high_value_low_confidence",
            "low_support_high_value",
            "section_high_value_low_confidence_elsewhere",
        ]:
            selected = audit.loc[audit["audit_class"].eq(audit_class)].copy()
            if selected.empty:
                continue
            selected["review_bucket"] = audit_class
            buckets.append(selected.sort_values("value", ascending=False).head(n_per_bucket))

    if "chord_ratio_p95" in section_df:
        selected = section_df.sort_values("chord_ratio_p95", ascending=False).head(n_per_bucket).copy()
        if not selected.empty:
            selected["feature"] = "chord_ratio"
            selected["audit_class"] = "rare_severe_chord_case"
            selected["review_bucket"] = "rare_severe_chord_case"
            selected["value"] = pd.to_numeric(selected["chord_ratio_p95"], errors="coerce")
            buckets.append(selected)

    if "ln_change_rate_gated_p95" in section_df:
        selected = section_df.sort_values("ln_change_rate_gated_p95", ascending=False).head(n_per_bucket).copy()
        if not selected.empty:
            selected["feature"] = "ln_change_rate_gated"
            selected["audit_class"] = "ln_change_case"
            selected["review_bucket"] = "ln_change_case"
            selected["value"] = pd.to_numeric(selected["ln_change_rate_gated_p95"], errors="coerce")
            buckets.append(selected)

    if not buckets:
        return pd.DataFrame()
    out = pd.concat(buckets, ignore_index=True, sort=False)
    keep = [
        "review_bucket",
        "audit_class",
        "feature",
        "contract",
        "value",
        "confidence_at_peak",
        "confidence_top_value_mean",
        "confidence_section_p20",
        "confidence_section_min",
        "n_eff_at_peak",
        "n_eff_top_value_mean",
        "filtered_index",
        "beatmap_id",
        "difficulty",
        "artist",
        "title",
        "version",
        "section_start_s",
        "section_end_s",
    ]
    keep = [column for column in keep if column in out.columns]
    return out[keep].reset_index(drop=True)
