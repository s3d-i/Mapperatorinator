from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from train.stage_2.events.audit_high_precision_grid_adapter import D_UNIVERSE_2TO6
from train.stage_2.timing.schema import FittedTimingGrid, TimingSegment


DEFAULT_DIAGNOSTICS_JSON = Path(
    "train/artifacts/timing_diagnostics/beatthis_oracle_500_unique_audio_seed20260501_current.json"
)
DEFAULT_OUTPUT_HTML = Path("train/artifacts/reports/events/snap_comparison_webui/index.html")
DEFAULT_OUTPUT_SUMMARY = Path("train/artifacts/reports/events/snap_comparison_webui/summary.json")


@dataclass(frozen=True)
class VisualNote:
    lane: int
    kind: str
    start_ms: float
    end_ms: float | None
    oracle_start_ms: float
    oracle_end_ms: float | None
    fitted_start_ms: float
    fitted_end_ms: float | None
    oracle_start_residual_ms: float
    oracle_end_residual_ms: float | None
    fitted_start_residual_ms: float
    fitted_end_residual_ms: float | None
    start_shift_ms: float
    end_shift_ms: float | None


def build_snap_comparison_webui(
    *,
    diagnostics_json: Path = DEFAULT_DIAGNOSTICS_JSON,
    output_html: Path = DEFAULT_OUTPUT_HTML,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
    top_k: int = 16,
    max_notes_per_map: int = 12000,
) -> dict[str, Any]:
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    rows = list(diagnostics.get("rows", []))
    ranked_rows = sorted(rows, key=_worst_score, reverse=True)[:top_k]

    maps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_rows, start=1):
        beatmap_path = Path(str(row["beatmap_path"]))
        try:
            parsed = _load_notes_with_reamber(beatmap_path)
            notes = parsed[:max_notes_per_map]
            oracle_grid = _grid_from_segments(row["oracle_segments"])
            fitted_grid = _grid_from_segments(row["predicted_segments"])
            visual_notes = [_snap_note(note, oracle_grid, fitted_grid) for note in notes]
            maps.append(_map_payload(rank, row, beatmap_path, visual_notes, truncated=len(parsed) > len(notes)))
        except Exception as exc:  # noqa: BLE001 - this is an artifact builder; keep going for other maps.
            failures.append({"beatmap_path": beatmap_path.as_posix(), "error": str(exc)})

    payload = {
        "schema_version": 1,
        "generated_by": "train.stage_2.events.build_snap_comparison_webui",
        "pinned_commit": _git_rev_parse("HEAD"),
        "diagnostics_json": diagnostics_json.as_posix(),
        "ranking_note": (
            "Worst maps are ranked from BeatThis-vs-oracle timing diagnostics, then rendered as raw .osu, "
            "oracle-grid snap, and fitted-grid snap with a shared vertical time axis."
        ),
        "divisors": list(D_UNIVERSE_2TO6),
        "map_count": len(maps),
        "failures": failures,
        "maps": maps,
    }

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_html(payload), encoding="utf-8")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(_summary_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_notes_with_reamber(beatmap_path: Path) -> list[dict[str, Any]]:
    from reamber.osu import OsuMap

    osu_map = OsuMap.read_file(beatmap_path)
    key_count = int(float(osu_map.circle_size))
    if key_count != 4:
        raise ValueError(f"expected 4K mania map, got circle_size={osu_map.circle_size!r}")

    notes: list[dict[str, Any]] = []
    for hit in osu_map.hits:
        notes.append(
            {
                "lane": int(hit.column),
                "kind": "tap",
                "start_ms": float(hit.offset),
                "end_ms": None,
            }
        )
    for hold in osu_map.holds:
        notes.append(
            {
                "lane": int(hold.column),
                "kind": "hold",
                "start_ms": float(hold.offset),
                "end_ms": float(hold.tail_offset),
            }
        )
    notes.sort(key=lambda note: (float(note["start_ms"]), int(note["lane"]), 0 if note["kind"] == "tap" else 1))
    return notes


def _snap_note(
    note: Mapping[str, Any],
    oracle_grid: FittedTimingGrid,
    fitted_grid: FittedTimingGrid,
) -> VisualNote:
    start = float(note["start_ms"])
    end = None if note["end_ms"] is None else float(note["end_ms"])
    oracle_start = _nearest_tick(start, oracle_grid)
    fitted_start = _nearest_tick(start, fitted_grid)
    oracle_end = _nearest_tick(end, oracle_grid) if end is not None else None
    fitted_end = _nearest_tick(end, fitted_grid) if end is not None else None

    return VisualNote(
        lane=int(note["lane"]),
        kind=str(note["kind"]),
        start_ms=start,
        end_ms=end,
        oracle_start_ms=oracle_start[0],
        oracle_end_ms=None if oracle_end is None else oracle_end[0],
        fitted_start_ms=fitted_start[0],
        fitted_end_ms=None if fitted_end is None else fitted_end[0],
        oracle_start_residual_ms=oracle_start[1],
        oracle_end_residual_ms=None if oracle_end is None else oracle_end[1],
        fitted_start_residual_ms=fitted_start[1],
        fitted_end_residual_ms=None if fitted_end is None else fitted_end[1],
        start_shift_ms=fitted_start[0] - start,
        end_shift_ms=None if end is None or fitted_end is None else fitted_end[0] - end,
    )


def _nearest_tick(time_ms: float | None, grid: FittedTimingGrid) -> tuple[float, float, int, int]:
    if time_ms is None:
        raise ValueError("time_ms must not be None")
    sections = tuple(grid.segments)
    section_id = _section_index(sections, float(time_ms))
    candidate_section_ids = {section_id, section_id - 1, section_id + 1}
    candidates: list[tuple[float, int, int, float, float]] = []
    for candidate_section_id in sorted(candidate_section_ids):
        if candidate_section_id < 0 or candidate_section_id >= len(sections):
            continue
        section = sections[candidate_section_id]
        start = float("-inf") if candidate_section_id == 0 else section.offset_ms
        end = None if candidate_section_id + 1 >= len(sections) else sections[candidate_section_id + 1].offset_ms
        for divisor in D_UNIVERSE_2TO6:
            step_ms = section.beat_length_ms / divisor
            tick_index = int(round((float(time_ms) - section.offset_ms) / step_ms))
            tick_time = section.offset_ms + tick_index * step_ms
            if tick_time < start - 1e-6:
                continue
            if end is not None and tick_time >= end - 1e-6:
                continue
            residual = float(time_ms) - tick_time
            candidates.append((abs(residual), int(divisor), tick_index, tick_time, residual))
    if not candidates:
        raise ValueError(f"no grid tick candidates for time {time_ms}")
    _, divisor, tick_index, tick_time, residual = min(candidates)
    return float(tick_time), float(residual), int(divisor), int(tick_index)


def _section_index(segments: Sequence[TimingSegment], time_ms: float) -> int:
    offsets = [segment.offset_ms for segment in segments]
    index = int(np.searchsorted(offsets, float(time_ms), side="right") - 1)
    return max(0, min(index, len(segments) - 1))


def _grid_from_segments(segments: Sequence[Mapping[str, Any]]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(
                offset_ms=float(segment["offset_ms"]),
                beat_length_ms=float(segment["beat_length_ms"]),
                meter=int(segment.get("meter", 4)),
            )
            for segment in segments
        )
    )


def _map_payload(
    rank: int,
    row: Mapping[str, Any],
    beatmap_path: Path,
    notes: Sequence[VisualNote],
    *,
    truncated: bool,
) -> dict[str, Any]:
    if not notes:
        raise ValueError(f"{beatmap_path} has no notes to render")
    min_time = min(_note_min_time(note) for note in notes)
    max_time = max(_note_max_time(note) for note in notes)
    max_abs_shift = max(max(abs(note.start_shift_ms), abs(note.end_shift_ms or 0.0)) for note in notes)
    max_shift_note = max(notes, key=lambda note: max(abs(note.start_shift_ms), abs(note.end_shift_ms or 0.0)))
    dense_center = _densest_window_center(notes)
    snapped_invalid_holds = sum(
        1
        for note in notes
        if note.kind == "hold"
        and note.fitted_end_ms is not None
        and note.fitted_end_ms <= note.fitted_start_ms + 1e-6
    )
    shifts = [abs(note.start_shift_ms) for note in notes]
    shifts.extend(abs(note.end_shift_ms) for note in notes if note.end_shift_ms is not None)
    residuals = [abs(note.fitted_start_residual_ms) for note in notes]
    residuals.extend(abs(note.fitted_end_residual_ms) for note in notes if note.fitted_end_residual_ms is not None)
    return {
        "rank": rank,
        "beatmap_path": beatmap_path.as_posix(),
        "audio_path": row.get("audio_path"),
        "title": row.get("title"),
        "artist": row.get("artist"),
        "version": row.get("version"),
        "difficulty": row.get("difficulty"),
        "note_count": len(notes),
        "truncated": truncated,
        "raw_time_min_ms": min_time,
        "raw_time_max_ms": max_time,
        "max_shift_center_ms": max_shift_note.start_ms,
        "dense_center_ms": dense_center,
        "metrics": {
            "worst_score": _worst_score(row),
            "fit_score": row.get("fit_score"),
            "mean_phase_error_ms": row.get("mean_phase_error_ms"),
            "max_phase_error_ms": row.get("max_phase_error_ms"),
            "local_bpm_mae": row.get("local_bpm_mae"),
            "first_offset_phase_error_ms": row.get("first_offset_phase_error_ms"),
            "first_bpm_alias_error": row.get("first_bpm_alias_error"),
            "segment_count_delta": row.get("segment_count_delta"),
            "predicted_segment_count": row.get("predicted_segment_count"),
            "oracle_segment_count": row.get("oracle_segment_count"),
            "max_abs_fitted_snap_shift_ms": max_abs_shift,
            "fitted_snap_abs_shift_ms": _stats(shifts),
            "fitted_snap_abs_residual_ms": _stats(residuals),
            "fitted_invalid_hold_count": snapped_invalid_holds,
        },
        "oracle_segments": row.get("oracle_segments", []),
        "fitted_segments": row.get("predicted_segments", []),
        "notes": [note.__dict__ for note in notes],
    }


def _note_min_time(note: VisualNote) -> float:
    values = [note.start_ms, note.oracle_start_ms, note.fitted_start_ms]
    if note.end_ms is not None:
        values.append(note.end_ms)
    if note.oracle_end_ms is not None:
        values.append(note.oracle_end_ms)
    if note.fitted_end_ms is not None:
        values.append(note.fitted_end_ms)
    return min(values)


def _note_max_time(note: VisualNote) -> float:
    values = [note.start_ms, note.oracle_start_ms, note.fitted_start_ms]
    if note.end_ms is not None:
        values.append(note.end_ms)
    if note.oracle_end_ms is not None:
        values.append(note.oracle_end_ms)
    if note.fitted_end_ms is not None:
        values.append(note.fitted_end_ms)
    return max(values)


def _densest_window_center(notes: Sequence[VisualNote], *, window_ms: float = 30000.0) -> float:
    starts = sorted(note.start_ms for note in notes)
    best_count = 0
    best_center = starts[len(starts) // 2]
    right = 0
    for left, start in enumerate(starts):
        while right < len(starts) and starts[right] <= start + window_ms:
            right += 1
        count = right - left
        if count > best_count:
            best_count = count
            best_center = start + window_ms / 2.0
    return float(best_center)


def _worst_score(row: Mapping[str, Any]) -> float:
    mean_phase = _finite(row.get("mean_phase_error_ms"))
    max_phase = _finite(row.get("max_phase_error_ms"))
    bpm_mae = _finite(row.get("local_bpm_mae"))
    first_offset = _finite(row.get("first_offset_phase_error_ms"))
    segment_delta = abs(_finite(row.get("segment_count_delta")))
    alias_bpm_error = _finite(row.get("first_bpm_alias_error"))
    fit_score = _finite(row.get("fit_score"))
    return (
        mean_phase
        + 0.35 * max_phase
        + 0.4 * bpm_mae
        + 0.5 * first_offset
        + 20.0 * segment_delta
        + 40.0 * alias_bpm_error
        + max(0.0, 0.85 - fit_score) * 40.0
    )


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(cleaned),
        "p50": _percentile(cleaned, 50.0),
        "p95": _percentile(cleaned, 95.0),
        "p99": _percentile(cleaned, 99.0),
        "max": cleaned[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "pinned_commit": payload["pinned_commit"],
        "diagnostics_json": payload["diagnostics_json"],
        "map_count": payload["map_count"],
        "failures": payload["failures"],
        "maps": [
            {
                "rank": item["rank"],
                "beatmap_path": item["beatmap_path"],
                "title": item["title"],
                "artist": item["artist"],
                "version": item["version"],
                "metrics": item["metrics"],
            }
            for item in payload["maps"]
        ],
    }


def _render_html(payload: Mapping[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title = "Stage 2 Snap Comparison"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #101214;
  --panel: #171a1f;
  --panel2: #1d2229;
  --line: #353c47;
  --text: #edf1f6;
  --muted: #aab4c0;
  --accent: #5dd0ff;
  --warn: #ffbf5d;
  --bad: #ff6b6b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.app {{
  height: 100vh;
  display: grid;
  grid-template-rows: auto auto 1fr;
  overflow: hidden;
}}
header {{
  padding: 14px 18px 10px;
  border-bottom: 1px solid var(--line);
  background: #11151a;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0;
}}
.controls {{
  display: grid;
  grid-template-columns: minmax(280px, 1fr) 170px 170px 120px;
  gap: 10px;
  align-items: center;
}}
select, button {{
  width: 100%;
  color: var(--text);
  background: var(--panel2);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 13px;
}}
button {{ cursor: pointer; }}
.meta {{
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 8px;
  padding: 8px 18px;
  border-bottom: 1px solid var(--line);
  background: #13171d;
}}
.metric {{
  min-width: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
}}
.metric b {{
  display: block;
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}}
.metric span {{
  display: block;
  margin-top: 3px;
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.stage {{
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 10px 18px 14px;
}}
.panel {{
  min-height: 0;
  display: grid;
  grid-template-rows: auto 1fr;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}}
.panel-title {{
  padding: 8px 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--line);
  font-size: 13px;
  font-weight: 700;
}}
.panel-title small {{
  color: var(--muted);
  font-weight: 500;
}}
.chart-wrap {{
  position: relative;
  min-height: 0;
}}
svg {{
  width: 100%;
  height: 100%;
  display: block;
  background: #0d1014;
}}
.tooltip {{
  position: fixed;
  pointer-events: none;
  z-index: 5;
  max-width: 360px;
  padding: 8px 9px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(14, 17, 22, .96);
  color: var(--text);
  font-size: 12px;
  line-height: 1.35;
  opacity: 0;
  transform: translate(10px, 10px);
}}
.axis-label {{
  fill: #96a2b0;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}}
.grid-line {{ stroke: #2d3540; stroke-width: 1; }}
.lane-line {{ stroke: #262d36; stroke-width: 1; }}
.tap {{ stroke-width: 1.2; }}
.hold-body {{ opacity: .64; }}
.hold-edge {{ stroke-width: 1.2; }}
@media (max-width: 1100px) {{
  .controls {{ grid-template-columns: 1fr 1fr; }}
  .meta {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .stage {{ grid-template-columns: 1fr; overflow: auto; }}
  .app {{ overflow: auto; }}
  .panel {{ height: 70vh; }}
}}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="controls">
      <select id="mapSelect"></select>
      <select id="rangeSelect">
        <option value="full">Full map</option>
        <option value="dense30">Densest 30s</option>
        <option value="shift30">Max shift 30s</option>
        <option value="shift10">Max shift 10s</option>
      </select>
      <select id="colorSelect">
        <option value="lane">Color by lane</option>
        <option value="shift">Color by fitted shift</option>
        <option value="residual">Color by fitted residual</option>
      </select>
      <button id="fitButton">Reset View</button>
    </div>
  </header>
  <section class="meta" id="meta"></section>
  <main class="stage">
    <section class="panel"><div class="panel-title">Raw .osu <small>integer-ms object times</small></div><div class="chart-wrap"><svg id="rawSvg"></svg></div></section>
    <section class="panel"><div class="panel-title">Oracle-grid snap <small>red timing reference</small></div><div class="chart-wrap"><svg id="oracleSvg"></svg></div></section>
    <section class="panel"><div class="panel-title">Fitted-grid snap <small>BeatThis + GridFitter</small></div><div class="chart-wrap"><svg id="fittedSvg"></svg></div></section>
  </main>
</div>
<div class="tooltip" id="tooltip"></div>
<script id="snap-data" type="application/json">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('snap-data').textContent);
const laneColors = ['#64d6ff', '#f4f4f4', '#ffd75d', '#ff8aa5'];
const state = {{ mapIndex: 0 }};
const mapSelect = document.getElementById('mapSelect');
const rangeSelect = document.getElementById('rangeSelect');
const colorSelect = document.getElementById('colorSelect');
const meta = document.getElementById('meta');
const tooltip = document.getElementById('tooltip');

DATA.maps.forEach((m, i) => {{
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = `#${{m.rank}} ${{m.artist || ''}} - ${{m.title || ''}} [${{m.version || ''}}]`;
  mapSelect.appendChild(opt);
}});

mapSelect.addEventListener('change', () => {{ state.mapIndex = Number(mapSelect.value); render(); }});
rangeSelect.addEventListener('change', render);
colorSelect.addEventListener('change', render);
document.getElementById('fitButton').addEventListener('click', () => {{ rangeSelect.value = 'full'; render(); }});
window.addEventListener('resize', render);

function render() {{
  const m = DATA.maps[state.mapIndex];
  renderMeta(m);
  const range = selectedRange(m);
  renderColumn(document.getElementById('rawSvg'), m, range, 'raw');
  renderColumn(document.getElementById('oracleSvg'), m, range, 'oracle');
  renderColumn(document.getElementById('fittedSvg'), m, range, 'fitted');
}}

function renderMeta(m) {{
  const metrics = m.metrics;
  const items = [
    ['rank', `#${{m.rank}}`],
    ['difficulty', fmt(metrics.fit_score) + ' fit score'],
    ['mean phase', fmt(metrics.mean_phase_error_ms) + ' ms'],
    ['max phase', fmt(metrics.max_phase_error_ms) + ' ms'],
    ['BPM MAE', fmt(metrics.local_bpm_mae)],
    ['max snap shift', fmt(metrics.max_abs_fitted_snap_shift_ms) + ' ms'],
    ['notes', `${{m.note_count}}${{m.truncated ? ' truncated' : ''}}`],
  ];
  meta.innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{escapeHtml(k)}}</b><span title="${{escapeHtml(v)}}">${{escapeHtml(v)}}</span></div>`).join('');
}}

function selectedRange(m) {{
  const mode = rangeSelect.value;
  if (mode === 'dense30') return around(m.dense_center_ms, 30000, m);
  if (mode === 'shift30') return around(m.max_shift_center_ms, 30000, m);
  if (mode === 'shift10') return around(m.max_shift_center_ms, 10000, m);
  return [m.raw_time_min_ms - 500, m.raw_time_max_ms + 500];
}}

function around(center, width, m) {{
  const lo = Math.max(m.raw_time_min_ms - 500, center - width / 2);
  const hi = Math.min(m.raw_time_max_ms + 500, center + width / 2);
  return [lo, hi];
}}

function renderColumn(svg, m, range, column) {{
  const rect = svg.getBoundingClientRect();
  const width = Math.max(240, rect.width || 400);
  const height = Math.max(420, rect.height || 700);
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  svg.textContent = '';
  const leftPad = 46;
  const rightPad = 8;
  const topPad = 10;
  const bottomPad = 18;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;
  const laneW = plotW / 4;
  const [start, end] = range;
  const dur = Math.max(1, end - start);
  const yOf = (t) => topPad + plotH - ((t - start) / dur) * plotH;
  drawGrid(svg, start, end, leftPad, topPad, plotW, plotH, laneW, yOf);
  for (const note of m.notes) {{
    const times = noteTimes(note, column);
    if (!rangeIntersects(times.start, times.end, start, end)) continue;
    drawNote(svg, note, times, column, leftPad, laneW, yOf, topPad, plotH);
  }}
}}

function noteTimes(note, column) {{
  if (column === 'raw') return {{ start: note.start_ms, end: note.end_ms }};
  if (column === 'oracle') return {{ start: note.oracle_start_ms, end: note.oracle_end_ms }};
  return {{ start: note.fitted_start_ms, end: note.fitted_end_ms }};
}}

function rangeIntersects(s, e, lo, hi) {{
  const end = e == null ? s : e;
  return end >= lo && s <= hi;
}}

function drawGrid(svg, start, end, x, y, w, h, laneW, yOf) {{
  for (let i = 0; i <= 4; i++) {{
    line(svg, x + i * laneW, y, x + i * laneW, y + h, 'lane-line');
  }}
  const interval = niceInterval((end - start) / 9);
  const first = Math.ceil(start / interval) * interval;
  for (let t = first; t <= end; t += interval) {{
    const yy = yOf(t);
    line(svg, x, yy, x + w, yy, 'grid-line');
    text(svg, x - 8, yy + 4, formatTime(t), 'axis-label', 'end');
  }}
}}

function drawNote(svg, note, times, column, x0, laneW, yOf, topPad, plotH) {{
  const x = x0 + note.lane * laneW + 3;
  const w = Math.max(4, laneW - 6);
  const yStart = clamp(yOf(times.start), topPad, topPad + plotH);
  const color = noteColor(note, column);
  const title = tooltipText(note, column);
  if (note.kind === 'hold' && times.end != null) {{
    const yEnd = clamp(yOf(times.end), topPad, topPad + plotH);
    const y = Math.min(yStart, yEnd);
    const h = Math.max(2, Math.abs(yEnd - yStart));
    rect(svg, x + w * .24, y, w * .52, h, color, 'hold-body', title);
    rect(svg, x, yStart - 2, w, 4, color, 'hold-edge', title);
    rect(svg, x, yEnd - 2, w, 4, color, 'hold-edge', title);
  }} else {{
    rect(svg, x, yStart - 2.5, w, 5, color, 'tap', title);
  }}
}}

function noteColor(note, column) {{
  const mode = colorSelect.value;
  if (mode === 'lane') return laneColors[note.lane % laneColors.length];
  const shift = Math.max(Math.abs(note.start_shift_ms || 0), Math.abs(note.end_shift_ms || 0));
  const residual = Math.max(Math.abs(note.fitted_start_residual_ms || 0), Math.abs(note.fitted_end_residual_ms || 0));
  const value = mode === 'shift' ? shift : residual;
  if (value <= 1) return '#64d6ff';
  if (value <= 5) return '#ffd75d';
  if (value <= 15) return '#ff9f43';
  return '#ff5f6d';
}}

function tooltipText(note, column) {{
  const times = noteTimes(note, column);
  return [
    `${{note.kind.toUpperCase()}} lane=${{note.lane + 1}}`,
    `raw=${{fmt(note.start_ms)}}${{note.end_ms == null ? '' : ' -> ' + fmt(note.end_ms)}} ms`,
    `oracle=${{fmt(note.oracle_start_ms)}}${{note.oracle_end_ms == null ? '' : ' -> ' + fmt(note.oracle_end_ms)}} ms`,
    `fitted=${{fmt(note.fitted_start_ms)}}${{note.fitted_end_ms == null ? '' : ' -> ' + fmt(note.fitted_end_ms)}} ms`,
    `fitted shift=${{fmt(note.start_shift_ms)}}${{note.end_shift_ms == null ? '' : ' / ' + fmt(note.end_shift_ms)}} ms`,
    `showing=${{column}} @ ${{fmt(times.start)}} ms`,
  ].join('\\n');
}}

function rect(svg, x, y, w, h, color, cls, title) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  el.setAttribute('x', x); el.setAttribute('y', y); el.setAttribute('width', w); el.setAttribute('height', h);
  el.setAttribute('rx', 1.5);
  el.setAttribute('fill', color);
  el.setAttribute('stroke', '#050608');
  el.setAttribute('class', cls);
  attachTooltip(el, title);
  svg.appendChild(el);
}}

function line(svg, x1, y1, x2, y2, cls) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  el.setAttribute('x1', x1); el.setAttribute('y1', y1); el.setAttribute('x2', x2); el.setAttribute('y2', y2);
  el.setAttribute('class', cls);
  svg.appendChild(el);
}}

function text(svg, x, y, content, cls, anchor='start') {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  el.setAttribute('x', x); el.setAttribute('y', y); el.setAttribute('class', cls); el.setAttribute('text-anchor', anchor);
  el.textContent = content;
  svg.appendChild(el);
}}

function attachTooltip(el, content) {{
  el.addEventListener('mousemove', (ev) => {{
    tooltip.style.opacity = '1';
    tooltip.style.left = ev.clientX + 'px';
    tooltip.style.top = ev.clientY + 'px';
    tooltip.textContent = content;
  }});
  el.addEventListener('mouseleave', () => {{ tooltip.style.opacity = '0'; }});
}}

function niceInterval(raw) {{
  const candidates = [500,1000,2000,5000,10000,15000,30000,60000];
  return candidates.find(v => v >= raw) || 120000;
}}

function formatTime(ms) {{
  const s = Math.max(0, Math.round(ms / 1000));
  return `${{Math.floor(s / 60)}}:${{String(s % 60).padStart(2, '0')}}`;
}}

function fmt(v) {{
  if (v == null || !Number.isFinite(Number(v))) return 'n/a';
  return Number(v).toFixed(Math.abs(Number(v)) >= 100 ? 1 : 3).replace(/\\.000$/, '');
}}

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

render();
</script>
</body>
</html>
"""


def _git_rev_parse(revision: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", revision], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a raw/oracle/fitted snap comparison Web UI.")
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-notes-per-map", type=int, default=12000)
    args = parser.parse_args(argv)
    payload = build_snap_comparison_webui(
        diagnostics_json=args.diagnostics_json,
        output_html=args.output_html,
        output_summary=args.output_summary,
        top_k=args.top_k,
        max_notes_per_map=args.max_notes_per_map,
    )
    print(f"maps {payload['map_count']}")
    print(f"failures {len(payload['failures'])}")
    print(f"output_html {args.output_html}")
    print(f"output_summary {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
