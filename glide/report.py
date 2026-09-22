"""Reproducible experiment report export for GLIDE."""

from __future__ import annotations

import csv
import html
import json
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict


def build_report(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the complete experiment configuration and provenance metadata."""
    return {
        'schema_version': 2,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'results': results,
    }


def summarize_trials(trials: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean, sample standard deviation, and 95% CI per numeric metric."""
    summary: Dict[str, Dict[str, float]] = {}
    if not trials:
        return summary
    for key in sorted({key for trial in trials for key, value in trial.items()
                       if isinstance(value, (int, float))}):
        values = [float(trial[key]) for trial in trials if isinstance(trial.get(key), (int, float))]
        mean = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        margin = 1.96 * stddev / (len(values) ** 0.5)
        summary[key] = {'mean': mean, 'stddev': stddev, 'ci95_low': mean - margin, 'ci95_high': mean + margin}
    return summary


def export_report(report: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Write JSON, CSV, and a self-contained HTML summary."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        'json': os.path.join(output_dir, 'report.json'),
        'csv': os.path.join(output_dir, 'report.csv'),
        'html': os.path.join(output_dir, 'report.html'),
    }
    with open(paths['json'], 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    rows = []
    for scenario, value in report.get('results', {}).items():
        schedulers = value.get('schedulers', value) if isinstance(value, dict) else {}
        for scheduler, metrics in schedulers.items():
            if isinstance(metrics, dict):
                rows.append({'scenario': scenario, 'scheduler': scheduler, **{
                    key: value for key, value in metrics.items()
                    if isinstance(value, (int, float, str))
                }})
    fields = sorted({key for row in rows for key in row}) or ['scenario', 'scheduler']
    with open(paths['csv'], 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(row.get(field, "")))}</td>' for field in fields) + '</tr>'
        for row in rows
    )
    headings = ''.join(f'<th>{html.escape(field)}</th>' for field in fields)
    document = (
        '<!doctype html><meta charset="utf-8"><title>GLIDE Experiment Report</title>'
        '<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}'
        'td,th{border:1px solid #ddd;padding:.4rem;text-align:left}</style>'
        '<h1>GLIDE Experiment Report</h1><p>Generated: '
        f'{html.escape(report.get("generated_at", ""))}</p><table><thead><tr>{headings}'
        f'</tr></thead><tbody>{table}</tbody></table>'
    )
    with open(paths['html'], 'w', encoding='utf-8') as handle:
        handle.write(document)
    return paths
