#!/usr/bin/env python3
"""Read-only training monitor and HTML summary; never trains or finalizes scores."""
import argparse
from datetime import datetime
from html import escape
import json
import math
import os
from pathlib import Path
import time


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def render(directory):
    launch = read(directory / "launch.json", {})
    active = alive(launch.get("pid"))
    comparison = read(directory / "comparison.json", None)
    plan = read(directory / "run_plan.json", {})
    variants = plan.get("variants", ["baseline", "enriched"])
    config = plan.get("config", {})
    samples, epochs = config.get("samples", 5000), config.get("epochs", 4)
    effective_batch = config.get("micro_batch", 2) * config.get("accumulation", 32)
    updates = epochs * math.ceil(samples / effective_batch)
    complete = comparison is not None and len(comparison.get("runs", [])) == len(variants)
    status = "Completed" if complete else "Running" if active else "Incomplete / stopped"
    lines = ['<!doctype html><meta charset="utf-8">',
        '<title>Head training</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}.note{color:#555}pre{white-space:pre-wrap}</style>']
    if active:
        lines.append('<meta http-equiv="refresh" content="30">')
    lines += [f'<h1>Head training: {escape(status)}</h1>',
        '<p>PyTorch/' + escape(config.get('device', 'mps').upper()) + '; original encoder frozen. Selected variants: ' + escape(', '.join(variants)) + f'. {samples:,} training records, {epochs} epochs and {updates} updates each. No calibration or confirmation scoring.</p>',
        '<table><tr><th>Variant</th><th>Updates</th><th>Epoch</th><th>Last loss</th><th>GPU GB after cleanup</th><th>Best completed development Brier ALC</th></tr>']
    summaries = {}
    for variant in variants:
        folder = directory / variant
        history = read(folder / 'history.json', [])
        epochs = read(folder / 'epochs.json', [])
        report = read(folder / 'report.json', {})
        last = history[-1] if history else {}
        scores = [epoch['brier_alc'] for epoch in epochs]
        score = f"{min(scores):.6f}" if scores else 'Pending'
        memory = last.get('mps_driver_bytes', last.get('cuda_reserved_bytes'))
        cells = [variant, f'{len(history)} / {updates}', str(last.get('epoch', 'Pending')),
                 f"{last['loss']:.4f}" if 'loss' in last else 'Pending',
                 f"{memory/1e9:.2f}" if memory is not None else 'Pending', score]
        lines.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in cells) + '</tr>')
        summaries[variant] = report
    lines += ['</table>', '<p class="note">Loss is measured on training batches and is not Brier ALC. Development scores appear only after a complete pass. GPU allocation is sampled after cache cleanup; it is not a peak measurement. Update history is not a resumable checkpoint.</p>',
              '<p>Always returning 50% gives Brier ALC 0.250000. Only two development benchmark families are covered. Model-property enrichment reaches one development system; other systems receive harness capabilities.</p>']
    if complete and comparison.get('enriched_minus_baseline') is not None:
        delta = comparison['enriched_minus_baseline']
        lines.append(f'<h2>Completed comparison</h2><p>Enriched minus baseline Brier ALC: {delta:+.6f}. Negative favors enrichment.</p>')
    for variant, report in summaries.items():
        if report:
            lines.append('<h2>' + escape(variant) + ' completed report</h2><pre>' + escape(json.dumps(report, indent=2)) + '</pre>')
    lines.append('<p class="note">Last refreshed: ' + escape(datetime.now().astimezone().isoformat(timespec='seconds')) + '</p>')
    temporary = directory / 'report.html.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(directory / 'report.html')
    return active


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2'))
    parser.add_argument('--watch', action='store_true', help='refresh the report while its recorded training PID is alive')
    args = parser.parse_args()
    while True:
        active = render(args.directory)
        if not args.watch or not active:
            break
        time.sleep(30)
