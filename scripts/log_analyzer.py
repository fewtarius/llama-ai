#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
"""
Parse llama-server log files and emit structured metrics plus a Markdown
report.

Usage (CLI):
    python3 log_analyzer.py <logfile>           # render markdown to stdout
    python3 log_analyzer.py <logfile> --json    # render structured metrics

Usage (library):
    from log_analyzer import analyze
    metrics = analyze(logfile_path)
    metrics["tasks"]   -> list[dict] per-task timings
    metrics["summary"] -> aggregate prompt/decode speeds
    metrics["model"]   -> model name from log
    metrics["context"] -> context size from log
"""
import argparse
import json
import re
import sys

# =============================================================================
# Parsing
# =============================================================================

# "prompt eval time =   17013.08 ms / 13911 tokens (    1.22 ms per token,   817.67 tokens per second)"
# "eval time =    5791.12 ms /   327 tokens (   17.71 ms per token,    56.47 tokens per second)"
_TIMING_RE = re.compile(
    r'(prompt eval time|eval time)\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens\s*\([^)]+,\s*([\d.]+)\s+tokens per second\)'
)
_TASK_RE = re.compile(r'id\s+(\d+)\s+\|\s+task\s+(-?\d+)')
_MODEL_RE = re.compile(r'^\s*Model:\s+(.+)$', re.MULTILINE)
_CONTEXT_RE = re.compile(r'Context size[:=]\s+(\d+)')
# Periodic slot timing: "n_decoded =    100, tg =  11.17 t/s, tg_3s =  11.17 t/s"
_NDECODED_RE = re.compile(
    r'id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+n_decoded\s*=\s*(\d+).*?tg\s*=\s*([\d.]+)\s*t/s'
)
# "loaded N checkpoints from <path>"
_LOADED_CKPTS_RE = re.compile(r'loaded\s+(\d+)\s+checkpoints\s+from')
# "prompt eval time" lines also include the prefill speed which we want
# separately from the aggregate t/s


def parse_log(text):
    """Extract per-task timings from a llama-server log.

    Returns a list of dicts sorted by task id; each dict has both pp and decode
    timings. Tasks missing one side are dropped.
    """
    task_data = {}
    current_task = None

    for line in text.splitlines():
        m = _TASK_RE.search(line)
        if m:
            current_task = int(m.group(2))
            task_data.setdefault(current_task, {
                'pp': None, 'decode': None,
                'prompt_tokens': None, 'decode_tokens': None,
                'prompt_time_ms': None, 'decode_time_ms': None,
                'n_decoded_samples': [],
                'tg_samples': [],
            })

        m = _TIMING_RE.search(line)
        if m and current_task is not None:
            typ, time_ms, tokens, speed = m.groups()
            entry = task_data[current_task]
            speed_f = float(speed)
            tokens_i = int(tokens)
            time_ms_f = float(time_ms)
            if typ == 'prompt eval time':
                entry['pp'] = speed_f
                entry['prompt_tokens'] = tokens_i
                entry['prompt_time_ms'] = time_ms_f
            else:
                entry['decode'] = speed_f
                entry['decode_tokens'] = tokens_i
                entry['decode_time_ms'] = time_ms_f

        m = _NDECODED_RE.search(line)
        if m and current_task is not None:
            _, _, n_dec, tg = m.groups()
            task_data[current_task]['n_decoded_samples'].append(int(n_dec))
            task_data[current_task]['tg_samples'].append(float(tg))

    tasks = []
    for tid, d in task_data.items():
        if d['pp'] is None or d['decode'] is None:
            continue
        tasks.append({
            'task_id': tid,
            'prompt_tokens': d['prompt_tokens'],
            'decode_tokens': d['decode_tokens'],
            'pp_speed': d['pp'],
            'decode_speed': d['decode'],
            'pp_time_ms': d['prompt_time_ms'],
            'decode_time_ms': d['decode_time_ms'],
            'tg_samples': d['tg_samples'],
        })
    tasks.sort(key=lambda x: x['task_id'])
    return tasks


def extract_context_size(text):
    m = _CONTEXT_RE.search(text)
    return int(m.group(1)) if m else None


def extract_model_name(text):
    m = _MODEL_RE.search(text)
    return m.group(1).strip() if m else "Unknown"


def extract_loaded_checkpoints(text):
    """Return count of checkpoints the server loaded on startup (>0 = SSD cache present)."""
    m = _LOADED_CKPTS_RE.search(text)
    return int(m.group(1)) if m else 0


# =============================================================================
# Aggregation
# =============================================================================

# Threshold separating "long context" workloads in the workload-type breakdown.
LONG_CONTEXT_THRESHOLD = 5000


def _avg(seq):
    return sum(seq) / len(seq) if seq else 0


def _stdev(seq):
    if len(seq) < 2:
        return 0
    mean = _avg(seq)
    var = sum((x - mean) ** 2 for x in seq) / (len(seq) - 1)
    return var ** 0.5


def summarize(tasks, long_threshold=LONG_CONTEXT_THRESHOLD):
    """Aggregate per-task timings into a workload-type breakdown.

    Returns a dict with mean pp/decode speeds for long-context (>threshold),
    short-context, and overall tasks; plus p50/p95 decode speed and tg
    stability (mean of stddev across all sampled decodes).
    """
    if not tasks:
        return {
            'n_tasks': 0,
            'n_long': 0, 'n_short': 0,
            'long_pp': 0, 'long_decode': 0,
            'short_pp': 0, 'short_decode': 0,
            'all_pp': 0, 'all_decode': 0,
            'decode_p50': 0, 'decode_p95': 0,
            'tg_stdev_mean': 0,
        }

    long_tasks = [t for t in tasks if t['prompt_tokens'] > long_threshold]
    short_tasks = [t for t in tasks if t['prompt_tokens'] <= long_threshold]

    decode_speeds = sorted(t['decode_speed'] for t in tasks)
    p50 = _percentile(decode_speeds, 0.50)
    p95 = _percentile(decode_speeds, 0.95)

    # Per-task tg stability: mean of stdev across the sampled decodes per task.
    # Zero variance means the per-task tg was rock-stable across the run.
    per_task_stdev = []
    for t in tasks:
        if len(t['tg_samples']) >= 2:
            per_task_stdev.append(_stdev(t['tg_samples']))

    return {
        'n_tasks': len(tasks),
        'n_long': len(long_tasks),
        'n_short': len(short_tasks),
        'long_pp': _avg([t['pp_speed'] for t in long_tasks]),
        'long_decode': _avg([t['decode_speed'] for t in long_tasks]),
        'short_pp': _avg([t['pp_speed'] for t in short_tasks]),
        'short_decode': _avg([t['decode_speed'] for t in short_tasks]),
        'all_pp': _avg([t['pp_speed'] for t in tasks]),
        'all_decode': _avg([t['decode_speed'] for t in tasks]),
        'decode_p50': p50,
        'decode_p95': p95,
        'tg_stdev_mean': _avg(per_task_stdev),
    }


def _percentile(sorted_data, p):
    """Linear-interpolation percentile matching numpy's default behaviour.

    For 3 sorted values [a, b, c] and p=0.95 the returned value is roughly
    b + 0.9 * (c - b), which gives a strictly monotonic p50 <= p95 even on
    tiny samples.
    """
    if not sorted_data:
        return 0
    if len(sorted_data) == 1:
        return sorted_data[0]
    rank = p * (len(sorted_data) - 1)
    lower = int(rank)
    frac = rank - lower
    if lower + 1 < len(sorted_data):
        return sorted_data[lower] + frac * (sorted_data[lower + 1] - sorted_data[lower])
    return sorted_data[lower]


# =============================================================================
# Markdown rendering
# =============================================================================

def _bar(value, max_value, width=40, char=''):
    if not max_value:
        return ''
    ratio = min(value / max_value, 1.0)
    return char * int(round(ratio * width))


def _fmt(value, default='N/A'):
    return f'{value:.1f}' if value else default


def render_markdown(metrics):
    """Render the structured metrics dict as a Markdown report.

    `metrics` must be the dict returned by analyze().
    """
    model = metrics.get('model', 'Unknown')
    context = metrics.get('context')
    summary = metrics['summary']
    tasks = metrics['tasks']

    if not tasks:
        return (
            f'# {model}\n\n'
            f'**Context size:** {context if context is not None else "?"} tokens\n\n'
            'No tasks with both prompt and decode timing found.'
        )

    lines = [f'# {model} — Cache Run Analysis', '']
    if context is not None:
        lines.append(f'**Context size:** {context} tokens')
    lines += [
        f'**Tasks captured:** {summary["n_tasks"]} '
        f'(long-context >{LONG_CONTEXT_THRESHOLD} tokens: {summary["n_long"]}, '
        f'short-context: {summary["n_short"]})',
        '',
        '## Performance by Workload Type',
        '',
        '| Metric | Long Context | Short Context | Overall |',
        '| :--- | :---: | :---: | :---: |',
    ]

    pp_max = max(summary['long_pp'], summary['short_pp'], summary['all_pp'], 1)
    dec_max = max(summary['long_decode'], summary['short_decode'], summary['all_decode'], 1)
    lines.append(
        f'| **Prompt Processing** | '
        f'{_bar(summary["long_pp"], pp_max, 30, "")} {_fmt(summary["long_pp"])} t/s | '
        f'{_bar(summary["short_pp"], pp_max, 30, "")} {_fmt(summary["short_pp"])} t/s | '
        f'{_bar(summary["all_pp"], pp_max, 30, "")} {_fmt(summary["all_pp"])} t/s |'
    )
    lines.append(
        f'| **Decode (Generation)** | '
        f'{_bar(summary["long_decode"], dec_max, 30, "")} {_fmt(summary["long_decode"])} t/s | '
        f'{_bar(summary["short_decode"], dec_max, 30, "")} {_fmt(summary["short_decode"])} t/s | '
        f'{_bar(summary["all_decode"], dec_max, 30, "")} {_fmt(summary["all_decode"])} t/s |'
    )
    lines.append('')

    lines.append('## Decode Speed Distribution')
    lines.append('')
    lines.append(f'- **p50 decode:** {_fmt(summary["decode_p50"])} t/s')
    lines.append(f'- **p95 decode:** {_fmt(summary["decode_p95"])} t/s')
    if summary['tg_stdev_mean']:
        lines.append(f'- **Mean per-task stdev:** {_fmt(summary["tg_stdev_mean"])} t/s')
    lines.append('')

    lines.append(f'## Decode Speed Stability Across {summary["n_tasks"]} Turns')
    lines.append('')
    max_decode = max(t['decode_speed'] for t in tasks) or 1
    for t in tasks:
        bar = _bar(t['decode_speed'], max_decode, 40, '')
        lines.append(
            f'Task {t["task_id"]:3d}  {bar} {_fmt(t["decode_speed"]):>5} t/s'
        )
    lines.append('')
    return '\n'.join(lines)


# =============================================================================
# Public API
# =============================================================================

def analyze(logfile):
    """Analyze a single llama-server log file.

    Returns a dict:
        tasks:   list of per-task timings
        summary: aggregate stats (see summarize())
        model:   model name from log
        context: context size from log
        loaded_checkpoints: int (0 if none)
    """
    with open(logfile) as f:
        text = f.read()
    tasks = parse_log(text)
    return {
        'tasks': tasks,
        'summary': summarize(tasks),
        'model': extract_model_name(text),
        'context': extract_context_size(text),
        'loaded_checkpoints': extract_loaded_checkpoints(text),
    }


def analyze_text(text):
    """Same as analyze() but takes a raw string (for tests)."""
    tasks = parse_log(text)
    return {
        'tasks': tasks,
        'summary': summarize(tasks),
        'model': extract_model_name(text),
        'context': extract_context_size(text),
        'loaded_checkpoints': extract_loaded_checkpoints(text),
    }


def _cli():
    p = argparse.ArgumentParser(
        description='Analyze a llama-server log and emit Markdown or JSON.',
    )
    p.add_argument('logfile', help='Path to llama-server log (use - for stdin)')
    p.add_argument('--json', action='store_true', help='Emit JSON instead of Markdown')
    args = p.parse_args()

    if args.logfile == '-':
        text = sys.stdin.read()
        tasks = parse_log(text)
        metrics = {
            'tasks': tasks,
            'summary': summarize(tasks),
            'model': extract_model_name(text),
            'context': extract_context_size(text),
            'loaded_checkpoints': extract_loaded_checkpoints(text),
        }
    else:
        metrics = analyze(args.logfile)

    if args.json:
        json.dump(metrics, sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        sys.stdout.write(render_markdown(metrics))


if __name__ == '__main__':
    _cli()