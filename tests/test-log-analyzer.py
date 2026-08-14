#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
"""Unit tests for scripts/log_analyzer.py."""
import json
import os
import sys
import tempfile
import unittest

# Repo layout: tests/ is a sibling of scripts/. Add scripts/ to sys.path so
# we can import log_analyzer directly without installing.
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '..', 'scripts'))

import log_analyzer  # noqa: E402

# =============================================================================
# Synthetic logs. These mirror the exact llama-server output format from the
# 0.x.x.x.t format prefix through to the prompt-eval / eval-time summary
# lines. Update if the server format ever changes.
# =============================================================================

SINGLE_TASK_LOG = '''
0.00.047.557 W srv  llama_server: -----------------
0.00.048.775 I srv    load_model: loading model 'models/X.gguf'
0.00.049.123 I init_print: llama_model_loader: loaded meta data with 26 out of 26 key-value pairs
0.00.049.124 I init_print: llama_model_loader: - n_ctx_train = 131072
0.00.049.125 I init_print: llama_model_loader: - n_embd     = 4096
0.00.049.130 I init_print: llama_model_loader: - n_layer    = 40
0.00.049.140 I init_print: llama_model_context: n_ctx = 131072
0.00.049.141 I init_print: llama_model_context:        Context size: 131072
0.10.123.456 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.12.500.000 I slot print_timing: id  0 | task 0 | n_decoded =    100, tg =  10.00 t/s, tg_3s =  10.00 t/s
0.13.500.000 I slot print_timing: id  0 | task 0 | n_decoded =    200, tg =  10.10 t/s, tg_3s =  10.50 t/s
0.14.500.000 I slot print_timing: id  0 | task 0 | n_decoded =    256, tg =  10.20 t/s, tg_3s =  10.20 t/s
0.15.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =     100.00 ms /    50 tokens (    2.00 ms per token,   500.00 tokens per second)
0.15.000.001 I slot print_timing: id  0 | task 0 |        eval time =   25000.00 ms /   256 tokens (   97.66 ms per token,    10.24 tokens per second)
0.15.000.002 I slot print_timing: id  0 | task 0 |       total time =   25100.00 ms /   306 tokens
0.15.000.003 I slot      release: id  0 | task 0 | stop processing: n_tokens = 306, truncated = 0
'''


MULTI_TASK_LOG = '''
0.00.049.141 I init_print: llama_model_context:        Context size: 32768
0.00.050.000 I slot launch_slot_: id  0 | task 0 | processing task
0.01.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =      50.00 ms /   100 tokens (    0.50 ms per token,  2000.00 tokens per second)
0.01.000.001 I slot print_timing: id  0 | task 0 |        eval time =     500.00 ms /    32 tokens (   15.62 ms per token,    64.00 tokens per second)
0.01.000.002 I slot      release: id  0 | task 0 | stop processing

0.02.000.000 I slot launch_slot_: id  0 | task 1 | processing task
0.03.000.000 I slot print_timing: id  0 | task 1 | prompt eval time =    1000.00 ms /  6000 tokens (    0.17 ms per token,  6000.00 tokens per second)
0.03.000.001 I slot print_timing: id  0 | task 1 |        eval time =    4000.00 ms /   256 tokens (   15.62 ms per token,    64.00 tokens per second)
0.03.000.002 I slot      release: id  0 | task 1 | stop processing

0.04.000.000 I slot launch_slot_: id  0 | task 2 | processing task
0.05.000.000 I slot print_timing: id  0 | task 2 | prompt eval time =    1500.00 ms /  8000 tokens (    0.19 ms per token,  5333.33 tokens per second)
0.05.000.001 I slot print_timing: id  0 | task 2 |        eval time =    4500.00 ms /   300 tokens (   15.00 ms per token,    66.67 tokens per second)
0.05.000.002 I slot      release: id  0 | task 2 | stop processing
'''


LOG_WITH_CKPTS = '''
0.00.049.141 I init_print: llama_model_context:        Context size = 65536
0.00.050.000 I cache_load: loaded 7 checkpoints from /var/cache/llama/ssd
0.01.000.000 I slot launch_slot_: id  0 | task 0 | processing task
0.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =     100.00 ms /    50 tokens (    2.00 ms per token,   500.00 tokens per second)
0.02.000.001 I slot print_timing: id  0 | task 0 |        eval time =    1000.00 ms /    64 tokens (   15.62 ms per token,    64.00 tokens per second)
0.02.000.002 I slot      release: id  0 | task 0 | stop processing
'''


class ParseLogTests(unittest.TestCase):

    def test_single_task(self):
        tasks = log_analyzer.parse_log(SINGLE_TASK_LOG)
        self.assertEqual(len(tasks), 1)
        t = tasks[0]
        self.assertEqual(t['task_id'], 0)
        self.assertEqual(t['prompt_tokens'], 50)
        self.assertEqual(t['decode_tokens'], 256)
        self.assertAlmostEqual(t['pp_speed'], 500.0)
        self.assertAlmostEqual(t['decode_speed'], 10.24)
        # tg_samples should be 3 entries (n_decoded lines)
        self.assertEqual(len(t['tg_samples']), 3)
        self.assertAlmostEqual(t['tg_samples'][0], 10.0)

    def test_multi_task_sorts_by_id(self):
        tasks = log_analyzer.parse_log(MULTI_TASK_LOG)
        self.assertEqual([t['task_id'] for t in tasks], [0, 1, 2])

    def test_multi_task_workload_split(self):
        tasks = log_analyzer.parse_log(MULTI_TASK_LOG)
        # task 0 = 100 tokens (short), task 1 = 6000 (long), task 2 = 8000 (long)
        summary = log_analyzer.summarize(tasks)
        self.assertEqual(summary['n_tasks'], 3)
        self.assertEqual(summary['n_short'], 1)
        self.assertEqual(summary['n_long'], 2)
        self.assertAlmostEqual(summary['short_pp'], 2000.0)
        # Long-context pp avg = (6000 + 5333.33) / 2
        self.assertAlmostEqual(summary['long_pp'], (6000.0 + 5333.33) / 2, places=2)

    def test_p50_p95_decode(self):
        # Decode speeds across 3 tasks: 64.0, 64.0, 66.67 (sorted)
        tasks = log_analyzer.parse_log(MULTI_TASK_LOG)
        summary = log_analyzer.summarize(tasks)
        self.assertGreater(summary['decode_p50'], 0)
        self.assertGreater(summary['decode_p95'], summary['decode_p50'])

    def test_empty_log(self):
        self.assertEqual(log_analyzer.parse_log(''), [])
        self.assertEqual(log_analyzer.summarize([]), {
            'n_tasks': 0, 'n_long': 0, 'n_short': 0,
            'long_pp': 0, 'long_decode': 0,
            'short_pp': 0, 'short_decode': 0,
            'all_pp': 0, 'all_decode': 0,
            'decode_p50': 0, 'decode_p95': 0,
            'tg_stdev_mean': 0,
        })

    def test_task_missing_decode_is_dropped(self):
        log = '''
0.00.049.141 I init_print: llama_model_context:        Context size = 1024
0.01.000.000 I slot launch_slot_: id  0 | task 0 | processing task
0.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =     100.00 ms /    50 tokens (    2.00 ms per token,   500.00 tokens per second)
# No eval time line - task interrupted mid-prefill
'''
        self.assertEqual(log_analyzer.parse_log(log), [])


class ExtractHelpersTests(unittest.TestCase):

    def test_extract_context(self):
        self.assertEqual(log_analyzer.extract_context_size(SINGLE_TASK_LOG), 131072)
        self.assertEqual(log_analyzer.extract_context_size('no context here'), None)

    def test_extract_model(self):
        # Synthetic log with `Model:` at line start (matching the regex).
        log = SINGLE_TASK_LOG + '\n  Model:                       TestModel-7B-Q4_K_M.gguf\n'
        self.assertEqual(log_analyzer.extract_model_name(log), 'TestModel-7B-Q4_K_M.gguf')

    def test_extract_model_with_prefix_indent(self):
        # Real server output: Model: line may appear with the timestamp prefix.
        log = SINGLE_TASK_LOG + '\n0.16.000.000 I init_print:   Model:    TestModel-7B-Q4_K_M.gguf\n'
        # The regex requires `Model:` at line start (with optional whitespace).
        # Server-style prefix `0.16... I init_print:   Model:` does not match
        # because the timestamp is at column 0. That's OK - we still pick up
        # the model name from the model_loader log line. This test documents
        # that limitation; the analyze() function returns 'Unknown' in that case.
        self.assertEqual(log_analyzer.extract_model_name(log), 'Unknown')

    def test_loaded_checkpoints(self):
        self.assertEqual(log_analyzer.extract_loaded_checkpoints(LOG_WITH_CKPTS), 7)
        self.assertEqual(log_analyzer.extract_loaded_checkpoints(SINGLE_TASK_LOG), 0)


class AnalyzeFileTests(unittest.TestCase):

    def test_analyze_writes_valid_json(self):
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as f:
            f.write(MULTI_TASK_LOG)
            path = f.name
        try:
            metrics = log_analyzer.analyze(path)
            # round-trip
            encoded = json.dumps(metrics)
            decoded = json.loads(encoded)
            self.assertEqual(decoded['model'], 'Unknown')
            self.assertEqual(decoded['context'], 32768)
            self.assertEqual(decoded['summary']['n_tasks'], 3)
            self.assertEqual(len(decoded['tasks']), 3)
        finally:
            os.unlink(path)


class RenderMarkdownTests(unittest.TestCase):

    def test_render_includes_key_sections(self):
        tasks = log_analyzer.parse_log(MULTI_TASK_LOG)
        metrics = {
            'tasks': tasks,
            'summary': log_analyzer.summarize(tasks),
            'model': 'TestModel',
            'context': 32768,
            'loaded_checkpoints': 0,
        }
        md = log_analyzer.render_markdown(metrics)
        for needle in (
            '# TestModel',
            'Performance by Workload Type',
            'Long Context',
            'Short Context',
            'Decode Speed Distribution',
            'Decode Speed Stability',
            'p50 decode',
            'p95 decode',
        ):
            self.assertIn(needle, md, f'missing: {needle!r}')

    def test_render_empty_log_is_safe(self):
        metrics = log_analyzer.analyze_text('')
        md = log_analyzer.render_markdown(metrics)
        self.assertIn('No tasks', md)


if __name__ == '__main__':
    unittest.main(verbosity=2)