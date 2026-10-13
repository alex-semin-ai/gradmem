"""Tests for the fairness fixes. The stop rule, the report's thresholds and budgets, and the checkpoint eval_runs picks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_runs
import run_gradmemgpt_on_mqar as G
import run_rmt_on_mqar as R

spec = importlib.util.spec_from_file_location('make_report', Path(__file__).parents[1] / 'vm' / 'make_report.py')
make_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_report)


@pytest.mark.parametrize('module', [G, R])
def test_stop_rule_needs_consecutive_evals(module):
    cb = module.StopOnMetricValue('all_queries_exact_match', 0.99, consecutive=3)
    control = SimpleNamespace(should_training_stop=False)
    for v in [0.995, 0.98, 0.991, 0.992]:          # a lucky eval, a drop, then two good ones
        cb.on_evaluate(None, None, control, {'eval_all_queries_exact_match': v})
        assert not control.should_training_stop
    cb.on_evaluate(None, None, control, {'eval_all_queries_exact_match': 0.993})
    assert control.should_training_stop


def test_stop_rule_default_is_the_old_one_eval_rule():
    cb = G.StopOnMetricValue('eval_all_queries_exact_match', 0.99)   # an eval_ prefix used to crash
    control = SimpleNamespace(should_training_stop=False)
    cb.on_evaluate(None, None, control, {'eval_all_queries_exact_match': 0.995})
    assert control.should_training_stop


def test_steps_to_threshold_ignore_single_lucky_evals():
    pts = [(500, 0.5), (1000, 0.99), (1500, 0.9), (2000, 0.99), (2500, 0.995), (3000, 0.991)]
    assert make_report.first_step(pts, 0.99) == 2000
    assert make_report.first_step([(500, 0.5), (1000, 0.995)], 0.99) is None   # stopped after one eval


def test_fixed_budget_carries_the_last_value_of_a_stopped_run():
    pts = [(5000, 0.9), (10000, 0.95), (28500, 0.999)]
    assert make_report.at_budget(pts, 10000) == (0.95, False)
    assert make_report.at_budget(pts, 30000) == (0.999, True)
    assert make_report.at_budget([(5000, 0.9), (30000, 0.95)], 20000) == (None, False)


def test_eval_runs_picks_the_trainer_best_checkpoint(tmp_path):
    for step in (59500, 60000):
        d = tmp_path / 'checkpoints' / f'checkpoint-{step}'
        d.mkdir(parents=True)
        json.dump({'best_model_checkpoint': '/tmp/runs/X/checkpoint-59500'}, open(d / 'trainer_state.json', 'w'))
    assert eval_runs.selected_checkpoint(tmp_path).name == 'checkpoint-59500'
