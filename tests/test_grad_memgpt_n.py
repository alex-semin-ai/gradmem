"""Tests for grad_memgpt_n.py and run_enc_gradmem_n_on_mqar.py (N1 write-step rule, C4 random start)."""
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import HfArgumentParser

import run_enc_gradmem_n_on_mqar as N
import run_enc_gradmem_on_mqar as S
from tests.test_enc_gradmem import batch
from tests.test_enc_gradmem_script import write_parent

BASE = ['--exp_path', 'x', '--per_device_batch_size', '4', '--vocab_size', '64', '--n_layer', '2',
        '--n_head', '2', '--n_embd', '16', '--n_mem_tokens', '2', '--K', '1', '--last_K_second_order', '1',
        '--inner_lr', '0.4', '--use_adam', 'false', '--grad_mode', 'second', '--base_model', 'llama']
PARENT = {}


def parent():
    if 'path' not in PARENT:
        a = HfArgumentParser(S.CombinedArgs).parse_args_into_dataclasses(BASE)[0]
        PARENT['path'] = write_parent(Path(tempfile.mkdtemp()), a)
    return PARENT['path']


def parse(module, *extra):
    a = HfArgumentParser(module.CombinedArgs).parse_args_into_dataclasses(BASE + list(extra))[0]
    a.init_checkpoint = parent()
    module.check_args(a)
    return a


def build(module, *extra):
    torch.manual_seed(1)
    return module.build_model(parse(module, *extra))


def run(model, b, train=False):
    model.train(train)
    with torch.set_grad_enabled(train):
        return model(b['input_ids'], labels=b['labels'])


def test_default_equals_the_old_combined_model():
    b = batch()
    old = run(build(S, '--start_mode', 'scaled', '--start_norm', '1.6', '--K', '2', '--last_K_second_order', '2'), b)
    new = run(build(N, '--start_mode', 'scaled', '--start_norm', '1.6', '--K', '2', '--last_K_second_order', '2'), b)
    assert torch.allclose(old['loss'], new['loss'], atol=1e-6)
    assert torch.allclose(old['predictions'].float(), new['predictions'].float())


def test_normalized_step_has_relative_step_alpha():
    m = build(N, '--inner_step_scale', 'normalized', '--inner_lr', '0.05')
    stats = run(m, batch())['inner_loop_stats']
    assert abs(stats['inner_rel_step_first'].item() - 0.05) < 1e-4
    assert abs(stats['inner_rel_step'].item() - 0.05) < 1e-4


def test_mem_sq_equals_plain_step_with_scaled_lr():
    """With the scaled start every example has norm L, so mem_sq with lr a equals the plain step with lr a (L/ref)^2."""
    b = batch()
    mem_sq = run(build(N, '--start_mode', 'scaled', '--start_norm', '3.0', '--inner_step_scale', 'mem_sq',
                       '--inner_step_ref_norm', '1.5', '--inner_lr', '0.1'), b)
    plain = run(build(N, '--start_mode', 'scaled', '--start_norm', '3.0', '--inner_lr', '0.4'), b)
    assert torch.allclose(mem_sq['loss'], plain['loss'], atol=1e-5)


def test_mem_sq_default_ref_is_the_mem_norm():
    m = build(N, '--inner_step_scale', 'mem_sq')
    assert m.gradmem.inner_step_ref_norm is None
    p = torch.ones(1, 2, 16)
    g = torch.ones(1, 2, 16)
    scaled = m.gradmem._scale_inner_grad(p, g)
    assert torch.allclose(scaled, g * (p.norm() / m.gradmem.mem.detach().norm()) ** 2)


def gradmem_grad(model, b):
    model.zero_grad(set_to_none=True)
    run(model, b, train=True)['loss'].backward()
    return model.gradmem.model.model.layers[0].self_attn.q_proj.weight.grad.clone()


def test_normalized_step_keeps_the_second_order_path():
    b = batch()
    second = gradmem_grad(build(N, '--inner_step_scale', 'normalized', '--inner_lr', '0.1'), b)
    first = gradmem_grad(build(N, '--inner_step_scale', 'normalized', '--inner_lr', '0.1',
                               '--last_K_second_order', '0'), b)
    assert not torch.allclose(second, first)


def test_random_start_has_length_l_and_no_encoder_direction():
    m = build(N, '--start_mode', 'random', '--start_norm', '1.6')
    ctx = batch()['input_ids']['context_input_ids']
    with torch.no_grad():
        s1, _, stats = m.start_memory(ctx)
        s2, _, _ = m.start_memory(ctx)
        h_enc, _ = N.encoder_write(m.encoder, ctx)
    assert torch.allclose(s1.norm(dim=(1, 2)), torch.full((s1.size(0),), 1.6), atol=1e-5)
    assert not torch.allclose(s1, s2)
    cos = (s1 * h_enc).sum(dim=(1, 2)) / (s1.norm(dim=(1, 2)) * h_enc.norm(dim=(1, 2)))
    assert cos.abs().max() < 0.9
    assert abs(stats['start_norm'].item() - 1.6) < 1e-4


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match='inner_step_scale'):
        parse(N, '--inner_step_scale', 'half')
    with pytest.raises(ValueError, match='use_adam'):
        parse(N, '--inner_step_scale', 'normalized', '--use_adam', 'true')
    with pytest.raises(ValueError, match='start_mode'):
        parse(N, '--start_mode', 'half')


def test_eval_runs_n_builds_both_combined_scripts():
    import eval_runs_n as E
    cfg = dict(per_device_batch_size=4, vocab_size=64, n_layer=2, n_head=2, n_embd=16, n_mem_tokens=2, K=1, last_K_second_order=1,
               inner_lr=0.1, use_adam=False, grad_mode='second', base_model='llama', start_mode='random',
               start_norm=1.6, inner_step_scale='normalized')
    model, a = E.build('run_enc_gradmem_n_on_mqar.py', cfg)
    assert isinstance(model, N.EncGradMemN) and model.gradmem.inner_step_scale == 'normalized'
    model, _ = E.build('run_enc_gradmem_on_mqar.py', {k: v for k, v in cfg.items()
                                                      if k not in ('inner_step_scale',)} | {'start_mode': 'scaled'})
    assert not isinstance(model, N.EncGradMemN)
