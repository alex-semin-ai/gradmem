"""Tests for the fused write (grad_memgpt_fused.py, run_gradmem_fused_on_mqar.py), on tiny models from a tiny RMT parent."""
import pytest
import torch
from transformers import HfArgumentParser

import grad_memgpt_fused as GF
import run_gradmem_fused_on_mqar as F
import run_gradmemgpt_on_mqar as G
from enc_gradmem import load_parent
from grad_memgpt import GradMemGPTConfig
from tests.test_enc_gradmem import MEM, VOCAB, batch, parent_state, tiny_base


def gm_kwargs(K, lr):
    return dict(base_config=tiny_base(), n_mem_tokens=MEM, K=K, last_K_second_order=K, lr=lr, use_adam=False,
                grad_mode='second', attn_implementation='eager', read_loss_alignment='query_position',
                mqar_vocab_size=VOCAB, mqar_dense_queries=True)


def original(sd, K=1, lr=0.4):
    torch.manual_seed(1)
    m = G.MQARGradMemGPT(GradMemGPTConfig(**gm_kwargs(K, lr)))
    load_parent(m, sd)
    return m


def fused(sd, mode, K=1, lr=0.4, norm=5.0, **extra):
    torch.manual_seed(1)
    m = F.MQARGradMemGPT(GF.GradMemGPTConfig(**gm_kwargs(K, lr), fused_mode=mode, fused_norm=norm, **extra))
    load_parent(m, sd)
    return m


def run(model, b, train=False):
    model.train(train)
    with torch.set_grad_enabled(train):
        return model(b['input_ids'], labels=b['labels'], return_mem=True)


@pytest.mark.parametrize('mode', ['none', 'probe'])
def test_none_and_probe_equal_plain_gradmem(mode):
    """The extra slots after the context cannot change the write loss or its gradient (causal attention)."""
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=2), b)
    out = run(fused(sd, mode, K=2), b)
    assert torch.allclose(ref['mem'], out['mem'], atol=1e-5)
    assert torch.allclose(ref['loss'], out['loss'], atol=1e-5)
    assert torch.allclose(ref['inner_loop_stats']['inner_loss'], out['inner_loop_stats']['inner_loss'], atol=1e-5)


def test_sum_is_rescaled_forward_memory_plus_gradient_step():
    _, sd = parent_state()
    b = batch()
    probe = fused(sd, 'probe')
    m_probe = run(probe, b)['mem']                       # mem0 - lr * grad
    mem0 = probe.mem.detach().unsqueeze(0).expand(m_probe.size(0), -1, -1)
    with torch.no_grad():                                # F(mem0) from an RMT-style pass [M; x; M]
        ctx = probe.model.get_input_embeddings()(b['input_ids']['context_input_ids'])
        h = probe.model(inputs_embeds=torch.cat([mem0, ctx, mem0], 1), output_hidden_states=True).hidden_states[-1]
        f = h[:, -MEM:]
    expected = 5.0 * f / f.norm(dim=(1, 2), keepdim=True) + (m_probe - mem0)
    out = run(fused(sd, 'sum'), b)
    assert torch.allclose(out['mem'], expected, atol=1e-4)
    assert torch.allclose(out['mem'].norm(dim=(1, 2)), expected.norm(dim=(1, 2)), atol=1e-4)
    stats = out['inner_loop_stats']
    assert 'fused_f_norm' in stats and 'fused_step_ratio' in stats


def test_sum_trains_through_the_forward_memory():
    """The outer loss reaches the weights through F(M) as well, so the gradient differs from probe."""
    _, sd = parent_state()
    b = batch()
    grads = {}
    for mode in ('probe', 'sum'):
        m = fused(sd, mode, K=2)
        run(m, b, train=True)['loss'].backward()
        grads[mode] = m.model.model.layers[-1].mlp.down_proj.weight.grad.clone()
    assert grads['sum'].norm() > 0
    assert not torch.allclose(grads['sum'], grads['probe'])


def test_sum_with_first_order_last_steps_still_runs():
    _, sd = parent_state()
    m = fused(sd, 'sum', K=2)
    m.last_K_second_order = 1
    run(m, batch(), train=True)['loss'].backward()


def test_bad_settings_are_refused():
    _, sd = parent_state()
    with pytest.raises(ValueError, match='fused_mode'):
        fused(sd, 'half')
    with pytest.raises(ValueError, match='n_ctrl_tokens'):
        torch.manual_seed(1)
        kw = gm_kwargs(1, 0.4)
        kw['n_ctrl_tokens'] = 1
        F.MQARGradMemGPT(GF.GradMemGPTConfig(**kw, fused_mode='sum'))


def test_runner_arguments():
    base = ['--exp_path', 'x', '--per_device_batch_size', '4']
    a = HfArgumentParser(F.ExperimentArgs).parse_args_into_dataclasses(base)[0]
    assert a.fused_mode == 'sum' and a.fused_norm == 5.0
    a = HfArgumentParser(F.ExperimentArgs).parse_args_into_dataclasses(base + ['--fused_mode', 'none'])[0]
    assert a.fused_mode == 'none'
    assert 'fused_f_norm' in F.TRAIN_COMPONENT_KEYS and 'fused_f_norm' not in G.TRAIN_COMPONENT_KEYS
