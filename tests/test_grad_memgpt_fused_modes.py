"""Tests for the fused modes that keep the full gradient step (ctx_add, focus, polyak) in grad_memgpt_fused.py."""
import torch

from tests.test_grad_memgpt_fused import batch, fused, gm_kwargs, original, parent_state, run
import run_gradmem_fused_on_mqar as F


def with_ctx_init(sd):
    """The parent has no ctx_add weights. Like the training script, keep the model's own init for them."""
    torch.manual_seed(1)
    m = F.MQARGradMemGPT(F.GradMemGPTConfig(**gm_kwargs(1, 0.4), fused_mode='ctx_add'))
    full = dict(sd)
    full.update({k: v for k, v in m.state_dict().items() if k.split('.')[0] in ('ctx_q', 'ctx_k', 'ctx_v', 'ctx_gate')})
    return full


def mem_diff(a, b):
    return (a - b).norm(dim=(1, 2))


def test_ctx_add_with_closed_gate_is_plain_gradmem():
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=2), b)
    m = fused(with_ctx_init(sd), 'ctx_add', K=2)
    with torch.no_grad():
        m.ctx_gate.fill_(-1e4)
    out = run(m, b)
    assert torch.allclose(ref['mem'], out['mem'], atol=1e-5)
    assert torch.allclose(ref['loss'], out['loss'], atol=1e-5)


def test_ctx_add_adds_a_gated_step_of_the_memory_length():
    """K=1. The added part has length sigmoid(gate) * ||M_0|| on top of the plain gradient step."""
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=1), b)['mem']
    m = fused(with_ctx_init(sd), 'ctx_add', K=1)
    out = run(m, b)
    gate = torch.sigmoid(m.ctx_gate).item()
    expected = gate * m.mem.detach().norm()
    assert torch.allclose(mem_diff(out['mem'], ref), expected.expand(ref.size(0)), rtol=1e-4)
    assert abs(out['inner_loop_stats']['fused_gate'].item() - gate) < 1e-6


def test_ctx_add_trains_its_new_weights():
    _, sd = parent_state()
    m = fused(with_ctx_init(sd), 'ctx_add', K=2)
    run(m, batch(), train=True)['loss'].backward()
    for p in (m.ctx_q.weight, m.ctx_k.weight, m.ctx_v.weight, m.ctx_gate):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_focus_with_gamma_zero_is_plain_gradmem():
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=2), b)
    out = run(fused(sd, 'focus', K=2, fused_focus_gamma=0.0), b)
    assert torch.allclose(ref['mem'], out['mem'], atol=1e-5)


def test_focus_reweights_the_gradient_but_logs_the_plain_loss():
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=1), b)
    out = run(fused(sd, 'focus', K=1, fused_focus_gamma=1.0), b)
    assert not torch.allclose(ref['mem'], out['mem'], atol=1e-5)
    plain = out['inner_loop_stats']['focus_plain_loss']
    assert torch.allclose(plain, ref['inner_loop_stats']['inner_reconstruction_loss'], atol=1e-5)


def test_polyak_never_steps_further_than_plain_gradmem():
    """The Polyak step is min(lr, L/||g||^2), so each example moves at most as far as with the plain lr."""
    _, sd = parent_state()
    b = batch()
    m0 = original(sd, K=1)
    start = m0.mem.detach().unsqueeze(0)
    ref = run(m0, b)['mem']
    out = run(fused(sd, 'polyak', K=1), b)
    plain_step, polyak_step = mem_diff(ref, start), mem_diff(out['mem'], start)
    assert (polyak_step <= plain_step + 1e-5).all()
    scale = out['inner_loop_stats']['polyak_scale'].item()
    assert 0 < scale <= 1


def test_polyak_with_a_tiny_lr_is_plain_gradmem():
    _, sd = parent_state()
    b = batch()
    ref = run(original(sd, K=1, lr=1e-6), b)
    out = run(fused(sd, 'polyak', K=1, lr=1e-6), b)
    assert torch.allclose(ref['mem'], out['mem'], atol=1e-7)


def test_runner_knows_the_new_settings():
    assert {'fused_gate', 'polyak_scale', 'focus_plain_loss'} <= set(F.TRAIN_COMPONENT_KEYS)
    fields = F.ExperimentArgs.__dataclass_fields__
    assert fields['fused_focus_gamma'].default == 1.0 and fields['fused_gate_init'].default == -4.0
