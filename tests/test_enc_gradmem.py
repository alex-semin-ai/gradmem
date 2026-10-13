"""Tests for the combined model (enc_gradmem.py), on tiny models built from a tiny RMT parent."""
import pytest
import torch

from enc_gradmem import EncGradMem, encoder_write, load_parent
from grad_memgpt import GradMemGPTConfig
from llama_config import llama_3_2_1b_config
from rmt import RMT2SegmConfig
from run_gradmemgpt_on_mqar import MQARGradMemGPT
from run_rmt_on_mqar import GradMemMQARDataset, MQARRMT, collate_fn
from zoology_mqar_data import build_mqar_datasets

VOCAB, PAIRS, MEM = 64, 4, 2


def tiny_base():
    c = llama_3_2_1b_config()
    c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads = 2, 2, 2
    c.hidden_size, c.head_dim, c.intermediate_size = 16, 8, 64
    c.rope_scaling, c.rope_theta, c.max_position_embeddings = None, 10000.0, 128
    c.torch_dtype, c.vocab_size, c.pad_token_id = 'float32', VOCAB, 0
    c.bos_token_id = c.eos_token_id = None
    c.use_cache = False
    return c


def parent_state():
    """A tiny RMT 'E1' trained with the reconstruction loss, saved like the Trainer saves it."""
    torch.manual_seed(0)
    rmt = MQARRMT(RMT2SegmConfig(base_config=tiny_base(), n_mem_tokens=MEM, K=1, use_reconstruction_loss=True,
                                 read_loss_alignment='query_position', mqar_vocab_size=VOCAB, mqar_dense_queries=True))
    with torch.no_grad():                       # make the weights differ from a fresh model
        for p in rmt.parameters():
            p.add_(0.05 * torch.randn_like(p))
    sd = {k: v.clone() for k, v in rmt.state_dict().items() if not k.endswith('lm_head.weight')}
    return rmt.eval(), sd


def build(sd, K=1, **kw):
    torch.manual_seed(1)
    enc = MQARRMT(RMT2SegmConfig(base_config=tiny_base(), n_mem_tokens=MEM, K=1, use_reconstruction_loss=True,
                                 read_loss_alignment='query_position', mqar_vocab_size=VOCAB, mqar_dense_queries=True))
    gm = MQARGradMemGPT(GradMemGPTConfig(base_config=tiny_base(), n_mem_tokens=MEM, K=K, last_K_second_order=K,
                                         lr=0.4, use_adam=False, grad_mode='second', attn_implementation='eager',
                                         read_loss_alignment='query_position', mqar_vocab_size=VOCAB,
                                         mqar_dense_queries=True))
    load_parent(enc, sd)
    load_parent(gm, sd)
    return EncGradMem(enc, gm, **kw)


def batch():
    train, _, _, _ = build_mqar_datasets(vocab_size=VOCAB, input_seq_len=3 * PAIRS, num_kv_pairs=PAIRS,
                                         train_num_examples=8, valid_num_examples=2, power_a=0.01,
                                         random_non_queries=False, data_seed=0, dense_queries=True)
    ds = GradMemMQARDataset(train, context_size=2 * PAIRS)
    return collate_fn([ds[i] for i in range(4)])


def test_k0_reproduces_the_rmt_parent():
    """GradMem θ started from E1 reads M_0 = h_enc exactly like RMT reads its own memory."""
    rmt, sd = parent_state()
    rmt.config.use_reconstruction_loss = False   # compare plain predictions
    rmt.use_reconstruction_loss = False
    model = build(sd, K=0).eval()
    b = batch()
    with torch.no_grad():
        p_rmt = rmt(b['input_ids'])['predictions']
        p_comb = model(b['input_ids'])['predictions'][:, 1:]   # prefix memory adds one pre-query logit
    assert torch.allclose(p_rmt, p_comb, atol=1e-5)


def test_shared_embedding_and_tied_output():
    _, sd = parent_state()
    m = build(sd)
    emb = m.gradmem.model.get_input_embeddings().weight
    assert m.encoder.model.get_input_embeddings().weight.data_ptr() == emb.data_ptr()
    assert m.encoder.model.get_output_embeddings().weight.data_ptr() == emb.data_ptr()
    assert m.gradmem.model.get_output_embeddings().weight.data_ptr() == emb.data_ptr()


def test_projection_starts_as_identity():
    _, sd = parent_state()
    b = batch()
    with torch.no_grad():
        plain = build(sd).eval()(b['input_ids'], labels=b['labels'])['loss']
        proj = build(sd, use_projection=True).eval()(b['input_ids'], labels=b['labels'])['loss']
    assert torch.allclose(plain, proj, atol=1e-5)


def encoder_grad(model, b):
    model.zero_grad(set_to_none=True)
    model.train()(b['input_ids'], labels=b['labels'])['loss'].backward()
    g = model.encoder.model.model.layers[0].self_attn.q_proj.weight.grad
    return 0.0 if g is None else g.norm().item()


def test_encoder_gets_gradient_through_write_steps():
    _, sd = parent_state()
    assert encoder_grad(build(sd, K=2), batch()) > 0


def test_frozen_encoder_gets_no_gradient():
    _, sd = parent_state()
    m = build(sd, K=2, freeze_encoder=True)
    assert encoder_grad(m, batch()) == 0.0
    assert m.gradmem.model.get_input_embeddings().weight.requires_grad   # the shared table still trains


def test_second_order_reaches_the_encoder_differently_than_k0():
    """The encoder's gradient depends on the write steps, so K = 0 and K = 2 give different gradients."""
    _, sd = parent_state()
    b = batch()
    g0 = build(sd, K=0)
    g2 = build(sd, K=2)
    encoder_grad(g0, b), encoder_grad(g2, b)
    a = g0.encoder.model.model.layers[0].self_attn.q_proj.weight.grad
    c = g2.encoder.model.model.layers[0].self_attn.q_proj.weight.grad
    assert not torch.allclose(a, c)


def test_encoder_reconstruction_loss_is_added():
    _, sd = parent_state()
    b = batch()
    with torch.no_grad():
        off = build(sd).eval()(b['input_ids'], labels=b['labels'])
        on = build(sd, encoder_reconstruction_loss=True).eval()(b['input_ids'], labels=b['labels'])
    rec = on['inner_loop_stats']['enc_rec_loss']
    assert rec > 0
    assert torch.allclose(on['loss'], off['loss'] + rec, atol=1e-5)


def test_encoder_write_matches_rmt():
    """encoder_write gives the same memory as RMT2Segm's own write pass."""
    rmt, _ = parent_state()
    rmt.use_reconstruction_loss = False
    b = batch()
    with torch.no_grad():
        mem_rmt = rmt(b['input_ids'], return_mem=True)['mem']
        h_enc, _ = encoder_write(rmt, b['input_ids']['context_input_ids'])
    assert torch.allclose(mem_rmt, h_enc, atol=1e-6)


def test_encoder_reconstruction_loss_equals_rmt_rec_loss():
    """E6rec and E7rec must train the encoder on exactly the reconstruction loss E1 was trained with."""
    rmt, _ = parent_state()
    assert rmt.use_reconstruction_loss
    b = batch()
    with torch.no_grad():
        rmt_rec = rmt(b['input_ids'], labels=b['labels'])['inner_loop_stats']['rec_loss']
        _, enc_rec = encoder_write(rmt, b['input_ids']['context_input_ids'], reconstruction_loss=True)
    assert torch.allclose(rmt_rec, enc_rec, atol=1e-5), (rmt_rec, enc_rec)


def test_invalid_combinations_are_refused():
    _, sd = parent_state()
    with pytest.raises(ValueError):
        build(sd, freeze_encoder=True, encoder_reconstruction_loss=True)
    m = build(sd)
    m.encoder.K = 2
    with pytest.raises(NotImplementedError):
        encoder_write(m.encoder, batch()['input_ids']['context_input_ids'])


def test_load_parent_rejects_missing_weights():
    _, sd = parent_state()
    bad = {('x' + k): v for k, v in sd.items()}
    with pytest.raises(RuntimeError):
        build(bad)


# ---------- start modes (fix for the scale problem) ----------

def start_of(model, b):
    with torch.no_grad():
        start, _, stats = model.start_memory(b['input_ids']['context_input_ids'])
        h_enc, _ = encoder_write(model.encoder, b['input_ids']['context_input_ids'])
    return start, h_enc, stats


def test_scaled_start_keeps_direction_and_sets_length():
    _, sd = parent_state()
    m = build(sd, start_mode='scaled', start_norm=1.6)
    start, h_enc, stats = start_of(m, batch())
    assert torch.allclose(start.norm(dim=(1, 2)), torch.full((start.size(0),), 1.6), atol=1e-5)
    cos = (start * h_enc).sum(dim=(1, 2)) / (start.norm(dim=(1, 2)) * h_enc.norm(dim=(1, 2)))
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)
    assert abs(stats['start_norm'].item() - 1.6) < 1e-4


def test_default_start_norm_is_the_parent_mem_norm_and_is_saved():
    _, sd = parent_state()
    m = build(sd, start_mode='scaled')
    assert abs(m.start_norm_t.item() - sd['mem'].norm().item()) < 1e-5
    assert 'start_norm_t' in m.state_dict()


def test_mixed_start_is_mem_plus_scaled_direction():
    _, sd = parent_state()
    m = build(sd, start_mode='mixed', start_norm=2.0, mix_beta=0.5)
    start, h_enc, _ = start_of(m, batch())
    d = h_enc / h_enc.norm(dim=(1, 2), keepdim=True)
    assert torch.allclose(start, m.gradmem.mem.unsqueeze(0) + 0.5 * 2.0 * d, atol=1e-5)


def test_projection_at_scaled_start_starts_equal_to_scaled_start():
    _, sd = parent_state()
    b = batch()
    with torch.no_grad():
        plain = build(sd, start_mode='scaled', start_norm=1.6).eval()(b['input_ids'], labels=b['labels'])
        proj = build(sd, start_mode='scaled', start_norm=1.6, use_projection=True).eval()(b['input_ids'], labels=b['labels'])
    assert torch.allclose(plain['loss'], proj['loss'], atol=1e-5)
    assert abs(proj['inner_loop_stats']['proj_scale'].item() - 1.0) < 1e-5


def test_shuffled_start_gives_each_example_another_memory():
    _, sd = parent_state()
    m = build(sd, start_mode='scaled', start_norm=1.6)
    b = batch()
    start, _, _ = start_of(m, b)
    m.shuffle_start = True
    shuffled, _, _ = start_of(m, b)
    assert torch.allclose(shuffled, start.roll(1, dims=0))
    assert not torch.allclose(shuffled, start)


def test_invalid_start_mode_is_refused():
    _, sd = parent_state()
    with pytest.raises(ValueError, match='start_mode'):
        build(sd, start_mode='half')
