"""Tests for run_enc_gradmem_on_mqar.py. Argument checks and building the combined model from an RMT parent file."""
import pytest
import torch
from safetensors.torch import save_file
from transformers import HfArgumentParser

import run_enc_gradmem_on_mqar as S
from rmt import RMT2SegmConfig
from run_rmt_on_mqar import MQARRMT


def parse(*extra):
    base = ['--exp_path', 'x', '--per_device_batch_size', '4', '--vocab_size', '64', '--n_layer', '2',
            '--n_head', '2', '--n_embd', '16', '--n_mem_tokens', '2', '--K', '1', '--last_K_second_order', '1',
            '--inner_lr', '0.4', '--use_adam', 'false', '--grad_mode', 'second', '--base_model', 'llama']
    return HfArgumentParser(S.CombinedArgs).parse_args_into_dataclasses(base + list(extra))[0]


def write_parent(tmp_path, args):
    torch.manual_seed(0)
    rmt = MQARRMT(RMT2SegmConfig(base_config=S.base_llama_config(args), n_mem_tokens=args.n_mem_tokens, K=1,
                                 use_reconstruction_loss=True, read_loss_alignment='query_position',
                                 mqar_vocab_size=args.vocab_size, mqar_dense_queries=True))
    sd = {k: v.contiguous() for k, v in rmt.state_dict().items() if not k.endswith('lm_head.weight')}
    path = tmp_path / 'model.safetensors'
    save_file(sd, str(path))
    return str(path)


def test_refuses_runs_without_parent_or_with_bad_flags():
    with pytest.raises(ValueError, match='init_checkpoint'):
        S.check_args(parse())
    with pytest.raises(ValueError, match='frozen encoder'):
        S.check_args(parse('--init_checkpoint', 'p', '--freeze_encoder', 'true', '--encoder_reconstruction_loss', 'true'))
    with pytest.raises(ValueError, match='mem_proj_mode'):
        S.check_args(parse('--init_checkpoint', 'p', '--mem_proj_mode', 'proj'))


def test_builds_combined_model_from_rmt_parent(tmp_path):
    args = parse('--use_projection', 'true')
    args.init_checkpoint = write_parent(tmp_path, args)
    S.check_args(args)
    model = S.build_model(args)
    assert model.proj is not None and torch.equal(model.proj.weight, torch.eye(16))
    emb = model.gradmem.model.get_input_embeddings().weight
    assert model.encoder.model.get_input_embeddings().weight.data_ptr() == emb.data_ptr()
    assert 'enc_rec_loss' in S.G.TRAIN_COMPONENT_KEYS


def test_parent_with_other_memory_size_is_refused(tmp_path):
    args = parse()
    args.init_checkpoint = write_parent(tmp_path, args)
    args.n_mem_tokens = 3
    with pytest.raises(ValueError, match='memory'):
        S.build_model(args)
