"""Tests for the RMT baseline on dense MQAR (run_rmt_on_mqar.py, rmt.py read_loss_alignment)."""
import numpy as np
import pytest
import torch
from torch import nn

from llama_config import llama_3_2_1b_config
from rmt import RMT2SegmConfig
from run_rmt_on_mqar import GradMemMQARDataset, MQARRMT, collate_fn
from zoology_mqar_data import build_mqar_datasets

VOCAB = 64
PAIRS = 4


def _tiny_llama_config():
    config = llama_3_2_1b_config()
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.hidden_size = 16
    config.head_dim = 8
    config.intermediate_size = 64
    config.rope_scaling = None
    config.rope_theta = 10000.0
    config.max_position_embeddings = 128
    config.torch_dtype = "float32"   # as in the training scripts; the Llama-3.2-1B config says bfloat16
    config.vocab_size = VOCAB
    config.pad_token_id = 0
    config.bos_token_id = None
    config.eos_token_id = None
    config.use_cache = False
    return config


def _tiny_model(read_loss_alignment):
    torch.manual_seed(0)
    config = RMT2SegmConfig(base_config=_tiny_llama_config(), n_mem_tokens=2, K=1,
                            read_loss_alignment=read_loss_alignment,
                            mqar_vocab_size=VOCAB, mqar_dense_queries=True)
    return MQARRMT(config).eval()


def _tiny_batch():
    train, _, _, _ = build_mqar_datasets(vocab_size=VOCAB, input_seq_len=3 * PAIRS, num_kv_pairs=PAIRS,
                                         train_num_examples=8, valid_num_examples=2, power_a=0.01,
                                         random_non_queries=False, data_seed=0, dense_queries=True)
    dataset = GradMemMQARDataset(train, context_size=2 * PAIRS)
    return collate_fn([dataset[i] for i in range(4)])


@pytest.mark.forward
@pytest.mark.all
def test_query_position_loss_scores_each_query_against_its_own_answer():
    model = _tiny_model('query_position')
    batch = _tiny_batch()
    with torch.no_grad():
        out = model(batch['input_ids'], labels=batch['labels'])
    logits, labels = out['predictions'], batch['labels']
    # one prediction per query position, aligned with the labels
    assert logits.shape[:2] == labels.shape
    expected = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(out['loss'], expected)


@pytest.mark.forward
@pytest.mark.all
def test_causal_alignment_differs_and_stays_the_default():
    assert RMT2SegmConfig().read_loss_alignment == 'causal'
    batch = _tiny_batch()
    with torch.no_grad():
        loss_query = _tiny_model('query_position')(batch['input_ids'], labels=batch['labels'])['loss']
        loss_causal = _tiny_model('causal')(batch['input_ids'], labels=batch['labels'])['loss']
    assert not torch.allclose(loss_query, loss_causal)


@pytest.mark.forward
@pytest.mark.all
def test_metrics_see_aligned_predictions():
    # compute_metrics_fn drops a first logit only when predictions are one longer than labels (GradMem prefix).
    # For RMT the lengths must be equal, so nothing is dropped and each prediction meets its own label.
    model = _tiny_model('query_position')
    batch = _tiny_batch()
    with torch.no_grad():
        preds = model(batch['input_ids'])['predictions'].argmax(-1).numpy()
    assert preds.shape == batch['labels'].numpy().shape
    assert np.all(batch['labels'].numpy() != -100)   # dense MQAR: every query position has an answer


def test_bad_alignment_name_is_rejected():
    with pytest.raises(AssertionError):
        RMT2SegmConfig(read_loss_alignment='next_token')
