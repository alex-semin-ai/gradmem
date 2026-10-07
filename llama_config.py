"""Local copy of the meta-llama/Llama-3.2-1B config.

The Hugging Face repo is gated, but the training scripts use only its architecture
fields and overwrite every size. Values copied from config.json of Llama-3.2-1B
(checked against the public mirror unsloth/Llama-3.2-1B).
"""
from transformers import LlamaConfig

LLAMA_3_2_1B = dict(
    architectures=["LlamaForCausalLM"],
    attention_bias=False, attention_dropout=0.0, mlp_bias=False,
    bos_token_id=128000, eos_token_id=128001,
    hidden_act="silu", hidden_size=2048, intermediate_size=8192, head_dim=64,
    num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
    initializer_range=0.02, rms_norm_eps=1e-05, pretraining_tp=1,
    max_position_embeddings=131072, rope_theta=500000.0,
    rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                  "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    tie_word_embeddings=True, torch_dtype="bfloat16", use_cache=True, vocab_size=128256,
)


def llama_3_2_1b_config():
    return LlamaConfig(**LLAMA_3_2_1B)
