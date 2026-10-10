"""Debug why RMT stalls at the copy baseline on dense MQAR.

Tier 0 (plumbing, works on an untrained or trained model):
  1. the memory path gets gradient           (self.mem appears only in the write input)
  2. the memory depends on the context       (change one value token)
  3. the memory depends on the binding       (re-pair keys and values, same tokens)
  4. the read depends on the memory          (swap memories between examples)

Tier 1 (meaningful on a trained checkpoint):
  normal accuracy, accuracy with swapped memory, accuracy with zero memory,
  predictions on a re-paired context, and how often a prediction is one of the
  example's own context values (the copy shortcut).

Usage (from the repo root):
  python debug_rmt.py                                    # Tier 0 on a fresh model, 8 pairs
  python debug_rmt.py --checkpoint <run>/checkpoint-N/model.safetensors
The model settings are read from the run's config.json when a checkpoint is given.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from llama_config import llama_3_2_1b_config
from rmt import RMT2SegmConfig
from run_rmt_on_mqar import GradMemMQARDataset, MQARRMT, collate_fn
from zoology_mqar_data import build_mqar_datasets


# ----------------------------------------------------------------------------- setup

def find_run_args(checkpoint):
    """Read the training arguments of the run that produced the checkpoint, if they can be found."""
    ckpt_dir = Path(checkpoint).resolve().parent
    candidates = [ckpt_dir.parent / 'config.json',                 # local run folder
                  ckpt_dir.parent.parent / 'files' / 'config.json',  # Drive mirror: runs/<RUN>/files
                  ckpt_dir.parent.parent / 'config.json']            # Drive mirror: notebook record
    for path in candidates:
        if path.is_file():
            data = json.load(open(path))
            if 'cli_args' in data:
                return data['cli_args'], path
            if 'config' in data:
                return data['config'], path
    return {}, None


def build_model(a, device):
    config = llama_3_2_1b_config()
    config.num_hidden_layers = a.n_layer
    config.num_attention_heads = a.n_head
    config.num_key_value_heads = a.n_head
    config.hidden_size = a.n_embd
    config.head_dim = a.n_embd // a.n_head
    config.intermediate_size = a.n_embd * 4
    config.rope_scaling = None
    config.rope_theta = 10000.0
    config.max_position_embeddings = 1024
    config.torch_dtype = 'float32'
    config.vocab_size = a.vocab_size
    config.pad_token_id = 0
    config.bos_token_id = None
    config.eos_token_id = None
    config.use_cache = False
    rmt_config = RMT2SegmConfig(base_config=config, n_mem_tokens=a.n_mem_tokens, K=a.K,
                                use_mem_proj=a.mem_proj_mode != 'none', mem_proj_mode=a.mem_proj_mode,
                                read_loss_alignment='query_position', attn_implementation='eager',
                                mqar_vocab_size=a.vocab_size, mqar_dense_queries=True)
    model = MQARRMT(rmt_config)
    if a.checkpoint:
        missing, unexpected = model.load_state_dict(load_file(a.checkpoint), strict=False)
        missing = [k for k in missing if not k.endswith('lm_head.weight')]   # tied to the embeddings
        print(f'checkpoint loaded. missing keys {len(missing)}, unexpected keys {len(unexpected)}')
        if missing:
            print('  MISSING', missing[:10])
        if unexpected:
            print('  UNEXPECTED', unexpected[:10])
    return model.to(device)


def eval_batches(a, device):
    _, valid, _, _ = build_mqar_datasets(vocab_size=a.vocab_size, input_seq_len=3 * a.pairs, num_kv_pairs=a.pairs,
                                         train_num_examples=8, valid_num_examples=a.n_eval, power_a=0.01,
                                         random_non_queries=False, data_seed=a.data_seed, dense_queries=True)
    dataset = GradMemMQARDataset(valid, context_size=2 * a.pairs)
    for start in range(0, len(dataset), a.batch):
        batch = collate_fn([dataset[i] for i in range(start, min(start + a.batch, len(dataset)))])
        yield {'context_input_ids': batch['input_ids']['context_input_ids'].to(device),
               'query_input_ids': batch['input_ids']['query_input_ids'].to(device)}, batch['labels'].to(device)


# ----------------------------------------------------------------------------- helpers

def derangement(n, generator=None):
    """A random permutation with no fixed point (n >= 2)."""
    perm = torch.randperm(n, generator=generator)
    out = torch.empty(n, dtype=torch.long)
    out[perm] = perm.roll(-1)
    return out


def write_memory(model, inputs):
    return model(inputs, return_mem=True)['mem']


def read_with_memory(model, mem, query_ids):
    """The RMT read pass with a memory given from outside (no control tokens, no read projection)."""
    if model.n_ctrl_tokens > 0 or model.mem_proj_mode == 'proj_rw':
        raise NotImplementedError('read_with_memory supports n_ctrl_tokens=0 and mem_proj_mode none or proj')
    qry_emb = model.model.get_input_embeddings()(query_ids)
    x = torch.cat([mem, qry_emb], dim=1)
    mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
    return model.model(inputs_embeds=x, attention_mask=mask).logits[:, model.n_mem_tokens:]


def repair_context(context, labels, query_ids, generator=None):
    """Keep the key and value tokens, but give every key a different value.

    Returns the new context and the new correct answers for the same queries."""
    context = context.clone()
    new_labels = labels.clone()
    keys, values = context[:, 0::2], context[:, 1::2]
    for b in range(context.size(0)):
        values_b = values[b][derangement(values.size(1), generator).to(values.device)]
        context[b, 1::2] = values_b
        answer_of = {int(k): int(v) for k, v in zip(keys[b], values_b)}
        new_labels[b] = torch.tensor([answer_of[int(q)] for q in query_ids[b]], device=labels.device)
    return context, new_labels


def rel_change(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# ----------------------------------------------------------------------------- tier 0

def tier0(model, inputs, labels, generator):
    print('\n=== Tier 0. Plumbing ===')
    ctx, qry = inputs['context_input_ids'], inputs['query_input_ids']

    # 1. gradient on the memory path
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(inputs, labels=labels)['loss']
    loss.backward()
    mem_grad = model.mem.grad.norm().item() if model.mem.grad is not None else 0.0
    q_proj = model.model.model.layers[0].self_attn.q_proj.weight.grad
    ref_grad = q_proj.norm().item() if q_proj is not None else float('nan')
    print(f'1. grad norm of self.mem (memory path only)  {mem_grad:.3e}')
    print(f'   grad norm of layer 0 q_proj, for scale      {ref_grad:.3e}')
    print('   ' + ('OK, gradient reaches the write pass' if mem_grad > 1e-8 else
                   'PROBLEM, no gradient reaches the write pass through the memory'))
    model.zero_grad(set_to_none=True)
    model.eval()

    with torch.no_grad():
        mem = write_memory(model, inputs)

        # 2. memory depends on the context: change one value token per example
        ctx2 = ctx.clone()
        ctx2[:, 1] = ctx2[:, 3]          # first value becomes the second value
        mem_value = write_memory(model, {'context_input_ids': ctx2, 'query_input_ids': qry})

        # scale reference: a completely different context (another example's)
        other = derangement(ctx.size(0), generator).to(ctx.device)
        mem_other = write_memory(model, {'context_input_ids': ctx[other], 'query_input_ids': qry})

        # 3. memory depends on the binding: same tokens, re-paired
        ctx_rep, _ = repair_context(ctx, labels, qry, generator)
        mem_rep = write_memory(model, {'context_input_ids': ctx_rep, 'query_input_ids': qry})

        print(f'2. memory change, one value token changed      {rel_change(mem_value, mem):.4f}')
        print(f'3. memory change, keys and values re-paired    {rel_change(mem_rep, mem):.4f}')
        print(f'   memory change, another example entirely     {rel_change(mem_other, mem):.4f}  (scale reference)')

        # 4. read depends on memory
        logits = read_with_memory(model, mem, qry)
        logits_swap = read_with_memory(model, mem[other], qry)
        print(f'4. logits change when memories are swapped     {rel_change(logits_swap, logits):.4f}')
    print('   Reading. 2 and 4 must be clearly above 0. If 3 is far below the scale reference,')
    print('   the memory hardly encodes WHICH value belongs to WHICH key.')


# ----------------------------------------------------------------------------- tier 1

def tier1(model, batches, generator):
    print('\n=== Tier 1. Behaviour on the validation set ===')
    n = 0
    hits = {'normal': 0, 'swap': 0, 'zero': 0, 'rep_vs_new': 0, 'rep_vs_old': 0, 'rep_same_pred': 0, 'in_context': 0}
    with torch.no_grad():
        for inputs, labels in batches:
            ctx, qry = inputs['context_input_ids'], inputs['query_input_ids']
            mem = write_memory(model, inputs)
            pred = read_with_memory(model, mem, qry).argmax(-1)
            other = derangement(ctx.size(0), generator).to(ctx.device)
            pred_swap = read_with_memory(model, mem[other], qry).argmax(-1)
            pred_zero = read_with_memory(model, torch.zeros_like(mem), qry).argmax(-1)
            ctx_rep, labels_rep = repair_context(ctx, labels, qry, generator)
            mem_rep = write_memory(model, {'context_input_ids': ctx_rep, 'query_input_ids': qry})
            pred_rep = read_with_memory(model, mem_rep, qry).argmax(-1)
            values = ctx[:, 1::2]
            in_ctx = (pred.unsqueeze(-1) == values.unsqueeze(1)).any(-1)

            n += labels.numel()
            hits['normal'] += (pred == labels).sum().item()
            hits['swap'] += (pred_swap == labels).sum().item()
            hits['zero'] += (pred_zero == labels).sum().item()
            hits['rep_vs_new'] += (pred_rep == labels_rep).sum().item()
            hits['rep_vs_old'] += (pred_rep == labels).sum().item()
            hits['rep_same_pred'] += (pred_rep == pred).sum().item()
            hits['in_context'] += in_ctx.sum().item()

    pct = {k: 100.0 * v / n for k, v in hits.items()}
    pairs = values.size(1)
    print(f'answers evaluated                               {n}')
    print(f'copy baseline, 1 / pairs                        {100.0 / pairs:.2f}%')
    print(f'accuracy, normal                                {pct["normal"]:.2f}%')
    print(f'accuracy, memory swapped with another example   {pct["swap"]:.2f}%')
    print(f'accuracy, memory set to zero                    {pct["zero"]:.2f}%')
    print(f'prediction is one of its own context values     {pct["in_context"]:.2f}%')
    print(f're-paired context, accuracy vs NEW answers      {pct["rep_vs_new"]:.2f}%')
    print(f're-paired context, accuracy vs OLD answers      {pct["rep_vs_old"]:.2f}%')
    print(f're-paired context, same prediction as before    {pct["rep_same_pred"]:.2f}%')
    print('Reading.')
    print(' - in-context near 100% and normal near the copy baseline: the copy shortcut.')
    print(' - swap far below normal: the memory carries this example\'s values.')
    print(' - re-paired vs NEW about equal to normal: the model binds keys to values.')
    print(' - re-paired vs NEW at the copy baseline: no binding, keys are ignored.')


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='model.safetensors of an RMT run')
    ap.add_argument('--pairs', type=int, default=None)
    ap.add_argument('--n_layer', type=int, default=None)
    ap.add_argument('--n_head', type=int, default=None)
    ap.add_argument('--n_embd', type=int, default=None)
    ap.add_argument('--n_mem_tokens', type=int, default=None)
    ap.add_argument('--K', type=int, default=None)
    ap.add_argument('--vocab_size', type=int, default=None)
    ap.add_argument('--mem_proj_mode', default=None, choices=[None, 'none', 'proj', 'proj_rw'])
    ap.add_argument('--data_seed', type=int, default=None)
    ap.add_argument('--n_eval', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    run_args, source = find_run_args(a.checkpoint) if a.checkpoint else ({}, None)
    if source:
        print(f'run settings read from {source}')
    defaults = {'pairs': run_args.get('num_kv_pairs', 8), 'n_layer': run_args.get('n_layer', 2),
                'n_head': run_args.get('n_head', 4), 'n_embd': run_args.get('n_embd', 128),
                'n_mem_tokens': run_args.get('n_mem_tokens', 8), 'K': run_args.get('K', 1),
                'vocab_size': run_args.get('vocab_size', 8192), 'data_seed': run_args.get('data_seed', 123),
                'mem_proj_mode': run_args.get('mem_proj_mode', 'none')}
    for key, value in defaults.items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    print(f'model: {a.n_layer} layers, hidden {a.n_embd}, {a.n_mem_tokens} memory tokens, K={a.K}, '
          f'{a.pairs} pairs, device {a.device}')

    torch.manual_seed(a.seed)
    generator = torch.Generator().manual_seed(a.seed)
    model = build_model(a, a.device)

    first_inputs, first_labels = next(eval_batches(a, a.device))
    tier0(model, first_inputs, first_labels, generator)
    tier1(model, eval_batches(a, a.device), generator)
    if not a.checkpoint:
        print('\nNo checkpoint given, so Tier 1 ran on an untrained model. Its numbers only show that the code runs.')


if __name__ == '__main__':
    main()
