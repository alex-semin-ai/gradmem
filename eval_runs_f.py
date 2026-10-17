"""Evaluate finished MQAR runs on a held-out test split, and benchmark their cost on one device.

Usage
    python eval_runs.py ~/gradmem_idea3                       all runs in the project folder
    python eval_runs.py ~/gradmem_idea3 --runs E2r_gm_L4_p16_1M_s1 E6_comb_K1_L4_p16_1M_s1
    python eval_runs.py ~/gradmem_idea3 --no_benchmark          test metrics only
    python eval_runs.py ~/gradmem_idea3 --k_eval 0 1 2 3 --shuffle_start --positions
        also evaluate with 0..3 write steps, with the encoder memory shuffled across examples (C3, combined runs),
        and split accuracy by the position of the pair in the context (pairs 1 to 5 against 6 to 16)

Writes results/test_results.json and results/test_results.csv. make_report.py shows them when they exist.

Why (fairness fixes 1, 5, 6, 9)
  - The validation set picks the best checkpoint and triggers the stop rule, so its numbers are optimistic.
    Here every run's selected checkpoint is evaluated once on a test split built from another seed
    (--test_seed, default 456), which no run has seen.
  - Cost is measured for all methods on the same device, batch and conditions. RMT is timed without its
    reconstruction pass at inference (it does not change the predictions, the read uses the memory of the K-th
    write), training steps are timed as trained (forward and backward, no optimizer step).


Copy of eval_runs.py that also evaluates runs of run_enc_gradmem_n_on_mqar.py (N1, N3). It handles every
other run type exactly like eval_runs.py, so it can replace it for all runs.

Copy of eval_runs_n.py that also evaluates runs of run_gradmem_fused_on_mqar.py (the fused write). It handles
every other run type exactly like eval_runs_n.py, so it can replace it for all runs.
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import HfArgumentParser

import run_enc_gradmem_on_mqar as C
import run_enc_gradmem_n_on_mqar as CN
import run_gradmem_fused_on_mqar as FU
import run_gradmemgpt_on_mqar as G
import run_rmt_on_mqar as R
from rmt import RMT2SegmConfig
from zoology_mqar_data import build_mqar_datasets

COMBINED = ('run_enc_gradmem_on_mqar.py', 'run_enc_gradmem_n_on_mqar.py')
TESTED = ['run_rmt_on_mqar.py', 'run_gradmemgpt_on_mqar.py', 'run_gradmem_fused_on_mqar.py', *COMBINED]


def to_args(cfg):
    out = []
    for k, v in cfg.items():
        if v is None:
            continue
        out += [f'--{k}', ('true' if v else 'false') if isinstance(v, bool) else str(v)]
    return out


def parse(dataclass, cfg):
    known = set(dataclass.__dataclass_fields__)
    args = ['--exp_path', 'unused'] + to_args({k: v for k, v in cfg.items() if k in known})
    return HfArgumentParser(dataclass).parse_args_into_dataclasses(args)[0]


def build(script, cfg):
    """An untrained model of the run's type, built exactly as its training script builds it."""
    if script == 'run_rmt_on_mqar.py':
        a = parse(R.ExperimentArgs, cfg)
        rmt_config = RMT2SegmConfig(
            pretrained_model=None, base_config=C.base_llama_config(a), n_mem_tokens=a.n_mem_tokens, K=a.K,
            n_ctrl_tokens=a.n_ctrl_tokens, use_mem_proj=a.use_mem_proj, mem_proj_mode=a.mem_proj_mode,
            use_reconstruction_loss=a.use_reconstruction_loss, reconstruction_loss_weight=a.reconstruction_loss_weight,
            use_write_head=a.use_write_head, use_mem_residual=a.use_mem_residual,
            use_gradient_checkpointing=False, attn_implementation=a.attn_implementation,
            read_loss_alignment='query_position', mqar_vocab_size=a.vocab_size, mqar_dense_queries=a.dense_queries)
        return R.MQARRMT(rmt_config), a
    if script == 'run_gradmemgpt_on_mqar.py':
        a = parse(G.ExperimentArgs, cfg)
        return G.MQARGradMemGPT(C.gradmem_config_from_args(a, C.base_llama_config(a))), a
    if script == 'run_gradmem_fused_on_mqar.py':
        a = parse(FU.ExperimentArgs, cfg)
        config = C.gradmem_config_from_args(a, C.base_llama_config(a))
        config.fused_mode, config.fused_norm = a.fused_mode, a.fused_norm
        return FU.MQARGradMemGPT(config), a
    if script == 'run_enc_gradmem_on_mqar.py':
        a = parse(C.CombinedArgs, cfg)
        return C.build_model(a, load_parent_weights=False), a
    if script == 'run_enc_gradmem_n_on_mqar.py':
        a = parse(CN.CombinedArgs, cfg)
        return CN.build_model(a, load_parent_weights=False), a
    raise ValueError(f'unknown script {script}')


def selected_checkpoint(run_dir):
    """The checkpoint the Trainer selected (best by token accuracy), else the newest one."""
    cks = {int(m.group(1)): p for p in (run_dir / 'checkpoints').glob('checkpoint-*')
           for m in [re.fullmatch(r'checkpoint-(\d+)', p.name)] if m and p.is_dir()}
    if not cks:
        return None
    newest = max(cks)
    try:
        state = json.load(open(cks[newest] / 'trainer_state.json'))
        m = re.search(r'checkpoint-(\d+)', state.get('best_model_checkpoint') or '')
        if m and int(m.group(1)) in cks:
            return cks[int(m.group(1))]
    except (OSError, ValueError):
        pass
    return cks[newest]


def load_weights(model, ckpt):
    if (ckpt / 'model.safetensors').exists():
        sd = load_file(str(ckpt / 'model.safetensors'))
    else:
        sd = torch.load(ckpt / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('lm_head.weight')   # tied to the embeddings
               and k != 'start_norm_t']                                  # older combined checkpoints, raw start
    if missing or unexpected:
        raise RuntimeError(f'{ckpt}: {len(missing)} missing and {len(unexpected)} unexpected weights, '
                           f'for example {(missing + unexpected)[:5]}')


def inference_mode(model, script):
    """Losses at evaluation are the read (target) loss only, for every method."""
    if script == 'run_rmt_on_mqar.py':
        model.use_reconstruction_loss = False   # the read uses the memory of the K-th write either way
    if script in COMBINED:
        model.encoder_reconstruction_loss = False


def test_batches(a, test_seed, n_test, batch, device):
    _, test, _, test_data_seed = build_mqar_datasets(
        vocab_size=a.vocab_size, input_seq_len=a.input_seq_len, num_kv_pairs=a.num_kv_pairs,
        train_num_examples=8, valid_num_examples=n_test, power_a=a.power_a,
        random_non_queries=a.random_non_queries, data_seed=test_seed, dense_queries=a.dense_queries,
        query_sampling=a.query_sampling)
    ds = G.GradMemMQARDataset(test, context_size=2 * a.num_kv_pairs)
    out = []
    for s in range(0, len(ds), batch):
        b = G.collate_fn([ds[i] for i in range(s, min(s + batch, len(ds)))])
        out.append(({k: v.to(device) for k, v in b['input_ids'].items()}, b['labels'].to(device)))
    return out, test_data_seed


def query_logits(out, labels):
    logits = out['predictions']
    return logits[:, 1:] if logits.shape[1] == labels.shape[1] + 1 else logits


def evaluate(model, batches, positions=False, suffix=''):
    model.eval()
    correct = total = 0
    em, loss_sum = [], 0.0
    pos_hits = pos_total = None
    with torch.no_grad():
        for inputs, labels in batches:
            logits = query_logits(model(inputs), labels).float()
            mask = labels != -100
            pred = logits.argmax(-1)
            ok = (pred == labels) & mask
            correct += ok.sum().item()
            total += mask.sum().item()
            em += (ok.sum(1) == mask.sum(1)).tolist()
            loss_sum += torch.nn.functional.cross_entropy(logits[mask], labels[mask], reduction='sum').item()
            if positions:
                keys = inputs['context_input_ids'][:, 0::2]          # key of pair j sits at context position 2j
                q = inputs['query_input_ids']
                where = (q.unsqueeze(-1) == keys.unsqueeze(1)).float().argmax(-1)   # pair index of every query
                if pos_hits is None:
                    pos_hits, pos_total = torch.zeros(keys.size(1)), torch.zeros(keys.size(1))
                pos_hits.index_add_(0, where[mask].cpu(), ok[mask].float().cpu())
                pos_total.index_add_(0, where[mask].cpu(), torch.ones(int(mask.sum())))
    out = {f'test_token_accuracy{suffix}': correct / total, f'test_exact_match{suffix}': float(np.mean(em)),
           f'test_target_loss{suffix}': loss_sum / total}
    if positions:
        acc = (pos_hits / pos_total.clamp_min(1)).tolist()
        out[f'test_acc_by_pair{suffix}'] = ' '.join(f'{a:.3f}' for a in acc)
        out[f'test_acc_pairs_1_5{suffix}'] = float(np.mean(acc[:5]))
        out[f'test_acc_pairs_6_end{suffix}'] = float(np.mean(acc[5:]))
    return out


def set_write_steps(model, script, k):
    target = model.gradmem if script in COMBINED else model
    old = target.K
    target.K = k
    return old


def timed(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / reps


def benchmark(model, script, batches, device, reps):
    inputs, labels = batches[0]

    def infer():
        model.eval()
        with torch.no_grad():
            model(inputs)

    def train_step():
        model.train()
        model.zero_grad(set_to_none=True)
        model(inputs, labels=labels)['loss'].backward()

    infer_ms = timed(infer, device, 3, reps)
    if script == 'run_rmt_on_mqar.py':     # training cost as trained, with the reconstruction pass
        model.use_reconstruction_loss = model.config.use_reconstruction_loss
    if script in COMBINED:
        model.encoder_reconstruction_loss = TRAINED_ENC_REC[0]
    train_ms = timed(train_step, device, 2, max(3, reps // 2))
    model.zero_grad(set_to_none=True)
    return dict(infer_ms_per_batch=infer_ms, train_ms_per_step=train_ms)


TRAINED_ENC_REC = [False]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('proj')
    ap.add_argument('--runs', nargs='*')
    ap.add_argument('--test_seed', type=int, default=456)
    ap.add_argument('--n_test', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--no_benchmark', action='store_true')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--k_eval', type=int, nargs='*', default=[],
                    help='also evaluate GradMem and combined runs with these numbers of write steps')
    ap.add_argument('--shuffle_start', action='store_true',
                    help='C3. Also evaluate combined runs with the encoder memory of another example')
    ap.add_argument('--positions', action='store_true', help='accuracy by the position of the pair in the context')
    a = ap.parse_args()
    proj = Path(a.proj).expanduser()
    device = torch.device(a.device)
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    out_path = proj / 'results' / 'test_results.json'
    results = json.load(open(out_path)) if out_path.exists() else {}

    run_dirs = sorted(p for p in (proj / 'runs').iterdir() if (p / 'config.json').exists())
    if a.runs:
        run_dirs = [p for p in run_dirs if p.name in a.runs]
    for d in run_dirs:
        rec = json.load(open(d / 'config.json'))
        script, cfg = rec.get('script'), rec.get('config', {})
        if script not in TESTED:
            continue
        ckpt = selected_checkpoint(d)
        if ckpt is None:
            print(f'{d.name}: no checkpoint, skipped')
            continue
        torch.manual_seed(0)
        model, args = build(script, cfg)
        load_weights(model, ckpt)
        TRAINED_ENC_REC[0] = bool(getattr(model, 'encoder_reconstruction_loss', False))
        inference_mode(model, script)
        model.to(device)
        batches, test_data_seed = test_batches(args, a.test_seed, a.n_test, a.batch, device)
        if test_data_seed in (cfg.get('data_seed'),):
            raise ValueError('the test seed must differ from the training data seed')
        row = dict(run=d.name, script=script, checkpoint=ckpt.name, test_seed=a.test_seed,
                   test_data_seed=test_data_seed, n_test=a.n_test,
                   params=sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()),
                   **evaluate(model, batches, positions=a.positions))
        if script != 'run_rmt_on_mqar.py':
            for k in a.k_eval:
                old = set_write_steps(model, script, k)
                row.update(evaluate(model, batches, positions=a.positions, suffix=f'_K{k}'))
                set_write_steps(model, script, old)
        if a.shuffle_start and script in COMBINED:
            model.shuffle_start = True
            row.update(evaluate(model, batches, positions=a.positions, suffix='_shuffled'))
            model.shuffle_start = False
        if not a.no_benchmark:
            row.update(benchmark(model, script, batches, device, a.reps), device=device_name, batch=a.batch)
            inference_mode(model, script)
        results[d.name] = row
        print(f'{d.name} ({ckpt.name}): token acc {row["test_token_accuracy"]:.4f}, '
              f'exact {row["test_exact_match"]:.4f}, target loss {row["test_target_loss"]:.4f}'
              + (f', infer {row["infer_ms_per_batch"]:.1f} ms, train step {row["train_ms_per_step"]:.1f} ms '
                 f'on {device_name}' if 'infer_ms_per_batch' in row else ''), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, 'w'), indent=2)
    cols = sorted({k for r in results.values() for k in r}, key=lambda k: (k != 'run', k))
    with open(out_path.with_suffix('.csv'), 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results.values())
    print('Written', out_path)


if __name__ == '__main__':
    sys.exit(main())
