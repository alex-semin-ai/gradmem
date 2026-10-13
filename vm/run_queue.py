"""Train a queue of MQAR runs on this machine, one run per GPU, with resume, logs and results.

Usage, from the repo root with the virtual environment active, inside tmux
    python vm/run_queue.py vm/queue.json             train the queue
    python vm/run_queue.py vm/queue.json --dry_run   build and check every configuration, train nothing

Everything is kept in the project folder (`proj` in the queue file, default ~/gradmem_idea3)
    runs/<RUN>/      config.json, summary.json, checkpoints/, files/ (TensorBoard, metrics.jsonl), logs/
    results/         results.csv and runs.md for all runs
    work/<RUN>/      the training folder the repo script writes to (copied into runs/<RUN> every few minutes)
    logs/            console log of every session of every run, and the log of this queue

The same rules as the Colab and Kaggle notebooks. GradMem starts only from an RMT parent trained with the
reconstruction loss (D1, D10, D11), the starting weights must load completely, one run name has one
configuration, finished and diverged runs are skipped unless FORCE is true, and a stopped run resumes from its newest checkpoint.
Ctrl+C (or SIGTERM) stops every run cleanly, and each run saves its newest checkpoint first.
"""
import argparse, glob, json, os, re, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_helper as H  # noqa: E402

# ---------- the shared base (plan, D11 to D14), the same as in the notebooks ----------
BASE = dict(
    vocab_size=8192, num_kv_pairs=16, input_seq_len=48,
    query_sampling='uniform', power_a=0.01,
    train_num_examples=1000000, valid_num_examples=3000, data_seed=123,
    base_model='llama', n_layer=4, n_head=4, n_embd=128, attn_implementation='eager',
    n_mem_tokens=8,
    per_device_batch_size=64, gradient_accumulation_steps=1, total_batch_size=64,
    learning_rate=1e-4, max_steps=60000, warmup_steps=1000,   # one warmup for every run (fairness fix 4)
    eval_steps=500, logging_steps=500,
    early_stopping_patience=500, stop_on_metric_value=0.99,
    stop_on_metric_consecutive=3,   # 99 % exact match in 3 evals in a row ends a run (fairness fix 2)
    seed=143,
)
SIZE_KEYS = ['vocab_size', 'num_kv_pairs', 'input_seq_len', 'n_layer', 'n_head', 'n_embd',
             'n_mem_tokens', 'train_num_examples', 'data_seed']
MODEL_KEYS = ['vocab_size', 'n_layer', 'n_head', 'n_embd', 'n_mem_tokens', 'num_kv_pairs']


def say(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf8') as f:
            f.write(msg + '\n')


LOG_FILE = None


def git(*args):
    r = subprocess.run(['git', '-C', str(REPO)] + list(args), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'unknown'


def fix_paths(proj):
    """Runs started on Colab or Kaggle keep their starting checkpoint under .../runs/<parent>/...
    Point it to the same parent in this project folder."""
    for cfg_path in Path(proj, 'runs').glob('*/config.json'):
        rec = json.load(open(cfg_path))
        init = (rec.get('config') or {}).get('init_checkpoint')
        if not init or '/runs/' not in init.replace('\\', '/'):
            continue
        new = f'{proj}/runs/' + init.replace('\\', '/').split('/runs/', 1)[1]
        if new != init:
            rec['config']['init_checkpoint'] = new
            json.dump(rec, open(cfg_path, 'w'), indent=2)
            say(f'{cfg_path.parent.name}: starting checkpoint path set to {new}')


def parent_checkpoint(proj, parent, step, cfg, allow_size_change):
    pdir = f'{proj}/runs/{parent}'
    if not os.path.exists(f'{pdir}/config.json'):
        raise ValueError(f'Parent {parent} not found ({pdir}). Copy its run folder into {proj}/runs first.')
    rec = json.load(open(f'{pdir}/config.json'))
    if rec.get('script') != 'run_rmt_on_mqar.py':
        raise ValueError(f'Parent {parent} is not an RMT run (script {rec.get("script")}).')
    pcfg = rec['config']
    if not pcfg.get('use_reconstruction_loss'):
        raise ValueError(f'Parent {parent} was trained without the reconstruction loss (plan, D11).')
    diff = {k: (pcfg.get(k), cfg.get(k)) for k in MODEL_KEYS if pcfg.get(k) != cfg.get(k)}
    if diff and not allow_size_change:
        raise ValueError(f'Parent {parent} has a different model size or pair count: {diff}')
    cks = {int(m.group(1)): p for p in glob.glob(f'{pdir}/checkpoints/checkpoint-*/model.safetensors')
           for m in [re.search(r'checkpoint-(\d+)', p)] if m}
    if not cks:
        raise ValueError(f'Parent {parent} has no checkpoint.')
    chosen = step
    if chosen is None:   # the best checkpoint, as recorded by the Trainer in the newest checkpoint
        chosen = max(cks)
        try:
            state = json.load(open(f'{pdir}/checkpoints/checkpoint-{chosen}/trainer_state.json'))
            m = re.search(r'checkpoint-(\d+)', state.get('best_model_checkpoint') or '')
            if m and int(m.group(1)) in cks:
                chosen = int(m.group(1))
        except (OSError, ValueError):
            pass
    if chosen not in cks:
        raise ValueError(f'Parent {parent} has no checkpoint-{chosen}. Available {sorted(cks)}')
    summary = f'{pdir}/summary.json'
    status = json.load(open(summary)).get('status') if os.path.exists(summary) else 'unknown'
    if status not in ('finished', 'stopped'):
        say(f'Note. Parent {parent} has status "{status}". Its best checkpoint may still change.')
    say(f'  parent {parent}, checkpoint-{chosen}')
    return cks[chosen].replace('\\', '/')


def to_args(cfg):
    out = []
    for k, v in cfg.items():
        if v is None:
            continue
        out += [f'--{k}', ('true' if v else 'false') if isinstance(v, bool) else str(v)]
    return out


def build(spec, q, proj, code):
    run, run_type, k = spec['RUN'], spec['RUN_TYPE'], spec['K']
    parent, extra = spec.get('PARENT_RMT_RUN'), dict(spec.get('EXTRA') or {})
    allow = q.get('allow_size_change', False)
    say(run)
    if run_type == 'rmt':
        script = 'run_rmt_on_mqar.py'
        cfg = {**BASE, **dict(K=k, use_reconstruction_loss=True, reconstruction_loss_weight=1.0)}
    elif run_type in ('gradmem', 'gradmem_reference'):
        script = 'run_gradmemgpt_on_mqar.py'
        cfg = {**BASE, **dict(memory_backend='prefix', K=k, last_K_second_order=k, inner_lr=0.4,
                              use_adam=False, grad_mode='second', freeze_backbone=False)}
    elif run_type == 'combined':
        script = 'run_enc_gradmem_on_mqar.py'
        cfg = {**BASE, **dict(memory_backend='prefix', K=k, last_K_second_order=k, inner_lr=0.4, use_adam=False, grad_mode='second', freeze_backbone=False, use_projection=False, encoder_reconstruction_loss=False, encoder_reconstruction_loss_weight=1.0, freeze_encoder=False)}
    else:
        raise ValueError(f"{run}: RUN_TYPE must be 'rmt', 'gradmem', 'gradmem_reference' or 'combined'")

    bad = [key for key in extra if key in SIZE_KEYS and extra[key] != cfg.get(key)]
    if bad and not allow:
        raise ValueError(f'{run}: EXTRA changes the model size or data {bad}. Set allow_size_change only on purpose.')
    if 'init_base_checkpoint' in extra:
        raise ValueError(f'{run}: init_base_checkpoint loads nothing for RMT checkpoints. Use PARENT_RMT_RUN.')
    if run_type == 'rmt' and extra.get('use_reconstruction_loss') is False:
        raise ValueError(f'{run}: RMT without the reconstruction loss stays at the copy baseline (plan, D11).')
    cfg.update(extra)

    run_dir = f'{proj}/runs/{run}'
    saved = f'{run_dir}/config.json'
    old = json.load(open(saved)) if os.path.exists(saved) else None
    summary_path = f'{run_dir}/summary.json'
    status = json.load(open(summary_path)).get('status') if os.path.exists(summary_path) else None
    if status == 'finished' and not spec.get('FORCE'):
        say('  finished already, skipped (set FORCE to true to train it further)')
        return None
    if status and status.startswith('diverged') and not spec.get('FORCE'):
        say(f'  {status}. Skipped. Fix the setting and use a new RUN name, or set FORCE to true to retry')
        return None

    if run_type in ('gradmem', 'combined') and not parent:
        raise ValueError(f"{run}: a '{run_type}' run must start from RMT weights. Set PARENT_RMT_RUN (plan, D1, D10).")
    if run_type == 'gradmem_reference' and parent:
        raise ValueError(f"{run}: 'gradmem_reference' runs train from scratch. Use RUN_TYPE 'gradmem' to start from RMT.")
    if parent:
        if old is not None and 'init_checkpoint' in old['config']:
            cfg['init_checkpoint'] = old['config']['init_checkpoint']   # a resumed run keeps its original start
            say('  resumed run, keeping its original start', cfg['init_checkpoint'])
        else:
            cfg['init_checkpoint'] = parent_checkpoint(proj, parent, spec.get('PARENT_STEP'), cfg, allow)
    elif run_type == 'gradmem_reference':
        say('  reference run from scratch (E2 or E3)')

    os.makedirs(run_dir, exist_ok=True)
    record = json.loads(json.dumps({'script': script, 'config': cfg}))
    if old is not None:
        changed = {key: (old['config'][key], v) for key, v in record['config'].items()
                   if key in old['config'] and old['config'][key] != v}
        if old.get('script') != record['script'] or changed:
            raise ValueError(f'{run} already exists with a different configuration {changed}. '
                             'Use a new RUN name, or restore the old configuration.')
        added = sorted(set(record['config']) - set(old['config']))
        if added:
            say('  settings added to the base after this run started are not applied to it:', added)
        cfg = dict(old['config'])   # a resumed run keeps its original configuration
        say('  same configuration as before, the run will resume')
    else:
        json.dump(record, open(saved, 'w'), indent=2)
        say('  new run, configuration saved')
    with open(f'{run_dir}/code_versions.txt', 'a') as f:
        f.write(code + '\n')
    work = f'{proj}/work/{run}'
    return dict(spec, SCRIPT=script, ARGS=['--exp_path', work] + to_args(cfg), WORK=work, RUN_DIR=run_dir)


def write_conclusions(q, proj):
    for run, text in (q.get('conclusions') or {}).items():
        p = Path(proj) / 'runs' / run / 'summary.json'
        if not text.strip():
            continue
        if not p.parent.exists():
            say('Conclusion skipped, no run named', run)
            continue
        summary = json.load(open(p)) if p.exists() else {'run': run, 'sessions': []}
        if summary.get('conclusion') == text:
            continue
        summary['conclusion'] = text
        H.write_json_atomic(p, summary)
        H.append_md(f'{proj}/results/runs.md', f'### {time.strftime("%Y-%m-%d %H:%M")} · {run} · conclusion\n- {text}')
        say('Conclusion saved for', run)


def last_eval(log):
    """The newest '[eval] ...' line of a run's console log."""
    try:
        with open(log, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 300_000))
            text = f.read().decode('utf8', 'replace').replace('\r', '\n')
    except OSError:
        return 'no log yet'
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    evals = [l for l in lines if l.startswith('[eval]')]
    return evals[-1][7:] if evals else 'no eval yet. ' + (lines[-1][-160:] if lines else '')


def detect_gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout
        return [int(x) for x in out.split()]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return []


def main():
    global LOG_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument('queue')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    q = json.load(open(a.queue))
    proj = os.path.expanduser(q.get('proj', '~/gradmem_idea3')).replace('\\', '/')
    for d in ('runs', 'results', 'work', 'logs'):
        os.makedirs(f'{proj}/{d}', exist_ok=True)
    LOG_FILE = f'{proj}/logs/queue_{time.strftime("%Y%m%d-%H%M%S")}.log'
    code = f'{git("rev-parse", "--abbrev-ref", "HEAD")} {git("rev-parse", "--short", "HEAD")}'
    gpus = q.get('gpus') or detect_gpus()
    if not gpus:
        raise SystemExit('No GPU found. Set "gpus" in the queue file.')
    say(f'Code {code}. Project {proj}. GPUs {gpus}.')

    names = [s['RUN'] for s in q['runs']]
    if len(set(names)) != len(names):
        raise SystemExit('Two runs have the same name. They would train into the same folder.')
    fix_paths(proj)
    planned = [r for r in (build(s, q, proj, code) for s in q['runs']) if r is not None]
    write_conclusions(q, proj)
    say('Queue:', [r['RUN'] for r in planned] or 'empty')
    if a.dry_run:
        say('Dry run. Every configuration was built and checked, nothing was trained.')
        return

    limit = q.get('time_limit_hours')
    deadline = time.time() + limit * 3600 if limit else None
    min_left = q.get('min_hours_to_start', 0.5)
    workers = str(q.get('workers_per_run', 2))
    queue, active, ended = list(planned), {}, []
    stop_requested = {'value': False}

    def request_stop(*_):
        stop_requested['value'] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def launch(r, gpu):
        log = f'{proj}/logs/{r["RUN"]}_{time.strftime("%Y%m%d-%H%M%S")}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),   # one GPU per run, so the batch never doubles
                   ACCELERATE_MIXED_PRECISION='no',             # full precision, the same as on Colab and Kaggle
                   GRADMEM_NUM_WORKERS=workers, USE_TF='0', USE_TORCH='1')
        cmd = [sys.executable, '-u', str(HERE / 'run_helper.py'), '--repo', str(REPO), '--script', r['SCRIPT'],
               '--local_dir', r['WORK'], '--drive_dir', r['RUN_DIR'], '--log', log,
               '--sync_minutes', str(q.get('sync_minutes', 5)),
               '--note', r.get('NOTE', ''), '--commit', code, '--run_type', r['RUN_TYPE'],
               '--runs_md', f'{proj}/results/runs.md', '--results_csv', f'{proj}/results/results.csv',
               '--'] + r['ARGS']
        kwargs = dict(start_new_session=True) if os.name == 'posix' else {}
        proc = subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT, env=env, **kwargs)
        say(f'{time.strftime("%H:%M")} started {r["RUN"]} on GPU {gpu}, pid {proc.pid}, log {log}')
        return (r['RUN'], proc, log)

    def stop_all(reason):
        say(f'{reason} Stopping the runs. Each run saves its newest checkpoint first.')
        for name, p, _ in active.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        end = time.time() + 600
        while any(p.poll() is None for _, p, _ in active.values()) and time.time() < end:
            time.sleep(5)
        for name, p, _ in active.values():
            if p.poll() is None:
                say(name, 'did not stop in 10 minutes, killed')
                p.kill()
            ended.append((name, p.returncode, _))

    next_report = 0
    while queue or active:
        for gpu, (name, p, log) in list(active.items()):
            if p.poll() is not None:
                say(f'{time.strftime("%H:%M")} {name} ended, exit {p.returncode}. {last_eval(log)}')
                ended.append((name, p.returncode, log))
                del active[gpu]
        if stop_requested['value']:
            stop_all('Stop requested (Ctrl+C or SIGTERM).')
            active.clear()
            break
        if deadline and time.time() > deadline:
            stop_all('Time limit reached.')
            active.clear()
            break
        left_h = (deadline - time.time()) / 3600 if deadline else float('inf')
        for gpu in gpus:
            if gpu not in active and queue:
                if left_h < min_left:
                    say(f'{left_h:.1f} h left, not starting {[r["RUN"] for r in queue]}')
                    queue = []
                    break
                active[gpu] = launch(queue.pop(0), gpu)
        if time.time() >= next_report and active:
            say(f'--- {time.strftime("%Y-%m-%d %H:%M")}')
            for gpu, (name, p, log) in sorted(active.items()):
                say(f'GPU {gpu} {name} | {last_eval(log)}')
            next_report = time.time() + 10 * 60
        if queue or active:
            time.sleep(5)

    H.rebuild_results(f'{proj}/runs', f'{proj}/results/results.csv')
    say('All done.' if not stop_requested['value'] else 'Stopped.', 'Results in', f'{proj}/results/results.csv')
    for name, code_, log in ended:
        say(f'{name}: exit {code_}')


if __name__ == '__main__':
    main()
