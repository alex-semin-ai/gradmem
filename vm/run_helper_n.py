"""Run a repo training script with resume and mirroring. Generated from the Colab notebook helper, keep them equal.

Usage
    python colab_run.py --repo R --script S --local_dir L --drive_dir D --log F --sync_minutes M -- <script args>
The script args must contain --exp_path equal to --local_dir.
"""
import argparse, csv, json, math, os, re, runpy, shutil, signal, sys, threading, time
from pathlib import Path

REQUIRED = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
MODEL_FILES = ('model.safetensors', 'pytorch_model.bin')


def step_of(p):
    m = re.fullmatch(r'checkpoint-(\d+)', p.name)
    return int(m.group(1)) if m else -1


def is_complete(p):
    ok = all((p / f).is_file() and (p / f).stat().st_size > 0 for f in REQUIRED)
    return ok and any((p / f).is_file() for f in MODEL_FILES)


def is_stable(p, seconds=3):
    """True when the Trainer has finished writing the checkpoint.

    The Trainer writes trainer_state.json last (after weights, optimizer, scheduler and RNG state),
    so a readable trainer_state.json plus a few quiet seconds means the folder is complete."""
    try:
        json.load(open(p / 'trainer_state.json'))
    except (OSError, ValueError):
        return False
    newest = max((f.stat().st_mtime for f in p.rglob('*') if f.is_file()), default=0)
    return time.time() - newest > seconds


def complete_ckpts(d, stable=False):
    if not d.is_dir():
        return []
    found = [p for p in d.iterdir() if p.is_dir() and step_of(p) >= 0 and is_complete(p)]
    if stable:
        found = [p for p in found if is_stable(p)]
    return sorted(found, key=step_of)


def copy_dir_atomic(src, dst):
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)


def copy_if_newer(src, dst):
    if dst.exists():
        s, d = src.stat(), dst.stat()
        if s.st_size == d.st_size and s.st_mtime <= d.st_mtime:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def restore(local, drive):
    """Copy checkpoints and files from Drive when the local run folder has no checkpoint."""
    if complete_ckpts(local) or not drive.is_dir():
        return
    for c in complete_ckpts(drive / 'checkpoints'):
        copy_dir_atomic(c, local / c.name)
    files = drive / 'files'
    if files.is_dir():
        for p in files.rglob('*'):
            if p.is_file():
                copy_if_newer(p, local / p.relative_to(files))
    n = len(complete_ckpts(local))
    if n:
        print(f'[run] restored {n} checkpoint(s) from Drive', flush=True)


def mirror(local, drive, log, final=False):
    """Mirror complete checkpoints first, then all other files of the run, to Drive."""
    (drive / 'checkpoints').mkdir(parents=True, exist_ok=True)
    # 1. checkpoints. During training only checkpoints the Trainer finished writing are copied.
    local_ck = complete_ckpts(local, stable=not final)
    names = {c.name for c in local_ck}
    for c in local_ck:
        if not (drive / 'checkpoints' / c.name).is_dir():
            copy_dir_atomic(c, drive / 'checkpoints' / c.name)
    # remove a Drive checkpoint only when it is gone locally AND a newer one is already on Drive,
    # so Drive always keeps the newest checkpoint it has seen
    if local_ck:
        newest = max(step_of(c) for c in local_ck)
        for c in (drive / 'checkpoints').iterdir():
            if c.is_dir() and 0 <= step_of(c) < newest and c.name not in names:
                shutil.rmtree(c, ignore_errors=True)
    # 2. small files (TensorBoard logs, config, console log), one by one so one failure blocks nothing
    files = [(p, drive / 'files' / p.relative_to(local)) for p in local.rglob('*')
             if p.is_file() and not p.relative_to(local).parts[0].startswith('checkpoint-')]
    if log and Path(log).is_file():
        files.append((Path(log), drive / 'logs' / Path(log).name))
    for src, dst in files:
        try:
            copy_if_newer(src, dst)
        except OSError as e:
            print(f'[run] could not copy {src.name}: {e}', flush=True)


# ---------- run log, summary.json, runs.md and results.csv ----------

def now():
    return time.strftime('%Y-%m-%d %H:%M')


def replace_retry(tmp, path, tries=50):
    # os.replace, retried briefly. On Windows a file another process has open cannot be replaced
    for _ in range(tries - 1):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, path)


def write_json_atomic(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf8')
    replace_retry(tmp, path)


def append_md(path, text):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', encoding='utf8') as f:
        f.write(text.rstrip('\n') + '\n\n')


NOT_METRICS = ('loss', 'runtime', 'per_second', 'epoch')


def eval_summary(log_history, best_metric, metric_name=None):
    """Accuracy key, best accuracy and the first eval step that reached 99%."""
    evals = [h for h in log_history if any(k.startswith('eval_') for k in h)]
    keys = sorted({k for h in evals for k in h
                   if k.startswith('eval_') and not any(x in k for x in NOT_METRICS)})
    key = None
    if metric_name:   # the metric the Trainer uses to pick the best model
        name = metric_name if metric_name.startswith('eval_') else f'eval_{metric_name}'
        key = name if name in keys else None
    for k in keys if key is None else []:   # else the key whose maximum equals the best metric
        vals = [h[k] for h in evals if k in h]
        if best_metric is not None and vals and abs(max(vals) - best_metric) < 1e-9:
            key = k
            break
    if key is None:
        named = [k for k in keys if any(x in k for x in ('acc', 'match', 'exact'))]
        key = named[0] if named else None
    if key is None:
        return None, None, None
    points = [(h.get('step', 0), h[key]) for h in evals if key in h]
    best = max(v for _, v in points)
    threshold = 0.99 if best <= 1.0 else 99.0
    hit = next((s for s, v in points if v >= threshold), None)
    return key, best, hit


def load_config(drive):
    p = drive / 'config.json'
    return json.load(open(p))['config'] if p.exists() else {}


def describe_settings(cfg):
    parts = [f"{cfg.get('num_kv_pairs')} pairs", f"K = {cfg.get('K')}", f"inner_lr {cfg.get('inner_lr')}",
             f"{cfg.get('n_layer')} layers", f"hidden {cfg.get('n_embd')}", f"seed {cfg.get('seed')}"]
    return ', '.join(p for p in parts if 'None' not in p)


def init_from(cfg):
    init = cfg.get('init_checkpoint') or cfg.get('init_base_checkpoint')
    if not init:
        return 'scratch'
    parts = Path(init).parts   # .../runs/<RUN>/<checkpoints/checkpoint-N | base>/model.safetensors
    if 'runs' in parts and parts.index('runs') + 2 < len(parts):
        i = parts.index('runs')
        return '/'.join(p for p in parts[i + 1:-1] if p != 'checkpoints')
    return init


class RunLog:
    """Keeps summary.json of one run up to date and writes its entries in runs.md."""

    def __init__(self, drive, note, commit, runs_md):
        self.drive, self.runs_md = drive, runs_md
        self.path = drive / 'summary.json'
        self.summary = json.load(open(self.path)) if self.path.exists() else {'run': drive.name, 'sessions': []}
        self.session = {'started': now(), 'commit': commit, 'note': note,
                        'start_step': None, 'end_step': None, 'seconds': 0.0}
        self.summary['sessions'].append(self.session)
        self.status, self.trainer, self.t0 = 'running', None, None

    def attach(self, trainer, start_step):
        self.trainer, self.t0 = trainer, time.time()
        self.session['start_step'] = start_step
        try:   # which device this session ran on, so speeds from different hardware are never mixed (fix 9)
            import torch
            self.session['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        except Exception:
            self.session['device'] = 'unknown'

    def update(self):
        s = self.summary
        s['status'], s['last_update'] = self.status, now()
        if self.trainer is not None:
            st = self.trainer.state
            self.session['end_step'] = st.global_step
            self.session['seconds'] = round(time.time() - self.t0, 1)
            key, best, hit = eval_summary(list(st.log_history), st.best_metric,
                                          getattr(self.trainer.args, 'metric_for_best_model', None))
            s.update(steps=st.global_step, metric=key, best=best, step_to_99=hit)
        self.drive.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, s)

    def launch_entry(self, resumed_from):
        cfg = load_config(self.drive)
        start = f'Resumed from {resumed_from}' if resumed_from else 'New run from step 0'
        append_md(self.runs_md, '\n'.join([
            f"### {now()} · {self.drive.name} · launch",
            f"- {start}. Initial weights from {init_from(cfg)}.",
            f"- Settings. {describe_settings(cfg)}.",
            f"- Code {self.session['commit']}.",
            f"- Why. {self.session['note'] or 'no note'}"]))

    def end_entry(self):
        s = self.summary
        best = s.get('best')
        best_txt = f"best accuracy {best:.4f}" if best is not None else 'no eval yet'
        hit = s.get('step_to_99')
        hit_txt = f'99% reached at step {hit}' if hit is not None else '99% not reached'
        done = self.session['end_step'] is not None and self.session['start_step'] is not None
        n = (self.session['end_step'] - self.session['start_step']) if done else 0
        speed = f"{1000 * self.session['seconds'] / n:.0f} ms per step in this session" if n > 0 else 'no steps in this session'
        append_md(self.runs_md, '\n'.join([
            f"### {now()} · {self.drive.name} · {self.status}",
            f"- Step {s.get('steps', 0)}, {best_txt}, {hit_txt}.",
            f"- {speed}.",
            "- Conclusion. Add it with the conclusion cell."]))


RESULT_COLUMNS = ['run', 'status', 'pairs', 'K', 'inner_lr', 'layers', 'hidden', 'seed', 'init_from',
                  'steps', 'best_acc', 'step_to_99', 'ms_per_step', 'devices', 'sessions', 'code', 'last_update',
                  'why', 'conclusion']


def rebuild_results(runs_dir, csv_path):
    """Write results.csv from every run folder. Safe to call from several runs at once."""
    runs_dir, csv_path = Path(runs_dir), Path(csv_path)
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg = load_config(d)
        s = json.load(open(d / 'summary.json')) if (d / 'summary.json').exists() else {}
        ses = s.get('sessions', [])
        done = [x for x in ses if x.get('start_step') is not None and x.get('end_step') is not None
                and x['end_step'] > x['start_step']]
        steps_done = sum(x['end_step'] - x['start_step'] for x in done)
        ms = round(1000 * sum(x['seconds'] for x in done) / steps_done, 1) if steps_done else None
        notes = [x['note'] for x in ses if x.get('note')]
        rows.append({
            'run': d.name, 'status': s.get('status', 'not started'),
            'pairs': cfg.get('num_kv_pairs'), 'K': cfg.get('K'), 'inner_lr': cfg.get('inner_lr'),
            'layers': cfg.get('n_layer'), 'hidden': cfg.get('n_embd'), 'seed': cfg.get('seed'),
            'init_from': init_from(cfg), 'steps': s.get('steps'), 'best_acc': s.get('best'),
            'step_to_99': s.get('step_to_99'), 'ms_per_step': ms,
            'devices': ' + '.join(sorted({x.get('device', 'unknown') for x in ses})) if ses else None,
            'sessions': len(ses),
            'code': ses[-1].get('commit') if ses else None, 'last_update': s.get('last_update'),
            'why': notes[-1] if notes else None, 'conclusion': s.get('conclusion')})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    replace_retry(tmp, csv_path)
    return rows


EVAL_KEYS = [('eval_loss', 'loss'), ('eval_target_loss', 'target'), ('eval_rec_loss', 'rec'),
             ('eval_inner_loss', 'inner'), ('eval_token_accuracy', 'tok_acc'),
             ('eval_all_queries_exact_match', 'exact')]


BAD_EVALS_TO_STOP = 3     # evals in a row with an eval loss above the limit
GUARD = {'reason': None}  # set when the divergence guard stopped the run


def make_metrics_log(callback_base, path, loss_limit):
    """A Trainer callback. Every logged value goes to metrics.jsonl (one JSON line per log event),
    and every evaluation prints one short console line with the main losses and accuracies.

    It is also the divergence guard. Training stops when a training loss is NaN or infinite, or when the
    eval target (read) loss is NaN, infinite or above loss_limit (2 ln(vocab size), far worse than random guessing) for
    BAD_EVALS_TO_STOP evaluations in a row. The run then ends with status 'diverged'."""

    class MetricsLog(callback_base):
        def __init__(self):
            self.last_train, self.mark, self.bad = {}, None, 0

        def stop(self, control, reason):
            GUARD['reason'] = reason
            control.should_training_stop = True
            print(f'\n[guard] {reason}. Training stopped, the run will be marked diverged.', flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            rec = {'step': state.global_step, 'time': round(time.time(), 1), **logs}
            try:
                with open(path, 'a', encoding='utf8') as f:
                    f.write(json.dumps(rec, default=float) + '\n')
            except OSError as e:
                print(f'[run] metrics.jsonl not written: {e}', flush=True)
            if GUARD['reason'] is None and state.global_step > 0:
                bad_train = [k for k, v in logs.items() if k.endswith('loss') and not k.startswith('eval_')
                             and isinstance(v, (int, float)) and not math.isfinite(v)]
                if bad_train:
                    self.stop(control, f'training {bad_train[0]} is {logs[bad_train[0]]} at step {state.global_step}')
                # the read (target) loss, so a reconstruction term never decides divergence (fairness fix 8)
                ev_loss = logs.get('eval_target_loss', logs.get('eval_loss'))
                if isinstance(ev_loss, (int, float)):
                    self.bad = self.bad + 1 if (not math.isfinite(ev_loss) or ev_loss > loss_limit) else 0
                    if self.bad >= BAD_EVALS_TO_STOP:
                        self.stop(control, f'eval target loss above {loss_limit:.1f} for {self.bad} evals in a row '
                                           f'(now {ev_loss:.4g} at step {state.global_step})')
            if not any(k.startswith('eval_') for k in logs):
                self.last_train = logs
                return control
            now, step = time.time(), state.global_step
            speed = ''
            if self.mark and step > self.mark[1]:
                speed = f' | {1000 * (now - self.mark[0]) / (step - self.mark[1]):.0f} ms/step'
            self.mark = (now, step)
            ev = ' '.join(f'{name} {logs[k]:.4f}' for k, name in EVAL_KEYS if k in logs)
            tr = ' '.join(f'{k} {self.last_train[k]:.4f}' for k in ('loss', 'target_loss', 'rec_loss', 'inner_loss')
                          if isinstance(self.last_train.get(k), (int, float)))
            print(chr(10) + f'[eval] step {step} | {ev}' + (f' | train {tr}' if tr else '') + speed, flush=True)
            return control

    return MetricsLog()


GRADMEM_SCRIPT, RMT_SCRIPT = 'run_gradmemgpt_on_mqar.py', 'run_rmt_on_mqar.py'
COMBINED_SCRIPT = 'run_enc_gradmem_on_mqar.py'
COMBINED_N_SCRIPT = 'run_enc_gradmem_n_on_mqar.py'   # combined_n (N1 write-step rule, C4 random start)


def check_run_type(run_type, script, script_args):
    """Sanity check before anything starts. Returns an error message, or None when the run is allowed.

    gradmem            GradMem must start from an RMT checkpoint (plan, D1 and D10).
    gradmem_reference  GradMem from scratch, only for the reference runs E2 and E3.
    rmt                RMT always with the reconstruction loss (plan, D11). A warm start must come from RMT.
    combined           encoder + GradMem (Idea 3). Both parts must start from an RMT checkpoint (plan, D1 to D3).
    combined_n         the same with run_enc_gradmem_n_on_mqar.py (N1 write-step rule, C4 random start).
    The checkpoint's own config.json names the class that saved it, so it shows where the weights come from."""
    if not run_type:
        return None
    a = {script_args[i][2:]: script_args[i + 1] for i in range(len(script_args) - 1)
         if script_args[i].startswith('--')}
    init = a.get('init_checkpoint')

    def rmt_parent_problem():
        cfg_path = Path(init).parent / 'config.json'
        if not Path(init).is_file():
            return f'the starting checkpoint does not exist ({init})'
        if not cfg_path.is_file():
            return f'no config.json next to the starting checkpoint ({cfg_path})'
        c = json.load(open(cfg_path))
        if c.get('model_type') != 'rmt_2segm':
            return f'the starting checkpoint is not an RMT checkpoint (model_type {c.get("model_type")!r})'
        if not c.get('use_reconstruction_loss'):
            return 'the RMT parent was trained without the reconstruction loss (plan, D11)'
        base = c.get('base_config') or {}
        pairs = [('num_hidden_layers', 'n_layer'), ('hidden_size', 'n_embd'), ('vocab_size', 'vocab_size')]
        diff = [f'{arg} {base.get(key)} vs {a.get(arg)}' for key, arg in pairs
                if a.get(arg) is not None and str(base.get(key)) != str(a.get(arg))]
        if a.get('n_mem_tokens') is not None and str(c.get('n_mem_tokens')) != str(a.get('n_mem_tokens')):
            diff.append(f'n_mem_tokens {c.get("n_mem_tokens")} vs {a.get("n_mem_tokens")}')
        if diff:
            return 'the RMT parent has a different model size: ' + ', '.join(diff)
        return None

    if 'init_base_checkpoint' in a:
        return 'init_base_checkpoint is not allowed. It loads nothing from RMT checkpoints'
    if run_type == 'gradmem':
        if Path(script).name != GRADMEM_SCRIPT:
            return f'a gradmem run must use {GRADMEM_SCRIPT}, not {script}'
        if not init:
            return 'a gradmem run must start from RMT weights, but no init_checkpoint is given (plan, D1, D10)'
        return rmt_parent_problem()
    if run_type in ('combined', 'combined_n'):
        expected = COMBINED_SCRIPT if run_type == 'combined' else COMBINED_N_SCRIPT
        if Path(script).name != expected:
            return f'a {run_type} run must use {expected}, not {script}'
        if not init:
            return 'a combined run must start from RMT weights, but no init_checkpoint is given (plan, D1)'
        return rmt_parent_problem()
    if run_type == 'gradmem_reference':
        if Path(script).name != GRADMEM_SCRIPT:
            return f'a gradmem_reference run must use {GRADMEM_SCRIPT}, not {script}'
        if init:
            return 'a gradmem_reference run trains from scratch, but init_checkpoint is given'
        return None
    if run_type == 'rmt':
        if Path(script).name != RMT_SCRIPT:
            return f'an rmt run must use {RMT_SCRIPT}, not {script}'
        if a.get('use_reconstruction_loss') != 'true':
            return 'an rmt run must use the reconstruction loss (plan, D11)'
        return rmt_parent_problem() if init else None
    return f'unknown run type {run_type!r}'


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--script', required=True)
    ap.add_argument('--local_dir', required=True)
    ap.add_argument('--drive_dir', required=True)
    ap.add_argument('--log', default='')
    ap.add_argument('--sync_minutes', type=float, default=10)
    ap.add_argument('--note', default='')
    ap.add_argument('--commit', default='')
    ap.add_argument('--runs_md', default='')
    ap.add_argument('--results_csv', default='')
    ap.add_argument('--run_type', default='', help='rmt, gradmem, gradmem_reference or combined. Empty skips the check')
    a = ap.parse_args(argv[:split])
    script_args = argv[split + 1:]

    problem = check_run_type(a.run_type, a.script, script_args)
    if problem:
        print(f'[run] REFUSED, {problem}. Nothing was started.', flush=True)
        sys.exit(2)
    if a.run_type:
        print(f'[run] run type {a.run_type} checked', flush=True)

    local, drive = Path(a.local_dir), Path(a.drive_dir)
    local.mkdir(parents=True, exist_ok=True)
    restore(local, drive)

    runlog = RunLog(drive, a.note, a.commit, a.runs_md)
    existing = complete_ckpts(local)
    runlog.launch_entry(existing[-1].name if existing else None)
    runlog.update()

    import transformers
    original_train = transformers.Trainer.train
    training_started = {'value': False}

    # Fail fast when a starting checkpoint leaves weights unloaded. The repo scripts load
    # init_checkpoint with strict=False and only log missing keys, so a wrong checkpoint or flag
    # would otherwise train from scratch without any error. Only loads before training are
    # checked; the Trainer's own resume loads are left alone.
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        original_load_state_dict = torch.nn.Module.load_state_dict

        def checked_load_state_dict(self, state_dict, strict=True, assign=False):
            result = original_load_state_dict(self, state_dict, strict=strict, assign=assign)
            if not strict and not training_started['value']:
                missing = [k for k in result.missing_keys if not k.endswith('lm_head.weight')]  # tied to embeddings
                if missing:
                    raise RuntimeError(
                        f'[run] the starting checkpoint left {len(missing)} weights unloaded, '
                        f'for example {missing[:5]}. Check init_checkpoint, and never use '
                        f'init_base_checkpoint for an RMT parent.')
                print(f'[run] starting checkpoint loaded, no missing weights '
                      f'({len(result.unexpected_keys)} unused keys)', flush=True)
            return result

        torch.nn.Module.load_state_dict = checked_load_state_dict

    def train_with_resume(self, *args, **kwargs):
        training_started['value'] = True
        start_step = 0
        if kwargs.get('resume_from_checkpoint') is None:
            ck = complete_ckpts(Path(self.args.output_dir))
            if ck:
                kwargs['resume_from_checkpoint'] = str(ck[-1])
                start_step = step_of(ck[-1])
                print(f'[run] resuming from {ck[-1].name}', flush=True)
            else:
                print('[run] no checkpoint found, starting from step 0', flush=True)
        runlog.attach(self, start_step)
        if hasattr(transformers, 'TrainerCallback') and hasattr(self, 'add_callback'):
            self.add_callback(make_metrics_log(transformers.TrainerCallback,
                                               Path(self.args.output_dir) / 'metrics.jsonl', loss_limit))
        return original_train(self, *args, **kwargs)

    transformers.Trainer.train = train_with_resume
    args_map = {script_args[i][2:]: script_args[i + 1] for i in range(len(script_args) - 1)
                if script_args[i].startswith('--')}
    loss_limit = 2 * math.log(int(args_map.get('vocab_size', 8192)))   # twice the loss of random guessing

    # Fewer DataLoader workers on machines with few CPU cores (the repo scripts use 4).
    # The workers only collate batches, so the data and its order stay the same.
    workers = os.environ.get('GRADMEM_NUM_WORKERS')
    if workers is not None and hasattr(transformers, 'TrainingArguments'):
        original_post_init = transformers.TrainingArguments.__post_init__

        def post_init_with_workers(self):
            self.dataloader_num_workers = int(workers)
            original_post_init(self)

        transformers.TrainingArguments.__post_init__ = post_init_with_workers

    # The repo scripts create exp_path with mkdir(parents=True) and stop if it exists.
    # The run folder always exists here (restored checkpoints, resume), so allow it for this one folder.
    import pathlib
    original_mkdir = pathlib.Path.mkdir

    def mkdir_allow_run_dir(self, mode=0o777, parents=False, exist_ok=False):
        if self.resolve() == local.resolve():
            exist_ok = True
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    pathlib.Path.mkdir = mkdir_allow_run_dir

    lock, stop = threading.Lock(), threading.Event()

    def safe_mirror(final=False):
        with lock:
            try:
                runlog.update()
                if a.results_csv:
                    rebuild_results(drive.parent, a.results_csv)
                mirror(local, drive, a.log, final=final)
            except Exception as e:  # a failed mirror must not stop training
                print(f'[run] mirror failed: {e}', flush=True)

    def loop():
        while not stop.wait(a.sync_minutes * 60):
            safe_mirror()

    threading.Thread(target=loop, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # so a stop still runs the final mirror

    os.chdir(a.repo)
    sys.path.insert(0, a.repo)
    sys.argv = [a.script] + script_args
    try:
        runpy.run_path(a.script, run_name='__main__')
        runlog.status = 'finished'
    except SystemExit as e:
        runlog.status = 'stopped' if e.code == 143 else ('finished' if e.code in (0, None) else f'failed (exit {e.code})')
        raise
    except BaseException as e:
        runlog.status = f'failed ({type(e).__name__})'
        raise
    finally:
        if GUARD['reason'] and runlog.status == 'finished':
            runlog.status = f"diverged ({GUARD['reason']})"
        stop.set()
        safe_mirror(final=True)
        try:
            runlog.end_entry()
        except Exception as e:
            print(f'[run] runs.md entry failed: {e}', flush=True)
        print(f'[run] {runlog.status}, final mirror to Drive done', flush=True)


if __name__ == '__main__':
    main()