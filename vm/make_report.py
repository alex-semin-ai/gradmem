"""Build a detailed report of all MQAR runs in a project folder.

Usage
    python vm/make_report.py ~/gradmem_idea3            writes results/report.html and results/report.md

Works on run folders from the server, Colab and Kaggle alike. Metrics come from files/metrics.jsonl when a run
has it, otherwise from files/trainer_state.json (runs trained before metrics.jsonl existed). No extra packages,
the charts are inline SVG.
"""
import html, json, math, sys, time
from pathlib import Path

THRESHOLDS = [0.90, 0.95, 0.99]
# (key in the logs, chart title, y-axis title, scale to percent)
EVAL_CHARTS = [
    ('eval_token_accuracy', 'Token accuracy (eval)', 'token accuracy, %', True),
    ('eval_all_queries_exact_match', 'Exact match, all 16 answers right (eval)', 'exact match, %', True),
    ('eval_loss', 'Total loss (eval)', 'loss, nats per token', False),
    ('eval_target_loss', 'Read (target) loss (eval)', 'loss, nats per token', False),
    ('eval_rec_loss', 'RMT reconstruction loss (eval)', 'loss, nats per token', False),
    ('eval_inner_loss', 'GradMem write loss before the last write step (eval)', 'loss, nats per token', False),
    ('eval_delta_mem_norm_mean', 'Memory change by the writes, mean norm (eval)', 'norm of M_K minus M_0', False),
    ('eval_mem_norm_mean', 'Memory norm, mean (eval)', 'norm of the memory', False),
    ('eval_inner_grad_norm', 'Gradient norm inside the write steps (eval)', 'gradient norm', False),
    ('eval_mem_attn_read', 'Attention of the read to the memory (eval)', 'attention share', False),
]
TRAIN_CHARTS = [
    ('loss', 'Total loss (train)', 'loss, nats per token', False),
    ('target_loss', 'Read (target) loss (train)', 'loss, nats per token', False),
    ('rec_loss', 'RMT reconstruction loss (train)', 'loss, nats per token', False),
    ('outer_loss', 'GradMem outer loss (train)', 'loss, nats per token', False),
    ('grad_norm', 'Gradient norm of the weights (train)', 'gradient norm', False),
    ('learning_rate', 'Learning rate', 'learning rate', False),
]
GLOSSARY = [
    ('Token accuracy', 'share of query answers that are right, over all queries of all validation examples'),
    ('Exact match', 'share of validation examples with all 16 answers right. About token accuracy to the power 16'),
    ('Read (target) loss', 'cross-entropy of the answers at the query positions, the loss the read is trained on'),
    ('RMT reconstruction loss', 'cross-entropy of rebuilding the context from the memory (RMT runs, D11)'),
    ('GradMem write loss', 'the reconstruction loss the write steps minimize, measured before the last write step'),
    ('Steps to X', 'first evaluation step that starts 3 evaluations in a row at or above X. A dash means never'),
    ('Test', 'the selected checkpoint evaluated once on a test split from another seed (eval_runs.py)'),
    ('bench', 'cost measured by eval_runs.py on one device for every run. RMT inference without its reconstruction pass'),
    ('GPU hours', 'wall time of all training sessions of the run, evaluation included'),
    ('RMT', 'Recurrent Memory Transformer. Writes the memory with forward passes (Bulatov et al., 2022)'),
    ('GradMem', 'writes the memory with gradient steps on a reconstruction loss, the method this project studies'),
]
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']


def load_run(d):
    rec = json.load(open(d / 'config.json'))
    summary = json.load(open(d / 'summary.json')) if (d / 'summary.json').exists() else {}
    events = []
    mj = d / 'files' / 'metrics.jsonl'
    if mj.exists():
        for line in open(mj, encoding='utf8'):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    if not events and (d / 'files' / 'trainer_state.json').exists():
        events = list(json.load(open(d / 'files' / 'trainer_state.json')).get('log_history', []))
    # one value per (step, key); a resumed session repeats the eval at its start step
    evals, trains = {}, {}
    for e in events:
        step = e.get('step')
        if step is None:
            continue
        target = evals if any(k.startswith('eval_') for k in e) else trains
        target.setdefault(step, {}).update({k: v for k, v in e.items() if isinstance(v, (int, float))})
    return dict(name=d.name, script=rec.get('script'), cfg=rec.get('config', {}), summary=summary,
                evals=dict(sorted(evals.items())), trains=dict(sorted(trains.items())))


def run_kind(r):
    cfg = r['cfg']
    init = cfg.get('init_checkpoint')
    parent = init.replace('\\', '/').split('/runs/')[1].split('/')[0] if init and '/runs/' in init.replace('\\', '/') else None
    model = {'run_rmt_on_mqar.py': 'RMT', 'run_gradmemgpt_on_mqar.py': 'GradMem',
             'run_enc_gradmem_on_mqar.py': 'Combined'}.get(r['script'], r['script'])
    return model, parent


def series(r, key, evals=True):
    src = r['evals'] if evals else r['trains']
    return [(s, v[key]) for s, v in src.items() if key in v and isinstance(v[key], (int, float)) and math.isfinite(v[key])]


def first_step(points, thr, in_a_row=3):
    """First eval step that starts a streak of in_a_row evals at or above thr, so one lucky eval does not count
    (fairness fix 2). A run that stopped before completing a streak (the old one-eval stop rule) gets none."""
    for i, (s, _) in enumerate(points):
        streak = points[i:i + in_a_row]
        if len(streak) == in_a_row and all(v >= thr for _, v in streak):
            return s
    return None


BUDGETS = [5000, 10000, 20000, 30000, 60000]


def at_budget(points, budget):
    """The metric at a fixed step budget. A run that ended earlier (the stop rule) keeps its last value."""
    exact = [v for s, v in points if s == budget]
    if exact:
        return exact[-1], False
    before = [v for s, v in points if s <= budget]
    if before and points and points[-1][0] < budget:
        return before[-1], True
    return None, False


def speed_by_device(r):
    """ms per training step from the sessions on the device with the most steps (fairness fix 9)."""
    by = {}
    for x in r['summary'].get('sessions', []):
        if x.get('start_step') is None or x.get('end_step') is None or x['end_step'] <= x['start_step']:
            continue
        d = by.setdefault(x.get('device', 'unknown device'), [0, 0.0])
        d[0] += x['end_step'] - x['start_step']
        d[1] += x.get('seconds') or 0
    if not by:
        return None, None
    dev, (steps, secs) = max(by.items(), key=lambda kv: kv[1][0])
    return 1000 * secs / steps, dev


def gpu_hours(r):
    return sum(x.get('seconds') or 0 for x in r['summary'].get('sessions', [])) / 3600


def ms_per_step(r):
    ses = [x for x in r['summary'].get('sessions', []) if x.get('start_step') is not None and x.get('end_step') is not None
           and x['end_step'] > x['start_step']]
    steps = sum(x['end_step'] - x['start_step'] for x in ses)
    return 1000 * sum(x['seconds'] for x in ses) / steps if steps else None


def fmt(v, pct=False, nd=4):
    if v is None:
        return '–'
    if pct:
        return f'{100 * v:.2f}'
    if isinstance(v, float):
        return f'{v:.{nd}g}' if abs(v) < 1e-3 or abs(v) >= 1e4 else f'{v:.{nd}f}'
    return str(v)


def svg_chart(lines, title, xlabel, ylabel, pct=False, width=560, height=300, logy=False):
    """Line chart. lines is a list of (label, color, [(x, y), ...])."""
    pts = [(x, y) for _, _, p in lines for x, y in p]
    if not pts:
        return ''
    tf = (lambda y: 100 * y) if pct else (lambda y: y)
    xs = [x for x, _ in pts]
    ys = [tf(y) for _, y in pts]
    if logy:
        ys = [y for y in ys if y > 0]
        if not ys:
            logy = False
            ys = [tf(y) for _, y in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = (min(ys), max(ys))
    if logy:
        y0, y1 = math.log10(y0), math.log10(y1)
    if pct:
        y0, y1 = 0.0, 100.0
    if y1 == y0:
        y1 = y0 + 1
    L, R, T, B = 62, 12, 30, 44
    W, Hh = width - L - R, height - T - B

    def X(x):
        return L + (x - x0) / ((x1 - x0) or 1) * W

    def Y(y):
        y = tf(y)
        if logy:
            y = math.log10(max(y, 1e-12))
        return T + (1 - (y - y0) / (y1 - y0)) * Hh

    out = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{L}" y="18" class="t">{html.escape(title)}</text>']
    for i in range(6):   # grid and tick labels
        gy = y0 + (y1 - y0) * i / 5
        py = T + (1 - i / 5) * Hh
        lab = f'{10 ** gy:.3g}' if logy else f'{gy:.3g}'
        out.append(f'<line x1="{L}" y1="{py:.1f}" x2="{L + W}" y2="{py:.1f}" class="g"/>'
                   f'<text x="{L - 6}" y="{py + 4:.1f}" class="k" text-anchor="end">{lab}</text>')
        gx = x0 + (x1 - x0) * i / 5
        px = L + i / 5 * W
        xl = f'{gx:,.0f}' if x1 - x0 >= 50 else f'{gx:.3g}'
        out.append(f'<text x="{px:.1f}" y="{T + Hh + 16}" class="k" text-anchor="middle">{xl}</text>')
    out.append(f'<text x="{L + W / 2}" y="{height - 6}" class="a" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{T + Hh / 2}" class="a" text-anchor="middle" transform="rotate(-90 14 {T + Hh / 2})">'
               f'{html.escape(ylabel + (" (log scale)" if logy else ""))}</text>')
    if not pct and not logy and y0 > 0:
        out.append(f'<text x="{L + W}" y="{T - 4}" class="k" text-anchor="end">y axis does not start at 0</text>')
    for label, color, p in lines:
        if not p:
            continue
        d = ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in p if not logy or tf(y) > 0)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.6"><title>{html.escape(label)}</title></polyline>')
        if len(p) <= 12:   # short lines (the write-steps plot) get a marker per point
            out.extend(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{color}"><title>{html.escape(label)}: '
                       f'{fmt(x)}, {tf(y):.4g}</title></circle>' for x, y in p if not logy or tf(y) > 0)
    out.append('</svg>')
    legend = ''.join(f'<span class="lg"><i style="background:{c}"></i>{html.escape(l)}</span>' for l, c, p in lines if p)
    return f'<figure>{"".join(out)}<figcaption>{legend}</figcaption></figure>'


def main():
    proj = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path('~/gradmem_idea3').expanduser()
    runs = [load_run(d) for d in sorted(Path(proj, 'runs').iterdir()) if (d / 'config.json').exists()]
    color = {r['name']: COLORS[i % len(COLORS)] for i, r in enumerate(runs)}
    test_path = Path(proj, 'results', 'test_results.json')
    tests = json.load(open(test_path)) if test_path.exists() else {}
    rows, md = [], []
    for r in runs:
        model, parent = run_kind(r)
        acc = series(r, 'eval_token_accuracy')
        em = series(r, 'eval_all_queries_exact_match')
        last = r['evals'][max(r['evals'])] if r['evals'] else {}
        t = tests.get(r['name'], {})
        ms, dev = speed_by_device(r)
        acc99 = first_step(acc, 0.99)
        rows.append(dict(
            run=r['name'], model=model, K=r['cfg'].get('K'), pairs=r['cfg'].get('num_kv_pairs'),
            start=parent or 'scratch', status=r['summary'].get('status', '?'),
            steps=max(r['evals']) if r['evals'] else None,
            best_acc=max((v for _, v in acc), default=None), best_em=max((v for _, v in em), default=None),
            **{f'acc{int(t * 100)}': first_step(acc, t) for t in THRESHOLDS},
            **{f'em{int(t * 100)}': first_step(em, t) for t in THRESHOLDS},
            target_loss=last.get('eval_target_loss', last.get('eval_loss')), ms=ms, device=dev, gpu_h=gpu_hours(r),
            sessions=len(r['summary'].get('sessions', [])), conclusion=r['summary'].get('conclusion') or '',
            test_acc=t.get('test_token_accuracy'), test_em=t.get('test_exact_match'), params=t.get('params'),
            infer_ms=t.get('infer_ms_per_batch'), train_ms=t.get('train_ms_per_step'),
            h_to_99=(acc99 * t['train_ms_per_step'] / 3.6e6) if acc99 is not None and t.get('train_ms_per_step') else None,
            **{f'b_acc{b // 1000}k': at_budget(acc, b) for b in BUDGETS},
            **{f'b_em{b // 1000}k': at_budget(em, b) for b in BUDGETS}))

    cols = [('run', 'Run'), ('model', 'Model'), ('K', 'K'), ('pairs', 'Pairs'), ('start', 'Starts from'),
            ('status', 'Status'), ('steps', 'Last eval step'), ('best_acc', 'Best token acc, %'),
            ('best_em', 'Best exact match, %'), ('acc90', 'Steps to 90 % acc'), ('acc95', 'to 95 %'),
            ('acc99', 'to 99 %'), ('em90', 'Steps to 90 % EM'), ('em99', 'to 99 % EM'),
            ('test_acc', 'Test token acc, %'), ('test_em', 'Test exact match, %'),
            ('target_loss', 'Last eval target loss'), ('params', 'Parameters'),
            ('infer_ms', 'Inference ms per batch (bench)'), ('train_ms', 'Train ms per step (bench)'),
            ('h_to_99', 'GPU h to 99 % acc (bench)'), ('ms', 'ms per step (logs)'), ('device', 'Device (logs)'),
            ('gpu_h', 'GPU hours'), ('sessions', 'Sessions')]
    budget_cols = ([('run', 'Run')] + [(f'b_acc{b // 1000}k', f'Token acc at {b // 1000}k, %') for b in BUDGETS]
                   + [(f'b_em{b // 1000}k', f'Exact match at {b // 1000}k, %') for b in BUDGETS])
    pct_cols = {'best_acc', 'best_em', 'test_acc', 'test_em'}

    def cell(row, k):
        v = row[k]
        if isinstance(v, tuple):   # a fixed-budget value, (value, carried from an earlier stop)
            return '–' if v[0] is None else f'{100 * v[0]:.2f}' + (' (stopped)' if v[1] else '')
        if k in pct_cols:
            return fmt(v, pct=True)
        if k in ('ms', 'gpu_h', 'infer_ms', 'train_ms', 'h_to_99') and v is not None:
            return f'{v:.1f}' if k in ('ms', 'infer_ms', 'train_ms') else f'{v:.2f}'
        if k == 'params' and v is not None:
            return f'{v / 1e6:.2f} M'
        return fmt(v)

    table = '<table><tr>' + ''.join(f'<th>{h}</th>' for _, h in cols) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(cell(row, k))}</td>' for k, _ in cols) + '</tr>' for row in rows) + '</table>'
    budget_table = '<table><tr>' + ''.join(f'<th>{h}</th>' for _, h in budget_cols) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(cell(row, k))}</td>' for k, _ in budget_cols) + '</tr>' for row in rows) + '</table>'
    for cs in (cols, budget_cols):
        md.append('| ' + ' | '.join(h for _, h in cs) + ' |')
        md.append('|' + '---|' * len(cs))
        for row in rows:
            md.append('| ' + ' | '.join(cell(row, k) for k, _ in cs) + ' |')
        md.append('')

    # K plot. Best token accuracy and best exact match against the number of write steps, GradMem runs by start
    kplot = ''
    groups = {}
    for row in rows:
        if row['model'] in ('GradMem', 'Combined') and row['K'] is not None:
            groups.setdefault((row['model'], row['start']), []).append(row)
    if groups:
        lines_acc, lines_em = [], []
        for i, ((model, start), rs) in enumerate(sorted(groups.items())):
            rs = sorted(rs, key=lambda x: x['K'])
            c = COLORS[i % len(COLORS)]
            label = f'{model} from {start}' + (' (encoder + K write steps)' if model == 'Combined' else '')
            # test values when eval_runs.py has run, else the best validation value
            lines_acc.append((label, c, [(x['K'], x['test_acc'] if x['test_acc'] is not None else x['best_acc'])
                                         for x in rs if x['best_acc'] is not None]))
            lines_em.append((label, c, [(x['K'], x['test_em'] if x['test_em'] is not None else x['best_em'])
                                        for x in rs if x['best_em'] is not None]))
        kplot = (svg_chart(lines_acc, 'Token accuracy against write steps (test if measured, else best validation)', 'backward write steps K', 'token accuracy, %', pct=True)
                 + svg_chart(lines_em, 'Exact match against write steps (test if measured, else best validation)', 'backward write steps K', 'exact match, %', pct=True))

    def charts(defs, evals, runs_):
        out = []
        for key, title, ylab, pct in defs:
            lines = [(r['name'], color[r['name']], series(r, key, evals)) for r in runs_]
            if any(p for _, _, p in lines):
                logy = 'loss' in key and not pct
                out.append(svg_chart(lines, title, 'training step', ylab, pct=pct, logy=logy))
        return ''.join(out)

    per_run = []
    for r in runs:
        model, parent = run_kind(r)
        ses = r['summary'].get('sessions', [])
        ses_rows = ''.join(f'<tr><td>{html.escape(str(x.get("started")))}</td><td>{x.get("start_step")}</td>'
                           f'<td>{x.get("end_step")}</td><td>{(x.get("seconds") or 0) / 3600:.2f}</td>'
                           f'<td>{html.escape(str(x.get("commit")))}</td><td>{html.escape(str(x.get("note") or ""))}</td></tr>'
                           for x in ses)
        last = r['evals'][max(r['evals'])] if r['evals'] else {}
        last_rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td></tr>' for k, v in sorted(last.items())
                            if k.startswith('eval_'))
        cfg_rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>' for k, v in r['cfg'].items())
        per_run.append(
            f'<section id="{html.escape(r["name"])}"><h2>{html.escape(r["name"])}</h2>'
            f'<p>{model}, K = {r["cfg"].get("K")}, starts from {html.escape(parent or "scratch")}. '
            f'Status {html.escape(str(r["summary"].get("status", "?")))}. '
            f'{html.escape(r["summary"].get("conclusion") or "")}</p>'
            f'<div class="charts">{charts(EVAL_CHARTS, True, [r])}{charts(TRAIN_CHARTS, False, [r])}</div>'
            f'<details><summary>Every metric of the last evaluation (step {max(r["evals"]) if r["evals"] else "–"})</summary>'
            f'<table>{last_rows}</table></details>'
            f'<details><summary>Sessions</summary><table><tr><th>Started</th><th>From step</th><th>To step</th>'
            f'<th>Hours</th><th>Code</th><th>Note</th></tr>{ses_rows}</table></details>'
            f'<details><summary>Configuration</summary><table>{cfg_rows}</table></details></section>')

    glossary = '<dl>' + ''.join(f'<dt>{html.escape(a)}</dt><dd>{html.escape(b)}</dd>' for a, b in GLOSSARY) + '</dl>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MQAR Runs Report</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;color:#1b1b1b;background:#fff;max-width:1400px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#f2f2f2}}
.charts{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0}}svg{{max-width:100%;height:auto;background:#fff}}
.t{{font-size:13px;font-weight:600}}.k{{font-size:10px;fill:#555}}.a{{font-size:11px;fill:#222}}.g{{stroke:#e5e5e5}}
.lg{{font-size:11px;margin-right:10px;white-space:nowrap}}.lg i{{display:inline-block;width:10px;height:10px;margin-right:4px}}
dt{{font-weight:600}}dd{{margin:0 0 6px 12px}}.wrap{{overflow-x:auto}}section{{border-top:1px solid #ddd;margin-top:16px}}
</style></head><body>
<h1>MQAR runs report</h1><p>Project {html.escape(str(proj))}. Built {time.strftime('%Y-%m-%d %H:%M')}. {len(runs)} runs.</p>
<h2>Overview</h2><p>Steps to X are the first step of 3 evals in a row at or above X. Test columns come from eval_runs.py
(the selected checkpoint, once, on a test split no run has seen). "bench" columns are measured by eval_runs.py for all runs
on one device. "logs" columns come from the training sessions.</p><div class="wrap">{table}</div>
<h2>Fixed step budgets</h2><p>Validation values at the same number of training steps for every run. "(stopped)" means the
run ended earlier by the stop rule and its last value is carried.</p><div class="wrap">{budget_table}</div>
<h2>Write steps</h2><div class="charts">{kplot or '<p>No GradMem runs yet.</p>'}</div>
<h2>All runs, evaluation</h2><div class="charts">{charts(EVAL_CHARTS, True, runs)}</div>
<h2>All runs, training</h2><div class="charts">{charts(TRAIN_CHARTS, False, runs)}</div>
{''.join(per_run)}
<h2>Glossary</h2>{glossary}
</body></html>"""
    out_dir = Path(proj, 'results')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.html').write_text(page, encoding='utf8')
    (out_dir / 'report.md').write_text('# MQAR runs report\n\n' + '\n'.join(md) + '\n\n' + '\n'.join(
        f'- **{row["run"]}**. {row["conclusion"]}' for row in rows if row['conclusion']) + '\n', encoding='utf8')
    print('Report written to', out_dir / 'report.html')


if __name__ == '__main__':
    main()
