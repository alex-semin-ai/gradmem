# Running the MQAR queue on a GPU server

Tested for a server with 2 × GTX 1080 Ti, driver 580, Python 3.12, no admin rights, no SLURM.

## One-time setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone -b my-idea3 https://github.com/alex-semin-ai/gradmem.git ~/gradmem
uv venv ~/venvs/gradmem --python 3.12
source ~/venvs/gradmem/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers==4.51.3 accelerate safetensors numpy pandas tensorboard
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list()); x = torch.randn(256, 256, device='cuda'); print((x @ x).sum().item())"
```

The last line must print `True`, an architecture list with `sm_61` (the 1080 Ti), and a number. If `sm_61` is missing or it fails with "no kernel image is available", reinstall torch from `https://download.pytorch.org/whl/cu118`.

## Each session

```bash
source ~/venvs/gradmem/bin/activate
cd ~/gradmem && git pull
tmux new -s train
python vm/run_queue.py vm/queue.json --dry_run
python vm/run_queue.py vm/queue.json
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t train`. Ctrl+C in the runner stops every run cleanly.

Parents must be in `~/gradmem_idea3/runs/` before a GradMem run starts (copy run folders with `scp -r`).

## Logs and results

| What | Where |
|---|---|
| Queue progress (every 10 minutes, starts and ends) | `~/gradmem_idea3/logs/queue_*.log` |
| Console log of a run | `~/gradmem_idea3/logs/<RUN>_*.log`. `grep -a "\[eval\]" <log>` shows one line per evaluation |
| Every logged metric | `~/gradmem_idea3/runs/<RUN>/files/metrics.jsonl` |
| Status | `~/gradmem_idea3/runs/<RUN>/summary.json` (`running`, `finished`, `stopped`, `failed`) |
| Table of all runs | `~/gradmem_idea3/results/results.csv` and `runs.md` |
| Detailed report | `python vm/make_report.py ~/gradmem_idea3`, then open `results/report.html` |
