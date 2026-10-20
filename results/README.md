# GradMem Idea 3, results of all runs

`gradmem_idea3_runs_no_ckpt.zip` (38 MB) holds every run of the Idea 3 experiments on MQAR (16 and 32 pairs):
RMT, GradMem from E1 and from scratch, the RMT encoder + GradMem model, ablations, step sweeps, forward reuse
variants (fused sum, context write, focused gradient, Polyak step), extra seeds and RMT xN at 32 pairs.
There are 131 runs. Model checkpoints are left out to keep the file small.

The final report is in `reports/gradmem_idea3_report_ru.tex` (LaTeX source, Russian).

## What is inside

```
gradmem_idea3/
  runs/<RUN>/
    config.json          the run configuration
    summary.json         the final status of the run
    code_versions.txt    the code version the run used
    files/               trainer_state.json, metrics.jsonl, TensorBoard events
    logs/                console logs of every training session
  extras/<batch>/        result tables and logs of each batch of runs, in the order they were made
```

The test numbers of all runs are in `gradmem_idea3/extras/rmt_p32_vm/results/test_results.csv`
(90 runs, test split with seed 456 and 3000 examples, timed on one GTX 1080 Ti).
The scripts that train and evaluate the runs are `vm/run_queue_f.py` and `eval_runs_f.py` in this repo.

## How to open

The archive uses LZMA compression. Python, 7-Zip and `tar` open it. The zip viewer built into Windows
Explorer may not.

    tar -xf gradmem_idea3_runs_no_ckpt.zip

or

    python -c "import zipfile; zipfile.ZipFile('gradmem_idea3_runs_no_ckpt.zip').extractall()"
