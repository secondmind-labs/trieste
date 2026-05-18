# Deep Ensembles Autoresearch

This is an experiment to improve the speed of Deep Ensembles model training, and possibly prediction.
You are an expert in Deep Learning, Tensorflow, Keras and how to optimise code using a GPU
under these frameworks.

## Goal

The objective is to get the fastest possible implementation of Deep Ensembles.
To achieve this, you will repeatedly run a benchmarking script (`train.py`) which trains
a Deep Ensemble model with a given architecture and hyperparameters on a fixed training and testing set.

**Simplicity Criterion**

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. 
Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. 
When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. 
A small time improvement that adds 20 lines of hacky code? Probably not worth it. A small time improvement from 
deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

### What you can do

- **You must only improve performance by changing the project Deep Ensemble's internal implementation**.

### What you cannot do

- Do not attempt to change the model public API, because other model types and the project in
its entirety depend on it.
- Modify any of the files in this directory (e.g. `data.py`, `train.py` etc.). Preventing the modification of `train.py` means that all improvements must come from the code, not the model hyperparameters.

## Experimentation

Each experiment runs on a single P100 16Gb GPU. The training/evaluation script `train.py` runs for a fixed
budget of 3 epochs on the same training data. You simply launch it with:

```bash
python train.py
```

Once the script finishes, it prints at the end a summary like:

```commandline
---
Test RMSE: 1.004589
Test NLPD: 1.422797
Training Time: 50.44s
```

which can be grepped to extract these 3 metrics. We're also interested in recording various
GPU metrics. While the experiment is running, repeatedly run on a separate shell the following:

```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
```

which outputs something like:

```commandline
utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]
0 %, 0 %, 16384 MiB, 16270 MiB, 0 MiB
```

and keep track of the max of the `utilization.gpu` and `utilization.memory` metrics.

## Logging Results

Once an experiment is done, log the results to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```commandline
Commit  Training Time   GPU Utilisation GPU Memory  RMSE    NLPD    Status  Description
```

1. Git commit hash (short, 7 chars)
2. Total training time in seconds
3. % GPU Utilisation (max)
4. % GPU Memory (max)
5. Test RMSE
6. Test NLPD
7. Status: `Keep`, `Discard`, or `Crash`
8. Short text description of what the experiment tried

## Phase 0: Setup

1. **Understand the repo**, the model interface and how Deep Ensembles are implemented,
   you can read the original paper at https://arxiv.org/pdf/1612.01474
2. **Read the in-scope files**:
    - data.py: how training/test data was created
    - train.npz, test.npz: training/testing data produced by the previous script
    - train.py: training and testing on the data produced by the previous script
3. **Agree on a run tag** based on today's date (e.g. 18-05-2025). The branch `de-autoresearch/<tag>` must not already exist
4. **Create the branch**: `git checkout -b de-autoresearch/<tag>` from the current branch
4. **Initialise results**: create results.tsv with just the header row. The baseline will be recorded after the first run
5. **Confirm and go**: Confirm you understand the aim of this experiment and that the set up looks good.

Once you get confirmation, kick off the experimentation.

## Phase 1: Confirm baseline

Run the `train.py` script in the current directory before making any changes to the model's implementation.

Record the baseline to `results.tsv`.

## Phase 2: The Experiment Loop

Loop for 1 hour:
1. Look at the git state: the current branch/commit you're on
2. Generate an implementation idea, plan how to implement and execute
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
    - While running, monitor `nvidia-smi` according the above instructions
5. Read out the results: `grep "^Test\|^Training" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If training time improved (lower), and RMSE/NLPD are as good as the baseline (lower), you "advance" the branch, keeping the git commit
9. If training time is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. 
And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind 
but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take a couple of minutes at most (including startup and eval overhead). If a run exceeds
that, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a 
typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the 
status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should 
continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from 
a computer and expects you to continue working indefinitely for 1 hour or until you are manually stopped. You are autonomous. If you 
run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining
previous near-misses, try more radical architectural changes. The loop runs for at most 1 hour or until the human interrupts you, period.