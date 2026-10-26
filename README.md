# SL-CAP

K robots with limited energy serve N tasks in the unit square and recharge at M stations, each of which
serves any number of robots at once. The objective is the makespan: the time when the last robot has
parked at a station after all tasks are served. SL-CAP is an attention policy that decides for each robot
when it becomes free whether to serve a task, recharge or park, trained by self-labeling. The
repository contains:

- the instance generator and the 12 test cases of the paper (`data/testcases`)
- training code for SL-CAP and for RL-CAP, the same policy trained by REINFORCE
- the baselines: the sequential greedy algorithm (SGA), exact branch-and-bound (B&B) and adaptive large
  neighborhood search (ALNS), all with the optimal recharging dynamic program
- the two trained models (`checkpoints`)

## Installation

Python 3.11. The policies need an NVIDIA GPU; the baselines run on the CPU. The paper used PyTorch 2.8 with
CUDA 12.9:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e .          # numpy, scipy, numba 0.67, alns 7.0.0
pytest tests              # under a minute; the first run compiles the numba kernels
```

## Evaluation

```bash
python scripts/evaluate.py data/testcases/K4_N30_M3.json --methods slcap,rlcap,sga,alns --workers 8
python scripts/summarize.py
```

`evaluate.py` writes `results/<case>/<method>.csv` with the makespan, completeness and time of every
instance. `summarize.py` prints the mean makespan, the mean gap to a reference and the median time of every
method and test case. The reference is B&B when it proved every optimum and ALNS otherwise.

| method | what it runs |
|---|---|
| `slcap`, `rlcap` | best of `--samples` (128) sampled solutions on the GPU in full fp32; `--time` adds the latency of solving one instance at a time |
| `sga` | sequential greedy algorithm |
| `bnb` | exact branch-and-bound; `--bnb-time-limit` stops it early (column `optimal`) |
| `alns` | ALNS with `--alns-iters` (25,000) iterations, or within a wall-clock budget `--alns-time` that includes its SGA start |

The classical methods run one thread per worker process. For timing, give each worker its own physical
core, for example with `taskset`.

## Reproducing the paper

Each test case `K<K>_N<N>_M<M>` has 32 instances with energy capacity E = 0.85.
`scripts/gen_testcases.py` regenerates them: instance i is drawn with `numpy.random.RandomState(i)`.

```bash
# Table 2: B&B, ALNS and SL-CAP with 10-20 tasks (B&B stops after 1 h on 29 of the 32 instances at N = 20)
python scripts/evaluate.py data/testcases/K3_N{10,15,20}_M3.json --methods bnb,alns,slcap --bnb-time-limit 3600 --workers 8

# Table 3: 30-150 tasks, 3-9 stations and robots. ALNS@25000 takes about 150 CPU-hours in total.
python scripts/evaluate.py data/testcases/K4_N{30,90,150}_M3.json data/testcases/K4_N120_M{3,6,9}.json \
    data/testcases/K{3,6,9}_N120_M3.json --methods alns,sga,rlcap,slcap --workers 32

# Equal-time comparison (Fig. 3): ALNS given the computation time of SL-CAP
python scripts/evaluate.py data/testcases/K4_N120_M3.json --methods slcap --time
python scripts/evaluate.py data/testcases/K4_N120_M3.json --methods alns --alns-time 1.36 --workers 8
python scripts/summarize.py
```

With the released checkpoints, the makespans of SL-CAP and RL-CAP match the paper exactly on an RTX 3090
with PyTorch 2.8 and CUDA 12.9 (`tests/test_slcap.py`); another GPU or library version may change the
sampled solutions. The classical methods are deterministic. Computation times depend on the hardware.
The paper ran the classical methods in a single thread on one core of an AMD Threadripper PRO 3995WX, and
the policies on one instance at a time on an RTX 3090.

## Training

```bash
python scripts/train_sl.py --out runs/slcap      # SL-CAP; defaults as in Table 1
python scripts/train_rl.py --out runs/rlcap      # RL-CAP
python scripts/evaluate.py data/testcases/K4_N30_M3.json --methods slcap --slcap runs/slcap/best.pth
```

Each run takes about 35 h on one RTX 3090 and writes `best.pth`, `latest.pth`, `train.csv` and `val.csv`
to its directory. `--resume` continues an interrupted run. Model selection uses a fixed validation set of
1,024 instances, never the test cases. Training runs fused kernels with TF32 matmuls
(`slcap/fastpath.py`); validation and evaluation run in full fp32.

## Layout

```
slcap/instances.py     instance generator (Appendix A), training distribution, test-case files
slcap/rollout.py       batched GPU simulator with the action mask; loss replay for training
slcap/policy.py        the attention policy
slcap/fastpath.py      fused kernels for training
slcap/train.py         batch prefetching, validation, checkpoints
slcap/baselines/       recharging dynamic program and plan replay (common.py), SGA, B&B, ALNS, numba kernels
scripts/               train_sl.py, train_rl.py, evaluate.py, summarize.py, gen_testcases.py
checkpoints/           slcap.pth (step 97,500), rlcap.pth (step 95,000)
data/testcases/        the 12 test cases
```

The attention blocks follow DAN (Cao, Sun and Sartoretti, 2021). The self-labeling scheme is that of
Corsini et al. (2024). ALNS follows Ropke and Pisinger (2006) and is built on the `alns` package (Wouda and
Lan, 2023).

## License

MIT (see `LICENSE`).
