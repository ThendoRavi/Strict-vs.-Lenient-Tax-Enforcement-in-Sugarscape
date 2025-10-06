# DQN Sugarscape Cluster Setup

## Quick Start for Cluster

1. **Test environment first:**

```bash
bash test_env.sh
```

2. **Submit job:**

```bash
sbatch batch_robust.sh
```

## Files Overview

- `batch.sh` - Basic SLURM batch script
- `batch_robust.sh` - Robust batch script that handles different cluster configurations
- `test_env.sh` - Environment testing script
- `dqn.py` - Main DQN simulation (now cluster-compatible)
- `requirements.txt` - Python package requirements

## Troubleshooting

### "python: command not found"

The robust batch script (`batch_robust.sh`) automatically tries:

- `python3`, `python`, `python3.9`, `python3.8`
- Loading Python modules: `python/3.9`, `python/3.8`, etc.

### Module Loading Issues

Check available modules on your cluster:

```bash
module avail python
module avail java
```

### NetLogo Issues

NetLogo requires Java. The script tries to load Java modules automatically.

## Customization

Edit the SLURM parameters in batch scripts:

- `--partition`: Change to your cluster's partition (stampede, bigbatch, biggpu)
- `--cpus-per-task`: Adjust CPU cores
- `--time`: Adjust time limit
- `--nodes`: Usually keep at 1 unless doing parallel computing

## Command Line Options

Run locally with options:

```bash
python3 dqn.py --full-mode  # Run full experiments instead of quick test
python3 dqn.py --cluster-mode --job-id 12345  # Simulate cluster mode locally
```

## Output Files

- `dqn_sugarscape_JOBID.log` - Standard output
- `dqn_sugarscape_err_JOBID.log` - Error output
- `dqn_run_JOBID.log` - Application log (cluster mode)
- `results/` - DQN results and plots
