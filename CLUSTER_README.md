# DQN Sugarscape Cluster Setup

## Quick Start for Cluster

1. **Test environment first:**

```bash
bash test_env.sh
```

2. **Submit job (recommended - most robust):**

```bash
sbatch batch_final.sh
```

**Alternative batch scripts (try if final doesn't work):**

```bash
sbatch batch_super_robust.sh  # Extra diagnostics
sbatch batch_robust.sh        # Standard robust version
sbatch batch.sh               # Basic version
```

## Files Overview

- `batch_final.sh` - **RECOMMENDED** - Most robust batch script with detailed logging
- `batch_super_robust.sh` - Extra robust version with extensive diagnostics
- `batch_robust.sh` - Standard robust batch script
- `batch.sh` - Basic SLURM batch script
- `test_env.sh` - Environment testing script
- `dqn.py` - Main DQN simulation (now cluster-compatible)
- `requirements.txt` - Original Python package requirements (strict versions)
- `requirements_cluster.txt` - Flexible requirements for cluster compatibility

## Troubleshooting

### "python: command not found" or "pip: command not found"

The new batch scripts automatically handle this by:

- Trying different Python commands: `python3`, `python`, `python3.9`, `python3.8`
- Loading Python modules: `python/3.9`, `python/3.8`, etc.
- Using `python -m pip` instead of `pip` directly
- Installing packages with `--user` flag to avoid permission issues

### Virtual environment creation failed

The updated scripts now:

- Check if venv module is available before trying to create virtual environment
- Fall back to global/user installation if venv fails
- Use flexible package versions that are more likely to be available

### Package installation failures

New strategies implemented:

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
