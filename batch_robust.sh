#!/bin/bash
#SBATCH --job-name=dqn_sugarscape
#SBATCH --output=dqn_sugarscape_%j.log
#SBATCH --error=dqn_sugarscape_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12
#SBATCH --partition=stampede
#SBATCH --time=3-00:00:00

echo "Starting DQN Sugarscape job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Try different Python commands until one works
PYTHON_CMD=""
for cmd in python3 python python3.9 python3.8; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        echo "Found Python: $PYTHON_CMD"
        $PYTHON_CMD --version
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No Python installation found!"
    echo "Trying to load Python module..."
    
    # Try common Python module names
    for module in python/3.9 python/3.8 python3/3.9 python3/3.8 Python/3.9; do
        if module avail $module 2>/dev/null | grep -q $module; then
            echo "Loading module: $module"
            module load $module
            PYTHON_CMD=python3
            break
        fi
    done
fi

# If still no Python, exit with error
if [ -z "$PYTHON_CMD" ] || ! command -v $PYTHON_CMD &> /dev/null; then
    echo "ERROR: Could not find or load Python. Available modules:"
    module avail python 2>&1 | head -20
    exit 1
fi

# Try to load Java for NetLogo
for java_module in java/11 java/8 openjdk/11 openjdk/8; do
    if module avail $java_module 2>/dev/null | grep -q $java_module; then
        echo "Loading Java module: $java_module"
        module load $java_module
        break
    fi
done

# Set environment variables
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export DISPLAY=""

# Check if we can create virtual environment
echo "Checking virtual environment capabilities..."
if $PYTHON_CMD -m venv --help > /dev/null 2>&1; then
    echo "venv module available"
    # Create and activate virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment, trying without venv..."
            USE_VENV=false
        else
            USE_VENV=true
        fi
    else
        USE_VENV=true
    fi
else
    echo "venv module not available, installing packages globally"
    USE_VENV=false
fi

# Activate virtual environment if available
if [ "$USE_VENV" = true ] && [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Find pip command
PIP_CMD=""
for cmd in pip pip3 "$PYTHON_CMD -m pip"; do
    if eval "$cmd --version" > /dev/null 2>&1; then
        PIP_CMD="$cmd"
        echo "Found pip: $PIP_CMD"
        break
    fi
done

if [ -z "$PIP_CMD" ]; then
    echo "No pip found, trying to install packages with python -m pip"
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Install packages
echo "Installing packages with: $PIP_CMD"
$PIP_CMD install --user --upgrade pip
$PIP_CMD install --user numpy pandas matplotlib tensorflow pynetlogo

# Verify key packages
echo "Verifying installations..."
$PYTHON_CMD -c "
try:
    import numpy, pandas, matplotlib, tensorflow
    print('✓ Core packages OK')
    import pynetlogo
    print('✓ PyNetLogo OK')
    print('All packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Run the DQN simulation
echo "Starting DQN simulation..."
$PYTHON_CMD dqn.py --cluster-mode --job-id "$SLURM_JOB_ID"

echo "Job completed at $(date)"