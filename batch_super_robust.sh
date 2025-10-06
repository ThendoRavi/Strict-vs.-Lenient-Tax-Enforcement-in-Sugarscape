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
echo "Working directory: $(pwd)"

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
            # Try to find python after loading module
            for cmd in python3 python; do
                if command -v $cmd &> /dev/null; then
                    PYTHON_CMD=$cmd
                    echo "Found Python after module load: $PYTHON_CMD"
                    break 2
                fi
            done
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
echo "Looking for Java..."
if ! command -v java &> /dev/null; then
    for java_module in java/11 java/8 openjdk/11 openjdk/8; do
        if module avail $java_module 2>/dev/null | grep -q $java_module; then
            echo "Loading Java module: $java_module"
            module load $java_module
            break
        fi
    done
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export DISPLAY=""
export PYTHONPATH="${PYTHONPATH}:${HOME}/.local/lib/python3.9/site-packages:${HOME}/.local/lib/python3.8/site-packages"

# Function to install packages
install_packages() {
    local install_cmd="$1"
    echo "Trying to install with: $install_cmd"
    
    # Try to install each package individually to see which ones fail
    for package in numpy pandas matplotlib tensorflow pynetlogo; do
        echo "Installing $package..."
        if eval "$install_cmd $package"; then
            echo "✓ $package installed successfully"
        else
            echo "✗ Failed to install $package"
        fi
    done
}

# Try different installation methods
echo "Installing Python packages..."

# Method 1: Try pip with user install
if command -v pip3 &> /dev/null; then
    echo "Method 1: Using pip3 with --user"
    install_packages "pip3 install --user --upgrade"
elif command -v pip &> /dev/null; then
    echo "Method 1: Using pip with --user"  
    install_packages "pip install --user --upgrade"
else
    echo "Method 1: Using python -m pip with --user"
    install_packages "$PYTHON_CMD -m pip install --user --upgrade"
fi

# Verify installations
echo "Verifying package installations..."
$PYTHON_CMD -c "
import sys
print('Python version:', sys.version)
print('Python executable:', sys.executable)

packages = ['numpy', 'pandas', 'matplotlib', 'tensorflow', 'pynetlogo']
for pkg in packages:
    try:
        exec(f'import {pkg}')
        print(f'✓ {pkg} - OK')
    except ImportError as e:
        print(f'✗ {pkg} - FAILED: {e}')

print('\\nPython path:')
for p in sys.path:
    print(f'  {p}')
"

# Check if we have the required files
echo "Checking required files..."
for file in dqn.py "Sugarscape 2 Constant Growback.nlogo"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        echo "Contents of current directory:"
        ls -la
        exit 1
    fi
done

# Run the DQN simulation
echo "Starting DQN simulation with: $PYTHON_CMD"
$PYTHON_CMD dqn.py --cluster-mode --job-id "$SLURM_JOB_ID"

echo "Job completed at $(date)"