#!/bin/bash
#SBATCH --job-name=dqn_sugarscape
#SBATCH --output=dqn_sugarscape_%j.log
#SBATCH --error=dqn_sugarscape_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12
#SBATCH --partition=stampede
#SBATCH --time=3-00:00:00

echo "==============================================="
echo "DQN Sugarscape Cluster Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "==============================================="

# Function to try loading a module
try_module() {
    local module_name="$1"
    if module list 2>&1 | grep -q "$module_name"; then
        echo "✓ Module $module_name already loaded"
        return 0
    elif module avail "$module_name" 2>&1 | grep -q "$module_name"; then
        echo "Loading module: $module_name"
        module load "$module_name"
        return $?
    else
        echo "✗ Module $module_name not available"
        return 1
    fi
}

# Try to load Python module
echo "Step 1: Loading Python module..."
PYTHON_LOADED=false
for python_mod in python/3.9 python/3.8 python3/3.9 python3/3.8 Python/3.9 Python/3.8; do
    if try_module "$python_mod"; then
        PYTHON_LOADED=true
        break
    fi
done

# Find Python executable
echo "Step 2: Finding Python executable..."
PYTHON_CMD=""
for cmd in python3 python python3.9 python3.8; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON_CMD="$cmd"
        echo "✓ Found Python: $PYTHON_CMD ($($cmd --version 2>&1))"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "✗ ERROR: No Python found!"
    echo "Available commands:"
    which python python3 python3.8 python3.9 2>/dev/null || echo "None found"
    exit 1
fi

# Try to load Java module for NetLogo
echo "Step 3: Loading Java module..."
for java_mod in java/11 java/8 openjdk/11 openjdk/8 Java/11 Java/8; do
    if try_module "$java_mod"; then
        break
    fi
done

# Check Java
if command -v java &> /dev/null; then
    echo "✓ Java found: $(java -version 2>&1 | head -1)"
else
    echo "⚠ Java not found - NetLogo may not work"
fi

# Set environment
echo "Step 4: Setting environment..."
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export DISPLAY=""
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$HOME/.local/lib/python3.8/site-packages:$HOME/.local/lib/python3.7/site-packages:$PYTHONPATH"

# Install packages using different strategies
echo "Step 5: Installing packages..."

# Strategy 1: Use pip with --user and flexible requirements
if [ -f "requirements_cluster.txt" ]; then
    echo "Using flexible cluster requirements..."
    $PYTHON_CMD -m pip install --user -r requirements_cluster.txt
else
    echo "Installing packages individually..."
    # Install essential packages one by one
    for package in "numpy>=1.19" "pandas>=1.3" "matplotlib>=3.3" "tensorflow>=2.6" "pynetlogo>=0.4"; do
        echo "Installing $package..."
        $PYTHON_CMD -m pip install --user "$package" || echo "Failed to install $package"
    done
fi

# Test imports
echo "Step 6: Testing package imports..."
$PYTHON_CMD -c "
import sys
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')

# Test essential imports
packages = {
    'numpy': 'np',
    'pandas': 'pd', 
    'matplotlib': None,
    'tensorflow': 'tf',
    'pynetlogo': None
}

success_count = 0
for pkg, alias in packages.items():
    try:
        if alias:
            exec(f'import {pkg} as {alias}')
        else:
            exec(f'import {pkg}')
        print(f'✓ {pkg}')
        success_count += 1
    except Exception as e:
        print(f'✗ {pkg}: {e}')

print(f'\\nSuccessfully imported {success_count}/{len(packages)} packages')
if success_count < len(packages):
    print('Some packages failed - the simulation may not work properly')
"

# Check required files
echo "Step 7: Checking required files..."
for file in "dqn.py" "Sugarscape 2 Constant Growback.nlogo"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        echo "Directory contents:"
        ls -la
        exit 1
    fi
done

# Run simulation
echo "Step 8: Running DQN simulation..."
echo "Command: $PYTHON_CMD dqn.py --cluster-mode --job-id $SLURM_JOB_ID"
echo "==============================================="

$PYTHON_CMD dqn.py --cluster-mode --job-id "$SLURM_JOB_ID"
exit_code=$?

echo "==============================================="
echo "Job completed at $(date)"
echo "Exit code: $exit_code"
echo "==============================================="

exit $exit_code