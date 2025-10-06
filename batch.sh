#!/bin/bash
#SBATCH --job-name=dqn_sugarscape
#SBATCH --output=dqn_sugarscape_%j.log
#SBATCH --error=dqn_sugarscape_err_%j.log
#SBATCH --nodes=1 # only use multiple if you are doing parallel computing e.g MPI
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12  # Change as you see fit               
#SBATCH --partition=stampede # 3 options stampede , bigbatch and biggpu
#SBATCH --time=3-00:00:00  # max time you want to give your program (this is 3 days)  

# Load required modules (adjust these based on your cluster's available modules)
module load python/3.9
module load java/11  # NetLogo requires Java
# module load cuda/11.8  # For TensorFlow GPU support if available

# Set up environment variables
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export DISPLAY=""  # Disable display for headless mode

# Define paths
WORK_DIR="$(pwd)"
RESULTS_DIR="$WORK_DIR/results"
JOB_NAME="dqn_sugarscape_${SLURM_JOB_ID}"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if virtual environment exists, create if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages if not already installed
echo "Installing/checking required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo "Verifying installations..."
python3 -c "import numpy, pandas, matplotlib, tensorflow, pynetlogo; print('All packages imported successfully')"

# Run the Python script
echo "Starting DQN run at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $WORK_DIR"

python3 dqn.py  # run file

# the comment below specifies how you would run a file with inputs
# python3 dqn.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --job_type "$JOB_NAME" --job_id "$SLURM_JOB_ID"

echo "Job done DQN run completed at $(date)"

