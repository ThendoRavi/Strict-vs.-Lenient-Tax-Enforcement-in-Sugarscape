#!/bin/bash
#SBATCH --job-name=dqn_sugarscape
#SBATCH --output=dqn_sugarscape_%j.log
#SBATCH --error=dqn_sugarscape_err_%j.log
#SBATCH --nodes=1 # only use multiple if you are doing parallel computing e.g MPI
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12  # Change as you see fit               
#SBATCH --partition=bigbatch  # 3 options stampede , bigbatch and biggpu
#SBATCH --time=3-00:00:00  # max time you want to give your program (this is 3 days)  

# Define paths
# INPUT_FILE="../../mass_spec_data/EMPA/17092024_cardiac tissues/h5 files/Sample 23_SGLT2 only_1-8007_SN1p0_centroid.imzml"
# OUTPUT_DIR="../PresentationData"
# JOB_NAME="sglt2_only"h

# Set Java headless mode BEFORE Python starts
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=true -Xmx4g"
export DISPLAY=""
export MPLBACKEND="Agg"

# Verify environment
echo "Environment variables set:"
echo "  JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS"
echo "  DISPLAY=$DISPLAY"
echo "  MPLBACKEND=$MPLBACKEND"

# Run the Python script
echo "Starting DQN run at $(date)"

python -u main.py  # -u flag disables output buffering

# the comment below specifies how you would run a file with inputs
# python temp.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --job_type "$JOB_NAME" --job_id "$SLURM_JOB_ID"


echo "Job done DQN run completed at $(date)"

