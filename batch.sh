#!/bin/bash
#SBATCH --job-name=template
#SBATCH --output=template_%j.log
#SBATCH --error=template__err_%j.log
#SBATCH --nodes=1 # only use multiple if you are doing parralel computing e.g MPI
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12  # Change as you see fit               
#SBATCH --partition=bigbatch # 3 options stampede , bigbatch and biggpu
#SBATCH --time=3-00:00:00  # max time you want to give your program (thius is 3 days)  

# Define paths
# INPUT_FILE="../../mass_spec_data/EMPA/17092024_cardiac tissues/h5 files/Sample 23_SGLT2 only_1-8007_SN1p0_centroid.imzml"
# OUTPUT_DIR="../PresentationData"
# JOB_NAME="sglt2_only"h

# Run the Python script
echo "Starting DA analysis at $(date)"

python temp.py  # run file

# the comment below specifies how you would run a file with inputs
# python temp.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" --job_type "$JOB_NAME" --job_id "$SLURM_JOB_ID"


echo "Job done analysis completed at $(date)"

