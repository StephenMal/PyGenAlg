#!/bin/bash
#SBATCH -o slurm_out.txt
#SBATCH -e slurm_err.txt
#SBATCH --time=120:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out


export file="run_bga.py"

mkdir "slurm/$SLURM_JOB_ID"
export debug_logs="slurm/$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="slurm/$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

echo "conda part"

source /home/${USER}/.bashrc
source activate pga_adhd_runs

echo "Made it passed the conda part"

## Enter Working Directory ##
cd $SLURM_SUBMIT_DIR

## Module debugging ##
module list >> $debug_logs
which mpirun >> $debug_logs

date >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Run job ##
python $file $args
sleep 3

date >> $benchmark_logs
echo "ulimit -l" >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv job.$SLURM_JOB_ID.err "slurm/$SLURM_JOB_ID/"
mv job.$SLURM_JOB_ID.out "slurm/$SLURM_JOB_ID/"
