#!/bin/sh
#PBS -N vqa_proj_ta
#PBS -P vqa_ta
#PBS -m bea
#PBS -M adichand20@gmail.com
#PBS -l select=1:ncpus=6:ngpus=1:host=dgx
#PBS -l walltime=25:00:00
#PBS -j oe

$PBS_O_WORKDIR="/home3/181ee103/"
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

singularity exec --nv /home3/181ee103/vqa_image python3 test_ta.py