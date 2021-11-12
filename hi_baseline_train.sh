#!/bin/sh
### The following command will set a user-friendly job name
#PBS -N vqa_proj
### The following command will set a user-friendly project name which is your department code by default
#PBS -P LTRANS
### The following command, specifies when email notifications can be sent: bea stand for the following : b- at begining of job execution, e-at the end of the job and a- when the job is aborted.
#PBS -m bea
### Specify your email address for notifications.
#PBS -M adichand20@gmail.com
#### This command will specify the resource list
#PBS –l select=1:mem=64G:ncpus=6:ngpus=1
#PBS -l walltime=025:00:00
#PBS -j oe
#$PBS_O_WORKDIR="/home3/181ee103/"
echo "==============================="
echo $PBS_JOBID
#cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
export WANDB_API_KEY=
#job
#singularity exec --nv /home3/181ee103/vqa_image python3 train_hi.py
unset WANDB_API_KEY