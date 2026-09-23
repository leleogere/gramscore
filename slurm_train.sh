#!/bin/bash
#SBATCH --output=/linkhome/rech/genfri01/ujt89sd/gramscore/jobs/%j/gramscore%j.out   # fichier de sortie (%j = job ID)
#SBATCH --error=/linkhome/rech/genfri01/ujt89sd/gramscore/jobs/%j/gramscore%j.err    # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100           # demander des GPU V100 16 Go
#SBATCH --nodes=1                   # reserver x nœuds
#SBATCH --ntasks=1                  # reserver x taches (ou processus)
#SBATCH --gres=gpu:1                # reserver x GPU par noeud
#SBATCH --cpus-per-task=20          # reserver x CPU par tache (et memoire associee)
#SBATCH --time=100:00:00            # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu_h100-t4       # QoS dev=2h / t3=20h / t4=100h
#SBATCH --hint=nomultithread        # desactiver l’hyperthreading
#SBATCH --account=cce@h100          # comptabilite V100
#SBATCH --mail-user=a2btkjumgd@pomail.net
#SBATCH --mail-type=ALL

module purge      # nettoyer les modules herites par defaut
#conda deactivate  # desactiver les environnements herites par defaut

module load arch/h100              # selectionner les modules compiles pour les H100
module load pytorch-gpu/py3/2.8.0  # charger les modules

set -x  # activer l’echo des commandes

export DATA_DIR="$SCRATCH/datasets/music"
export WANDB_MODE="offline"
export HYDRA_FULL_ERROR=1
export TMPDIR=$JOBSCRATCH
export WANDB_ENTITY=leleogere
export WANDB_PROJECT=gramscore_tokens
export PYTHONPATH="$HOME/gramscore:$HOME/gramscore/src:"
export WANDB_COMMUNICATION_DIR="$SCRATCH/.wandb_communication_dir"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"  # See https://github.com/wandb/wandb/issues/1193
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb-data"

if [ -z "$SLURM_JOB_NAME" ]; then
    echo "Error: job name (hence experiment name) not set."
    echo "Usage: sbatch -J $0 <experiment_name>"
    exit 1
fi
experiment="$SLURM_JOB_NAME"

experiment_path="$HOME/gramscore/configs/experiment/$experiment.yaml"
if ! [ -f "$experiment_path" ]; then
    echo "Error: experiment '$experiment' does not seem to exist at '$experiment_path'."
    exit 1
fi

srun python $HOME/gramscore/src/train.py --config-name common_config paths=jz trainer=gpu +experiment=$experiment +slurm_id=$SLURM_JOB_ID
