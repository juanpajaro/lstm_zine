#!/bin/bash
#SBATCH --job-name=red_lstm    ## Name of the job
#SBATCH -p debug
#SBATCH -o logs/resultado_%j.out      ## Output file
#SBATCH -e logs/error_%j.err
#SBATCH --time=10:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      ## The number of threads the code will use
#SBATCH --mem-per-cpu=100M     ## Real memory(MB) per CPU required by the job.

## Load the python interpreter
## module load python

#**********************************************************
# Creamos variables de entorno
export PATH="/apps/bpike/miniforge3/bin:$PATH"
echo "creamos variable entorno para CONDS"

# Inciamos CONDA
conda init
echo " "
echo "Iniciamos CONDA"
# Cargamos el nuevo archivo bashrc
source /zine/HPC0251/miniconda3/etc/profile.d/conda.sh
echo "Reiniciamos sistema equivalente al /.bashrc"
echo " "
# Configuramos Conda para no iniciar con env=base
#conda config --set auto_activate_base

# Activamos el entorno con CONDA
conda activate pajaroenv
echo "**********************************************"
echo "Activamos el ambiente:"
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "**********************************************"
echo " "
#ejecutar
echo "**********************************************"
echo "corriendo programa en python..."
echo "**********************************************"
#*********
python3 entrenamiento.py
