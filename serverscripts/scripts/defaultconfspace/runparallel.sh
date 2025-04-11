#!/bin/sh
spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC
cd /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/parallel

OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/blum/parallel/$1.out"
ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/blum/parallel/$1.err"

python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/defaultconfspace/defaultparallel.py $1 $2 $3 $4 $5 > $OUT 2> $ERR