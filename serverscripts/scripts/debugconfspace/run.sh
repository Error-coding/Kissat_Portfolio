#!/bin/sh
spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/debug1/$1.out"
ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/debug1/$1.err"

python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/debugconfspace/kissatparallel.py $1 $2 $3 $4 $5 > $OUT 2> $ERR