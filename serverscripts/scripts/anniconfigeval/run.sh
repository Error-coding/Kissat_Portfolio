#!/bin/sh
spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconfanni/liskov/anni9.out"
ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconfanni/liskov/anni9.err"

python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/defaultconfanni/parallelanni.py  > $OUT 2> $ERR