#!/bin/bash

spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/extend/liskov/17.out"
ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/extend/liskov/17.err"
python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/extendedconfspace/extended_optimization.py 17 1800 200 32 > $OUT 2> $ERR
