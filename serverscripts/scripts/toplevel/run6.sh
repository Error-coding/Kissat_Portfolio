#!/bin/bash

spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

cd /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel


OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/6.out"
ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/6.err"

python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/parallelToplevel.py 6 1800 32 50 32 > $OUT 2> $ERR
