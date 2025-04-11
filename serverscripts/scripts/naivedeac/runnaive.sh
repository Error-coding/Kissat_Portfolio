#!/bin/bash

spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

cd /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel

for i in $(seq 0 34); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/naive/naur/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/naive/naur/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/naivedeac/naive.py $i 1000 16 16  > $OUT 2> $ERR" 
    fi
done | parallel -j 1