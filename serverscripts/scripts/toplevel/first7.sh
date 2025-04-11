#!/bin/bash

spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

cd /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel

for i in $(seq 0 6); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/parallelToplevel.py $i $1 $2 $3 $4 > $OUT 2> $ERR" 
    fi
done | parallel -j 1