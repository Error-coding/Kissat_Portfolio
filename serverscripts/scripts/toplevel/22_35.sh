#!/bin/bash

for i in $(seq 21 34); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/parallelToplevel.py $i 1800 12 50 12 > $OUT 2> $ERR" 
    fi
done | tac | parallel -j 1
