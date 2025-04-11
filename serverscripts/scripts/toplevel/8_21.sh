#!/bin/bash

for i in $(seq 7 20); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/parallelToplevel.py $i 1800 20 50 20 > $OUT 2> $ERR" 
    fi
done | parallel -j 1
