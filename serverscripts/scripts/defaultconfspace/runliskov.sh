#!/bin/bash

spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

# Run i from 0-6 with 32 threads
for i in $(seq 0 6); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/defaultconfspace/defaultparallel.py $i 1800 32 0 32 > $OUT 2> $ERR"
    fi
done | parallel -j 1

# Run i from 7-20 with 20 threads in parallel
for i in $(seq 7 20); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/defaultconfspace/defaultparallel.py $i 1800 20 0 20 > $OUT 2> $ERR"
    fi
done | parallel -j 1 &

# Run i from 21-34 in reverse with 12 threads in parallel
for i in $(seq 34 -1 21); do
    OUT="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.out"
    ERR="/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/liskov/$i.err"

    if [ ! -f $OUT ]; then
        echo "python3 /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/defaultconfspace/defaultparallel.py $i 1800 12 0 12 > $OUT 2> $ERR"
    fi
done | parallel -j 1 &

wait