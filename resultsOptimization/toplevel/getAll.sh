#!/bin/bash

rm */*

for i in $(seq 0 34); do
	scp rzipperer@login.ae.iti.kit.edu:/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/outputs/toplevel/liskov/$i/*/*/* ./$i/
	scp rzipperer@login.ae.iti.kit.edu:/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/toplevel/liskov/$i.out ./$i/log.out
	python3 csvify.py $i
done
