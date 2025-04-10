#!/bin/bash

for i in $(seq 0 34); do
	scp rzipperer@login.ae.iti.kit.edu:/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/defaultconf/naur/$i.out ./$i/default.out
done
