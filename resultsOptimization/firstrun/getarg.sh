#!/bin/bash

for i in $(seq 0 34); do
	a=$(($i + 1))
	head -n $a ../../git/Kissat_hyperparamoptimization/instances_families.txt | tail -1
	grep -i $1 ./$i/log.out
done

