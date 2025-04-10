#!/bin/bash

rm sorted/*
rm raw/*
rm ordered/*

for i in $(seq 0 27); do
	scp rzipperer@login.ae.iti.kit.edu:/nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scriptout/naive/naur/$i.out ./raw/$i.out
	cat ./raw/$i.out | grep score | sort -n -t' ' -k5,5 | awk '{print $1, $NF}' | cut -c 5- > sorted/$i.out
	python3 csvify.py $i
	python3 makesolved.py $i
done
