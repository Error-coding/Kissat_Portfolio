#!/bin/bash

for i in $(seq 0 34); do
	tail -1 $i/default.out | cut --delimiter=' ' -f3
done > ./defaults.txt
