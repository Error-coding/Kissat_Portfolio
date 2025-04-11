#!/bin/bash
spack env activate smac3
source /nfs/home/rzipperer/.bashrc
conda activate SMAC

comm=$"sh /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/8_21.sh
sh /nfs/home/rzipperer/git/Kissat_hyperparamoptimization/scripts/toplevel/22_35.sh"
echo "$comm" | parallel -j 2