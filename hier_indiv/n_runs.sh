#!/bin/bash

num_runs=5
echo $num_runs
for (( run_idx=0; run_idx<=$num_runs; run_idx++ ))
do
	python GDLN.py $run_idx
done
