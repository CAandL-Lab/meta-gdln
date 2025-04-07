#!/bin/bash

num_runs=10
for (( run_idx=0; run_idx<=$num_runs; run_idx++ ))
do
	python GDLN_uniform.py $run_idx
        python GDLN_binomial.py $run_idx
	python GDLN_habit.py $run_idx
done
