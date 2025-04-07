#!/bin/bash

n_runs=5

#if [ ! -d "hier/n_runs" ]; then
#  mkdir hier/n_runs
#fi
#if [ ! -d "hier/end_gates" ]; then
#  mkdir hier/end_gates
#fi
#if [ ! -d "hier/losses" ]; then
#  mkdir hier/losses
#fi
#if [ ! -d "hier/train_norms" ]; then
#  mkdir hier/train_norms
#fi
#if [ ! -d "hier/plots" ]; then
#  mkdir hier/plots
#fi
#cd hier
#for k in $(seq 0 $n_runs);
#do
#  python GDLN_uniform.py $k
#  python GDLN_binomial.py $k
#  python GDLN_habit.py $k
#done
#python replot.py
#cd ..
#
#if [ ! -d "hier_indiv/n_runs" ]; then
#  mkdir hier_indiv/n_runs
#fi
#if [ ! -d "hier_indiv/end_gates" ]; then
#  mkdir hier_indiv/end_gates
#fi
#if [ ! -d "hier_indiv/losses" ]; then
#  mkdir hier_indiv/losses
#fi
#if [ ! -d "hier_indiv/plots" ]; then
#  mkdir hier_indiv/plots
#fi
#cd hier_indiv
#for k in $(seq 0 $n_runs);
#do
#  python GDLN_uniform.py $k
#  python GDLN_binomial.py $k
#  python GDLN_habit.py $k
#done
#python replot.py
#cd ..
#
#if [ ! -d "hier_loss_knows_gates/n_runs" ]; then
#  mkdir hier_loss_knows_gates/n_runs
#fi
#if [ ! -d "hier_loss_knows_gates/end_gates" ]; then
#  mkdir hier_loss_knows_gates/end_gates
#fi
#if [ ! -d "hier_loss_knows_gates/losses" ]; then
#  mkdir hier_loss_knows_gates/losses
#fi
#if [ ! -d "hier_loss_knows_gates/train_norms" ]; then
#  mkdir hier_loss_knows_gates/train_norms
#fi
#if [ ! -d "hier_loss_knows_gates/plots" ]; then
#  mkdir hier_loss_knows_gates/plots
#fi
#cd hier_loss_knows_gates
#for k in $(seq 0 $n_runs);
#do
#  python GDLN_uniform.py $k
#  python GDLN_binomial.py $k
#  python GDLN_habit.py $k
#done 
#python replot.py
#cd ..
#
#if [ ! -d "hier_indiv_loss_knows_gates/n_runs" ]; then
#  mkdir hier_indiv_loss_knows_gates/n_runs
#fi
#if [ ! -d "hier_indiv_loss_knows_gates/end_gates" ]; then
#  mkdir hier_indiv_loss_knows_gates/end_gates
#fi
#if [ ! -d "hier_indiv_loss_knows_gates/losses" ]; then
#  mkdir hier_indiv_loss_knows_gates/losses
#fi
#if [ ! -d "hier_indiv_loss_knows_gates/plots" ]; then
#  mkdir hier_indiv_loss_knows_gates/plots
#fi
#cd hier_indiv_loss_knows_gates
#for k in $(seq 0 $n_runs);
#do
#  python GDLN_uniform.py $k
#  python GDLN_binomial.py $k
#  python GDLN_habit.py $k
#done
#python replot.py
#cd ..

if [ ! -d "sys/n_runs" ]; then
  mkdir sys/n_runs
fi
if [ ! -d "sys/end_gates" ]; then
  mkdir sys/end_gates
fi
if [ ! -d "sys/losses" ]; then
  mkdir sys/losses
fi
if [ ! -d "sys/sys_losses" ]; then
  mkdir sys/sys_losses
fi
if [ ! -d "sys/non_losses" ]; then
  mkdir sys/non_losses
fi
if [ ! -d "sys/train_norms" ]; then
  mkdir sys/train_norms
fi
if [ ! -d "sys/test_norms" ]; then
  mkdir sys/test_norms
fi
if [ ! -d "sys/plots" ]; then
  mkdir sys/plots
fi
cd sys
for k in $(seq 0 $n_runs);
do
  python GDLN_uniform.py $k
  python GDLN_binomial.py $k
  python GDLN_habit.py $k
  python GDLN_baseline_uniform.py $k
  python GDLN_baseline_binomial.py $k
  python GDLN_baseline_habit.py $k
done
python replot.py
cd ..

if [ ! -d "sys_loss_knows_gates/n_runs" ]; then
  mkdir sys_loss_knows_gates/n_runs
fi
if [ ! -d "sys_loss_knows_gates/end_gates" ]; then
  mkdir sys_loss_knows_gates/end_gates
fi
if [ ! -d "sys_loss_knows_gates/losses" ]; then
  mkdir sys_loss_knows_gates/losses
fi
if [ ! -d "sys_loss_knows_gates/sys_losses" ]; then
  mkdir sys_loss_knows_gates/sys_losses
fi
if [ ! -d "sys_loss_knows_gates/non_losses" ]; then
  mkdir sys_loss_knows_gates/non_losses
fi
if [ ! -d "sys_loss_knows_gates/train_norms" ]; then
  mkdir sys_loss_knows_gates/train_norms
fi
if [ ! -d "sys_loss_knows_gates/test_norms" ]; then
  mkdir sys_loss_knows_gates/test_norms
fi
if [ ! -d "sys_loss_knows_gates/plots" ]; then
  mkdir sys_loss_knows_gates/plots
fi
cd sys_loss_knows_gates
for k in $(seq 0 $n_runs);
do
  python GDLN_uniform.py $k
  python GDLN_binomial.py $k
  python GDLN_habit.py $k
  python GDLN_baseline_uniform.py $k
  python GDLN_baseline_binomial.py $k
  python GDLN_baseline_habit.py $k
done
python replot.py
cd ..

if [ ! -d "sys_indiv/n_runs" ]; then
  mkdir sys_indiv/n_runs
fi
if [ ! -d "sys_indiv/end_gates" ]; then
  mkdir sys_indiv/end_gates
fi
if [ ! -d "sys_indiv/losses" ]; then
  mkdir sys_indiv/losses
fi
if [ ! -d "sys_indiv/sys_losses" ]; then
  mkdir sys_indiv/sys_losses
fi
if [ ! -d "sys_indiv/non_losses" ]; then
  mkdir sys_indiv/non_losses
fi
if [ ! -d "sys_indiv/plots" ]; then
  mkdir sys_indiv/plots
fi
cd sys_indiv
for k in $(seq 0 $n_runs);
do
  python GDLN_uniform.py $k
  python GDLN_binomial.py $k
  python GDLN_habit.py $k
  python GDLN_baseline_uniform.py $k
  python GDLN_baseline_binomial.py $k
  python GDLN_baseline_habit.py $k
done
python replot.py
cd ..

if [ ! -d "sys_indiv_loss_knows_gates/n_runs" ]; then
  mkdir sys_indiv_loss_knows_gates/n_runs
fi
if [ ! -d "sys_indiv_loss_knows_gates/end_gates" ]; then
  mkdir sys_indiv_loss_knows_gates/end_gates
fi
if [ ! -d "sys_indiv_loss_knows_gates/losses" ]; then
  mkdir sys_indiv_loss_knows_gates/losses
fi
if [ ! -d "sys_indiv_loss_knows_gates/sys_losses" ]; then
  mkdir sys_indiv_loss_knows_gates/sys_losses
fi
if [ ! -d "sys_indiv_loss_knows_gates/non_losses" ]; then
  mkdir sys_indiv_loss_knows_gates/non_losses
fi
if [ ! -d "sys_indiv_loss_knows_gates/plots" ]; then
  mkdir sys_indiv/plots
fi
cd sys_indiv_loss_knows_gates
for k in $(seq 0 $n_runs);
do
  python GDLN_uniform.py $k
  python GDLN_binomial.py $k
  python GDLN_habit.py $k
  python GDLN_baseline_uniform.py $k
  python GDLN_baseline_binomial.py $k
  python GDLN_baseline_habit.py $k
done
python replot.py
cd ..
