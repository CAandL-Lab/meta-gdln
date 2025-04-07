#!/bin/bash

cd hier
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses train_norms plots
rm -rf __pycache__
cd ..

cd hier_indiv
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses plots
rm -rf __pycache__
cd ..

cd hier_loss_knows_gates
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses train_norms plots
rm -rf __pycache__
cd ..

cd hier_indiv_loss_knows_gates
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses plots
rm -rf __pycache__
cd ..

cd sys
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates non_losses test_norms losses sys_losses train_norms plots
rm -rf __pycache__
cd ..

cd sys_loss_knows_gates
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates non_losses test_norms losses sys_losses train_norms plots
rm -rf __pycache__
cd ..

cd sys_indiv
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses sys_losses non_losses plots
rm -rf __pycache__
cd ..

cd sys_indiv_loss_knows_gates
rm *.pdf
rm *.png
rm -rf n_runs
rm -rf end_gates losses sys_losses non_losses plots
rm -rf __pycache__
cd ..
