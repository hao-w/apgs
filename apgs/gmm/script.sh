#!/bin/sh
for i in 1 2 3 4 5
do
  echo $i
  python apg_training.py --run_index=$i --num_sweeps=5 --grad_use=full
done