#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
  python apg_training.py --run_index=i --num_sweeps=5 --grad_use=full
done