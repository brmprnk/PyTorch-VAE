#!/bin/bash


betas=(0 0.001 0.01 0.1 1 2 5 10 20)
h_dims=("None" "2091 1394" "1394")


for beta in "${betas[@]}"; do
    for h_dim in "${h_dims[@]}"; do
      python run.py -c 'configs/lindel.yaml' --beta "$beta" --h_dim "$h_dim" -s --umap
    done
done
