#!/bin/bash

for j in 0.5
do
        k=1  # Initialize k to 1
	for i in 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05
	do
    	python evaluations.py --tr 0.6 --path datasets/ --dropout 0.1 --weight_decay 1e-4 --lr 1e-2 --epochs 200 --extrastr $j --dataset BlogCatalog \
	                   --normy 1 --gamma 0.75 --threshold $i --mode WithSN --count $k
	k=$((k + 1))  # Increment k by 1
        done
done
