#!/bin/bash

for i in BlogCatalog  # set the dataset name
do
	for j in 0.5    # set the imbalance k value
	do
    	python main.py --tr 0.6 --path datasets/ --dropout 0.1 --weight_decay 1e-4 --lr 1e-2 --epochs 200 --extrastr $j --dataset $i \
	                   --normy 1 --gamma 0.75 --mode WithSN
	done
done
