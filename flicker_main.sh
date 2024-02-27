#!/bin/bash

for i in Flickr
do
	for j in 0.5 1 2
	do
    	python main.py --tr 0.6 --path datasets/ --dropout 0.1 --weight_decay 1e-4 --lr 5e-3 --epochs 150 --extrastr $j --dataset $i \
	                   --normy 1 --gamma 0.75 --mode WithSN
	done
done
