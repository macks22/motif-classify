#!/bin/bash

#for i in 1 2 3 4 5
for i in 1 2
do
    name="dataset${i}"
    python grid_search.py \
        -tr "data/${name}/train.txt" \
        -te "data/${name}/test.txt" \
        -o "outputs/${name}.csv" 2>&1 | tee "outputs/${name}.log"
done
