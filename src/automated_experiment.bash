#!/bin/bash

methods=(
    'IDDPG'
    'SNDDPG'
    'SEDDPG'
    'FLDDPG'
)


for ((i=0;i<3;i+=1))
do
    for method in ${methods[*]}
    do
        python experiment.py \
        --mode='learn' \
        $method \
        --seed $(expr $i + 100)
    done
done
