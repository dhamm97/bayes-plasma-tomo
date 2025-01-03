#!/bin/bash

for i in $(seq 0 9);
do
    python ula_iteration_number_tuning.py $((i * 10 + 900)) $(((i+1) * 10 + 900)) &
done
