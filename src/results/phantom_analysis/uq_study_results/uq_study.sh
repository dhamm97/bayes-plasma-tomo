#!/bin/bash

for i in $(seq 0 9);
do
    python uq_study.py $((i * 90)) $(((i+1) * 90))  &
done
