#!/bin/bash

for FILE in ./data/HDL_dataset_te/resolution_0.200_occ/*; do
    pu_ray --input "$FILE" --r 4  --patch-k 16 --query-k 32 --marching-steps 6 --model puray_supervised --output-dir resolution_0.20_occ --real-scanned --fps
done

