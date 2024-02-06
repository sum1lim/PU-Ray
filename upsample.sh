#!/bin/bash

for FILE in ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/*; do
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_supervised --num-op 128 --output-dir 4x_supervised --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_train2test --num-op 128 --output-dir 4x_train2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_test2test --num-op 128 --output-dir 4x_test2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_simple2test --num-op 128 --output-dir 4x_simple2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_medium2test --num-op 128 --output-dir 4x_medium2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_complex2test --num-op 128 --output-dir 4x_complex2test --fps --implicit-points
done

for FILE in ./data/PU-GAN/test_pointcloud/input_2048_4X_noise_0.01/input_2048/*; do
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_supervised --num-op 128 --output-dir noise1_supervised --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_train2test --num-op 128 --output-dir noise1_train2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_test2test --num-op 128 --output-dir noise1_test2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_simple2test --num-op 128 --output-dir noise1_simple2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_medium2test --num-op 128 --output-dir noise1_medium2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_complex2test --num-op 128 --output-dir noise1_complex2test --fps --implicit-points
done

for FILE in ./data/PU-GAN/test_pointcloud/input_2048_4X_noise_0.02/input_2048/*; do
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_supervised --num-op 128 --output-dir noise2_supervised --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_train2test --num-op 128 --output-dir noise2_train2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_test2test --num-op 128 --output-dir noise2_test2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_simple2test --num-op 128 --output-dir noise2_simple2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_medium2test --num-op 128 --output-dir noise2_medium2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 10 --marching-steps 6 --model puray_complex2test --num-op 128 --output-dir noise2_complex2test --fps --implicit-points
done

for FILE in ./data/PU-GAN/test_pointcloud/input_2048_16X/input_2048/*; do
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_supervised --num-op 128 --output-dir 16x_supervised --fps --implicit-points
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_train2test --num-op 128 --output-dir 16x_train2test --fps --implicit-points
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_test2test --num-op 128 --output-dir 16x_test2test --fps --implicit-points
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_simple2test --num-op 128 --output-dir 16x_simple2test --fps --implicit-points
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_medium2test --num-op 128 --output-dir 16x_medium2test --fps --implicit-points
    pu_ray --input "$FILE" --r 16 --query-k 10 --marching-steps 6 --model puray_complex2test --num-op 128 --output-dir 16x_complex2test --fps --implicit-points
done

for FILE in ./data/PU1K/test/input_2048/input_2048/*; do
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_supervised --num-op 128 --output-dir pu1k_supervised --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_train2test --num-op 128 --output-dir pu1k_train2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_test2test --num-op 128 --output-dir pu1k_test2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_simple2test --num-op 128 --output-dir pu1k_simple2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_medium2test --num-op 128 --output-dir pu1k_medium2test --fps --implicit-points
    pu_ray --input "$FILE" --r 4 --query-k 8 --marching-steps 6 --model puray_complex2test --num-op 128 --output-dir pu1k_complex2test --fps --implicit-points
done
