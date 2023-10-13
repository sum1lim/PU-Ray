#!/bin/bash

# Train on different experimentation setups
train_model --input-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --log puray_supervised --num-query 983040 --num-op 128
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_supervised --num-query 221184 --num-op 128

train_model --input-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/input_2048 --marching-steps 6 --log puray_train2test --num-query 245760 --num-op 128 --num-epochs 15
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_train2test --num-query 221184 --num-op 128

train_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --marching-steps 6 --log puray_test2test --num-query 55296 --num-op 128 --num-epochs 15
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_test2test --num-query 221184 --num-op 128 

train_model --input-dir ./data/PU-GAN/simple_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/simple_pointcloud/input_2048_4X/input_2048 --marching-steps 6 --log puray_simple2test --num-query 2048 --num-sample 1 --num-op 128 --num-epochs 30
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_simple2test --num-query 221184 --num-op 128 

train_model --input-dir ./data/PU-GAN/medium_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/medium_pointcloud/input_2048_4X/input_2048 --marching-steps 6 --log puray_medium2test --num-query 2048 --num-sample 1 --num-op 128 --num-epochs 30
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_medium2test --num-query 221184 --num-op 128 

train_model --input-dir ./data/PU-GAN/complex_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/complex_pointcloud/input_2048_4X/input_2048 --marching-steps 6 --log puray_complex2test --num-query 2048 --num-sample 1 --num-op 128 --num-epochs 30
test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps 6 --model puray_complex2test --num-query 221184 --num-op 128 
