# Ablation Study

for seed in 2 4 8 16 32 64 128 256; do
    echo "********************** Random seed: $seed **********************" 
    for i in 2048 4096 8192 16384; do
        for j in 0 2 4 6 8; do
            train_model --input-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/train_pointcloud/input_2048_4X/gt_8192 --marching-steps $j --log ablation_"$i"queries_"$j"  --num-query "$i" --seed "$seed" --num-op 128
            test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$j" --model ablation_"$i"queries_"$j" --num-query 221184 --verbose --num-op 128
            rm ./models/ablation_"$i"queries_"$j.pt"
            rm ./log/ablation_"$i"queries_"$j.log"
        done
    done
done

for step in 4 6 8; do
    echo "********************** Num marching steps: $step **********************" 
    test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$step" --model wo_raymarching_loss --num-query 221184 --verbose --num-op 128
    test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$step" --model wo_epsilon_loss --num-query 221184 --verbose --num-op 128
    test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$step" --model wo_tangent_loss --num-query 221184 --verbose --num-op 128
    test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$step" --model wo_ms_loss --num-query 221184 --verbose --num-op 128
    test_model --input-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048 --query-dir ./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192 --marching-steps "$step" --model puray_supervised --num-query 221184 --verbose --num-op 128
done
