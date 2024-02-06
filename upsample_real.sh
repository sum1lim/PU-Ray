#!/bin/bash

for FILE in ./data/kitti_snippet/*; do
    pu_ray --input "$FILE" --r 4  --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti-snippet --real-scanned --min-dist 0 --fps --implicit-points
done

generate_gt --aoi ./data/kitti_test/resolution_h0.200_v0.400_occ --reference ./data/kitti_test/reference --min-dist 15000
rm ./log/kitti_puray.log
rm ./log/kitti_train2test.log
rm ./log/kitti_test2test.log
rm ./log/kitti_simple2test.log
rm ./log/kitti_medium2test.log
rm ./log/kitti_complext2test.log

for FILE in ./data/kitti_test/resolution_h0.200_v0.400_occ/*; do
    echo "******************** $(basename ${FILE%.*}) ********************"
    start=`date +%s.%N`
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_puray --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_puray/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_puray.log --wait  --min-dist 15000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_train2test --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_train2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_train2test.log --wait  --min-dist 15000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_test2test --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_test2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_test2test.log --wait  --min-dist 15000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_simple2test --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_simple2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_simple2test.log --wait  --min-dist 15000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_medium2test --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_medium2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_medium2test.log --wait  --min-dist 15000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir kitti_complext2test --real-scanned --min-dist 15000 --fps
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_complext2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_complext2test.log --wait  --min-dist 15000
done

generate_gt --aoi ./data/highway_test/resolution_h0.200_v0.400_occ --reference ./data/highway_test/reference --min-dist 30000
rm ./log/highway_puray.log
rm ./log/highway_train2test.log
rm ./log/highway_test2test.log
rm ./log/highway_simple2test.log
rm ./log/highway_medium2test.log
rm ./log/highway_complext2test.log

for FILE in ./data/highway_test/resolution_h0.200_v0.400_occ/*; do
    echo "******************** $(basename ${FILE%.*}) ********************"
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_puray --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_puray/"$(basename ${FILE%.*}).xyz" --log ./log/highway_puray.log --wait --min-dist 30000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_train2test --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_train2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_train2test.log --wait --min-dist 30000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_test2test --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_test2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_test2test.log --wait --min-dist 30000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_simple2test --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_simple2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_simple2test.log --wait --min-dist 30000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_medium2test --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_medium2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_medium2test.log --wait --min-dist 30000
    pu_ray --input "$FILE" --r 4  --batch-size 2500 --patch-k 64 --query-k 8 --marching-steps 6 --model puray_supervised --output-dir highway_complext2test --real-scanned --min-dist 30000 --fps
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_complext2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_complext2test.log --wait --min-dist 30000
done
