#!/bin/bash

rm ./log/kitti_low_res.log
rm ./log/kitti_high_res.log
rm ./log/kitti_high_res.log
rm ./log/kitti_puray.log
rm ./log/kitti_punet.log
rm ./log/kitti_mpu.log
rm ./log/kitti_pugan.log
rm ./log/kitti_pugcn.log
rm ./log/kitti_gradpu.log
rm ./log/kitti_train2test.log
rm ./log/kitti_test2test.log
rm ./log/kitti_simple2test.log
rm ./log/kitti_medium2test.log
rm ./log/kitti_complext2test.log

for FILE in ./data/kitti_test/resolution_h0.200_v0.400_occ/*; do
    echo "******************** $(basename ${FILE%.*}) ********************"
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./data/kitti_test/resolution_h0.200_v0.400_occ/"$(basename ${FILE%.*}).csv" --log ./log/kitti_low_res.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./data/kitti_test/resolution_h0.100_v0.200_occ/"$(basename ${FILE%.*}).csv" --log ./log/kitti_high_res.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_punet/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_punet.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_mpu/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_mpu.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_pugan/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_pugan.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_pugcn/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_pugcn.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_gradpu/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_gradpu.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_puray/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_puray.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_train2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_train2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_test2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_test2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_simple2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_simple2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_medium2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_medium2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/kitti_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/kitti_complext2test/"$(basename ${FILE%.*}).xyz" --log ./log/kitti_complext2test.log --wait  --min-dist 15000
done

rm ./log/highway_low_res.log
rm ./log/highway_high_res.log
rm ./log/highway_high_res.log
rm ./log/highway_puray.log
rm ./log/highway_punet.log
rm ./log/highway_mpu.log
rm ./log/highway_pugan.log
rm ./log/highway_pugcn.log
rm ./log/highway_gradpu.log
rm ./log/highway_train2test.log
rm ./log/highway_test2test.log
rm ./log/highway_simple2test.log
rm ./log/highway_medium2test.log
rm ./log/highway_complext2test.log

for FILE in ./data/highway_test/resolution_h0.200_v0.400_occ/*; do
    echo "******************** $(basename ${FILE%.*}) ********************"
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./data/highway_test/resolution_h0.200_v0.400_occ/"$(basename ${FILE%.*}).csv" --log ./log/highway_low_res.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./data/highway_test/resolution_h0.100_v0.200_occ/"$(basename ${FILE%.*}).csv" --log ./log/highway_high_res.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_punet/"$(basename ${FILE%.*}).xyz" --log ./log/highway_punet.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_mpu/"$(basename ${FILE%.*}).xyz" --log ./log/highway_mpu.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_pugan/"$(basename ${FILE%.*}).xyz" --log ./log/highway_pugan.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_pugcn/"$(basename ${FILE%.*}).xyz" --log ./log/highway_pugcn.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_gradpu/"$(basename ${FILE%.*}).xyz" --log ./log/highway_gradpu.log --wait --min-dist 30000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_puray/"$(basename ${FILE%.*}).xyz" --log ./log/highway_puray.log --wait  --min-dist 15000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_train2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_train2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_test2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_test2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_simple2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_simple2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_medium2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_medium2test.log --wait  --min-dist 15000
    evaluate --pc1 ./data/highway_test/resolution_h0.200_v0.400_occ_gt/"$(basename ${FILE%.*}).csv" --pc2 ./output/highway_complext2test/"$(basename ${FILE%.*}).xyz" --log ./log/highway_complext2test.log --wait  --min-dist 15000
done
