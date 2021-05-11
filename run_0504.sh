#./tools/dist_train.sh configs/vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py 2
#python ./tools/train.py configs/vfnet/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_coco.py # 1
#./tools/dist_train.sh configs/vfnet/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_coco.py 8
./tools/dist_train.sh configs/vfnet/vfnetx_r2_101_fpn_59e_pcbv3_lr0001.py 8
#./tools/dist_train.sh configs/vfnet/vfnetx_r2_101_fpn_59e_pcbv3_scale.py 8
