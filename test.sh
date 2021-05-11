python demo/eval.py /work/u5216579/ctr/data/PCB_v3/LKE0007357_2302_4502_2352_4552_1.jpg configs/vfnet/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_coco.py /home/u5216579/vf/work_dirs/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_pcbv3_f+b/epoch_10.pth --score-thr 0.1 --read_img path --exp_id pcb_base_ep4

#python demo/image_demo.py /work/u5216579/ctr/data/PCB_v3/LKE0007357_2302_4502_2352_4552_1.jpg configs/vfnet/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_coco.py work_dirs/vfnetx_r2_101_fpn_mdconv_c3-c5_mstrain_59e_coco/best_bbox_mAP.pth --score-thr 0.2 --read_img path
