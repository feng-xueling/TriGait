
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 opengait/Heatmap.py --cfgs ./configs/trigait/trigait_casiab.yaml --phase train
#CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/trigait/trigait_othersil.yaml --phase train
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/trigait/trigait_casiab.yaml --phase train