python -m torch.distributed.launch --nproc_per_node 8 main_simmim.py --cfg configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml --data-path /work/usr/lei.yang/ImageNet_ILSVRC2012/train --batch-size 128 --output checkpoints/