python -m torch.distributed.launch --nproc_per_node 8 main_simmim.py --cfg configs/swin_base__800ep/simmim_pretrain__swin_base__img224_window7__800ep.yaml --data-path /work/usr/lei.yang/ImageNet_ILSVRC2012/train --batch-size 64 --output checkpoints/