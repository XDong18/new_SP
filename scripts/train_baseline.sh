# Classwise semi-supervision (VOC categories)
# Train baseline Mask R-CNN
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_port=3000 \
#     shapeprop/tools/train_net.py \
#     --config-file configs/coco_voc_mask_rcnn_r50_fpn_1x.yml
# # Evaluate
#python shapeprop/tools/evaluate.py runs/coco_voc_mask_rcnn_r50_fpn_1x/inference/coco_2017_val/segm.json


# Classwise semi-supervision (VOC categories)
# Train baseline Mask R-CNN
export CUDA_VISIBLE_DEVICES=5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=3000 \
    shapeprop/tools/train_net.py \
    --config-file configs/bdd_yanzhao_r50.yml \
    --local_rank 1 \
    --num_gpu 3
# Evaluate
