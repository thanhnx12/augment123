# python train.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type full --epoch 8 --epoch_mem 6
# python train.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type current --epoch 8 --epoch_mem 6
# python train.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --SAM --SAM_type full --epoch 8 --epoch_mem 6
# python train.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type current --epoch 8 --epoch_mem 6
python train.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type current --epoch 8 --epoch_mem 6
python train.py --task_name FewRel --num_k 5 --num_gen 2 --SAM --SAM_type current --epoch 8 --epoch_mem 6
python train.py --task_name FewRel --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --num_k 5 --num_gen 2 --SAM --SAM_type current --epoch 8 --epoch_mem 6


# for t in FewRel
# do
#     for i in 8 10
#     do
#         for j in 6 10
#         do
#             for l in 0.25 0.5
#             do
#                 for m in 0.25 0.5
#                 do
#                     for r in 0.1 0.05
#                     do
#                         CUDA_VISIBLE_DEVICES=0 python train.py \
#                             --task_name $t \
#                             --num_k 5 \
#                             --num_gen 5 \
#                             --mixup \
#                             --mixup_loss_1 $l \
#                             --mixup_loss_2 $m \
#                             --rho $r \
#                             --SAM \
#                             --SAM_type full \
#                             --epoch $i \
#                             --epoch_mem $j
#                     done
                        
#                 done
#             done
#         done
#     done
# done
# python train.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --SAM --SAM_type full --epoch 8 --epoch_mem 6
