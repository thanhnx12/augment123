CUDA_VISIBLE_DEVICES=0 python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --SAM --SAM_type current --epoch 10 --epoch_mem 10
CUDA_VISIBLE_DEVICES=0 python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.5 --mixup_loss_2 0.25 --SAM --SAM_type current --epoch 10 --epoch_mem 10
