python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --epoch 8 --epoch_mem 6
python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --epoch 8 --epoch_mem 6
python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type full --epoch 8 --epoch_mem 6
python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --SAM --SAM_type full --epoch 8 --epoch_mem 6
python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --SAM --SAM_type current --epoch 8 --epoch_mem 6
python train_llm.py --task_name Tacred --num_k 5 --num_gen 5 --mixup --mixup_loss_1 0.25 --mixup_loss_2 0.25 --SAM --SAM_type current --epoch 8 --epoch_mem 6