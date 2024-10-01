for t in tacred FewRel
do
    for i in 10
    do
        for j in 5
        do
            for m in 5
            do
                for k in 0.1
                do
                    for l in 1e-5
                    do
                        CUDA_VISIBLE_DEVICES=1 python train-ignore-wandb.py \
                            --task_name $t \
                            --epoch $i \
                            --epoch_mem $j \
                            --num_gen $m \
                            --lr $l \
                            --rho $k \
                            --SAM \
                            --mixup_loss_1 0.25 \
                            --mixup_loss_2 0.25 \
                            --mixup
                    done
                done
            done
        done
    done
done

