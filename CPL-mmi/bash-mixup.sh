for t in FewRel
do
    for i in 10
    do
        for j in 5
        do
            for m in 2
            do
                for k in 0.25 0.5
                do
                    for l in 0.25 0.5
                    do
                        for p in 0.05 0.1
                        do
                            CUDA_VISIBLE_DEVICES=0 python train-ignore-wandb.py \
                                --task_name $t \
                                --epoch $i \
                                --epoch_mem $j \
                                --num_gen $m \
                                --mixup_loss_1 $k \
                                --mixup_loss_2 $l \
                                --mixup \
                                --rho $p
                        done
                    done
                done
            done
        done
    done
done

