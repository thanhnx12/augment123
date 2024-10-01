for t in FewRel tacred
do
    for i in 5
    do
        for j in 10
        do
            for k in 10
            do
                for l in 0.5 0.75 1
                do
                    for m in 0.5 0.75 1
                    do
                        for n in 0.05 0.1 0.01
                        do
                            CUDA_VISIBLE_DEVICES=1 python main-mmi.py \
                                --task $t \
                                --step1_epochs $i \
                                --step2_epochs $j \
                                --step3_epochs $k \
                                --loss1_factor $l \
                                --loss2_factor $m \
                                --mixup \
                                --SAM \
                                --rho $n \
                                --SAM_type full
                        done
                    done
                done
            done
        done
    done
done