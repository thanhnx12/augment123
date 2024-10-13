for t in FewRel
do
    for i in 10 15
    do
        for j in 10 5
        do
            for k in 10 5
            do
                for l in 0.5 0.75
                do
                    for m in 0.5 0.75
                    do
                        CUDA_VISIBLE_DEVICES=0 python main-mmi.py \
                            --task $t \
                            --step1_epochs $i \
                            --step2_epochs $j \
                            --step3_epochs $k \
                            --loss1_factor $l \
                            --loss2_factor $m \
                            --mixup \
                            --SAM \
                            --SAM_type current
                    done
                done
            done
        done
    done
done