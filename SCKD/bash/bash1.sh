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
                        CUDA_VISIBLE_DEVICES=0 python main.py \
                            --task $t \
                            --step1-epochs $i \
                            --step2-epochs $j \
                            --step3-epochs $k \
                            --loss1-factor $l \
                            --loss2-factor $m \
                            --mixup \
                            --SAM \
                            --SAM_type current
                    done
                done
            done
        done
    done
done