for t in FewRel
do
    for i in 5 10
    do
        for j in 10 5
        do
            for k in 10 5
            do
                for l in 0.5 0.75 1
                do
                    for m in 0.5 0.75 1
                    do
                        CUDA_VISIBLE_DEVICES=0 python main-mmi.py \
                            --task $t \
                            --step1-epochs $i \
                            --step2-epochs $j \
                            --step3-epochs $k \
                            --loss1-factor $l \
                            --loss2-factor $m
                    done
                done
            done
        done
    done
done