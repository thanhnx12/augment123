python main.py --task tacred --shot 5 --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main.py --task tacred --shot 5 --mixup --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main.py --task FewRel --shot 5 --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main.py --task FewRel --shot 5 --mixup --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10
python main-mmi.py --task tacred --shot 5 --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main-mmi.py --task tacred --shot 5 --mixup --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main-mmi.py --task FewRel --shot 5 --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 
python main-mmi.py --task FewRel --shot 5 --mixup --SAM --SAM_type current --step1_epochs 5 --step2_epochs 10 --step3_epochs 10


# python main-mmi.py --task tacred --shot 5 --mixup --step1_epochs 5 --step2_epochs 10 --step3_epochs 10 >> log.txt