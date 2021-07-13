#python train.py --nlayer 18 --name 201230_newdata_mixup_lamb_prob1 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --augment mixup

python train_mask.py --name 0713_mask --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --cutoff 284 --arch t2t
#python train.py --name paper-0506 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --cutoff 284 --arch t2t



# Valid
#python valid.py --gpu-num 0 --batch-size 64 --augment None --mask --vis --arch mask_resnet --cutoff 284
