#python train.py --nlayer 18 --name 201230_newdata_mixup_lamb_prob1 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --augment mixup

#python train.py --nlayer 18 --name 210105_follow --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --multi-gpu --normalize --gaussian
#python valid.py --multi-gpu 0 --gpu-num 1 --nlayer 18



# Train
#python train.py --nlayer 18 --name 210117_aug --batch-size 64 --gammas 0.1 --schedule 10 20 --nepoch 30 --lr 0.001 --gpu-num 1 --augment all --flatten
#python train.py --name 210218_trans --batch-size 16 --nepoch 16 --cutoff 256 --gpu-num 0 --augment cutmix --flatten --arch transpose

#python train.py --name 210323_transr --batch-size 32 --nepoch 20 --multi-gpu --flatten --arch transr
#python train.py --name paper_time_test --batch-size 64 --nepoch 1 --gpu-num 1 --cutoff 284 --flatten --arch resnet
#python train.py --name paper_time_test --batch-size 64 --nepoch 1 --gpu-num 1 --cutoff 284 --flatten --arch t2t
#python train.py --name paper-210412 --batch-size 64 --nepoch 30 --cutoff 284 --flatten --gpu-num 1 --arch t2t --augment mixup --frame-stack-num 1
python train.py --name paper-0414 --batch-size 64 --nepoch 30 --cutoff 284 --flatten --gpu-num 1 --arch t2t --frame-stack-num 1

# Valid
#python valid.py --gpu-num 1 --batch-size 64 --cutoff 284 --flatten --arch t2t --frame-stack-num 1
#python valid.py --gpu-num 1 --batch-size 32 --cutoff 284 --flatten --arch t2t --frame-stack-num 1
