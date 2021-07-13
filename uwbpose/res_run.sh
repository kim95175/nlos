#python train.py --name paper-0419 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch resnet_one
#python train.py --name paper-0419 --batch-size 64 --nepoch 30 --augment cutmix --gpu-num 0 --arch resnet_one
#python train.py --name paper-0419 --batch-size 64 --nepoch 30 --augment mixup --gpu-num 0 --arch resnet_one

#python train.py --name paper-0426-1 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch t2t_one
#python train.py --name paper-0426-2 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch t2t_one
#python train.py --name paper-0426-3 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch t2t_one
#python train.py --name paper-0426-4 --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch t2t_one
#python train.py --name 0703_4d --batch-size 64 --nepoch 30 --augment None --gpu-num 0 --arch resnet


python valid.py --gpu-num 0 --batch-size 64 --augment None --arch resnet --vis
#python valid.py --gpu-num 0 --batch-size 64 --augment None --arch resnet_one
