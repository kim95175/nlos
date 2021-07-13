############################################
#
#   Visualize results of trained model
#
############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from tqdm import tqdm

from pose_dataset2 import *
#from model.pose_resnet_one import *
#from model.pose_resnet_1d import *
from model.mask_resnet_2d import *              
#from model.t2t_one import T2TViT_One
#from model.t2t_one_new import T2TViT_One
from model.t2t120 import T2TViT
import arguments
from make_log import *
from evaluate import *
from loss import *

args = arguments.get_arguments()

# model name
model_name = '{}_arch-{}_lr{}_batch{}_nepoch{}_cutoff{}_aug-{}_stack{}'.format(
        args.name,
        args.arch,
        #args.hm_size,
        args.lr,
        args.batch_size,
        args.nepochs,
        args.cutoff,
        args.augment,
        args.frame_stack_num
    )
logger = make_logger(log_file=model_name)
logger.info("saved model name "+model_name)        

arguments.print_arguments(args, logger)


multi_gpu = args.multi_gpu
set_gpu_num = args.gpu_num

if torch.cuda.is_available():
    print("gpu", torch.cuda.current_device(), torch.cuda.get_device_name())
else:
    print("cpu")
    
#----- model -----
if args.arch =='hrnet':
    model = get_pose_hrnet()
elif args.arch =='transh':
    model = get_transpose_h_net()
#elif args.arch =='transr':
#    model = get_transpose_r_net(num_layer=args.nlayer)
elif args.arch == 'multitrans':
    model = get_multi_trans_net()
elif args.arch =='vit':
    model = ViT(
            image_size=40,
            patch_size=8,
            dim = 1024, #1024,
            depth = 6,
            heads = 16,
            mlp_dim = 512, #256 * 14 # feedforward hidden dim
            dropout = 0.1,
            channels = 1,
            emb_dropout = 0.1
    )
    
    logger.info("vit model hyperparam")
    logger.info("image size : {image_size}\t patch size : {patch_size}\t dim : {dim}\t depth : {depth}\t heads : {heads}\tmlp_dim : {mlp_dim}\t dropout : {dropout}\t emb_droput : {emb_dropout}\n".format(
            image_size=40,
            patch_size=8,
            dim = 1024, #1024,
            depth = 6,
            heads = 16,
            mlp_dim = 512, #256 * 14 # feedforward hidden dim
            dropout = 0.1,
            channels = 1,
            emb_dropout = 0.1
    ))
elif args.arch =='t2t':
    model = T2TViT(
        dim = 900,  #
        image_size = 126,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        channels = 1, #args.frame_stack_num,
        t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        #t2t_layers = ((9, 5), (3, 2), (3, 2))    # 74 32 32
                                                # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )
    logger.info("tkt model hyperparam")
    logger.info("dim : {dim}\t image size : {image_size}\t depth : {depth}\t heads : {heads}\tmlp_dim : {mlp_dim}\t t2t_layers : {t2t_layers}\n".format(
        dim = 900,
        image_size = 126,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        channels = 1, #args.frame_stack_num,
        t2t_layers = ((7, 4), (3, 2), (3, 2))
        #t2t_layers = ((9, 5), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    ))
    #print(model)
elif args.arch =='t2t_one':
    model = T2TViT_One(
        dim = 900,  #
        image_size = 40,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        channels = 9,
        t2t_layers = ((7, 4), (3, 2), (1, 1)) # t2t_one_new
        #t2t_layers = ((7, 4), (3, 2), (3, 2))    # 74 32 32
                                                # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )
    logger.info("tkt model hyperparam")
    logger.info("dim : {dim}\t image size : {image_size}\t depth : {depth}\t heads : {heads}\tmlp_dim : {mlp_dim}\t t2t_layers : {t2t_layers}\n".format(
        dim = 900,
        image_size = 40,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        channels = 9,
        #t2t_layers = ((7, 4), (3, 2), (3, 2))
        t2t_layers = ((7, 4), (3, 2), (1, 1)) # t2t_one_new
        #t2t_layers = ((9, 5), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    ))
    #print(model)
elif args.arch == 'resnet_one':
    model = get_one_pose_net(num_layer=args.nlayer, input_depth=1)
else:
    #if args.flatten:
    #model = get_2d_pose_net(num_layer=args.nlayer, input_depth=1)
    model = get_2d_mask_net(num_layer=args.nlayer, input_depth=1)
    #else:
    #    model = get_pose_net(num_layer=args.nlayer, input_depth=2048-args.cutoff)

if multi_gpu is True:
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
else:
    torch.cuda.set_device(set_gpu_num)
    logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
    #model.cuda()
    model = torch.nn.DataParallel(model, device_ids = [set_gpu_num]).cuda()
#model.cuda() # torch.cuda_set_device(device) 로 send
#model.to(device) # 직접 device 명시



#----- loss function -----
criterion = nn.MSELoss().cuda()
cr = JointsMSELoss().cuda()
#----- optimizer and scheduler -----
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info('use adam optimizer')
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=False)
    logger.info('use sgd optimizer')

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gammas)

#----- dataset -----
train_data = PoseDataset(mode='train', args=args)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

#----- training -----
max_acc = 0
max_acc_epoch = 0

#load_model_name = '210410_arch-t2t_120_lr0.001_batch64_nepoch50_cutoff284_aug-None_stack1_epoch6.pt'
#load_model_name = '210410_arch-t2t_120_lr0.001_batch64_nepoch50_cutoff284_aug-None_stack1_epoch5.pt'
#load_model_name = 'paper-0414_arch-t2t_lr0.001_batch64_nepoch30_cutoff284_aug-None_stack1_epoch29.pt'
#print("load model = ", load_model_name)
#model.module.load_state_dict(torch.load('./save_model/'+load_model_name))

param = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Initialized model with {} parameters".format(param))


begin_time = datetime.now()
print(begin_time)
#exit(-1)
for epoch in range(args.nepochs):
    logger.info("Epoch {}\tcurrent lr : {} {}".format(epoch, optimizer.param_groups[0]['lr'], lr_scheduler.get_last_lr()))
    epoch_loss = []
    avg_acc = 0
    sum_acc = 0
    total_cnt = 0
    iterate = 0
    for rf, target in tqdm(train_dataloader):
        #print(rf.shape, target_heatmap.shape)
        #print(rf.dtype, target_heatmap.dtype)
        rf, target_heatmap, target_mask = rf.cuda(), target[0].cuda(), target[1].cuda()
        #print("rf.shape : ", rf.shape)
        #print("heatmap.shape : ", target_heatmap.shape)
        #print("mask.shape : ", target_mask.shape)
        
        out, mask = model(rf)
        #loss = 0.5 * criterion(out, target_heatmap)
        loss_pose = cr(out, target_heatmap)
        loss_mask = cr(mask, target_mask)
        loss = loss_pose + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss)
        _, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
                target_heatmap.detach().cpu().numpy())
        sum_acc += temp_avg_acc * cnt
        total_cnt += cnt
        avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
        if iterate % 500 == 0:
            logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
        iterate += 1

    logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item())
    logger.info("epoch acc on train data : %.4f"%(avg_acc))
    
    if avg_acc > max_acc:
        logger.info("epoch {} acc {} > max acc {} epoch {}".format(epoch, avg_acc, max_acc, max_acc_epoch))
        max_acc = avg_acc
        max_acc_epoch = epoch
        #if args.multi_gpu == 1:
        torch.save(model.module.state_dict(), "save_model/" + model_name + "_best.pt")
        #else:
            #torch.save(model.state_dict(), "save_model/" + model_name + "_best.pt")
    lr_scheduler.step()
    #if args.multi_gpu == 1:
    if (epoch >= args.nepochs-3) or (epoch % 10 == 0 and epoch != 0):
        torch.save(model.module.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))
    #else:
    #    torch.save(model.state_dict(), "save_model/" + model_name + "_epoch{}.pt".format(epoch))

logger.info("training end | elapsed time = " + str(datetime.now() - begin_time))


