import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

import arguments
from pose_dataset2 import *
#from pose_dataset3 import *
#from model.pose_resnet_4d import *
#from model.pose_resnet_2d import *
#from model.pose_resnet_1d import *
from model.mask_resnet_2d import *
from model.pose_resnet_one import *
from model.pose_hrnet import *
from model.transpose_h import *
from model.transpose_r import *
from model.vit import ViT
#from model.t2t_one import T2TViT_One
from model.t2t_one_new import T2TViT_One
from model.t2t120 import T2TViT
from loss import *
from visualize import *
from inference import *
from make_log import *
from evaluate import *



def prediction(model, rf, target_heatmap, criterion, debug_mask=True):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    #start.record()
    if debug_mask:
        out, mask = model(rf)
    else:
        out = model(rf)

    #end.record()

    #torch.cuda.synchronize()
    #print(start.elapsed_time(end))

    loss = criterion(out, target_heatmap)

    _, temp_avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
            target_heatmap.detach().cpu().numpy())
    
    #print(out.shape, target_heatmap.shape)
    preds, maxvals = get_final_preds(out.clone().cpu().numpy())

    target_label, target_maxvals = get_final_preds(target_heatmap.clone().cpu().numpy())
    
    #print(preds.shape, target_label.shape)
    temp_true_det, temp_whole_cnt = pck(preds*4, target_label*4) # *4

    if debug_mask:
        return out, mask, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt
    else:
        return out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt

def validate(dataloader, model, logger, criterion, debug_img=False, debug_mask=False):
    model.eval()
    criterion = JointsMSELoss().cuda()
    vis = Visualize(show_debug_idx=False) 
    with torch.no_grad():
        epoch_loss = []
        avg_acc = 0
        sum_acc = 0
        total_cnt = 0

        iterate = 0

        true_detect = np.zeros((4, 13))
        whole_count = np.zeros((4, 13))
        
        if debug_mask == True:
            for rf, target_heatmap, img, mask in tqdm(dataloader):
                rf, target_heatmap, img, mask = rf.cuda(), target_heatmap.cuda(), img.cuda(), mask.cuda()
                out, mask, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion, debug_mask=True)
                
                epoch_loss.append(loss)
                sum_acc += temp_avg_acc * cnt
                total_cnt += cnt
                avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
                true_detect += temp_true_det
                whole_count += temp_whole_cnt

                #save_debug_images(img, target_label*4, target_heatmap, preds*4, out, './vis/batch_{}'.format(iterate))
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), preds*4, iterate, 'pred')
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), target_label*4, iterate, 'gt')
                print_masks(img.clone(), mask, iterate)
                #vis.compare_visualize(img.clone().cpu().numpy(), preds*4, target_label*4, iterate)
                
                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1

        elif debug_img == True:
            for rf, target_heatmap, img in tqdm(dataloader):
                rf, target_heatmap, img = rf.cuda(), target_heatmap.cuda(), img.cuda()
                out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion)
                
                epoch_loss.append(loss)
                sum_acc += temp_avg_acc * cnt
                total_cnt += cnt
                avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
                true_detect += temp_true_det
                whole_count += temp_whole_cnt

                #save_debug_images(img, target_label*4, target_heatmap, preds*4, out, './vis/batch_{}'.format(iterate))
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), preds*4, iterate, 'pred')
                #vis.detect_and_draw_person(img.clone().cpu().numpy(), target_label*4, iterate, 'gt')
                vis.compare_visualize(img.clone().cpu().numpy(), preds*4, target_label*4, iterate)

                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1

        else:
            for rf, target_heatmap in tqdm(dataloader):
                rf, target_heatmap = rf.cuda(), target_heatmap.cuda()
                out, loss, temp_avg_acc, cnt, preds, target_label, temp_true_det, temp_whole_cnt = prediction(model, rf, target_heatmap, criterion)
                
                epoch_loss.append(loss)
                sum_acc += temp_avg_acc * cnt
                total_cnt += cnt
                avg_acc = sum_acc / total_cnt if total_cnt != 0 else 0
                true_detect += temp_true_det
                whole_count += temp_whole_cnt

                #if iterate % 100 == 0:
                #    logger.info("iteration[%d] batch loss %.6f\tavg_acc %.4f\ttotal_count %d"%(iterate, loss.item(), avg_acc, total_cnt))
                iterate += 1
        
        logger.info("epoch loss : %.6f"%torch.tensor(epoch_loss).mean().item())
        logger.info("epoch acc on test data : %.4f"%(avg_acc))
        pck_res = true_detect / whole_count * 100
        thr = [0.1, 0.2, 0.3, 0.5]
        for t in range(4):
            logger.info("PCK {} average {} - {}".format(thr[t], np.average(pck_res[t]), pck_res[t]))


if __name__ == '__main__':

    args = arguments.get_arguments()
   
    model_name = args.model_name
    #model_name = 'paper-0419_arch-t2t_one_lr0.001_batch64_nepoch30_cutoff448_aug-None_stack1'
    #model_name = 'paper-0419_arch-resnet_one_lr0.001_batch64_nepoch30_cutoff448_aug-None_stack1'
    #model_name = 'paper-0506_arch-resnet_lr0.001_batch64_nepoch15_cutoff284_aug-None_stack1'
    #model_name = 'paper-0506_arch-t2t_lr0.001_batch64_nepoch30_cutoff284_aug-None_stack1'    
    model_name = '0703_mask_arch-resnet_lr0.001_batch64_nepoch30_cutoff284_aug-None_stack1'

    if len(model_name) == 0:
        print("You must enter the model name for testing")
        sys.exit()
    
    #if model_name[-3:] != '.pt':
    #    print("You must enter the full name of model")
    #    sys.exit()

    #model_name = model_name.split('_epoch')[0]
    print("vaildate mode = ", model_name)
    log_name = model_name.split('/')[-1]
    print("log_name = ", log_name)
    logger = make_logger(log_file='valid_'+log_name)
    logger.info("saved valid log file "+'valid_'+log_name)        

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
    elif args.arch =='transr':
        model = get_transpose_r_net(num_layer=args.nlayer)
    elif args.arch =='vit':
        model = ViT(
            image_size=40, #126,
            patch_size=8, #18,
            dim = 1024, #1024,
            depth = 6,
            heads = 16,
            mlp_dim = 512,  #980, #256 * 14 # feedforward hidden dim
            dropout = 0.1,
            channels = 1,
            emb_dropout = 0.1
        )
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
    elif args.arch =='t2t_one':
        model = T2TViT_One(
            dim = 900,  #
            image_size = 40,
            depth = 5,
            heads = 8,
            mlp_dim = 512,
            channels =1, # 1, #args.frame_stack_num,
            #t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
            t2t_layers = ((7, 4), (3, 2), (1, 1))    # 74 32 32
                                                    # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        )
        logger.info("tkt model hyperparam")
        logger.info("dim : {dim}\t image size : {image_size}\t depth : {depth}\t heads : {heads}\tmlp_dim : {mlp_dim}\t t2t_layers : {t2t_layers}\n".format(
            dim = 900,
            image_size = 40,
            depth = 5,
            heads = 8,
            mlp_dim = 512,
            channels = 1, #9, #1, #args.frame_stack_num,
            #t2t_layers = ((7, 4), (3, 2), (3, 2))
            t2t_layers = ((7,4), (3, 2), (1, 1))
            #t2t_layers = ((9, 5), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        ))
    elif args.arch == 'resnet_one':
        model = get_one_pose_net(num_layer=args.nlayer, input_depth=1) 
    elif args.arch == 'mask_resnet':
        model =get_2d_mask_net(num_layer=args.nlayer, input_depth=1)
    else:
        model = get_4d_pose_net(num_layer=args.nlayer, input_depth=1)

    if multi_gpu is True:
        model = torch.nn.DataParallel(model).cuda()
        logger.info("Let's use multi gpu\t# of gpu : {}".format(torch.cuda.device_count()))
    else:
        device = torch.device(f'cuda:{set_gpu_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        logger.info("Let's use single gpu\t now gpu : {}".format(set_gpu_num))
        model.cuda()
        #model.to(device)
        model = torch.nn.DataParallel(model, device_ids = [set_gpu_num]).cuda()

    #model.cuda() # torch.cuda_set_device(device) 로 send
    #model.to(device) # 직접 device 명시
    

    #----- loss function -----
    #criterion = nn.MSELoss().cuda()
    criterion = JointsMSELoss().cuda()

    #----- dataset -----
    test_data = PoseDataset(mode='test', args=args)
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model_name = model_name + '_epoch{}.pt'
    #model_name = model_name + '_best.pt'
    # 원하는 모델 구간 지정해야함.
    #for i in [30]:
    #for i in range(18, 20):
    for i in [27, 28, 29]:
    #for i in [0]:
    #for i in range(5, 10):
        logger.info("epoch %d"%i)
        logger.info('./save_model/' + model_name.format(i))
        model.module.load_state_dict(torch.load('./save_model/'+model_name.format(i)))
        #model.module.load_state_dict(torch.load('./save_model/'+model_name))

        #model.module.load_state_dict(torch.load(model_name))
        validate(dataloader, model, logger, criterion, debug_img=args.vis, debug_mask=args.mask)
