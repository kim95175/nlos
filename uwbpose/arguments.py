import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='RF Single Pose')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
            help='learning rate ( default: 0.001) ')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
            help='input batch size for training (default: 32)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
            help='learning rate scheduler. decay gamma at these epochs ')
    parser.add_argument('--gammas', type=float, default=0.1,
            help='gamma. LR decay ')
    parser.add_argument('--momentum', type=float, default=0.9,
            help='SGD Momentum')
    parser.add_argument('--nepochs', type=int, default=30,
            help = 'Number of epochs')
    parser.add_argument('--multi-gpu', action='store_true',  default=False,
            help = 'Use multi gpu ( default: single gpu )')
    parser.add_argument('--gpu-num', type=int, default=0,
            help = 'gpu number if you use a single gpu, set this device number')
    parser.add_argument('--optimizer', type=str, default='adam',
            help = 'set optimizer adam or sgd')
    parser.add_argument('--name', type=str, default='Pose',
            help='model name for saving')
    parser.add_argument('--nlayer', type=int, default=18,
            help = 'number of resnet layer, default:18')
    # -----
    parser.add_argument('--gaussian', action='store_true', default=False,
            help='add gaussian noise')
    parser.add_argument('--normalize', action='store_true', default=False,
            help='data normalization by dividing the data with std-dev')
    
    parser.add_argument('--cutoff', type=int, default=284, #448, # 284, 246
            help='cut off the front of the input data,   --> length = 2048 - cutoff')

    parser.add_argument('--hm-size', type=int, default=120, # 128, 246   
            help='size of heatmap (64, 120 )')

    parser.add_argument('--vis', action='store_true', default=False,
            help='visualize the image for debugging')

    parser.add_argument('--model-name', type=str, default='',
            help='load model name')

    parser.add_argument('--frame-stack-num', type=int, default=1, 
            help='number of stacked frame')
    
    parser.add_argument('--mask', action='store_true', default=False,
            help='train and test mask')

    
    parser.add_argument('--augment', default='None', type=str,
            choices=['cutmix',
                'mixup',
                'intensity',
                'all',
                'None'],
            help='rf data augmentation')

    parser.add_argument('--flatten', action='store_true', default=True,
                        help='flatten raw data to 2d [126, 126]')
    parser.add_argument('--arch', default='t2t', type=str,
            choices=['hrnet',
                'resnet',
                'resnet_one',
                'vit',
                't2t',
                't2t_one',
                'mask_resnet'],
            help='backbone architecture')

    return parser.parse_args()

def print_arguments(args, logger):
    #print("training batch size :", args.batch_size,"\tnumber of layer :", args.nlayer)
    logger.info("training batch size : {}\tnumber of layer : {}".format(args.batch_size, args.nlayer))
    #print("optimizer :", args.optimizer, "\tlearning rate :",args.lr, "\tmomentum :",args.momentum)
    logger.info("optimizer : {}\tlearning rate : {}\tmomentum : {}".format(args.optimizer, args.lr, args.momentum))
    #print("number of epochs :", args.nepochs, "\tscheduler :", args.schedule, "\tgammas :", args.gammas)
    logger.info("number of epochs : {}\tscheduler : {}\tgammas : {}".format(args.nepochs, args.schedule, args.gammas))
    
    logger.info("gaussian noise : {}\tnormalize : {}\tcutoff : {}\tvisualize : {}".format(args.gaussian, args.normalize, args.cutoff, args.vis))
    logger.info("data augmentation : {}\tflatten : {}\t hm size : {}\t frame_stack_num :{}".format(args.augment, args.flatten, args.hm_size, args.frame_stack_num))
    logger.info("Model : {}".format(args.arch))
