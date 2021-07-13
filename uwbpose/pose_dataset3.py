import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import os 
import glob
import numpy as np
import time
import cv2
import einops
from einops import rearrange, reduce, repeat
from collections import deque
import math



class PoseDataset(Dataset):
    def __init__(self, args, is_correlation=False, mode='train'):
        
        '''
        dataset 처리
        rf와 이미지의 경우에는 init 할 때부터 읽어와서 메모리에 올리지만 gt는 데이터를 활용할 때마다 load함.
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 
                valid: valid를 위함(demo). rf, img만 있는 경우
        '''
        self.is_correlation = is_correlation
        self.load_img = args.vis
        self.mode = mode
        
        self.is_gaussian = args.gaussian
        self.std = 0.1
        self.mean = 0
        
        self.is_normalize = args.normalize
        self.cutoff = args.cutoff
        print("self.cutoff =", self.cutoff)
        
        self.augmentation = args.augment
        self.augmentation_prob = 1
        self.intensity = Intensity(scale=0.05)
        self.gt_shape = args.hm_size

        self.flatten = args.flatten
        self.arch = args.arch

        self.one = False
        self.print_once = True

        data_path = '/data/nlos/save_data_ver3'
        self.gt_shape = 120
        #data_path_list = os.listdir(data_path)
        #print("data list", data_path_list)
        #img_list = sorted(data_path_list)
        #print(data_path_list)
        rf_data = []  # rf data list
        gt_list = []  # ground truth
        img_list = []
        print("start - data read")

        dir_count = 0
        
        idx = 0

        ma_train_num = 5900
        ma_train_set = range(idx, idx+ma_train_num)
        idx += ma_train_num
        
        human_train_num = 32000 #4000
        human_train_set = range(idx, idx+human_train_num)
        idx += human_train_num

        ma_train_num2 = 4000
        ma_train_set2 = range(idx, idx+ma_train_num2)
        idx += ma_train_num2

        ma_test_num = 2000
        ma_test_set = range(idx, idx+ma_test_num)
        idx += ma_test_num        
         
        human_test_num = 4000
        human_test_set = range(idx, idx+human_test_num)
        idx += human_test_num

        #human_test_set = range(35900, 37900)

        rf_index = -1
        gt_index = -1
        img_index = -1
        
     
        #for file, file2 in zip(data_path_list,data_path_list2):
        # 각 폴더 안의 npy 데이터
        rf_file_list = glob.glob(data_path + '/raw/*.csv')
        rf_file_list = sorted(rf_file_list)
        img_file_list = glob.glob(data_path + '/img/*.jpg')
        img_file_list = sorted(img_file_list)
        del img_file_list[0]
        gt_file_list = glob.glob(data_path + '/gt/*.npy')
        gt_file_list = sorted(gt_file_list)
        del gt_file_list[0]
        print('dir(raw):', data_path, '\t# of data :', len(rf_file_list))
        print('dir(img):', data_path, '\t# of data :', len(img_file_list))
        print('dir(gt):', data_path, '\t# of data :', len(gt_file_list))
        
        len_data = len(rf_file_list)
        for i in range(len_data):
            rf_index += 1
            if (self.mode == 'train') and (rf_index not in human_train_set):
                continue

            if (self.mode == 'test') and (rf_index not in human_test_set):
                continue
                
            rf = rf_file_list[i]
            temp_raw_rf = np.genfromtxt(rf, delimiter=',')# skip_header = 1)

            if i % 500 == 0:
                print(i, temp_raw_rf.shape)

            #----- normalization ------
            if self.is_normalize is True:
                for i in range(temp_raw_rf.shape[0]):
                    stdev = np.std(temp_raw_rf[i])
                    temp_raw_rf[i] = temp_raw_rf[i]/stdev
            
            #temp_raw_rf = np.transpose(temp_raw_rf, (2, 1, 0)).transpose(0,2,1)
            temp_raw_rf = torch.tensor(temp_raw_rf).float()
            #---------- 2차원으로 만들기 -----------
            temp_raw_rf = rearrange(temp_raw_rf, 'x (len1 len2) -> (len1 x) len2', len1=8)
            temp_raw_rf = temp_raw_rf.unsqueeze(0)

            if self.print_once:
                print("now shape",temp_raw_rf.shape)
                self.print_once = False
            
            rf_data.append(temp_raw_rf)
            gt_list.append(gt_file_list[i])
            
            #np.load = np_load_old
            if self.load_img is True:
                temp_img = cv2.imread(img_file_list[i])
                img_list.append(temp_img)   

        self.rf_data = rf_data
        self.gt_list = gt_list
        print(len(gt_list))
        if self.mode == 'valid' and len(self.gt_list) == 0:
            for i in range(len(self.rf_data)):
                self.gt_list.append(np.zeros((13, self.gt_shape, self.gt_shape)))
        self.img_list = img_list
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        if self.mode == 'valid':
            gt = np.zeros((13, self.gt_shape, self.gt_shape))
        else:
            gt = np.load(self.gt_list[idx])
            #print("loaded gt", gt.shape)
        gt = torch.tensor(gt).float()
        gt = gt.reshape(13, self.gt_shape, self.gt_shape)
        
        rf = self.rf_data[idx] 

        #---- augmentation  ----#
        random_prob = torch.rand(1)

        if self.mode == 'train' and self.augmentation != 'None' and random_prob < self.augmentation_prob:
            random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item() 
            
            #while random_target == idx: # random target이 동일하다면 다시 뽑음.
            #    random_target = torch.randint(low=0, high=len(self.rf_data), size=(1,)).item()
            
            target_gt = np.load(self.gt_list[random_target])
            target_gt = torch.tensor(target_gt).reshape(gt.shape)
            target_rf = self.rf_data[random_target]
            #print("augmetatied rf = ", rf.shape)
            #print("augmented gt = ", gt.shape)
            if self.augmentation == 'cutmix':
                rf, gt = cutmix(rf, target_rf, gt, target_gt)
            elif self.augmentation == 'mixup':
                rf, gt = mixup(rf, target_rf, gt, target_gt)
            elif self.augmentation =='intensity':
                rf = self.intensity(rf)
            elif self.augmentation =='all':
                r = np.random.rand(1)
                if r < 0.4:
                    rf, gt = cutmix(rf, target_rf, gt, target_gt)
                #elif r < 0.7:
                    #rf = self.intensity(rf)
                elif r < 0.8:
                    rf, gt = mixup(rf, target_rf, gt, target_gt)
            else:
                print('wrong augmentation')

        if self.load_img is False:
            #gaussian noise
            if self.mode == 'train' and self.is_gaussian is True:
                gt = gt + torch.randn(gt.size()) * self.std + self.mean
                    
            return rf, gt
            # return self.rf_data[idx], self.gt_list[idx]
        else:
            return rf, gt, self.img_list[idx]

def cutmix(rf, target_rf, gt, target_gt):
    beta = 1.0
    lam = np.random.beta(beta, beta)
    # print("rf.size ", rf.size())
    # print(rf.size()[-2])
    bbx1, bby1, bbx2, bby2 = rand_bbox(rf.size(), lam)
    # print(bbx1, bbx2, bby1, bby2)
    rf[:, bbx1:bbx2, bby1:bby2] = target_rf[:, bbx1:bbx2, bby1:bby2]
    # print((bbx2-bbx1)*(bby2-bby1))
    # print(rf.size()[-1] * rf.size()[-2])
    lam = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) / (rf.size()[-1] * rf.size()[-2]))
    new_rf = rf
    new_gt = lam * gt + (1 - lam) * target_gt
    return new_rf, new_gt

def mixup(rf, target_rf, gt, target_gt):
    '''
    논문에서는 배치 내에서 섞지만, 전체 데이터에서 mixup.
    '''
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)

    new_gt = lam * gt + (1 - lam) * target_gt
    new_rf = lam * rf + (1 - lam) * target_rf
    return new_rf, new_gt

class Intensity(nn.Module):
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, ))
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
