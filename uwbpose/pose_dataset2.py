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

from visualize import print_masks



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
        self.load_mask = True
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

        data_path = '/data/nlos/save_data_ver2'
        if self.gt_shape == 32:
            data_path2 = '/data/nlos/gt2'
        elif self.gt_shape == 64:
            data_path2 = '/home/bansh123/deep-high-resolution-net.pytorch/gt6464/'
        else:
            data_path2 = data_path
            self.gt_shape = 120
        #data_path_list = os.listdir(data_path)
        data_path_list = glob.glob(data_path + '/*')
        data_path_list2 = glob.glob(data_path2 + '/*')
        #print("data list", data_path_list)
        data_path_list = sorted(data_path_list)
        data_path_list2 = sorted(data_path_list2)
        #print(data_path_list)
        rf_data = []  # rf data list
        gt_list = []  # ground truth
        mask_list = []
        img_list = []
        print("start - data read")
        #test_dir = [8, 9] # past version - 1
        test_dir = [2, 5, 10, 14, 16, 19] # cur version - 2
        #test_dir = [7]
        #test_dir = [2, 5, 10] # cur version - 2
        #test_dir = [2]
        #test_dir = [2, 5, 10] # los
        #test_dir = [14, 16, 19]  #nlos
        #test_dir = [10, 19] # demo - with mask  ,  los , nlos
        remove_dir = [3, 4] 
        #valid_dir = [25, 26, 27]
        #valid_dir = [21]
        #valid_dir = [28, 29] # nlos wall
        valid_dir = [x for x in range(21, 40)]
        #valid_dir = [x for x in range(1, 40)]  # Model test
        dir_count = 0


        rf_index = 0
        if mode == 'train':
            outlier_list = range(49500, 50000)
        else:
            outlier_list = range(18000, 19000)
            #outlier_list = []
        rf_index = -1
        gt_index = -1
        img_index = -1
        mask_index = -1
        
        frame_stack = deque(maxlen=args.frame_stack_num)        
        for file, file2 in zip(data_path_list,data_path_list2):
            if dir_count in remove_dir:
                dir_count += 1
                continue

            if mode == 'train' and (dir_count in test_dir or dir_count in valid_dir):
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue
            elif mode == 'valid' and dir_count not in valid_dir:
                dir_count += 1
                continue

            not_stacked_list = []
            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                rf_file_list = glob.glob(file + '/raw/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('dir_count:', dir_count,'dir(raw):', file, '\t# of data :', len(rf_file_list))
                #print(rf_file_list)
                frame_stack.clear()
                for rf in rf_file_list:
                    #print(rf)
                    rf_index += 1
                    if rf_index in outlier_list:
                        frame_stack.clear()
                        continue
                    temp_raw_rf = np.load(rf)
                    if self.one:
                        temp_raw_rf = temp_raw_rf[0,0,self.cutoff:]
                    else:
                        temp_raw_rf = temp_raw_rf[:, :, self.cutoff:]

                    #----- normalization ------
                    if self.is_normalize is True:
                        for i in range(temp_raw_rf.shape[0]):
                            for j in range(temp_raw_rf.shape[1]):
                                stdev = np.std(temp_raw_rf[i, j])
                                temp_raw_rf[i, j] = temp_raw_rf[i, j]/stdev
                    
                    #temp_raw_rf = np.transpose(temp_raw_rf, (2, 1, 0)).transpose(0,2,1)
                    temp_raw_rf = torch.tensor(temp_raw_rf).float()

                    #---------- 2차원으로 만들기 -----------
                    if self.flatten:
                        if self.one:
                            #print(temp_raw_rf.shape)
                            temp_raw_rf = rearrange(temp_raw_rf, '(len1 len2) -> len1 len2', len1=int(math.sqrt(temp_raw_rf.shape[0])))
                            #print(temp_raw_rf.shape)
                            #pass 
                        else:
                            #temp_raw_rf = temp_raw_rf.view(-1, 1764)
                            temp_raw_rf = rearrange(temp_raw_rf, 'tx rx len -> (tx rx) len')
                            #temp_raw_rf = rearrange(temp_raw_rf, 'x (len1 len2) -> x len1 len2', len1=int(math.sqrt(temp_raw_rf.shape[1])))
                            temp_raw_rf = rearrange(temp_raw_rf, 'x (len1 len2) -> (len1 x) len2', len1=14)
                            temp_raw_rf = temp_raw_rf.unsqueeze(0)
                        #print(temp_raw_rf.shape)
                    
                    frame_stack.append(temp_raw_rf)
                    if len(frame_stack) == args.frame_stack_num:
                        stacked_raw_rf = torch.stack(tuple(frame_stack), 0)
                        #stacked_raw_rf = rearrange(stacked_raw_rf, 'x (len1 len2) -> (len1 x) len2', len1=14)
                        stacked_raw_rf = stacked_raw_rf.squeeze(0)
                        rf_data.append(stacked_raw_rf)
                    else:
                        not_stacked_list.append(rf_index)
                        #print(not_stacked_list)
                    
                    if self.print_once:
                        print("now shape",stacked_raw_rf.shape)
                        self.print_once = False
                    #rf_data.append(temp_raw_rf) 
                    
                
                #break
                '''
                ground truth data 읽어오기.
                heatmap 형태. 총 데이터의 개수* keypoint * width * height
                '''
                gt_file_list = glob.glob(file2 + '/gt/*')
                gt_file_list = sorted(gt_file_list)
                print('dir(gt):', file2, '\t# of data :', len(gt_file_list)) 
                for gt in gt_file_list:
                    gt_index += 1
                    if gt_index in outlier_list or gt_index in not_stacked_list:
                        continue
                    gt_list.append(gt)

                #img_file_list = glob.glob(file2 + '/img/*.jpg')
                #img_file_list = sorted(img_file_list)

                mask_file_list = glob.glob(file2 + '/mask/*')
                mask_file_list = sorted(mask_file_list)
                target_file_list = glob.glob(file2 + '/target/*')
                target_file_list = sorted(target_file_list)
                print('dir(mask):', file2, '\t# of data :', len(mask_file_list))
                print('dir(target):', file2, '\t# of data :', len(target_file_list)) 
                for i in range(len(mask_file_list)):
                    
                    mask = mask_file_list[i]
                    target = target_file_list[i]
                    #img = img_file_list[i]
                    assert mask[-9:] == target[-9:]
                    #print(mask[-9:], target[-9:])
                    #print("mask shape = ", mask, np.load(mask).shape)
                    #print("target shape = ", np.load(target), np.load(target).shape)
                    
                    mask = np.load(mask)
                    target = np.load(target)
                    mask_with_obj = np.zeros((3, 120, 120))
                    #if target.shape[0] > mask.shape[0]:
                        #print(target)
                        #continue
                    for i in range(mask.shape[0]):
                        find_human = False
                        find_chair = False
                        #print(target[i][4])
                        
                        if target[i][4] ==0:
                            if find_human == False:
                                mask_with_obj[0] = cv2.resize(mask[i], (120, 120))
                                find_human = True
                            #else:
                            #    mask_with_obj[2] = cv2.resize(mask[i], (120, 120))
                        elif target[i][4] == 56:
                            if find_chair == False:
                                mask_with_obj[1] = cv2.resize(mask[i], (120, 120))
                                find_chair = True
                        else:
                            mask_with_obj[2] = cv2.resize(mask[i], (120, 120))
                    #for i in range(mask_with_obj.shape[0]):
                    #    print_masks(cv2.imread(img), mask_with_obj[i], i)

                    #print("mask shape = ", mask.shape)
                    mask_index += 1
                    if mask_index in outlier_list or mask_index in not_stacked_list:
                        continue
                    mask_list.append(mask_with_obj)
                    
                    if mask_index ==0:
                        print("mask_shape = ", mask_with_obj.shape)

                    #break
                
                '''
                mask_file_list = glob.glob(file2 + '/human_mask/*')
                mask_file_list = sorted(mask_file_list)
                print('dir(mask):', file2, '\t# of data :', len(mask_file_list)) 
                for mask in mask_file_list:
                    if mask_index == 1:
                        print("mask shape = ", mask, np.load(mask).shape)
                    mask_index += 1
                    if mask_index in outlier_list or mask_index in not_stacked_list:
                        continue
                    mask_list.append(mask)
                '''
                #np.load = np_load_old
                if self.load_img is True:
                    img_file_list = glob.glob(file + '/img/*.jpg')
                    img_file_list = sorted(img_file_list)
                    print('dir(img):', file, '\t# of data :', len(img_file_list))
                    for img in img_file_list:
                        img_index += 1
                        if img_index in outlier_list:
                            img_index += 1
                            continue
                        temp_img = cv2.imread(img)
                        img_list.append(temp_img)


            dir_count += 1
        self.rf_data = rf_data
        self.gt_list = gt_list
        self.mask_list = mask_list
        print(len(gt_list))
        if self.mode == 'valid' and len(self.gt_list) == 0:
            for i in range(len(self.rf_data)):
                self.gt_list.append(np.zeros((13, self.gt_shape, self.gt_shape)))
        if self.mode == 'valid' and len(self.mask_list) == 0:
            for i in range(len(self.rf_data)):
                self.mask_list.append(np.zeros((1, self.gt_shape, self.gt_shape)))
        self.img_list = img_list
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        if self.mode == 'valid':
            gt = np.zeros((13, self.gt_shape, self.gt_shape))
            mask = np.zeros((3, self.gt_shape, self.gt_shape))
        else:
            gt = np.load(self.gt_list[idx])
            #mask = np.load(self.mask_list[idx])
            mask = self.mask_list[idx]           
            #print("loaded gt", gt.shape)

        gt = torch.tensor(gt).float()
        gt = gt.reshape(13, self.gt_shape, self.gt_shape)
        
        mask = torch.tensor(mask).float()
        #mask = mask.reshape(1, self.gt_shape, self.gt_shape)

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
            if self.load_mask is False:
                if self.mode == 'train' and self.is_gaussian is True:
                    gt = gt + torch.randn(gt.size()) * self.std + self.mean
                        
                return rf, gt
            else:
                return rf, (gt, mask)
            # return self.rf_data[idx], self.gt_list[idx]
        else:
            return rf, gt, self.img_list[idx], mask

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
