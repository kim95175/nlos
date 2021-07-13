############################################
#
#   Visualize results of trained model
#
############################################
import math

import datetime
import os
import numpy as np
import torchvision
import cv2
import torch

from inference import get_max_preds
from transforms import get_affine_transform

COLORS = (244,  67,  54)

def print_masks(img, masks, idx):
    threshold = 0.7
    mask_alpha=0.45
    color = (244,  67,  54)
    color = (color[2], color[1], color[0])
    #print(img.shape)
    #img = img.cpu().numpy()
    img = cv2.resize(img, dsize=(480, 480), interpolation=cv2.INTER_AREA)
    #mask = masks[0][0]#.cpu().numpy()
    mask = masks
    
    #print(mask[45:75,45:75])   
    #mask = [0 if v_ > threshold else v_ for v_ in mask]
    mask[mask < threshold] = 0
    #print(mask[45:75,45:75])    

    mask = mask[:, :, None] # 120, 120, 1
    mask = mask.repeat(3, axis=2) * color
    mask_numpy = cv2.resize(mask, (480, 480))
    #print(mask_numpy.shape, img.shape)
    
    '''
    masks = masks[:, :, :, None]

    color = torch.Tensor(color).cpu() / 255.
    color = color.view(1,1,1,3)
    #print(masks.shape, color.shape, img.shape)
    masks_color =  masks.repeat(1, 1, 1, 3) * color * mask_alpha
    mask_gpu = masks_color[0]
    mask_numpy = (mask_gpu * 255).byte().cpu().numpy()
    #print(mask_numpy.shape)
    mask_numpy = cv2.resize(mask_numpy, dsize=(480, 480), interpolation=cv2.INTER_AREA)
    '''
    save_dir = './vis/mask_0713/'
    os.makedirs(save_dir, exist_ok=True)

    #res = np.concatenate((img, mask_numpy), axis=1)
    res = img + mask_numpy * mask_alpha
    res= np.concatenate((res, mask_numpy), axis=1)
    #print(res.shape)
    cv2.imwrite(os.path.join(save_dir, str(idx)) +'_masks_fusion{}.png'.format(threshold), res)

def save_batch_image_with_joints(batch_image, batch_joints,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]

            for joint in joints:
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    dst =  cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, dst)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        # 1,2,0
                              
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    #dst =  cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name,  grid_image)


def save_debug_images(input, label, target, joints_pred, output,
                      prefix):
    print(input.shape, label.shape, target.shape, joints_pred.shape, output.shape)
    save_batch_image_with_joints(
        input, label, '{}_gt.jpg'.format(prefix)
    )

    save_batch_image_with_joints(
        input, joints_pred, '{}_pred.jpg'.format(prefix)
    )

    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )


class Visualize:
    def __init__(self, show_debug_idx=False):
        self.keypoints = [ "padding", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]
        # delete eye, ear
        #skeleton_pair = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        #        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


        self.skeleton_pair = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.skeleton_pair = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [1, 1]] # [1,6], [1,7] 제거 하고 머리랑 어깨 사이 추가
    
        self.skeleton_color = [(100, 0, 0), (100, 100, 0), (100, 100, 100), (0, 100, 0), (0, 100, 100), (0, 0, 100), (100, 0, 100),
            (180, 0, 0), (180, 180, 0), (0, 180, 0), (0, 180, 180), (0, 0, 180), (180, 0, 180), (180, 180, 180)] 
        
        self.show_debug_idx = show_debug_idx
        now = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm')
        self.save_dir = './vis/{}/'.format(now)
        os.makedirs(self.save_dir, exist_ok=True)

    def detect_and_draw_person(self, img, pred, idx, prefix):
        start_idx = idx*32
        #print("img shape", img.shape, "pred shape", pred.shape)
        num_img = img.shape[0]
        num_joint = pred.shape[1]
        center = np.array([320, 240], dtype=np.float32)
        scale = np.array([4, 4], dtype=np.float32)
        rotation = 0
        for i in range(num_img):
            #dst = cv2.resize(img[i], dsize=(640, 480), interpolation=cv2.INTER_AREA)
            dst = img[i].copy()
            dst = np.zeros(dst.shape)
            #trans = get_affine_transform(center, scale, rotation, (480, 480))
            #dst = cv2.warpAffine(img[i], trans, (int(480), int(480)), flags=cv2.INTER_LINEAR)
            for j in range(num_joint):
                x_coord, y_coord = int(pred[i][j][0]*4/3), int(pred[i][j][1])
                cv2.circle(dst, (x_coord, y_coord), 3, (0, 0, 255), -1)
        
            col_idx = 0

            for k in self.skeleton_pair:
                # index가 1부터 시작하니까 1씩 빼줌
                start_point = k[0]-1
                end_point = k[1]-1
                if start_point > 1:
                    start_point -= 4
                if end_point > 1:
                    end_point -= 4


                pt1 = (int(pred[i][start_point][0]*4/3), int(pred[i][start_point][1]))
                pt2 = (int(pred[i][end_point][0]*4/3), int(pred[i][end_point][1]))
                if start_point == 0: # nose
                    point = (pred[i][1] + pred[i][2])/2
                    pt2 = (int(point[0]*4/3), int(point[1]))

                cv2.line(dst, pt1, pt2, self.skeleton_color[col_idx], thickness=3)
                col_idx += 1

            cv2.imwrite(self.save_dir + prefix + '_%05d.jpg'%(start_idx+i), dst)


    def compare_visualize(self, img, pred, target, idx, mask=None):
   
        start_idx = idx*32
        if mask is not None:
            print("img shape", img.shape, "pred shape", pred.shape, "mask shape", mask.shape)
        num_img = img.shape[0]
        num_joint = pred.shape[1]
        center = np.array([320, 240], dtype=np.float32)
        scale = np.array([4, 4], dtype=np.float32)
        rotation = 0
        for i in range(num_img):
            #dst = cv2.resize(img[i], dsize=(480, 480), interpolation=cv2.INTER_AREA)
            #trans = get_affine_transform(center, scale, rotation, (480, 480))
            #dst = cv2.warpAffine(img[i], trans, (int(480), int(480)), flags=cv2.INTER_LINEAR)
            #if i % 5 != 0 :
            #    continue
            original = img[i].copy()
            
            if self.show_debug_idx is True:
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (30, 30)
                fontScale = 1
                color = (255, 255, 255)
                thickness = 2
                original = cv2.putText(original, '{}'.format(start_idx+i), org, font, fontScale, color, thickness, cv2.LINE_AA)
        
            dst = np.zeros(original.shape)
            gt_img = dst.copy()
            for j in range(num_joint):
            
                # prediction
                x_coord, y_coord = int(pred[i][j][0]*4/3), int(pred[i][j][1])
                cv2.circle(dst, (x_coord, y_coord), 3, (0, 0, 255), -1)

                # gt
                x_coord, y_coord = int(target[i][j][0]*4/3), int(target[i][j][1])
                cv2.circle(gt_img, (x_coord, y_coord), 3, (0, 0, 255), -1)
            col_idx = 0
            for k in self.skeleton_pair:
                # index가 1부터 시작하니까 1씩 빼줌
                start_point = k[0]-1
                end_point = k[1]-1
                # 눈, 귀 빼서 4개 똑같이 빼줌 1보다 크면
                if start_point > 1:
                    start_point -= 4
                if end_point > 1:
                    end_point -= 4


                # pred
                pt1 = (int(pred[i][start_point][0]*4/3), int(pred[i][start_point][1]))
                pt2 = (int(pred[i][end_point][0]*4/3), int(pred[i][end_point][1]))
                if start_point == 0: # nose
                    point = (pred[i][1] + pred[i][2])/2
                    pt2 = (int(point[0]*4/3), int(point[1]))

                cv2.line(dst, pt1, pt2, self.skeleton_color[col_idx], thickness=3)
            
                # gt
                pt1 = (int(target[i][start_point][0]*4/3), int(target[i][start_point][1]))
                pt2 = (int(target[i][end_point][0]*4/3), int(target[i][end_point][1]))
                if start_point == 0: # nose
                    point = (target[i][1] + target[i][2])/2
                    pt2 = (int(point[0]*4/3), int(point[1]))

                cv2.line(gt_img, pt1, pt2, self.skeleton_color[col_idx], thickness=3)
            
            
                col_idx += 1

            res = dst
            #res = np.concatenate((gt_img, dst), axis=1)
            res = np.concatenate((original, res), axis=1)
            #cv2.imwrite(self.save_dir + 'compare_%05d.jpg'%(start_idx+i), res)

