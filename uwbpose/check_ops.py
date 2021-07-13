import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

dummy = torch.zeros(3, 3, 1764)
for i in range(3):
    for j in range(3):
        num = (i+1)*(j+1)*10000
        for k in range(dummy.shape[2]):
            tmp_num = num+k
            dummy[i][j][k] = tmp_num

print(dummy[0][0])

temp_raw_rf = dummy.view(-1, 1764)
temp_raw_rf1 = torch.cat([temp_raw_rf[:,:126],temp_raw_rf[:,126:126*2],temp_raw_rf[:,126*2:126*3],temp_raw_rf[:,126*3:126*4],temp_raw_rf[:,126*4:126*5],temp_raw_rf[:,126*5:126*6],temp_raw_rf[:,126*6:126*7],temp_raw_rf[:,126*7:126*8],temp_raw_rf[:,126*8:126*9],temp_raw_rf[:,126*9:126*10],temp_raw_rf[:,126*10:126*11],temp_raw_rf[:,126*11:126*12],temp_raw_rf[:,126*12:126*13],temp_raw_rf[:,126*13:126*14]])

print(temp_raw_rf1.shape)       
print(temp_raw_rf1[1])

temp_raw_rf = rearrange(dummy, 'tx rx len -> (tx rx) len')
temp_raw_rf2 = rearrange(temp_raw_rf, 'x (len1 len2) -> (len1 x) len2', len1=14)

print(temp_raw_rf2.shape)
for i in range(15):
    print(temp_raw_rf2[i])

print(torch.equal(temp_raw_rf1, temp_raw_rf2))

temp_raw_rf = rearrange(dummy, 'tx rx len -> (tx rx) len')
temp_raw_rf3 = rearrange(temp_raw_rf, 'x (len1 len2) -> (x len2) len1', len1=126)

print(temp_raw_rf3.shape)
for i in range(15):
    print(temp_raw_rf3[i])


k1 = temp_raw_rf2.unsqueeze(0)
k2_list = []
k2_list.append(temp_raw_rf2)
k2 = torch.stack(k2_list, 0)

print(k1.shape, k2.shape)
print(torch.equal(k1, k2))


new_raw = dummy[0, 0, :1600]
temp_raw_rf4 = rearrange(new_raw, '(len1 len2) -> 1 len1 len2', len1=int(math.sqrt(new_raw.shape[0])))
#for i in range(15):
#    print(temp_raw_rf4[i])

frame_stack = []
for i in range(3):
    for j in range(3):
        frame_stack.append(dummy[i,j,:])

temp_raw_rf = torch.stack(frame_stack, 0)
temp_raw_rf2 = rearrange(temp_raw_rf, 'x (len1 len2) -> (len1 x) len2', len1=14)
print(temp_raw_rf2.shape)
for i in range(15):
    print(temp_raw_rf2[i])
