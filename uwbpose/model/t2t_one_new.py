import math
import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

BN_MOMENTUM = 0.1

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

# helpers
def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# classes
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# main class

class T2TViT_One(nn.Module):
    def __init__(self, *, image_size, dim, depth = None, heads = None, mlp_dim = None, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., transformer = None, t2t_layers = ((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        print("-------------- T2TVit RF Transformer Model(1x1) new--------------")
        layers = []
        layer_dim = channels
        output_image_size = image_size
        self.deconv_with_bias = False

        self.t2t_module = []
        #t2t_layers = ((7, 4), (3, 2), (1, 1))
        print(t2t_layers)
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            if i == 0:
                layer_dim *= kernel_size ** 2
                output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
                #output_image_size = 19
                layers.extend([
                    nn.Identity(),
                    nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                    Rearrange('b c n -> b n c'),
                    Transformer(dim = layer_dim, heads = 1, depth = 1, dim_head = layer_dim, mlp_dim = layer_dim, dropout = dropout),
                ])
                #self.t2t_module.append(nn.Sequential(*layers))
                #layers = []
            else:
                layer_dim *= kernel_size ** 2
                output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
                layers.extend([
                    RearrangeImage(),
                    nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                    Rearrange('b c n -> b n c'),
                    Transformer(dim = layer_dim, heads = 1, depth = 1, dim_head = layer_dim, mlp_dim = layer_dim, dropout = dropout),
                ])
                #self.t2t_module.append(nn.Sequential(*layers))
                #layers = [] 
        
        #layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)
        #self.linear_projection = nn.Linear(layer_dim, dim)
        dim = layer_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.inplanes = output_image_size ** 2 + 1 #65

        self.deconv_layers = self._make_deconv_layer(
            4,  # NUM_DECONV_LAYERS
            [256,256,256,256],  # NUM_DECONV_FILTERS
            [3,4,4,4],  # NUM_DECONV_KERNERLS
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=13,  #cfg['MODEL']['NUM_JOINTS'],
            kernel_size=1,  #extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=0
        )

        dummy = torch.zeros(64, 1,  40, 40)
        #dummy = torch.zeros(32, channels, 9, 1809)  
        self.check_dim(dummy)


    def forward(self, x):
        x = self.to_patch_embedding(x)
        #x = self.linear_projection(x)
        x += self.pos_embedding
        x = self.transformer(x)
        #x = rearrange(x, 'b n c -> b c n')
        x = rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1]))).contiguous()
        x = self.deconv_layers(x) # batch, 256, 64, 64
        x = self.final_layer(x)
        return x

   
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 5:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):  
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        s = 2
        self.inplanes = 441#900#441
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            if i==0:
                s=3
            else:
                s=2
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def check_dim(self, x):

        print("input : ", x.shape)
        print (x.is_contiguous())
        x = self.to_patch_embedding(x)
        print("to_patch_embedding : ", x.shape)
        print (x.is_contiguous())
        #x = self.linear_projection(x)
        #print("linear : ", x.shape)
        x += self.pos_embedding
        print("pose_embedding : ", x.shape)
        print (x.is_contiguous())
        #x = self.dropout(x)
        x = self.transformer(x)
        print("transformer : ", x.shape)
        print (x.is_contiguous())
        #x = rearrange(x, 'b n c -> b c n')
        #print("reshape ", x.shape)
        #print (x.is_contiguous())
        #x = rearrange(x, 'b c (h w) -> b c h w', h = int(math.sqrt(x.shape[2])))
        
        x = rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1]))).contiguous()
        print("reshape ", x.shape)
        print (x.is_contiguous())
        x = self.deconv_layers(x) # batch, 256, 64, 64
        print("deconv : ", x.shape)
        print (x.is_contiguous())
        x = self.final_layer(x)
        print("reshape ", x.shape)
        print (x.is_contiguous())
    
        return x
