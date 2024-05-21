# -*- coding: utf-8 -*-
"""
@Time ： 2021/8/5 12:06
@Auth ： xxxyyyzin
@File ：main_SeismicTrans.py
@IDE ：PyCharm
"""

from functools import partial
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, velocity_size=128, patch_size=16, in_c=1, embed_dim=768, norm_layer=None):
        super().__init__()
        velocity_size= (velocity_size, velocity_size)
        patch_size = (patch_size, patch_size)
        self.velocity_size= velocity_size
        self.patch_size = patch_size
        self.grid_size = (velocity_size[0] // patch_size[0], velocity_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape      #[batch,channel,height,width]
        assert H == self.velocity_size[0] and W == self.velocity_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.velocity_size[0]}*{self.velocity_size[1]})."

        #proj(x): [batch,3,224,224] ->[batch,768,14,14]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)    
        x = self.norm(x)
        return x

# Attention mechanism
class Attention(nn.Module):
    def __init__(self,
                 dim,   # Input the dim token vector sequence for the token: [num_token, token_dim]
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP Block in Encoder Block
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# An Encoder Block
class Block(nn.Module):
    def __init__(self,
                 dim,   # Input the dim token vector sequence for the token: [num_token, token_dim]
                 num_heads,
                 mlp_ratio=4.,   # The number of output nodes in the first fully connected layer of MLP is four times that of the input
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,   # Drop path in encoder block
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)   #multi-head attention
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity() # Nn Identity () does not take any action
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SeismicTransformer(nn.Module):
    def __init__(self, velocity_size=128, patch_size=16, in_c=1,embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True,qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.,drop_bn_ratio=0.02, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            velocity_size(int, tuple): input velocity size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer                    # The number of times encoder blocks are stacked
            num_heads (int): number of attention heads           # The number of heads in multi heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            drop_bn_ratio(float):the rate of convolution and anti-convolution operation
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(SeismicTransformer, self).__init__()
        self.name = "SeismicTransformer"
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(velocity_size=velocity_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)  # patch embedding
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim))   # position embedding
        self.pos_drop = nn.Dropout(p=drop_ratio) # Dropout before entering transformer

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # Stochastic depth cause rule (DropPath in Encoder Block)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])  # Stacking of encoder blocks
        self.norm = norm_layer(embed_dim)  # Layer norm after transformer encoder
        self.relu= F.relu

        ####ENCODER
        self.conv1=nn.Conv2d(768,768,(2,2),(2,2),0)
        self.conv1_bn = nn.BatchNorm2d(768)
        self.drop1 = nn.Dropout2d(drop_bn_ratio)

        self.conv2=nn.Conv2d(768,768,(2,2),(2,2),0)
        self.conv2_bn = nn.BatchNorm2d(768)
        self.drop2 = nn.Dropout2d(drop_bn_ratio)

        self.conv3=nn.Conv2d(768,1024,(2,2),(2,2),0)
        self.conv3_bn = nn.BatchNorm2d(1024)
        self.drop3 = nn.Dropout2d(drop_bn_ratio)

        ## DECODER

        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        self.convT1 = nn.ConvTranspose2d(1025, 1025, (2, 2), (2, 2), 0)
        self.convT1_bn = nn.BatchNorm2d(1025)
        self.dropT1 = nn.Dropout2d(drop_bn_ratio)

        self.convT2 = nn.ConvTranspose2d(1025, 512, (2, 4), (2, 4), 0)
        self.convT2_bn = nn.BatchNorm2d(512)
        self.dropT2 = nn.Dropout2d(drop_bn_ratio)

        self.convT2a = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.convT2a_bn = nn.BatchNorm2d(512)
        self.dropT2a = nn.Dropout2d(drop_bn_ratio)

        self.convT2b = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.convT2b_bn = nn.BatchNorm2d(512)
        self.dropT2b = nn.Dropout2d(drop_bn_ratio)

        self.convT3 = nn.ConvTranspose2d(512, 256, (2, 4), (2, 4), 0)
        self.convT3_bn = nn.BatchNorm2d(256)
        self.dropT3 = nn.Dropout2d(drop_bn_ratio)

        self.convT3a = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.convT3a_bn = nn.BatchNorm2d(256)
        self.dropT3a = nn.Dropout2d(drop_bn_ratio)

        self.convT3b = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.convT3b_bn = nn.BatchNorm2d(256)
        self.dropT3b = nn.Dropout2d(drop_bn_ratio)

        self.convT4 = nn.ConvTranspose2d(256, 64, (2, 4), (2, 4), 0)
        self.convT4_bn = nn.BatchNorm2d(64)
        self.dropT4 = nn.Dropout2d(drop_bn_ratio)

        self.convT4a = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.convT4a_bn = nn.BatchNorm2d(64)
        self.dropT4a = nn.Dropout2d(drop_bn_ratio)

        self.convT4b = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.convT4b_bn = nn.BatchNorm2d(64)
        self.dropT4b = nn.Dropout2d(drop_bn_ratio)

        self.convT5 = nn.ConvTranspose2d(64, 8, (2, 4), (2, 4), 0)
        self.convT5_bn = nn.BatchNorm2d(8)
        self.dropT5 = nn.Dropout2d(drop_bn_ratio)

        self.convT5a = nn.Conv2d(8, 8, (3, 3), (1, 1), (1, 1))
        self.convT5a_bn = nn.BatchNorm2d(8)
        self.dropT5a = nn.Dropout2d(drop_bn_ratio)

        self.convT5b = nn.Conv2d(8, 8, (3, 3), (1, 1), (1, 1))
        self.convT5b_bn = nn.BatchNorm2d(8)
        self.dropT5b = nn.Dropout2d(drop_bn_ratio)

        self.convT6 = nn.Conv2d(8, 1, (1, 1), (1, 1), 0)  # final linear layer





    def forward(self,x,s):
        x = self.patch_embed(x)  # [B,1,128,128]->[B, 64, 768]
        x = self.pos_drop(x + self.pos_embed)   # Add position embedding before dropout
        x = self.blocks(x)  # transformer encoder （Encoder Block X L(12) ）
        x = self.norm(x)   # layer norm
        B,C,H,W = x.shape[0],x.shape[2],int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1]))
        x = x.reshape(B,C,H,W)    #[B,768,8,8]

        x = self.drop1(self.relu(self.conv1_bn(self.conv1(x))))                   #[B,768,4,4]
        x = self.drop2(self.relu(self.conv2_bn(self.conv2(x))))                   #[B,768,2,2]
        x = self.drop3(self.relu(self.conv3_bn(self.conv3(x))))                   #[B,1024,1,1]

        x = torch.cat((x, s[:, 0:1, :, :]), dim=1)                                #[B,1025,1,1]

        x = self.dropT1(self.relu(self.convT1_bn(self.convT1(x))))                #[B,1025,2,2]

        x = self.dropT2(self.relu(self.convT2_bn(self.convT2(x))))                #[B,512,4,8]
        x = self.dropT2a(self.relu(self.convT2a_bn(self.convT2a(x))))             #[B,512,4,8]
        x = self.dropT2b(self.relu(self.convT2b_bn(self.convT2b(x))))             #[B,512,4,8]

        x = self.dropT3(self.relu(self.convT3_bn(self.convT3(x))))                #[B,256,8,32]
        x = self.dropT3a(self.relu(self.convT3a_bn(self.convT3a(x))))             #[B,256,8,32]
        x = self.dropT3b(self.relu(self.convT3b_bn(self.convT3b(x))))             #[B,256,8,32]

        x = self.dropT4(self.relu(self.convT4_bn(self.convT4(x))))                #[B,64,16,128]
        x = self.dropT4a(self.relu(self.convT4a_bn(self.convT4a(x))))             #[B,64,16,128]
        x = self.dropT4b(self.relu(self.convT4b_bn(self.convT4b(x))))             #[B,64,16,128]

        x = self.dropT5(self.relu(self.convT5_bn(self.convT5(x))))                #[B,8,32,512]
        x = self.dropT5a(self.relu(self.convT5a_bn(self.convT5a(x))))             #[B,8,32,512]
        x = self.dropT5b(self.relu(self.convT5b_bn(self.convT5b(x))))             #[B,8,32,512]

        x = self.convT6(x)  # final linear layer                                   #[B,1,32,512]

        return x



def SeismicTrans_base_patch16_128():
    model=SeismicTransformer(velocity_size=128,patch_size=16,embed_dim=768,depth=12,num_heads=6,drop_bn_ratio=0.01)
    return model

model_param=SeismicTrans_base_patch16_128()
print("Model: %s" % (model_param.name))
total_params = sum(p.numel() for p in model_param.parameters())
total_trainable_params = sum(p.numel() for p in model_param.parameters() if p.requires_grad)
print("Total number of parameters: %i" % (total_params))
print("Total number of trainable parameters: %i" % (total_trainable_params))
