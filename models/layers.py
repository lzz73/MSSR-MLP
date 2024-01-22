
import math
import torch.nn as nn
import torch
import einops

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def patch_partition(x,patch_size):
    b,c,h,w = x.size()
    x1 = x.view(b,c,h//patch_size,patch_size,w//patch_size,patch_size)
    patches = x1.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, c, patch_size, patch_size)
    return patches

def patch_reverse(patches,patch_size,h,w):
    B = int(patches.shape[0]/(h*w/patch_size/patch_size))
    x = patches.view(B,-1,h//patch_size,w//patch_size,patch_size,patch_size)
    x1 = x.permute(0,1,2,4,3,5).contiguous().view(B,-1,h,w)
    return x1

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class CA(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        y = x * y
        return y


class MS_MLP(nn.Module):
    def __init__(self, dim, image_size, patch_size=4):
        super(MS_MLP,self).__init__()
        self.patch_size = patch_size
        self.image_size = (image_size,image_size)
        self.dim = dim
        self.fc1 = nn.Conv2d(dim,dim*3,kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(3*dim,dim,kernel_size=1,bias=False)
        self.ffn = nn.Sequential(LayerNorm2d(dim),
                                 nn.Conv2d(dim,dim*4,kernel_size=1,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim*4,kernel_size=3,padding=1,groups=dim*4,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim,kernel_size=1,bias=False))

        self.spatial = patch_size*patch_size

        self.linear1 = nn.Linear(self.spatial,self.spatial)
        self.linear2 = nn.Linear(self.spatial,self.spatial)
        self.linear3 = nn.Linear(self.spatial,self.spatial)

        self.norm = LayerNorm2d(dim)


    def forward(self,x):
        x0 = x[0]
        m1 = x[1].cuda()
        n1 = x[2].cuda()
        m2 = x[3].cuda()
        n2 = x[4].cuda()
        k1 = x[5].cuda()
        k2 = x[6].cuda()
        _,_,h,w = x0.size()
        x1 = self.norm(x0)
        x2 = self.fc1(x1)
        x2_1,x2_2,x2_3 = torch.chunk(x2,3,1)
        ######### multi_scale_fuse ##############
        #####stage1#####
        patches0 = patch_partition(x2_1,self.patch_size) #patches = 8*8
        _,_,h1,w1 = patches0.size()
        ##### patch to token ####
        patches1 = einops.rearrange(patches0,'b c h w -> b c (h w)')
        ###### mlp in spatial#####
        patches1_spatial = self.linear1(patches1)        
        ### token to patch ######
        patches1_reverse = einops.rearrange(patches1_spatial,'b c (h w) -> b c h w',h=h1,w=w1)
        ####### reverse to image ########
        x3 = patch_reverse(patches1_reverse,self.patch_size,h,w)
        x_g1 = torch.mul(x2_1,x3)

        #####stage2###
        x4 = k1@x2_2@k2
        patches2 = patch_partition(x4,self.patch_size)
        _,_,h2,w2 = patches2.size()
        patches2_1 = einops.rearrange(patches2,'b c h w -> b c (h w)')
        patches2_spatial = self.linear2(patches2_1)
        patches2_reverse = einops.rearrange(patches2_spatial,'b c (h w) -> b c h w',h=h2,w=w2)
        x5 = patch_reverse(patches2_reverse,self.patch_size,h,w)
        x6 = k1@x5@k2
        x_g2 = torch.mul(x2_2,x6)

        ######stage3#####
        x7 = m1@x2_3@n1
        patches3 = patch_partition(x7,self.patch_size)
        _,_,h3,w3 = patches3.size()
        patches3_1 = einops.rearrange(patches3,'b c h w -> b c (h w)')
        patches3_spatial = self.linear3(patches3_1)
        patches3_reverse = einops.rearrange(patches3_spatial,'b c (h w) -> b c h w',h=h3,w=w3)
        x8 = patch_reverse(patches3_reverse,self.patch_size,h,w)
        x9 = m2@x8@n2
        x_g3 = torch.mul(x2_3,x9)

        ####fuse#####
        x_cat = torch.cat((x_g1,x_g2,x_g3),1)
        x_sp_fuse = self.fc2(x_cat)
        x_out1 = x_sp_fuse + x0

        x_cp = self.ffn(x_out1)
        x_out2 = x_cp + x_out1
        return [x_out2,m1,n1,m2,n2,k1,k2]

    def flops(self):
        h,w = self.image_size
        n = h*w/self.spatial
        flops = 0
        #matrix mul
        flops += 4*2*self.dim*h*w
        #norm1 and norm2
        flops += 2*h*w*self.dim
        #fc1 and fc2
        flops += 2*h*w*self.dim*self.dim*3
        #depthwise
        flops += h*w*(3*3)*self.dim*4
        #linear
        flops += 3*n*self.dim*self.spatial*self.spatial
        #fc3 and fc4
        flops += 2*h*w*self.dim*4*self.dim
        return flops

class MS_MLP_tail(nn.Module):
    def __init__(self, dim, image_size, patch_size=4):
        super(MS_MLP_tail,self).__init__()
        self.image_size = (image_size,image_size)
        self.patch_size = patch_size
        self.dim = dim
        self.fc1 = nn.Conv2d(dim,dim*3,kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(3*dim,dim,kernel_size=1,bias=False)
        self.ffn = nn.Sequential(LayerNorm2d(dim),
                                 nn.Conv2d(dim,dim*4,kernel_size=1,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim*4,kernel_size=3,padding=1,groups=dim*4,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim,kernel_size=1,bias=False))

        self.spatial = patch_size*patch_size

        self.linear1 = nn.Linear(self.spatial,self.spatial)
        self.linear2 = nn.Linear(self.spatial,self.spatial)
        self.linear3 = nn.Linear(self.spatial,self.spatial)

        self.norm = LayerNorm2d(dim)


    def forward(self,x):
        x0 = x[0]
        m1 = x[1].cuda()
        n1 = x[2].cuda()
        m2 = x[3].cuda()
        n2 = x[4].cuda()
        k1 = x[5].cuda()
        k2 = x[6].cuda()
        _,_,h,w = x0.size()
        x1 = self.norm(x0)
        x2 = self.fc1(x1)
        x2_1,x2_2,x2_3 = torch.chunk(x2,3,1)
        ######### multi_scale_fuse ##############
        #####stage1#####
        patches0 = patch_partition(x2_1,self.patch_size) #patches = 8*8
        _,_,h1,w1 = patches0.size()
        ##### patch to token ####
        patches1 = einops.rearrange(patches0,'b c h w -> b c (h w)')
        ###### mlp in spatial#####
        patches1_spatial = self.linear1(patches1)        
        ### token to patch ######
        patches1_reverse = einops.rearrange(patches1_spatial,'b c (h w) -> b c h w',h=h1,w=w1)
        ####### reverse to image ########
        x3 = patch_reverse(patches1_reverse,self.patch_size,h,w)
        x_g1 = torch.mul(x2_1,x3)

        #####stage2###
        x4 = k1@x2_2@k2
        patches2 = patch_partition(x4,self.patch_size)
        _,_,h2,w2 = patches2.size()
        patches2_1 = einops.rearrange(patches2,'b c h w -> b c (h w)')
        patches2_spatial = self.linear2(patches2_1)
        patches2_reverse = einops.rearrange(patches2_spatial,'b c (h w) -> b c h w',h=h2,w=w2)
        x5 = patch_reverse(patches2_reverse,self.patch_size,h,w)
        x6 = k1@x5@k2
        x_g2 = torch.mul(x2_2,x6)

        ######stage3#####
        x7 = m1@x2_3@n1
        patches3 = patch_partition(x7,self.patch_size)
        _,_,h3,w3 = patches3.size()
        patches3_1 = einops.rearrange(patches3,'b c h w -> b c (h w)')
        patches3_spatial = self.linear3(patches3_1)
        patches3_reverse = einops.rearrange(patches3_spatial,'b c (h w) -> b c h w',h=h3,w=w3)
        x8 = patch_reverse(patches3_reverse,self.patch_size,h,w)
        x9 = m2@x8@n2
        x_g3 = torch.mul(x2_3,x9)

        ####fuse#####
        x_cat = torch.cat((x_g1,x_g2,x_g3),1)
        x_sp_fuse = self.fc2(x_cat)
        x_out1 = x_sp_fuse + x0

        x_cp = self.ffn(x_out1)
        x_out2 = x_cp + x_out1
        return x_out2

    def flops(self):
        h,w = self.image_size
        n = h*w/self.spatial
        flops = 0
        #matrix mul
        flops += 4*2*self.dim*h*w
        #norm1 and norm2
        flops += 2*h*w*self.dim
        #fc1 and fc2
        flops += 2*h*w*self.dim*self.dim*3
        #depthwise
        flops += h*w*(3*3)*self.dim*4
        #linear
        flops += 3*n*self.dim*self.spatial*self.spatial
        #fc3 and fc4
        flops += 2*h*w*self.dim*4*self.dim
        return flops

class MS_MLP_v1(nn.Module):
    def __init__(self, dim, image_size, patch_size=4):
        super(MS_MLP_v1,self).__init__()
        self.dim = dim
        self.image_size = (image_size,image_size)
        self.patch_size = patch_size
        self.fc1 = nn.Conv2d(dim,dim*2,kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(dim*2,dim,kernel_size=1,bias=False)
        self.ffn = nn.Sequential(LayerNorm2d(dim),
                                 nn.Conv2d(dim,dim*4,kernel_size=1,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim*4,kernel_size=3,padding=1,groups=dim*4,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim,kernel_size=1,bias=False))

        self.spatial = patch_size*patch_size

        self.linear1 = nn.Linear(self.spatial,self.spatial)
        self.linear2 = nn.Linear(self.spatial,self.spatial)
        self.norm = LayerNorm2d(dim)


    def forward(self,x):
        x0 = x[0]
        m1 = x[1].cuda()
        m2 = x[2].cuda()
        _,_,h,w = x0.size()
        x1 = self.norm(x0)
        x2 = self.fc1(x1)
        x2_1,x2_2 = torch.chunk(x2,2,1)
        ######### multi_scale_fuse ##############
        #####stage1#####
        patches0 = patch_partition(x2_1,self.patch_size) #patches = 8*8
        _,_,h1,w1 = patches0.size()
        ##### patch to token ####
        patches1 = einops.rearrange(patches0,'b c h w -> b c (h w)')
        ###### mlp in spatial#####
        patches1_spatial = self.linear1(patches1)        
        ### token to patch ######
        patches1_reverse = einops.rearrange(patches1_spatial,'b c (h w) -> b c h w',h=h1,w=w1)
        ####### reverse to image ########
        x3 = patch_reverse(patches1_reverse,self.patch_size,h,w)
        x_g1 = torch.mul(x2_1,x3)

        #####stage2###
        x4 = m1@x2_2@m2
        patches2 = patch_partition(x4,self.patch_size)
        _,_,h2,w2 = patches2.size()
        patches2_1 = einops.rearrange(patches2,'b c h w -> b c (h w)')
        patches2_spatial = self.linear2(patches2_1)
        patches2_reverse = einops.rearrange(patches2_spatial,'b c (h w) -> b c h w',h=h2,w=w2)
        x5 = patch_reverse(patches2_reverse,self.patch_size,h,w)
        x6 = m1@x5@m2
        x_g2 = torch.mul(x2_2,x6)
        ####fuse#####
        x_cat = torch.cat((x_g1,x_g2),1)
        x_sp_fuse = self.fc2(x_cat)
        x_out1 = x_sp_fuse + x0

        x_cp = self.ffn(x_out1)
        x_out2 = x_cp + x_out1
        return [x_out2,m1,m2]

    def flops(self):
        h,w = self.image_size
        n = h*w/self.spatial
        flops = 0
        #matrix mul
        flops += 2*2*self.dim*h*w
        #norm1 and norm2
        flops += 2*h*w*self.dim
        #fc1 and fc2
        flops += 2*h*w*self.dim*self.dim*3
        #depthwise
        flops += h*w*(3*3)*self.dim*4
        #linear
        flops += 2*n*self.dim*self.spatial*self.spatial
        #fc3 and fc4
        flops += 2*h*w*self.dim*4*self.dim
        return flops

class MS_MLP_tail_v1(nn.Module):
    def __init__(self, dim, image_size, patch_size=4):
        super(MS_MLP_tail_v1,self).__init__()
        self.dim = dim
        self.image_size = (image_size,image_size)
        self.patch_size = patch_size
        self.fc1 = nn.Conv2d(dim,dim*2,kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(dim*2,dim,kernel_size=1,bias=False)
        self.ffn = nn.Sequential(LayerNorm2d(dim),
                                 nn.Conv2d(dim,dim*4,kernel_size=1,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim*4,kernel_size=3,padding=1,groups=dim*4,bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim*4,dim,kernel_size=1,bias=False))

        self.spatial = patch_size*patch_size

        self.linear1 = nn.Linear(self.spatial,self.spatial)
        self.linear2 = nn.Linear(self.spatial,self.spatial)
        self.norm = LayerNorm2d(dim)


    def forward(self,x):
        x0 = x[0]
        m1 = x[1].cuda()
        m2 = x[2].cuda()
        _,_,h,w = x0.size()
        x1 = self.norm(x0)
        x2 = self.fc1(x1)
        x2_1,x2_2 = torch.chunk(x2,2,1)
        ######### multi_scale_fuse ##############
        #####stage1#####
        patches0 = patch_partition(x2_1,self.patch_size) #patches = 8*8
        _,_,h1,w1 = patches0.size()
        ##### patch to token ####
        patches1 = einops.rearrange(patches0,'b c h w -> b c (h w)')
        ###### mlp in spatial#####
        patches1_spatial = self.linear1(patches1)        
        ### token to patch ######
        patches1_reverse = einops.rearrange(patches1_spatial,'b c (h w) -> b c h w',h=h1,w=w1)
        ####### reverse to image ########
        x3 = patch_reverse(patches1_reverse,self.patch_size,h,w)
        x_g1 = torch.mul(x2_1,x3)

        #####stage2###
        x4 = m1@x2_2@m2
        patches2 = patch_partition(x4,self.patch_size)
        _,_,h2,w2 = patches2.size()
        patches2_1 = einops.rearrange(patches2,'b c h w -> b c (h w)')
        patches2_spatial = self.linear2(patches2_1)
        patches2_reverse = einops.rearrange(patches2_spatial,'b c (h w) -> b c h w',h=h2,w=w2)
        x5 = patch_reverse(patches2_reverse,self.patch_size,h,w)
        x6 = m1@x5@m2
        x_g2 = torch.mul(x2_2,x6)
        ####fuse#####
        x_cat = torch.cat((x_g1,x_g2),1)
        x_sp_fuse = self.fc2(x_cat)
        x_out1 = x_sp_fuse + x0

        x_cp = self.ffn(x_out1)
        x_out2 = x_cp + x_out1
        return x_out2

    def flops(self):
        h,w = self.image_size
        n = h*w/self.spatial
        flops = 0
        #matrix mul
        flops += 2*2*self.dim*h*w
        #norm1 and norm2
        flops += 2*h*w*self.dim
        #fc1 and fc2
        flops += 2*h*w*self.dim*self.dim*3
        #depthwise
        flops += h*w*(3*3)*self.dim*4
        #linear
        flops += 2*n*self.dim*self.spatial*self.spatial
        #fc3 and fc4
        flops += 2*h*w*self.dim*4*self.dim
        return flops