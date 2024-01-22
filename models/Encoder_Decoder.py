from email.mime import base
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from scipy.linalg import block_diag

def matrix_construct_1(x):
    matrix = torch.tensor([[0,0,1,0],
                           [0,0,0,1],
                           [1,0,0,0],
                           [0,1,0,0]])
    _,_,h,w = x.size()
    row = h // 4 
    col = w // 4 
    list1 = []
    list2 = []

    for i in range(row):
        list1.append(matrix)
    matrix_row_1 = torch.from_numpy(block_diag(*list1)).to(torch.float32)
    del list1[0:row//2]
    matrix_row_2 = torch.from_numpy(block_diag(*list1)).to(torch.float32)
    del list1[0:row//4]
    matrix_row_3 = torch.from_numpy(block_diag(*list1)).to(torch.float32)
    del list1[0:row//8]
    matrix_row_4 = torch.from_numpy(block_diag(*list1)).to(torch.float32)
    
    for j in range(col):
        list2.append(matrix)  
    matrix_col_1 = torch.from_numpy(block_diag(*list2)).to(torch.float32)
    del list2[0:col//2]
    matrix_col_2 = torch.from_numpy(block_diag(*list2)).to(torch.float32)
    del list2[0:col//4]
    matrix_col_3 = torch.from_numpy(block_diag(*list2)).to(torch.float32)
    del list2[0:col//8]
    matrix_col_4 = torch.from_numpy(block_diag(*list2)).to(torch.float32)

    return [matrix_row_1,matrix_col_1],[matrix_row_2,matrix_col_2],[matrix_row_3,matrix_col_3],[matrix_row_4,matrix_col_4]

def matrix_construct_2(a):
    matrix = torch.tensor([[0,0,1,0],
                           [0,0,0,1],
                           [1,0,0,0],
                           [0,1,0,0]])
    matrix1 = torch.eye(2)
    _,_,h,w = a.size()
    row = h // 4 - 1
    col = w // 4 - 1
    list1 = []
    list2 = []

    for i in range(row):
        list1.append(matrix)
    matrix_row_1 = torch.from_numpy(block_diag(matrix1,*list1,matrix1)).to(torch.float32)
    del list1[0:row//2+1]
    matrix_row_2 = torch.from_numpy(block_diag(matrix1,*list1,matrix1)).to(torch.float32)
    del list1[0:row//4+1]
    matrix_row_3 = torch.from_numpy(block_diag(matrix1,*list1,matrix1)).to(torch.float32)
    del list1[0:row//8+1]
    matrix_row_4 = torch.from_numpy(block_diag(matrix1,*list1,matrix1)).to(torch.float32)
    
    for j in range(col):
        list2.append(matrix)  
    matrix_col_1 = torch.from_numpy(block_diag(matrix1,*list2,matrix1)).to(torch.float32)
    del list2[0:col//2+1]
    matrix_col_2 = torch.from_numpy(block_diag(matrix1,*list2,matrix1)).to(torch.float32)
    del list2[0:col//4+1]
    matrix_col_3 = torch.from_numpy(block_diag(matrix1,*list2,matrix1)).to(torch.float32)
    del list2[0:col//8+1]
    matrix_col_4 = torch.from_numpy(block_diag(matrix1,*list2,matrix1)).to(torch.float32)
    
    return [matrix_row_1,matrix_col_1],[matrix_row_2,matrix_col_2],[matrix_row_3,matrix_col_3],[matrix_row_4,matrix_col_4]

class EBlock_fuse(nn.Module):
    def __init__(self, out_channel, num_res, image_size):
        super(EBlock_fuse, self).__init__()
        self.num_res = num_res
        layers = [MS_MLP(out_channel, image_size) for _ in range(self.num_res-1)]
        self.block = MS_MLP_tail(out_channel, image_size)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.num_res > 1:
            [x1,m1,n1,m2,n2,k1,k2] = self.layers(x)
            x2 = self.block([x1,m1,n1,m2,n2,k1,k2])
            return x2
        
        if self.num_res == 1:
            return self.block(x)
        
    def flops(self):
        flops = 0
        if self.num_res == 1:
            flops += self.block.flops()
            return flops
        else:
            for blk in self.layers:
                flops += blk.flops()
            flops += self.block.flops()
            return flops
        
class DBlock_fuse(nn.Module):
    def __init__(self, channel, num_res, image_size):
        super(DBlock_fuse, self).__init__()
        self.num_res = num_res
        layers = [MS_MLP(channel, image_size) for _ in range(self.num_res-1)]
        self.block = MS_MLP_tail(channel, image_size)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.num_res > 1:
            [x1,m1,n1,m2,n2,k1,k2] = self.layers(x)
            x2 = self.block([x1,m1,n1,m2,n2,k1,k2])
            return x2
        
        if self.num_res == 1:
            return self.block(x)

    def flops(self):
        flops = 0
        if self.num_res == 1:
            flops += self.block.flops()
            return flops
        else:
            for blk in self.layers:
                flops += blk.flops()
            flops += self.block.flops()
            return flops


class EBlock_fuse_v1(nn.Module):
    def __init__(self, out_channel, num_res, image_size):
        super(EBlock_fuse_v1, self).__init__()
        self.num_res = num_res
        layers = [MS_MLP_v1(out_channel, image_size) for _ in range(self.num_res-1)]
        self.block = MS_MLP_tail_v1(out_channel, image_size)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.num_res > 1:
            [x1,k1,k2] = self.layers(x)
            x2 = self.block([x1,k1,k2])
            return x2
        
        if self.num_res == 1:
            return self.block(x)

    def flops(self):
        flops = 0
        if self.num_res == 1:
            flops += self.block.flops()
            return flops
        else:
            for blk in self.layers:
                flops += blk.flops()
            flops += self.block.flops()
            return flops


        
class DBlock_fuse_v1(nn.Module):
    def __init__(self, channel, num_res, image_size):
        super(DBlock_fuse_v1, self).__init__()
        self.num_res = num_res
        layers = [MS_MLP_v1(channel, image_size) for _ in range(self.num_res-1)]
        self.block = MS_MLP_tail_v1(channel, image_size)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.num_res > 1:
            [x1,k1,k2] = self.layers(x)
            x2 = self.block([x1,k1,k2])
            return x2
        
        if self.num_res == 1:
            return self.block(x)

    def flops(self):
        flops = 0
        if self.num_res == 1:
            flops += self.block.flops()
            return flops
        else:
            for blk in self.layers:
                flops += blk.flops()
            flops += self.block.flops()
            return flops

class UNet(nn.Module):
    def __init__(self, image_size = 256, base_channel = 42, num_res=[2,4,12,2]):
        super(UNet, self).__init__()
        self.image_size = (image_size,image_size)
        self.base_channel = base_channel

        self.Encoder = nn.ModuleList([
            EBlock_fuse(base_channel,num_res[0], image_size),
            EBlock_fuse(base_channel*2, num_res[1], image_size//2),
            EBlock_fuse_v1(base_channel*4, num_res[2], image_size//4),
            EBlock_fuse_v1(base_channel*8, num_res[3], image_size//8)
        ])


        self.downsample = nn.ModuleList([
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*8,kernel_size=3, relu=True, stride=2)

        ])

        self.Decoder = nn.ModuleList([
            DBlock_fuse_v1(base_channel * 8, num_res[3],image_size//8),
            DBlock_fuse_v1(base_channel * 4, num_res[2],image_size//4),
            DBlock_fuse(base_channel * 2, num_res[1],image_size//2),
            DBlock_fuse(base_channel,num_res[0],image_size)
        ])
        
        self.upsample = nn.ModuleList([
            nn.Sequential(  nn.Conv2d(base_channel*8, base_channel*16, 1, bias=False),
                            nn.PixelShuffle(2)),

            nn.Sequential(  nn.Conv2d(base_channel*4, base_channel*8, 1, bias=False),
                            nn.PixelShuffle(2)),

            nn.Sequential(  nn.Conv2d(base_channel*2, base_channel*4, 1, bias=False),
                            nn.PixelShuffle(2))      

        ])
        

        self.Convs = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])


    def forward(self, x):
        z = self.Convs[0](x)#3->32
        [m1,m2],[m3,m4],[m5,m6],[m7,m8] = matrix_construct_1(z)
        [n1,n2],[n3,n4],[n5,n6],[n7,n8] = matrix_construct_2(z)
        matrix1_raw = n1@m1
        matrix1_col = m2@n2
        matrix1_raw_back = m1@n1
        matrix1_col_back = n2@m2
        res1 = self.Encoder[0]([z,matrix1_raw,matrix1_col,matrix1_raw_back,matrix1_col_back,n1,n2])#32->32, 32,h

        z = self.downsample[0](res1)#32->64,h->h/2
        matrix2_raw = n3@m3
        matrix2_col = m4@n4
        matrix2_raw_back = m3@n3
        matrix2_col_back = n4@m4
        res2 = self.Encoder[1]([z,matrix2_raw,matrix2_col,matrix2_raw_back,matrix2_col_back,n3,n4])#64->64,h/2

        z = self.downsample[1](res2)#64->128,h/2->h/4
        #matrix3_raw = n5@m5
       # matrix3_col = m6@n6
        #matrix3_raw_back = m5@n5
        #matrix3_col_back = n6@m6
        res3 = self.Encoder[2]([z,n5,n6])#128->128,h/4

        z = self.downsample[2](res3)#128->256,h/4->h/8
       # matrix4_raw = n7@m7
       # matrix4_col = m8@n8
       # matrix4_raw_back = m7@n7
        #matrix4_col_back = n8@m8
        res4 = self.Encoder[3]([z,n7,n8])#256->256,h/8

        out1 = self.Decoder[0]([res4,n7,n8])#256->256
        z = self.upsample[0](out1)#256->128,h/8->h/4

        z = z + res3#h/4,128
        out2 = self.Decoder[1]([z,n5,n6])#h/4,128->128

        z = self.upsample[1](out2)#h/2,128->64
        
        z = z + res2#h/2,64
        out3 = self.Decoder[2]([z,matrix2_raw,matrix2_col,matrix2_raw_back,matrix2_col_back,n3,n4])#h/2,64->64

        z= self.upsample[2](out3)#h,64->32

        z = z + res1#h,32
        z = self.Decoder[3]([z,matrix1_raw,matrix1_col,matrix1_raw_back,matrix1_col_back,n1,n2])#32->32

        z = self.Convs[1](z)#32->3

        return z+x
    
    def flops(self):
        h,w = self.image_size
        flops = 0
        #convs
        flops += h*w*self.base_channel*(3*3*3)
        flops += h*w*3*(self.base_channel*3*3)
        #Encoders
        flops += self.Encoder[0].flops()
        flops += self.Encoder[1].flops()
        flops += self.Encoder[2].flops()
        flops += self.Encoder[3].flops()
        #Decoder
        flops += self.Decoder[0].flops()
        flops += self.Decoder[1].flops()
        flops += self.Decoder[2].flops()
        flops += self.Decoder[3].flops()
        #Downsample
        flops += (h/2)*(w/2)*self.base_channel*2*(self.base_channel*3*3)
        flops += (h/4)*(w/4)*self.base_channel*4*(self.base_channel*2*3*3)       
        flops += (h/8)*(w/8)*self.base_channel*8*(self.base_channel*4*3*3) 
        #Upsample
        flops += (h/8)*(w/8)*self.base_channel*16*(self.base_channel*8)
        flops += (h/4)*(w/4)*self.base_channel*8*(self.base_channel*4)       
        flops += (h/2)*(w/2)*self.base_channel*4*(self.base_channel*2) 
        print("Net_flops:%.2f"%(flops/1e9))
        return flops/1e9