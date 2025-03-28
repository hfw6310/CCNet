
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        out_channel = int(out_channel)
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        #layers = [DRB(out_channel) for _ in range(num_res-1)]

        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        channel = int(channel)
        # layers = [pointConvsAndScConv(channel, channel) for _ in range(num_res-1)]
        # layers = [DRB(channel) for _ in range(num_res-1)]
        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        out_plane = int(out_plane)
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class SANet(nn.Module):
    def __init__(self,base_channel_=32, num_res=4):
        super(SANet, self).__init__()

        base_channel = base_channel_

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2), # 1
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2), # 2
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True), # 3
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), # 4
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        # print('x.shape:',x.shape)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        # print('z.shape:', z.shape)
        z = self.FAM2(z, z2)
        # print('z.shape:', z.shape)
        res2 = self.Encoder[1](z)
        # print('res2.shape:', res2.shape)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

# class SANet(nn.Module):
#     def __init__(self, base_channel_=32, num_res=4):
#         super(SANet, self).__init__()

#         base_channel = base_channel_

#         self.Encoder = nn.ModuleList([
#             EBlock(base_channel, num_res),
#             EBlock(base_channel*0.5, num_res),
#             EBlock(base_channel*0.25, num_res),
#         ])

#         self.feat_extract = nn.ModuleList([
#             BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
#             BasicConv(base_channel, base_channel*0.5, kernel_size=4, relu=True, stride=2, transpose=True), # 1
#             BasicConv(base_channel*0.5, base_channel*0.25, kernel_size=4, relu=True, stride=2, transpose=True), # 2
#             BasicConv(base_channel*0.25, base_channel*0.5, kernel_size=3, relu=True, stride=2), # 3
#             BasicConv(base_channel*0.5, base_channel, kernel_size=3, relu=True, stride=2), # 4

#             BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
#         ])

#         self.Decoder = nn.ModuleList([
#             DBlock(base_channel * 0.25, num_res),
#             DBlock(base_channel * 0.5, num_res),
#             DBlock(base_channel * 1, num_res)
#         ])

#         self.Convs = nn.ModuleList([
#             BasicConv(base_channel * 1, base_channel * 0.5, kernel_size=1, relu=True, stride=1),
#             BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
#         ])

#         self.ConvsOut = nn.ModuleList(
#             [
#                 BasicConv(base_channel * 0.25, 3, kernel_size=3, relu=False, stride=1),
#                 BasicConv(base_channel * 0.5, 3, kernel_size=3, relu=False, stride=1),
#             ]
#         )

#         self.FAM1 = FAM(base_channel * 0.25)
#         self.SCM1 = SCM(base_channel * 0.25)
#         self.FAM2 = FAM(base_channel * 0.5)
#         self.SCM2 = SCM(base_channel * 0.5)

#     def forward(self, x):
#         # print('x.shape:',x.shape)
#         x_2 = F.interpolate(x, scale_factor=2)
#         x_4 = F.interpolate(x_2, scale_factor=2)
#         z2 = self.SCM2(x_2) 
#         z4 = self.SCM1(x_4)

#         outputs = list()
#         # 256
#         x_ = self.feat_extract[0](x)
#         res1 = self.Encoder[0](x_)
#         # 512
#         z = self.feat_extract[1](res1) # 0.5 * base_channel
#         z = self.FAM2(z, z2) # 0.5
#         res2 = self.Encoder[1](z) # 0.5
#         # 1024
#         z = self.feat_extract[2](res2)
#         z = self.FAM1(z, z4) # 0.25
#         z = self.Encoder[2](z) # 0.25

#         z = self.Decoder[0](z) # 0.25
#         z_ = self.ConvsOut[0](z) 
#         # 512
#         z = self.feat_extract[3](z)
#         outputs.append(z_+x_4)

#         z = torch.cat([z, res2], dim=1)
#         # print('z.shape:',z.shape)
#         z = self.Convs[0](z) # 0.5
        
#         z = self.Decoder[1](z) # 0.5
#         z_ = self.ConvsOut[1](z)
#         # 256
#         # print('z.shape:',z.shape)
#         z = self.feat_extract[4](z)
#         outputs.append(z_+x_2)
        

#         z = torch.cat([z, res1], dim=1) 
#         z = self.Convs[1](z)
#         z = self.Decoder[2](z)
#         z = self.feat_extract[5](z)
#         outputs.append(z+x)

#         return outputs

def build_net(base_channel_):
    return SANet(base_channel_)
