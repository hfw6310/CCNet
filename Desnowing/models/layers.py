import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.SCSA import SCSA
# from models.ddpconv import DynamicDepthwiseConv2d



### CSU module
class SpanConvUX(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        self.pixel_norm = nn.LayerNorm(in_channels)  # channel-wise
        self.acti = nn.GELU()

        # pointwise1
        self.pw1=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise1
        self.dw1 = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

        # pointwise2
        self.pw2=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

    def forward(self, fea):
        
        fea = fea.permute(0, 2, 3, 1)  # (B, H, W, C)
        fea = self.pixel_norm(fea)
        fea = fea.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        fea1 = self.pw1(fea)
        fea1 = self.dw1(fea1)
        fea1 = self.acti(fea1)

        fea2 = self.pw2(fea)

        return fea1 * fea2

class BasicConv1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv1, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                # nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
                SpanConvUX(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
                )
            
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
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
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        
### ERSM / RSAM
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv1(in_channel, out_channel, kernel_size=9, stride=1, relu=True)
        # self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:

            self.cubic_11 = cubic_attention(in_channel//2, group=1, kernel=11)
            self.cubic_7 = cubic_attention(in_channel//2, group=1, kernel=7)
            
        self.filter = filter
        self.acti = nn.GELU()
    def forward(self, x):
        out = self.conv1(x)
        
        if self.filter:
            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            out = torch.cat((out_11, out_7), dim=1)

        

        out = self.conv2(out)

        return out + x
    
class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = nn.Sequential(
                                    spatial_strip_att(dim, group=group, kernel=kernel),
                                    spatial_strip_att_dilated(dim, group=group, kernel=kernel, dilated=(kernel+1)//2)
                                    ) # dilated=(kernel+1)//2
        self.W_spatial_att = nn.Sequential(
                                    spatial_strip_att(dim, group=group, kernel=kernel, H=False),
                                    spatial_strip_att_dilated(dim, group=group, kernel=kernel, H=False, dilated=(kernel+1)//2)
                                    ) # 
        
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))
    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta # out 

class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=1, H=True, dilated=1) -> None:
        super().__init__()

        self.k0 = kernel
        self.dilated =  dilated
        pad = dilated * (kernel - 1) // 2
        self.k = pad * 2 + 1
        self.H = H
        self.kernel = (self.k, 1) if H else (1, self.k) 
        self.pad = pad
        self.group = group
        self.groups = dim//group
        self.padding = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x): 
        filter = self.ap(x)
        filter = self.conv(filter)     
        n, c, h, w = x.shape
        x = self.padding(x)

        if self.H:
            filter = filter.reshape(n, self.group, 1, self.k0).repeat((1,c//self.group,1,1)).reshape(n*c, 1, 1, self.k0)
            h_pad = h
            w_pad = w+2*self.pad
        else:
            filter = filter.reshape(n, self.group, self.k0, 1).repeat((1,c//self.group,1,1)).reshape(n*c, 1, self.k0, 1)
            h_pad = h+2*self.pad
            w_pad = w
        filter = self.filter_act(filter)
        # print('x.shape:',x.shape)
        # print('x[0,0,0,:,0]:',x[0,0,0,:,0])
        # print('filter1.shape:',filter.shape) 
        # print('x.reshape(1, -1, h_pad, w_pad).shape:',x.reshape(1, -1, h_pad, w_pad).shape)

        out = F.conv2d(x.reshape(1, -1, h_pad, w_pad), weight=filter, bias=None, stride=1, 
                            padding=0,dilation=self.dilated,groups=n*c)
        # print('out.shape:',out.shape)  
        return out.view(n, c, h, w)

class spatial_strip_att_dilated(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True, dilated=2) -> None:
        super().__init__()

        self.k0 = kernel
        self.dilated =  dilated
        pad = dilated * (kernel - 1) // 2
        self.k = pad * 2 + 1
        self.H = H
        self.kernel = (self.k, 1) if H else (1, self.k) 
        self.pad = pad
        self.group = group
        self.groups = dim//group
        self.padding = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x): 
        filter = self.ap(x)
        filter = self.conv(filter)     
        n, c, h, w = x.shape
        x = self.padding(x)

        if self.H:
            filter = filter.reshape(n, self.group, 1, self.k0).repeat((1,c//self.group,1,1)).reshape(n*c, 1, 1, self.k0)
            h_pad = h
            w_pad = w+2*self.pad
        else:
            filter = filter.reshape(n, self.group, self.k0, 1).repeat((1,c//self.group,1,1)).reshape(n*c, 1, self.k0, 1)
            h_pad = h+2*self.pad
            w_pad = w
        filter = self.filter_act(filter)
        # print('x.shape:',x.shape)
        # print('x[0,0,0,:,0]:',x[0,0,0,:,0])
        # print('filter1.shape:',filter.shape) 
        # print('x.reshape(1, -1, h_pad, w_pad).shape:',x.reshape(1, -1, h_pad, w_pad).shape)

        out = F.conv2d(x.reshape(1, -1, h_pad, w_pad), weight=filter, bias=None, stride=1, 
                            padding=0,dilation=self.dilated,groups=n*c)
        # print('out.shape:',out.shape)  
        return out.view(n, c, h, w)
