import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # # depthwise2
        # self.dw2 = torch.nn.Conv2d(
        #         in_channels=out_channels,
        #         out_channels=out_channels,
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         dilation=dilation,
        #         groups=out_channels,
        #         bias=bias,
        #         padding_mode=padding_mode,
        # )

    def forward(self, fea):
        
        fea = fea.permute(0, 2, 3, 1)  # (B, H, W, C)
        fea = self.pixel_norm(fea)
        fea = fea.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        fea1 = self.pw1(fea)
        fea1 = self.dw1(fea1)
        fea1 = self.acti(fea1)

        fea2 = self.pw2(fea)
        # fea2 = self.dw2(fea2)

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
    
class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        # spatial attention
        sa_x = self.conv_sa(input_x)  
        # channel attention
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out  = sa_x + ca_x
        return out

# Adaptice Filter Generation 
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, 3)
        self.conv = nn.Conv2d(in_channels, in_channels*kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size, h, w])

        return filter_x

# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg_h = AFG(in_channels, kernel_size)
        self.afg_v = AFG(in_channels, kernel_size)
        self.unfold_h = nn.Unfold(kernel_size=(kernel_size,1), dilation=1, padding=(kernel_size//2,0), stride=1)
        self.unfold_v = nn.Unfold(kernel_size=(1,kernel_size), dilation=1, padding=(0,kernel_size//2), stride=1)
        self.filter_act = nn.Tanh()

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x_h = self.filter_act(self.afg_h(input_x))
        unfold_x_h = self.unfold_h(input_x).reshape(b, c, -1, h, w)
        out = (unfold_x_h * filter_x_h).sum(2)

        filter_x_v = self.filter_act(self.afg_v(out))
        unfold_x_v = self.unfold_v(out).reshape(b, c, -1, h, w)
        out = (unfold_x_v * filter_x_v).sum(2)

        return out
    
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

# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, filter=False):
#         super(ResBlock, self).__init__()
#         self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
#         self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
#         if filter:
#             # self.cubic_11 = cubic_attention(in_channel//2, group=1, kernel=11)
#             # self.cubic_7 = cubic_attention(in_channel//2, group=1, kernel=7)
#             # self.attention = attention(in_channel)

#             # self.cubic_self1 = LargeStripAttention(in_channel//2,11)
#             # self.cubic_self2 = LargeStripAttention(in_channel//2,7)
            
#             self.cubic_self1 = LargeStripAttention(in_channel//2,11)
#             self.cubic_self2 = LargeStripAttention(in_channel//2,7)

#             # self.attention = EMA(in_channel)
            
#         self.filter = filter
#         self.acti = nn.GELU()
#     def forward(self, x):
#         out = self.conv1(x)
        
#         if self.filter:
#             # out = torch.chunk(out, 2, dim=1)
#             # out_11 = self.cubic_11(out[0])
#             # out_7 = self.cubic_7(out[1])
#             # out = torch.cat((out_11, out_7), dim=1)

#             # out = self.attention(out)

#             out = torch.chunk(out, 2, dim=1)
#             out_11 = self.cubic_self1(out[0])
#             out_7 = self.cubic_self2(out[1])
#             out = torch.cat((out_11, out_7), dim=1)

#         # out = self.conv2(out)
        
#             # out = self.attention(out)
#         out = self.conv2(out)

#         return out + x

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv1(in_channel, out_channel, kernel_size=9, stride=1, relu=True)
        # self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:
            # self.cubic_11 = DyConv(in_channel//2, 11)
            # self.cubic_7 = DyConv(in_channel//2, 7)

            self.cubic_11 = cubic_attention(in_channel//2, group=1, kernel=11)
            self.cubic_7 = cubic_attention(in_channel//2, group=1, kernel=7)

            # self.attention = attention(in_channel)

            # self.cubic_self1 = LargeStripAttention(in_channel//2,11)
            # self.cubic_self2 = LargeStripAttention(in_channel//2,7)
            
            # self.cubic_self1 = LargeStripAttention(in_channel//2,11)
            # self.cubic_self2 = LargeStripAttention(in_channel//2,7)
            # self.attention = Attention(in_channel)
            
            # self.attention = EMA(in_channel)
            
        self.filter = filter
        self.acti = nn.GELU()
    def forward(self, x):
        out = self.conv1(x)
        
        if self.filter:
            # out1 = out.clone()
            # out1 = self.attention(out1)

            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            out = torch.cat((out_11, out_7), dim=1)

            # out = self.attention(out)


            # out = torch.chunk(out, 2, dim=1)
            # out_11 = self.cubic_self1(out[0])
            # out_7 = self.cubic_self2(out[1])
            # out = torch.cat((out_11, out_7), dim=1)

            # out = self.attention(out)

            # out = out+out1
            # out = out1
        # out = self.conv2(out)
        

        out = self.conv2(out)

        return out + x
    
class LargeStripAttention(nn.Module):
    def __init__(self, dim,kernel):
        super().__init__()
        # self.pointwise = nn.Conv2d(dim, dim, 1)
        # self.depthwise = cubic_attention(dim, group=1, kernel=kernel)
        # self.depthwise = cubic_attention(dim, group=dim, kernel=7)

        self.scsa = SCSA(dim)

    def forward(self, x):
        # u = x.clone()
        # attn = self.pointwise(x)
        # attn = self.depthwise(x)

        attn = self.scsa(x)

        return attn # u * attn
    
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        # self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

        self.depthwise = DynamicDepthwiseConv2d(dim, 5, dilation=1)
        self.depthwise_dilated = DynamicDepthwiseConv2d(dim, 5, dilation=3)
        

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn
        
# class LargeStripAttention(nn.Module):
#     def __init__(self, dim,kernel):
#         super().__init__()
#         # self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = cubic_attention(dim, group=1, kernel=kernel)
#         # self.depthwise = cubic_attention(dim, group=dim, kernel=7)
        

#     def forward(self, x):
#         # u = x.clone()
#         # attn = self.pointwise(x)
#         attn = self.depthwise(x)
#         return attn # u * attn
    
# class attention(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()

#         ker = 7
#         dilation = ker//2 + 1 + 1
#         pad = ker // 2 * dilation

#         ker1  = 7
#         dilation1 = ker1//2 + 1 
#         pad1 = ker1 // 2 * dilation1
        
#         self.in_conv = nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
#                     # nn.GELU()
#                     )
#         #self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
#         # self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,ker // 2), stride=1, groups=dim)
#         # self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(ker // 2,0), stride=1, groups=dim)
#         # self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=ker // 2, stride=1, groups=dim)
        
#         self.dw_13 = nn.Sequential(
#                         nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,ker//2), stride=1, groups=dim),
#                         nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim, dilation=dilation)
#                         )
            
#         self.dw_31 = nn.Sequential(
#                         nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(ker//2,0), stride=1, groups=dim),
#                         nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim, dilation=dilation)
#                         )
        
#         self.dw_33 = nn.Sequential(
#                         nn.Conv2d(dim, dim, kernel_size=ker1, padding=ker1//2, stride=1, groups=dim),
#                         nn.Conv2d(dim, dim, kernel_size=ker1, padding=pad1, stride=1, groups=dim, dilation=dilation1)
#                         )
#         # self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

#         self.act = nn.ReLU()

#     def forward(self, x):
#         out = self.in_conv(x)
        
#         out =  x * self.dw_33(out)  + x * self.dw_31(self.dw_13(out))# + x * self.dw_11(out)
        
#         # out = x * self.dw_11(out)  # x * self.dw_13(out) + x * self.dw_31(out) +
#         # out = self.dw_13(out)
#         # out = self.dw_31(out)
#         # out = x * out
#         # out = x * self.dw_33(out)

#         # out =  self.dw_13(x) + self.dw_31(x) + self.dw_33(x) # + self.dw_11(x) 
#         # out = x * self.in_conv(out)



#         # out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
#         # out = self.act(out)

#         return out # self.out_conv(out) # out # 
    
# class cubic_attention(nn.Module):
#     def __init__(self, dim, group, kernel) -> None:
#         super().__init__()

#         self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
#         self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
#         self.gamma = nn.Parameter(torch.zeros(dim,1,1))
#         self.beta = nn.Parameter(torch.ones(dim,1,1))

#     def forward(self, x):
#         out = self.H_spatial_att(x)
#         out = self.W_spatial_att(out)
#         return self.gamma * out + x * self.beta

# class cubic_attention(nn.Module):
#     def __init__(self, dim, group, kernel) -> None:
#         super().__init__()

#         self.H_spatial_att = nn.Sequential(#spatial_strip_att(dim, group=group, kernel=kernel),
#                                     spatial_strip_att_dilated(dim, group=group, kernel=kernel,dilated=3)) # dilated=(kernel+1)//2
#         self.W_spatial_att = nn.Sequential(#spatial_strip_att(dim, group=group, kernel=kernel, H=False),
#                                     spatial_strip_att_dilated(dim, group=group, kernel=kernel, H=False,dilated=3)) # (kernel+1)//2
        
#         self.gamma = nn.Parameter(torch.zeros(dim,1,1))
#         self.beta = nn.Parameter(torch.ones(dim,1,1))

#     def forward(self, x):
#         out = self.H_spatial_att(x)
#         out = self.W_spatial_att(out)
#         return self.gamma * out + x * self.beta

# class cubic_attention(nn.Module):
#     def __init__(self, dim, group, kernel) -> None:
#         super().__init__()

#         self.H_spatial_att = nn.Sequential(spatial_strip_att(dim, group=group, kernel=kernel),
#                                     spatial_strip_att_dilated(dim, group=group, kernel=kernel,dilated=(kernel+1)//2)) # 
#         self.W_spatial_att = nn.Sequential(spatial_strip_att(dim, group=group, kernel=kernel, H=False),
#                                     spatial_strip_att_dilated(dim, group=group, kernel=kernel, H=False,dilated=(kernel+1)//2)) # 
        
#         self.gamma = nn.Parameter(torch.zeros(dim,1,1))
#         self.beta = nn.Parameter(torch.ones(dim,1,1))

#     def forward(self, x):
#         out = self.H_spatial_att(x)
#         out = self.W_spatial_att(out)
#         return self.gamma * out + x * self.beta

# class cubic_attention(nn.Module):
#     def __init__(self, dim, group, kernel) -> None:
#         super().__init__()

#         self.H_spatial_att = nn.Sequential(#spatial_strip_att(dim, group=group, kernel=kernel),
#                                     large_spatial_strip_att_dilated(dim, group=group, kernel=kernel, dilated=3)) # dilated=(kernel+1)//2
#         self.W_spatial_att = nn.Sequential(#spatial_strip_att(dim, group=group, kernel=kernel, H=False),
#                                     large_spatial_strip_att_dilated(dim, group=group, kernel=kernel, H=False, dilated=3)) # (kernel+1)//2
#         self.gamma = nn.Parameter(torch.zeros(dim,1,1))
#         self.beta = nn.Parameter(torch.ones(dim,1,1))
#     def forward(self, x):
#         out = self.H_spatial_att(x)
#         out = self.W_spatial_att(out)
#         return self.gamma * out + x * self.beta # out 

class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        # self.H_spatial_att = nn.Sequential(
        #                             spatial_strip_att(dim, group=group, kernel=kernel),
        #                             #large_spatial_strip_att_dilated(dim, group=group, kernel=kernel, dilated=3)
        #                             ) # dilated=(kernel+1)//2
        # self.W_spatial_att = nn.Sequential(
        #                             spatial_strip_att(dim, group=group, kernel=kernel, H=False),
        #                             #large_spatial_strip_att_dilated(dim, group=group, kernel=kernel, H=False, dilated=3)
        #                             ) # (kernel+1)//2

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
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out

# class large_spatial_strip_att_dilated(nn.Module):
#     def __init__(self, dim, kernel=5, group=2, H=True, dilated=2) -> None:
#         super().__init__()
#         # self.dw_conv0 = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, stride=1,groups=dim)
#         self.dw_conv = nn.Conv2d(dim, dim, kernel, padding=(kernel//2+1)*(kernel - 1) // 2, stride=1,groups=dim,dilation=kernel//2+1)
#         self.k = kernel
#         pad = kernel // 2
#         self.kernel = (1, kernel) if H else (kernel, 1)
#         self.padding = (kernel//2, 1) if H else (1, kernel//2)

#         self.group = group
#         self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
#         self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
#         self.ap = nn.AdaptiveAvgPool2d((1, 1))
#         self.filter_act = nn.Sigmoid()

#     def forward(self, x):
#         x = self.dw_conv(x)
#         filter = self.ap(x)
#         filter = self.conv(filter)
#         n, c, h, w = x.shape
#         x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
#         n, c1, p, q = filter.shape
#         filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
#         filter = self.filter_act(filter)
#         out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
#         # out = self.dw_conv0(x)

#         return out
    
def interleave(aa,bb):
    # print('aa[0,0,0,:,0]:', aa[0,0,0,:,0])
    shape = aa.size()
    # print('shape:',shape)
    aa = aa.view(-1,1)
    bb = bb.view(-1,1)
    
    new_tensor = torch.stack((aa,bb),dim=1).view(shape[0],shape[1],shape[2],shape[3]*2,shape[4])
    
    # print('new_tensor[0,0,0,:,0]:', new_tensor[0,0,0,:,0])
    return new_tensor[:,:,:,:shape[3]*2-1,:]

# class spatial_strip_att_dilated(nn.Module):
#     def __init__(self, dim, kernel=5, group=2, H=True, dilated=2) -> None:
#         super().__init__()

#         self.k0 = kernel
#         self.dilated = dilated
#         pad = dilated * (kernel - 1) // 2
#         self.k = pad * 2 + 1

#         self.kernel = (1, self.k) if H else (self.k, 1)
#         # self.padding = (kernel//2, 1) if H else (1, kernel//2)

#         self.group = group
#         self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
#         self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
#         self.ap = nn.AdaptiveAvgPool2d((1, 1))
#         self.filter_act = nn.Sigmoid()

#     def forward(self, x):
#         filter = self.ap(x)
#         filter = self.conv(filter)
#         # print('x.shape:',x.shape)        
#         n, c, h, w = x.shape
#         x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
#         n, c1, p, q = filter.shape
#         # print('filter0.shape:',filter.shape)
#         filter = filter.reshape(n, c1//self.k0, self.k0, p*q).unsqueeze(2)
#         filter = self.filter_act(filter)
#         # print('x.shape:',x.shape)
#         # print('x[0,0,0,:,0]:',x[0,0,0,:,0])
        
#         # x = x[:,:,:,::self.dilated,:] 

#         ### for dilated strip
#         filter_dilated = torch.zeros(n, c1//self.k0, 1, self.dilated*(self.k0-1)+1, p*q).to("cuda")
#         filter_dilated[:,:,:,::self.dilated,:] = filter
#         # filter = filter.cpu()
#         filter = filter_dilated
        
#         # print('x1[0,0,0,:,0]:',x[0,0,0,:,0])
#         # print('filter[0,0,0,:,0]:',filter[0,0,0,:,0])

#         out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
#         return out

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

class large_spatial_strip_att_dilated(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True, dilated=2) -> None:
        super().__init__()
        
        self.dw_conv = nn.Conv2d(dim, dim, 2*(dilated-1)+1, padding=dilated-1, stride=1,groups=dim)

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
        x = self.dw_conv(x)
        filter = self.ap(x)
        
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
        # print('self.dilated:',self.dilated)

        out = F.conv2d(x.reshape(1, -1, h_pad, w_pad), weight=filter, bias=None, stride=1, 
                            padding=0,dilation=self.dilated,groups=n*c)
        # print('out.shape:',out.shape)  
        return out.view(n, c, h, w)

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = dynamic_conv(channels // self.groups, kernel_size=1, stride=1)
        self.conv3x3 = dynamic_conv(channels // self.groups, kernel_size=3, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class dynamic_conv(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=1):
        super(dynamic_conv, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)
        out = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        return out   
    
    