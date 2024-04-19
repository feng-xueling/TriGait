# -*- coding: utf-8 -*-
"""
   File Name：     smplgait
   Author :       jinkai Zheng
   E-mail:        zhengjinkai3@hdu.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper

from torchvision.utils import save_image



class backbone_3d(nn.Module):
    def __init__(self,_in_channels = 1, _channels = [32, 64, 128,128]):
        super(backbone_3d, self).__init__()
        # 3D Convolution
        self.conv3d_1 = Conv3d(_in_channels, _channels[0])
        self.LTA = nn.Sequential(
            BasicConv3d(_channels[0], _channels[0], kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3d_2 = Conv3d(_channels[0], _channels[1])
        self.conv3d_3 = Conv3d(_channels[1], _channels[2])
        self.conv3d_4 = Conv3d(_channels[2], _channels[3])

    def forward(self,x):
        x = self.conv3d_1(x)
        x = self.LTA(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        return x





class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(Conv3d, self).__init__()
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        feat = F.leaky_relu(gob_feat)
        return feat




class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio,
                                             gate_channel // reduction_ratio,
                                             kernel_size=3,
                                             padding=dilation_val,
                                             dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):

        att = 1+torch.sigmoid(self.gate_s(in_tensor).expand_as(in_tensor))
        x = att*in_tensor

        return x

class GLSpatialGate(nn.Module):
    def __init__(self,gate_channel, having=3, fm_sign=True):
        super(GLSpatialGate, self).__init__()
        self.halving = having
        self.fm_sign = fm_sign
        self.global_SpatialGate = SpatialGate(gate_channel)
        self.local_SpatialGate0 = SpatialGate(gate_channel)
        self.local_SpatialGate1 = SpatialGate(gate_channel)
        self.local_SpatialGate2 = SpatialGate(gate_channel)
        self.local_SpatialGate3 = SpatialGate(gate_channel)
        self.local_SpatialGate4 = SpatialGate(gate_channel)
        self.local_SpatialGate5 = SpatialGate(gate_channel)
        self.local_SpatialGate6 = SpatialGate(gate_channel)
        self.local_SpatialGate7 = SpatialGate(gate_channel)



    def forward(self, x):



        gob_feat = self.global_SpatialGate(x)
        if self.halving == 0:
            lcl_feat = self.local_SpatialGate(x)
        else:
            h = x.size(2)

            split_size = int(h // 2 ** self.halving)

            lcl_feat = x.split(split_size, 2)

            lcl= [self.local_SpatialGate0(lcl_feat[0]),self.local_SpatialGate1(lcl_feat[1]),self.local_SpatialGate2(lcl_feat[2]),
                  self.local_SpatialGate3(lcl_feat[3]),self.local_SpatialGate4(lcl_feat[4]),self.local_SpatialGate5(lcl_feat[5]),
                  self.local_SpatialGate6(lcl_feat[6]),self.local_SpatialGate7(lcl_feat[7])]
            lcl_feat = torch.cat([_ for _ in lcl], 2)
            #lcl_feat = torch.cat([self.local_SpatialGate(_) for _ in lcl_feat], 2)


        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=2))


        return feat


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class TemporalGate(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TemporalGate, self).__init__()
        self.HPP = GeMHPP()
        self.dilate_conv_1 = BasicConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(1,0), dilation=(1,1))

        self.dilate_conv_2 = BasicConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(2,0), dilation=(2,1))

        self.dilate_conv_4 = BasicConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(4,0), dilation=(4,1))

        #self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU()


    def forward(self,x):
        #in [n,c,s,h,w]
        #out [n,c,p]
        n,c,s,h,w = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(n*s,c,h,w)
        x = self.HPP(x)  #ns,c,t,p
        x = x.view(n,s,c,-1).permute(0,2,1,3).contiguous()

        t_f = self.dilate_conv_1(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)
        t_s = self.dilate_conv_2(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)+t_f
        t_l = self.dilate_conv_4(x).permute(0, 2, 1, 3).contiguous().unsqueeze(-1)+t_s


        t = torch.cat([t_f, t_s, t_l], -1)


        return t

def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)

class ATA(nn.Module):
    def __init__(self, in_planes, part_num=64, div=16):
        super(ATA, self).__init__()
        self.in_planes = in_planes
        self.part_num = part_num
        self.div = div

        self.in_conv = conv1d(part_num * 3 * in_planes, part_num * 3 * in_planes // div, 1, False, groups=part_num)

        self.out_conv = conv1d(part_num * 3 * in_planes // div, part_num * 3 * in_planes, 1, False, groups=part_num)

        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, t):

        n, s, c, h, _ = t.size()
        t = t.permute(0, 3, 4, 2, 1).contiguous().view(n, h * 3 * c, s)

        t_inter = self.leaky_relu(self.in_conv(t))
        t_attention = self.out_conv(t_inter).sigmoid()
        weighted_sum = (t_attention * t).view(n, h, 3, c, s).sum(2).sum(-1) / t_attention.view(n, h, 3, c, s).sum(
            2).sum(-1)
        weighted_sum = self.leaky_relu(weighted_sum).permute(0, 2, 1).contiguous()

        return weighted_sum









class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1) * p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class Cat_sil_pose(nn.Module):
    def __init__(self, p=5):
        super(Cat_sil_pose, self).__init__()
        self.p = nn.Parameter(
            torch.ones(1) * p)
    def forward(self,sil,pose):
        return torch.cat([sil*self.p, pose], 2)

class SILTEST(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']
        self.ES = backbone_3d(_in_channels=1,_channels=sil_channels)
        self.attention_t = TemporalGate(sil_channels[-1], sil_channels[-1])
        self.aggre_t = ATA(sil_channels[-1])
        self.TP = PackSequenceWrapper(torch.max)

        self.attention_s = GLSpatialGate(sil_channels[-1])

        hidden_dim = model_cfg['hidden_dim']
        self.Head0 = SeparateFCs(**model_cfg['SeparateFCs'])
        # self.Head0 = SeparateFCs(64, 128, hidden_dim)
        self.Bn = nn.BatchNorm1d(hidden_dim)
        self.Head1 = SeparateFCs(64, hidden_dim, model_cfg['class_num'])

        self.HPP = GeMHPP()



    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)  ## [n, 1,s, h, w]

        del ipts


        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)




        '''extract sil features '''
        f = self.ES(sils)
        f_te = self.attention_t(f)
        f_te = self.aggre_t(f_te)
        f_pa = self.TP(f, seqL, options={"dim": 2})[0]
        f_pa = self.attention_s(f_pa)
        f_pa = self.HPP(f_pa)  # [n, c, p]

        outs = torch.cat([f_te,f_pa],1)







        gait = self.Head0(outs)  # [p, n, c]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft)  # [p, n, c]
        embed=bnft








        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval