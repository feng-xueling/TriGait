import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torch.autograd import Variable
import numpy as np
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, Attention, DividPart, FusePart2,SeparateBNNecks,HorizontalPoolingPyramid
from PIL import Image
from torchvision.utils import save_image

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d









class ST(nn.Module):
    def __init__(self, channle,head_num = 8,res = True):
        super(ST, self).__init__()
        self.bn0 = nn.BatchNorm2d(channle)
        self.atten = Attention(channle, head_num, qkv_bias=False)
        self.mish = nn.Mish(inplace=True)
        self.bn1 = nn.BatchNorm2d(channle)
        self.res = res

    def forward(self, x):
        """
            x  : [n, c, t, v]
            ret: [n, c, p]
        """
        x = self.bn0(x)
        n, c, t, vv = x.size()
        # print("inp", x.shape)
        x = x.permute(0, 2, 3, 1).reshape(n * t, vv, c).contiguous()
        x = self.atten(x) + x if self.res else self.atten(x)
        x = x.reshape(n, t, vv, c).permute(0,3,1,2).contiguous()
        x = self.bn1(self.mish(x))
        return x

class TemporalGate(nn.Module):
    def __init__(self,in_channels,out_channels,bin_num=[64]):
        super(TemporalGate, self).__init__()
        #self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.dilate_conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(1,0), dilation=(1,1))

        self.dilate_conv_2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(2,0), dilation=(2,1))

        self.dilate_conv_4 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1),
            stride=(1,1), padding=(4,0), dilation=(4,1))




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

class TCN(nn.Module):
    def __init__(self,in_c, out_c,kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),res = True):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(nn.Dropout(),
                      nn.Conv2d(in_c, out_c, kernel_size, stride, padding,
                                bias=False),
                      nn.Mish(),
                      nn.BatchNorm2d(out_c))


    def forward(self, x):
        """
            x  : [n, in_c, t, v]
            ret: [n, out_c, t,v]
        """
        return self.tcn(x)

class TCNST(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        super(TCNST, self).__init__()
        self.f = False
        if in_c !=out_c:
            self.cc = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                    nn.Mish(),
                                    nn.BatchNorm2d(out_c)
                                    )
            self.f = True

        self.tcn = TCN(in_c,out_c,kernel_size, stride, padding)
        self.st = ST(out_c)



    def forward(self, x):
        """
            x  : [n, c, t, v]
            ret: [n, c, p]
        """
        f_t = self.tcn(x)
        x = self.cc(x) if self.f else x
        f_st = x + f_t
        f_st = self.st(f_st)
        return x + f_st







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
        return torch.cat([sil*self.p, pose], 1)




class Earlyfusion(BaseModel):
    def __init__(self, *args, **kargs):
        super(Earlyfusion, self).__init__(*args, **kargs)



    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']

        self.conv3d = nn.Sequential(
            BasicConv3d(1, sil_channels[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(sil_channels[0]),
            #nn.ReLU()
        )
        self.TPmean = PackSequenceWrapper(torch.mean)





        # pose branch

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']
        head_num = model_cfg['pos_cfg']['num_heads']

        self.BN2d = nn.BatchNorm2d(in_c2[0])


        #fuse
        self.posepretreatment = TCN(in_c2[0], in_c2[1])
        self.cc = nn.Sequential(
            nn.Conv2d(in_c2[0], in_c2[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Mish(),
            nn.BatchNorm2d(in_c2[1])
            )

        self.divid = DividPart(pose_cfg['head'], pose_cfg['shoulder'], pose_cfg['elbow'], pose_cfg['wrist'],
                               pose_cfg['hip'], pose_cfg['knee'], pose_cfg['ankle'], if_ou=pose_cfg[
                'if_ou'])  # pose_cfg['head'],pose_cfg['shoulder'],pose_cfg['elbow'],pose_cfg['wrist'],pose_cfg['hip'],pose_cfg['knee'],pose_cfg['ankle'],if_ou=True
        fuse_channel = model_cfg['SeparateFCs']['in_channels']
        self.fuse = FusePart2(sil_channels[0], in_c2[1], out_c=fuse_channel, atten_depth=2)


        '''
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        '''
        self.Head0 = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Bn = nn.BatchNorm1d(model_cfg['hidden_dim'])
        self.Head1 = SeparateFCs(7, model_cfg['hidden_dim'], model_cfg['class_num'])






    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL

        sils = ipts[0].unsqueeze(1)  ## [n, 1,s, h, w]
        poses = ipts[1].permute(0, 3, 1, 2).contiguous()  # [n, s, v, c]->n,c,s,v
        del ipts


        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)
            poses = poses.repeat(1, 1, repeat, 1)



        '''extract sil features '''


        #f,fuse_sil = self.ES(sils)
        fuse_sil = self.conv3d(sils)



        x = self.BN2d(poses)  # [128,10,30,17]
        fuse_pose = self.posepretreatment(x)+self.cc(x)


        ma, mi = self.divid(poses)


        fuse_sil=self.TPmean(fuse_sil, seqL, options={"dim": 2})

        fuse_pose=self.TPmean(fuse_pose, seqL, options={"dim": 2})



        fuse = self.fuse(fuse_sil,fuse_pose,ma,mi)  #n,p,c
        fuse = fuse.permute(0, 2, 1).contiguous()  # ncp



        #fuse = self.TPmean(fuse, seqL, options={"dim": 2})

        '''
        embed_1 = self.FCs(fuse)  # [n, c, p]
        embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        '''
        embed_1 = self.Head0(fuse)  # [n, c, p]
        bnft = self.Bn(embed_1)  # [n, c, p]
        logi = self.Head1(bnft)  # [n, c, p]
        embed = bnft




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
        #return logi.sum(-1) #logi.max(-1)[0]+logi.min(-1)[0]