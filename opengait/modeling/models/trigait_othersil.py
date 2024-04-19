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

class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat

class backbone(nn.Module):
    def __init__(self,in_c):
        super(backbone, self).__init__()
        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.MaxPool0 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))


    def forward(self,x):
        outs = self.conv3d(x)
        fuse = outs.clone()
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        return outs,fuse



class TriGait_othersil(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)



    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']

        self.ES = backbone(sil_channels)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])






        # pose branch

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']
        head_num = model_cfg['pos_cfg']['num_heads']

        self.BN2d = nn.BatchNorm2d(in_c2[0])

        self.tcnst0 = nn.Sequential(TCNST(in_c2[0],in_c2[1]),
                                    TCNST(in_c2[1], in_c2[1]))

        self.tcnst1 = nn.Sequential(TCNST(in_c2[1],in_c2[2]))

        self.tcnst2 = nn.Sequential(TCNST(in_c2[2],in_c2[3]))
        self.Avg = nn.AdaptiveAvgPool1d(model_cfg["bin_num"])
        self.TPmean = PackSequenceWrapper(torch.mean)

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

        self.cat_s_p = Cat_sil_pose()



        #self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        #self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.Head0 = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Bn = nn.BatchNorm1d(model_cfg['hidden_dim'])
        self.Head1 = SeparateFCs(71, model_cfg['hidden_dim'], model_cfg['class_num'])



    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
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
        f,fuse_sil = self.ES(sils)

        outs = self.TP(f, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]



        '''extract pose features'''


        # print('device:', poses.device)
        x = self.BN2d(poses)  # [128,10,30,17]
        fuse_pose = self.posepretreatment(x)+self.cc(x)
        x = self.tcnst0(x)
        x = self.tcnst1(x)
        x = self.tcnst2(x)
        n, c, _, _ = x.shape  # n,c,t,v
        x = self.TPmean(x, seqL, options={"dim": 2})

        x = self.Avg(x)  # ncp


        '''fuse stage 1'''
          #[n,c1,p]

        ma, mi = self.divid(poses)

        fuse_pose = self.TPmean(fuse_pose, seqL, options={"dim": 2})
        fuse_sil = self.TPmean(fuse_sil, seqL, options={"dim": 2})


        fuse = self.fuse(fuse_sil,fuse_pose,ma,mi)  #n,p,c
        fuse = fuse.permute(0,2,1).contiguous() #ncp
        fuse_sp = self.cat_s_p(outs,x)





        fuse = torch.cat([fuse_sp, fuse], 2)


        #embed_1 = self.FCs(fuse)  # [n, c, p]
        #embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        #embed = embed_1
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