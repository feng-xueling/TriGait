import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, PackSequenceWrapper, Attention, FusePart3,SeparateBNNecks,HorizontalPoolingPyramid,FusePart2,DividPart
from torchvision.utils import save_image

class backbone_2d(nn.Module):
    def __init__(self,in_c = [32, 64, 128,256]):
        super(backbone_2d, self).__init__()
        # 3D Convolution

        self.conv1 = nn.Sequential(
            BasicConv2d(1, in_c[0], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True)
        )

        self.ConvA0 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[0], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[0]),
            BasicConv2d(in_c[0], in_c[0], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True)
        )

        self.ConvA1 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[1], kernel_size=(3, 3),
                        stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(in_c[1]),
            BasicConv2d(in_c[1], in_c[1], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True)
        )

        self.ConvA2 = nn.Sequential(
            BasicConv2d(in_c[1], in_c[2], kernel_size=(3, 3),
                        stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(in_c[2]),
            BasicConv2d(in_c[2], in_c[2], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[2]),
            nn.ReLU(inplace=True)
        )

        self.ConvA3 = nn.Sequential(
            BasicConv2d(in_c[2], in_c[3], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[3]),
            BasicConv2d(in_c[3], in_c[3], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[3]),
            nn.ReLU(inplace=True)
        )


    def forward(self,inputs):
        x = self.conv1(inputs)
        sil = x.clone()
        x = self.ConvA0(x)
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x,sil



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
    def __init__(self, gate_channel, having=3, fm_sign=True):
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
            lcl = [self.local_SpatialGate0(lcl_feat[0]), self.local_SpatialGate1(lcl_feat[1]),
                   self.local_SpatialGate2(lcl_feat[2]),
                   self.local_SpatialGate3(lcl_feat[3]), self.local_SpatialGate4(lcl_feat[4]),
                   self.local_SpatialGate5(lcl_feat[5]),
                   self.local_SpatialGate6(lcl_feat[6]), self.local_SpatialGate7(lcl_feat[7])]
            lcl_feat = torch.cat([_ for _ in lcl], 2)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=2))
        return feat


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
    def __init__(self,in_channels,out_channels):
        super(TemporalGate, self).__init__()
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        #self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
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
    def __init__(self, in_planes, part_num=16, div=16):
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


class TriGait_Gait3D_2dbackbone(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']
        self.ES = backbone_2d(sil_channels)
        self.attention_t = TemporalGate(sil_channels[-1], sil_channels[-1])#,model_cfg['bin_num'])
        self.aggre_t = ATA(sil_channels[-1])
        self.TP = PackSequenceWrapper(torch.max)
        self.attention_s = GLSpatialGate(sil_channels[-1])
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        #self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])



        # pose branch

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']
        head_num = model_cfg['pos_cfg']['num_heads']

        self.BN2d = nn.BatchNorm2d(in_c2[0])

        self.tcnst0 = nn.Sequential(TCNST(in_c2[0],in_c2[1]),
                                    TCNST(in_c2[1], in_c2[1]))

        self.tcnst1 = nn.Sequential(TCNST(in_c2[1],in_c2[2]))

        self.tcnst2 = nn.Sequential(TCNST(in_c2[2],in_c2[3]))
        self.Avg = nn.AdaptiveAvgPool1d(16)
        self.TPmean = PackSequenceWrapper(torch.mean)

        #fuse
        self.posepretreatment = TCN(in_c2[0], in_c2[1])
        self.cc = nn.Sequential(nn.Conv2d(in_c2[0], in_c2[1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                nn.Mish(),
                                nn.BatchNorm2d(in_c2[1])
                                )
        self.divid = DividPart(pose_cfg['head'],pose_cfg['shoulder'],pose_cfg['elbow'],pose_cfg['wrist'],pose_cfg['hip'],pose_cfg['knee'],pose_cfg['ankle'],if_ou=pose_cfg['if_ou']) #pose_cfg['head'],pose_cfg['shoulder'],pose_cfg['elbow'],pose_cfg['wrist'],pose_cfg['hip'],pose_cfg['knee'],pose_cfg['ankle'],if_ou=True
        fuse_channel = 1280
        self.fuse = FusePart2(sil_channels[0],in_c2[1],out_c=fuse_channel,atten_depth=4,pose=[pose_cfg['head'],pose_cfg['shoulder'],pose_cfg['elbow'],pose_cfg['wrist'],pose_cfg['hip'],pose_cfg['knee'],pose_cfg['ankle']])


        #self.fuse_tcnst = TCN(fuse_channel[1],fuse_channel[1])
        #self.switch_ske = Switch_Ske()

        self.cat_s_p = Cat_sil_pose()
        '''

        hidden_dim = model_cfg['hidden_dim']
        sil_dim = sil_channels[-1]*2+in_c2[-1]
        self.Head0 = SeparateFCs(23, sil_dim, hidden_dim)
        self.Bn = nn.BatchNorm1d(hidden_dim)
        self.Head1 = SeparateFCs(23, hidden_dim, model_cfg['class_num'])'''


        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])



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
        n, c, s, h, w = sils.size()
        x = sils.transpose(1, 2).reshape(-1, c, h, w)
        f,fuse_sil = self.ES(x)
        size_f = f.size()
        f = f.reshape(n, s, *size_f[1:]).transpose(1, 2).contiguous()
        size_fuse_sil = fuse_sil.size()
        fuse_sil = fuse_sil.reshape(n, s, *size_fuse_sil[1:]).transpose(1, 2).contiguous()

        f_te = self.attention_t(f)
        f_te = self.aggre_t(f_te)
        f_pa = self.TP(f, seqL, options={"dim": 2})[0]
        f_pa = self.attention_s(f_pa)
        f_pa = self.HPP(f_pa)  # [n, c, p]

        outs = torch.cat([f_te,f_pa],1) #ncp



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
        ma,mi = self.divid(poses)
        fuse_pose = self.TPmean(fuse_pose, seqL, options={"dim": 2})
        fuse_sil = self.TPmean(fuse_sil, seqL, options={"dim": 2})
        fuse = self.fuse(fuse_sil,fuse_pose,ma,mi)  #n,p,c
        fuse = fuse.permute(0,2,1).contiguous() #ncp
        '''
        ma, mi = self.divid(poses)
        fuse = self.fuse(fuse_sil, fuse_pose, ma, mi)  # n,c,sp
        # n, c, _, _ = fuse.shape
        fuse = self.TP(fuse, dim=2, seq_dim=2, seqL=seqL)[0]  # ncp
        fuse = fuse.permute(2, 0, 1).contiguous()'''


        '''concate along c dimention'''


        fuse_sp = self.cat_s_p(outs,x)

        fuse = torch.cat([fuse_sp, fuse], 2)
        '''
        gait = self.Head0(fuse)  # [ncp]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft)  # [n, c, p]
        embed = bnft
        '''
        embed_1 = self.FCs(fuse)  # [n, c, p]
        embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1




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