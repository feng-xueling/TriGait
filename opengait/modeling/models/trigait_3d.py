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

class backbone_3d(nn.Module):
    def __init__(self,in_c = [32, 64, 128,256]):
        super(backbone_3d, self).__init__()
        # 3D Convolution



        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA0 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(inplace=True)
        )


        self.ConvA1 = nn.Sequential(

            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            BasicConv3d(in_c[1], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(inplace=True),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            nn.ReLU(inplace=True),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            BasicConv3d(in_c[3], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            nn.ReLU(inplace=True),
        )


    def forward(self,inputs):
        x = self.conv3d(inputs)
        sil = x.clone()
        x = self.ConvA0(x)
        #x = self.MaxPool0(x)
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x,sil

class backbone_3d_ca(nn.Module):
    def __init__(self,in_c = [32, 64, 128,256]):
        super(backbone_3d_ca, self).__init__()
        # 3D Convolution



        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            #nn.ReLU()
        )


        self.ConvA0 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[0]),
            #nn.ReLU()
        )


        self.ConvA1 = nn.Sequential(

            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[1]),
            #nn.ReLU(),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[2]),
            #nn.ReLU(),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_c[3]),
            #nn.ReLU(),
        )
        '''
        self.conv3d = nn.Sequential(
            BasicConv2d(1, in_c[0], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True)
        )

        self.ConvA0 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[0], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True)
        )

        self.ConvA1 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[1], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
        )

        self.ConvA2 = nn.Sequential(
            BasicConv2d(in_c[1], in_c[2], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[2]),
            nn.ReLU(inplace=True),
        )

        self.ConvA3 = nn.Sequential(
            BasicConv2d(in_c[2], in_c[3], kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c[3]),
            nn.ReLU(inplace=True),
        )'''


    def forward(self,inputs):
        n,c,s,h,w = inputs.shape;

        x = self.conv3d(inputs)
        sil = x.clone()
        x = self.ConvA0(x)
        #x = self.MaxPool0(x)
        x = self.ConvA1(x)
        x = self.ConvA2(x)
        x = self.ConvA3(x)
        return x, sil



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




class TriGait_baseline(BaseModel):
    def __init__(self, *args, **kargs):
        super(TriGait_baseline, self).__init__(*args, **kargs)
    def savefig(self,fuse_sil):
        processed = []
        output_directory = r'/code/CodeTest/fxl/Opengait/sil_feature_map/'
        # Save the activation map as an image file

        import matplotlib.pyplot as plt
        for frame_count in range(fuse_sil.shape[2]):
            feature_map = fuse_sil[0, :, frame_count, :, :]  # c h w
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[
                0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
            processed.append(
                gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy
        for i in range(len(processed)):  # len(processed) = 17
            output_directory = r'/code/CodeTest/fxl/Opengait/sil_feature_map/'
            output_path = os.path.join(output_directory, f'frame_{i}.png')
            cv2.imwrite(output_path, processed[i] * 255)

    def savefig2(self, fuse_sil):

        output_directory = r'/code/CodeTest/fxl/Opengait/sil_feature_map/'
        # Save the activation map as an image file
        feature_map = fuse_sil[0, :, :, :]  # c h w
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed = gray_scale.data.cpu().numpy()
        output_directory = r'/code/CodeTest/fxl/Opengait/sil_feature_map/'
        output_path = os.path.join(output_directory, 'mean.png')
        cv2.imwrite(output_path, processed * 255)


    def build_network(self, model_cfg):
        sil_channels = model_cfg['channels']

        self.ES = backbone_3d_ca(sil_channels)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

        #self.TP = PackSequenceWrapper(torch.max)
        self.attention_s = GLSpatialGate(sil_channels[-1])
        self.attention_t = TemporalGate(sil_channels[-1], sil_channels[-1],bin_num=model_cfg["bin_num"])
        self.aggre_t = ATA(sil_channels[-1], part_num=model_cfg["bin_num"][0])




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

        '''
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        '''
        self.Head0 = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Bn = nn.BatchNorm1d(model_cfg['hidden_dim'])
        self.Head1 = SeparateFCs(71, model_cfg['hidden_dim'], model_cfg['class_num'])




    def save_gradient(self, grad):
        self.gradients = grad


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
        f,fuse_sil = self.ES(sils)




        f_s = self.TP(f, seqL, options={"dim": 2})[0]
        f_s = self.attention_s(f_s)
        f_s = self.HPP(f_s)  # [n, c, p]
        f_te = self.attention_t(f)
        f_te = self.aggre_t(f_te)

        outs = torch.cat((f_s,f_te),1)
        #outs = f_s


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
            },
            'logits_feat': {
                'logits': outs
            }
        }
        return retval
        #return logi.sum(-1) #logi.max(-1)[0]+logi.min(-1)[0]