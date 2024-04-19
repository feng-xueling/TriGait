import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, Attention, DividPart, FusePart2,SeparateBNNecks,HorizontalPoolingPyramid

from torchvision.utils import save_image

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Graph:
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='coco',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,6), (5,6),
                             (5,7), (7,9), (6,8), (8,10), (5,11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'nonlocal-coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            edge=[]
            for i in range(0,  self.num_node ):
                for j in range(0,  self.num_node ):
                    edge.append((i, j))
            self.edge = edge
            self.center = 1
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class TCN(nn.Module):
    def __init__(self,in_c, out_c,kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),res = True):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size,
                stride,
                padding,
            ),
            nn.BatchNorm2d(out_c),
            nn.Dropout(inplace=True),
        )


    def forward(self, x):
        """
            x  : [n, in_c, t, v]
            ret: [n, out_c, t,v]
        """
        return self.tcn(x)


class ST(nn.Module):
    def __init__(self, channle,head_num = 8,res = True):
        super(ST, self).__init__()
        self.bn0 = nn.BatchNorm2d(channle)
        self.atten = Attention(channle, head_num, qkv_bias=False)
        self.mish = nn.ReLU(inplace=True)
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

class TCNST(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        super(TCNST, self).__init__()
        self.f = False
        if in_c !=out_c:
            self.cc = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm2d(out_c),
            )
            self.cc2 = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm2d(out_c),
            )
            self.f = True

        if in_c==10:
             num_head=1
        else:
             num_head=8

        self.tcn = TCN(in_c,out_c,kernel_size, stride, padding)
        self.st = ST(in_c,num_head)



    def forward(self, x):
        """
            x  : [n, c, t, v]
            ret: [n, c, p]
        """
        f_t = self.st(x) + x
        f_st = self.tcn(f_t)
        f_t = self.cc2(f_t) if self.f else f_t
        x = self.cc(x) if self.f else x
        f_st = f_st + f_t
        return x + f_st


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=[3,3],
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class SKE_GCN(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']
        head_num = model_cfg['pos_cfg']['num_heads']

        self.BN2d = nn.BatchNorm2d(in_c2[0])

        self.graph = Graph(strategy='spatial', layout='coco')
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        temporal_kernel_size = 9
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(10, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1),
        ))


        self.edge_importance = [1] * len(self.st_gcn_networks)







        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])



    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))

        poses = ipts[0].permute(0, 3, 1, 2).contiguous()  # [n, s, v, c]->n,c,s,v
        del ipts


        n, _, s, v = poses.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            poses = poses.repeat(1, 1, repeat, 1)




        '''extract pose features'''


        # print('device:', poses.device)
        x = self.BN2d(poses)  # [128,10,30,17]


        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)


        n, c, _, _ = x.shape  # n,c,t,v

        x = F.avg_pool2d(x, x.size()[2:]).squeeze(3)


        embed_1 = self.FCs(x)  # [n, c, p]
        embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1



        n, _, s, v = poses.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': poses.view(n * s, 1,10,v )
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


class SKE(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):

        pose_cfg = model_cfg['pos_cfg']
        in_c2 = model_cfg['pos_cfg']['in_channels']
        head_num = model_cfg['pos_cfg']['num_heads']

        self.BN2d = nn.BatchNorm2d(in_c2[0])

        self.tcnst0 = nn.Sequential(TCNST(in_c2[0],in_c2[1]),
                                    TCNST(in_c2[1], in_c2[1]))

        self.tcnst1 = nn.Sequential(TCNST(in_c2[1],in_c2[2]))

        self.tcnst2 = nn.Sequential(TCNST(in_c2[2],in_c2[3]))
        self.Avg = nn.AdaptiveAvgPool1d(1)
        self.TPmean = PackSequenceWrapper(torch.mean)



        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])



    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))

        poses = ipts[0].permute(0, 3, 1, 2).contiguous()  # [n, s, v, c]->n,c,s,v
        del ipts


        n, _, s, v = poses.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            poses = poses.repeat(1, 1, repeat, 1)



        '''extract pose features'''


        # print('device:', poses.device)
        x = self.BN2d(poses)  # [128,10,30,17]
        x = self.tcnst0(x)
        x = self.tcnst1(x)
        x = self.tcnst2(x)
        n, c, _, _ = x.shape  # n,c,t,v
        x = self.TPmean(x, seqL, options={"dim": 2})

        x = self.Avg(x)  # ncp

        embed_1 = self.FCs(x)  # [n, c, p]
        embed_2, logi = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1



        n, _, s, v = poses.size()

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': poses.view(n * s, 1, 10,v )
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

class Attention2(nn.Module):
    def __init__(self,dim,use_pes=False,num_heads=8,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(Attention2, self).__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim, bias=False)
        self.proj_drop=nn.Dropout(proj_drop)
        self.dim = dim



    def forward(self,x):
        n,c,t,vv=x.size()
        #print("inp", x.shape)
        x = x.permute(0,2,3,1).reshape(n*t,vv,c).contiguous()
        #print("gai",x.shape)
        B, N, C = x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).contiguous()
        q,k,v=qkv[0],qkv[1],qkv[2]
        #print("q:",q.shape)

        attn = (q @ k.transpose(-2,-1).contiguous())*self.scale
        #print("attn1:",attn.shape)
        attn = attn.softmax(dim=1)
        #print("attn2:", attn.shape)
        attn = self.attn_drop(attn)
        #print("attn3:", attn.shape)

        x=(attn@v).transpose(1,2).reshape(B,N,C).contiguous()
        #print("attn4:", x.shape)
        x= self.proj(x)
        #print("attn5:", x.shape)
        x=self.proj_drop(x).reshape(n, t, vv, c).permute(0,3,1,2).contiguous()
        #print("attn6:", x.shape)
        return x

class GaitTR(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        in_c = model_cfg['pos_cfg']['in_channels']
        self.BN2d = nn.BatchNorm2d(in_c[0])

        self.fc0 = nn.Linear(in_c[0], in_c[1], bias=False)
        self.cc0 = nn.Sequential( nn.Mish(),
                                  nn.BatchNorm2d(in_c[1]))

        self.fc2 = nn.Linear(in_c[1], in_c[2], bias=False)
        self.cc2 = nn.Sequential(nn.Mish(),
                                 nn.BatchNorm2d(in_c[2]))

        self.fc3 = nn.Linear(in_c[2], in_c[3], bias=False)
        self.cc3 = nn.Sequential(nn.Mish(),
                                 nn.BatchNorm2d(in_c[3]))

        '''self.cc0 = nn.Conv2d(in_c[0], in_c[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.cc2 = nn.Conv2d(in_c[1], in_c[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.cc3 = nn.Conv2d(in_c[2], in_c[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)'''

        # self.TC = nn.Conv2d(in_c[0], in_c[1], kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.TCN0 = nn.Sequential(nn.Dropout(),
                                  nn.Conv2d(in_c[0], in_c[1], kernel_size=(3,3), stride=(1, 1), padding=(1, 1),
                                            bias=False),
                                  nn.Mish(),
                                  nn.BatchNorm2d(in_c[1]))

        self.ST0 = nn.Sequential(nn.BatchNorm2d(in_c[1]),
                                 Attention2(in_c[1], model_cfg['pos_cfg']['num_heads'], qkv_bias=False),
                                 nn.Mish(),
                                 nn.BatchNorm2d(in_c[1]))

        self.TCN1 = nn.Sequential(nn.Dropout(),
                                  nn.Conv2d(in_c[1], in_c[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                            bias=False),
                                  nn.Mish(),
                                  nn.BatchNorm2d(in_c[1]))

        self.ST1 = nn.Sequential(nn.BatchNorm2d(in_c[1]),
                                 Attention2(in_c[1], model_cfg['pos_cfg']['num_heads'], qkv_bias=False),
                                 nn.Mish(),
                                 nn.BatchNorm2d(in_c[1]))

        self.TCN2 = nn.Sequential(nn.Dropout(),
                                  nn.Conv2d(in_c[1], in_c[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                            bias=False),
                                  nn.Mish(),
                                  nn.BatchNorm2d(in_c[2]))

        self.ST2 = nn.Sequential(nn.BatchNorm2d(in_c[2]),
                                 Attention2(in_c[2],model_cfg['pos_cfg']['num_heads'], qkv_bias=False),
                                 nn.Mish(),
                                 nn.BatchNorm2d(in_c[2]))

        self.TCN3 = nn.Sequential(nn.Dropout(),
                                  nn.Conv2d(in_c[2], in_c[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                            bias=False),
                                  nn.Mish(),
                                  nn.BatchNorm2d(in_c[3]))

        self.ST3 = nn.Sequential(nn.BatchNorm2d(in_c[3]),
                                 Attention2(in_c[3], model_cfg['pos_cfg']['num_heads'], qkv_bias=False),
                                 nn.Mish(),
                                 nn.BatchNorm2d(in_c[3]))

        # k1 = model_cfg['frame_num']
        # k2 = model_cfg['keynode_num']

        # self.Avg = nn.AvgPool2d(kernel_size=(k1, k2), stride=(1, 1), padding=(0, 0))
        self.Avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_c[3], 128, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.fcsftx = nn.Linear(128, 74, bias=False)

    def forward(self, inputs):
        ipts, labs, type1, view, seqL = inputs
        poses = ipts[0]  # [n, t, c, v] [256,10,60,17]
        del ipts
        poses = poses.permute(0, 3, 1, 2).contiguous()
        n, c, t, v = poses.shape
        '''if c != 10:
            r = 10 - c
            a = torch.ones(n, r, t, v).to(torch.distributed.get_rank())
            poses = torch.cat((a, poses), 1)
        # print("pose:", poses.shape, labs,type1, view )'''
        # print('device:', poses.device)
        x = self.BN2d(poses)  # [256,10,60,17]
        #print("x",x[0,:,0:3,3])
        # print(x.shape)

        x0 = self.fc0(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x0 = self.cc0(x0)
        # print(x0.shape)
        x1 = x0 + self.TCN0(x)  # [256,64,60,17]



        x2 = x1 + self.ST0(x1)
        x = x0 + x2
        # x = x2
        # print(x.shape)

        x1 = x + self.TCN1(x)  # [256,64,60,17]
        x2 = x1 + self.ST1(x1)
        x = x + x2
        # print(x.shape)

        x0 = self.fc2(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x0 = self.cc2(x0)
        x1 = x0 + self.TCN2(x)  # [256,128,60,17]
        # x1 = self.TCN2(x)
        x2 = x1 + self.ST2(x1)
        x = x0 + x2
        # x = x2
        # print(x.shape)

        x0 = self.fc3(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x0 = self.cc3(x0)
        x1 = x0 + self.TCN3(x)  # [256,256,60,17]
        # x1 = self.TCN3(x)
        x2 = x1 + self.ST3(x1)
        x = x0 + x2
        # x = x2
        # print(x.shape)
        n, c, _, _ = x.shape  # n,c,t,v
        x = self.Avg(x).reshape(n, c, -1).permute(0, 2, 1).contiguous()  # ncp->npc
        # print(x.shape)
        embs = self.fc(x).permute(0,2,1).contiguous()
        # print(embs.shape)
        logi = self.bn(embs).permute(0,2,1).contiguous()
        logi = self.fcsftx(logi).permute(0,2,1).contiguous()

        n, s, h, w = poses.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': poses.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval