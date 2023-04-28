import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from thirds.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from ._dgcnn import DGCNN_cls, DGCNN_partseg, DGCNN_semseg
from ._pct import Pct, Pct_partseg
from ._pointmlp import pointMLP, pointMLPElite, pointMLP_partseg
from ._curvenet import CurveNet
from ._gbnet import get_model as GBNet
from ._pointnet2 import get_model as PointNet2, get_seg_model as PointNet2_partseg

def knn(xyz_k, xyz_q=None, k=32): # bs, N, 3; bs, M, 3

    if xyz_q is None:
        xyz_q = xyz_k
    inner = 2*torch.matmul(xyz_q, xyz_k.transpose(-2, -1))
    len_k = torch.sum(xyz_k**2, dim=2, keepdim=True).transpose(-2, -1)
    len_q = torch.sum(xyz_q**2, dim=2, keepdim=True)

    dist = inner - len_k - len_q

    return dist.topk(k=k, dim=-1)[1] # bs, M, k

def augment(xyz: torch.Tensor, mask_num=12, fps_num=16) -> torch.Tensor:
    bs, n, _ = xyz.shape
    k = n // fps_num
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, fps_num)
    fps_idx = fps_idx[:, np.random.choice(np.arange(fps_num), mask_num, replace=False)]
    fps_data = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()

    idx = knn(xyz, fps_data, k).reshape(bs, -1)
    all_idx = torch.arange(n).to(idx).unsqueeze(1)
    idx = [all_idx[~all_idx.eq(idx[i]).any(1)][:n-mask_num*k].view(1,-1) for i in range(bs)]
    idx = torch.cat(idx, dim=0).type_as(fps_idx)
    
    fps_data = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1,2).contiguous()
    return fps_data

class autoencoder(nn.Module):
    def __init__(self, encoder=None, fps_num = 16, mask_ratio=0.5) -> None:
        super().__init__()
        self.fps_num = fps_num
        self.mask_num = int(mask_ratio * self.fps_num)

        if encoder == 'dgcnn':
            self.encoder = DGCNN_cls(return_feature=True)
        elif encoder == 'dgcnn_partseg':
            self.encoder = DGCNN_partseg(return_feature=True)
        elif encoder == 'dgcnn_semseg':
            self.encoder = DGCNN_semseg(return_feature=True)
        elif encoder == 'pct':
            self.encoder = Pct(return_feature=True)
        elif encoder == 'pct_partseg':
            self.encoder = Pct_partseg(return_feature=True)
        elif encoder == 'gbnet':
            self.encoder = GBNet(return_feature=True)
        elif encoder == 'pointmlp':
            self.encoder = pointMLP(return_feature=True)
        elif encoder == 'pointmlpelite':
            self.encoder = pointMLPElite(return_feature=True)
        elif encoder == 'pointmlp_partseg':
            self.encoder = pointMLP_partseg(return_feature=True)
        elif encoder == 'curvenet':
            self.encoder = CurveNet(return_feature=True)
        else:
            raise NotImplementedError()
        self.decoder = nn.Linear(self.encoder.feature_channel, 3*1024, 1)

        self.loss_func = ChamferDistanceL1()
    
    def cal_loss(self, xyz: torch.Tensor, rec_xyz: torch.Tensor):
        return self.loss_func(xyz, rec_xyz)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        bs, n, _ = xyz.shape
        mask_xyz = augment(xyz, mask_num=self.mask_num, fps_num=self.fps_num)
        x = mask_xyz.permute(0, 2, 1)
        feat = self.encoder(x)
        rec_xyz = self.decoder(feat).reshape(bs, n, -1)
        return xyz, rec_xyz

class classify(nn.Module):
    def __init__(self, encoder=None, num_classes=40) -> None:
        super().__init__()

        if encoder == 'dgcnn':
            self.encoder = DGCNN_cls(output_channels=num_classes)
        elif encoder == 'pct':
            self.encoder = Pct(output_channels=num_classes)
        elif encoder == 'pointmlp':
            self.encoder = pointMLP(num_classes=num_classes)
        elif encoder == 'pointmlpelite':
            self.encoder = pointMLPElite(num_classes=num_classes)
        elif encoder == 'gbnet':
            self.encoder = GBNet(output_channels=num_classes)
        elif encoder == 'curvenet':
            self.encoder = CurveNet(num_classes=num_classes)=
        else:
            raise NotImplementedError()

    def cal_loss(self, gold: torch.Tensor, pred: torch.Tensor, smoothing=True) -> torch.Tensor:
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

    def class_acc(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        cls_num = true.max() + 1
        mAcc = torch.zeros(cls_num)
        for i in range(cls_num):
            true_label = true==i
            mAcc[i] = torch.sum(~true_label == (pred-i)*2)/torch.sum(true_label)
        return 100. * mAcc.mean().item()
        
    def instance_acc(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return 100. * torch.sum(pred == true).item() / true.shape[0]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz.permute(0, 2, 1)
        cls = self.encoder(x)
        return cls

class partseg(nn.Module):
    def __init__(self, encoder: str=None, seg_num_all: int=50) -> None:
        super().__init__()

        if encoder == 'dgcnn':
            self.encoder = DGCNN_partseg(seg_num_all=seg_num_all)
        elif encoder == 'pct':
            self.encoder = Pct_partseg(seg_num_all=seg_num_all)
        elif encoder == 'pointmlpn':
            self.encoder = pointMLP_partseg(num_classes=seg_num_all, use_norm=True)
        elif encoder == 'pointmlp':
            self.encoder = pointMLP_partseg(num_classes=seg_num_all, use_norm=False)
        elif encoder == 'pointnet2':
            self.encoder = PointNet2_partseg(num_classes=seg_num_all)
        else:
            raise NotImplementedError()

    def cal_loss(self, pred: torch.Tensor, gt: torch.Tensor):
        gold = gt.contiguous().view(-1)
        loss = F.cross_entropy(pred, gold, reduction='mean')
        return loss

    def forward(self, xyz: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        x = xyz.permute(0, 2, 1)
        feat = self.encoder(x, l)
        return feat