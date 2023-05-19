import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class DownMSELoss(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()
        return self.mse(dmap, gt_density)
class InfoNCELoss(nn.Module):
    def __init__(self, device='cpu', downsampling_rate=16):
        super().__init__()
        self.downsampling_rate = downsampling_rate
        self.kernel = torch.ones(1, 1, downsampling_rate, downsampling_rate).to(device)
        
    def forward(self, corr_map, pt_map):
        # resize the pt_map to the shape of features
        pt_map = F.conv2d(pt_map.float(), self.kernel, stride=self.downsampling_rate).bool()
        bs, _, h, w = pt_map.shape
        pt_map = pt_map.flatten(2).view(bs, h*w)
        
        # corr_map: shape of B * HW * query_number
        corr = torch.exp(corr_map)
        corr = corr.mean(dim=-1, keepdim=False) # shape of B * HW
        
        loss = 0
        for idx in range(bs):
            pos_corr = corr[idx][pt_map[idx]].sum()
            neg_corr = corr[idx][~pt_map[idx]].sum()
            sample_loss = -1 * torch.log(pos_corr / (neg_corr + pos_corr + 1e-10))
            loss += sample_loss
            
        return loss / bs