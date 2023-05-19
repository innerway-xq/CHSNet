import torch
import torch.nn as nn
from einops import rearrange
import cv2


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

def MincountLoss(inputs,examplar_list,dcsize,device):
    ones=torch.ones(1).to(device)
    inputs = inputs.squeeze(0)
    loss_fn=nn.MSELoss(reduction='sum')
    Loss=0.0

    for i,(dmap,y1,x1,y2,x2) in enumerate(examplar_list):
        pred_box=inputs[:,x1//dcsize:x2//dcsize,y1//dcsize:y2//dcsize]
        X=torch.sum(pred_box,dim=(1,2))/100.0
        if X.item()<=1:
            Loss+=loss_fn(X,ones)

    return Loss

def PerturbationLoss(input,examplar,desize,device):
    input=input.squeeze()
    loss_fn=nn.MSELoss(reduction='sum')
    Loss=0.0

    for i,(dmap,y1,x1,y2,x2) in enumerate(examplar):
        if y2[0]-y1[0]<desize or x2[0]-x1[0]<desize:
            continue
        pred_box=input[y1[0]//desize : y2[0]//desize, x1[0]//desize : x2[0]//desize]/100.0
        box=dmap.squeeze().cpu().numpy()
        # if y2[0]//desize>input.shape[0] or x2[0]//desize>input.shape[1]:
        #     print(input.shape)
        #     print(y2[0],x2[0])
        box=cv2.resize(box,(pred_box.shape[1],pred_box.shape[0]))
        box=torch.from_numpy(box).to(device)

        Loss+=loss_fn(box,pred_box)

    return Loss
        
        


    