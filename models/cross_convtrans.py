import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import ConvBlock,exp_ConvBlock,OutputNet
from models.crossvit import CrossAttentionBlock


class Cross_VGG16Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False,decoder_depth=2,norm_layer=nn.LayerNorm,decoder_embed_dim=512,decoder_num_heads=4,mlp_ratio=4.0):
        super().__init__()
        self.shot_token = nn.Parameter(torch.zeros(512))
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.scale_factor = 16//dcsize
        self.encoder = nn.Sequential(
            ConvBlock(cin=3, cout=64),
            ConvBlock(cin=64, cout=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=64, cout=128),
            ConvBlock(cin=128, cout=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=128, cout=256),
            ConvBlock(cin=256, cout=256),
            ConvBlock(cin=256, cout=256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=256, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
        )

        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        # self.conv_decoder = nn.Sequential(
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        # )
        # self.conv_decoder_p2 = OutputNet(dim=512)

        self._initialize_weights()
        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
            else:
                mod = torchvision.models.vgg16(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.encoder.state_dict().items())):
                temp_key = list(self.encoder.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.encoder.load_state_dict(fsd)
        
        self.exp_encoder = nn.Sequential(
            exp_ConvBlock(cin=3, cout=64),
            nn.MaxPool2d(2),
            exp_ConvBlock(cin=64, cout=128),
            nn.MaxPool2d(2),
            exp_ConvBlock(cin=128, cout=256),
            nn.MaxPool2d(2),
            exp_ConvBlock(cin=256, cout=512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.Cross_attention = nn.ModuleList(
            [CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                                qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def exp_cross_attention(self, x,y_,shot_num=3):
        y_ = y_.transpose(0, 1)  # y_ [N,3,3,64,64]->[3,N,3,64,64]
        y1 = []
        C = 0
        N = 0
        cnt = 0
        for yi in y_:
            cnt += 1
            if cnt > shot_num:
                break
            yi=self.exp_encoder(yi)
            N, C, _, _ = yi.shape
            y1.append(yi.squeeze(-1).squeeze(-1))  # yi [N,C,1,1]->[N,C]

        if shot_num > 0:
            y = torch.cat(y1, dim=0).reshape(shot_num, N, C).to(x.device)
        else:
            y = self.shot_token.repeat(
                y_.shape[1], 1).unsqueeze(0).to(x.device)
        y = y.transpose(0, 1)  # y [3,N,C]->[N,3,C]

        # apply Transformer blocks
        for blk in self.Cross_attention:
            x = blk(x, y)
        x = self.decoder_norm(x)

        return x
    
    def forward(self, x,y_,shot_num=3):
        raw_x = self.encoder(x)
        bs, c, h, w = raw_x.shape

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        #cross-attention
        x_ = x.permute(1,0,2)
        x_ = self.exp_cross_attention(x_,y_,shot_num=shot_num)
        x_ = x_.permute(1,0,2)
        x = self.tran_decoder(x, (h, w))
        x= x+x_
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y
