import torch
import torch.nn as nn
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import ConvBlock, OutputNet
from models.similaritymatcher import DynamicSimilarityMatcher
import copy

class VGG16Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
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
        self.exampler_encoder = nn.Sequential(
            ConvBlock(cin=3, cout=64),
            ConvBlock(cin=64, cout=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=64, cout=128),
            ConvBlock(cin=128, cout=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=128, cout=256),
            ConvBlock(cin=256, cout=256),
            ConvBlock(cin=256, cout=256),
        )
        self.exampler_encoder.append(ConvBlock(cin=256, cout=512))
        # self.exampler_encoder.eval() 
        self.matcher = DynamicSimilarityMatcher(512, 512, 512)

        self.tran_decoder = Transformer(layers=4, dim = 513)
        self.tran_decoder_p2 = OutputNet(dim=513)

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
            
            ex_fsd = collections.OrderedDict()
            for i in range(len(self.exampler_encoder.state_dict().items())):
                temp_key = list(self.exampler_encoder.state_dict().items())[i][0]
                ex_fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.exampler_encoder.load_state_dict(ex_fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ex_list):
        raw_x = self.encoder(x)
        bs, c, h, w = raw_x.shape
        bs_ex_feature_vec_list = []
        for examplers in ex_list:
            ex_feature_vec_list = []
            for exampler in examplers:
                ex_feature_vec_list.append(torch.mean(self.exampler_encoder(torch.unsqueeze(exampler,dim=0)), dim=(2, 3), keepdim=False))
            # print(len(ex_feature_vec_list))
            bs_ex_feature_vec_list.append(torch.stack(ex_feature_vec_list[:3]).squeeze(1))
        bs_ex_feature_vec_list = torch.stack(bs_ex_feature_vec_list)
        # print(raw_x.shape, bs_ex_feature_vec_list.shape)
        

        
        # print(raw_x.shape, bs_ex_feature_vec_list.shape)
        simi_map, corr = self.matcher(raw_x, bs_ex_feature_vec_list)
        # print(simi_map.shape)
        
        # path-transformer
        x = simi_map.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c+1

        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c+1 , h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y, corr