import torch
import pdb
import numpy as np
import torch
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
from model.layers.utils import select_point_of_interest
from model.backbone.DCNv2.dcn_v2 import DCNv2

from model.layers.coord_conv import CoordConv
from inplace_abn import InPlaceABN

@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = len(cfg.DATASETS.DETECT_CLASSES)
        
        self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_HEADS
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': norm_func = group_norm
        else: norm_func = nn.Identity

        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.use_inplace_abn = cfg.MODEL.INPLACE_ABN
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.abn_activision = 'leaky_relu'

        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        if self.use_inplace_abn:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        else:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            if self.use_inplace_abn:
                feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision))
            else:
                feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True))
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################

        # edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

        # self.register_buffer('offset_std', torch.tensor([20.0244, 8.3567]).view(1,2,1,1))
        # self.register_buffer('offset_mean', torch.tensor([0.1060, -6.5576]).view(1,2,1,1))
        self.seprate_depth_channels = cfg.MODEL.HEAD.SEPRATE_DEPTH_CHANNELS
        self.add_ground_depth = cfg.MODEL.HEAD.ADD_GROUND_DEPTH
        self.detach_ground_depth = cfg.MODEL.HEAD.DETACH_GROUND_DEPTH
        self.dilated_ground_depth = cfg.MODEL.HEAD.DILATED_GROUND_DEPTH 
        ground_depth_output_channel = 2 if not cfg.MODEL.HEAD.GD_XY else 6
        self.freeze_backbone = cfg.MODEL.HEAD.FREEZE_BACKBONE
        
        if cfg.MODEL.HEAD.UNCERTAINTY_ALIGN:
            if self.add_ground_depth:
                self.register_buffer('uncertainty_alignment_k', torch.ones(4 + ground_depth_output_channel//2))
                self.register_buffer('uncertainty_alignment_b', torch.zeros(4 + ground_depth_output_channel//2))
            else:
                self.register_buffer('uncertainty_alignment_k', torch.ones(4))
                self.register_buffer('uncertainty_alignment_b', torch.zeros(4))

        if self.add_ground_depth:

            gd_conv_cls = CoordConv if cfg.MODEL.HEAD.GD_DEPTH_COORD_CONV else nn.Conv2d

            self.ground_depth_predictor = nn.Sequential(
                gd_conv_cls(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                # InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision) if self.use_inplace_abn else \
                #     norm_func(self.head_conv, momentum=self.bn_momentum), 
                nn.BatchNorm2d(self.head_conv), 
                nn.ReLU(inplace=True),
                gd_conv_cls(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.head_conv), 
                nn.ReLU(inplace=True),
                gd_conv_cls(self.head_conv, ground_depth_output_channel, kernel_size=1, padding=1 // 2, bias=True)
                # 2 for ground depth and ground depth uncertainty
            ) if not self.dilated_ground_depth else \
                nn.Sequential(
                gd_conv_cls(in_channels, self.head_conv, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(self.head_conv), 
                nn.ReLU(inplace=True),
                gd_conv_cls(self.head_conv, self.head_conv, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(self.head_conv), 
                nn.ReLU(inplace=True),
                gd_conv_cls(self.head_conv, ground_depth_output_channel, kernel_size=1, padding=1 // 2, bias=True)
                # 2 for ground depth and ground depth uncertainty
            )
            if cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                torch.nn.init.xavier_normal_(self.ground_depth_predictor[-1].weight[-1], gain=0.01)
                if cfg.MODEL.HEAD.GD_XY:
                    torch.nn.init.xavier_normal_(self.ground_depth_predictor[-1].weight[-3], gain=0.01)
                    torch.nn.init.xavier_normal_(self.ground_depth_predictor[-1].weight[-5], gain=0.01)

                # get the weight of ground depth uncertainty
            
            if self.freeze_backbone:
                for param in self.class_head.parameters():
                    param.requires_grad = False
                for param in self.reg_features.parameters():
                    param.requires_grad = False
                for param in self.reg_heads.parameters():
                    param.requires_grad = False
                if self.enable_edge_fusion:
                    for param in self.trunc_heatmap_conv.parameters():
                        param.requires_grad = False
                    for param in self.trunc_offset_conv.parameters():
                        param.requires_grad = False

    def forward(self, features, targets):
        b, c, h, w = features.shape                   # bs, 64, 96, 320
        # output classification
        feature_cls = self.class_head[:-1](features)  # bs, 256, 96, 320
        output_cls = self.class_head[-1](feature_cls) # bs, 3, 96, 320

        output_regs = []
        # output regression
        reg_feature = self.reg_features[0](features)
        if self.seprate_depth_channels:
            reg_fearure_for_depth = self.reg_features[1](features)
        for i, reg_feature_head in enumerate(self.reg_features):

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                if (i == 6 or i == 7) and self.seprate_depth_channels:
                    output_reg = reg_output_head(reg_fearure_for_depth)
                else:
                    output_reg = reg_output_head(reg_feature)

                # apply edge feature enhancement
                if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    # output_reg = output_reg * self.offset_std + self.offset_mean

                    edge_indices = torch.stack([t.get_field("edge_indices") for t in targets]) # B x K x 2
                    edge_lens = torch.stack([t.get_field("edge_len") for t in targets]) # B
                    
                    # normalize
                    grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
                    grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1
                    grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

                    # apply edge fusion for both offset and heatmap
                    feature_for_fusion = torch.cat((feature_cls, reg_feature), dim=1)
                    edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)

                    edge_cls_feature = edge_features[:, :self.head_conv, ...]
                    edge_offset_feature = edge_features[:, self.head_conv:, ...]
                    edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
                    edge_offset_output = self.trunc_offset_conv(edge_offset_feature)
                    
                    for k in range(b):
                        edge_indice_k = edge_indices[k, :edge_lens[k]]
                        output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                        output_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset_output[k, :, :edge_lens[k]]
                
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        if self.add_ground_depth:
            ground_du = self.ground_depth_predictor(features.detach() if self.detach_ground_depth else features)
            return {'cls': output_cls, 'reg': output_regs, 'ground_du': ground_du}


        return {'cls': output_cls, 'reg': output_regs}
    def train(self, mode=True):
        if self.freeze_backbone:
            self.class_head.eval()
            self.reg_features.eval()
            self.reg_heads.eval()
            self.trunc_heatmap_conv.eval()
            self.trunc_offset_conv.eval()
            self.ground_depth_predictor.train()
        else:
            super(_predictor, self).train(mode)

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)