from sre_compile import _ASSERT_CODES
import torch
import math
import torch.distributed as dist
import pdb

from torch.nn import functional as F
from utils.comm import get_world_size

from model.anno_encoder import Anno_Encoder
from model.layers.utils import select_point_of_interest
from model.utils import Uncertainty_Reg_Loss, Laplace_Loss

from model.layers.focal_loss import *
from model.layers.iou_loss import *
from model.head.depth_losses import *
from model.layers.utils import Converter_key2channel, soft_get

def make_loss_evaluator(cfg):
	loss_evaluator = Loss_Computation(cfg=cfg)
	return loss_evaluator

class Loss_Computation():
	def __init__(self, cfg):
		
		self.anno_encoder = Anno_Encoder(cfg)
		self.key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
		
		self.max_objs = cfg.DATASETS.MAX_OBJECTS
		self.center_sample = cfg.MODEL.HEAD.CENTER_SAMPLE
		self.regress_area = cfg.MODEL.HEAD.REGRESSION_AREA
		self.heatmap_type = cfg.MODEL.HEAD.HEATMAP_TYPE
		self.corner_depth_sp = cfg.MODEL.HEAD.SUPERVISE_CORNER_DEPTH
		self.loss_keys = cfg.MODEL.HEAD.LOSS_NAMES

		self.world_size = get_world_size()
		self.dim_weight = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_WEIGHT).view(1, 3)
		self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

		# loss functions
		loss_types = cfg.MODEL.HEAD.LOSS_TYPE
		self.cls_loss_fnc = FocalLoss(cfg.MODEL.HEAD.LOSS_PENALTY_ALPHA, cfg.MODEL.HEAD.LOSS_BETA) # penalty-reduced focal loss
		self.iou_loss = IOULoss(loss_type=loss_types[2]) # iou loss for 2D detection

		# depth loss
		if loss_types[3] == 'berhu': self.depth_loss = Berhu_Loss()
		elif loss_types[3] == 'inv_sig': self.depth_loss = Inverse_Sigmoid_Loss()
		elif loss_types[3] == 'log': self.depth_loss = Log_L1_Loss()
		elif loss_types[3] == 'L1': self.depth_loss = F.l1_loss
		else: raise ValueError

		# regular regression loss
		self.reg_loss = loss_types[1]
		self.reg_loss_fnc = F.l1_loss if loss_types[1] == 'L1' else F.smooth_l1_loss
		self.keypoint_loss_fnc = F.l1_loss

		# multi-bin loss setting for orientation estimation
		self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
		self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
		self.trunc_offset_loss_type = cfg.MODEL.HEAD.TRUNCATION_OFFSET_LOSS

		self.loss_weights = {}
		for key, weight in zip(cfg.MODEL.HEAD.LOSS_NAMES, cfg.MODEL.HEAD.INIT_LOSS_WEIGHT): self.loss_weights[key] = weight

		# whether to compute corner loss
		self.compute_direct_depth_loss = 'depth_loss' in self.loss_keys
		self.compute_keypoint_depth_loss = 'keypoint_depth_loss' in self.loss_keys
		self.compute_weighted_depth_loss = 'weighted_avg_depth_loss' in self.loss_keys
		self.compute_corner_loss = 'corner_loss' in self.loss_keys
		self.separate_trunc_offset = 'trunc_offset_loss' in self.loss_keys
		
		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.compute_keypoint_corner = 'corner_offset' in self.key2channel.keys
		self.corner_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys

		self.uncertainty_weight = cfg.MODEL.HEAD.UNCERTAINTY_WEIGHT # 1.0
		self.keypoint_xy_weights = cfg.MODEL.HEAD.KEYPOINT_XY_WEIGHT # [1, 1]
		self.keypoint_norm_factor = cfg.MODEL.HEAD.KEYPOINT_NORM_FACTOR # 1.0
		self.modify_invalid_keypoint_depths = cfg.MODEL.HEAD.MODIFY_INVALID_KEYPOINT_DEPTH

		# depth used to compute 8 corners
		self.corner_loss_depth = cfg.MODEL.HEAD.CORNER_LOSS_DEPTH
		self.eps = 1e-5
		self.dense_depth = cfg.MODEL.HEAD.DENSE_DEPTH
		self.direct_3d_loss = cfg.MODEL.HEAD.DIRECT_3D_LOSS
		if self.direct_3d_loss:
			self.direct_3d_loss_func = F.l1_loss
			self.direct_3d_loss_weight = cfg.MODEL.HEAD.DIRECT_3D_LOSS_WEIGHT
			self.direct_3d_loss_sched = cfg.MODEL.HEAD.DIRECT_3D_LOSS_SCHED
		self.freeze_backbone = cfg.MODEL.HEAD.FREEZE_BACKBONE and cfg.MODEL.HEAD.ADD_GROUND_DEPTH
		self.add_ground_depth = cfg.MODEL.HEAD.ADD_GROUND_DEPTH
		self.ground_depth_loss_weight = cfg.MODEL.HEAD.GROUND_DEPTH_LOSS_WEIGHT
		self.gd_depth_decode_mode = cfg.MODEL.HEAD.GD_DEPTH_DECODE_MODE
		self.gd_depth_decode_pos = cfg.MODEL.HEAD.GD_DEPTH_DECODE_POS
		assert self.gd_depth_decode_pos in ['before', 'after']
		self.gd_xy = cfg.MODEL.HEAD.GD_XY
		self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
		self.uncertainty_align = cfg.MODEL.HEAD.UNCERTAINTY_ALIGN
		self.uncertainty_align_loss_weight = cfg.MODEL.HEAD.UNCERTAINTY_ALIGN_LOSS_WEIGHT

	def prepare_targets(self, targets):
		# clses
		heatmaps = torch.stack([t.get_field("hm") for t in targets])
		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# 2d detection
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		bboxes = torch.stack([t.get_field("2d_bboxes") for t in targets])
		# 3d detection
		keypoints = torch.stack([t.get_field("keypoints") for t in targets])
		keypoints_depth_mask = torch.stack([t.get_field("keypoints_depth_mask") for t in targets])
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		alphas = torch.stack([t.get_field("alphas") for t in targets])
		orientations = torch.stack([t.get_field("orientations") for t in targets])
		# utils
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
		reg_weight = torch.stack([t.get_field("reg_weight") for t in targets])
		ori_imgs = torch.stack([t.get_field("ori_img") for t in targets])
		trunc_mask = torch.stack([t.get_field("trunc_mask") for t in targets])

		return_dict = dict(cls_ids=cls_ids, target_centers=target_centers, bboxes=bboxes, keypoints=keypoints, dimensions=dimensions,
			locations=locations, rotys=rotys, alphas=alphas, calib=calibs, pad_size=pad_size, reg_mask=reg_mask, reg_weight=reg_weight,
			offset_3D=offset_3D, ori_imgs=ori_imgs, trunc_mask=trunc_mask, orientations=orientations, keypoints_depth_mask=keypoints_depth_mask,
		)

		if self.dense_depth or self.add_ground_depth:
			dense_depth_pts = torch.stack([t.get_field("dense_depth_pts") for t in targets])
			return_dict['dense_depth_pts'] = dense_depth_pts

		return heatmaps, return_dict

	def prepare_predictions(self, targets_variables, predictions):
		pred_regression = predictions['reg']
		batch, channel, feat_h, feat_w = pred_regression.shape

		# 1. get the representative points
		targets_bbox_points = targets_variables["target_centers"] # representative points

		reg_mask_gt = targets_variables["reg_mask"].bool()
		flatten_reg_mask_gt = reg_mask_gt.view(-1)

		# the corresponding image_index for each object, used for finding pad_size, calib and so on
		batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1)
		batch_idxs = batch_idxs[flatten_reg_mask_gt].to(reg_mask_gt.device) 

		valid_targets_bbox_points = targets_bbox_points.view(-1, 2)[flatten_reg_mask_gt]

		# fcos-style targets for 2D
		target_bboxes_2D = targets_variables['bboxes'].view(-1, 4)[flatten_reg_mask_gt]
		target_bboxes_height = target_bboxes_2D[:, 3] - target_bboxes_2D[:, 1]
		target_bboxes_width = target_bboxes_2D[:, 2] - target_bboxes_2D[:, 0]

		target_regression_2D = torch.cat((valid_targets_bbox_points - target_bboxes_2D[:, :2], target_bboxes_2D[:, 2:] - valid_targets_bbox_points), dim=1)
		mask_regression_2D = (target_bboxes_height > 0) & (target_bboxes_width > 0)
		target_regression_2D = target_regression_2D[mask_regression_2D]

		# targets for 3D
		target_clses = targets_variables["cls_ids"].view(-1)[flatten_reg_mask_gt]
		target_depths_3D = targets_variables['locations'][..., -1].view(-1)[flatten_reg_mask_gt]
		target_rotys_3D = targets_variables['rotys'].view(-1)[flatten_reg_mask_gt]
		target_alphas_3D = targets_variables['alphas'].view(-1)[flatten_reg_mask_gt]
		target_offset_3D = targets_variables["offset_3D"].view(-1, 2)[flatten_reg_mask_gt]
		target_dimensions_3D = targets_variables['dimensions'].view(-1, 3)[flatten_reg_mask_gt]
		
		target_orientation_3D = targets_variables['orientations'].view(-1, targets_variables['orientations'].shape[-1])[flatten_reg_mask_gt]
		target_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, target_offset_3D, target_depths_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)

		target_corners_3D = self.anno_encoder.encode_box3d(target_rotys_3D, target_dimensions_3D, target_locations_3D)
		target_bboxes_3D = torch.cat((target_locations_3D, target_dimensions_3D, target_rotys_3D[:, None]), dim=1)

		target_trunc_mask = targets_variables['trunc_mask'].view(-1)[flatten_reg_mask_gt]
		obj_weights = targets_variables["reg_weight"].view(-1)[flatten_reg_mask_gt]

		# 2. extract corresponding predictions
		pred_regression_pois_3D = select_point_of_interest(batch, targets_bbox_points, pred_regression).view(-1, channel)[flatten_reg_mask_gt]
		
		pred_regression_2D = F.relu(pred_regression_pois_3D[mask_regression_2D, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois_3D[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets_3D = pred_regression_pois_3D[:, self.key2channel('3d_dim')]
		pred_orientation_3D = torch.cat((pred_regression_pois_3D[:, self.key2channel('ori_cls')], 
									pred_regression_pois_3D[:, self.key2channel('ori_offset')]), dim=1)
		
		# decode the pred residual dimensions to real dimensions
		pred_dimensions_3D = self.anno_encoder.decode_dimension(target_clses, pred_dimensions_offsets_3D)

		# preparing outputs
		targets = { 'reg_2D': target_regression_2D, 'offset_3D': target_offset_3D, 'depth_3D': target_depths_3D, 'orien_3D': target_orientation_3D,
					'dims_3D': target_dimensions_3D, 'corners_3D': target_corners_3D, 'width_2D': target_bboxes_width, 'rotys_3D': target_rotys_3D,
					'cat_3D': target_bboxes_3D, 'trunc_mask_3D': target_trunc_mask, 'height_2D': target_bboxes_height,
				}

		preds = {'reg_2D': pred_regression_2D, 'offset_3D': pred_offset_3D, 'orien_3D': pred_orientation_3D, 'dims_3D': pred_dimensions_3D}
		
		reg_nums = {'reg_2D': mask_regression_2D.sum(), 'reg_3D': flatten_reg_mask_gt.sum(), 'reg_obj': flatten_reg_mask_gt.sum()}
		weights = {'object_weights': obj_weights}

		# predict the depth with direct regression
		if self.pred_direct_depth:
			if self.dense_depth:
				# pred_regression: b,c,h,w

				pred_du_dense = torch.cat( (pred_regression[:, self.key2channel('depth'), :, :], pred_regression[:, self.key2channel('depth_uncertainty'), :, :]), dim = 1)
				# du for depth and uncertainty

				target_center_float = targets_bbox_points + targets_variables["offset_3D"]

				dense_coords = targets_variables['dense_depth_pts'][...,:2]
				dense_coords_depths = targets_variables['dense_depth_pts'][...,-1]
				# b, max_objs, sample_num x sample_num, 2
				# b, max_objs, sample_num x sample_num

				pred_depths_offset_3D = []
				pred_depths_offset_dense = []
				depths_dense_target = []

				pred_uncertainty = []
				pred_uncertainty_dense = []
				for b_idx in range(batch):
					pred_du_iter, pred_du_valid = soft_get(pred_du_dense[b_idx], target_center_float[b_idx])
					assert torch.all(pred_du_valid)
					# all direct depth should be valid

					pred_depths_offset_3D_iter = pred_du_iter[0, reg_mask_gt[b_idx]]
					pred_depths_offset_3D.append(pred_depths_offset_3D_iter)

					pred_uncertainty_iter = pred_du_iter[1, reg_mask_gt[b_idx]]
					pred_uncertainty.append(pred_uncertainty_iter)

					dense_coords_this_bacth = dense_coords[b_idx, reg_mask_gt[b_idx]].view(-1,2)
					dense_coords_depths_this_bacth = dense_coords_depths[b_idx, reg_mask_gt[b_idx]].flatten()
					if dense_coords_this_bacth.size(0) == 0:
						continue
					pred_du_dense_iter, pred_du_dense_valid_iter = soft_get(pred_du_dense[b_idx], dense_coords_this_bacth )
					# n
					pred_depths_offset_dense.append(pred_du_dense_iter[0])
					pred_uncertainty_dense.append(pred_du_dense_iter[1])
					assert torch.all(dense_coords_depths_this_bacth[pred_du_dense_valid_iter] > 0)
					depths_dense_target.append(dense_coords_depths_this_bacth[pred_du_dense_valid_iter])


				pred_depths_offset_3D = torch.cat(pred_depths_offset_3D, dim = 0)
				pred_depths_offset_dense = torch.cat(pred_depths_offset_dense, dim = 0)
				depths_dense_target = torch.cat(depths_dense_target, dim = 0)

				pred_uncertainty = torch.cat(pred_uncertainty, dim = 0)
				pred_uncertainty_dense = torch.cat(pred_uncertainty_dense, dim = 0)

				pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
				preds['depth_3D'] = pred_direct_depths_3D

				pred_direct_depths_dense = self.anno_encoder.decode_depth(pred_depths_offset_dense)
				preds['dense_depth'] = pred_direct_depths_dense
				targets['dense_depth'] = depths_dense_target
			else:
				pred_depths_offset_3D = pred_regression_pois_3D[:, self.key2channel('depth')].squeeze(-1)
				pred_direct_depths_3D = self.anno_encoder.decode_depth(pred_depths_offset_3D)
				preds['depth_3D'] = pred_direct_depths_3D

		# predict the uncertainty of depth regression
		if self.depth_with_uncertainty:
			if self.dense_depth:
				preds['depth_uncertainty'] = pred_uncertainty
				preds['pred_uncertainty_dense'] = pred_uncertainty_dense
				if self.uncertainty_range is not None:
					preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])
					preds['pred_uncertainty_dense'] = torch.clamp(preds['pred_uncertainty_dense'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])
				
			else:
				preds['depth_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('depth_uncertainty')].squeeze(-1)
				
				if self.uncertainty_range is not None:
					preds['depth_uncertainty'] = torch.clamp(preds['depth_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])


		if self.compute_keypoint_corner:
			# targets for keypoints
			target_corner_keypoints = targets_variables["keypoints"].view(flatten_reg_mask_gt.shape[0], -1, 3)[flatten_reg_mask_gt]
			targets['keypoints'] = target_corner_keypoints[..., :2]
			targets['keypoints_mask'] = target_corner_keypoints[..., -1]
			reg_nums['keypoints'] = targets['keypoints_mask'].sum()

			# mask for whether depth should be computed from certain group of keypoints
			target_corner_depth_mask = targets_variables["keypoints_depth_mask"].view(-1, 3)[flatten_reg_mask_gt]
			targets['keypoints_depth_mask'] = target_corner_depth_mask

			# predictions for keypoints
			pred_keypoints_3D = pred_regression_pois_3D[:, self.key2channel('corner_offset')]
			pred_keypoints_3D = pred_keypoints_3D.view(flatten_reg_mask_gt.sum(), -1, 2)
			pred_keypoints_depths_3D = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoints_3D, pred_dimensions_3D,
														targets_variables['calib'], batch_idxs)

			preds['keypoints'] = pred_keypoints_3D			
			preds['keypoints_depths'] = pred_keypoints_depths_3D

		# predict the uncertainties of the solved depths from groups of keypoints
		if self.corner_with_uncertainty:
			preds['corner_offset_uncertainty'] = pred_regression_pois_3D[:, self.key2channel('corner_uncertainty')]

			if self.uncertainty_range is not None:
				preds['corner_offset_uncertainty'] = torch.clamp(preds['corner_offset_uncertainty'], min=self.uncertainty_range[0], max=self.uncertainty_range[1])


		# compute the corners of the predicted 3D bounding boxes for the corner loss
		if self.corner_loss_depth == 'direct':
			pred_corner_depth_3D = pred_direct_depths_3D

		elif self.corner_loss_depth == 'keypoint_mean':
			pred_corner_depth_3D = preds['keypoints_depths'].mean(dim=1)
		
		else:
			assert self.corner_loss_depth in ['soft_combine', 'hard_combine']
			# make sure all depths and their uncertainties are predicted
			pred_combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(-1), preds['corner_offset_uncertainty']), dim=1).exp()
			pred_combined_depths = torch.cat((pred_direct_depths_3D.unsqueeze(-1), preds['keypoints_depths']), dim=1)
			
			if self.corner_loss_depth == 'soft_combine':
				pred_uncertainty_weights = 1 / pred_combined_uncertainty
				pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
				pred_corner_depth_3D = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)
				preds['weighted_depths'] = pred_corner_depth_3D
			
			elif self.corner_loss_depth == 'hard_combine':
				pred_corner_depth_3D = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		# compute the corners
		pred_locations_3D = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_corner_depth_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)
		# decode rotys and alphas
		pred_rotys_3D, _ = self.anno_encoder.decode_axes_orientation(pred_orientation_3D, pred_locations_3D)
		# encode corners
		pred_corners_3D = self.anno_encoder.encode_box3d(pred_rotys_3D, pred_dimensions_3D, pred_locations_3D)
		# concatenate all predictions
		pred_bboxes_3D = torch.cat((pred_locations_3D, pred_dimensions_3D, pred_rotys_3D[:, None]), dim=1)

		preds.update({'corners_3D': pred_corners_3D, 'rotys_3D': pred_rotys_3D, 'cat_3D': pred_bboxes_3D})

		if self.direct_3d_loss:
			preds['pred_locations_3D'] = self.anno_encoder.decode_location_flatten(valid_targets_bbox_points, pred_offset_3D, pred_direct_depths_3D, 
										targets_variables['calib'], targets_variables['pad_size'], batch_idxs)[:,:2]
			targets['target_locations_3D'] = target_locations_3D[:,:2]

		return targets, preds, reg_nums, weights

	def __call__(self, predictions, targets, iters = None, max_iter = None):
		if self.freeze_backbone:
			return self.freeze_call(predictions, targets, iters, max_iter)
		elif self.add_ground_depth:
			d1, d2, all_ground_depths, all_ground_uncertainty = self.freeze_call(predictions, targets, iters, max_iter, return_depths = True)
			d3, d4 = self.normal_call(predictions, targets, iters, max_iter, all_ground_depths, all_ground_uncertainty)
			return dict(d1, **d3), dict(d2, **d4)
		else:
			return self.normal_call(predictions, targets, iters, max_iter)
	def freeze_call(self, predictions, targets, iters = None, max_iter = None, return_depths = False):
		ground_du = predictions['ground_du']
		if self.gd_depth_decode_pos == 'before':
			ground_du = torch.cat([self.anno_encoder.decode_depth_with_mode(ground_du[:,:1], self.gd_depth_decode_mode), ground_du[:,1:]], dim = 1)
		targets_heatmap, targets_variables = self.prepare_targets(targets)

		targets_bbox_points = targets_variables["target_centers"]
		reg_mask_gt = targets_variables["reg_mask"].bool()

		target_center_float = targets_bbox_points + targets_variables["offset_3D"]

		dense_coords = targets_variables['dense_depth_pts'][...,:2]
		dense_coords_depths = targets_variables['dense_depth_pts'][...,-1]
		# b, max_objs, sample_num x sample_num, 2
		# b, max_objs, sample_num x sample_num

		# pred_depths_offset_3D = []
		pred_depths_offset_dense = []
		depths_dense_target = []

		# pred_uncertainty = []
		pred_uncertainty_dense = []

		if self.gd_xy:
			dense_x = []
			dense_x_uncertainty = []
			dense_y = []
			dense_y_uncertainty = []

			dense_x_target = targets_variables['dense_depth_pts'][...,-3]
			dense_y_target = targets_variables['dense_depth_pts'][...,-2]

			dense_x_target_all = []
			dense_y_target_all = []


		if return_depths:
			assert self.add_ground_depth
			kpts = targets_variables['keypoints'][...,:2] + targets_variables["target_centers"].unsqueeze(2)
			kpts_02 = kpts[:,:,[0,2],:]
			kpts_13 = kpts[:,:,[1,3],:]
			kpts_bot = kpts[:,:,8,:]

			all_ground_depths = []
			all_ground_uncertainty = []

		for b_idx in range(ground_du.shape[0]):

			dense_coords_this_bacth = dense_coords[b_idx, reg_mask_gt[b_idx]].view(-1,2)
			dense_coords_depths_this_bacth = dense_coords_depths[b_idx, reg_mask_gt[b_idx]].flatten()
			if dense_coords_this_bacth.size(0) == 0:
				continue
			pred_du_dense_iter, pred_du_dense_valid_iter = soft_get(ground_du[b_idx], dense_coords_this_bacth )
			# n
			pred_depths_offset_dense.append(pred_du_dense_iter[0])
			pred_uncertainty_dense.append(pred_du_dense_iter[1])
			if self.gd_xy:

				dense_x.append(pred_du_dense_iter[2])
				dense_x_uncertainty.append(pred_du_dense_iter[3])
				dense_y.append(pred_du_dense_iter[4])
				dense_y_uncertainty.append(pred_du_dense_iter[5])

				dense_x_target_this_batch = dense_x_target[b_idx, reg_mask_gt[b_idx]].flatten()
				dense_y_target_this_batch = dense_y_target[b_idx, reg_mask_gt[b_idx]].flatten()

				dense_x_target_all.append(dense_x_target_this_batch[pred_du_dense_valid_iter])
				dense_y_target_all.append(dense_y_target_this_batch[pred_du_dense_valid_iter])

				xxa = targets_variables['calib'][b_idx].P[0,0]
				xxb = targets_variables['calib'][b_idx].P[0,-1]
				xxc = targets_variables['calib'][b_idx].P[-1,-1]
				xxd = targets_variables['calib'][b_idx].P[0,2]

				yya = targets_variables['calib'][b_idx].P[1,1]
				yyb = targets_variables['calib'][b_idx].P[1,-1]
				yyc = targets_variables['calib'][b_idx].P[-1,-1]
				yyd = targets_variables['calib'][b_idx].P[1,2]

			assert torch.all(dense_coords_depths_this_bacth[pred_du_dense_valid_iter] > 0)
			depths_dense_target.append(dense_coords_depths_this_bacth[pred_du_dense_valid_iter])

			if return_depths:
				kpts_02_this_batch = kpts_02[b_idx, reg_mask_gt[b_idx]].view(-1,2)
				kpts_13_this_batch = kpts_13[b_idx, reg_mask_gt[b_idx]].view(-1,2)
				# n*2*2, first 2 means 02 or 13
				kpts_bot_this_batch = kpts_bot[b_idx, reg_mask_gt[b_idx]]
				# n*2
				pred_ground_du_02, _ = soft_get(ground_du[b_idx], kpts_02_this_batch, force_get=True, force_get_uncertainty = True)
				pred_ground_du_13, _ = soft_get(ground_du[b_idx], kpts_13_this_batch, force_get=True, force_get_uncertainty = True)
				pred_ground_du_bot, _ = soft_get(ground_du[b_idx], kpts_bot_this_batch, force_get=True, force_get_uncertainty = True)

				pred_ground_du_02_depth = pred_ground_du_02[0]
				pred_ground_du_13_depth = pred_ground_du_13[0]
				pred_ground_du_bot_depth = pred_ground_du_bot[0]

				pred_ground_du_02_uncertainty = pred_ground_du_02[1]
				pred_ground_du_13_uncertainty = pred_ground_du_13[1]
				pred_ground_du_bot_uncertainty = pred_ground_du_bot[1]

				if self.gd_xy:
					pred_ground_du_02_x = pred_ground_du_02[2]
					pred_ground_du_13_x = pred_ground_du_13[2]
					pred_ground_du_bot_x = pred_ground_du_bot[2]

					pred_ground_du_02_x_2d = kpts_02_this_batch[:,0] * self.down_ratio -  targets_variables['pad_size'][b_idx][0]
					pred_ground_du_13_x_2d = kpts_13_this_batch[:,0] * self.down_ratio - targets_variables['pad_size'][b_idx][0]
					pred_ground_du_bot_x_2d = kpts_bot_this_batch[:,0] * self.down_ratio - targets_variables['pad_size'][b_idx][0]

					pred_ground_du_02_x_uncertainty = pred_ground_du_02[3].view(-1,2).mean(1, keepdim=True)
					pred_ground_du_13_x_uncertainty = pred_ground_du_13[3].view(-1,2).mean(1, keepdim=True)
					pred_ground_du_bot_x_uncertainty = pred_ground_du_bot[3]

					ground_depth_from_x_02 = (pred_ground_du_02_x * xxa + xxb - pred_ground_du_02_x_2d * xxc) / (pred_ground_du_02_x_2d - xxd)
					ground_depth_from_x_13 = (pred_ground_du_13_x * xxa + xxb - pred_ground_du_13_x_2d * xxc) / (pred_ground_du_13_x_2d - xxd)
					ground_depth_from_x_bot = (pred_ground_du_bot_x * xxa + xxb - pred_ground_du_bot_x_2d * xxc) / (pred_ground_du_bot_x_2d - xxd)

					ground_depth_from_x_02 = ground_depth_from_x_02.view(-1,2).mean(1, keepdim=True)
					ground_depth_from_x_13 = ground_depth_from_x_13.view(-1,2).mean(1, keepdim=True)

					pred_ground_du_02_y = pred_ground_du_02[4]
					pred_ground_du_13_y = pred_ground_du_13[4]
					pred_ground_du_bot_y = pred_ground_du_bot[4]

					pred_ground_du_02_y_2d = kpts_02_this_batch[:,1]* self.down_ratio - targets_variables['pad_size'][b_idx][1]
					pred_ground_du_13_y_2d = kpts_13_this_batch[:,1]* self.down_ratio - targets_variables['pad_size'][b_idx][1]
					pred_ground_du_bot_y_2d = kpts_bot_this_batch[:,1]* self.down_ratio - targets_variables['pad_size'][b_idx][1]

					pred_ground_du_02_y_uncertainty = pred_ground_du_02[5].view(-1,2).mean(1, keepdim=True)
					pred_ground_du_13_y_uncertainty = pred_ground_du_13[5].view(-1,2).mean(1, keepdim=True)
					pred_ground_du_bot_y_uncertainty = pred_ground_du_bot[5]


					ground_depth_from_y_02 = (pred_ground_du_02_y * yya + yyb - pred_ground_du_02_y_2d * yyc) / (pred_ground_du_02_y_2d - yyd)
					ground_depth_from_y_13 = (pred_ground_du_13_y * yya + yyb - pred_ground_du_13_y_2d * yyc) / (pred_ground_du_13_y_2d - yyd)
					ground_depth_from_y_bot = (pred_ground_du_bot_y * yya + yyb - pred_ground_du_bot_y_2d * yyc) / (pred_ground_du_bot_y_2d - yyd)

					ground_depth_from_y_02 = ground_depth_from_y_02.view(-1,2).mean(1, keepdim=True)
					ground_depth_from_y_13 = ground_depth_from_y_13.view(-1,2).mean(1, keepdim=True)


				if self.gd_depth_decode_pos == 'after':
					pred_ground_du_02_depth = self.anno_encoder.decode_depth_with_mode(pred_ground_du_02_depth, self.gd_depth_decode_mode)
					pred_ground_du_13_depth = self.anno_encoder.decode_depth_with_mode(pred_ground_du_13_depth, self.gd_depth_decode_mode)
					pred_ground_du_bot_depth = self.anno_encoder.decode_depth_with_mode(pred_ground_du_bot_depth, self.gd_depth_decode_mode)

				# 2nx2
				pred_ground_du_02_depth = pred_ground_du_02_depth.view(-1,2).mean(1, keepdim=True)
				pred_ground_du_13_depth = pred_ground_du_13_depth.view(-1,2).mean(1, keepdim=True)

				pred_ground_du_02_uncertainty = pred_ground_du_02_uncertainty.view(-1,2).mean(1, keepdim=True)
				pred_ground_du_13_uncertainty = pred_ground_du_13_uncertainty.view(-1,2).mean(1, keepdim=True)

				ground_depths = torch.cat((pred_ground_du_02_depth, pred_ground_du_13_depth, pred_ground_du_bot_depth.view(-1,1)), dim = 1)

				ground_depths_uncertainty = torch.cat((pred_ground_du_02_uncertainty, pred_ground_du_13_uncertainty, pred_ground_du_bot_uncertainty.view(-1,1)), dim = 1)

				if self.gd_xy:
					ground_depths = torch.cat([ground_depths, ground_depth_from_x_02, ground_depth_from_x_13, ground_depth_from_x_bot.view(-1,1),  \
						ground_depth_from_y_02, ground_depth_from_y_13, ground_depth_from_y_bot.view(-1,1),], dim = 1)
					ground_depths_uncertainty = torch.cat([ground_depths_uncertainty, \
						pred_ground_du_02_x_uncertainty, pred_ground_du_13_x_uncertainty, pred_ground_du_bot_x_uncertainty.view(-1,1),\
						pred_ground_du_02_y_uncertainty, pred_ground_du_13_y_uncertainty, pred_ground_du_bot_y_uncertainty.view(-1,1),], dim = 1)

				all_ground_depths.append(ground_depths)
				all_ground_uncertainty.append(ground_depths_uncertainty)

		if len(pred_depths_offset_dense) != 0:
			pred_direct_depths_dense = torch.cat(pred_depths_offset_dense, dim = 0)
			depths_dense_target = torch.cat(depths_dense_target, dim = 0)

		if self.gd_xy:
			dense_x = torch.cat(dense_x, dim = 0)
			dense_y = torch.cat(dense_y, dim = 0)

			dense_x_uncertainty = torch.cat(dense_x_uncertainty, dim = 0)
			dense_y_uncertainty = torch.cat(dense_y_uncertainty, dim = 0)

			dense_x_target_all = torch.cat(dense_x_target_all, dim = 0)
			dense_y_target_all = torch.cat(dense_y_target_all, dim = 0)

		if len(pred_depths_offset_dense) != 0:
			pred_uncertainty_dense = torch.cat(pred_uncertainty_dense, dim = 0)

			if self.uncertainty_range is not None:
				pred_uncertainty_dense = torch.clamp(pred_uncertainty_dense, min=self.uncertainty_range[0], max=self.uncertainty_range[1])

				if self.gd_xy:
					dense_x_uncertainty = torch.clamp(dense_x_uncertainty, min=self.uncertainty_range[0], max=self.uncertainty_range[1])
					dense_y_uncertainty = torch.clamp(dense_y_uncertainty, min=self.uncertainty_range[0], max=self.uncertainty_range[1])


		if self.gd_depth_decode_pos == 'after' and len(pred_depths_offset_dense) != 0:
			pred_direct_depths_dense = self.anno_encoder.decode_depth_with_mode(pred_direct_depths_dense, self.gd_depth_decode_mode)


		if len(pred_depths_offset_dense) != 0:
			depth_3D_loss = self.ground_depth_loss_weight * self.depth_loss(pred_direct_depths_dense, depths_dense_target, reduction='none')
			depth_3D_loss = depth_3D_loss * torch.exp(- pred_uncertainty_dense) + \
									pred_uncertainty_dense * self.ground_depth_loss_weight * self.uncertainty_weight
			depth_3D_loss = depth_3D_loss.mean()
			loss_dict = {'ground_depth_loss': depth_3D_loss}
			log_loss_dict = {'ground_depth_loss': depth_3D_loss.item()}
		else:
			loss_dict = {'ground_depth_loss': 0}
			log_loss_dict = {'ground_depth_loss': 0}


		if self.gd_xy:
			ground_x_loss = self.ground_depth_loss_weight * self.depth_loss(dense_x, dense_x_target_all, reduction = 'none')
			ground_x_loss = ground_x_loss * torch.exp(- dense_x_uncertainty) + dense_x_uncertainty * self.ground_depth_loss_weight * self.uncertainty_weight
			ground_x_loss = ground_x_loss.mean()

			ground_y_loss = self.ground_depth_loss_weight * self.depth_loss(dense_x, dense_y_target_all, reduction = 'none')
			ground_y_loss = ground_y_loss * torch.exp(- dense_y_uncertainty) + dense_y_uncertainty * self.ground_depth_loss_weight * self.uncertainty_weight
			ground_y_loss = ground_y_loss.mean()

			loss_dict['ground_x_loss'] = ground_x_loss
			loss_dict['ground_y_loss'] = ground_y_loss

			log_loss_dict['ground_x_loss'] = ground_x_loss.item()
			log_loss_dict['ground_y_loss'] = ground_y_loss.item()
		
		for k, v in loss_dict.items():
			if torch.isnan(v).sum() > 0:
				print(loss_dict)
				print(k+' loss is nan')
				# pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				print(loss_dict)
				print(k+' loss is inf')
				# pdb.set_trace()

		if return_depths:
			return loss_dict, log_loss_dict, all_ground_depths, all_ground_uncertainty

		return loss_dict, log_loss_dict

	def normal_call(self, predictions, targets, iters = None, max_iter = None, all_ground_depths = None, all_ground_uncertainty = None):
		targets_heatmap, targets_variables = self.prepare_targets(targets)

		pred_heatmap = predictions['cls']
		pred_targets, preds, reg_nums, weights = self.prepare_predictions(targets_variables, predictions)

		# heatmap loss
		if self.heatmap_type == 'centernet':
			hm_loss, num_hm_pos = self.cls_loss_fnc(pred_heatmap, targets_heatmap)
			hm_loss = self.loss_weights['hm_loss'] * hm_loss / torch.clamp(num_hm_pos, 1)

		else: raise ValueError

		# synthesize normal factors
		num_reg_2D = reg_nums['reg_2D']
		num_reg_3D = reg_nums['reg_3D']
		num_reg_obj = reg_nums['reg_obj']
		
		trunc_mask = pred_targets['trunc_mask_3D'].bool()
		num_trunc = trunc_mask.sum()
		num_nontrunc = num_reg_obj - num_trunc

		# IoU loss for 2D detection
		if num_reg_2D > 0:
			reg_2D_loss, iou_2D = self.iou_loss(preds['reg_2D'], pred_targets['reg_2D'])
			reg_2D_loss = self.loss_weights['bbox_loss'] * reg_2D_loss.mean()
			iou_2D = iou_2D.mean()
			depth_MAE = (preds['depth_3D'] - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

		if num_reg_3D > 0:
			# direct depth loss
			if self.compute_direct_depth_loss:

				if self.dense_depth:
					depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['dense_depth'], pred_targets['dense_depth'], reduction='none')
					real_depth_3D_loss = depth_3D_loss.detach().mean()
					
					if self.depth_with_uncertainty:
						depth_3D_loss = depth_3D_loss * torch.exp(- preds['pred_uncertainty_dense']) + \
								preds['pred_uncertainty_dense'] * self.loss_weights['depth_loss'] * self.uncertainty_weight
				else:
					depth_3D_loss = self.loss_weights['depth_loss'] * self.depth_loss(preds['depth_3D'], pred_targets['depth_3D'], reduction='none')
					real_depth_3D_loss = depth_3D_loss.detach().mean()
					
					if self.depth_with_uncertainty:
						depth_3D_loss = depth_3D_loss * torch.exp(- preds['depth_uncertainty']) + \
								preds['depth_uncertainty'] * self.loss_weights['depth_loss'] * self.uncertainty_weight
				
				depth_3D_loss = depth_3D_loss.mean()
				
			# offset_3D loss
			offset_3D_loss = self.reg_loss_fnc(preds['offset_3D'], pred_targets['offset_3D'], reduction='none').sum(dim=1)

			# use different loss functions for inside and outside objects
			if self.separate_trunc_offset:
				if self.trunc_offset_loss_type == 'L1':
					trunc_offset_loss = offset_3D_loss[trunc_mask]
				
				elif self.trunc_offset_loss_type == 'log':
					trunc_offset_loss = torch.log(1 + offset_3D_loss[trunc_mask])

				trunc_offset_loss = self.loss_weights['trunc_offset_loss'] * trunc_offset_loss.sum() / torch.clamp(trunc_mask.sum(), min=1)
				if offset_3D_loss[~trunc_mask].shape[0] == 0:
					offset_3D_loss = 0.
					print("skip offset 3d loss")
				else:
					offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss[~trunc_mask].mean()
			else:
				offset_3D_loss = self.loss_weights['offset_loss'] * offset_3D_loss.mean()

			# orientation loss
			if self.multibin:
				orien_3D_loss = self.loss_weights['orien_loss'] * \
								Real_MultiBin_loss(preds['orien_3D'], pred_targets['orien_3D'], num_bin=self.orien_bin_size)

			# dimension loss
			dims_3D_loss = self.reg_loss_fnc(preds['dims_3D'], pred_targets['dims_3D'], reduction='none') * self.dim_weight.type_as(preds['dims_3D'])
			dims_3D_loss = self.loss_weights['dims_loss'] * dims_3D_loss.sum(dim=1).mean()

			# with torch.no_grad(): pred_IoU_3D = get_iou_3d(preds['corners_3D'], pred_targets['corners_3D']).mean()
			
			# corner loss
			if self.compute_corner_loss:
				# N x 8 x 3
				corner_3D_loss = self.loss_weights['corner_loss'] * \
							self.reg_loss_fnc(preds['corners_3D'], pred_targets['corners_3D'], reduction='none').sum(dim=2).mean()

			if self.compute_keypoint_corner:
				# N x K x 3
				keypoint_loss = self.loss_weights['keypoint_loss'] * self.keypoint_loss_fnc(preds['keypoints'],
								pred_targets['keypoints'], reduction='none').sum(dim=2) * pred_targets['keypoints_mask']
				
				keypoint_loss = keypoint_loss.sum() / torch.clamp(pred_targets['keypoints_mask'].sum(), min=1)

				if self.compute_keypoint_depth_loss:
					pred_keypoints_depth, keypoints_depth_mask = preds['keypoints_depths'], pred_targets['keypoints_depth_mask'].bool()
					target_keypoints_depth = pred_targets['depth_3D'].unsqueeze(-1).repeat(1, 3)
					
					valid_pred_keypoints_depth = pred_keypoints_depth[keypoints_depth_mask]
					invalid_pred_keypoints_depth = pred_keypoints_depth[~keypoints_depth_mask].detach()
					
					# valid and non-valid
					valid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(valid_pred_keypoints_depth, 
															target_keypoints_depth[keypoints_depth_mask], reduction='none')
					
					invalid_keypoint_depth_loss = self.loss_weights['keypoint_depth_loss'] * self.reg_loss_fnc(invalid_pred_keypoints_depth, 
															target_keypoints_depth[~keypoints_depth_mask], reduction='none')
					
					# for logging
					log_valid_keypoint_depth_loss = valid_keypoint_depth_loss.detach().mean()

					if self.corner_with_uncertainty:
						# center depth, corner 0246 depth, corner 1357 depth
						pred_keypoint_depth_uncertainty = preds['corner_offset_uncertainty']

						valid_uncertainty = pred_keypoint_depth_uncertainty[keypoints_depth_mask]
						invalid_uncertainty = pred_keypoint_depth_uncertainty[~keypoints_depth_mask]

						valid_keypoint_depth_loss = valid_keypoint_depth_loss * torch.exp(- valid_uncertainty) + \
												self.loss_weights['keypoint_depth_loss'] * valid_uncertainty

						invalid_keypoint_depth_loss = invalid_keypoint_depth_loss * torch.exp(- invalid_uncertainty)

					# average
					valid_keypoint_depth_loss = valid_keypoint_depth_loss.sum() / torch.clamp(keypoints_depth_mask.sum(), 1)
					invalid_keypoint_depth_loss = invalid_keypoint_depth_loss.sum() / torch.clamp((~keypoints_depth_mask).sum(), 1)

					# the gradients of invalid depths are not back-propagated
					if self.modify_invalid_keypoint_depths:
						keypoint_depth_loss = valid_keypoint_depth_loss + invalid_keypoint_depth_loss
					else:
						keypoint_depth_loss = valid_keypoint_depth_loss
				
				# compute the average error for each method of depth estimation
				keypoint_MAE = (preds['keypoints_depths'] - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)
				
				center_MAE = keypoint_MAE[:, 0].mean()
				keypoint_02_MAE = keypoint_MAE[:, 1].mean()
				keypoint_13_MAE = keypoint_MAE[:, 2].mean()

				if self.corner_with_uncertainty:
					if self.pred_direct_depth and self.depth_with_uncertainty:
						if all_ground_depths is None or all_ground_uncertainty is None:
							combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths']), dim=1)
							combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty']), dim=1).exp()
							combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE), dim=1)
						else:
							all_ground_depths = torch.cat(all_ground_depths, dim = 0)
							all_ground_uncertainty = torch.cat(all_ground_uncertainty, dim = 0)

							ground_MAE = (all_ground_depths - pred_targets['depth_3D'].unsqueeze(-1)).abs() \
									/ pred_targets['depth_3D'].unsqueeze(-1)

							ground_02_MAE = ground_MAE[:,0][all_ground_uncertainty[:,0] < 5].mean()
							ground_13_MAE = ground_MAE[:,1][all_ground_uncertainty[:,1] < 5].mean()
							ground_bot_MAE = ground_MAE[:,2][all_ground_uncertainty[:,2] < 5].mean()


							if self.gd_xy:
								ground_x_02_MAE = ground_MAE[:,3][all_ground_uncertainty[:,3] < 5].mean()
								ground_x_13_MAE = ground_MAE[:,4][all_ground_uncertainty[:,4] < 5].mean()
								ground_x_bot_MAE = ground_MAE[:,5][all_ground_uncertainty[:,5] < 5].mean()

								ground_y_02_MAE = ground_MAE[:,6][all_ground_uncertainty[:,6] < 5].mean()
								ground_y_13_MAE = ground_MAE[:,7][all_ground_uncertainty[:,7] < 5].mean()
								ground_y_bot_MAE = ground_MAE[:,8][all_ground_uncertainty[:,8] < 5].mean()

							combined_depth = torch.cat((preds['depth_3D'].unsqueeze(1), preds['keypoints_depths'],all_ground_depths ), dim=1)
							combined_uncertainty = torch.cat((preds['depth_uncertainty'].unsqueeze(1), preds['corner_offset_uncertainty'], all_ground_uncertainty), dim=1).exp()
							combined_MAE = torch.cat((depth_MAE.unsqueeze(1), keypoint_MAE, ground_MAE), dim=1)
					else:
						combined_depth = preds['keypoints_depths']
						combined_uncertainty = preds['corner_offset_uncertainty'].exp()
						combined_MAE = keypoint_MAE

					# the oracle MAE
					lower_MAE = torch.min(combined_MAE, dim=1)[0]
					# the hard ensemble
					hard_MAE = combined_MAE[torch.arange(combined_MAE.shape[0]), combined_uncertainty.argmin(dim=1)]
					# the soft ensemble
					combined_weights = 1 / combined_uncertainty
					combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
					soft_depths = torch.sum(combined_depth * combined_weights, dim=1)
					soft_MAE = (soft_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']
					# the average ensemble
					mean_depths = combined_depth.mean(dim=1)
					mean_MAE = (mean_depths - pred_targets['depth_3D']).abs() / pred_targets['depth_3D']

					# average
					lower_MAE, hard_MAE, soft_MAE, mean_MAE = lower_MAE.mean(), hard_MAE.mean(), soft_MAE.mean(), mean_MAE.mean()
				
					if self.compute_weighted_depth_loss:
						soft_depth_loss = self.loss_weights['weighted_avg_depth_loss'] * \
										self.reg_loss_fnc(soft_depths, pred_targets['depth_3D'], reduction='mean')

			depth_MAE = depth_MAE.mean()
			if self.direct_3d_loss:
				direct_3d_loss = self.direct_3d_loss_func(preds['pred_locations_3D'], pred_targets['target_locations_3D']).mean() * self.direct_3d_loss_weight
				if self.direct_3d_loss_sched == 'sigmoid':
					assert iters is not None
					assert max_iter is not None
					frac = iters * 1.0 / max_iter - 0.66
					# -0.66 to 0.33
					direct_3d_loss_sched_weight = 1. / (1 + math.exp(- frac * 15 ))
					direct_3d_loss *= direct_3d_loss_sched_weight
		loss_dict = {
			'hm_loss':  hm_loss,
			'bbox_loss': reg_2D_loss,
			'dims_loss': dims_3D_loss,
			'orien_loss': orien_3D_loss,
		}

		log_loss_dict = {
			'2D_IoU': iou_2D.item(),
			# '3D_IoU': pred_IoU_3D.item(),
			'3D_IoU': -1.0,
		}

		MAE_dict = {}

		if self.separate_trunc_offset:
			loss_dict['offset_loss'] = offset_3D_loss
			loss_dict['trunc_offset_loss'] = trunc_offset_loss
		else:
			loss_dict['offset_loss'] = offset_3D_loss

		if self.compute_corner_loss:
			loss_dict['corner_loss'] = corner_3D_loss

		if self.pred_direct_depth:
			loss_dict['depth_loss'] = depth_3D_loss
			log_loss_dict['depth_loss'] = real_depth_3D_loss.item()
			MAE_dict['depth_MAE'] = depth_MAE.item()

		if self.compute_keypoint_corner:
			loss_dict['keypoint_loss'] = keypoint_loss

			MAE_dict.update({
				'center_MAE': center_MAE.item(),
				'02_MAE': keypoint_02_MAE.item(),
				'13_MAE': keypoint_13_MAE.item(),
			})

			if self.corner_with_uncertainty:
				MAE_dict.update({
					'lower_MAE': lower_MAE.item(),
					'hard_MAE': hard_MAE.item(),
					'soft_MAE': soft_MAE.item(),
					'mean_MAE': mean_MAE.item(),
				})
			if all_ground_depths is not None and all_ground_uncertainty is not None:
				MAE_dict.update({
					'ground_02_MAE':ground_02_MAE.item(),
					'ground_13_MAE':ground_13_MAE.item(),
					'ground_bot_MAE':ground_bot_MAE.item(),
				})
				if self.gd_xy:
					MAE_dict.update({
						'ground_x_02_MAE':ground_x_02_MAE.item(),
						'ground_x_13_MAE':ground_x_13_MAE.item(),
						'ground_x_bot_MAE':ground_x_bot_MAE.item(),

						'ground_y_02_MAE':ground_y_02_MAE.item(),
						'ground_y_13_MAE':ground_y_13_MAE.item(),
						'ground_y_bot_MAE':ground_y_bot_MAE.item(),
					})
		if self.compute_keypoint_depth_loss:
			loss_dict['keypoint_depth_loss'] = keypoint_depth_loss
			log_loss_dict['keypoint_depth_loss'] = log_valid_keypoint_depth_loss.item()

		if self.compute_weighted_depth_loss:
			loss_dict['weighted_avg_depth_loss'] = soft_depth_loss
		
		if self.direct_3d_loss:
			loss_dict['direct_3d_loss'] = direct_3d_loss
			if self.direct_3d_loss_sched == 'sigmoid':
				log_loss_dict['direct_3d_loss_weight'] = direct_3d_loss_sched_weight
		# loss_dict ===> log_loss_dict
		for key, value in loss_dict.items():
			if key not in log_loss_dict:
				if hasattr(value, 'item'):
					log_loss_dict[key] = value.item()
				else:
					log_loss_dict[key] = float(value)

		# stop when the loss has NaN or Inf
		for k, v in loss_dict.items():
			if not isinstance(v, torch.Tensor):
				continue
			if torch.isnan(v).sum() > 0:
				print(loss_dict)
				pdb.set_trace()
			if torch.isinf(v).sum() > 0:
				print(loss_dict)
				pdb.set_trace()

		log_loss_dict.update(MAE_dict)

		return loss_dict, log_loss_dict

def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
	gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst

	cls_losses = 0
	reg_losses = 0
	reg_cnt = 0
	for i in range(num_bin):
		# bin cls loss
		cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
		# regression loss
		valid_mask_i = (gt_ori[:, i] == 1)
		cls_losses += cls_ce_loss.mean()
		if valid_mask_i.sum() > 0:
			s = num_bin * 2 + i * 2
			e = s + 2
			pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
			reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
						F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

			reg_losses += reg_loss.sum()
			reg_cnt += valid_mask_i.sum()

	return cls_losses / num_bin + reg_losses / reg_cnt
