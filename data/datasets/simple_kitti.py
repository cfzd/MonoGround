import numpy as np
from data.datasets.kitti_utils import Calibration

class SimpleKITTIDataset(Dataset):
    def __init__(self, cfg, root, is_train=True, transforms=None, augment=True):

        self.check_cfg()
		self.imageset_txt = os.path.join("ImageSets", "train.txt")
		assert os.path.exists(self.imageset_txt), "ImageSets file not exist, dir = {}".format(self.imageset_txt)

        self.all_file_names = open(self.imageset_txt, "r").readlines()
        self.all_file_names = [name.strip() for name in self.all_file_names]

		self.all_data = {}
        for name in self.all_file_names:
            self.all_data[name] = [np.load(os.path.join(root, name+'.npz')), np.load(os.path.join(root, name+'f.npz'))]
        
        self.transforms = transforms

        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.dense_depth = cfg.MODEL.HEAD.DENSE_DEPTH

        assert is_train

    def __len__(self):
        return len(all_file_names)

	def __getitem__(self, idx):
        name = self.all_file_names[idx]
        data = self.all_data[name]
        if random.random() < 0.5:
            data = data[0]
        else:
            data = data[1]
        # flip augmentation
        
        target = ParamsList(image_size=(1280, 384), is_train=True) 
		target.add_field("cls_ids", data['cls_ids'])
		target.add_field("target_centers", data['target_centers'])
		target.add_field("keypoints", data['keypoints'])
		target.add_field("keypoints_depth_mask", data['keypoints_depth_mask'])
		target.add_field("dimensions", data['dimensions'])
		target.add_field("locations", data['locations'])
		target.add_field("calib", Calibration(P= data['calib']))
		target.add_field("reg_mask", data['reg_mask'])
		target.add_field("reg_weight", data['reg_weight'])
		target.add_field("offset_3D", data['offset_3D'])
		target.add_field("2d_bboxes", data['bboxes'])
		target.add_field("pad_size", data['pad_size'])
		target.add_field("ori_img", data['ori_img'])
		target.add_field("rotys", data['rotys'])
		target.add_field("trunc_mask", data['trunc_mask'])
		target.add_field("alphas", data['alphas'])
		target.add_field("orientations", data['orientations'])
		target.add_field("hm", data['heat_map'])
		target.add_field("gt_bboxes", data['gt_bboxes']) # for validation visualization
		target.add_field("occlusions", data['occlusions'])
		target.add_field("truncations", data['truncations'])

		if self.enable_edge_fusion:
			target.add_field('edge_len', data['input_edge_count'])
			target.add_field('edge_indices', data['input_edge_indices'])
		if self.dense_depth:
			target.add_field('dense_depth_pts', data['dense_depth_pts'])
        
        img = data['ori_img']

		if self.transforms is not None: img, target = self.transforms(img, target)
		return img, target, '004025' 
        # 004025 is not used during training, just a place holder
        # for val , we use original kitti implementation


    def check_cfg(self, cfg):

        assert cfg.DATASETS.DETECT_CLASSES ==  ("Car", "Pedestrian", "Cyclist")
        assert cfg.DATASETS.TRAIN_SPLIT == 'train'

        # whether to use right-view image
        assert not cfg.DATASETS.USE_RIGHT_IMAGE


        # input and output shapes
        assert cfg.INPUT.WIDTH_TRAIN == 1280
        assert cfg.INPUT.HEIGHT_TRAIN == 384
        assert cfg.MODEL.BACKBONE.DOWN_RATIO == 4
        
        # maximal length of extracted feature map when appling edge fusion
        assert cfg.DATASETS.MAX_OBJECTS == 40
        
        # filter invalid annotations
        assert cfg.DATASETS.FILTER_ANNO_ENABLE
        assert cfg.DATASETS.FILTER_ANNOS == [0.9, 20]

        # handling truncation
        assert not cfg.DATASETS.CONSIDER_OUTSIDE_OBJS
        assert cfg.INPUT.APPROX_3D_CENTER == 'intersect'

        # True
        assert cfg.INPUT.KEYPOINT_VISIBLE_MODIFY


        assert cfg.INPUT.ORIENTATION == 'multi-bin'
        assert cfg.INPUT.ORIENTATION_BIN_SIZE == 4

        # use '2D' or '3D' center for heatmap prediction
        assert cfg.INPUT.HEATMAP_CENTER == '3D'
        # infact not true


        assert cfg.INPUT.ADJUST_BOUNDARY_HEATMAP # True
        assert cfg.INPUT.HEATMAP_RATIO == 0.5 # radius / 2d box, 0.5

        self.logger = logging.getLogger("monoflex.simple dataset")

        assert cfg.DATASETS.KITTI_TYPE == 'original'
        assert cfg.MODEL.HEAD.DENSE_DEPTH
        assert cfg.MODEL.HEAD.DENSE_DEPTH_SAMPLE_NUM == 21
        assert cfg.INPUT.AUG_PARAMS == [[0.5]]