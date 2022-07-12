from nuscenes_eval.my_kitti_db import MyKittiDB
from nuscenes.utils.data_classes import Box
from typing import List, Dict, Any
import os, json

def _box_to_sample_result(sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
    # Prepare data
    translation = box.center
    size = box.wlh
    rotation = box.orientation.q
    velocity = box.velocity
    detection_name = box.name
    detection_score = box.score

    # Create result dict
    sample_result = dict()
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation.tolist()
    sample_result['size'] = size.tolist()
    sample_result['rotation'] = rotation.tolist()
    sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
    sample_result['detection_name'] = detection_name
    sample_result['detection_score'] = detection_score
    sample_result['attribute_name'] = attribute_name

    return sample_result

def kitti_pred_to_nuscenes(kitti_root, kitti_pred_root, split, nusc = None):

    meta = {
        'use_camera': True,
        'use_lidar': False,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }

    # Init.
    results = {}

    # Load the KITTI dataset.
    kitti = MyKittiDB(root=kitti_root, splits=(split, ), pred_root = kitti_pred_root)

    for kitti_token in kitti.tokens:
        # Get the KITTI boxes we just generated in LIDAR frame.
        _, sample_token = kitti_token.split('_')

        sample = nusc.get('sample', sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
        cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        boxes = kitti.get_boxes(token=kitti_token, pose_record=pose_record, cs_record=cs_record)

        # Convert KITTI boxes to nuScenes detection challenge result format.
        sample_results = [_box_to_sample_result(sample_token, box) for box in boxes]

        # Store all results for this image.
        results[sample_token] = sample_results

    # Store submission file to disk.
    submission = {
        'meta': meta,
        'results': results
    }
    submission_path = os.path.join(kitti_pred_root, 'submission.json')
    print('Writing submission to: %s' % submission_path)
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)