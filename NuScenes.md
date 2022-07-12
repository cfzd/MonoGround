We provide the experiments on the NuScenes dataset. The results are as follows:
| Camera      |   mAP  |  mATE | mASE  | mAOE  | mAVE  | mAAE | NDS    |
|-------------|:------:|:-----:|-------|-------|-------|------|--------|
| front       |  1.670 | 0.882 | 0.699 | 0.720 | 1.506 | 1.0  | 7.814  |
| front_left  | 0.066  | 1.0   | 1.0   | 1.0   | 1.0   | 1.0  | 0.033  |
| front_right | 0.138  | 0.956 | 0.934 | 1.0   | 1.0   | 1.0  | 1.163  |
| back        | 2.683  | 0.835 | 0.701 | 0.721 | 1.300 | 1.0  | 8.767  |
| back_left   | 0.040  | 1.0   | 1.0   | 1.0   | 1.0   | 1.0  | 0.020  |
| back_right  | 0.138  | 1.0   | 1.0   | 1.0   | 1.0   | 1.0  | 0.069  |
| all         | 18.220 | 0.717 | 0.414 | 0.826 | 1.452 | 1.0  | 19.540 |

To run our code on NuScenes, one can run
```Shell
python scripts/nuscenes2kitti.py
```
to convert to NuScenes dataset to KITTI format. Then change the path of KITTI to NuScenes and kick off the training.