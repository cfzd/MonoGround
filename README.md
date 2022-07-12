# MonoGround
Officail PyTorch implementation of the paper: "[MonoGround: Detecting Monocular 3D Objects from the Ground](https://arxiv.org/abs/2206.07372)".


## Installation
Please see [INSTALL.md](./INSTALL.md).

## Get started
To verify the results of the trained model, please run:
```Shell
python tools/plain_train_net.py --batch_size 8 --config runs/monoground.yaml --ckpt /path/to/model --eval --output ./tmp
```

To train the model by yourself, please run:
```Shell
python tools/plain_train_net.py --batch_size 8 --config runs/monoground.yaml --output ./tmp
```

## Model and log
We provide the trained model on KITTI and corresponding logs.

| Model | Log | AP easy | AP mod | AP hard |
|:-----:|:---:|:-------:|---------|---------|
|   [Google](https://drive.google.com/file/d/1Cp36t6E9m6_8P-oXW2aJLmxSeot5FIOW/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/12CI2J_GY0rDFUvkYaZcIlw?pwd=2chf)    |  [Google](https://drive.google.com/file/d/1EnyUaTarrUZbkPfmxrn5eIlfN2mYrWuw/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/14oZfH8QJdymNyCxRtV9WWw?pwd=az78)   |  25.24    |   18.69   |   15.58   |

## Exp on NuScenes
We also tested our method on the NuScenes dataset. Please see [NuScenes.md](./NuScenes.md) for details.

## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{qin2022monoground,
  title={MonoGround: Detecting Monocular 3D Objects From the Ground},
  author={Qin, Zequn and Li, Xi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3793--3802},
  year={2022}
}
```

## Acknowlegment

The code is heavily borrowed from [MonoFlex](https://github.com/zhangyp15/MonoFlex) and [SMOKE](https://github.com/lzccccc/SMOKE) and thanks for their contribution.
