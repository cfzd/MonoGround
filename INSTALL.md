## Install
```
git clone https://github.com/cfzd/MonoGround

cd MonoGround

conda create -n monoground python=3.7

conda activate monoground

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

cd model/backbone/DCNv2

sh make.sh

cd ../../..

python setup.py develop
```

## Data Preparation
Please make the KITTI dataset looks like:
```
#KITTI	
  |training/
    |calib/
    |image_2/
    |label/
    |ImageSets/
  |testing/
    |calib/
    |image_2/
    |ImageSets/
```
The `ImageSets` can be copied from the `ImageSets` folder in our code.

## Config your dataset path
Once you have finished data preparation, you have to set the `DATA_DIR = "/path/to/your/kitti/"` in the `config/paths_catalog.py` according to your environment.