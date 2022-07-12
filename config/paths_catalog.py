import os

class DatasetCatalog():
    DATA_DIR = "/path/to/your/kitti/"
    DATASETS = {
        "kitti_train": {
            "root": "training/",
        },
        "kitti_test": {
            "root": "testing/",
        },
    }

    @staticmethod
    def get(name): 
        if "kitti" in name or 'nuscenes' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        elif name == 'simple':
            return dict(
                factory="SimpleKITTIDataset",
                args=dict(root=attrs["root"]),
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
