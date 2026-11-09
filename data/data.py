import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        root = getattr(self, "data_root", None) or "./data/datasets"
        download = not getattr(self, "offline", False)
        train_dataset = datasets.cifar.CIFAR100(root, train=True, download=download)
        test_dataset = datasets.cifar.CIFAR100(root, train=False, download=download)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )




class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self):
        root = getattr(self, "data_root", None) or "./data/datasets/imagenet-r"
        train_dset = datasets.ImageFolder(os.path.join(root, "train"))
        test_dset = datasets.ImageFolder(os.path.join(root, "test"))
        if len(train_dset.classes) != 200 or train_dset.classes != test_dset.classes:
            raise ValueError("ImageNet-R requires the same 200 class folders in train and test")
        self.class_names = list(train_dset.classes)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root = getattr(self, "data_root", None) or "./data/datasets/imagenet-a"
        train_dir = os.path.join(root, "train")
        test_dir = os.path.join(root, "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        train_classes = list(train_dset.classes)
        test_classes = list(test_dset.classes)
        if len(train_classes) != 200 or len(test_classes) != 200:
            raise ValueError(
                "ImageNet-A protocol requires exactly 200 train and 200 test classes"
            )
        if train_classes != test_classes:
            raise ValueError(
                "ImageNet-A train/test class folders differ; their label maps are unsafe"
            )
        self.class_names = train_classes

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        root = getattr(self, "data_root", None) or "./data/datasets/cub"
        train_dset = datasets.ImageFolder(os.path.join(root, "train"))
        test_dset = datasets.ImageFolder(os.path.join(root, "test"))
        if len(train_dset.classes) != 200 or train_dset.classes != test_dset.classes:
            raise ValueError("CUB-200 requires the same 200 class folders in train and test")
        self.class_names = list(train_dset.classes)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


def prepare_cub_split(source, dest):
    """Build <dest>/{train,test}/<class>/ from a raw CUB_200_2011 folder with the official split.

    ``source`` holds images/, images.txt and train_test_split.txt (5 994 train / 5 794 test).
    Files are symlinked (copied if symlinks are unavailable); returns the image counts.
    """
    ids = dict(line.split() for line in open(os.path.join(source, "images.txt"), encoding="utf-8")
               if line.strip())
    counts = {"train": 0, "test": 0}
    for line in open(os.path.join(source, "train_test_split.txt"), encoding="utf-8"):
        if not line.strip():
            continue
        image_id, is_train = line.split()
        relative = ids[image_id]
        split = "train" if is_train == "1" else "test"
        target = os.path.join(dest, split, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.exists(target):
            original = os.path.abspath(os.path.join(source, "images", relative))
            try:
                os.symlink(original, target)
            except OSError:
                import shutil
                shutil.copy2(original, target)
        counts[split] += 1
    return counts



class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/datasets/vtab/train/"
        test_dir = "./data/datasets/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/datasets/omnibenchmark/train/"
        test_dir = "./data/datasets/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
