import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from data.data import  iCIFAR224, iImageNetR,iImageNetA,CUB, vtab, omnibenchmark


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment,
                 data_root=None, offline=False):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.offline = offline
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets, sample_ids = [], [], []
        for idx in indices:
            class_ids = np.flatnonzero((y >= idx) & (y < idx + 1))
            if m_rate is not None and m_rate != 0:
                selected = np.random.randint(
                    0, len(class_ids), size=int((1 - m_rate) * len(class_ids))
                )
                class_ids = np.sort(class_ids[selected])
            class_data, class_targets = x[class_ids], y[class_ids]
            
            data.append(class_data)
            targets.append(class_targets)
            sample_ids.append(class_ids)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            sample_ids.append(np.arange(len(x), len(x) + len(appendent_data)))

        data, targets = np.concatenate(data), np.concatenate(targets)
        sample_ids = np.concatenate(sample_ids)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, sample_ids)
        else:
            return DummyDataset(data, targets, trsf, self.use_path, sample_ids)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_validation(self, indices, val_ratio, seed):
        """Return a deterministic stratified train/validation split.

        Training samples retain training augmentation.  Validation samples use
        the deterministic test transform so hyperparameter selection and QKSR
        calibration do not depend on random crops or flips.
        """
        val_ratio = float(val_ratio)
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1).")

        rng = np.random.default_rng(int(seed))
        train_data, train_targets = [], []
        val_data, val_targets = [], []
        train_ids, val_ids = [], []
        for idx in indices:
            class_ids = np.flatnonzero(self._train_targets == idx)
            class_data, class_targets = self._select(
                self._train_data, self._train_targets, low_range=idx, high_range=idx + 1
            )
            if len(class_data) < 2:
                raise ValueError(
                    "Class {} needs at least two samples for validation splitting.".format(idx)
                )
            val_count = int(round(len(class_data) * val_ratio))
            val_count = min(max(val_count, 1), len(class_data) - 1)
            permutation = rng.permutation(len(class_data))
            val_indices = permutation[:val_count]
            train_indices = permutation[val_count:]
            train_data.append(class_data[train_indices])
            train_targets.append(class_targets[train_indices])
            val_data.append(class_data[val_indices])
            val_targets.append(class_targets[val_indices])
            train_ids.append(class_ids[train_indices])
            val_ids.append(class_ids[val_indices])

        train_transform = transforms.Compose([*self._train_trsf, *self._common_trsf])
        val_transform = transforms.Compose([*self._test_trsf, *self._common_trsf])
        return (
            DummyDataset(
                np.concatenate(train_data),
                np.concatenate(train_targets),
                train_transform,
                self.use_path,
                np.concatenate(train_ids),
            ),
            DummyDataset(
                np.concatenate(val_data),
                np.concatenate(val_targets),
                val_transform,
                self.use_path,
                np.concatenate(val_ids),
            ),
        )

    def get_eval_view(self, dataset):
        """Create a deterministic-transform view over an existing dataset."""
        if not isinstance(dataset, DummyDataset):
            raise TypeError("get_eval_view expects a DummyDataset.")
        eval_transform = transforms.Compose([*self._test_trsf, *self._common_trsf])
        return DummyDataset(
            dataset.images, dataset.labels, eval_transform, dataset.use_path,
            dataset.sample_ids,
        )

    def get_seen_validation_dataset(self, task_sizes, val_ratio, seed):
        """Reconstruct validation for all seen classes without training on it."""
        validation_ids = []
        lower = 0
        for task, size in enumerate(task_sizes):
            _, validation = self.get_dataset_with_validation(
                np.arange(lower, lower + size), val_ratio, int(seed) + task
            )
            validation_ids.append(validation.sample_ids)
            lower += size
        ids = np.concatenate(validation_ids)
        transform = transforms.Compose([*self._test_trsf, *self._common_trsf])
        return DummyDataset(
            self._train_data[ids], self._train_targets[ids], transform,
            self.use_path, ids,
        )

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.data_root = getattr(self, "data_root", None)
        idata.offline = getattr(self, "offline", False)
        idata.download_data()
        # train_data和train_target和test_data和test_target

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, sample_ids=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.sample_ids = np.asarray(
            np.arange(len(images)) if sample_ids is None else sample_ids,
            dtype=np.int64,
        )
        if len(self.sample_ids) != len(images):
            raise ValueError("Sample ID count must match dataset size.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return int(self.sample_ids[idx]), image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name== "cifar224":
        return iCIFAR224()
    elif name== "imagenetr":
        return iImageNetR()
    elif name=="imageneta":
        return iImageNetA()
    elif name=="cub":
        return CUB()
    elif name=="vtab":
        return vtab()
    elif name == "omnibenchmark":
        return omnibenchmark()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
