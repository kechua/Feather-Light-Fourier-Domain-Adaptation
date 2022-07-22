import os
from os.path import isdir
from typing import Optional, Type, Callable, Dict

from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import albumentations
import torch
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Subset
from parameters.path import Paths

from albumentations.pytorch.transforms import ToTensorV2 as AlbToTensor, ToTensorV2

from dataset.DA_dataloader import CustomPictDataset


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class AbstractLoader:
    pass


# dataset = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/BackUp')
# dataset.domain_preproc('/raid/data/DA_BrainDataset/old_data/philips_15', 'philips_15')

# DL_DS = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)


class DALoader(AbstractLoader):
    batch_size = 8
    test_batch_size = 8
    image_size = 256

    # transforms = transforms.Compose([transforms.Resize([image_size, image_size]),
    #                                              transforms.PILToTensor(),
    #                                              ThresholdTransform(thr_255=240)])

    def __init__(self, path="philips_15"):
        print("/raid/data/DA_BrainDataset/BackUp")
        print(f'/raid/data/DA_BrainDataset/{path}', f'{path}')
        self.dataset = CustomPictDataset(None, None, load_dir='/raid/data/DA_BrainDataset/BackUp')
        self.dataset.domain_preproc(f'/raid/data/DA_BrainDataset/{path}', f'{path}')
        N = self.dataset.__len__()

        self.dataset_train = Subset(self.dataset, range(int(N * 0.8)))

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=DALoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=10
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.dataset_test = Subset(self.dataset, range(int(N * 0.8), N))

        self.test_loader = data.DataLoader(
            self.dataset_test,
            batch_size=DALoader.test_batch_size,
            drop_last=False,
            num_workers=10
        )

        print("DA initialize")
        print(f'train size: {len(self.dataset_train)}, test size: {len(self.dataset_test)}')


class Celeba:

    image_size = 256
    batch_size = 8

    transform = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(image_size, image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=50, alpha_affine=1, sigma=10),
            # albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=(-0.1, 0.3)),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
    ])

    def __init__(self):
        print("init calaba")

        dataset = ImageDataset(
            f"{Paths.default.data()}/celeba",
            img_transform=Celeba.transform
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=Celeba.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=Celeba.batch_size
        )

        print("batch size: ", Celeba.batch_size)

        self.loader = sample_data(self.loader)


class Metfaces:

    image_size = 256
    batch_size = 8

    transform = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(image_size, image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=50, alpha_affine=1, sigma=10),
            # albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=(-0.1, 0.3)),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
    ])

    def __init__(self):
        print("init Metfaces")

        dataset = ImageDataset(
            f"{Paths.default.data()}/metfaces/",
            img_transform=Metfaces.transform
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=Metfaces.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=Metfaces.batch_size
        )

        print("batch size: ", Metfaces.batch_size)

        self.loader = sample_data(self.loader)


class LazyLoader:
    saved = {}

    domain_adaptation_philips15_save: Optional[DALoader] = None
    domain_adaptation_siemens15_save: Optional[DALoader] = None
    domain_adaptation_siemens3_save: Optional[DALoader] = None
    domain_adaptation_ge3_save: Optional[DALoader] = None
    domain_adaptation_ge15_save: Optional[DALoader] = None
    domain_adaptation_philips3_save: Optional[DALoader] = None
    celeba_save: Optional[Celeba] = None
    metfaces_save: Optional[Metfaces] = None

    @staticmethod
    def register_loader(cls: Type[AbstractLoader]):
        LazyLoader.saved[cls.__name__] = None

    @staticmethod
    def domain_adaptation_philips15() -> DALoader:
        if not LazyLoader.domain_adaptation_philips15_save:
            LazyLoader.domain_adaptation_philips15_save = DALoader(path="philips15")
        return LazyLoader.domain_adaptation_philips15_save

    @staticmethod
    def domain_adaptation_philips3() -> DALoader:
        if not LazyLoader.domain_adaptation_philips3_save:
            LazyLoader.domain_adaptation_philips3_save = DALoader(path="philips3")
        return LazyLoader.domain_adaptation_philips3_save

    @staticmethod
    def domain_adaptation_siemens15() -> DALoader:
        if not LazyLoader.domain_adaptation_siemens15_save:
            LazyLoader.domain_adaptation_siemens15_save = DALoader(path="siemens15")
        return LazyLoader.domain_adaptation_siemens15_save

    @staticmethod
    def domain_adaptation_siemens3() -> DALoader:
        if not LazyLoader.domain_adaptation_siemens3_save:
            LazyLoader.domain_adaptation_siemens3_save = DALoader(path="siemens3")
        return LazyLoader.domain_adaptation_siemens3_save

    @staticmethod
    def domain_adaptation_ge3() -> DALoader:
        if not LazyLoader.domain_adaptation_ge3_save:
            LazyLoader.domain_adaptation_ge3_save = DALoader(path="ge3")
        return LazyLoader.domain_adaptation_ge3_save

    @staticmethod
    def domain_adaptation_ge15() -> DALoader:
        if not LazyLoader.domain_adaptation_ge15_save:
            LazyLoader.domain_adaptation_ge15_save = DALoader(path="ge15")
        return LazyLoader.domain_adaptation_ge15_save

    @staticmethod
    def celeba():
        if not LazyLoader.celeba_save:
            LazyLoader.celeba_save = Celeba()
        return LazyLoader.celeba_save

    @staticmethod
    def metfaces():
        if not LazyLoader.metfaces_save:
            LazyLoader.metfaces_save = Metfaces()
        return LazyLoader.metfaces_save


class ImageDataset(Dataset):
    def __init__(self,
                 images_path,
                 img_transform):

        image_folders = [x for x in os.listdir(images_path)]

        self.imgs = []

        for folder in image_folders:
            if not isdir(os.path.join(images_path, folder)):
                continue
            for img in os.listdir(os.path.join(images_path, folder)):
                img_path = os.path.join(images_path, folder, img)
                self.imgs += [img_path]

        self.img_transform = img_transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.imgs[index]).convert('RGB'))

        dict_transfors = self.img_transform(image=image)
        image = dict_transfors['image']

        return image

    def __len__(self):
        return len(self.imgs)


