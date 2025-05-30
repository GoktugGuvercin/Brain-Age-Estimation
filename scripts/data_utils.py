import os
from tqdm import tqdm
from functools import partial
from typing import List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

DATA_DIR = './data/brain_age'


def get_image_dataloaders(
    img_size: int,
    batch_size: int,
    num_workers: int = 0
):
    print('Loading data. This might take a while...')
    train_ds = BrainAgeImageDataset('train', img_size)
    val_ds = BrainAgeImageDataset('val', img_size)
    test_ds = BrainAgeImageDataset('test', img_size)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False,
                         num_workers=num_workers)

    return {'train': train_dl, 'val': val_dl, 'test': test_dl}


class BrainAgeImageDataset(Dataset):
    def __init__(self, mode: str, img_size: int):
        assert mode in ['train', 'val', 'test']
        print(f'Loading {mode} data...')

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train.csv'))
        elif mode == 'val':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_segmentation_train.csv'))
        if mode == 'test':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_test.csv'))

        self.ages = self.df["age"].tolist()
        self.img_ids = self.df['subject_id'].tolist()
        self.load_fn = partial(load_and_preprocess, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a brain MR-image and it's corresponding age

        :param idx: Sample index of the dataset
        :return img: Loaded brain MRI. Shape (C, H, W, D)
        """

        img = self.load_fn(self.img_ids[idx])
        img = np.expand_dims(img, axis=0)
        age = np.array([self.ages[idx]])

        img = torch.from_numpy(img)
        age = torch.from_numpy(age).float()
        return img, age


def normalize(img: np.ndarray, mask: np.ndarray):
    """
    * Instance based normalization

    Normalize a brain MR-image to have zero mean and unit variance.
    For getting the normalization parameters, only the brain pixels should be
    used. All background pixels should be 0 in the end.

    :param img: Brain MR-image. Shape (H, W, D)
    :param mask: Mask to indicate which voxels correspond to the brain (1s) and
                 which are background (0s). Shape (H, W, D)
    :return normalized_img: Brain MR-image normalized to zero mean and unit
                            variance. Shape (H, W, D)
    """

    normalized_img = img.copy()
    non_background = normalized_img[mask != 0]
    normalized_img[mask != 0] = (
        non_background - non_background.mean()
    ) / non_background.std()

    return normalized_img


def load_nii(path: str, dtype: str = 'float32') -> np.ndarray:
    """Load an MRI scan from disk and convert it to a given datatype

    :param path: Path to file
    :param dtype: Target dtype
    :return img: Loaded image. Shape (H, W, D)
    """
    return nib.load(path).get_fdata().astype(np.dtype(dtype))


def preprocess(img: np.ndarray, mask: np.ndarray, img_size: int) -> np.ndarray:
    """Preprocess an MRI.

    :param img: MR-image. Shape (H, W, D)
    :param mask: Brain mask. Shape (H, W, D)
    :param img_size: Target size, a scalar.
    :return preprocessed_img: The preprocessed image.
                              Shape (img_size, img_size, img_size)
    """

    preprocessed_img = normalize(img, mask)
    if img_size != 0 and img_size != -1:
        preprocessed_img = resize(preprocessed_img, [img_size] * 3)
    return preprocessed_img


def load_and_preprocess(ID: str, img_size: int) -> np.ndarray:
    """Load an MRI from disk and preprocess it"""
    img = load_nii(os.path.join(DATA_DIR, f"images/sub-{ID}_T1w_unbiased.nii.gz"))
    mask = load_nii(
        os.path.join(DATA_DIR, f"masks/sub-{ID}_T1w_brain_mask.nii.gz"),
        dtype='int'
    )
    return preprocess(img, mask, img_size)


def prefetch_samples(IDs: List[str], img_size: int) -> np.ndarray:

    load_fn = partial(load_and_preprocess, img_size=img_size)
    res = [load_fn(ID) for ID in IDs]
    return np.array(res)[:, None]


def load_segmentations(paths: str):
    """Load all segmentations and associated subject_ids"""
    filenames, segmentations = [], []
    for im in tqdm(paths):
        id = im.split('_brain_')[0].split('/')[-1].split('-')[1].split('_')[0]
        segmentations.append(load_nii(im))
        filenames.append(id)
    return filenames, np.array(segmentations)
