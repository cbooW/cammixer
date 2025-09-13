"""
Dataset for handling int16 NPY files.
File should be saved as: basicsr/data/paired_int16_dataset.py
The filename must end with '_dataset.py' for auto-registration in BasicSR.
"""

import io
import os
import glob
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY


def npy2tensor(npy_array, float32=True, normalize_max=8000.0):
    """Convert numpy array to tensor.
    
    Args:
        npy_array: Input numpy array (H, W) or (H, W, C)
        float32: Whether convert to float32
        normalize_max: Maximum value for normalization
    
    Returns:
        Tensor: (C, H, W) tensor, normalized to [0, 1]
    """
    if npy_array.ndim == 2:
        npy_array = np.expand_dims(npy_array, axis=2)
    
    # Normalize to [0, 1]
    if float32:
        npy_array = npy_array.astype(np.float32) / normalize_max
    
    # HWC to CHW
    tensor = torch.from_numpy(npy_array.transpose(2, 0, 1))
    
    return tensor


def paired_paths_from_folder_npy(folders, keys, filename_tmpl='{}.npy'):
    """Generate paired paths from folders with npy extension."""
    assert len(folders) == 2, f'Need 2 folders, but got {len(folders)}.'
    assert len(keys) == 2, f'Need 2 keys, but got {len(keys)}.'
    
    paths = []
    lq_folder, gt_folder = folders
    
    # Get all npy files
    lq_files = sorted(glob.glob(os.path.join(lq_folder, '*.npy')))
    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.npy')))
    
    # Match files by name
    lq_names = {os.path.splitext(os.path.basename(f))[0]: f for f in lq_files}
    gt_names = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    
    # Find common files
    common_names = set(lq_names.keys()) & set(gt_names.keys())
    
    for name in sorted(common_names):
        paths.append({
            f'{keys[0]}_path': lq_names[name],
            f'{keys[1]}_path': gt_names[name]
        })
    
    return paths


@DATASET_REGISTRY.register()
class PairedImageDatasetInt16(data.Dataset):
    """Paired image dataset for int16 NPY files.

    Read LQ and GT image pairs from NPY files.
    
    Args:
        opt (dict): Config for train datasets. Contains:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            normalize_max (float): Maximum value for normalization. Default: 8000.0
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (int): Scale factor.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetInt16, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        # Get normalization maximum value
        self.normalize_max = float(opt.get('normalize_max', 8000.0))
        
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        
        # Get paired paths
        self.paths = paired_paths_from_folder_npy(
            [self.lq_folder, self.gt_folder],
            ['lq', 'gt'],
            opt.get('filename_tmpl', '{}.npy')
        )
        
        if len(self.paths) == 0:
            raise ValueError(f'No paired npy files found in {self.lq_folder} and {self.gt_folder}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), 
                **self.io_backend_opt
            )

        scale = self.opt.get('scale', 1)
        
        # Load gt and lq npy files
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        
        # Read npy files directly from disk
        img_gt = np.load(gt_path)
        img_lq = np.load(lq_path)
        
        # Ensure 3D array (H, W, C) for compatibility with augmentation
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
        
        # Convert to float32 if needed
        if img_gt.dtype != np.float32:
            img_gt = img_gt.astype(np.float32)
        if img_lq.dtype != np.float32:
            img_lq = img_lq.astype(np.float32)
        
        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size', 256)
            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Flip and rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], 
                self.opt.get('use_hflip', True),
                self.opt.get('use_rot', True)
            )

        # Convert to tensor and normalize to [0, 1]
        img_gt = npy2tensor(img_gt, float32=True, normalize_max=self.normalize_max)
        img_lq = npy2tensor(img_lq, float32=True, normalize_max=self.normalize_max)

        # Additional normalization if specified
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq, 
            'gt': img_gt, 
            'lq_path': lq_path, 
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class SingleImageDatasetInt16(data.Dataset):
    """Single image dataset for int16 NPY files (for inference).

    Args:
        opt (dict): Config for test datasets. Contains:
            dataroot_lq (str): Data root path for lq.
            normalize_max (float): Maximum value for normalization. Default: 8000.0
    """

    def __init__(self, opt):
        super(SingleImageDatasetInt16, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.lq_folder = opt['dataroot_lq']
        
        # Get normalization maximum value
        self.normalize_max = float(opt.get('normalize_max', 8000.0))
        
        # Get all npy files
        self.paths = sorted(glob.glob(os.path.join(self.lq_folder, '*.npy')))
        
        if len(self.paths) == 0:
            raise ValueError(f'No npy files found in {self.lq_folder}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        # Load lq npy file
        lq_path = self.paths[index]
        
        # Read npy file
        img_lq = np.load(lq_path)
        
        # Ensure 3D array
        if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=2)
            
        # Convert to float32 if needed
        if img_lq.dtype != np.float32:
            img_lq = img_lq.astype(np.float32)

        # Convert to tensor and normalize to [0, 1]
        img_lq = npy2tensor(img_lq, float32=True, normalize_max=self.normalize_max)

        # Additional normalization if specified
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.paths)