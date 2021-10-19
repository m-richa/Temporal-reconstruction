import torch
import torch.nn as nn
import os
import glob
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from base.base_dataloader import BaseDataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TemporalDataloader(BaseDataLoader):
    
    r"""Return the scheduler object.
    Args: 

    Returns:
        data = [rgb1_rgb2_scf, curr_dep, next_dep]
        rgb1_rgb2_scf1 = BxKx18xHxW (12+3+12)
        curr_dep = BxKx4xHxW
        next_dep = BxKx4xHxW
    """
    
    
    def __init__(self, dataset_path, seq_ID, K_frames, batch_size,
                 shuffle=False, validation_split = 0.0, 
                 num_workers = 1, training =True):
        
        self.dataset_path = dataset_path
        self.seq_ID = seq_ID
        self.K_frames = K_frames
        self.dataset = RGBDataset(self.dataset_path, self.seq_ID, self.K_frames)
        
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
        
        

class TemporalDataset(Dataset):
    
    def __init__(self, dataset_path, seq_ID, K_frames):
        """

        Args:
            folder_path (string): path to image folder 
            s.t. dataset_path/rgb/seq_ID/Frame00......
            
            seq_ID: seq_ID of video
        """
        # Get rgb images and scf list
        self.rgb_path = os.path.join(dataset_path, 'rgb', seq_ID)
        self.scf_path = os.path.join(dataset_path, 'scf', seq_ID)
        self.depth_path = os.path.join(dataset_path, 'depth', seq_ID)
        self.K_frames = K_frames
        frames_list = glob.glob(self.rgb_path+'/*')
        self.total_frames = len(frames_list)
        
        
        # Transformations for rgb and scf
        self.trsfm_rgb = transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        self.trsfm_scf = transforms.Normalize(mean=[-0.3]*12, std=[0.6]*12)
        

    def __getitem__(self, start_index):
        
        #K = no. of frames; start with 4
        assert start_index < (self.total_frames - self.K_frames), "Frame index not in range"
        
        rgb1_rgb2_scf = []
        curr_depth = []
        next_depth = []

        for i in range(start_index, start_index+self.K_frames):
            
        
            curr_rgb_path = os.path.join(self.rgb_path, 'Frame{num:03d}'.format(num=i))
            next_rgb_path = os.path.join(self.rgb_path, 'Frame{num:03d}'.format(num =i+1))
    
            curr_rgb1_img = io.imread(os.path.join(curr_rgb_path, 'rgb_01.png'))
            next_rgb1_img = io.imread(os.path.join(next_rgb_path, 'rgb_01.png'))

            curr_rgb1 = (torch.from_numpy(np.array(curr_rgb1_img))).float().permute(2,0,1) #3,H,W
            curr_rgb1 = self.trsfm_rgb(curr_rgb1)
            next_rgb1 = (torch.from_numpy(np.array(next_rgb1_img))).float().permute(2,0,1) #3,H,W
            next_rgb1 = self.trsfm_rgb(next_rgb1)
            
            """depth"""
            
            curr_depth_path = os.path.join(self.depth_path, 'Frame{num:03d}'.format(num=i))
            next_depth_path = os.path.join(self.depth_path, 'Frame{num:03d}'.format(num =i+1))
            
            curr_dep_arr1 = np.load(os.path.join(curr_depth_path, 'dep_1.npz'))
            curr_dep_arr2 = np.load(os.path.join(curr_depth_path, 'dep_2.npz'))
            curr_dep_arr3 = np.load(os.path.join(curr_depth_path, 'dep_3.npz'))
            curr_dep_arr4 = np.load(os.path.join(curr_depth_path, 'dep_4.npz'))
            curr_dep = np.stack([curr_dep_arr1.f.a, curr_dep_arr2.f.a,
                                 curr_dep_arr3.f.a, curr_dep_arr4.f.a]) #4xWxH
            curr_dep = torch.Tensor(curr_dep)
            
            
            next_dep_arr1 = np.load(os.path.join(next_depth_path, 'dep_1.npz'))
            next_dep_arr2 = np.load(os.path.join(next_depth_path, 'dep_2.npz'))
            next_dep_arr3 = np.load(os.path.join(next_depth_path, 'dep_3.npz'))
            next_dep_arr4 = np.load(os.path.join(next_depth_path, 'dep_4.npz'))
            next_dep = np.stack([next_dep_arr1.f.a, next_dep_arr2.f.a,
                                 next_dep_arr3.f.a, next_dep_arr4.f.a]) #4xWxH
            next_dep = torch.Tensor(next_dep)
            
            
            """scf"""
            
            scf_path = os.path.join(self.scf_path, 'Frame{num:03d}'.format(num=i))
            
            scf_arr1 = np.load(os.path.join(scf_path, 'scf_1.npz'))
            scf_arr2 = np.load(os.path.join(scf_path, 'scf_2.npz'))
            scf_arr3 = np.load(os.path.join(scf_path, 'scf_3.npz'))
            scf_arr4 = np.load(os.path.join(scf_path, 'scf_4.npz'))
            scf = np.concatenate([scf_arr1.f.a, scf_arr2.f.a, scf_arr3.f.a, scf_arr4.f.a], axis=2) # H,W,12
            
            scf = torch.Tensor(scf).permute(2,1,0) #12,H,W
            #print('max: '+str(torch.max(scf))+' min: '+str(torch.min(scf)))
            scf = self.trsfm_scf(scf)
            #print('max_after: '+str(torch.max(scf))+' min_after: '+str(torch.min(scf)))
            output_frame = torch.cat([curr_rgb1, next_rgb1, scf], dim=0) # 18xBxW
            rgb1_rgb2_scf.append(output_frame)
            curr_depth.append(curr_dep)
            next_depth.append(next_dep)
            
            
        rgb1_rgb2_scf = torch.stack(rgb1_rgb2_scf)
        curr_depth = torch.stack(curr_depth)
        next_depth = torch.stack(next_depth)
        
        return rgb1_rgb2_scf, curr_depth, next_depth # Kx18xWxH, Kx4xWxH, Kx4xWxH

    def __len__(self):
        
        return self.total_frames - self.K_frames
    
    
    
class RGBDataset(Dataset):
    
    def __init__(self, dataset_path, seq_ID, K_frames):
        """

        Args:
            folder_path (string): path to image folder 
            s.t. dataset_path/rgb/seq_ID/Frame00......
            
            seq_ID: seq_ID of video
        """
        # Get rgb images and scf list
        self.rgb_path = os.path.join(dataset_path, 'rgb', seq_ID)
        self.depth_path = os.path.join(dataset_path, 'depth', seq_ID)
        self.scf_path = os.path.join(dataset_path, 'scf', seq_ID)
        self.K_frames = K_frames
        frames_list = glob.glob(self.rgb_path+'/*')
        self.total_frames = len(frames_list)
        
        
        # Transformations for rgb and scf
        self.trsfm_rgb = transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        self.trsfm_scf = transforms.Normalize(mean=[-0.3]*12, std=[0.6]*12)
        

    def __getitem__(self, start_index):
        
        #K = no. of frames; start with 4
        assert start_index < (self.total_frames - self.K_frames), "Frame index not in range"
        
        rgb1_rgb2_scf = []
        curr_depth = []
        next_rgb = []

        for i in range(start_index, start_index+self.K_frames):
            
        
            curr_rgb_path = os.path.join(self.rgb_path, 'Frame{num:03d}'.format(num=i))
            next_rgb_path = os.path.join(self.rgb_path, 'Frame{num:03d}'.format(num =i+1))
    
            curr_rgb1_img = io.imread(os.path.join(curr_rgb_path, 'rgb_01.png'))
            curr_rgb2_img = io.imread(os.path.join(curr_rgb_path, 'rgb_02.png'))
            curr_rgb3_img = io.imread(os.path.join(curr_rgb_path, 'rgb_03.png'))
            curr_rgb4_img = io.imread(os.path.join(curr_rgb_path, 'rgb_04.png'))
            next_rgb1_img = io.imread(os.path.join(next_rgb_path, 'rgb_01.png'))
            next_rgb2_img = io.imread(os.path.join(next_rgb_path, 'rgb_02.png'))
            next_rgb3_img = io.imread(os.path.join(next_rgb_path, 'rgb_03.png'))
            next_rgb4_img = io.imread(os.path.join(next_rgb_path, 'rgb_04.png'))

            curr_rgb1 = (torch.from_numpy(np.array(curr_rgb1_img))).float().permute(2,0,1) #3,H,W
            curr_rgb2 = (torch.from_numpy(np.array(curr_rgb2_img))).float().permute(2,0,1)
            curr_rgb3 = (torch.from_numpy(np.array(curr_rgb3_img))).float().permute(2,0,1)
            curr_rgb4 = (torch.from_numpy(np.array(curr_rgb4_img))).float().permute(2,0,1)
            curr_rgb1 = self.trsfm_rgb(curr_rgb1)
            curr_rgb2 = self.trsfm_rgb(curr_rgb2)
            curr_rgb3 = self.trsfm_rgb(curr_rgb3)
            curr_rgb4 = self.trsfm_rgb(curr_rgb4)
            
            next_rgb1 = (torch.from_numpy(np.array(next_rgb1_img))).float().permute(2,0,1) #3,H,W
            next_rgb2 = (torch.from_numpy(np.array(next_rgb2_img))).float().permute(2,0,1)
            next_rgb3 = (torch.from_numpy(np.array(next_rgb3_img))).float().permute(2,0,1)
            next_rgb4 = (torch.from_numpy(np.array(next_rgb4_img))).float().permute(2,0,1)
            next_rgb1 = self.trsfm_rgb(next_rgb1)
            next_rgb2 = self.trsfm_rgb(next_rgb2)
            next_rgb3 = self.trsfm_rgb(next_rgb3)
            next_rgb4 = self.trsfm_rgb(next_rgb4)

            
            """depth"""
            curr_depth_path = os.path.join(self.depth_path, 'Frame{num:03d}'.format(num=i))
            
            curr_dep_arr1 = np.load(os.path.join(curr_depth_path, 'dep_1.npz'))
            curr_dep_arr2 = np.load(os.path.join(curr_depth_path, 'dep_2.npz'))
            curr_dep_arr3 = np.load(os.path.join(curr_depth_path, 'dep_3.npz'))
            curr_dep_arr4 = np.load(os.path.join(curr_depth_path, 'dep_4.npz'))
            curr_dep = np.stack([curr_dep_arr1.f.a, curr_dep_arr2.f.a,
                                 curr_dep_arr3.f.a, curr_dep_arr4.f.a]) #4xWxH
            curr_dep = torch.Tensor(curr_dep)
            
            
            """scf"""
            
            scf_path = os.path.join(self.scf_path, 'Frame{num:03d}'.format(num=i))
            
            scf_arr1 = np.load(os.path.join(scf_path, 'scf_1.npz'))
            scf_arr2 = np.load(os.path.join(scf_path, 'scf_2.npz'))
            scf_arr3 = np.load(os.path.join(scf_path, 'scf_3.npz'))
            scf_arr4 = np.load(os.path.join(scf_path, 'scf_4.npz'))
            scf = np.concatenate([scf_arr1.f.a, scf_arr2.f.a, scf_arr3.f.a, scf_arr4.f.a], axis=2) # H,W,12
            
            scf = torch.Tensor(scf).permute(2,1,0) #12,H,W
            #print('max: '+str(torch.max(scf))+' min: '+str(torch.min(scf)))
            scf = self.trsfm_scf(scf)
            #print('max_after: '+str(torch.max(scf))+' min_after: '+str(torch.min(scf)))
            output_frame = torch.cat([curr_rgb1, curr_rgb2, curr_rgb3, curr_rgb4,
                                      next_rgb1, scf], dim=0) # 27xBxW
            
            next_rgb.append(torch.cat([next_rgb1, next_rgb2, next_rgb3, next_rgb4], dim=0))

            rgb1_rgb2_scf.append(output_frame)
            curr_depth.append(curr_dep)
            
            
        rgb1_rgb2_scf = torch.stack(rgb1_rgb2_scf)
        next_rgb = torch.stack(next_rgb) #B,12,H,W
        curr_depth = torch.stack(curr_depth)# B,4,H,W
        
        return rgb1_rgb2_scf, next_rgb, curr_depth  # Kx27xWxH, Kx4xWxH

    def __len__(self):
        
        return self.total_frames - self.K_frames
