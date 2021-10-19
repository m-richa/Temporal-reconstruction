from utils.utils import get_peeled_intersections, get_depth_peels
from utils.sceneflow import get_scf_peels

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import trimesh
import os

def get_depth_peels_next(root, path_curr, path_next):
    
    intersections, ray_param, ray_intersect_param, scene = get_peeled_intersections(root, path_curr)
    
    depth_curr, _ = get_depth_peels(root, path_curr, intersections, ray_param, ray_intersect_param, scene)
    
    scf, K = get_scf_peels(root, path_curr, path_next, intersections, ray_param, ray_intersect_param, scene)
    
    name = path_curr.split('/')[0]
    print(name)
    os.makedirs(root+'/PeeledMaps/'+name, exist_ok=True)
    mesh_folder = root+'/PeeledMaps/'+name
    """
    Project the depth from depth_peel_maps to 3D add scf and project back
    """
    for i in range(4):
    
        mask = torch.where(torch.tensor(depth_curr[i])>0)
        y, x = mask
        K = torch.tensor(K).float()
        im_points = torch.stack((x, y, torch.ones(len(x))), 0)
        #print(im_points.shape)
        world_points = (torch.linalg.inv(K) @ (im_points * depth_curr[i][mask])).cpu().numpy()
        world_points_next = (world_points.T + scf[i][mask]).T
        
        im_points_next = K @ world_points_next
        
        #Use grid-interpolation for assigning im_points_next to pixels
        
        gridx, gridy = torch.meshgrid(torch.arange(0, depth_curr[0].shape[1]),
                              torch.arange(0, depth_curr[0].shape[0]))
        depth_next_intp = griddata((im_points_next[:2, :] / im_points_next[2, :]).T,
                   im_points_next[2, :],
                   (gridy, gridx), method='nearest', fill_value = 0)
        
        
        
        plt.imsave(mesh_folder + '/' + 'dep_next_{}.jpg'.format(i), dep1n_intp, cmap='gray')
        np.savez_compressed(mesh_folder + '/' + 'dep_next_{}'.format(i), a = depth_next_intp)
        
        
        
    
