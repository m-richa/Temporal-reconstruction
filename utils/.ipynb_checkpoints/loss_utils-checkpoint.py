import trimesh
import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import os
from torch.nn.functional import grid_sample

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


T = torch.Tensor([[0.0, 0.0, 0.0]])#.to(verts.device)
R = torch.Tensor([[[-6.1232e-17,  1.0000e+00, -7.4988e-33],
                   [ 1.0000e+00,  6.1232e-17,  1.2246e-16],
                   [ 1.2246e-16,  0.0000e+00, -1.0000e+00]]])#.to(verts.device)


K = np.array([[443.40500674,   0.        , 256.        ],
              [  0.        , 443.40500674, 256.        ],
              [  0.        ,   0.        ,   1.        ]])

K = torch.tensor(K).float()

rotate_1 = trimesh.transformations.rotation_matrix(angle=np.radians(90), direction=[0, 0, -1])
rotate_2 = trimesh.transformations.rotation_matrix(angle=np.radians(180), direction=[0,-1,0])
EXTRINSIC = torch.tensor(rotate_1[:3, :3] @ rotate_2[:3, :3]).float()

#if torch.cuda.is_available():
#    EXTRINSIC = EXTRINSIC.cuda()
#    K = K.cuda()


        
def gridsample(input1, input2):

    return grid_sample(input1, input2, mode='bilinear', padding_mode='border')

        
def warp_rgb(scf, next_rgb, depth_curr, K=K, extrinsic=EXTRINSIC):
    """
    Args:
        scf: B,3,H,W
        next_rgb1: B,3,H,W
        depth_curr: B,H,W
        
    Returns:
        output: B,3,H,W
    """
    
    B, _, H, W = scf.shape
    K = torch.tensor(K, device = scf.device).float()
    extrinsic = torch.tensor(extrinsic, device = scf.device).float()
    
    output_rgb = []
    flow_grid = []
        
    world_points_next = project_peels_pred(scf, depth_curr, is_mask=False) #B,3,512*512
    
    for b in range(B):
        im_points = K @ extrinsic @ world_points_next[b,:,:] #3,512*512
        u, v = im_points[:2, :]/ (im_points[2, :] + 1e-8)
        u = (u.reshape(1, H, W))
        v = (v.reshape(1, H, W))
    
        flow = torch.cat([u, v], dim=0)
        flow = (2*(flow)-511)/511
        flow = (flow).permute(1, 2, 0) #512,512,2
        flow_grid.append(flow)
    
    flow_grid = torch.stack(flow_grid) #B,512,512,2
    
    output = gridsample(next_rgb, flow_grid) #B,3,H,W
    print(output.shape)
    return output
    
    
def depth_predicted(depth_curr, depth_next, scf, K= K, EXTRINSIC=EXTRINSIC):
    """
    Args: 
        depth_curr = 4xHxW tensor depth gt frame_curr peels
        depth_next = 4xHxW tensor depth gt frame_next peels
        extrinsic_mat = extrinsic matrix
        K = intrinsic matrix
    
    Returns:
        frame_next predicted depth peel maps 4xHxW
        
    
    """
    scf = scf.permute(1,2,0)
    pred = []
    target = []
    peel_loss = []
    
    for i in range(4):
        ####Check the interpolation without mask_next#####
        #mask_curr = torch.where(depth_curr[i]>0)
        #mask_next = torch.where(depth_next[i]>0)
        

        world_points_curr = projection(depth_curr[i], EXTRINSIC, K)
        world_points_next_pred = (world_points_curr.T+ scf[:,:, i*3:i*3 + 3].reshape(-1, 3)).T
        
        im_points_next = K @ EXTRINSIC @ world_points_next_pred
        
        mask_next = torch.zeros_like(depth_next[i])
        mask_next[torch.where(depth_next[i] > 0)] = 1
        depth_next_pred = mask_next * (-im_points_next[2,:].reshape(512, 512))
        
        l1_loss = 100*torch.abs(depth_next_pred - depth_next)
        peel_loss.append(l1_loss)
        #pred.append(world_points_next_pred)
        #target.append(world_points_next)
        
    #pred = torch.stack(pred)
    #target =torch.stack(target)
    peel_loss = torch.stack(peel_loss)
    
    return peel_loss #pred, target
    
    
def backproject(depth, extrinsic=EXTRINSIC, K=K, is_mask=True):
    """
    Args: 
        depth_curr = HxW tensor depth gt frame_curr peels
        extrinsic_mat = extrinsic matrix
        K = intrinsic matrix
    
    Returns:
        3D world_points [X, Y, Z] 3xN   
    """
    K = torch.tensor(K, device = depth.device).float()
    extrinsic = torch.tensor(extrinsic, device = depth.device).float()

    if is_mask:
    
        mask = torch.where(depth > 0)
        y, x = mask
        im_points = torch.stack((x, y, torch.ones(len(x), device=depth.device)), 0)
        world_points = (torch.linalg.inv(extrinsic) @ torch.linalg.inv(K) @ (im_points * depth[mask]))
        #3xN
        return world_points, mask
    
    else:
        
        x, y = torch.meshgrid(torch.arange(0, depth.shape[1]), 
                              torch.arange(0, depth.shape[0]))

        x = x.reshape(-1)
        y = y.reshape(-1)
        ones = torch.ones(len(x), device=x.device)
        im_points = torch.stack((y, x, ones), 0).to(device=depth.device).float()
        
        world_points = ((torch.linalg.inv(extrinsic) @ torch.linalg.inv(K) @ im_points) * depth.reshape(-1))
        return world_points #3xN

    
    
def project_fullbody(depth_peels):
    
    world_points = []
    
    for i in range(4):
        
        peel_projection, _ = backproject(depth_peels[i])
        
        world_points.extend([*peel_projection.T])
        
    return torch.stack(world_points)

def project_fullbody_pred(scf, depth_curr_gt, extrinsic=EXTRINSIC, K=K, is_mask=True):
    
    world_points_next_pred = []
    scf = scf.permute(1,2,0) #HxWx12
    
    for i in range(4):
        
        if is_mask:
            peel_curr, mask = backproject(depth_curr_gt[i], is_mask=is_mask)
            peel_next_pred = peel_curr.T + scf[:,:,3*i:3*i+3][mask]
            
        else:
            peel_curr = backproject(depth_curr_gt[i], is_mask=is_mask) #3x262144
            peel_next_pred = peel_curr.T + scf[:,:,3*i:3*i+3].reshape(-1,3)
        
        world_points_next_pred.extend([*peel_next_pred])
        
    return torch.stack(world_points_next_pred) #Nx3


def project_peels(depth_peels, extrinsic=EXTRINSIC, K=K):

    P, H, W = depth_peels.shape
    world_points_peels = []
    
    for i in range(P):
        if is_mask:
            world_points_peel, mask = backproject(depth_peels[i].reshape(H, W), extrinsic, K)
            world_points_peels.append(world_points_peel.T)#4xNx3
            
        else:
            world_points_peel = backproject(depth_peels[i], extrinsic, K)
            world_points_peels.append(world_points_peel.T)
    
    return world_points_peels

def project_peels_pred(scf, depth_curr, extrinsic=EXTRINSIC, K=K, is_mask=True):
    
    B, H, W = depth_curr.shape
    world_points_next_pred = []
    scf = scf.permute(0,2,3,1) #B,H,W,3
    
    for b in range(B):
        
        if is_mask:
            peel_curr, mask = backproject(depth_curr[b], is_mask=is_mask)
            peel_next_pred = (peel_curr.T + scf[b,:,:,:][mask]).T
            world_points_next_pred.append(peel_next_pred)
            
        else:
            peel_curr = backproject(depth_curr[b], is_mask=is_mask)
            peel_next_pred = (peel_curr.T + scf[b,:,:,:].reshape(-1,3)).T
            world_points_next_pred.append(peel_next_pred)
        
    return torch.stack(world_points_next_pred) #B,3,512*512



def batch_pointclouds(depth):
    
    """
    Make a batch of pointclouds for every peel 
    depth: B,H,W
    """
    
    B, H, W = depth.shape
    pointcloud_batch = []
    
    for b in range(B):
        
        p_cld, _ = backproject(depth[b])
        pointcloud_batch.append(p_cld.T)
        
    #torch.stack(pointcloud_batch)
        
    return Pointclouds(points=pointcloud_batch)
        
    
def batch_pointclouds_pred(scf, depth):
    
    """
    depth: B,H,W
    scf: B,H,W,3
    """
    
    B, H, W = depth.shape
    scf = scf.permute(0,2,3,1)
    pointcloud_batch = []
    
    for b in range(B):
        p_cld, mask = backproject(depth[b])
        p_cld_next = p_cld.T + scf[b,:,:,:][mask]
        pointcloud_batch.append(p_cld_next)
        
    return Pointclouds(points=pointcloud_batch)


def project(scf, depth_curr):
    
    """
    Args:
        scf: (B*K, 12, H, W)
        depth_curr: (B*K, 4, H, W)
        
    returns: backprojected depth maps
        depth_next_pred:(B*K, 4, H, W)
        
    """
    
    B, P, H, W = depth_curr.shape
    depth_pred = []
    #scf = scf.permute(0,2,3,1)
    cameras = OpenGLPerspectiveCameras(device=depth_curr.device, R=R, T=T, fov=-60)
    raster_settings = PointsRasterizationSettings(image_size=512, radius=0.005, points_per_pixel=1)

    rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    for i in range(P):
        
        pointclouds_next = batch_pointclouds_pred(scf[:,i*3:i*3+3,:,:] , depth_curr[:,i,:,:].reshape(B,H,W))
    
        frag = rasterizer(pointclouds_next)
        depth = frag.zbuf
        depth = depth.reshape(B,H,W)
        depth[torch.where(depth<0)] = 0
        
        depth_pred.append(depth)
        
    depth_pred = torch.stack(depth_pred).permute(1,0,2,3) #B,4,H,W
    
    return depth_pred
        

    
    
