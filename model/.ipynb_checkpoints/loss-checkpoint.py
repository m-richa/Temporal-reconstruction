import torch.nn.functional as F
import numpy as np
import torch
from utils.loss_utils import *
from pytorch3d.loss import chamfer_distance
from torchgeometry.losses import SSIM


def ssim_L1_loss(scf_predicted, curr_rgb, next_rgb, depth_curr):
    """
    Args:
    Returns:
    """
    
    B, _, H, W = scf_predicted.shape
    
    #un-normalize
    
    scf_predicted = scf_predicted*0.6 - 0.3
    curr_rgb = curr_rgb*127.5 + 127.5
    next_rgb = next_rgb*127.5 + 127.5
    
    ssim_loss = SSIM(5, reduction='mean')
    loss = 0
    
    for i in range(4):
        warp_rgb_peel = warp_rgb(scf_predicted[:,i*3:i*3+3,:,:], next_rgb[:,i*3:i*3+3,:,:], depth_curr[:,i,:,:])
        ssim_peel_loss = 1- ssim_loss(warp_rgb_peel, curr_rgb[:,i*3:i*3+3,:,:])
        l1_peel_loss = F.l1_loss(warp_rgb_peel, curr_rgb[:,3*i:3*i+3,:,:])
        
        loss = loss+ssim_peel_loss+l1_peel_loss
        
    print(loss)
        
    return loss


def depth_L1_loss(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    scf_predicted = scf_predicted*(0.6) - 0.3 #Un-normalize
    
    depth_next_pred = backproject(scf_predicted, depth_curr_gt)
    loss = F.l1_loss(depth_next_gt, depth_next_pred)

    return loss

def chamfer_loss_peel1(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    scf_predicted =scf_predicted*0.3
    
    pointclouds_next1 = batch_pointclouds(depth_next_gt[:,0,:,:])
    pointclouds_next1_pred = batch_pointclouds_pred(scf_predicted[:,:3,:,:], depth_curr_gt[:,0,:,:])
    
    loss_peel1 = chamfer_distance(pointclouds_next1, pointclouds_next1_pred)
    print(loss_peel1[0])
    
    return loss_peel1[0]


def chamfer_L1_loss(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    scf_predicted =scf_predicted*0.6 - 0.3
    loss = 0
    for i in range(4):
    
        pointclouds_next = batch_pointclouds(depth_next_gt[:,i,:,:]) #B,N,3
        pointclouds_next_pred = batch_pointclouds_pred(scf_predicted[:,i*3:i*3+3,:,:], depth_curr_gt[:,i,:,:]) #B,N,3
        loss_peel = chamfer_distance(pointclouds_next, pointclouds_next_pred)
        loss = loss+loss_peel[0]  
        
    print(loss)
    loss_L1 = depth_L1_loss(scf_predicted, depth_curr_gt, depth_next_gt)
    
    return loss+loss_L1

    
    
def chamfer_peel1_L1_loss(scf_predicted, depth_curr_gt, depth_next_gt):
    
    loss_L1 = depth_L1_loss(scf_predicted, depth_curr_gt, depth_next_gt)
    loss_chamfer = chamfer_loss_peel1(scf_predicted, depth_curr_gt, depth_next_gt)
    
    return loss_L1+1000*loss_chamfer

    
def chamfer_loss(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    #scf_predicted = scf_predicted.permute()
    
    loss=[]
    prediction = []
    gt = []

    for b in range(B):
        
        for i in range(4):
            world_points_curr = project_peel(depth_curr_gt[b,i,:,:])
            world_points_next = project_peel(depth_next_gt[b,i,:,:])

            world_points_next_pred = world_points_curr + scf_predicted[b, i*3:i*3 + 3,:,:].reshape(3, H*W)
            
            world_points_next_pred = world_points_next_pred.T
            world_points_next = world_points_next.T
            
            loss_per_peel = chamfer_distance(world_points_next.unsqueeze(0), world_points_next_pred.unsqueeze(0))
            loss.append(loss_per_peel[0])
            #gt.append(torch.stack(world_points_next))
            #prediction.append(world_points_next_pred)

        #world_points_curr = projection_batch(depth_curr_gt[i,:,:,:])#.reshape(4,H*W,3)
        #world_points_next = projection_batch(depth_next_gt[i,:,:,:])#.reshape(4,H*W,3)
        #world_points_next_pred = torch.stack([world_points_curr[b] + scf_predicted[b, i*3:i*3 + 3,:,:].reshape(H*W, 3) for b in range(B)])
        
        #loss_per_peel = chamfer_distance(torch.stack(world_points_next), world_points_next_pred)

        #loss.append(loss_per_peel[0])
        #gt.append(torch.stack(world_points_next))
        #prediction.append(world_points_next_pred)
    
    loss = torch.stack(loss)
    #prediction = torch.stack(prediction)
    #gt = torch.stack(gt)
    
    return torch.mean(loss)#, prediction, gt

def chamfer_loss_body(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    
    loss = []
    for i in range(B):
        
        scf_predicted = scf_predicted*0.2 #Un-normalize
        
        world_points_next_gt = project_fullbody(depth_next_gt[i])
        world_points_next_pred = project_fullbody_pred(scf_predicted[i,:,:,:], depth_curr_gt[i])

        loss1 = 100*chamfer_distance(world_points_next_gt.unsqueeze(0), world_points_next_pred.unsqueeze(0))
        loss.append(loss1[0])
        
    loss = torch.stack(loss)
    return torch.mean(loss)

def chamfer_loss_body_peels(scf_predicted, depth_curr_gt, depth_next_gt):
    
    B, _, H, W = scf_predicted.shape
    scf_predicted = scf_predicted*0.2
    loss = []
    for b in range(B):
    

        world_points_next_gt = project_peels(depth_next_gt[b])
        world_points_next_pred = project_peels_pred(scf_predicted[b,:,:,:], depth_curr_gt[b])
        
        loss_peels = []
        for i in range(4):

            loss1 = chamfer_distance(world_points_next_gt[i].unsqueeze(0), world_points_next_pred[i].unsqueeze(0))

            loss_peels.append(loss1[0])
            
        loss.append(torch.sum(torch.stack(loss_peels)))

    return torch.mean(torch.stack(loss))

def chamfer_loss_bodyandpeels(scf_predicted, depth_curr_gt, depth_next_gt):
    
    return chamfer_loss_body(scf_predicted, depth_curr_gt, depth_next_gt) + chamfer_loss_body_peels(scf_predicted, depth_curr_gt, depth_next_gt)
