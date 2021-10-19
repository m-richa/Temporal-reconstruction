import trimesh
import torch
import pyembree
import PIL.Image
import cv2

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from utils.utils import mesh_init, get_peeled_intersections


def get_scf_per_intersection(mesh_curr, mesh_next, intersections, ray_param, ray_intersect_param):
    
    #rotate = trimesh.transformations.rotation_matrix(angle=np.radians(90), direction=[0,0,1])
    #rotate2 = trimesh.transformations.rotation_matrix(angle=np.radians(180), direction=[0,1,0])
    #extrinsic = rotate2[:3, :3] @ rotate[:3, :3]
        
    
    scf_per_vertex = np.array(mesh_next.vertices) - np.array(mesh_curr.vertices)
    
    #scf_per_vertex = (extrinsic @ (np.array(mesh_next.vertices) - np.array(mesh_curr.vertices)).T).T
    
    """
    Calculate the intersection points for the mesh in the current frame.
    """
    
    #intersections, ray_param, ray_intersect_param, scene = get_peeled_intersections(mesh_curr)
    
    first, second, third, fourth = intersections
    _, _, pixels = ray_param
    locations, index_ray, index_tri = ray_intersect_param
    
    """
    Map scf_per_vertex to scf_per_intersection using barycentric interpolation
    """
    scf_per_intersection = np.zeros((len(locations), 3))
    n = len(locations) #number of interections
    
    for i in range(n):
        v1_i, v2_i, v3_i = np.array(mesh_curr.faces[index_tri[i]])
        v1, v2, v3 = mesh_curr.vertices[[v1_i, v2_i, v3_i]]
        
        v2v1 = v2 - v1
        v2v1_d = v2v1/np.linalg.norm(v2v1)
        v3v1 = v3 - v1 
        v3v1_d = v3v1/np.linalg.norm(v3v1)
        v3v2 = v3 - v2
        v3v2_d = v3v2/np.linalg.norm(v3v2)
        pv1 = locations[i] - v1
        pv2 = locations[i] - v2
        
        p_to_v1v2 = np.linalg.norm(pv1 - pv1.dot(v2v1)* v2v1_d)
        p_to_v1v3 = np.linalg.norm(pv1 - pv1.dot(v3v1)* v3v1_d)
        p_to_v3v2 = np.linalg.norm(pv2 - pv1.dot(v3v2)* v3v2_d)
        c = 1/(p_to_v1v2 + p_to_v1v3 + p_to_v3v2)
        
        t = c*p_to_v3v2
        u = c*p_to_v1v3
        v = 1-u-t
        
        #bary_coor = np.array([t,u,v]).T
        scf_per_intersection[i] = t*scf_per_vertex[v1_i] + u*t*scf_per_vertex[v2_i] + t*scf_per_vertex[v3_i]
        
    return scf_per_intersection



def get_scf_peels(root, frame_curr, frame_next, intersections, ray_param, ray_intersect_param, scene, out_dir, is_smpl=True):
    
    mesh_curr = mesh_init(root, frame_curr, is_smpl)
    mesh_next = mesh_init(root, frame_next, is_smpl)
    
    scf_per_intersection = get_scf_per_intersection(mesh_curr, mesh_next,
                                   intersections, ray_param,
                                   ray_intersect_param)
    
    """
    rotate = trimesh.transformations.rotation_matrix(angle=np.radians(90), direction=[0,0,1])
    rotate2 = trimesh.transformations.rotation_matrix(angle=np.radians(180), direction=[0,1,0])
    extrinsic = rotate2[:3, :3] @ rotate[:3, :3]
    scf_per_intersection = (extrinsic @ scf_per_intersection.T).T
    """
    first, second, third, fourth = intersections
    _, _, pixels = ray_param
    _, index_ray, _ = ray_intersect_param
    
    """ Pixel coordinates for each layer """
    pixel_ray_1 = pixels[index_ray[first]]
    pixel_ray_2 = pixels[index_ray[second]]
    pixel_ray_3 = pixels[index_ray[third]]
    pixel_ray_4 = pixels[index_ray[fourth]]
    
    """ Scene flow """
    scf_1_arr = np.zeros((scene.camera.resolution[0], scene.camera.resolution[1], 3), dtype=np.float32)
    scf_2_arr = np.zeros((scene.camera.resolution[0], scene.camera.resolution[1], 3), dtype=np.float32)
    scf_3_arr = np.zeros((scene.camera.resolution[0], scene.camera.resolution[1], 3), dtype=np.float32)
    scf_4_arr = np.zeros((scene.camera.resolution[0], scene.camera.resolution[1], 3), dtype=np.float32)
    
    
    scf_1 = scf_per_intersection[first]
    scf_2 = scf_per_intersection[second]
    scf_3 = scf_per_intersection[third]
    scf_4 = scf_per_intersection[fourth]
    
    temp = [(pixel_ray_1, scf_1, scf_1_arr), (pixel_ray_2, scf_2, scf_2_arr),
           (pixel_ray_3, scf_3, scf_3_arr), (pixel_ray_4, scf_4, scf_4_arr)]
    
    for (a, b, c) in temp:
        ind0 = np.where(a[:, 0] < scene.camera.resolution[0])
        ind1 = np.where(a[:, 1] < scene.camera.resolution[0])
        ind = np.intersect1d(ind0, ind1)
        a = a[ind]
        
        c[a[:, 0], a[:, 1], :] = b[ind]
        
    #name = path_curr.split('/')[0]
    #print(name)
    #os.makedirs(root+'/PeeledMaps/'+name, exist_ok=True)
    mesh_folder = out_dir
        
    np.savez_compressed(mesh_folder + '/' + 'scf_1', a = scf_1_arr)
    np.savez_compressed(mesh_folder + '/' + 'scf_2', a = scf_2_arr)
    np.savez_compressed(mesh_folder + '/' + 'scf_3', a = scf_3_arr)
    np.savez_compressed(mesh_folder + '/' + 'scf_4', a = scf_4_arr)
    
    return [scf_1_arr, scf_2_arr, scf_3_arr, scf_4_arr], scene.camera.K
