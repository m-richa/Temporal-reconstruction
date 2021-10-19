import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyembree
import torch
import cv2
import PIL.Image

import os
import sys


def mesh_init(root, frame, is_smpl=True):
    
    """
    Use is_smpl=True initialization for scf peels
    """
    smpl_path = str(frame)+'/human/body.obj'
    
    if is_smpl:

        mesh_path = os.path.join(root, smpl_path)
        mesh = trimesh.load(mesh_path, prefer_color = 'vertex', process=False) 
        mesh.vertices -= mesh.center_mass
    else:

        mesh_path = os.path.join(root, str(frame)+'.obj')
        smpl_path = os.path.join(root, smpl_path)
        mesh = trimesh.load(mesh_path, prefer_color = 'vertex', process=False)
        smpl_mesh = trimesh.load(smpl_path, prefer_color = 'vertex', process=False)
        
        mesh.vertices -= smpl_mesh.center_mass

       
    """
    Translate and rotate mesh
    """
    trans = trimesh.transformations.translation_matrix(np.array([0, 0, -2.5]))
    rotate1 = trimesh.transformations.rotation_matrix(angle=np.radians(90), direction=[0, 0, -1])
    rotate2 = trimesh.transformations.rotation_matrix(angle = np.radians(90), direction = [0, -1, 0])
    
    mesh.apply_transform(trans @ rotate2 @ rotate1)
    
    return mesh



def get_peeled_intersections(root, path, is_smpl=True):

    """
    Create scene and set camera at origin in world-coordinate system
    """
    mesh = mesh_init(root, path, is_smpl)
    
    scene = mesh.scene()
    scene.camera.resolution = [512, 512]
    scene.camera.fov = [60, 60]
    scene.camera_transform = np.eye(4)   
        
    
    """Ray tracing"""
    
    origins, ray_vectors, pixels = scene.camera_rays()
     
    locations, index_ray, index_tri = mesh.ray.intersects_location(origins, ray_vectors, multiple_hits=True)
    
    """
    1. locations store the locations of the intersection in world_corordinate system.
    2. index_ray stores the index of the ray_vector that intersected the mesh
    3. index_tri stores the index of the mesh face that the ray_vector intersects
    4. locations, index_ray, index_tri all all synced and store in the order in which the intersections occur for all intersections.
    """
    
    unique_rays = np.unique(np.array(index_ray))
    occurences = [list(np.where(index_ray == ray)[0]) for ray in unique_rays]
    
    intersections_1 = list(filter(lambda x: len(x) > 0, occurences))
    intersections_2 = list(filter(lambda x: len(x) > 1, occurences))
    intersections_3 = list(filter(lambda x: len(x) > 2, occurences))
    intersections_4 = list(filter(lambda x: len(x) > 3, occurences))
    
    """list of indices for first, second, thind and fourth intersections"""
    first = [x[0] for x in intersections_1]
    second = [x[1] for x in intersections_2]
    third = [x[2] for x in intersections_3]
    fourth = [x[3] for x in intersections_4]
    
    #print(len(first), len(second), len(third), len(fourth))
    
    return [first, second, third, fourth], [origins, ray_vectors, pixels], [locations, index_ray, index_tri], scene



def get_rgb_peels(root, path, intersections, ray_param, ray_intersect_param, scene, out_dir, is_smpl):
  
    mesh = mesh_init(root, path, is_smpl)
    #intersections, ray_param, ray_intersect_param, scene = get_peeled_intersections(mesh)
  
    """Initialize rgb images with white pixels"""
    
    rgb_1 = np.ones([scene.camera.resolution[0], scene.camera.resolution[1], 3], dtype=np.uint8) * 255
    rgb_2 = np.ones([scene.camera.resolution[0], scene.camera.resolution[1], 3], dtype=np.uint8) * 255
    rgb_3 = np.ones([scene.camera.resolution[0], scene.camera.resolution[1], 3], dtype=np.uint8) * 255
    rgb_4 = np.ones([scene.camera.resolution[0], scene.camera.resolution[1], 3], dtype=np.uint8) * 255
  
    first, second, third, fourth = intersections
    _, _, pixels = ray_param
    _, index_ray, index_tri = ray_intersect_param
    
    #print(len(first), len(second), len(third), len(fourth))
    
    ind = [2, 1, 0]
    rgb_colors = np.array(mesh.visual.face_colors[index_tri])[:, :3][:, ind]
    
    pixels_1 = pixels[index_ray[first]]
    pixels_2 = pixels[index_ray[second]]
    pixels_3 = pixels[index_ray[third]]
    pixels_4 = pixels[index_ray[fourth]]
    
    rgb_1_list = rgb_colors[first]
    rgb_2_list = rgb_colors[second]
    rgb_3_list = rgb_colors[third]
    rgb_4_list = rgb_colors[fourth]
    
    temp = [(pixels_1, rgb_1_list, rgb_1),
            (pixels_2, rgb_2_list, rgb_2),
            (pixels_3, rgb_3_list, rgb_3),
            (pixels_4, rgb_4_list, rgb_4)]
    
  
    for (a, b, c) in temp:
        ind0 = np.where(a[:, 0]< scene.camera.resolution[0])
        ind1 = np.where(a[:, 1]< scene.camera.resolution[0])
        ind = np.intersect1d(ind0, ind1)
        a = a[ind]
    
        c[a[:, 0], a[:, 1], :] = b[ind]
      
    #name = path.split('/')[0]
    #print(name)
    #os.makedirs(root+'/PeeledMaps/'+name, exist_ok=True)
    mesh_folder = out_dir
    cv2.imwrite(mesh_folder + '/' + 'rgb_01.png', rgb_1)
    cv2.imwrite(mesh_folder + '/' + 'rgb_02.png', rgb_2)
    cv2.imwrite(mesh_folder + '/' + 'rgb_03.png', rgb_3)
    cv2.imwrite(mesh_folder + '/' + 'rgb_04.png', rgb_4)
    
    
def get_depth_peels(root, frame, intersections, ray_param, ray_intersect_param, scene, out_dir, is_smpl):
  
    mesh = mesh_init(root, frame, is_smpl)
    #intersections, ray_param, ray_intersect_param, scene = get_peeled_intersections(mesh)
    
    """Initialize depth maps"""
    
    a_raw = np.zeros(scene.camera.resolution, dtype=np.float32)
    b_raw = np.zeros(scene.camera.resolution, dtype=np.float32)
    c_raw = np.zeros(scene.camera.resolution, dtype=np.float32)
    d_raw = np.zeros(scene.camera.resolution, dtype=np.float32)
    
    first, second, third, fourth = intersections
    origins, _, pixels = ray_param
    locations, index_ray, index_tri = ray_intersect_param
    
    principal_axis = np.tile(np.expand_dims(np.array([0, 0, -1]), 0), (len(pixels), 1)) #why -1 and not 1
    
    depth = trimesh.util.diagonal_dot(locations - origins[0], principal_axis[index_ray])
    
    
    """Pixels"""
    pixels_1 = pixels[index_ray[first]]
    pixels_2 = pixels[index_ray[second]]
    pixels_3 = pixels[index_ray[third]]
    pixels_4 = pixels[index_ray[fourth]]
    
    depth_1 = depth[first]
    depth_2 = depth[second]
    depth_3 = depth[third]
    depth_4 = depth[fourth]
    
    temp = [(pixels_1, depth_1, a_raw), (pixels_2, depth_2, b_raw),
      (pixels_3, depth_3, c_raw), (pixels_4, depth_4, d_raw)]
    
    for (a, b, c) in temp:
        ind0 = np.where(a[:, 0] < scene.camera.resolution[0])
        ind1 = np.where(a[:, 1] < scene.camera.resolution[1])
        ind = np.intersect1d(ind0, ind1)
        a = a[ind]
        
        c[a[:, 0], a[:, 1]] = b[ind]
    
    #name = path.split('/')[0]
    #print(name)
    #os.makedirs(root+'/PeeledMaps/'+name, exist_ok=True)
    mesh_folder = out_dir
    plt.imsave(mesh_folder + '/' + 'dep_1.jpg', a_raw, cmap='gray')
    plt.imsave(mesh_folder + '/' + 'dep_2.jpg', b_raw, cmap='gray')
    plt.imsave(mesh_folder + '/' + 'dep_3.jpg', c_raw, cmap='gray')
    plt.imsave(mesh_folder + '/' + 'dep_4.jpg', d_raw, cmap='gray')
    np.savez_compressed(mesh_folder + '/' + 'dep_1', a = a_raw)
    np.savez_compressed(mesh_folder + '/' + 'dep_2', a = b_raw)
    np.savez_compressed(mesh_folder + '/' + 'dep_3', a = c_raw)
    np.savez_compressed(mesh_folder + '/' + 'dep_4', a = d_raw)
    
    return [a_raw, b_raw, c_raw, d_raw], scene.camera.K
