import trimesh
import numpy as np
import os
import cv2
import cv2
from utils.ray_trace_rendering import *
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

from utils.utils import clothed_mesh_init

scene = None
mesh = None
objects = None


def init_frame(frame):
    if 'Frame' in frame:
        return frame[5:]
    else:
        return 0


""" parallise"""
def parallise(light, camera, index_ray_all, index_tri_all, locations_all, ray_origins, ray_directions, idx):

    global mesh
    global objects
    where = np.where(index_ray_all == idx)
    index_tri = index_tri_all[where]
    locations = locations_all[where]
    index_ray = list(np.zeros(len(where[0].tolist())))

    face_color1 = np.array([0, 0, 0])
    face_color2 = np.array([0, 0, 0])
    face_color3 = np.array([0, 0, 0])
    face_color4 = np.array([0, 0, 0])
    iters = len(index_tri)
    if(iters):

        """ Separate ray hits """
        unique_index = np.unique(np.array(index_ray))
        occurences = [list(np.where(index_ray == unique)[0])
                      for unique in unique_index]
        intersections_1 = list(filter(lambda x: len(x) > 0, occurences))
        intersections_2 = list(filter(lambda x: len(x) > 1, occurences))
        intersections_3 = list(filter(lambda x: len(x) > 2, occurences))
        intersections_4 = list(filter(lambda x: len(x) > 3, occurences))

        first = [x[0] for x in intersections_1]
        second = [x[1] for x in intersections_2]
        third = [x[2] for x in intersections_3]
        fourth = [x[3] for x in intersections_4]

        # depth_maps
        z_vec = np.array([0, 0, -1])
        depth = trimesh.util.diagonal_dot(locations - ray_origins[0], z_vec)
        depth1 = 0.0
        depth2 = 0.0
        depth3 = 0.0
        depth4 = 0.0
        if len(first) > 0:
            depth1 = depth[first][0]
        if len(second) > 0:
            depth2 = depth[second][0]
        if len(third) > 0:
            depth3 = depth[third][0]
        if len(fourth) > 0:
            depth4 = depth[fourth][0]

        ind = [2, 1, 0]

        try:
            ind = [2, 1, 0]
            rgb_colors = np.array(mesh.visual.face_colors[index_tri])[
                :, :3][:, ind]


        except:
            ind = [2, 1, 0]
            mesh.visual = mesh.visual.to_color()

            rgb_colors = np.array(mesh.visual.face_colors[index_tri])[
                :, :3][:, ind]

        face_color1 = np.array([0, 0, 0])
        face_color2 = np.array([0, 0, 0])
        face_color3 = np.array([0, 0, 0])
        face_color4 = np.array([0, 0, 0])

        if(len(rgb_colors[first])):
            face_color1 = rgb_colors[first][0]/255

        if(len(rgb_colors[second])):
            face_color2 = rgb_colors[second][0]/255

        if(len(rgb_colors[third])):
            face_color3 = rgb_colors[third][0]/255

        if(len(rgb_colors[fourth])):
            face_color4 = rgb_colors[fourth][0]/255

    if(iters):
        color1 = np.array([0, 0, 0])
        color2 = np.array([0, 0, 0])
        color3 = np.array([0, 0, 0])
        color4 = np.array([0, 0, 0])
        if len(first) > 0:
            color1 = RecursiveRayTracing(
                objects, ray_origins[idx], ray_directions[idx], light, camera, np.array(face_color1), 4, 1.0, 1.0, 1)
        if len(second) > 0:
            color2 = RecursiveRayTracing(
                objects, ray_origins[idx], ray_directions[idx], light, camera, np.array(face_color2), 4, 1.0, 1.0, 2)
        if len(third) > 0:
            color3 = RecursiveRayTracing(
                objects, ray_origins[idx], ray_directions[idx], light, camera, np.array(face_color3), 4, 1.0, 1.0, 3)
        if len(fourth) > 0:
            color4 = RecursiveRayTracing(
                objects, ray_origins[idx], ray_directions[idx], light, camera, np.array(face_color4), 4, 1.0, 1.0, 4)
        # print(color1.shape,seg_colors_1.shape)

    else:
        color1 = np.array([0, 0, 0])
        color2 = np.array([0, 0, 0])
        color3 = np.array([0, 0, 0])
        color4 = np.array([0, 0, 0])

        depth1 = 0.0
        depth2 = 0.0
        depth3 = 0.0
        depth4 = 0.0

    return color1, color2, color3, color4, depth1, depth2, depth3, depth4


def peelGen(root, path, out_dir_rgb, out_dir_depth, nproc=20):

    global scene
    global mesh
    global mesh_seg
    global objects
    scene = None
    mesh = None
    objects = None

    """ Load meshes """
    mesh = clothed_mesh_init(root, path)

    """ Create Scene """
    scene = mesh.scene()
    scene.camera_transform = np.eye(4)
    scene.camera.resolution = [512, 512]
    scene.camera.fov = [60, 60]

    mesh.unmerge_vertices()


    """create light area"""
    len_x = np.abs(np.max(mesh.vertices[:, 0])-np.min(mesh.vertices[:, 0]))
    len_y = np.abs(np.max(mesh.vertices[:, 1])-np.min(mesh.vertices[:, 1]))
    len_z = np.abs(np.max(mesh.vertices[:, 2])-np.min(mesh.vertices[:, 2]))

    light_area = trimesh.creation.box([len_x/3, len_y/3, 10e-5])
    scene.camera_transform[:, 3][:3] = scene.camera_transform[:, 3][:3]
    translate = np.copy(scene.camera_transform[:, 3][:3])
    translate[2] = translate[2]+len_z/100
    light_area.apply_translation(translate)
    scene.add_geometry(light_area)
    ray_origins, ray_directions, pixels = scene.camera_rays()

    """parameters for lighting """
    width, height = scene.camera.resolution
    camera = scene.camera_transform[:, 3][:3]
    pos = 4
    light = {'position': np.array(translate)+pos, 'ambient': np.array([0.6, 0.6, 0.6]),
             'diffuse': np.array([1.0, 1.0, 1.0]), 'specular': np.array([1.0, 1.0, 1.0])}
    
    obj = SolidObjects(mesh=mesh, name='human',
                       ambient=np.array([1.0, 1.0, 1.0]), diffuse=np.array([1.0, 1.0, 1.0]),
                       specular=np.array([1.0, 1.0, 1.0]), shininess=100, reflection=0.0)
    objects = [obj]

    """Render scene via ray tracing"""
    image_res = scene.camera.resolution

    locations_all, index_ray_all, index_tri_all = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True)

    """multi-processing"""
    with mp.Pool(processes=nproc) as pooled_row:
        func = partial(parallise, light, camera, index_ray_all,
                       index_tri_all, locations_all, ray_origins, ray_directions)
        results = pooled_row.map(func, range(0, image_res[0]*image_res[1]))

    color1 = []
    color2 = []
    color3 = []
    color4 = []

    depth1 = []
    depth2 = []
    depth3 = []
    depth4 = []


    for res in results:
        color1.append(res[0])
        color2.append(res[1])
        color3.append(res[2])
        color4.append(res[3])
        depth1.append(res[4])
        depth2.append(res[5])
        depth3.append(res[6])
        depth4.append(res[7])


    color1 = np.array(color1).reshape(image_res[0], image_res[1], 3)
    color2 = np.array(color2).reshape(image_res[0], image_res[1], 3)
    color3 = np.array(color3).reshape(image_res[0], image_res[1], 3)
    color4 = np.array(color4).reshape(image_res[0], image_res[1], 3)

    color1 = ((color1 - color1.min()) / (color1.max() - color1.min()))*255
    color1 = cv2.flip(np.array(color1, dtype=np.uint8), 1)

    color2 = ((color2 - color2.min()) / (color2.max() - color2.min()))*255
    color2 = cv2.flip(np.array(color2, dtype=np.uint8), 1)

    color3 = ((color3 - color3.min()) / (color3.max() - color3.min()))*255
    color3 = cv2.flip(np.array(color3, dtype=np.uint8), 1)

    color4 = ((color4 - color4.min()) / (color4.max() - color4.min()))*255
    color4 = cv2.flip(np.array(color4, dtype=np.uint8), 1)

    depth1 = cv2.flip(np.array(depth1, dtype=np.float32).reshape(
        image_res[0], image_res[1]), 1)
    depth2 = cv2.flip(np.array(depth2, dtype=np.float32).reshape(
        image_res[0], image_res[1]), 1)
    depth3 = cv2.flip(np.array(depth3, dtype=np.float32).reshape(
        image_res[0], image_res[1]), 1)
    depth4 = cv2.flip(np.array(depth4, dtype=np.float32).reshape(
        image_res[0], image_res[1]), 1)

    cv2.imwrite(out_dir_rgb+'/rgb_01.png', cv2.cvtColor(color1, cv2.COLOR_BGR2RGB))
    cv2.imwrite(out_dir_rgb+'/rgb_02.png', cv2.cvtColor(color2, cv2.COLOR_BGR2RGB))
    cv2.imwrite(out_dir_rgb+'/rgb_03.png', cv2.cvtColor(color3, cv2.COLOR_BGR2RGB))
    cv2.imwrite(out_dir_rgb+'/rgb_04.png', cv2.cvtColor(color4, cv2.COLOR_BGR2RGB))

    plt.imsave(out_dir_depth+'/dep_1.png', depth1, cmap='gray')
    plt.imsave(out_dir_depth+'/dep_2.png', depth2, cmap='gray')
    plt.imsave(out_dir_depth+'/dep_3.png', depth3, cmap='gray')
    plt.imsave(out_dir_depth+'/dep_4.png', depth4, cmap='gray')
    np.savez_compressed(out_dir_depth+'/dep_1', a=depth1)
    np.savez_compressed(out_dir_depth+'/dep_2', a=depth2)
    np.savez_compressed(out_dir_depth+'/dep_3', a=depth3)
    np.savez_compressed(out_dir_depth+'/dep_4', a=depth4)
    #np.savez_compressed(out_dir+'/com', a=mass2, b=mass)
    #np.save(out_dir+'/rotation.npy', rotate_mesh)

    scene = None
    mesh = None
    objects = None

