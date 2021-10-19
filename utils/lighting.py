import trimesh
import numpy as np
import os
import cv2
import cv2
from ray_trace_rendering import *
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt


def parallise_rgb(light, camera, index_ray_all, index_tri_all, locations_all, ray_origins, ray_directions, idx):

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

    else:
        color1 = np.array([0, 0, 0])
        color2 = np.array([0, 0, 0])
        color3 = np.array([0, 0, 0])
        color4 = np.array([0, 0, 0])

    return color1, color2, color3, color4


def parallise_rgb(light, camera, index_ray_all, index_tri_all, locations_all, ray_origins, ray_directions, idx):

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
            rgb_colors = np.array(mesh.visual.face_colors[index_tri])[:, :3][:, ind]

        except:
            ind = [2, 1, 0]
            mesh.visual = mesh.visual.to_color()
            rgb_colors = np.array(mesh.visual.face_colors[index_tri])[:, :3][:, ind]

        face_color1 = np.array([0, 0, 0])
        face_color2 = np.array([0, 0, 0])
        face_color3 = np.array([0, 0, 0])
        face_color4 = np.array([0, 0, 0])

        if(len(rgb_colors[first])):
            face_color1 = rgb_colors[first][0]/255
            # seg_colors_1 = seg_colors[first][0]
        if(len(rgb_colors[second])):
            face_color2 = rgb_colors[second][0]/255
            # seg_colors_2 = seg_colors[second][0]
        if(len(rgb_colors[third])):
            face_color3 = rgb_colors[third][0]/255
            # seg_colors_3 = seg_colors[third][0]
        if(len(rgb_colors[fourth])):
            face_color4 = rgb_colors[fourth][0]/255
            # seg_colors_4 = seg_colors[fourth][0]

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

        # seg_colors_1 = np.array([0,0,0])
        # seg_colors_2 = np.array([0,0,0])
        # seg_colors_3 = np.array([0,0,0])
        # seg_colors_4 = np.array([0,0,0])
        depth1 = 0.0
        depth2 = 0.0
        depth3 = 0.0
        depth4 = 0.0

    return depth1, depth2, depth3, depth4


    
    
    