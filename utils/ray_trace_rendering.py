import trimesh
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from tqdm import tqdm
import time
import cv2
from operator import itemgetter

class SolidObjects:
    
    def __init__(self,mesh,ambient=None,diffuse=None,specular=None,shininess=None,
                 reflection=0.0,refraction=0.0,ior=None,name=None,cast_shadow=True):
        self.mesh = mesh
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction
        self.ior = ior
        self.name = name
        self.cast_shadow = cast_shadow
        
    def ray_intersection(self,ray_origin,ray_direction,layer):
        ray_origins = np.array([ray_origin])
        ray_directions = np.array([ray_direction])
        # run the mesh-ray query
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,multiple_hits=True)
        min_distance = np.inf
        normal = None
        face_color = None
        distances = []
        for k in range(len(locations)):
            dist = np.linalg.norm(locations[k]-ray_origin)
            distances.append((dist,k,locations[k]))
        if len(distances)>0:
            distances = sorted(distances,key=itemgetter(0))
            min_distance, indx,loc = distances[layer-1]
            nearest_face_index = index_tri[indx]
            normal = self.mesh.face_normals[nearest_face_index]
            face_color = self.mesh.visual.face_colors[nearest_face_index][:3]
            face_color = np.array(face_color,dtype=np.float)
        return min_distance, normal, face_color


# HELPER FUNCTIONS

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def nearest_intersected_object(objects, ray_origin, ray_direction,layer):
    values = [obj.ray_intersection(ray_origin, ray_direction,layer) for obj in objects]
    min_distance = values[0][0]
    normal = values[0][1]
    face_color = values[0][2]
    return objects[0], min_distance, normal, face_color

#===================================================================================================================================

#RECURSIVE RAY TRACER
def RecursiveRayTracing(objects,origin,direction,light,camera,peel_color,depth,reflection,ior, layer):
    
    color = np.zeros(3,)
    
    if depth==0 or reflection==0.0:
        return color
    
    # check for intersections
    nearest_object, min_distance, normal, face_color = nearest_intersected_object(objects, origin, direction,layer)
    if nearest_object is None:
        return color
    if min_distance==np.inf:
        return color

    # return face_color
    intersection = origin + min_distance * direction
    normal_to_surface = normalize(normal)
    shifted_point = intersection +  1e-5*normal_to_surface
    intersection_to_light = normalize(light['position'] - shifted_point)
    intersection_to_light_distance = np.linalg.norm((light['position'] - intersection))        

    
    #shadow calculation
    shadow_factor = 1.0
    
    _, min_distance,_, _ = nearest_intersected_object(objects, shifted_point, intersection_to_light,layer=1)            
    is_shadowed = min_distance < intersection_to_light_distance

    if is_shadowed:
        shadow_factor = 0.8


    illumination = np.zeros((3))

    nearest_object.ambient = face_color
    nearest_object.diffuse = face_color

    # ambient
    illumination += nearest_object.ambient * light['ambient']

    # diffuse
    if(np.dot(intersection_to_light, normal_to_surface) <0):
        dot = -np.dot(intersection_to_light, normal_to_surface)
    else: dot=np.dot(intersection_to_light, normal_to_surface)
    illumination += nearest_object.diffuse * light['diffuse'] * dot


    # specular
    intersection_to_camera = normalize(camera - intersection)
    H = normalize(intersection_to_light + intersection_to_camera)
    illumination += nearest_object.specular * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object.shininess / 4)

    # reflection
    color += reflection * illumination
    reflection *= nearest_object.reflection
    
    reflection_weight = 0.0
    
    origin = shifted_point
    direction = reflected(direction, normal_to_surface)
    if layer==2 or layer==4:
        direction *= -1
    layer = 1

    RRT = RecursiveRayTracing(objects,origin, direction,light,camera,face_color,depth-1,reflection,ior,layer)
    return shadow_factor * ( color +  reflection_weight*np.array(RRT)  )
