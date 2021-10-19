import os
import argparse
from utils.utils import get_depth_peels, get_rgb_peels, get_peeled_intersections
from utils.sceneflow import get_scf_peels

def generate_data(dataroot, dataroot_gt, use_lighting = True):

    root = dataroot
    root_gt = dataroot_gt
    seqs = os.listdir(root+'/cloth3d_goodseqs')
    os.makedirs('dataset', exist_ok=True)
    
    print(seqs)
    
    os.makedirs(root_gt+'/rgb', exist_ok=True )
    os.makedirs(root_gt+'/depth', exist_ok=True )
    #os.makedirs(root_gt+'/scf', exist_ok=True )
    
    for seq in seqs:
        print('Sequence: '+str(seq))
        os.makedirs(root_gt+'/rgb/'+str(seq), exist_ok=True)
        os.makedirs(root_gt+'/depth/'+str(seq), exist_ok=True)
        #os.makedirs(root_gt+'/scf/'+str(seq), exist_ok=True)
        
        path_seq = root + '/cloth3d_goodseqs/' + str(seq)
        n = sum(os.path.isdir(i) for i in [os.path.join(path_seq, f) for f in os.listdir(path_seq)])
        print(n)
        
        for i in range(n):
    
            rgb_out_dir = root_gt+'/rgb/'+str(seq)+'/Frame{num:03d}'.format(num=i)
            depth_out_dir = root_gt+'/depth/'+str(seq)+'/Frame{num:03d}'.format(num=i)
            #scf_out_dir = root_gt+'/scf/'+str(seq)+'/Frame{num:03d}'.format(num=i)
            
            os.makedirs(rgb_out_dir, exist_ok=True)
            os.makedirs(depth_out_dir, exist_ok=True)
            #os.makedirs(scf_out_dir, exist_ok=True)
            
            frame = '/Frame{num:03d}'.format(num=i)
            
            intersections, ray_param, ray_intersect_param, scene = get_peeled_intersections(path_seq, frame, is_smpl=False)
            #intersections, ray_param_scf, ray_intersect_param_scf, scene_scf = get_peeled_intersections(path_seq, frame, is_smpl=True)
            
            get_rgb_peels(path_seq, frame, intersections, ray_param, ray_intersect_param, scene, rgb_out_dir, is_smpl=False)
            get_depth_peels(path_seq, path, intersections, ray_param, ray_intersect_param, scene, depth_out_dir, is_smpl=False)
            print('Frame: '+str(i))
            
            """
            if(i<(n-1)):
                path_next = path_seq + '/Frame{num:03d}'.format(num=i+1) + '/human/body.obj'
                _, _ = get_scf_peels(root, path_scf, path_next,
                                 intersections, ray_param,
                                 ray_intersect_param, scene, scf_out_dir)"""
            
            
if __name__ == '__main__':
    print("INSIDE MAIN")
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--dataroot', metavar='path', required=True,
                        help='the path to the smpl models in the following format ..../dataroot/cloth_3d_goodseqs/...')
    parser.add_argument('--dataroot_gt', metavar='path', required=True,
                        help='path to dataroot where gt peel maps will be stored')
    #parser.add_argument('--num_frames', metavar='', required=True, help='path to dem')
    
    args = parser.parse_args()
    
    generate_data(args.dataroot, args.dataroot_gt)
