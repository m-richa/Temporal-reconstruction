import os
import argparse
from utils.get_peelmaps_lighting import *

def generate_data(dataroot, dataroot_gt, use_lighting = True):

    root = dataroot
    root_gt = dataroot_gt
    seqs = os.listdir(root+'/cloth3d_goodseqs')
    os.makedirs('dataset', exist_ok=True)
    
    print(seqs)
    
    rgb_path = os.makedirs(root_gt+'/rgb_light', exist_ok=True )
    depth_path = os.makedirs(root_gt+'/depth_light', exist_ok=True )

    
    for seq in seqs:
        print('Sequence: '+str(seq))
        os.makedirs(root_gt+'/rgb_light/'+str(seq), exist_ok=True)
        os.makedirs(root_gt+'/depth_light/'+str(seq), exist_ok=True)

        path_seq = root + '/cloth3d_goodseqs/' + str(seq)
        n = sum(os.path.isdir(i) for i in [os.path.join(path_seq, f) for f in os.listdir(path_seq)])
        print(n)
        
        for i in range(n):
    
            rgb_out_dir = root_gt+'/rgb/'+str(seq)+'/Frame{num:03d}'.format(num=i)
            depth_out_dir = root_gt+'/depth/'+str(seq)+'/Frame{num:03d}'.format(num=i)

            os.makedirs(rgb_out_dir, exist_ok=True)
            os.makedirs(depth_out_dir, exist_ok=True)
         
            
            path = path_seq + '/Frame{num:03d}'.format(num=i) + '.obj'
            
            peelGen(root, path, rgb_out_dir, depth_out_dir, nproc=20)

            print('Frame: '+str(i))
            
            
if __name__ == '__main__':
    print("INSIDE MAIN")
    parser = argparse.ArgumentParser(description='Create a schema')
    parser.add_argument('--dataroot', metavar='path', required=True,
                        help='the path to the 3D models in the following format ..../dataroot/cloth_3d_goodseqs/...')
    parser.add_argument('--dataroot_gt', metavar='path', required=True,
                        help='path to dataroot where gt peel maps will be stored')
    #parser.add_argument('--num_frames', metavar='', required=True, help='path to dem')
    
    args = parser.parse_args()
    
    generate_data(args.dataroot, args.dataroot_gt)