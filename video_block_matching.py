# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:31:28 2019

@author: pauls
"""

import numpy as np

def video_block_matching(Patches, Param):
    # Return an array with each patch index of the centered frame as the 
    # first element of each line and the K nearest patches completing the line
    # Patches is a 2D array of size (patch size, number of patches on each 
    # frame*number of frame)
    # Param is a dataclass containing the following parameters
    lsize = Param.x_patches_num
    csize = Param.y_patches_num
    patches_num = Param.x_patches_num * Param.y_patches_num
    frame_num = Param.frame_num
    Param.patches_num = patches_num
    K = Param.similar_patches_number
    t_range = Param.temporal_search_range
    x_win_size = Param.x_window_size
    y_win_size = Param.y_window_size
    BM = np.zeros((patches_num*frame_num, K), dtype=int)
    index = []
    for t in range(frame_num):
        for i in range(csize):
            for j in range(lsize):
            
                for k in range(-x_win_size//2, (x_win_size+1)//2):
                    for l in range(-y_win_size//2, (y_win_size+1)//2):
                        if i+k >= 0 and i+k < csize and j+l >= 0 and j+l < lsize:
                            index.append(i+k+(j+l)*csize)
            
                index = np.array(index)
                complete_index = np.array([], dtype=int)
                for l in range(-t_range//2, (t_range+1)//2):
                    if t+l >= 0 and t+l < frame_num:
                        complete_index = np.append(complete_index, 
                                                   index+(t+l)*patches_num)

                Patch_candidate = Patches[:,complete_index]
                Center = Patches[:,i+j*csize + t*patches_num]
                Center_patches = np.repeat(Center[:,np.newaxis], 
                                           len(complete_index), axis=-1)
                dis = np.sum((Patch_candidate - Center_patches)**2, axis=0)
                KNN = np.argsort(dis, axis=None, kind='quicksort')[:K]
                BM[i+j*csize+t*patches_num] = complete_index[KNN]
                index=[]
    
        print(BM)
    return BM