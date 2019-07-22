# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:55:32 2019

@author: pauls
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

import patches
import video_block_matching as vbm
import sparse_coding
import transform_update as tu
from LR_approx import LR_approx
import parameters
import aggregation as ag

def SALT(Video, new=True, BM=None, Patches=None, Param=None):
    if new:
        parameters.set_parameters(Video)
        Param = parameters.Param
    w_lr = Param.low_rank_weight
    w_s = Param.sparse_weight
    w_d = Param.data_weight
    m = Param.temporal_search_range + 1
    patches_num = Param.patches_num
    psize = Param.psize
    frame_num = Param.frame_num
    if new:
        Patches = patches.overlapping_patches(Video, Param)    
        BM = vbm.video_block_matching(Patches, Param)
    transform = scipy.fftpack.dct(np.eye(psize*m), norm='ortho', axis=0)
    Denoised_video = np.zeros(Video.shape)
    
    for t in range(frame_num):
        data_to_sparse = np.zeros((psize*m, patches_num))        
        for i in range(patches_num):
            Patches[:,BM[i+t*patches_num]] = LR_approx(
                    Patches[:,BM[i+t*patches_num]], Param)
            data_to_sparse[:,i] = np.ravel(Patches[:,BM[i+t*patches_num,:m]])
        
        sparsed = sparse_coding.sparse(data_to_sparse, Param, transform)        
        transform = tu.trans_update(sparsed, data_to_sparse)      
        sparsed = sparse_coding.sparse(data_to_sparse, Param, transform)
        
        for i in range(patches_num):
            s = np.zeros((psize, m))
            KNN = Patches[:,BM[i+t*patches_num]]
            D = LR_approx(KNN, Param)
            index1 = BM[i+t*patches_num,:m]
            index2 = BM[i+t*patches_num,m:]
            s = (transform.T @ sparsed[:,i]).reshape((psize, m))
            Patches[:,index1] = ((w_s*s + w_lr*D[:,:m] + w_d*Patches[:,index1])
                / (w_s+w_d+w_lr))
            Patches[:,index2] = ((w_lr*D[:,m:] + w_d*Patches[:,index2]) 
                / (w_lr+w_d))
        
        Denoised_img = ag.aggregation(Patches, Param, t)
        #plt.imshow(Denoised_img)
        #plt.gray()
        #plt.show()
        Denoised_video[:,:,t] = Denoised_img
    
    return Denoised_video