# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:51:43 2019

@author: pauls
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from alternating_exact_min import alt
from gradient_descent import gd
from hard_tresholding import hard_treshold
import patches
import video_block_matching as vbm
import sparse_coding
import transform_update as tu
from LR_approx import LR_approx
import parameters
import aggregation as ag

def SALT_gd(Video, new=True, BM=None, Patches=None, Param=None):
    if new:
        parameters.set_parameters(Video)
        Param = parameters.Param
    K = Param.similar_patches_number
    sp_treshold = Param.sparse_treshold
    w_lr = Param.low_rank_weight
    w_s = Param.sparse_weight
    w_d = Param.data_weight
    m = Param.temporal_search_range //2
    patches_num = Param.patches_num
    psize = Param.psize
    frame_num = Param.frame_num
    rank = 7
    if new:
        Patches = patches.overlapping_patches(Video, Param)    
        BM = vbm.video_block_matching(Patches, Param)
    transform1 = scipy.fftpack.dct(np.eye(psize*m), norm='ortho', axis=0)
    transform2 = scipy.fftpack.dct(np.eye(K*m), norm='ortho', axis=0)
    Denoised_video = np.zeros(Video.shape)

    rand1 = np.random.random_sample((psize, rank))
    rand2 = np.random.random_sample((rank, K))
    for t in range(frame_num):
        for i in range(patches_num):
            #u, s, vt  = scipy.sparse.linalg.svds(Patches[:,BM[i+t*patches_num,:]],
                                                 #rank)
            #s = np.diag(s)
            #P = u @ s
            #L = vt
            M = Patches[:,BM[i+t*patches_num,:]]
            P = rand1
            L = rand2
                
            for j in range(8):
                
                Ppinv = np.linalg.pinv(P)
                L =  Ppinv @ M
                Lpinv = np.linalg.pinv(L)
                P = M @ Lpinv
                
                if j%4 == 1:
                    Pm = P[:,:m]
                    p = np.ravel(Pm)
                    sparsed_p = sparse_coding.sparse(p, Param, transform1)
                    transform1 = tu.trans_update(sparsed_p, p)
                    p = transform1.T @ hard_treshold(transform1 @ p, sp_treshold)
                    Pm = p.reshape(Pm.shape)
                    P[:,:m] = Pm
                    
                    Lm = L[:m,:]
                    l = np.ravel(Lm)
                    sparsed_l = sparse_coding.sparse(l, Param, transform2)
                    transform2 = tu.trans_update(sparsed_l, l)
                    l = transform2.T @ hard_treshold(transform2 @ l, sp_treshold)
                    Lm = l.reshape(Lm.shape)
                    L[:m,:] = Lm
                    
            Patches[:,BM[i+t*patches_num,:]] = P @ L
            
            
        
        Denoised_img = ag.aggregation(Patches, Param, t)
        #plt.imshow(Denoised_img)
        #plt.gray()
        #plt.show()
        Denoised_video[:,:,t] = Denoised_img
    
    return Denoised_video