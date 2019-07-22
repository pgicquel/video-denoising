# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:53:31 2019

@author: pauls
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io

from SALT_gradient_descent import SALT_gd
import SALT_denoising as sd
import newSALT as ns
import video_block_matching as vbm
import patches
import parameters

def positive(X):
    return X * (X >= 0)


sigma = 15
mat_Video = sio.loadmat("salesman.mat")
Video = mat_Video['clean']
#Video = skvideo.io.vread("video\Man_texting.mp4", 
                         #outputdict={"-pix_fmt": "gray"})[:, :, :, 0]

Video = Video[:,:,:]
#Video = np.transpose(Video, (1,2,0))
Video = Video[:,:,:2]
Noisy_video = positive(Video + np.random.normal(0, sigma, Video.shape))
plt.imshow(Video[:,:,0])
plt.gray()
plt.show

plt.imshow(Noisy_video[:,:,0])
plt.gray()
plt.show


parameters.set_parameters(Noisy_video)
Para = parameters.Param
pat = patches.overlapping_patches(Noisy_video, Para)
Blocks = vbm.video_block_matching(pat, Para)
denoised = sd.SALT(Video, new=False, BM=Blocks, Patches=pat, Param=Para)
plt.imshow(denoised[:,:,0])
#denoised = np.transpose(denoised, (2,0,1))
#skvideo.io.vwrite("denoised_video.mp4", denoised, inputdict={"-r" : "13"})
#Noisy_video = np.transpose(Noisy_video, (2,0,1))
#skvideo.io.vwrite("noisy_video.mp4", Noisy_video, inputdict={"-r" : "13"})
