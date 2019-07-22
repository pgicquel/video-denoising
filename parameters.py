# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:32:34 2019

@author: pauls
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Param:
    # settings
    estimated_sigma : float
    stride : int
    stride_temporal : int
    xpsize : int
    ypsize : int
    psize : int
    x_window_size : int
    y_window_size : int
    temporal_search_range : int
    similar_patches_number : int
    forgetting_factor : float
    # other parameters
    height : int
    width : int
    frame_num : int
    sparse_treshold : float
    LR_treshold : float
    x_patches_num : int
    y_patches_num : int
    patches_num : int
    sparse_weight : float
    low_rank_weight : float
    data_weight : float

def set_parameters(Video):
    sigma = 15
    Param.estimated_sigma = sigma
    Param.stride = 4
    Param.stride_temporal = 1
    Param.xpsize = 8
    Param.ypsize = 8
    Param.psize = Param.xpsize*Param.ypsize
    Param.x_window_size = 12
    Param.y_window_size = 12
    Param.temporal_search_range = 2*3+1
    K = 15
    Param.similar_patches_number = K
    Param.forgetting_factor = 0.2
    Param.height = Video.shape[0]
    Param.width = Video.shape[1]
    Param.frame_num = Video.shape[2]
    Param.LR_treshold = 0.7*sigma*(np.sqrt(K)+np.sqrt(Param.psize))
    Param.sparse_treshold = np.sqrt(sigma) * 5
    Param.x_patches_num = int((Param.width - Param.xpsize) // Param.stride) + 1
    Param.y_patches_num = int((Param.height - Param.ypsize) // Param.stride) + 1
    Param.patches_num = Param.x_patches_num * Param.y_patches_num
    Param.sparse_weight = 0.9
    Param.low_rank_weight = 1
    Param.data_weight = 0
    
    return Param