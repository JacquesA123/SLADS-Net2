#! /usr/bin/env python3

import numpy as np

def generateInitialMask(InitialMask,SizeImage):

    if InitialMask.MaskType == 'R':

        # 'R': Randomly distributed mask; can choose any percentage
        Mask = np.zeros((SizeImage[0], SizeImage[1]))
        UnifMatrix = np.random.rand(SizeImage[0], SizeImage[1])
        Mask = UnifMatrix<(InitialMask.Percentage / 100)

    elif InitialMask.MaskType == 'U':
        
        # 'U': Uniform mask; can choose any percentage
    
        Mask = np.zeros((SizeImage[0], SizeImage[1]))
        ModVal = int(100 / InitialMask.Percentage)
        for r in range(0 , SizeImage[0]):
            for s in range(0 , SizeImage[1]): 
                LinIdx = r * SizeImage[1] + s
                if np.remainder(LinIdx, ModVal) == 0:
                    Mask[r][s] = 1
    
    # 'H': low-dsicrepacy mask; can only choose 1% mask
    # Not in this function for some reason
    
    return Mask
        
        