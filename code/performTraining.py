#! /usr/bin/env python3

import os
import numpy as np
from scipy.io import loadmat

from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPRegressor as nnr
import pickle

from imageio import imread
import glob
import sys
import random

from computeOrupdateERD import FindNeighbors
from computeOrupdateERD import ComputeRecons
from computeFeatures import computeFeatures
from computeDifference import computeDifference

def performTraining(MeasurementPercentageVector,TrainingDataPath,ImageType,ImageExtension,SizeImage,TrainingInfo,Resolution,WindowSize,c_vec,PercOfRD):
    
    ImNum = 0
    loadPathImage = TrainingDataPath + 'Images' + os.path.sep   
    NumTrainingImages = np.size(glob.glob(loadPathImage + '*' + ImageExtension))
    
    for image_path in glob.glob(loadPathImage + '*' + ImageExtension):
        
        # Import image data based on their image extension
        if ImageExtension=='.mat':
            ImgDat=loadmat(image_path)
            Img=ImgDat['img']
        else:
            Img = imread(image_path)
                    
        if SizeImage[0] != Img.shape[0] or SizeImage[1] != Img.shape[1]:
            sys.exit('Error!!! The dimensions you entered in "SizeImage" do not match the dimensions of the training images')
        
        if not os.path.exists(TrainingDataPath + 'FeaturesRegressCoeffs'):
            os.makedirs(TrainingDataPath + 'FeaturesRegressCoeffs')

        for m in range(0, np.size(MeasurementPercentageVector)):

            SaveFolder = 'Image_' + str(ImNum+1) + '_Perc_' + str(MeasurementPercentageVector[m])
            SavePath = TrainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + SaveFolder
            if not os.path.exists(SavePath):
                os.makedirs(SavePath)

            # Generate a random mask for the image
            Mask = np.zeros((SizeImage[0], SizeImage[1]))
            UnifMatrix = np.random.rand(SizeImage[0], SizeImage[1])
            Mask = UnifMatrix < (MeasurementPercentageVector[m] / 100)
            
            # Get the indices of the measured and unmeasured pixels
            MeasuredIdxs = np.transpose(np.where(Mask==1))
            UnMeasuredIdxs = np.transpose(np.where(Mask==0))
            MeasuredValues = Img[Mask==1]

            # Find the neighbors of the measured pixels
            NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)
            
            ReconValues,ReconImage = ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)
            
            AllPolyFeatures=computeFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfo,ReconValues,ReconImage,Resolution,ImageType)
            
            ### ESTIMATING ERD FROM FEATURES AND RECONSTRUCTIONS USING GAUSSIAN KERNEL TO APPROXIMATE W/ RD ATTENUATION
            
            # Calculate the number of random choices for the RDPP
            NumRandChoices =  int(PercOfRD * MeasurementPercentageVector[m] * \
                                  SizeImage[1] * SizeImage[0] / (100*100)) # Scale by downsample of original image size
            
            # Generate a random order for the unmeasured pixels
            OrderForRD = random.sample(range(0, UnMeasuredIdxs.shape[0]), NumRandChoices) 
            PolyFeatures = AllPolyFeatures[OrderForRD,:]
            
            # RDPP = difference between original and reconstructed image
            # I presume it stands for Reduction in Disortion (RD) per pixel, but I could be wrong
            RDPP = computeDifference(Img, ReconImage, ImageType).astype(int)
            
            # Pad the RDPP with zeros to account for the window size
            RDPPWithZeros = np.pad(RDPP, (int(np.floor(WindowSize[0]/2)), int(np.floor(WindowSize[1]/2))), \
                                   'constant', constant_values=0)
            
            # Gives local neighborhoods as columns in this array
            ImgAsBlocks = im2col(RDPPWithZeros, WindowSize)
            
            # Convert mask into boolean list
            MaskVect = np.ravel(Mask)
            
            # Select only local neighborhoods 
            ImgAsBlocksOnlyUnmeasured = ImgAsBlocks[:, np.logical_not(MaskVect)]
            
            # ??
            temp = np.zeros((WindowSize[0]*WindowSize[1], NumRandChoices))
            
            for c in c_vec:
                sigma = NeighborDistances[:, 0] / c
                cnt = 0;
                for l in OrderForRD:
                    Filter = generateGaussianKernel(sigma[l], WindowSize)
                    temp[:, cnt] = ImgAsBlocksOnlyUnmeasured[:, l] * Filter
                    cnt = cnt + 1
                RD = np.sum(temp, axis=0)
                SavePath_c = SavePath + os.path.sep + 'c_' + str(c)

                if not os.path.exists(SavePath_c):
                    os.makedirs(SavePath_c)
                
                np.save(SavePath_c + os.path.sep + 'RD', RD)        
                np.save(SavePath_c + os.path.sep + 'OrderForRD', OrderForRD) 
                  
            np.save(SavePath + os.path.sep + 'Mask', Mask)   
            np.save(SavePath + os.path.sep + 'ReconImage', ReconImage)
            np.save(SavePath + os.path.sep + 'PolyFeatures', PolyFeatures)
            
        if ImNum == 0:
            print('Feature Extraction Complete for ' + str(ImNum+1) + ' Image' )
        else:
            print('Feature Extraction Complete for ' + str(ImNum+1) + ' Images' )
        
        ImNum = ImNum + 1
        
    #try:
    #    Img
    #except NameError:
    #    sys.exit('Error!!! There are no images in ' + loadPathImage + ' that have the extention ' + ImageExtension)
        
    for c in c_vec:
        FirstLoop = 1
        for ImNum in range(0,NumTrainingImages):
            for m in range(0,np.size(MeasurementPercentageVector)):
                
                LoadFolder = 'Image_' + str(ImNum+1) + '_Perc_' + str(MeasurementPercentageVector[m])
                LoadPath = TrainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + LoadFolder
                PolyFeatures = np.load(LoadPath + os.path.sep + 'PolyFeatures.npy')
                LoadPath_c = LoadPath + os.path.sep + 'c_' + str(c)
                RD = np.load(LoadPath_c + os.path.sep + 'RD.npy')
                if ImageType=='D':
                    if FirstLoop==1:
                        BigPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))
                        BigRD = RD
                        FirstLoop = 0                  
                    else:
                        TempPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))                    
                        BigPolyFeatures = np.vstack((BigPolyFeatures,TempPolyFeatures))
                        BigRD = np.append(BigRD,RD)
                else:
                    if FirstLoop==1:
                        BigPolyFeatures = PolyFeatures
                        BigRD = RD
                        FirstLoop = 0                  
                    else:
                        TempPolyFeatures = PolyFeatures               
                        BigPolyFeatures = np.vstack((BigPolyFeatures,TempPolyFeatures))
                        BigRD = np.append(BigRD,RD)                    
                    
                       
        #regr = linear_model.LinearRegression()
        
#        regr = svm.SVR(kernel='rbf')
        
        regr = nnr(activation='identity', solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        
        regr.fit(BigPolyFeatures, BigRD)

#        Theta = np.zeros((PolyFeatures.shape[1]))    
#        if ImageType=='D':            
#            Theta[0:24]=regr.coef_[0:24]
#            Theta[26]=regr.coef_[25]
#        else:
#            Theta = regr.coef_
        SavePath_c = TrainingDataPath + os.path.sep + 'c_' + str(c)
        del BigRD,BigPolyFeatures

        if not os.path.exists(SavePath_c):
            os.makedirs(SavePath_c) 
#        np.save(SavePath_c + os.path.sep + 'Theta', Theta)
        with open(SavePath_c + os.path.sep + 'Theta.pkl', 'wb') as fid:
            pickle.dump(regr, fid)
        
        
        print("Regressions Complete for c = " + str(c))
            
def im2col(Matrix,WindowSize):
    """
    Generates an array of windows given by size WindowSize over Matrix, transforms the windows into columns, and then returns the hstack of these columns.
    """
    M,N = Matrix.shape
    col_extent = N - WindowSize[1] + 1
    row_extent = M - WindowSize[0] + 1
    start_idx  = np.arange(WindowSize[0])[:,None] * N + np.arange(WindowSize[1])
    offset_idx = np.arange(row_extent)[:,None] * N    + np.arange(col_extent)
    out = np.take(Matrix,start_idx.ravel()[:,None] + offset_idx.ravel())

    return(out)
    # http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python

def generateGaussianKernel(sigma,WindowSize):
    FilterMat = np.ones((WindowSize[0],WindowSize[1]))
    for i in range(0,WindowSize[0]):
        for j in range(0,WindowSize[1]):
            FilterMat[i][j] = np.exp( -(1/(2*sigma**2)) * np.absolute( ( (i-np.floor(WindowSize[0]/2))**2 +  (j-np.floor(WindowSize[1]/2))**2 ) )  )
    FilterMat = FilterMat / np.amax(FilterMat)
    FilterMat = np.transpose(FilterMat)
    Filter=np.ravel(FilterMat)
    return Filter

