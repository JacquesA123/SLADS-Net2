
import sys
sys.path.append('code')
import numpy as np

from computeOrupdateERD import ComputeRecons
from computeOrupdateERD import FindNeighbors
from computeDifference import computeDifference
from loader import loadTestImage

def performReconOnce(SavePath,TrainingInfo,Resolution,SizeImage,ImageType,CodePath,TestingImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues):
    """
    My guess (Nolan) is that Recon stands for reconstruction. This function identifies neighbors, uses them to perform the reconstruction, and performs a comparison against the test image to yield distortion D.
    """

    NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)
    ReconValues,ReconImage = ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)    
    
    Img = loadTestImage(CodePath,TestingImageSet,ImNum,ImageExtension,SimulationRun)
    
    Difference = np.sum(computeDifference(Img,ReconImage,ImageType))
    return(Difference,ReconImage)