#! /usr/bin/env python3
import numpy as np

def computeNeighborWeights(NeighborDistances,TrainingInfo):
    """
    Computes weighted norms corresponding to the distances between neighbors.
    """
    
    UnNormNeighborWeights = 1 / np.power(NeighborDistances, TrainingInfo.p)
    SumOverRow = (np.sum(UnNormNeighborWeights, axis=1))
    NeighborWeights = UnNormNeighborWeights / SumOverRow[:, np.newaxis]
    
    return NeighborWeights
