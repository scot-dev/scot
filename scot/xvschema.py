# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Cross-validation schemas """

from numpy import sort

def singletrial(t,T):
    """Single-trial cross-validation schema: use one trial for training, all others for testing."""
    trainset = [t]
    testset = [i for i in range(trainset[0])] + [i for i in range(trainset[-1]+1,T)]

    testset = sort([t%T for t in testset])
    
    return trainset, testset
    
def multitrial(t,T):
    """Multi-trial cross-validation schema: use one trial for testing, all others for training."""
    testset = [t]    
    trainset = [i for i in range(testset[0])] + [i for i in range(testset[-1]+1,T)]
    
    trainset = sort([t%T for t in testset])
    
    return trainset, testset