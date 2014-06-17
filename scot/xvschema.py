# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Cross-validation schemas """

from numpy import sort


def singletrial(num_trials, skipstep):
    """ Single-trial cross-validation schema

    Use one trial for training, all others for testing.

    Parameters
    ----------
    num_trials : int
        Total number of trials
    skipstep : int
        only use every `skipstep` trial for training

    Returns
    -------
    gen : generator object
        the generator returns tuples (trainset, testset)
    """
    for t in range(0, num_trials, skipstep):
        trainset = [t]
        testset = [i for i in range(trainset[0])] + [i for i in range(trainset[-1] + 1, num_trials)]
        testset = sort([t % num_trials for t in testset])
        yield trainset, testset


def multitrial(num_trials, skipstep):
    """ Multi-trial cross-validation schema

    Use one trial for testing, all others for training.

    Parameters
    ----------
    num_trials : int
        Total number of trials
    skipstep : int
        only use every `skipstep` trial for testing

    Returns
    -------
    gen : generator object
        the generator returns tuples (trainset, testset)
    """
    for t in range(0, num_trials, skipstep):
        testset = [t]
        trainset = [i for i in range(testset[0])] + [i for i in range(testset[-1] + 1, num_trials)]
        trainset = sort([t % num_trials for t in trainset])
        yield trainset, testset


def splitset(num_trials, skipstep):
    """ Split-set cross validation

    Use half the trials for training, and the other half for testing. Then
    repeat the other way round.

    Parameters
    ----------
    num_trials : int
        Total number of trials
    skipstep : int
        unused

    Returns
    -------
    gen : generator object
        the generator returns tuples (trainset, testset)
    """
    split = num_trials // 2

    a = list(range(0, split))
    b = list(range(split, num_trials))
    yield a, b
    yield b, a