
"""
This example shows how to decompose motor imagery EEG into sources using
SCPVARICA and visualize a connectivity measure.
"""
import numpy as np

import scot.backend.sklearn     # use scikit-learn backend
#import scot.backend.builtin     # use builtin (default) backend
import scot

from eegtopo.eegpos3d import positions as eeg_locations
from eegtopo.topoplot import topoplot

"""
The example data set contains a continuous 45 channel EEG recording of a motor
imagery experiment. The data was preprocessed to reduce eye movement artifacts
and resampled to a sampling rate of 100 Hz.
With a visual cue the subject was instructed to perform either hand of foot
motor imagery. The the trigger time points of the cues are stored in 'tr', and
'cl' contains the class labels (hand: 1, foot: -1). Duration of the motor 
imagery period was approximately 6 seconds.
"""

from motorimagery import data as midata

raweeg = midata.eeg
triggers = midata.triggers
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


"""
Prepare the data

Here we cut segments from 3s to 4s following each trigger out of the EEG. This
is right in the middle of the motor imagery period.
"""

data = scot.datatools.cut_segments(raweeg, triggers, 3*fs, 4*fs)

"""
Set up the analysis object

We simply choose a VAR model order of 30, and reduction to 4 components (that's not a lot!).
"""

api = scot.SCoT(30, reducedim=4, locations=locs)

"""
Perform MVARICA
"""

api.setData(data)
api.doMVARICA()

api.plotComponents()

"""
Connectivity Analysis

We will extract the full frequency directed transfer function (ffDTF) from the
activations of each class and plot them with matplotlib.

We define a function "topo" that multiplot2 calls to draws scalp projections
of the components.
"""

api.setData(data, classes)
api.fitVAR()

api.plotConnectivity('ffDTF')

api.showPlots()