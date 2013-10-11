
"""
This example shows how to decompose motor imagery EEG into sources using
SCPVARICA and visualize a connectivity measure.
"""
import numpy as np

import scot
import scot.backend.sklearn     # use scikit-learn backend
#import scot.backend.builtin     # use builtin (default) backend

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
Set up the analysis object

We simply choose a VAR model order of 30, and reduction to 4 components (that's not a lot!).
"""
ws = scot.Workspace(30, reducedim=4, fs=fs, locations=locs)


"""
Prepare the data

Here we cut segments from 3s to 4s following each trigger out of the EEG. This
is right in the middle of the motor imagery period.
"""
data = scot.datatools.cut_segments(raweeg, triggers, 3*fs, 4*fs)

"""
Perform MVARICA
"""
ws.setData(data, time_offset=3)
ws.doMVARICA()


"""
Prepare the data

Here we cut segments from -1s to 7s around each trigger out of the EEG. This
covers the whole trial
"""
data = scot.datatools.cut_segments(raweeg, triggers, -1*fs, 7*fs)

"""
Connectivity Analysis

Extract the full frequency directed transfer function (ffDTF) from the
activations of each class and plot them with matplotlib.
"""
ws.setData(data, classes, time_offset=-1)
ws.plotTFConnectivity('ffDTF', 1*fs, int(0.2*fs))

ws.showPlots()
