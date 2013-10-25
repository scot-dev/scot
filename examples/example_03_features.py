
"""
This example shows how to decompose EEG signals into source activations with MVARICA, and subsequently extract single-trial connectivity as features for LDA.
"""

from __future__ import print_function

import numpy as np

import scot
import scot.backend.sklearn     # use scikit-learn backend
import scot.xvschema

from sklearn.lda import LDA

import matplotlib.pyplot as plt


# The example data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement artifacts
# and resampled to a sampling rate of 100 Hz.
# With a visual cue the subject was instructed to perform either hand of foot
# motor imagery. The the trigger time points of the cues are stored in 'tr', and
# 'cl' contains the class labels (hand: 1, foot: -1). Duration of the motor
# imagery period was approximately 6 seconds.
from motorimagery import data as midata

raweeg = midata.eeg
triggers = midata.triggers
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


# Set up the analysis object
# We simply choose a VAR model order of 30, and reduction to 4 components.
ws = scot.Workspace(30, reducedim=4, fs=fs)


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3*fs, 4*fs)


# Perform MVARICA
ws.set_data(data)
ws.do_mvarica()

# Find optimal regularization parameter for single-trial fitting
ws.optimize_regularization(scot.xvschema.singletrial, 30)

freq = np.linspace(0,fs,ws.nfft_)

# Single-Trial Fitting and feature extraction
features = np.zeros((len(triggers), 32))
for t in range(len(triggers)):
    print('Trial: %d   '%t, end='\r')
    ws.set_data(data[:,:,t])
    ws.fit_var()

    con = ws.get_connectivity('ffPDC')
    
    alpha = np.mean(con[:,:,np.logical_and(7<freq, freq<13)], axis=2)
    beta = np.mean(con[:,:,np.logical_and(15<freq, freq<25)], axis=2)
    
    features[t,:] = np.array([alpha, beta]).flatten()
print('')
    
lda = LDA( )
lda.fit(features, classes)

llh = lda.transform(features)
    
plt.hist([llh[classes==-1,:], llh[classes==1,:]])
plt.show()
