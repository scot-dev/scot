"""
This example shows how to decompose EEG signals into source activations with MVARICA, and subsequently extract single-trial connectivity as features for LDA.
"""

from __future__ import print_function

import numpy as np
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import scot
import scot.backend.sklearn     # use scikit-learn backend
import scot.xvschema


# The example data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement artifacts
# and resampled to a sampling rate of 100 Hz.
# With a visual cue the subject was instructed to perform either hand of foot
# motor imagery. The the trigger time points of the cues are stored in 'tr', and
# 'cl' contains the class labels (hand: 1, foot: -1). Duration of the motor
# imagery period was approximately 6 seconds.
import scotdata.motorimagery as midata

raweeg = midata.eeg
triggers = midata.triggers
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


# Set up the analysis object
# We simply choose a VAR model order of 30, and reduction to 4 components.
ws = scot.Workspace({'model_order': 30}, reducedim=4, fs=fs)
freq = np.linspace(0, fs, ws.nfft_)


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)

# Initialize Cross Validation
nfolds = 10
kf = KFold(len(triggers), n_folds=nfolds, indices=False)

# LDA requires numeric class labels
clunique = np.unique(midata.classes)
classids = np.array([dict(zip(clunique, range(len(clunique))))[c] for c in midata.classes])

# Perform Cross Validation
lda = LDA()
cm = np.zeros((2, 2))
fold = 0
for train, test in kf:
    fold += 1

    # Perform MVARICA
    ws.set_data(data[:, :, train], classes[train])
    ws.do_cspvarica()

    # Find optimal regularization parameter for single-trial fitting
    #ws.var_.xvschema = scot.xvschema.singletrial
    #ws.optimize_var()
    ws.var_.delta = 1

    # Single-Trial Fitting and feature extraction
    features = np.zeros((len(triggers), 32))
    for t in range(len(triggers)):
        print('Fold %d/%d, Trial: %d   ' %(fold, nfolds, t), end='\r')
        ws.set_data(data[:, :, t])
        ws.fit_var()

        con = ws.get_connectivity('ffPDC')

        alpha = np.mean(con[:, :, np.logical_and(7 < freq, freq < 13)], axis=2)
        beta = np.mean(con[:, :, np.logical_and(15 < freq, freq < 25)], axis=2)

        features[t, :] = np.array([alpha, beta]).flatten()

    lda.fit(features[train, :], classids[train])

    acc_train = lda.score(features[train, :], classids[train])
    acc_test = lda.score(features[test, :], classids[test])

    print('Fold %d/%d, Acc Train: %.4f, Acc Test: %.4f' %(fold, nfolds, acc_train, acc_test))

    pred = lda.predict(features[test, :])
    cm += confusion_matrix(classids[test], pred)
print('Confusion Matrix:\n', cm)

print('Total Accuracy: %.4f'%(np.sum(np.diag(cm))/np.sum(cm)))
