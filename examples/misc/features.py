"""
This example shows how to decompose EEG signals into source activations with
CSPVARICA, and subsequently extract single-trial connectivity as features for
LDA classification.
"""

from __future__ import print_function

import numpy as np
try:  # new in sklearn 0.19
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except ImportError:
    from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix

import scot
import scot.xvschema

# The data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement
# artifacts and resampled to a sampling rate of 100 Hz. With a visual cue, the
# subject was instructed to perform either hand or foot motor imagery. The
# trigger time points of the cues are stored in 'triggers', and 'classes'
# contains the class labels. Duration of the motor imagery period was
# approximately six seconds.
import scotdata.motorimagery as midata

raweeg = midata.eeg.T
triggers = np.asarray(midata.triggers, dtype=int)
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


# Set random seed for repeatable results
np.random.seed(42)


# Switch backend to scikit-learn
scot.backend.activate('sklearn')


# Set up analysis object
#
# We simply choose a VAR model order of 30, and reduction to 4 components.
ws = scot.Workspace({'model_order': 30}, reducedim=4, fs=fs)
freq = np.linspace(0, fs, ws.nfft_)


# Prepare data
#
# Here we cut out segments from 3s to 4s after each trigger. This is right in
# the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)

# Initialize cross-validation
nfolds = 10
kf = KFold(len(triggers), n_folds=nfolds)

# LDA requires numeric class labels
cl = np.unique(midata.classes)
classids = np.array([dict(zip(cl, range(len(cl))))[c] for c in midata.classes])

# Perform cross-validation
lda = LDA()
cm = np.zeros((2, 2))
fold = 0
for train, test in kf:
    fold += 1

    # Perform CSPVARICA
    ws.set_data(data[train, :, :], classes[train])
    ws.do_cspvarica()

    # Find optimal regularization parameter for single-trial fitting
    # ws.var_.xvschema = scot.xvschema.singletrial
    # ws.optimize_var()
    ws.var_.delta = 1

    # Single-trial fitting and feature extraction
    features = np.zeros((len(triggers), 32))
    for t in range(len(triggers)):
        print('Fold {:2d}/{:2d}, trial: {:d}   '.format(fold, nfolds, t),
              end='\r')
        ws.set_data(data[t, :, :])
        ws.fit_var()

        con = ws.get_connectivity('ffPDC')

        alpha = np.mean(con[:, :, np.logical_and(7 < freq, freq < 13)], axis=2)
        beta = np.mean(con[:, :, np.logical_and(15 < freq, freq < 25)], axis=2)

        features[t, :] = np.array([alpha, beta]).flatten()

    lda.fit(features[train, :], classids[train])

    acc_train = lda.score(features[train, :], classids[train])
    acc_test = lda.score(features[test, :], classids[test])

    print('Fold {:2d}/{:2d}, '
          'acc train: {:.3f}, '
          'acc test: {:.3f}'.format(fold, nfolds, acc_train, acc_test))

    pred = lda.predict(features[test, :])
    cm += confusion_matrix(classids[test], pred)

print('\nConfusion Matrix:\n', cm)
print('\nTotal Accuracy: {:.3f}'.format(np.sum(np.diag(cm))/np.sum(cm)))
