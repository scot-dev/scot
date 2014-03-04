"""
This example shows how to decompose EEG signals into source activations with
MVARICA, and visualize a connectivity.
"""

import scot.backend.builtin     # use builtin (default) backend
import scot

import numpy as np
from scot.utils import cuthill_mckee
import matplotlib.pyplot as plt
from eegtopo.topoplot import Topoplot
from scot import plotting


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


# Prepare the data
#
# Here we cut segments from 2s to 5s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 2 * fs, 5 * fs)


# Set up the analysis object
#
# We simply choose a VAR model order of 30, and reduction to 15 components.
ws = scot.Workspace({'model_order': 30}, reducedim=15, fs=fs, locations=locs)


# Perform MVARICA to obtain
ws.set_data(data, classes)
ws.do_cspvarica()


# Connectivity Analysis
#
# Extract the full frequency directed transfer function (ffDTF) from the
# activations of each class and calculate the average value over the alpha band (8-12Hz).

freq = np.linspace(0, fs, ws.nfft_)
alpha, beta = {}, {}
for c in np.unique(classes):
    ws.set_used_labels([c])
    ws.fit_var()
    con = ws.get_connectivity('ffDTF')
    alpha[c] = np.mean(con[:, :, np.logical_and(8 < freq, freq < 12)], axis=2)

# Prepare topography plots
topo = Topoplot()
topo.set_locations(locs)
mixmaps = plotting.prepare_topoplots(topo, ws.mixing_)

# Force diagonal (self-connectivity) to 0
np.fill_diagonal(alpha['hand'], 0)
np.fill_diagonal(alpha['foot'], 0)

order = None
for cls in ['hand', 'foot']:
    np.fill_diagonal(alpha[cls], 0)

    w = alpha[cls]
    m = alpha[cls] > 4

    # use same ordering of components for each class
    if not order:
        order = cuthill_mckee(m)

    # fixed color, but alpha varies with connectivity strength
    r = np.ones(w.shape)
    g = np.zeros(w.shape)
    b = np.zeros(w.shape)
    a = (alpha[cls]-4) / max(np.max(alpha['hand']-4), np.max(alpha['foot']-4))
    c = np.dstack([r, g, b, a])

    plotting.plot_circular(colors=c, widths=w, mask=m, topo=topo, topomaps=mixmaps, order=order)
    plt.title(cls)

plotting.show_plots()

