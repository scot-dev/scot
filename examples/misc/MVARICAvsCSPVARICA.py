# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

"""
This example shows how to decompose motor imagery EEG into sources using
CSPVARICA and visualize a connectivity measure.
"""

import numpy as np

import scot

# The data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement
# artifacts and resampled to a sampling rate of 100 Hz. With a visual cue, the
# subject was instructed to perform either hand or foot motor imagery. The
# trigger time points of the cues are stored in 'triggers', and 'classes'
# contains the class labels. Duration of the motor imagery period was
# approximately six seconds.
from scot.datasets import fetch


midata = fetch("mi")[0]

raweeg = midata["eeg"]
triggers = midata["triggers"]
classes = midata["labels"]
fs = midata["fs"]
locs = midata["locations"]


# Set random seed for repeatable results
np.random.seed(42)


# Prepare data
#
# Here we cut out segments from 3s to 4s after each trigger. This is right in
# the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)


# Set up analysis object
#
# We simply choose a VAR model order of 30, and reduction to 4 components.
ws = scot.Workspace({'model_order': 30}, reducedim=4, fs=fs, locations=locs)

# Configure plotting options
ws.plot_f_range = [0, 30]  # only show 0-30 Hz
ws.plot_diagonal = 'S'  # put spectral density plots on the diagonal
ws.plot_outside_topo = True  # plot topos above and to the left

# Perform MVARICA
ws.set_data(data, classes)
ws.do_mvarica()
fig1 = ws.plot_connectivity_topos()
ws.set_used_labels(['foot'])
ws.fit_var()
ws.get_connectivity('ffDTF', fig1)
ws.set_used_labels(['hand'])
ws.fit_var()
ws.get_connectivity('ffDTF', fig1)
fig1.suptitle('MVARICA')

# Perform CSPVARICA
ws.set_data(data, classes)
ws.do_cspvarica()
fig2 = ws.plot_connectivity_topos()
ws.set_used_labels(['foot'])
ws.fit_var()
ws.get_connectivity('ffDTF', fig2)
ws.set_used_labels(['hand'])
ws.fit_var()
ws.get_connectivity('ffDTF', fig2)
fig2.suptitle('CSPVARICA')

ws.show_plots()
